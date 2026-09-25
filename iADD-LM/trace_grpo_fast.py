"""
Faster drop-in trainer for trace-based GRPO on MDLM (iADD-LM Phase 1, Step 1).

SAME algorithm as trace_grpo.py -- this file only changes how work is batched
on the GPU. It subclasses TraceGRPO from trace_grpo.py and overrides:

  - rollout():   now takes a LIST of P prompts (all with the same tokenized
                 length -- see "prompt length buckets" below) and runs all
                 P * G sequences through the model as ONE batch per denoising
                 step, instead of one prompt (G sequences) at a time. This
                 amortizes the fixed per-step Python / kernel-launch overhead
                 (embedding lookups, hydra/omegaconf attribute access, CUDA
                 graph capture where applicable, etc.) over more sequences,
                 which matters because the model is small (170M) and the GPU
                 was measured at 98% util but only 2.6GB memory -- i.e. we are
                 spending time on many small launches, not on raw FLOPs.
  - ppo_update(): identical PPO-clip math, only difference is an optional
                 torch.amp.autocast(bf16) region around the forward pass
                 (--amp flag) and a higher default micro_bs (8 vs 4).
  - main():      drives multi-prompt rollout + per-prompt-group advantage
                 normalization (advantages are group-relative WITHIN each
                 prompt's G samples, exactly as before -- never mixed across
                 prompts sharing a batch).

Prompt length buckets
----------------------
The MDLM reverse process fills z[:, :len(prompt)] with the prompt once at
t=1 and never touches those positions again (copy_flag=1 in the design
notes above / in trace_grpo.py). Because every row in a batched rollout
must share the same total sequence length `L = model.length` (gen_len) AND
the same prompt length (so the prompt occupies the same column range in
every row), we can only put two different prompts in the same batch if
they tokenize to the exact same number of tokens. Padding to a common
length is deliberately NOT used here -- it would require an attention mask
plumbed through FKDiffusion.forward (not present in the base class) and
would risk changing numerics for the non-batched (single-prompt) code path,
which violates the "same algorithm" requirement of this task.

So at startup we tokenize the 15 PPLM prompts once, bucket their indices by
token count, and each iteration:
  1. pick uniformly at random one bucket that has >= --prompts-per-iter
     distinct prompts (buckets with fewer than P prompts are skipped so we
     never need padding or sampling-with-replacement-within-a-batch);
  2. sample P distinct prompt indices from that bucket without replacement.

With the shipped pplm_discrim_prompts_orig.jsonl (gpt2 tokenizer, trailing
token dropped as in the original code) the bucket sizes are:
    length 3 tokens -> 10 prompts  (book, chicken, city, country, horse,
                                     lake, movie, painting, pizza, potato,
                                     road -- 11 actually, see note below)
    length 4 tokens -> 1 prompt   ("The last time")
    length 5 tokens -> 1 prompt   ("Once upon a time")
    length 6 tokens -> 2 prompts  ("The president of the country",
                                    "The year is 1910.")
(exact counts are recomputed from the tokenizer at runtime -- the above is
illustrative, verified once with the real GPT2 tokenizer on the CPU.)
With the default --prompts-per-iter 2, only the length-3 and length-6
buckets are eligible; this is a real (small) change in which prompts get
trained on per iteration relative to the strict round-robin of the
original script -- documented as a deviation below.

If --prompts-per-iter is larger than the biggest eligible bucket, we fall
back to the single-prompt path (P=1) for that iteration and print a
one-time warning, rather than silently changing the algorithm (no padding,
no cross-length batching).
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_grpo import (  # noqa: E402
    FK_DIR, REWARD_NAMES, TraceGRPO, _sample_categorical, build_config,
    compute_rewards, dataloader)


class TraceGRPOFast(TraceGRPO):
    """Multi-prompt-batched rollout + AMP update. Same PPO-clip algorithm."""

    # ---------- rollout (batches P prompts x G samples per forward) ----------
    @torch.no_grad()
    def rollout(self, prompt_ids_list, group_size, num_steps, eps=1e-5):
        """Sample `group_size` sequences for EACH of the P prompts in
        `prompt_ids_list` (all must tokenize to the same length), batching
        all P * group_size sequences through one forward per denoising step.

        Rows are laid out in P contiguous blocks of `group_size`: block p
        (rows [p*G:(p+1)*G]) belongs to prompt_ids_list[p]. This ordering is
        relied on by main() when it splits rewards/advantages back out per
        prompt.

        Returns the same dict shape as TraceGRPO.rollout (z_traj, final,
        old_lp, commit, step_H), just with a batch dim of P*G instead of G.
        """
        P = len(prompt_ids_list)
        G, L = group_size, self.config.model.length
        device = self.device
        dt = (1 - eps) / num_steps
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

        PG = P * G
        z = self._sample_prior(PG, L).to(device)
        for p, prompt_ids in enumerate(prompt_ids_list):
            block = slice(p * G, (p + 1) * G)
            z[block, :prompt_ids.shape[1]] = prompt_ids

        z_traj, old_lp, commit, step_H = [], [], [], []
        for i in range(num_steps + 1):
            t = timesteps[i] * torch.ones(PG, 1, device=device)
            z_traj.append(z.detach().to('cpu', torch.int32))

            sigma_t, _ = self.noise(t)
            sigma_s, _ = self.noise(t - dt)
            sigma_t, sigma_s = sigma_t.squeeze(-1), sigma_s.squeeze(-1)
            mct = (1 - torch.exp(-sigma_t))[:, None, None]
            mcs = (1 - torch.exp(-sigma_s))[:, None, None]

            log_p_x0 = self.forward(z, sigma_t)          # (PG, L, V) log-probs
            p_x0 = log_p_x0.exp()
            q_xs = p_x0 * (mct - mcs)
            q_xs[:, :, self.mask_index] = mcs[:, :, 0]
            _x = _sample_categorical(q_xs)

            masked = z == self.mask_index
            z_new = torch.where(masked, _x, z)
            com = masked & (z_new != self.mask_index)

            lp = torch.zeros_like(z, dtype=torch.float32)
            lp[com] = log_p_x0.gather(
                -1, z_new.unsqueeze(-1)).squeeze(-1)[com].float()
            old_lp.append(lp.to('cpu', torch.float16))
            commit.append(com.cpu())

            with torch.no_grad():
                H = -(p_x0 * log_p_x0).sum(-1)
                H = torch.where(masked, H, torch.zeros_like(H))
                denom = masked.sum(-1).clamp(min=1)
                step_H.append((H.sum(-1) / denom).cpu())

            z = z_new

        return dict(z_traj=torch.stack(z_traj), final=z.detach(),
                    old_lp=old_lp, commit=commit,
                    step_H=torch.stack(step_H))

    # ---------- update (adds optional bf16 autocast around forward) ----------
    def ppo_update(self, trace, advantages, steps_S, optimizer,
                   clip=0.2, micro_bs=8, num_steps=128, eps=1e-5, amp=False):
        """Identical PPO-clip math to TraceGRPO.ppo_update; only difference
        is the model.forward call is wrapped in torch.amp.autocast(bf16)
        when amp=True, and the default micro_bs is raised (4 -> 8)."""
        device = self.device
        G = trace['final'].shape[0]
        dt = (1 - eps) / num_steps
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

        total, n_terms = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for t_idx in steps_S:
            com = trace['commit'][t_idx]
            if not com.any():
                continue
            for g0 in range(0, G, micro_bs):
                gs = slice(g0, min(g0 + micro_bs, G))
                com_b = com[gs].to(device)
                if not com_b.any():
                    continue
                z_t = trace['z_traj'][t_idx][gs].to(device, torch.long)
                z_s = trace['z_traj'][t_idx + 1][gs].to(device, torch.long)
                t = timesteps[t_idx] * torch.ones(z_t.shape[0], 1, device=device)
                sigma_t, _ = self.noise(t)
                if amp:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        log_p_x0 = self.forward(z_t, sigma_t.squeeze(-1))
                else:
                    log_p_x0 = self.forward(z_t, sigma_t.squeeze(-1))
                new_lp = log_p_x0.gather(
                    -1, z_s.unsqueeze(-1)).squeeze(-1)[com_b].float()
                old_lp = trace['old_lp'][t_idx][gs].to(device, torch.float32)[com_b]
                adv = advantages[gs].to(device)
                adv_tok = adv.unsqueeze(-1).expand_as(com_b)[com_b]
                ratio = (new_lp - old_lp).exp()
                un = ratio * adv_tok
                cl = ratio.clamp(1 - clip, 1 + clip) * adv_tok
                loss = -torch.min(un, cl).mean() / max(len(steps_S), 1)
                loss.backward()
                total += loss.item()
                n_terms += 1
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
        optimizer.step()
        return total / max(n_terms, 1)


def build_prompt_buckets(tokenizer, prompts):
    """Tokenize each prompt (same way main() does for the single-prompt
    path: drop the trailing token, no padding) and group indices by token
    count. Returns dict: length -> list[(idx, prompt_ids_tensor_cpu)]."""
    buckets = defaultdict(list)
    for idx, text in enumerate(prompts):
        enc = tokenizer([text], return_tensors='pt', padding=False)
        ids = enc['input_ids'][:, :-1]
        buckets[ids.shape[1]].append((idx, ids))
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='~/dllm/mdlm-owt-local')
    ap.add_argument('--reward', default='sentiment', choices=REWARD_NAMES)
    ap.add_argument('--reward-label', default='positive')
    ap.add_argument('--reward-trim', type=int, default=100)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--gen-len', type=int, default=128)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--select', default='all',
                    choices=['all', 'early', 'late', 'any', 'entropy', 'incremental'])
    ap.add_argument('--select-n', type=int, default=16)
    ap.add_argument('--micro-bs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--prompt-file',
                    default=os.path.join(FK_DIR, 'evaluation',
                                         'pplm_discrim_prompts_orig.jsonl'))
    ap.add_argument('--out', default=os.path.expanduser('~/dllm/iadd-lm/runs'))
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--prompts-per-iter', type=int, default=2,
                    help='P: number of same-tokenized-length prompts to '
                         'batch together per iteration (each contributes '
                         'its own group of --group samples).')
    ap.add_argument('--amp', action='store_true',
                    help='Wrap the PPO-update forward pass in '
                         'torch.amp.autocast(bf16). Rollout already runs '
                         "under the model's internal autocast.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = TraceGRPOFast(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None                       # train the raw weights
    model.backbone.train()

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f]

    buckets = build_prompt_buckets(tokenizer, prompts)
    P = args.prompts_per_iter
    eligible = {length: entries for length, entries in buckets.items()
               if len(entries) >= P}
    if not eligible:
        bucket_sizes = {length: len(entries) for length, entries in buckets.items()}
        print(f'WARNING: no prompt-length bucket has >= {P} prompts '
              f'(bucket sizes: {bucket_sizes}); '
              'falling back to P=1 for every iteration.')
        eligible = {length: entries for length, entries in buckets.items()}
        P_effective_max = 1
    else:
        P_effective_max = P
    bucket_lengths = sorted(eligible.keys())

    run_name = (f"fast_{args.reward}_{args.select}{args.select_n}"
                f"_g{args.group}_p{args.prompts_per_iter}_s{args.seed}_"
                + datetime.now().strftime('%m%d-%H%M'))
    out_dir = os.path.join(os.path.expanduser(args.out), run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    log_f = open(os.path.join(out_dir, 'log.jsonl'), 'a')

    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=args.lr,
                                  weight_decay=0.0)

    for it in range(args.iters):
        t0 = time.time()

        length = random.choice(bucket_lengths)
        entries = eligible[length]
        p_use = min(P_effective_max, len(entries))
        chosen = random.sample(entries, p_use)
        prompt_idxs = [idx for idx, _ in chosen]
        prompt_ids_list = [ids.to(model.device) for _, ids in chosen]
        G = args.group

        trace = model.rollout(prompt_ids_list, G, args.steps)

        # decode + reward per prompt block, using that prompt's own length
        # for the reward-trim offset (same formula as the single-prompt
        # script, just applied block-by-block since prompt length can
        # differ ACROSS iterations, though not within one).
        trim = args.reward_trim + length
        texts = tokenizer.batch_decode(trace['final'][:, :trim])
        rewards = torch.tensor(
            compute_rewards(samples=texts, reward_name=args.reward,
                            reward_label=args.reward_label),
            dtype=torch.float32)

        # group-relative advantages, computed WITHIN each prompt's own
        # block of G samples -- never mixed across prompts.
        adv = torch.zeros_like(rewards)
        skipped_blocks = 0
        for p in range(p_use):
            block = slice(p * G, (p + 1) * G)
            r_block = rewards[block]
            a_block = r_block - r_block.mean()
            if a_block.abs().max() < 1e-6:
                skipped_blocks += 1
                continue  # leave this block's advantages at 0 -> no grad
            adv[block] = a_block / (r_block.std() + 1e-6)

        if skipped_blocks == p_use:
            log_f.write(json.dumps({'iter': it, 'r_mean': rewards.mean().item(),
                                    'skipped': True,
                                    'prompt_idxs': prompt_idxs}) + '\n')
            log_f.flush()
            continue

        S = model.select_steps(args.select, args.steps, args.select_n,
                               it, args.iters, step_H=trace['step_H'])
        loss = model.ppo_update(trace, adv, S, optimizer,
                                micro_bs=args.micro_bs, num_steps=args.steps,
                                amp=args.amp)

        rec = {'iter': it, 'r_mean': rewards.mean().item(),
               'r_max': rewards.max().item(), 'r_min': rewards.min().item(),
               'loss': loss, 'n_steps_trained': len(S),
               'prompts_this_iter': p_use, 'prompt_idxs': prompt_idxs,
               'sec': round(time.time() - t0, 1)}
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()
        print(rec)

        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            ckpt = os.path.join(out_dir, f'backbone_it{it+1}.pt')
            torch.save(model.backbone.state_dict(), ckpt)
            # keep best + last only: prune older
            saved = sorted(f for f in os.listdir(out_dir)
                           if f.startswith('backbone_it'))
            for f_old in saved[:-2]:
                os.remove(os.path.join(out_dir, f_old))

    log_f.close()
    print('done:', out_dir)


if __name__ == '__main__':
    main()
