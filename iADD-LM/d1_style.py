"""
d1-style (diffu-GRPO baseline) one-step mean-field GRPO for MDLM (iADD-LM).

Faithful-lite reimplementation of the d1/diffu-GRPO objective for our 170M
leaderboard, sitting alongside trace_grpo.py (trace-based GRPO). Where
trace_grpo.py trains on the *realized denoising trace* (per-step categorical
log-probs recorded during rollout), this file instead follows d1: treat the
whole completion as committed in one shot and approximate its log-probability
with a single forward pass of the model at (near-)fully-masked conditioning,
i.e. the "one-step mean-field likelihood"

    log p(y | q) ~= sum_i log p_x0(y_i | x)

where x = [q_masked ; MASK * len(y)], q_masked is the prompt with each token
independently replaced by MASK w.p. p_mask (d1's random-prompt-masking
regularizer, meant to make the estimator robust / act as a light data aug),
and log p_x0(. | x) is read off a single self.forward(x, sigma(t=1)) call
(the model's x0-predictor conditioned on the noisiest timestep, since x's
completion region is fully masked and the prompt region is partially masked).

Rollout is a plain (untraced) ancestral reverse-diffusion sampler -- we only
need the finished sequences, not the per-step trace, so there is no trace
bookkeeping here (see trace_grpo.TraceGRPO.rollout for the traced version).

PPO-clip ratios are computed against the same one-step estimate recomputed
under the current theta each inner epoch (--inner-epochs, default 2, matching
d1's point of doing multiple gradient updates per rollout batch via the
importance ratio). The prompt mask used to build x is sampled once per
iteration and reused for the old (rollout-time) and every inner-epoch new
log-prob estimate, so the PPO ratio reflects only the parameter change, not
masking noise. Group-relative advantages are computed exactly like
trace_grpo: G samples per prompt, A = (r - mean) / (std + eps), skip
zero-variance groups.

Run from Fk-Diffusion-Steering/discrete_diffusion/:
  python ~/dllm/iadd-lm/d1_style.py \
      --reward sentiment --reward-label positive \
      --steps 128 --gen-len 128 --group 8 --iters 300 --inner-epochs 2
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import torch

FK_DIR = os.path.expanduser('~/dllm/Fk-Diffusion-Steering/discrete_diffusion')
sys.path.insert(0, FK_DIR)
sys.path.insert(0, os.path.join(FK_DIR, 'mdlm'))
os.chdir(FK_DIR)  # hydra searchpath file://mdlm/configs is cwd-relative

import dataloader  # noqa: E402  (mdlm)
from fk_diffusion import (  # noqa: E402
    FKDiffusion, _sample_categorical, compute_rewards)

from hydra import compose, initialize_config_dir  # noqa: E402

REWARD_NAMES = ['sentiment', 'toxicity', 'cola', 'gpt2_perp']


def build_config(args):
    with initialize_config_dir(config_dir=os.path.join(FK_DIR, 'configs'),
                               version_base=None):
        cfg = compose(config_name='fk_steering_config', overrides=[
            f'seed={args.seed}',
            f'eval.checkpoint_path={os.path.expanduser(args.checkpoint)}',
            'data=openwebtext-split',
            f'model.length={args.gen_len}',
            'sampling.predictor=ddpm',
            f'sampling.steps={args.steps}',
            'loader.eval_batch_size=1',
            'backbone=hf_dit',
            f'fk_steering.reward_fn={args.reward}',
            f'fk_steering.reward_label={args.reward_label}',
            f'fk_steering.reward_trim_length={args.reward_trim}',
        ])
    return cfg


class D1Style(FKDiffusion):
    """Untraced rollout + d1-style one-step mean-field PPO objective."""

    # ---------- rollout (no trace, only the finished sequences) ----------
    @torch.no_grad()
    def rollout(self, prompt_ids, group_size, num_steps, eps=1e-5):
        """Sample `group_size` sequences via plain ancestral reverse
        diffusion. Returns dict(final=(G, L) long) -- unlike
        TraceGRPO.rollout we do not record per-step log-probs/commit masks,
        since the d1 objective is estimated post-hoc from the finished
        sequence rather than from the realized trace.
        """
        G, L = group_size, self.config.model.length
        device = self.device
        dt = (1 - eps) / num_steps
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

        z = self._sample_prior(G, L).to(device)
        if prompt_ids is not None:
            z[:, :prompt_ids.shape[1]] = prompt_ids

        for i in range(num_steps + 1):
            t = timesteps[i] * torch.ones(G, 1, device=device)
            sigma_t, _ = self.noise(t)
            sigma_s, _ = self.noise(t - dt)
            sigma_t, sigma_s = sigma_t.squeeze(-1), sigma_s.squeeze(-1)
            mct = (1 - torch.exp(-sigma_t))[:, None, None]
            mcs = (1 - torch.exp(-sigma_s))[:, None, None]

            log_p_x0 = self.forward(z, sigma_t)          # (G, L, V) log-probs
            p_x0 = log_p_x0.exp()
            q_xs = p_x0 * (mct - mcs)
            q_xs[:, :, self.mask_index] = mcs[:, :, 0]
            _x = _sample_categorical(q_xs)

            masked = z == self.mask_index
            z = torch.where(masked, _x, z)

        return dict(final=z.detach())

    # ---------- d1 one-step mean-field log-prob estimate ----------
    def seq_logprob(self, prompt_ids, completion, p_mask=0.15, prompt_mask=None):
        """Estimate log p(completion | prompt) with a single forward pass.

        prompt_ids : (1, P) long -- shared prompt, broadcast over the batch.
        completion : (B, C) long -- committed (non-mask) completion tokens.
        p_mask     : prob. of masking each prompt token, used only when
                     `prompt_mask` is not given (d1's random-prompt-masking
                     regularizer).
        prompt_mask: optional pre-sampled (B, P) bool mask; pass the SAME
                     mask for the rollout-time ("old") and every inner-epoch
                     ("new") call so the PPO ratio isolates the parameter
                     change rather than masking noise.

        Construction: x = [q_masked ; MASK * C], one forward pass at
        sigma(t=1) (the noisiest / fully-masked conditioning -- appropriate
        here since the whole completion region of x is masked and the prompt
        region is partially masked), then sum the completion tokens'
        log p_x0(y_i | x).

        Returns (seq_lp: (B,) float32, tok_lp: (B, C) float32).
        """
        device = self.device
        B, C = completion.shape
        P = prompt_ids.shape[-1]
        prompt = prompt_ids.reshape(1, P).expand(B, P).to(device)

        if prompt_mask is None:
            prompt_mask = torch.rand(B, P, device=device) < p_mask
        else:
            prompt_mask = prompt_mask.to(device)
        mask_id = torch.full_like(prompt, self.mask_index)
        q_masked = torch.where(prompt_mask, mask_id, prompt)

        comp_masked = torch.full((B, C), self.mask_index, dtype=torch.long,
                                 device=device)
        x = torch.cat([q_masked, comp_masked], dim=1)

        t1 = torch.ones(B, 1, device=device)
        sigma_1, _ = self.noise(t1)
        log_p_x0 = self.forward(x, sigma_1.squeeze(-1))   # (B, P+C, V) bf16

        tok_lp = log_p_x0[:, P:, :].float().gather(
            -1, completion.unsqueeze(-1).to(device)).squeeze(-1)  # (B, C)
        return tok_lp.sum(dim=1), tok_lp

    # ---------- PPO-clip update over one inner epoch ----------
    def ppo_update(self, prompt_ids, completion, old_tok_lp, advantages, optimizer,
                   clip=0.2, micro_bs=4, prompt_mask=None):
        """PER-TOKEN PPO-clip: ratio_j = exp(tok_lp_new_j - tok_lp_old_j) for
        each completion token j, clipped and weighted by the *sequence's*
        (scalar) advantage, then averaged over tokens. This matches diffu-GRPO
        and avoids the sequence-level ratio exp(sum_j(...)) which explodes
        for ~100-token completions.

        old_tok_lp : (G, C) float32 -- per-token log-probs recorded at
                     rollout time (the second return of seq_logprob).
        """
        device = self.device
        G = completion.shape[0]
        total, n_terms = 0.0, 0
        any_nonfinite = False
        optimizer.zero_grad(set_to_none=True)
        for g0 in range(0, G, micro_bs):
            gs = slice(g0, min(g0 + micro_bs, G))
            comp_b = completion[gs].to(device)
            pm_b = None if prompt_mask is None else prompt_mask[gs]
            _, new_tok_lp = self.seq_logprob(prompt_ids, comp_b, prompt_mask=pm_b)
            old_tok_lp_b = old_tok_lp[gs].to(device)
            adv_b = advantages[gs].to(device)
            log_ratio = (new_tok_lp - old_tok_lp_b).clamp(-20, 20)  # (b, C)
            ratio = log_ratio.exp()
            un = ratio * adv_b[:, None]
            cl = ratio.clamp(1 - clip, 1 + clip) * adv_b[:, None]
            per_tok_loss = -torch.min(un, cl)               # (b, C)
            loss = per_tok_loss.mean(dim=1).sum() / G        # mean over tokens, sum over batch, normalize by G
            if not torch.isfinite(loss):
                any_nonfinite = True
                print(f'[ppo_update] WARNING: non-finite loss ({loss.item()}) '
                      f'in micro-batch [{g0}:{gs.stop}), skipping its backward')
                continue
            loss.backward()
            total += loss.item()
            n_terms += 1
        if any_nonfinite or n_terms == 0:
            print('[ppo_update] WARNING: skipping optimizer.step() this update '
                  'due to non-finite loss(es)')
            optimizer.zero_grad(set_to_none=True)
            return total / max(n_terms, 1)
        torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
        optimizer.step()
        return total / max(n_terms, 1)


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
    ap.add_argument('--inner-epochs', type=int, default=2)
    ap.add_argument('--p-mask', type=float, default=0.15)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--micro-bs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--prompt-file',
                    default=os.path.join(FK_DIR, 'evaluation',
                                         'pplm_discrim_prompts_orig.jsonl'))
    ap.add_argument('--out', default=os.path.expanduser('~/dllm/iadd-lm/runs'))
    ap.add_argument('--save-every', type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = D1Style(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None                       # train the raw weights
    model.backbone.train()

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f]

    run_name = (f"d1_{args.reward}_g{args.group}_s{args.seed}_"
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
        prompt_text = prompts[it % len(prompts)]
        enc = tokenizer([prompt_text], return_tensors='pt', padding=False)
        prompt_ids = enc['input_ids'][:, :-1].to(model.device)
        P = prompt_ids.shape[1]

        trace = model.rollout(prompt_ids, args.group, args.steps)
        final = trace['final']
        completion = final[:, P:].to(torch.long)

        trim = args.reward_trim + P
        texts = tokenizer.batch_decode(final[:, :trim])
        rewards = torch.tensor(
            compute_rewards(samples=texts, reward_name=args.reward,
                            reward_label=args.reward_label),
            dtype=torch.float32)

        r_mean, r_std = rewards.mean(), rewards.std()
        adv = rewards - r_mean
        if adv.abs().max() < 1e-6:
            log_f.write(json.dumps({'iter': it, 'r_mean': r_mean.item(),
                                    'skipped': True}) + '\n')
            log_f.flush()
        else:
            adv = adv / (r_std + 1e-6)

            # Prompt mask fixed for this iteration: shared by old (rollout-time)
            # and all inner-epoch (new) log-prob estimates, so the PPO ratio
            # reflects only the parameter update, not fresh masking noise.
            prompt_mask = torch.rand(args.group, P, device=model.device) < args.p_mask
            with torch.no_grad():
                _, old_tok_lp = model.seq_logprob(prompt_ids, completion,
                                                   prompt_mask=prompt_mask)
                old_tok_lp = old_tok_lp.float()

            loss = 0.0
            for _ in range(args.inner_epochs):
                loss = model.ppo_update(prompt_ids, completion, old_tok_lp, adv,
                                        optimizer, clip=args.clip,
                                        micro_bs=args.micro_bs,
                                        prompt_mask=prompt_mask)

            rec = {'iter': it, 'r_mean': rewards.mean().item(),
                   'r_max': rewards.max().item(), 'r_min': rewards.min().item(),
                   'loss': loss, 'inner_epochs': args.inner_epochs,
                   'sec': round(time.time() - t0, 1)}
            log_f.write(json.dumps(rec) + '\n')
            log_f.flush()
            print(rec)

        # Runs every --save-every iterations regardless of whether the
        # update above was skipped (zero-variance group), so a run that
        # skips most iterations still checkpoints.
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
