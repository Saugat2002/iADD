"""
DM-style (DMPO/TraFL) distribution-matching baseline for MDLM (iADD-LM Phase 1,
diversity family).

Where TraceGRPO (trace_grpo.py) does a mode-seeking, reverse-KL-flavored PPO
update over the denoising trace, this trains the *forward-KL* / mass-covering
update: pull p_theta toward the reward-tilted mixture

    q(y) = softmax_i(r_i / alpha)  ~  Sum_i w_i * delta(y_i)

by taking a group of G on-policy rollouts for one prompt, weighting each
completion by its softmax-reward weight w_i, and doing one step of ordinary
masked-diffusion cross-entropy (the standard continuous-time MDLM/SUBS loss,
CE weighted by 1/t) on each completion, scaled by w_i. Summed over the group
this is a weighted MLE step toward the reward-tilted empirical distribution,
i.e. forward KL / distribution matching rather than a policy-gradient ratio
update. This is the "honest simplified" version: no learned partition
function, no replay buffer, no importance-reweighting correction for the
completions having been sampled from an earlier theta -- just fresh on-policy
rollouts each iteration.

Run from Fk-Diffusion-Steering/discrete_diffusion/:
  python ~/dllm/iadd-lm/dm_style.py \
      --reward sentiment --reward-label positive \
      --steps 128 --gen-len 128 --group 8 --iters 300 --alpha 0.5
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
from fk_diffusion import compute_rewards  # noqa: E402

sys.path.insert(0, os.path.expanduser('~/dllm/iadd-lm'))
from trace_grpo import TraceGRPO, build_config, REWARD_NAMES  # noqa: E402

# Fixed diffusion-time mask-rate range for the DM loss (matches the task
# spec's t ~ U(0.05, 0.95); not exposed as a CLI knob to keep this an honest,
# minimal baseline rather than a second hyperparameter surface).
T_LOW, T_HIGH = 0.05, 0.95


def dm_update(model, prompt_len, final_seqs, weights, optimizer, micro_bs=4):
    """One weighted-MLE (forward-KL) update over a group of completions.

    final_seqs: (G, L) long, full sequences (prompt + completion), cpu/gpu.
    weights:    (G,) float, per-sequence group weights (need not sum to 1
                if --center was used).

    For each sequence i: draw t_i ~ U(T_LOW, T_HIGH), mask each *completion*
    token independently with prob t_i (prompt tokens are never masked), do
    one forward pass, take masked-diffusion cross-entropy on the masked
    completion tokens, weight by 1/t_i (standard continuous-time MDLM/SUBS
    NLL weighting), average over the masked tokens in that sequence to get a
    per-sequence loss, then scale by w_i and sum over the group.
    """
    device = model.device
    G, L = final_seqs.shape
    comp_len = L - prompt_len
    assert comp_len > 0

    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for g0 in range(0, G, micro_bs):
        gs = slice(g0, min(g0 + micro_bs, G))
        x0 = final_seqs[gs].to(device, torch.long)
        bs = x0.shape[0]

        t = torch.empty(bs, 1, device=device).uniform_(T_LOW, T_HIGH)

        comp_region = torch.zeros(bs, L, dtype=torch.bool, device=device)
        comp_region[:, prompt_len:] = True
        mask_pos = (torch.rand(bs, L, device=device) < t) & comp_region
        # guard against a (rare, low-probability) all-clean draw so every
        # sequence contributes at least one supervised token
        empty = ~mask_pos.any(dim=-1)
        if empty.any():
            fallback_idx = prompt_len + torch.randint(
                0, comp_len, (bs,), device=device)
            mask_pos[empty, fallback_idx[empty]] = True

        xt = torch.where(mask_pos, torch.full_like(x0, model.mask_index), x0)
        sigma_t, _ = model.noise(t)
        log_p_x0 = model.forward(xt, sigma_t.squeeze(-1))  # (bs, L, V)
        ce = -log_p_x0.gather(-1, x0.unsqueeze(-1)).squeeze(-1).float()

        ce_masked = ce * mask_pos
        n_masked = mask_pos.sum(dim=-1).clamp(min=1)
        seq_ce = ce_masked.sum(dim=-1) / n_masked          # mean CE, masked tokens
        seq_loss = seq_ce / t.squeeze(-1)                  # 1/t weighting

        w = weights[gs].to(device)
        loss = (w * seq_loss).sum()
        loss.backward()
        total += loss.item()

    torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
    optimizer.step()
    return total


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
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='softmax temperature for reward-tilted weights '
                         'w_i = softmax(r_i / alpha)')
    ap.add_argument('--center', action='store_true',
                    help='subtract the uniform baseline 1/G from the '
                         'softmax weights before applying them (allows '
                         'negative weights)')
    ap.add_argument('--center-clamp', action='store_true',
                    help='with --center, clamp the centered weights at 0 '
                         '(default off: negative weights are kept)')
    ap.add_argument('--micro-bs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--prompt-file',
                    default=os.path.join(FK_DIR, 'evaluation',
                                         'pplm_discrim_prompts_orig.jsonl'))
    ap.add_argument('--out', default=os.path.expanduser('~/dllm/iadd-lm/runs'))
    ap.add_argument('--save-every', type=int, default=50)
    args = ap.parse_args()

    if args.center_clamp and not args.center:
        ap.error('--center-clamp requires --center')

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = TraceGRPO(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None                       # train the raw weights
    model.backbone.train()

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f]

    run_name = (f"dm_{args.reward}_a{args.alpha}_g{args.group}_s{args.seed}_"
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
        prompt_len = prompt_ids.shape[1]

        trace = model.rollout(prompt_ids, args.group, args.steps)
        final_seqs = trace['final']  # (G, L) long, on model.device already

        trim = args.reward_trim + prompt_len
        texts = tokenizer.batch_decode(final_seqs[:, :trim])
        rewards = torch.tensor(
            compute_rewards(samples=texts, reward_name=args.reward,
                            reward_label=args.reward_label),
            dtype=torch.float32)

        w = torch.softmax(rewards / args.alpha, dim=0)
        if args.center:
            w = w - 1.0 / args.group
            if args.center_clamp:
                w = w.clamp(min=0.0)

        loss = dm_update(model, prompt_len, final_seqs, w, optimizer,
                         micro_bs=args.micro_bs)

        rec = {'iter': it, 'r_mean': rewards.mean().item(),
               'r_max': rewards.max().item(), 'r_min': rewards.min().item(),
               'w_mean': w.mean().item(), 'w_max': w.max().item(),
               'w_min': w.min().item(), 'loss': loss,
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
