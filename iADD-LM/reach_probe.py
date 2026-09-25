"""
Reach probe for iADD-LM (Phase 1, Step 5) — empirical backing for the
discrete Prop. 1 ("generative reach").

For rollouts of the BASE model we record the trace, then at strided probe
steps t: take the recorded post-step state z_{t+1}, flip ONE token that was
committed at step t to the model's 2nd choice (from a forward at z_t), and
continue the remaining denoising DETERMINISTICALLY (argmax of the same
reverse rule) for both the perturbed state and an unperturbed control.
reach(t) := mean Hamming distance between the two final sequences over the
completion region, excluding the flipped position itself (downstream
influence only).

Hypothesis: early steps (low i = high mask rate) have larger reach.

  python ~/dllm/iadd-lm/reach_probe.py --prompts 5 --group 4 --steps 128 --stride 8
"""
import argparse
import json
import os
import sys

import torch

FK_DIR = os.path.expanduser('~/dllm/Fk-Diffusion-Steering/discrete_diffusion')
sys.path.insert(0, FK_DIR)
sys.path.insert(0, os.path.join(FK_DIR, 'mdlm'))
sys.path.insert(0, os.path.expanduser('~/dllm/iadd-lm'))
os.chdir(FK_DIR)

import dataloader  # noqa: E402
from trace_grpo import TraceGRPO, build_config  # noqa: E402


@torch.no_grad()
def deterministic_continue(model, z, start_i, num_steps, eps=1e-5):
    """Continue denoising from state z (BEFORE step start_i) with the same
    reverse rule as rollout() but argmax instead of sampling. Returns final z."""
    device = model.device
    B = z.shape[0]
    dt = (1 - eps) / num_steps
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    for i in range(start_i, num_steps + 1):
        t = timesteps[i] * torch.ones(B, 1, device=device)
        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        sigma_t, sigma_s = sigma_t.squeeze(-1), sigma_s.squeeze(-1)
        mct = (1 - torch.exp(-sigma_t))[:, None, None]
        mcs = (1 - torch.exp(-sigma_s))[:, None, None]
        log_p_x0 = model.forward(z, sigma_t)
        q_xs = log_p_x0.exp() * (mct - mcs)
        q_xs[:, :, model.mask_index] = mcs[:, :, 0]
        _x = q_xs.argmax(-1)
        z = torch.where(z == model.mask_index, _x, z)
    if (z == model.mask_index).any():        # residual masks: argmax fill
        t = timesteps[-1] * torch.ones(B, 1, device=device)
        sigma_t, _ = model.noise(t)
        log_p_x0 = model.forward(z, sigma_t.squeeze(-1))
        fill = log_p_x0.argmax(-1)
        z = torch.where(z == model.mask_index, fill, z)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='~/dllm/mdlm-owt-local')
    ap.add_argument('--reward', default='sentiment')        # unused; build_config needs it
    ap.add_argument('--reward-label', default='positive')
    ap.add_argument('--reward-trim', type=int, default=100)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--gen-len', type=int, default=128)
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--prompts', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--prompt-file', default=os.path.join(
        FK_DIR, 'evaluation', 'pplm_discrim_prompts_orig.jsonl'))
    ap.add_argument('--out', default=os.path.expanduser('~/dllm/iadd-lm/reach.json'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = TraceGRPO(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None
    model.backbone.eval()
    device = model.device

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f][:args.prompts]

    probe_steps = list(range(0, args.steps, args.stride))
    reach = {t: [] for t in probe_steps}          # t -> list of per-sample distances

    for p_i, prompt_text in enumerate(prompts):
        enc = tokenizer([prompt_text], return_tensors='pt', padding=False)
        prompt_ids = enc['input_ids'][:, :-1].to(device)
        p_len = prompt_ids.shape[1]

        trace = model.rollout(prompt_ids, args.group, args.steps)

        for t in probe_steps:
            com = trace['commit'][t]              # (G, L) bool cpu
            z_t = trace['z_traj'][t].to(device, torch.long)
            z_next = trace['z_traj'][t + 1].to(device, torch.long)

            # pick one committed completion-region position per sample
            g_idx, pos_idx = [], []
            for g in range(args.group):
                cand = com[g].nonzero().flatten()
                cand = cand[cand >= p_len]
                if len(cand):
                    g_idx.append(g)
                    pos_idx.append(int(cand[torch.randint(len(cand), (1,))]))
            if not g_idx:
                continue
            g_idx_t = torch.tensor(g_idx, device=device)

            # 2nd-choice token at the flipped position, from a forward at z_t
            tt = torch.linspace(1, 1e-5, args.steps + 1, device=device)[t]
            sigma_t, _ = model.noise(tt * torch.ones(len(g_idx), 1, device=device))
            log_p = model.forward(z_t[g_idx_t], sigma_t.squeeze(-1))
            top2 = log_p.topk(2, dim=-1).indices                # (n, L, 2)

            ctrl = z_next[g_idx_t].clone()
            pert = z_next[g_idx_t].clone()
            for j, pos in enumerate(pos_idx):
                first, second = top2[j, pos, 0], top2[j, pos, 1]
                new_tok = second if ctrl[j, pos] == first else first
                pert[j, pos] = new_tok

            both = torch.cat([ctrl, pert], dim=0)
            final = deterministic_continue(model, both, t + 1, args.steps)
            n = len(g_idx)
            f_ctrl, f_pert = final[:n], final[n:]

            for j, pos in enumerate(pos_idx):
                diff = (f_ctrl[j, p_len:] != f_pert[j, p_len:])
                rel = pos - p_len
                if 0 <= rel < diff.shape[0]:
                    diff[rel] = False                           # exclude flipped pos
                reach[t].append(int(diff.sum()))
        print(f'prompt {p_i + 1}/{len(prompts)} done', flush=True)

    ts = [t for t in probe_steps if reach[t]]
    means = [sum(reach[t]) / len(reach[t]) for t in ts]
    result = dict(steps=args.steps, stride=args.stride, group=args.group,
                  prompts=len(prompts), t=ts, reach_mean=means,
                  n_per_t=[len(reach[t]) for t in ts],
                  raw={str(t): reach[t] for t in ts})
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'step t':>8} {'mask%':>7} {'reach (tokens)':>15} {'n':>4}")
    for t, m in zip(ts, means):
        mask_frac = 1 - t / args.steps                          # approx: linear schedule
        print(f'{t:>8} {100 * mask_frac:>6.0f}% {m:>15.2f} {len(reach[t]):>4}')
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
