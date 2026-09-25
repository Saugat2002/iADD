"""
Trace-based GRPO for MDLM (iADD-LM Phase 1, Step 1).

Records the actual denoising trace (which tokens were committed at which step,
with their exact categorical log-probs) and runs PPO-clip on a selectable
subset S of steps. Step-selection rules: all / early / late / any / entropy /
incremental.  FK branch/resample rollouts arrive in Step 3 (this file keeps
G independent rollouts per prompt = plain trace-GRPO).

Run from Fk-Diffusion-Steering/discrete_diffusion/:
  python ~/dllm/iadd-lm/trace_grpo.py \
      --reward sentiment --reward-label positive \
      --steps 128 --gen-len 128 --group 8 --iters 300 --select all

Design notes (why this is correct):
- MDLM reverse step: for a currently-masked position, P(commit v) =
  p_x0(v)*(mct-mcs)/mct and P(stay masked) = mcs/mct.  The stay-masked branch
  and the (mct-mcs)/mct factor are schedule-only (no theta), so the
  theta-dependent log-prob of a realized step is exactly
      sum_{i committed at step} log p_x0(v_i | z_t)
  which is what we record and what PPO ratios are computed on.
- Prompt tokens are pre-filled (copy_flag=1) and never contribute.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F

FK_DIR = os.path.expanduser('~/dllm/Fk-Diffusion-Steering/discrete_diffusion')
sys.path.insert(0, FK_DIR)
sys.path.insert(0, os.path.join(FK_DIR, 'mdlm'))
os.chdir(FK_DIR)  # hydra searchpath file://mdlm/configs is cwd-relative

import dataloader  # noqa: E402  (mdlm)
from fk_diffusion import (  # noqa: E402
    FKDiffusion, _sample_categorical, compute_rewards)

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

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


class TraceGRPO(FKDiffusion):
    """Adds trace-recording rollouts and a PPO update over selected steps."""

    # ---------- rollout ----------
    @torch.no_grad()
    def rollout(self, prompt_ids, group_size, num_steps, eps=1e-5):
        """Sample `group_size` sequences; return final z + per-step trace.

        Returns dict with:
          z_traj  : (T+1, G, L) int32 cpu  -- state BEFORE each step i
          final   : (G, L) long           -- final sequences
          old_lp  : list length T of (G, L) float16 cpu -- log p_x0 of the
                    token committed at that step, 0 where nothing committed
          commit  : list length T of (G, L) bool cpu    -- committed mask
          step_H  : (T, G) float cpu      -- mean entropy over masked pos.
        """
        G, L = group_size, self.config.model.length
        device = self.device
        dt = (1 - eps) / num_steps
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)

        z = self._sample_prior(G, L).to(device)
        if prompt_ids is not None:
            z[:, :prompt_ids.shape[1]] = prompt_ids

        z_traj, old_lp, commit, step_H = [], [], [], []
        for i in range(num_steps + 1):
            t = timesteps[i] * torch.ones(G, 1, device=device)
            z_traj.append(z.detach().to('cpu', torch.int32))

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

    # ---------- step selection ----------
    @staticmethod
    def select_steps(rule, T, N, it, iters, step_H=None):
        """Return sorted list of step indices in [0, T) to train on."""
        if rule == 'all':
            return list(range(T))
        if rule == 'early':          # early in DENOISING = high mask rate = small i
            return list(range(N))
        if rule == 'late':
            return list(range(T - N, T))
        if rule == 'any':
            return sorted(random.sample(range(T), N))
        if rule == 'entropy':        # EGSPO-style: top-N mean entropy
            assert step_H is not None
            # step_H has T+1 rows; only steps < T are trainable (need z_traj[t+1])
            return sorted(step_H[:T].mean(dim=1).topk(min(N, T)).indices.tolist())
        if rule == 'incremental':    # iADD: grow N over training, sample any-N
            stages = [N, 2 * N, 4 * N, 8 * N]
            n = stages[min(int(it / max(iters, 1) * len(stages)), len(stages) - 1)]
            n = min(n, T)
            return sorted(random.sample(range(T), n))
        if rule == 'ent_inc':        # hybrid: incremental budget, entropy-guided pick
            assert step_H is not None
            stages = [N, 2 * N, 4 * N, 8 * N]
            n = stages[min(int(it / max(iters, 1) * len(stages)), len(stages) - 1)]
            n = min(n, T)
            return sorted(step_H[:T].mean(dim=1).topk(n).indices.tolist())
        raise ValueError(rule)

    # ---------- update ----------
    def ppo_update(self, trace, advantages, steps_S, optimizer,
                   clip=0.2, micro_bs=4, num_steps=128, eps=1e-5):
        """One PPO epoch over selected steps. advantages: (G,) tensor."""
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
                    choices=['all', 'early', 'late', 'any', 'entropy', 'incremental', 'ent_inc'])
    ap.add_argument('--select-n', type=int, default=16)
    ap.add_argument('--micro-bs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--reward-threshold', type=float, default=None,
                    help='SPARSE/RARE-SUCCESS mode. When set, the RL reward '
                         'used for advantage computation is binarized: 1.0 '
                         'if the raw reward score >= threshold else 0.0. '
                         'The zero-variance-advantage skip (all-success or '
                         'all-failure group) is evaluated on this BINARY '
                         'reward -- that skipping is the starvation '
                         'phenomenon under study, and is logged explicitly '
                         '(n_success / group_size / skipped). Raw '
                         'continuous scores are still logged via '
                         'r_mean/r_max/r_min for diagnostics; they are NOT '
                         'used for the advantage in this mode.')
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
    model = TraceGRPO(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None                       # train the raw weights
    model.backbone.train()

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f]

    run_name = (f"{args.reward}_{args.select}{args.select_n}"
                f"_g{args.group}_s{args.seed}_"
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

        trace = model.rollout(prompt_ids, args.group, args.steps)

        trim = args.reward_trim + prompt_ids.shape[1]
        texts = tokenizer.batch_decode(trace['final'][:, :trim])
        # raw_rewards: continuous reward score, always the ground truth for
        # logging/diagnostics.
        raw_rewards = torch.tensor(
            compute_rewards(samples=texts, reward_name=args.reward,
                            reward_label=args.reward_label),
            dtype=torch.float32)

        sparse = args.reward_threshold is not None
        if sparse:
            # SPARSE mode: the RL reward driving the advantage is the
            # binarized indicator 1{raw_score >= threshold}, NOT the raw
            # continuous score. This is the "rare success" regime: a group
            # is only informative if at least one (but not all) samples
            # cleared the threshold.
            rl_rewards = (raw_rewards >= args.reward_threshold).float()
            n_success = int(rl_rewards.sum().item())
        else:
            rl_rewards = raw_rewards
            n_success = None

        adv = rl_rewards - rl_rewards.mean()
        if adv.abs().max() < 1e-6:
            rec = {'iter': it, 'r_mean': raw_rewards.mean().item(),
                   'skipped': True}
            if sparse:
                rec.update(sparse=True, n_success=n_success,
                           group_size=args.group)
            log_f.write(json.dumps(rec) + '\n')
            log_f.flush()
            continue
        adv = adv / (rl_rewards.std() + 1e-6)

        S = model.select_steps(args.select, args.steps, args.select_n,
                               it, args.iters, step_H=trace['step_H'])
        loss = model.ppo_update(trace, adv, S, optimizer,
                                micro_bs=args.micro_bs, num_steps=args.steps)

        rec = {'iter': it, 'r_mean': raw_rewards.mean().item(),
               'r_max': raw_rewards.max().item(), 'r_min': raw_rewards.min().item(),
               'loss': loss, 'n_steps_trained': len(S),
               'sec': round(time.time() - t0, 1)}
        if sparse:
            rec.update(sparse=True, n_success=n_success, group_size=args.group,
                       skipped=False)
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()
        print(rec)

        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            ckpt = os.path.join(out_dir, f'backbone_it{it+1}.pt')
            torch.save(model.backbone.state_dict(), ckpt)
            # keep best + last only: prune older
            # (IADD_KEEP_ALL_CKPTS=1 retains every checkpoint for curve evals)
            if os.environ.get("IADD_KEEP_ALL_CKPTS") != "1":
                saved = sorted(f for f in os.listdir(out_dir)
                               if f.startswith('backbone_it'))
                for f_old in saved[:-2]:
                    os.remove(os.path.join(out_dir, f_old))

    log_f.close()
    print('done:', out_dir)


if __name__ == '__main__':
    main()
