"""
FK branch/resample TRAINING rollouts for iADD-LM (Phase 1, Step 3).

Extends TraceGRPO: one prompt per iteration is rolled out with a shared
prefix, branched into k particles, and Feynman-Kac resampled at fixed
fractions of the trajectory using rewards computed on the free one-step
x0-fill (argmax over the logits already produced by that step's forward).
The recorded per-step trace follows each particle through resampling, so
PPO ratios stay exact for whatever lineage survives.

Run from anywhere:
  python ~/dllm/iadd-lm/fk_trace_grpo.py \
      --reward sentiment --reward-label positive \
      --steps 128 --gen-len 128 --k-particles 4 --iters 300 \
      --select incremental --select-n 8

Design notes:
- Shared prefix is rolled with batch 1 and replicated across particles when
  the final trace is assembled.  Prefix transitions therefore appear once
  per particle in the PPO loss; with group-relative (mean-centred)
  advantages their contributions cancel in expectation, and in --pair-mode
  (+A / -A) they cancel exactly - matching iADD's contrastive pairs.
- Resampling permutes particles; every post-branch per-step record is
  reindexed with the same indices, so each stored row remains the true
  history of the particle now occupying that slot.
- The potential is iADD's max form: w = exp(lmbda * max(R_hist, R_now)).
- --branch-frac 0.0 removes the shared prefix entirely: k independent
  prior samples are drawn instead of one prefix cloned into k particles,
  so lineages are distinct from step 0.
- --resample-mode selects the resampling scheme (multinomial [default,
  original behaviour] / residual / keep-distinct); see resample_idx().
- --adv-mode selects how advantages are formed (group [default, original
  mean-centred] / pair [alias: --pair-mode] / lineage [group-relative over
  distinct lineages only, tracked via a lin_id vector permuted alongside
  resampling]). Each resample event appends a line to
  <out_dir>/resample_stats.jsonl with distinct lineage counts before/after.
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.expanduser('~/dllm/iadd-lm'))

from trace_grpo import (  # noqa: E402  (also sets up FK_DIR paths + chdir)
    TraceGRPO, build_config, REWARD_NAMES, FK_DIR)

import dataloader  # noqa: E402
from fk_diffusion import _sample_categorical, compute_rewards  # noqa: E402


def resample_idx(w, k, mode='multinomial'):
    """Return a (k,) LongTensor of survivor indices for potential w (k,).

    mode:
      multinomial   - standard FK resampling, sample-with-replacement
                       proportional to w (original behaviour).
      residual      - floor(k*w_norm) deterministic copies per particle,
                       remaining slots filled by multinomial-without-
                       replacement on the residual weights.  Reduces
                       duplicate lineages relative to plain multinomial.
      keep-distinct - identity permutation except the single worst
                       particle, which is replaced by a copy of the
                       single best particle (at most one duplication per
                       resample event).  Guarantees >= k-1 distinct
                       lineages survive.
    """
    w = torch.clamp(w, 1e-20, 1e10)
    wn = w / w.sum()

    if mode == 'multinomial':
        return torch.multinomial(wn, k, replacement=True)

    if mode == 'residual':
        counts = torch.floor(wn * k).long()
        idx_list = []
        for i, c in enumerate(counts.tolist()):
            idx_list += [i] * c
        remainder = k - len(idx_list)
        if remainder > 0:
            resid_w = torch.clamp(wn * k - counts.float(), min=0.0)
            nonzero = int((resid_w > 0).sum().item())
            if nonzero >= remainder:
                extra = torch.multinomial(resid_w, remainder, replacement=False)
            else:
                # not enough distinct residual mass; fall back to
                # sampling-with-replacement so we still fill the batch
                safe_w = torch.clamp(resid_w, min=1e-12)
                extra = torch.multinomial(safe_w, remainder, replacement=True)
            idx_list += extra.tolist()
        return torch.tensor(idx_list, dtype=torch.long)

    if mode == 'keep-distinct':
        k_eff = w.shape[0]
        idx = torch.arange(k_eff)
        best = int(torch.argmax(w))
        worst = int(torch.argmin(w))
        if best != worst:
            idx[worst] = best
        return idx

    raise ValueError(f'unknown resample_mode: {mode}')


class FKTraceGRPO(TraceGRPO):
    """Trace-recording rollouts with FK branch/resample during training."""

    @torch.no_grad()
    def _step(self, z, t_val, dt):
        """One reverse step for a batch. Returns (z_new, log_p_x0, commit, lp, H)."""
        B = z.shape[0]
        t = t_val * torch.ones(B, 1, device=self.device)
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        sigma_t, sigma_s = sigma_t.squeeze(-1), sigma_s.squeeze(-1)
        mct = (1 - torch.exp(-sigma_t))[:, None, None]
        mcs = (1 - torch.exp(-sigma_s))[:, None, None]

        log_p_x0 = self.forward(z, sigma_t)          # (B, L, V) log-probs
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

        H = -(p_x0 * log_p_x0).sum(-1).float()
        H = torch.where(masked, H, torch.zeros_like(H))
        H = H.sum(-1) / masked.sum(-1).clamp(min=1)
        return z_new, log_p_x0, com, lp, H

    def _x0_fill(self, z, log_p_x0, n_samples):
        """Complete drafts from the current state without extra forwards."""
        fills = [torch.where(z == self.mask_index, log_p_x0.argmax(-1), z)]
        for _ in range(n_samples - 1):
            fills.append(torch.where(z == self.mask_index,
                                     _sample_categorical(log_p_x0.exp()), z))
        return fills

    @torch.no_grad()
    def rollout_fk(self, prompt_ids, num_steps, k_particles=4,
                   branch_frac=0.2, resample_fracs=(0.4, 0.6, 0.8),
                   lmbda=2.0, n_x0_samples=2, reward_name='sentiment',
                   reward_label='positive', reward_trim=100, eps=1e-5,
                   resample_mode='multinomial', resample_log_path=None,
                   iter_num=None, potential='max'):
        """FK rollout for ONE prompt. Returns (trace, final_rewards (k,))."""
        L = self.config.model.length
        device = self.device
        dt = (1 - eps) / num_steps
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        b_step = int(branch_frac * num_steps)
        res_steps = {int(f * num_steps) for f in resample_fracs}
        trim = reward_trim + (prompt_ids.shape[1] if prompt_ids is not None else 0)

        # branch_frac == 0.0 (b_step == 0): no shared prefix at all - start
        # from k independent prior samples so lineages are distinct from
        # step 0.  Otherwise: shared single-particle prefix, cloned into k
        # particles at i == b_step (original behaviour).
        if b_step == 0:
            z = self._sample_prior(k_particles, L).to(device)
            if prompt_ids is not None:
                z[:, :prompt_ids.shape[1]] = prompt_ids.expand(
                    k_particles, -1)
            lin_id = torch.arange(k_particles)
        else:
            z = self._sample_prior(1, L).to(device)
            if prompt_ids is not None:
                z[:, :prompt_ids.shape[1]] = prompt_ids
            lin_id = None

        pre_z, pre_lp, pre_com, pre_H = [], [], [], []
        post_z, post_lp, post_com, post_H = [], [], [], []
        R_hist = torch.full((k_particles,), -1e9)
        R_prev = None  # only used by potential == 'diff'
        log_p_x0 = None

        for i in range(num_steps + 1):
            branched = i >= b_step
            if i == b_step and b_step > 0:
                z = z.repeat(k_particles, 1)          # clone into k particles
                lin_id = torch.arange(k_particles)

            (pre_z if not branched else post_z).append(
                z.detach().to('cpu', torch.int32))
            z, log_p_x0, com, lp, H = self._step(z, timesteps[i], dt)
            (pre_lp if not branched else post_lp).append(lp.to('cpu', torch.float16))
            (pre_com if not branched else post_com).append(com.cpu())
            (pre_H if not branched else post_H).append(H.cpu())

            if branched and i in res_steps:
                # FK RESAMPLING POTENTIAL: computed from RAW continuous
                # x0-draft scores (`rs`), ALWAYS -- independent of any
                # --reward-threshold binarization applied later (in the
                # caller's main loop) to the RL advantage. Dense guidance
                # toward rare success via this raw-score potential is the
                # mechanism this experiment tests; it must never see the
                # binarized reward.
                texts, rs = [], []
                for fill in self._x0_fill(z, log_p_x0, n_x0_samples):
                    texts += self.tokenizer.batch_decode(fill[:, :trim])
                flat = compute_rewards(samples=texts, reward_name=reward_name,
                                       reward_label=reward_label)
                rs = torch.tensor(flat, dtype=torch.float32).reshape(
                    n_x0_samples, k_particles).mean(0)

                if potential == 'max':
                    # iADD's max form: w = exp(lmbda * max(R_hist, R_now)).
                    R_hist = torch.maximum(R_hist, rs)
                    w = torch.exp(lmbda * R_hist)
                else:
                    # telescoping 'diff' form: weight by the INCREMENT
                    # since this particle's previous resample event.
                    # First event has no previous event per particle, so
                    # R_prev is seeded with the common baseline (mean of
                    # R_now across particles) so first-event weights
                    # reflect relative standing only.
                    if R_prev is None:
                        R_prev = torch.full_like(rs, rs.mean())
                    w = torch.exp(lmbda * (rs - R_prev))

                idx = resample_idx(w, k_particles, mode=resample_mode)

                if lin_id is not None:
                    distinct_before = int(torch.unique(lin_id).numel())
                    lin_id = lin_id[idx]
                    distinct_after = int(torch.unique(lin_id).numel())
                    if resample_log_path is not None:
                        with open(resample_log_path, 'a') as rf:
                            rf.write(json.dumps({
                                'iter': iter_num, 'step': i,
                                'distinct_before': distinct_before,
                                'distinct_after': distinct_after,
                            }) + '\n')

                z = z[idx.to(device)]
                if potential == 'max':
                    R_hist = R_hist[idx]
                else:
                    # R_prev for the surviving particle at this slot
                    # becomes its just-observed raw score, permuted by
                    # the same resample index as everything else.
                    R_prev = rs[idx]
                idx_cpu = idx
                post_z = [x[idx_cpu] for x in post_z]
                post_lp = [x[idx_cpu] for x in post_lp]
                post_com = [x[idx_cpu] for x in post_com]
                post_H = [x[idx_cpu] for x in post_H]

        final = z.detach()
        texts = self.tokenizer.batch_decode(final[:, :trim])
        rewards = torch.tensor(
            compute_rewards(samples=texts, reward_name=reward_name,
                            reward_label=reward_label), dtype=torch.float32)

        k = k_particles
        # pre_z / pre_lp / ... are [] when b_step == 0 (no shared prefix);
        # the list-comprehension + concatenation below degenerates cleanly
        # to just the post-branch lists in that case.
        z_traj = torch.stack(
            [x.expand(k, -1) for x in pre_z] + post_z)          # (T+1, k, L)
        old_lp = [x.expand(k, -1) for x in pre_lp] + post_lp
        commit = [x.expand(k, -1) for x in pre_com] + post_com
        step_H = torch.stack(
            [x.expand(k) for x in pre_H] + post_H)              # (T+1, k)

        trace = dict(z_traj=z_traj, final=final, old_lp=old_lp,
                     commit=commit, step_H=step_H, lin_id=lin_id)
        return trace, rewards

    @staticmethod
    def subset_trace(trace, idx):
        """Restrict a trace to particle indices idx (list of ints)."""
        idx_t = torch.tensor(idx)
        out = dict(
            z_traj=trace['z_traj'][:, idx_t],
            final=trace['final'][idx_t],
            old_lp=[x[idx_t] for x in trace['old_lp']],
            commit=[x[idx_t] for x in trace['commit']],
            step_H=trace['step_H'][:, idx_t],
        )
        if trace.get('lin_id') is not None:
            out['lin_id'] = trace['lin_id'][idx_t]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='~/dllm/mdlm-owt-local')
    ap.add_argument('--reward', default='sentiment', choices=REWARD_NAMES)
    ap.add_argument('--reward-label', default='positive')
    ap.add_argument('--reward-trim', type=int, default=100)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--gen-len', type=int, default=128)
    ap.add_argument('--k-particles', type=int, default=4)
    ap.add_argument('--branch-frac', type=float, default=0.2)
    ap.add_argument('--resample-fracs', type=float, nargs='+',
                    default=[0.4, 0.6, 0.8])
    ap.add_argument('--lmbda', type=float, default=2.0)
    ap.add_argument('--potential', default='max', choices=['max', 'diff'],
                    help='max (default) = original iADD potential, '
                         'w = exp(lmbda * max(R_hist, R_now)), tracking a '
                         'running max of raw draft scores per particle. '
                         'diff = telescoping potential, w = exp(lmbda * '
                         '(R_now - R_prev)), where R_prev is that '
                         'particle raw draft score at the PREVIOUS '
                         'resample event (mean of R_now across particles '
                         'at the first event).')
    ap.add_argument('--n-x0-samples', type=int, default=2)
    ap.add_argument('--resample-mode', default='multinomial',
                    choices=['multinomial', 'residual', 'keep-distinct'],
                    help='multinomial = original FK resampling; residual '
                         'reduces duplicate lineages; keep-distinct kills '
                         'only the single worst particle per event')
    ap.add_argument('--adv-mode', default='group',
                    choices=['group', 'pair', 'lineage'],
                    help='group = mean-centred advantage over all k '
                         'particles (original); pair = best/worst +/-A '
                         '(same as --pair-mode); lineage = group-relative '
                         'advantage over DISTINCT lineages only, dropping '
                         'duplicate lineage slots from the trace')
    ap.add_argument('--pair-mode', action='store_true',
                    help='deprecated alias for --adv-mode pair')
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--select', default='all',
                    choices=['all', 'early', 'late', 'any', 'entropy',
                             'incremental'])
    ap.add_argument('--select-n', type=int, default=16)
    ap.add_argument('--micro-bs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--reward-threshold', type=float, default=None,
                    help='SPARSE/RARE-SUCCESS mode. When set, the RL reward '
                         'used for the PPO advantage (group / pair / '
                         'lineage) is binarized: 1.0 if the raw final-'
                         'sequence reward score >= threshold else 0.0. The '
                         'zero-variance-advantage skip is evaluated on this '
                         'BINARY reward and logged explicitly (n_success / '
                         'k_particles / skipped). CRITICAL: this does NOT '
                         'touch the FK resampling potential -- R_hist / '
                         'w = exp(lmbda * max(R_hist, R_now)) inside '
                         'rollout_fk is always computed from the RAW '
                         'continuous x0-draft scores, regardless of this '
                         'flag, because dense guidance toward rare success '
                         'is exactly the mechanism under test. Raw scores '
                         'are still logged via r_mean/r_best/r_worst.')
    ap.add_argument('--prompt-file',
                    default=os.path.join(FK_DIR, 'evaluation',
                                         'pplm_discrim_prompts_orig.jsonl'))
    ap.add_argument('--out', default=os.path.expanduser('~/dllm/iadd-lm/runs'))
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--init-ckpt', default=None,
                    help='Optional path to a backbone_itN.pt state dict '
                         '(as saved by this script / trace_grpo.py) to '
                         'load into the freshly-constructed model before '
                         'any rollouts, loaded the same way eval.py loads '
                         'checkpoints (torch.load + backbone.load_state_dict, '
                         'strict).')
    ap.add_argument('--eval-only', type=int, default=0,
                    help='If > 0, run this many FK rollouts (rollout_fk, '
                         'using the run\'s normal args) with NO optimizer '
                         'step / ppo_update, log per-rollout final rewards '
                         '+ n_success (same --reward-threshold logic as '
                         'training) to log.jsonl in a run dir tagged '
                         '"evalonly", then exit. Default 0 = normal '
                         'training behaviour, unchanged.')
    ap.add_argument('--run-tag', default=None,
                    help='Optional override for the run directory name '
                         'prefix (before the timestamp). Only affects the '
                         'run directory name.')
    ap.add_argument('--eval-save-texts', action='store_true',
                    help='Only meaningful with --eval-only > 0. When set, '
                         'each eval-only rollout also decodes every '
                         'particle\'s final sequence to text (same trim '
                         'convention as the reward computation inside '
                         'rollout_fk) and appends one JSONL line '
                         '{rollout, prompt_idx, texts, raw_rewards, '
                         'best_idx} to finals.jsonl in the run dir. '
                         'Default False = no change to eval-only behaviour.')
    args = ap.parse_args()

    if args.pair_mode:
        args.adv_mode = 'pair'

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = FKTraceGRPO(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None
    model.backbone.train()

    if args.init_ckpt is not None:
        # Same loading convention as eval.py: plain torch.load + strict
        # backbone.load_state_dict, no key remapping.
        ckpt_path = os.path.expanduser(args.init_ckpt)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.expanduser('~/dllm/iadd-lm'),
                                     ckpt_path)
        sd = torch.load(ckpt_path, map_location='cuda')
        model.backbone.load_state_dict(sd)

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f]

    mode = {'group': 'grp', 'pair': 'pair', 'lineage': 'lin'}[args.adv_mode]
    if args.run_tag is not None:
        run_name = args.run_tag + '_' + datetime.now().strftime('%m%d-%H%M')
    elif args.eval_only > 0:
        run_name = (f"evalonly_{args.reward}_{mode}_k{args.k_particles}"
                    f"_s{args.seed}_" + datetime.now().strftime('%m%d-%H%M'))
    else:
        run_name = (f"fk_{args.reward}_{args.select}{args.select_n}"
                    f"_k{args.k_particles}_{mode}_s{args.seed}_"
                    + datetime.now().strftime('%m%d-%H%M'))
    out_dir = os.path.join(os.path.expanduser(args.out), run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    log_f = open(os.path.join(out_dir, 'log.jsonl'), 'a')
    resample_log_path = os.path.join(out_dir, 'resample_stats.jsonl')

    if args.eval_only > 0:
        # EVAL-ONLY MODE: N FK rollouts with the run's normal rollout_fk
        # args, NO optimizer / ppo_update. Model stays in eval + no-grad
        # for all rollouts (rollout_fk is itself @torch.no_grad()).
        model.backbone.eval()
        sparse = args.reward_threshold is not None
        n_success_total = 0
        n_finals_total = 0
        raw_reward_sum = 0.0
        finals_f = None
        if args.eval_save_texts:
            finals_f = open(os.path.join(out_dir, 'finals.jsonl'), 'a')
        with torch.no_grad():
            for it in range(args.eval_only):
                t0 = time.time()
                prompt_idx = it % len(prompts)
                prompt_text = prompts[prompt_idx]
                enc = tokenizer([prompt_text], return_tensors='pt',
                                padding=False)
                prompt_ids = enc['input_ids'][:, :-1].to(model.device)

                trace, raw_rewards = model.rollout_fk(
                    prompt_ids, args.steps, k_particles=args.k_particles,
                    branch_frac=args.branch_frac,
                    resample_fracs=tuple(args.resample_fracs),
                    lmbda=args.lmbda, n_x0_samples=args.n_x0_samples,
                    reward_name=args.reward, reward_label=args.reward_label,
                    reward_trim=args.reward_trim,
                    resample_mode=args.resample_mode,
                    resample_log_path=resample_log_path, iter_num=it,
                    potential=args.potential)

                rec = {'iter': it, 'eval_only': True,
                       'r_mean': raw_rewards.mean().item(),
                       'r_best': raw_rewards.max().item(),
                       'r_worst': raw_rewards.min().item(),
                       'raw_rewards': raw_rewards.tolist(),
                       'k_particles': args.k_particles,
                       'sec': round(time.time() - t0, 1)}
                if sparse:
                    success = (raw_rewards >= args.reward_threshold).float()
                    n_success = int(success.sum().item())
                    rec.update(sparse=True, n_success=n_success,
                               reward_threshold=args.reward_threshold)
                    n_success_total += n_success
                    n_finals_total += args.k_particles
                raw_reward_sum += raw_rewards.sum().item()
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                print(rec)

                if finals_f is not None:
                    trim = args.reward_trim + prompt_ids.shape[1]
                    texts = tokenizer.batch_decode(trace['final'][:, :trim])
                    finals_rec = {
                        'rollout': it,
                        'prompt_idx': prompt_idx,
                        'texts': texts,
                        'raw_rewards': raw_rewards.tolist(),
                        'best_idx': int(raw_rewards.argmax()),
                    }
                    finals_f.write(json.dumps(finals_rec) + '\n')
                    finals_f.flush()

        if finals_f is not None:
            finals_f.close()

        summary = {'eval_only_summary': True,
                   'n_rollouts': args.eval_only,
                   'k_particles': args.k_particles,
                   'n_finals': args.eval_only * args.k_particles,
                   'raw_reward_mean': raw_reward_sum
                                      / max(args.eval_only * args.k_particles, 1)}
        if sparse:
            summary.update(n_success_total=n_success_total,
                           n_finals_total=n_finals_total,
                           success_frac=n_success_total
                                        / max(n_finals_total, 1))
        log_f.write(json.dumps(summary) + '\n')
        log_f.flush()
        log_f.close()
        print(summary)
        print('eval-only done:', out_dir)
        return

    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=args.lr,
                                  weight_decay=0.0)

    for it in range(args.iters):
        t0 = time.time()
        prompt_text = prompts[it % len(prompts)]
        enc = tokenizer([prompt_text], return_tensors='pt', padding=False)
        prompt_ids = enc['input_ids'][:, :-1].to(model.device)

        # raw_rewards: RAW continuous final-sequence scores. rollout_fk's
        # internal FK resampling potential (R_hist / w = exp(lmbda *
        # max(R_hist, R_now))) is computed separately, INSIDE rollout_fk,
        # from raw x0-draft scores -- it is never touched by
        # --reward-threshold. Only the advantage below is binarized.
        trace, raw_rewards = model.rollout_fk(
            prompt_ids, args.steps, k_particles=args.k_particles,
            branch_frac=args.branch_frac,
            resample_fracs=tuple(args.resample_fracs), lmbda=args.lmbda,
            n_x0_samples=args.n_x0_samples, reward_name=args.reward,
            reward_label=args.reward_label, reward_trim=args.reward_trim,
            resample_mode=args.resample_mode,
            resample_log_path=resample_log_path, iter_num=it,
            potential=args.potential)

        sparse = args.reward_threshold is not None
        if sparse:
            # SPARSE mode: the RL reward driving the advantage (group /
            # pair / lineage, all below) is the binarized indicator
            # 1{raw_score >= threshold}, NOT the raw continuous score.
            rewards = (raw_rewards >= args.reward_threshold).float()
            n_success = int(rewards.sum().item())
        else:
            rewards = raw_rewards
            n_success = None

        if args.adv_mode == 'pair':
            best = int(rewards.argmax())
            worst = int(rewards.argmin())
            if best == worst:
                rec = {'iter': it, 'skipped': True}
                if sparse:
                    rec.update(sparse=True, n_success=n_success,
                               k_particles=args.k_particles)
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                continue
            scale = (rewards[best] - rewards[worst]).item() / 2
            trace = model.subset_trace(trace, [best, worst])
            adv = torch.tensor([scale, -scale])
        elif args.adv_mode == 'lineage':
            lin_id = trace['lin_id']
            uniq_ids = torch.unique(lin_id)
            rep_idx, lineage_rewards = [], []
            for u in uniq_ids.tolist():
                members = (lin_id == u).nonzero(as_tuple=True)[0]
                rep_idx.append(int(members[0]))
                lineage_rewards.append(rewards[members].mean().item())
            lineage_rewards = torch.tensor(lineage_rewards)
            adv = lineage_rewards - lineage_rewards.mean()
            if adv.abs().max() < 1e-6:
                rec = {'iter': it, 'skipped': True,
                       'n_distinct_lineages': len(rep_idx)}
                if sparse:
                    rec.update(sparse=True, n_success=n_success,
                               k_particles=args.k_particles)
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                continue
            adv = adv / (lineage_rewards.std() + 1e-6)
            trace = model.subset_trace(trace, rep_idx)
        else:
            adv = rewards - rewards.mean()
            if adv.abs().max() < 1e-6:
                rec = {'iter': it, 'skipped': True}
                if sparse:
                    rec.update(sparse=True, n_success=n_success,
                               k_particles=args.k_particles)
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()
                continue
            adv = adv / (rewards.std() + 1e-6)

        S = model.select_steps(args.select, args.steps, args.select_n,
                               it, args.iters, step_H=trace['step_H'])
        loss = model.ppo_update(trace, adv, S, optimizer,
                                micro_bs=args.micro_bs, num_steps=args.steps)

        rec = {'iter': it, 'r_mean': raw_rewards.mean().item(),
               'r_best': raw_rewards.max().item(),
               'r_worst': raw_rewards.min().item(),
               'loss': loss, 'n_steps_trained': len(S),
               'sec': round(time.time() - t0, 1)}
        if sparse:
            rec.update(sparse=True, n_success=n_success,
                       k_particles=args.k_particles, skipped=False)
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()
        print(rec)

        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            ckpt = os.path.join(out_dir, f'backbone_it{it+1}.pt')
            torch.save(model.backbone.state_dict(), ckpt)
            # keep best + last only: prune older
            # (IADD_KEEP_ALL_CKPTS=1 retains every checkpoint for curve evals)
            if os.environ.get("IADD_KEEP_ALL_CKPTS") != "1":
                saved = sorted([f for f in os.listdir(out_dir)
                                if f.startswith('backbone_it')],
                               key=lambda f: int(f[len('backbone_it'):-3]))
                for f_old in saved[:-2]:
                    os.remove(os.path.join(out_dir, f_old))

    log_f.close()
    print('done:', out_dir)


if __name__ == '__main__':
    main()
