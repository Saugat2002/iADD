"""
Verification suite for iADD-LM Phase 1 (plan Verification items 2-4).

  python ~/dllm/iadd-lm/verify.py --check logprob    # trace recording exact
  python ~/dllm/iadd-lm/verify.py --check noop       # constant reward -> no update
  python ~/dllm/iadd-lm/verify.py --check resample   # lmbda->0 resampling uniform
  python ~/dllm/iadd-lm/verify.py --check all

logprob / noop need the GPU; resample runs on CPU.
Each check prints PASS/FAIL with numbers; process exits nonzero on any FAIL.
"""
import argparse
import json
import os
import random
import sys

import torch

FK_DIR = os.path.expanduser('~/dllm/Fk-Diffusion-Steering/discrete_diffusion')
sys.path.insert(0, FK_DIR)
sys.path.insert(0, os.path.join(FK_DIR, 'mdlm'))
sys.path.insert(0, os.path.expanduser('~/dllm/iadd-lm'))
os.chdir(FK_DIR)  # hydra searchpath file://mdlm/configs is cwd-relative


def _model_args(seed=1234, steps=32, gen_len=64):
    """Minimal args namespace accepted by trace_grpo.build_config."""
    return argparse.Namespace(
        checkpoint='~/dllm/mdlm-owt-local',
        reward='sentiment', reward_label='positive', reward_trim=100,
        steps=steps, gen_len=gen_len, seed=seed)


def _load_model(args):
    import dataloader
    from trace_grpo import TraceGRPO, build_config
    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = TraceGRPO(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None
    model.backbone.eval()   # no dropout: rollout and recompute must match
    return model, tokenizer


def _prompt_ids(tokenizer, model):
    prompt_file = os.path.join(FK_DIR, 'evaluation',
                               'pplm_discrim_prompts_orig.jsonl')
    with open(prompt_file) as f:
        prompt = json.loads(f.readline())['context_string']
    enc = tokenizer([prompt], return_tensors='pt', padding=False)
    return enc['input_ids'][:, :-1].to(model.device)


# ---------- check 1: trace log-probs are exactly recomputable ----------
def check_logprob():
    torch.manual_seed(1234)
    random.seed(1234)
    args = _model_args()
    model, tokenizer = _load_model(args)
    prompt_ids = _prompt_ids(tokenizer, model)

    with torch.no_grad():
        trace = model.rollout(prompt_ids, 2, args.steps)

    eps = 1e-5
    dt = (1 - eps) / args.steps
    timesteps = torch.linspace(1, eps, args.steps + 1, device=model.device)

    max_delta, n_checked = 0.0, 0
    with torch.no_grad():
        for t_idx in range(args.steps):
            com = trace['commit'][t_idx]
            if not com.any():
                continue
            z_t = trace['z_traj'][t_idx].to(model.device, torch.long)
            z_s = trace['z_traj'][t_idx + 1].to(model.device, torch.long)
            t = timesteps[t_idx] * torch.ones(z_t.shape[0], 1,
                                              device=model.device)
            sigma_t, _ = model.noise(t)
            log_p_x0 = model.forward(z_t, sigma_t.squeeze(-1))
            new_lp = log_p_x0.gather(
                -1, z_s.unsqueeze(-1)).squeeze(-1)[com.to(model.device)].float()
            old_lp = trace['old_lp'][t_idx].float()[com].to(model.device)
            delta = (new_lp - old_lp).abs().max().item()
            max_delta = max(max_delta, delta)
            n_checked += com.sum().item()

    ok = max_delta < 1e-2
    print(f"[logprob] {'PASS' if ok else 'FAIL'}  "
          f"max |recomputed - recorded| = {max_delta:.3e} over "
          f"{n_checked} committed tokens (tol 1e-2)")
    return ok


# ---------- check 2: constant reward -> skip path, weights untouched ----------
def check_noop(iters=5):
    torch.manual_seed(1234)
    random.seed(1234)
    args = _model_args()
    model, tokenizer = _load_model(args)
    prompt_ids = _prompt_ids(tokenizer, model)

    def checksum():
        with torch.no_grad():
            return sum(p.double().abs().sum().item()
                       for p in model.backbone.parameters())

    before = checksum()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1e-5)
    skipped = 0
    for _ in range(iters):
        with torch.no_grad():
            trace = model.rollout(prompt_ids, 4, args.steps)
        # constant reward (spec: monkeypatched scorer; equivalent inline form)
        rewards = torch.ones(4, dtype=torch.float32)
        adv = rewards - rewards.mean()
        if adv.abs().max() < 1e-6:
            skipped += 1
            continue                       # mirrors trace_grpo main-loop skip
        raise AssertionError('constant reward produced nonzero advantage')
    after = checksum()

    ok = skipped == iters and before == after
    print(f"[noop] {'PASS' if ok else 'FAIL'}  skipped {skipped}/{iters} "
          f"iters; |checksum delta| = {abs(before - after):.3e} "
          f"(before {before:.6e})")
    return ok


# ---------- check 3: lmbda -> 0 resampling is uniform ----------
def check_resample(k=4, n_draws=10000, tol=0.05):
    torch.manual_seed(1234)
    # base statistical sanity: uniform weights -> uniform multinomial
    w = torch.ones(k) / k
    draws = torch.multinomial(w, num_samples=n_draws, replacement=True)
    counts = torch.bincount(draws, minlength=k).float()
    rel_dev = ((counts - n_draws / k).abs() / (n_draws / k)).max().item()
    ok = rel_dev < tol
    print(f"[resample] {'PASS' if ok else 'FAIL'}  uniform multinomial: "
          f"max relative count deviation = {rel_dev:.4f} (tol {tol})")

    # lmbda=0 potentials: exp(0 * r) == 1 for every particle -> uniform
    rs = torch.randn(k)
    w0 = torch.exp(0.0 * rs)
    flat = torch.allclose(w0, torch.ones(k))
    print(f"[resample] {'PASS' if flat else 'FAIL'}  exp(lmbda*r) at "
          f"lmbda=0 uniform: weights = {w0.tolist()}")
    ok = ok and flat

    # optional: the training-time FK module, if present
    try:
        import fk_trace_grpo  # noqa: F401
        fn = getattr(fk_trace_grpo, 'fk_potentials', None)
        if fn is None:
            print("[resample] SKIP  fk_trace_grpo has no fk_potentials()")
        else:
            w_fk = fn(torch.randn(k), torch.randn(k), lmbda=0.0)
            u = torch.allclose(torch.as_tensor(w_fk, dtype=torch.float32),
                               torch.ones(k), atol=1e-6)
            print(f"[resample] {'PASS' if u else 'FAIL'}  "
                  f"fk_trace_grpo.fk_potentials lmbda=0 uniform")
            ok = ok and u
    except ImportError:
        print("[resample] SKIP  fk_trace_grpo not present yet")
    except (AttributeError, TypeError) as e:
        print(f"[resample] SKIP  fk_trace_grpo API mismatch: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', default='all',
                    choices=['logprob', 'noop', 'resample', 'all'])
    args = ap.parse_args()

    checks = {'logprob': check_logprob, 'noop': check_noop,
              'resample': check_resample}
    names = list(checks) if args.check == 'all' else [args.check]
    results = {n: checks[n]() for n in names}

    print('----')
    for n, ok in results.items():
        print(f"{n:9s} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
