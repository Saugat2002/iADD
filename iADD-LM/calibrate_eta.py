"""
Calibrate the rare-success threshold eta* for the FK-in-rare-regime experiment.

Samples G=32 continuations per hard prompt from the BASE model (untrained
backbone, same TraceGRPO.rollout used by eval.py / trace_grpo.py), scores them
with the sentiment reward, and picks eta* as the pooled-quantile threshold such
that the base model's success rate P(score >= eta*) ~= 0.05 (5%) over all
G * num_prompts samples pooled together.

Run (GPU, ~10 min for 5 prompts x 32 samples):
  python ~/dllm/iadd-lm/calibrate_eta.py \
      --prompt-file ~/dllm/iadd-lm/hard_prompts.jsonl --group 32 \
      --target-rate 0.05 --out ~/dllm/iadd-lm/eta_star.json
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
from fk_diffusion import compute_rewards  # noqa: E402
from trace_grpo import TraceGRPO, build_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='~/dllm/mdlm-owt-local')
    ap.add_argument('--reward', default='sentiment')
    ap.add_argument('--reward-label', default='positive')
    ap.add_argument('--reward-trim', type=int, default=100)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--gen-len', type=int, default=128)
    ap.add_argument('--group', type=int, default=32,
                     help='G continuations sampled per hard prompt (base model)')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--prompt-file',
                     default=os.path.expanduser('~/dllm/iadd-lm/hard_prompts.jsonl'))
    ap.add_argument('--target-rate', type=float, default=0.05,
                     help='desired pooled base success rate P(score >= eta*)')
    ap.add_argument('--out', default=os.path.expanduser('~/dllm/iadd-lm/eta_star.json'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config(args)
    tokenizer = dataloader.get_tokenizer(cfg)
    model = TraceGRPO(cfg, tokenizer=tokenizer).to('cuda')
    model.ema = None
    model.backbone.eval()  # BASE model, no training

    with open(args.prompt_file) as f:
        prompts = [json.loads(l)['context_string'] for l in f]

    per_prompt = []
    all_scores = []
    for p in prompts:
        enc = tokenizer([p], return_tensors='pt', padding=False)
        prompt_ids = enc['input_ids'][:, :-1].to(model.device)
        with torch.no_grad():
            tr = model.rollout(prompt_ids, args.group, args.steps)
        trim = args.reward_trim + prompt_ids.shape[1]
        texts = tokenizer.batch_decode(tr['final'][:, :trim])
        rs = [float(r) for r in compute_rewards(
            samples=texts, reward_name=args.reward, reward_label=args.reward_label)]
        per_prompt.append({'prompt': p, 'scores': rs})
        all_scores += rs
        print(f'[{p!r}] n={len(rs)} mean={sum(rs)/len(rs):.4f} '
              f'max={max(rs):.4f} min={min(rs):.4f}')

    # eta* = pooled quantile such that P(score >= eta*) ~= target_rate.
    # i.e. eta* is the (1 - target_rate) quantile of the pooled scores.
    sorted_scores = sorted(all_scores)
    n = len(sorted_scores)
    q = 1.0 - args.target_rate
    idx = min(int(round(q * (n - 1))), n - 1)
    eta_star = sorted_scores[idx]

    base_rate_pooled = sum(1 for s in all_scores if s >= eta_star) / n

    per_prompt_rates = []
    for rec in per_prompt:
        rate = sum(1 for s in rec['scores'] if s >= eta_star) / len(rec['scores'])
        per_prompt_rates.append({'prompt': rec['prompt'], 'success_rate': rate,
                                  'n': len(rec['scores'])})

    result = {
        'eta_star': eta_star,
        'target_rate': args.target_rate,
        'pooled_base_success_rate': base_rate_pooled,
        'n_pooled': n,
        'group_per_prompt': args.group,
        'reward': args.reward,
        'reward_label': args.reward_label,
        'per_prompt_base_success_rate': per_prompt_rates,
    }

    print('\n=== eta* calibration result ===')
    print(json.dumps(result, indent=2))

    with open(os.path.expanduser(args.out), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
