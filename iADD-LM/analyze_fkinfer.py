"""
Analyze finals.jsonl from an fk_trace_grpo.py --eval-only --eval-save-texts run.

Per prompt (grouped by prompt_idx), takes the BEST-reward text of each of the
8 independent FK rollouts assigned to that prompt (that is what FK-inference
would actually return per generation), then computes:
  - mean reward of those 8 selected finals
  - semantic_div across those 8 texts, using the SAME embedding + pairwise
    cosine-distance code as eval.py (imported directly from eval.py).
Averages these per-prompt numbers over prompts. Also reports the same for a
random (not best) particle per rollout as a secondary row.

Usage:
  python analyze_fkinfer.py <run_dir>/finals.jsonl
"""
import json
import random
import sys
import os

sys.path.insert(0, os.path.expanduser('~/dllm/iadd-lm'))
import eval as eval_mod  # noqa: E402  (reuse semantic_div exactly)


def strip_eot(t):
    return t.replace('<|endoftext|>', ' ').strip()


def main():
    finals_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 777
    rng = random.Random(seed)

    rollouts = []
    with open(finals_path) as f:
        for line in f:
            rollouts.append(json.loads(line))

    by_prompt = {}
    for r in rollouts:
        by_prompt.setdefault(r['prompt_idx'], []).append(r)

    per_prompt_best = []
    per_prompt_rand = []
    for pidx, rolls in sorted(by_prompt.items()):
        best_texts, best_rewards = [], []
        rand_texts, rand_rewards = [], []
        for r in rolls:
            bi = r['best_idx']
            best_texts.append(strip_eot(r['texts'][bi]))
            best_rewards.append(r['raw_rewards'][bi])
            ri = rng.randrange(len(r['texts']))
            rand_texts.append(strip_eot(r['texts'][ri]))
            rand_rewards.append(r['raw_rewards'][ri])

        per_prompt_best.append(dict(
            prompt_idx=pidx, n=len(rolls),
            r_mean=sum(best_rewards) / len(best_rewards),
            semantic_div=eval_mod.semantic_div(best_texts),
        ))
        per_prompt_rand.append(dict(
            prompt_idx=pidx, n=len(rolls),
            r_mean=sum(rand_rewards) / len(rand_rewards),
            semantic_div=eval_mod.semantic_div(rand_texts),
        ))

    def avg(recs, k):
        vals = [d[k] for d in recs if d[k] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = dict(
        n_prompts=len(per_prompt_best),
        n_rollouts=len(rollouts),
        selected=dict(
            r_mean_selected=avg(per_prompt_best, 'r_mean'),
            semantic_div_selected=avg(per_prompt_best, 'semantic_div'),
            per_prompt=per_prompt_best,
        ),
        random_particle=dict(
            r_mean_random=avg(per_prompt_rand, 'r_mean'),
            semantic_div_random=avg(per_prompt_rand, 'semantic_div'),
            per_prompt=per_prompt_rand,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
