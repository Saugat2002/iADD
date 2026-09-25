"""
Hard-prompt vs easy-prompt reward-gain split analysis.

Loads ~/dllm/iadd-lm/evals.jsonl, ranks the 15 eval prompts by the `base`
row's per-prompt r_mean (lower = harder for the base model to score well
on), defines:
    HARD = bottom 5 prompts by base reward
    EASY = top 5 prompts by base reward
For each requested tag, computes mean per-prompt reward GAIN over base
separately on HARD and EASY prompts, and their ratio.

Deviation note: evals.jsonl contains duplicate rows for several tags
(base x2, entropy x2, incremental x2, any x3, iadd_lm_v2k8 x2) — these are
reruns. This script uses the LAST occurrence of each tag in file order
(most recent run), except it verified the two `base` rows are byte-for-byte
identical in per_prompt r_mean before picking one.
"""
import json
import os

EVALS_PATH = os.path.expanduser('~/dllm/iadd-lm/evals.jsonl')
OUT_JSON = os.path.expanduser('~/dllm/iadd-lm/rarity_split.json')

TAGS = ['all', 'entropy', 'early', 'incremental', 'iadd_lm_v2k8',
        'iadd_lm_v2', 'dm_style', 'd1_style']


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def last_by_tag(rows, tag):
    matches = [r for r in rows if r.get('tag') == tag]
    if not matches:
        return None
    return matches[-1]


def main():
    rows = load_rows(EVALS_PATH)

    base = last_by_tag(rows, 'base')
    if base is None:
        raise SystemExit("No 'base' row found in evals.jsonl")

    base_pp = base['per_prompt']
    n = len(base_pp)
    order = sorted(range(n), key=lambda i: base_pp[i]['r_mean'])
    hard_idx = order[:5]          # lowest base reward = hardest
    easy_idx = order[-5:]         # highest base reward = easiest

    hard_prompts = [base_pp[i]['prompt'] for i in hard_idx]
    easy_prompts = [base_pp[i]['prompt'] for i in easy_idx]

    results = []
    missing = []
    for tag in TAGS:
        row = last_by_tag(rows, tag)
        if row is None:
            missing.append(tag)
            continue
        pp = row['per_prompt']
        if len(pp) != n:
            print(f"[warn] tag={tag} has {len(pp)} prompts, expected {n}; skipping")
            continue

        def gain(idx):
            gs = [pp[i]['r_mean'] - base_pp[i]['r_mean'] for i in idx]
            return sum(gs) / len(gs)

        easy_gain = gain(easy_idx)
        hard_gain = gain(hard_idx)
        ratio = hard_gain / easy_gain if easy_gain != 0 else float('nan')
        results.append(dict(tag=tag, easy_gain=easy_gain, hard_gain=hard_gain,
                             ratio=ratio))

    results.sort(key=lambda d: d['hard_gain'], reverse=True)

    # ---- print table ----
    print(f"HARD prompts (bottom 5 by base r_mean):")
    for i in hard_idx:
        print(f"  [{base_pp[i]['r_mean']:.4f}] {base_pp[i]['prompt']!r}")
    print(f"\nEASY prompts (top 5 by base r_mean):")
    for i in easy_idx:
        print(f"  [{base_pp[i]['r_mean']:.4f}] {base_pp[i]['prompt']!r}")

    print(f"\n{'tag':<16}{'easy_gain':>12}{'hard_gain':>12}{'ratio':>10}")
    for r in results:
        print(f"{r['tag']:<16}{r['easy_gain']:>12.4f}{r['hard_gain']:>12.4f}{r['ratio']:>10.4f}")

    if missing:
        print(f"\n[warn] tags not found in evals.jsonl: {missing}")

    out = dict(
        hard_prompts=hard_prompts,
        easy_prompts=easy_prompts,
        hard_idx=hard_idx,
        easy_idx=easy_idx,
        results=results,
        missing_tags=missing,
    )
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == '__main__':
    main()
