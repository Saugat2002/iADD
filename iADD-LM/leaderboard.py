"""
Leaderboard for iADD-LM evals.

Reads evals.jsonl (one JSON record per eval run, appended by eval.py),
deduplicates by tag (keeping the latest record per tag), and emits:
  - leaderboard.md : ranked markdown tables (by r_mean, and by a
    reward/diversity tradeoff score)
  - leaderboard.png: scatter of r_mean vs jaccard_div, labeled by tag

CPU-only, no GPU / torch dependency.

Usage:
  python leaderboard.py [--evals ~/dllm/iadd-lm/evals.jsonl]
                         [--md ~/dllm/iadd-lm/leaderboard.md]
                         [--png ~/dllm/iadd-lm/leaderboard.png]
"""
import argparse
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


REQUIRED_FIELDS = ('tag', 'r_mean', 'distinct2', 'self_bleu', 'jaccard_div')


def load_records(path):
    """Read evals.jsonl, skip malformed/incomplete lines, dedup by tag
    keeping the latest record (last occurrence in file order)."""
    by_tag = {}
    if not os.path.exists(path):
        return []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping malformed JSON at line {lineno}",
                      file=sys.stderr)
                continue
            if not isinstance(rec, dict):
                print(f"warning: skipping non-object record at line {lineno}",
                      file=sys.stderr)
                continue
            if not all(k in rec and rec[k] is not None for k in REQUIRED_FIELDS):
                print(f"warning: skipping record missing required fields "
                      f"at line {lineno}", file=sys.stderr)
                continue
            tag = rec['tag'] if rec['tag'] not in (None, '') else rec.get('ckpt', f'line{lineno}')
            rec['tag'] = tag
            by_tag[tag] = rec  # later occurrence overwrites -> "latest"
    return list(by_tag.values())


def zscore(values):
    """Standardize a list of floats: (x - mean) / stdev.
    If stdev is 0 or there's only one value, returns all zeros."""
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return [0.0 for _ in values]
    return [(v - mean) / stdev for v in values]


def build_table(rows, sort_key, header, extra_cols=None):
    """rows: list of dicts with tag, r_mean, distinct2, self_bleu, jaccard_div,
    tradeoff. Returns a markdown table string sorted descending by sort_key."""
    ordered = sorted(rows, key=lambda r: r[sort_key], reverse=True)
    cols = ['rank', 'tag', 'r_mean', 'distinct2', 'self_bleu (lower=more diverse)',
            'jaccard_div', 'tradeoff score']
    lines = [header, '', '| ' + ' | '.join(cols) + ' |',
             '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for i, r in enumerate(ordered, 1):
        lines.append(
            f"| {i} | {r['tag']} | {r['r_mean']:.4f} | {r['distinct2']:.4f} | "
            f"{r['self_bleu']:.4f} | {r['jaccard_div']:.4f} | {r['tradeoff']:.4f} |"
        )
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evals', default=os.path.expanduser('~/dllm/iadd-lm/evals.jsonl'))
    ap.add_argument('--md', default=os.path.expanduser('~/dllm/iadd-lm/leaderboard.md'))
    ap.add_argument('--png', default=os.path.expanduser('~/dllm/iadd-lm/leaderboard.png'))
    args = ap.parse_args()

    records = load_records(args.evals)

    if not records:
        msg = "no evals yet"
        print(msg)
        with open(args.md, 'w') as f:
            f.write(f"# iADD-LM Leaderboard\n\n{msg} (no valid records found in "
                     f"`{args.evals}`).\n")
        # No PNG when there's nothing to plot.
        return

    r_means = [r['r_mean'] for r in records]
    jaccards = [r['jaccard_div'] for r in records]
    z_r = zscore(r_means)
    z_j = zscore(jaccards)
    for r, zr, zj in zip(records, z_r, z_j):
        r['tradeoff'] = zr + zj

    table1 = build_table(records, 'r_mean', '## Ranked by mean reward (r_mean)')
    table2 = build_table(records, 'tradeoff', '## Ranked by tradeoff score')

    footnote = (
        "\n\n---\n"
        "**Tradeoff score** = z(r_mean) + z(jaccard_div), where z(x) = "
        "(x - mean(x)) / stdev(x) computed across all rows in this table "
        "(population stdev; 0 if fewer than 2 rows or stdev is 0). Higher "
        "tradeoff score means a run is jointly above-average on reward and "
        "on diversity (jaccard_div). self_bleu is included for reference "
        "only and is *not* part of the composite score; lower self_bleu "
        "means more diverse samples.\n"
    )

    md = f"# iADD-LM Leaderboard\n\n{table1}\n\n{table2}{footnote}"
    with open(args.md, 'w') as f:
        f.write(md)
    print(md)

    # ---- scatter plot ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in records:
        is_base = (r['tag'] == 'base')
        ax.scatter(r['r_mean'], r['jaccard_div'],
                   s=140 if is_base else 80,
                   marker='*' if is_base else 'o',
                   edgecolors='black' if is_base else 'none',
                   linewidths=1.5 if is_base else 0,
                   color='crimson' if is_base else 'steelblue',
                   zorder=3 if is_base else 2)
        ax.annotate(r['tag'], (r['r_mean'], r['jaccard_div']),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)

    ax.set_xlabel('reward (r_mean) →')
    ax.set_ylabel('diversity (jaccard_div) →')
    ax.set_title('iADD-LM: reward vs. diversity')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.png, dpi=150)
    print(f"wrote {args.md}")
    print(f"wrote {args.png}")


if __name__ == '__main__':
    main()
