"""
Analysis for the rare-success FK-vs-plain-GRPO experiment (Phase: sparse
reward, --reward-threshold eta*).

Reads log.jsonl from the two sparse runs produced by run_rare_exp_chain.sh
(baseline trace_grpo.py and FK fk_trace_grpo.py), and reports:
  - success-rate-over-iterations for both (rolling window), plotted to
    ~/dllm/iadd-lm/rare_plot.png
  - first-success iteration for both
  - fraction of skipped (zero-signal / zero-variance-advantage) iterations
  - a verdict table (printed) comparing the two

Does NOT run any training or GPU code -- CPU-only, matplotlib for the plot.

Usage:
  python ~/dllm/iadd-lm/rare_analysis.py \
      --baseline-dir ~/dllm/iadd-lm/runs/<baseline_run_name> \
      --fk-dir ~/dllm/iadd-lm/runs/<fk_run_name> \
      --window 20 \
      --out-plot ~/dllm/iadd-lm/rare_plot.png

If --baseline-dir / --fk-dir are omitted, this script picks the most
recently modified run directory under ~/dllm/iadd-lm/runs whose args.json
has reward_threshold set and (for FK) has k_particles set, respectively --
i.e. it tries to auto-find the two runs launched by run_rare_exp_chain.sh.
"""
import argparse
import glob
import json
import os


def load_log(run_dir):
    path = os.path.join(run_dir, 'log.jsonl')
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def find_latest_sparse_run(runs_root, want_fk):
    """Best-effort auto-discovery of a sparse (reward_threshold set) run,
    matching baseline (no k_particles in args.json) or FK (has k_particles).
    """
    candidates = []
    for d in glob.glob(os.path.join(runs_root, '*')):
        args_path = os.path.join(d, 'args.json')
        if not os.path.isdir(d) or not os.path.isfile(args_path):
            continue
        try:
            with open(args_path) as f:
                args = json.load(f)
        except Exception:
            continue
        if args.get('reward_threshold') is None:
            continue
        is_fk = 'k_particles' in args
        if is_fk != want_fk:
            continue
        candidates.append((os.path.getmtime(d), d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def success_rate_over_iterations(recs, window):
    """Rolling success rate: for sparse recs, an iteration "succeeds" if it
    was NOT skipped AND n_success > 0 (at least one binary success in the
    group/lineage/pair). Skipped iterations (zero-variance advantage --
    either all-fail or all-succeed) are included in the rolling denominator
    since they are still an iteration of the run, but flagged separately.
    Returns (iters, rolling_rate, skipped_flags, first_success_iter).
    """
    iters, succ_flags, skipped_flags = [], [], []
    first_success_iter = None
    for r in recs:
        it = r['iter']
        n_success = r.get('n_success')
        skipped = bool(r.get('skipped', False))
        if n_success is not None:
            had_success = n_success > 0
        else:
            # Non-sparse record fallback (shouldn't happen for these runs):
            # treat a positive mean reward as "success" is meaningless here,
            # so just mark unknown as no-success.
            had_success = False
        if had_success and first_success_iter is None:
            first_success_iter = it
        iters.append(it)
        succ_flags.append(1 if had_success else 0)
        skipped_flags.append(1 if skipped else 0)

    rolling = []
    for i in range(len(succ_flags)):
        lo = max(0, i - window + 1)
        w = succ_flags[lo:i + 1]
        rolling.append(sum(w) / len(w))
    return iters, rolling, skipped_flags, first_success_iter


def summarize(name, recs, window):
    iters, rolling, skipped_flags, first_success = success_rate_over_iterations(
        recs, window)
    n_total = len(recs)
    n_skipped = sum(skipped_flags)
    skip_frac = n_skipped / max(n_total, 1)
    overall_success_rate = sum(
        1 for r in recs if (r.get('n_success') or 0) > 0) / max(n_total, 1)
    return dict(name=name, n_total=n_total, n_skipped=n_skipped,
                skip_frac=skip_frac, first_success_iter=first_success,
                overall_success_rate=overall_success_rate,
                iters=iters, rolling=rolling)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', default=os.path.expanduser('~/dllm/iadd-lm/runs'))
    ap.add_argument('--baseline-dir', default=None)
    ap.add_argument('--fk-dir', default=None)
    ap.add_argument('--window', type=int, default=20,
                     help='rolling window (in iterations) for success rate')
    ap.add_argument('--out-plot', default=os.path.expanduser(
        '~/dllm/iadd-lm/rare_plot.png'))
    args = ap.parse_args()

    baseline_dir = args.baseline_dir or find_latest_sparse_run(
        args.runs_root, want_fk=False)
    fk_dir = args.fk_dir or find_latest_sparse_run(args.runs_root, want_fk=True)

    if baseline_dir is None or fk_dir is None:
        raise SystemExit(
            f'Could not auto-find both runs (baseline_dir={baseline_dir}, '
            f'fk_dir={fk_dir}). Pass --baseline-dir / --fk-dir explicitly.')

    print(f'baseline run: {baseline_dir}')
    print(f'FK run:       {fk_dir}')

    baseline_recs = load_log(baseline_dir)
    fk_recs = load_log(fk_dir)

    base_summary = summarize('baseline (plain sparse GRPO)', baseline_recs,
                             args.window)
    fk_summary = summarize('FK (rare-success resampling)', fk_recs, args.window)

    # ---- verdict table ----
    def fmt_iter(x):
        return 'never' if x is None else str(x)

    rows = [
        ('n iterations logged', base_summary['n_total'], fk_summary['n_total']),
        ('n skipped (zero-signal) iters', base_summary['n_skipped'],
         fk_summary['n_skipped']),
        ('fraction skipped', f"{base_summary['skip_frac']:.3f}",
         f"{fk_summary['skip_frac']:.3f}"),
        ('first-success iteration', fmt_iter(base_summary['first_success_iter']),
         fmt_iter(fk_summary['first_success_iter'])),
        ('overall per-iter success rate', f"{base_summary['overall_success_rate']:.3f}",
         f"{fk_summary['overall_success_rate']:.3f}"),
    ]
    name_w = max(len(r[0]) for r in rows) + 2
    col_w = 28
    print('\n=== VERDICT TABLE ===')
    header = f"{'metric':<{name_w}}{'baseline':<{col_w}}{'FK':<{col_w}}"
    print(header)
    print('-' * len(header))
    for label, b, f in rows:
        print(f"{label:<{name_w}}{str(b):<{col_w}}{str(f):<{col_w}}")

    if (base_summary['first_success_iter'] is not None
            and fk_summary['first_success_iter'] is not None):
        if fk_summary['first_success_iter'] < base_summary['first_success_iter']:
            print('\nVerdict: FK reaches first success earlier than plain '
                  'sparse GRPO (consistent with the design hypothesis: '
                  'dense potential guidance helps find rare successes '
                  'faster than starved zero-advantage groups).')
        elif fk_summary['first_success_iter'] > base_summary['first_success_iter']:
            print('\nVerdict: baseline reaches first success earlier than '
                  'FK in this run -- inspect resample_stats.jsonl / lmbda '
                  'before concluding FK underperforms.')
        else:
            print('\nVerdict: tie on first-success iteration.')
    else:
        print('\nVerdict: at least one run never logged a success within '
              'the iterations recorded so far -- inconclusive, rerun with '
              'more iterations or re-check eta_star calibration.')

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(base_summary['iters'], base_summary['rolling'],
                label='baseline (plain sparse GRPO)', color='tab:blue')
        ax.plot(fk_summary['iters'], fk_summary['rolling'],
                label='FK (rare-success resampling)', color='tab:orange')
        if base_summary['first_success_iter'] is not None:
            ax.axvline(base_summary['first_success_iter'], color='tab:blue',
                       linestyle=':', alpha=0.6)
        if fk_summary['first_success_iter'] is not None:
            ax.axvline(fk_summary['first_success_iter'], color='tab:orange',
                       linestyle=':', alpha=0.6)
        ax.set_xlabel('iteration')
        ax.set_ylabel(f'rolling success rate (window={args.window})')
        ax.set_title('Rare-success rate over training: plain GRPO vs FK')
        ax.legend()
        fig.tight_layout()
        out_path = os.path.expanduser(args.out_plot)
        fig.savefig(out_path, dpi=150)
        print(f'\nSaved plot to {out_path}')
    except ImportError:
        print('\n[warn] matplotlib not available; skipping plot.')


if __name__ == '__main__':
    main()
