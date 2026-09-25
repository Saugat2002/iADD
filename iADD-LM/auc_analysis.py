#!/usr/bin/env python
"""
AUC / operating-point / Pareto analysis for iADD-LM curve evals.

Implements the iADD paper's evaluation protocol on top of
~/dllm/iadd-lm/evals.jsonl:

  1. For each method m in {all, inc, ent, fk}: gather curve points
     (r_mean, semantic_div) from tags m_c<ckpt>, prepend the shared
     `base` row as the iteration-0 point of every curve.
  2. Jointly min-max normalize both axes (reward, semantic_div) across
     the union of ALL methods' points (base + all m_c* points), so the
     curves are comparable in [0,1] x [0,1].
  3. AUC per method: sort by normalized reward, trapezoid-integrate
     normalized diversity over normalized reward. Report as %.

     Two AUC conventions are reported side by side:
       - auc_pct        ("own-range" AUC): integrated over THIS
         method's own normalized-reward span [min_nr, max_nr]. This is
         the original convention -- methods with wider reward coverage
         can end up with larger or smaller AUC purely because they
         integrate over a different x-range than other methods, so
         auc_pct values are NOT directly comparable across methods with
         different reward ranges.
       - auc_common_pct ("common-support" AUC): integrated only over
         the INTERSECTION of all valid methods' raw reward ranges
         (converted into the same normalized-reward space used
         everywhere else). Every method's auc_common_pct is computed
         over the identical x-interval, so these values ARE directly
         comparable across methods. Curve values at the intersection's
         boundaries are linearly interpolated when a boundary falls
         between two observed checkpoints. If fewer than 2 methods are
         valid, or the intersection of reward ranges is empty/degenerate,
         auc_common_pct is None for all methods.
  4. Operating point per method: checkpoint minimizing Euclidean
     distance to (1,1) in normalized space.
  5. Diversity-at-matched-reward: interpolate each method's curve at a
     grid of 5 reward levels spanning the overlap region of all curves,
     print level x method table + per-level winner.
  6. Pareto dominance check between every pair of methods.
  7. Report to stdout, JSON to auc_report.json, figure to
     tradeoff_curves.png.

Robust to: missing m_c* checkpoints, methods with <2 points (skipped
with a warning), duplicate tags (last one wins), and the case where NO
m_c* tags exist yet at all (prints 'no curve evals yet' and exits 0).
"""
import json
import os
import re
import sys

METHODS = ["all", "inc", "ent", "fk", "hyb"]
EVALS_PATH = os.path.expanduser("~/dllm/iadd-lm/evals.jsonl")
REPORT_JSON_PATH = os.path.expanduser("~/dllm/iadd-lm/auc_report.json")
FIGURE_PATH = os.path.expanduser("~/dllm/iadd-lm/tradeoff_curves.png")

# tag pattern: <method>_c<checkpoint number>, e.g. all_c75, ent_c600
CURVE_TAG_RE = re.compile(r"^(all|inc|ent|fk|hyb)_c(\d+)$")


def load_rows(path):
    """Load JSONL, keep-last on duplicate tags."""
    rows_by_tag = {}
    order = []
    if not os.path.exists(path):
        print(f"ERROR: evals file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {lineno}: {e}", file=sys.stderr)
                continue
            tag = d.get("tag")
            if tag is None:
                print(f"WARNING: line {lineno} has no 'tag', skipping", file=sys.stderr)
                continue
            if tag not in rows_by_tag:
                order.append(tag)
            rows_by_tag[tag] = d  # keep-last on duplicates
    return rows_by_tag, order


def extract_point(d):
    """Pull (r_mean, semantic_div) out of an eval row, tolerating missing fields."""
    r = d.get("r_mean")
    div = d.get("semantic_div")
    if r is None or div is None:
        return None
    return (float(r), float(div))


def build_curves(rows_by_tag):
    """
    Returns dict: method -> list of (ckpt_int, tag, r_mean, semantic_div),
    sorted by ckpt, with the shared base row prepended as ckpt=0.
    Also returns the list of curve tags actually found (for the 'no curve
    evals yet' check) and any per-method warnings.
    """
    base_row = rows_by_tag.get("base")
    base_point = extract_point(base_row) if base_row is not None else None

    curve_tags_found = []
    per_method_ckpts = {m: [] for m in METHODS}

    for tag in rows_by_tag:
        m = CURVE_TAG_RE.match(tag)
        if m:
            method, ckpt = m.group(1), int(m.group(2))
            curve_tags_found.append(tag)
            per_method_ckpts[method].append((ckpt, tag))

    curves = {}
    warnings = []

    for method in METHODS:
        pts = []
        if base_point is not None:
            pts.append((0, "base", base_point[0], base_point[1]))
        else:
            warnings.append("no 'base' tag found in evals.jsonl; curves will not include iteration-0 point")

        for ckpt, tag in sorted(per_method_ckpts[method]):
            row = rows_by_tag[tag]
            p = extract_point(row)
            if p is None:
                warnings.append(f"tag '{tag}' missing r_mean/semantic_div, skipping")
                continue
            pts.append((ckpt, tag, p[0], p[1]))

        if len(pts) < 2:
            if per_method_ckpts[method]:
                warnings.append(
                    f"method '{method}' has fewer than 2 usable points "
                    f"({len(pts)}); skipping this method's curve"
                )
            curves[method] = None
        else:
            curves[method] = pts

    return curves, curve_tags_found, warnings


def minmax_normalize_all(curves):
    """Jointly min-max normalize reward & diversity across union of all methods' points."""
    all_r = []
    all_d = []
    for method, pts in curves.items():
        if pts is None:
            continue
        for (_, _, r, d) in pts:
            all_r.append(r)
            all_d.append(d)

    r_min, r_max = min(all_r), max(all_r)
    d_min, d_max = min(all_d), max(all_d)
    r_span = (r_max - r_min) or 1.0
    d_span = (d_max - d_min) or 1.0

    norm_curves = {}
    for method, pts in curves.items():
        if pts is None:
            norm_curves[method] = None
            continue
        norm_pts = []
        for (ckpt, tag, r, d) in pts:
            nr = (r - r_min) / r_span
            nd = (d - d_min) / d_span
            norm_pts.append((ckpt, tag, r, d, nr, nd))
        norm_curves[method] = norm_pts

    return norm_curves, dict(r_min=r_min, r_max=r_max, d_min=d_min, d_max=d_max)


def trapezoid_auc(xs, ys):
    """Standard trapezoid rule, xs assumed sorted ascending."""
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        area += dx * (ys[i] + ys[i - 1]) / 2.0
    return area


def compute_auc(norm_curves):
    aucs = {}
    for method, pts in norm_curves.items():
        if pts is None:
            aucs[method] = None
            continue
        sorted_pts = sorted(pts, key=lambda p: p[4])  # sort by normalized reward
        xs = [p[4] for p in sorted_pts]
        ys = [p[5] for p in sorted_pts]
        # dedupe identical x (keep first) to avoid zero-width weirdness only if needed
        area = trapezoid_auc(xs, ys)
        span = xs[-1] - xs[0]
        # normalize by reward-span so AUC is a % of the achievable box (matches iADD Table 3 style)
        auc_pct = 100.0 * area / span if span > 0 else 0.0
        aucs[method] = auc_pct
    return aucs


def _interp_at_x(xs, ys, x):
    """Linear-interpolate ys at x, given xs sorted ascending. Assumes
    xs[0] <= x <= xs[-1]; returns None otherwise."""
    if x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i - 1] <= x <= xs[i]:
            if xs[i] == xs[i - 1]:
                return ys[i]
            frac = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + frac * (ys[i] - ys[i - 1])
    return None


def compute_common_support_auc(norm_curves, norm_stats):
    """
    Common-support AUC: each valid method's AUC integrated only over the
    intersection of all valid methods' RAW reward ranges (mapped into
    normalized-reward space via the same affine transform used for
    joint normalization). Directly comparable across methods, unlike
    the own-range auc_pct.

    Returns dict method -> auc_common_pct (or None), and the
    (lo_norm, hi_norm) intersection interval used (or None if undefined).
    """
    valid_methods = [m for m, pts in norm_curves.items() if pts is not None]
    common_aucs = {m: None for m in norm_curves}

    if len(valid_methods) < 2:
        return common_aucs, None

    raw_ranges = {}
    for m in valid_methods:
        rs = [p[2] for p in norm_curves[m]]
        raw_ranges[m] = (min(rs), max(rs))

    lo_raw = max(raw_ranges[m][0] for m in valid_methods)
    hi_raw = min(raw_ranges[m][1] for m in valid_methods)
    if lo_raw >= hi_raw:
        return common_aucs, None

    r_min = norm_stats["r_min"]
    r_span = (norm_stats["r_max"] - norm_stats["r_min"]) or 1.0
    lo_norm = (lo_raw - r_min) / r_span
    hi_norm = (hi_raw - r_min) / r_span

    for method in valid_methods:
        sorted_pts = sorted(norm_curves[method], key=lambda p: p[4])
        xs = [p[4] for p in sorted_pts]
        ys = [p[5] for p in sorted_pts]

        y_lo = _interp_at_x(xs, ys, lo_norm)
        y_hi = _interp_at_x(xs, ys, hi_norm)
        if y_lo is None or y_hi is None:
            # shouldn't happen (intersection derived from these same
            # methods' raw ranges), but guard defensively
            continue

        clipped_xs = [lo_norm]
        clipped_ys = [y_lo]
        for x, y in zip(xs, ys):
            if lo_norm < x < hi_norm:
                clipped_xs.append(x)
                clipped_ys.append(y)
        clipped_xs.append(hi_norm)
        clipped_ys.append(y_hi)

        area = trapezoid_auc(clipped_xs, clipped_ys)
        span = clipped_xs[-1] - clipped_xs[0]
        common_aucs[method] = 100.0 * area / span if span > 0 else 0.0

    return common_aucs, (lo_norm, hi_norm)


def compute_operating_points(norm_curves):
    ops = {}
    for method, pts in norm_curves.items():
        if pts is None:
            ops[method] = None
            continue
        best = None
        best_dist = None
        for (ckpt, tag, r, d, nr, nd) in pts:
            dist = ((1.0 - nr) ** 2 + (1.0 - nd) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (ckpt, tag, r, d, nr, nd, dist)
        ops[method] = best
    return ops


def interp_diversity_at_reward(pts, r_level):
    """
    pts: list of (ckpt, tag, r, d, nr, nd) sorted by raw r ascending.
    Linear-interpolate raw semantic_div at raw reward r_level.
    Returns None if r_level is outside this curve's [min_r, max_r] range.
    """
    sorted_pts = sorted(pts, key=lambda p: p[2])
    rs = [p[2] for p in sorted_pts]
    ds = [p[3] for p in sorted_pts]
    if r_level < rs[0] or r_level > rs[-1]:
        return None
    for i in range(1, len(rs)):
        if rs[i - 1] <= r_level <= rs[i]:
            if rs[i] == rs[i - 1]:
                return ds[i]
            frac = (r_level - rs[i - 1]) / (rs[i] - rs[i - 1])
            return ds[i - 1] + frac * (ds[i] - ds[i - 1])
    return None


def compute_matched_reward_table(norm_curves, n_levels=5):
    valid_methods = [m for m, pts in norm_curves.items() if pts is not None]
    if len(valid_methods) < 2:
        return None, valid_methods

    # overlap region = intersection of [min_r, max_r] across all valid methods' curves
    lo = max(min(p[2] for p in norm_curves[m]) for m in valid_methods)
    hi = min(max(p[2] for p in norm_curves[m]) for m in valid_methods)
    if lo >= hi:
        return None, valid_methods

    levels = [lo + (hi - lo) * i / (n_levels - 1) for i in range(n_levels)]
    table = []
    for level in levels:
        row = {"reward_level": level, "diversity": {}}
        for m in valid_methods:
            row["diversity"][m] = interp_diversity_at_reward(norm_curves[m], level)
        usable = {k: v for k, v in row["diversity"].items() if v is not None}
        row["winner"] = max(usable, key=usable.get) if usable else None
        table.append(row)
    return table, valid_methods


def compute_pareto(matched_table, valid_methods):
    """
    For each pair (a, b), check if a dominates b: at every overlapping
    reward level where both have a value, a's diversity >= b's (and
    strictly > at least once). Symmetric check gives b-dominates-a too.
    """
    result = {}
    if matched_table is None:
        return result
    for a in valid_methods:
        for b in valid_methods:
            if a == b:
                continue
            pairs = []
            for row in matched_table:
                da, db = row["diversity"].get(a), row["diversity"].get(b)
                if da is not None and db is not None:
                    pairs.append((da, db))
            if not pairs:
                result[f"{a}_vs_{b}"] = "no overlap"
                continue
            a_dominates = all(da >= db for da, db in pairs) and any(da > db for da, db in pairs)
            result[f"{a}_vs_{b}"] = "dominates" if a_dominates else "does_not_dominate"
    return result


def make_figure(norm_curves, ops, base_point_raw, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"all": "tab:blue", "inc": "tab:orange", "ent": "tab:green", "fk": "tab:red", "hyb": "tab:purple"}

    for method, pts in norm_curves.items():
        if pts is None:
            continue
        sorted_pts = sorted(pts, key=lambda p: p[2])  # by raw reward
        rs = [p[2] for p in sorted_pts]
        ds = [p[3] for p in sorted_pts]
        ax.plot(rs, ds, marker="o", label=method, color=colors.get(method))

        op = ops.get(method)
        if op is not None:
            _, _, op_r, op_d, _, _, _ = op
            ax.scatter([op_r], [op_d], s=220, facecolors="none",
                       edgecolors=colors.get(method), linewidths=2, zorder=5)

    if base_point_raw is not None:
        ax.scatter([base_point_raw[0]], [base_point_raw[1]], marker="*", s=300,
                   color="black", zorder=6, label="base")

    ax.set_xlabel("reward (r_mean) →  higher is better")
    ax.set_ylabel("semantic diversity →  higher is better")
    ax.set_title("Reward vs. semantic diversity trade-off curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_report(curves, norm_curves, norm_stats, aucs, common_aucs, common_range, ops, matched_table, valid_methods, pareto, warnings):
    print("=" * 70)
    print("iADD-LM curve-eval AUC / operating-point / Pareto report")
    print("=" * 70)

    print("\n-- Normalization ranges (joint, across all methods) --")
    print(f"  reward:    [{norm_stats['r_min']:.4f}, {norm_stats['r_max']:.4f}]")
    print(f"  diversity: [{norm_stats['d_min']:.4f}, {norm_stats['d_max']:.4f}]")

    print("\n-- Curve points per method (raw) --")
    for method in METHODS:
        pts = curves.get(method)
        if pts is None:
            print(f"  {method}: <skipped, insufficient data>")
            continue
        print(f"  {method}:")
        for (ckpt, tag, r, d) in pts:
            print(f"      ckpt={ckpt:>4}  tag={tag:<12}  r_mean={r:.4f}  semantic_div={d:.4f}")

    print("\n-- AUC (%) per method (trapezoid area, normalized reward -> normalized diversity) --")
    print("  auc_pct: integrated over each method's OWN normalized-reward range (not directly")
    print("           comparable across methods with different reward coverage)")
    print("  auc_common_pct: integrated over the INTERSECTION of all valid methods' raw reward")
    print("           ranges (mapped to normalized space) -- directly comparable across methods")
    if common_range is not None:
        print(f"  common-support normalized-reward interval: [{common_range[0]:.4f}, {common_range[1]:.4f}]")
    else:
        print("  common-support interval: n/a (fewer than 2 valid methods, or empty intersection)")
    header = "  " + "method".ljust(7) + "auc_pct".ljust(12) + "auc_common_pct"
    print(header)
    for method in METHODS:
        v = aucs.get(method)
        cv = common_aucs.get(method)
        v_str = "n/a" if v is None else f"{v:.2f}%"
        cv_str = "n/a" if cv is None else f"{cv:.2f}%"
        print(f"  {method:<7}{v_str:<12}{cv_str}")

    print("\n-- Operating point per method (closest to ideal corner (1,1), normalized) --")
    for method in METHODS:
        op = ops.get(method)
        if op is None:
            print(f"  {method:<5}: n/a")
            continue
        ckpt, tag, r, d, nr, nd, dist = op
        print(f"  {method:<5}: tag={tag:<12} r_mean={r:.4f}  semantic_div={d:.4f}  (dist_to_corner={dist:.4f})")

    print("\n-- Diversity at matched reward levels --")
    if matched_table is None:
        print("  (fewer than 2 methods have usable curves, or no reward overlap; skipped)")
    else:
        header = "  level".ljust(10) + "reward".ljust(12) + "".join(m.ljust(10) for m in valid_methods) + "winner"
        print(header)
        for row in matched_table:
            line = f"  {'':<8}{row['reward_level']:<12.4f}"
            for m in valid_methods:
                v = row["diversity"].get(m)
                line += (f"{v:.4f}".ljust(10) if v is not None else "n/a".ljust(10))
            line += str(row["winner"])
            print(line)

    print("\n-- Pareto dominance (pairwise) --")
    if not pareto:
        print("  (skipped, fewer than 2 valid methods)")
    else:
        for k, v in pareto.items():
            print(f"  {k}: {v}")

    if warnings:
        print("\n-- Warnings --")
        for w in warnings:
            print(f"  * {w}")


def main():
    rows_by_tag, order = load_rows(EVALS_PATH)
    curves, curve_tags_found, warnings = build_curves(rows_by_tag)

    if not curve_tags_found:
        print("no curve evals yet")
        return 0

    any_valid = any(v is not None for v in curves.values())
    if not any_valid:
        print("no curve evals yet")
        for w in warnings:
            print(f"  * {w}", file=sys.stderr)
        return 0

    norm_curves, norm_stats = minmax_normalize_all(curves)
    aucs = compute_auc(norm_curves)
    common_aucs, common_range = compute_common_support_auc(norm_curves, norm_stats)
    ops = compute_operating_points(norm_curves)
    matched_table, valid_methods = compute_matched_reward_table(norm_curves)
    pareto = compute_pareto(matched_table, valid_methods)

    print_report(curves, norm_curves, norm_stats, aucs, common_aucs, common_range, ops, matched_table, valid_methods, pareto, warnings)

    base_row = rows_by_tag.get("base")
    base_point_raw = extract_point(base_row) if base_row is not None else None
    try:
        make_figure(norm_curves, ops, base_point_raw, FIGURE_PATH)
        print(f"\nSaved figure: {FIGURE_PATH}")
    except Exception as e:
        print(f"\nWARNING: failed to save figure: {e}", file=sys.stderr)

    report_obj = {
        "curve_tags_found": sorted(curve_tags_found),
        "norm_stats": norm_stats,
        "curves_raw": {
            m: (None if pts is None else [
                {"ckpt": c, "tag": t, "r_mean": r, "semantic_div": d} for (c, t, r, d) in pts
            ])
            for m, pts in curves.items()
        },
        "auc_pct": aucs,
        "auc_common_pct": common_aucs,
        "auc_common_support_range_normalized": common_range,
        "operating_points": {
            m: (None if op is None else {
                "ckpt": op[0], "tag": op[1], "r_mean": op[2], "semantic_div": op[3],
                "dist_to_corner": op[6],
            })
            for m, op in ops.items()
        },
        "matched_reward_table": matched_table,
        "pareto": pareto,
        "warnings": warnings,
    }
    try:
        with open(REPORT_JSON_PATH, "w") as f:
            json.dump(report_obj, f, indent=2)
        print(f"Saved report JSON: {REPORT_JSON_PATH}")
    except Exception as e:
        print(f"WARNING: failed to save report JSON: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
