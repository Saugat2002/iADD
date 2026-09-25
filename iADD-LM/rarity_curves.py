#!/usr/bin/env python3
"""
Rarity-capture analysis for iADD-LM curve evals.

eta = 75th percentile of the BASE row's pooled r_samples over the 5 HARD
prompts (iADD Q3 rule). NOTE: the 'base' row in evals.jsonl has no
'r_samples' field on any of its per_prompt entries (checked both
occurrences of tag == 'base' in the file). As a documented fallback, eta
is instead computed as the 75th percentile of the BASE row's per-prompt
r_mean values over the 5 hard prompts (5 values, not 40 samples). This is
flagged clearly in stdout and in the saved JSON.

For the same reason, the operating-point comparison for 'base' reports an
approximate rare-capture fraction computed at the per-prompt r_mean level
(fraction of hard/overall per-prompt means >= eta) rather than a true
per-sample fraction, and this is also flagged.
"""
import json
import re
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/dllm/iadd-lm")
EVALS_PATH = os.path.join(ROOT, "evals.jsonl")
SPLIT_PATH = os.path.join(ROOT, "rarity_split.json")
OUT_JSON = os.path.join(ROOT, "rarity_curves.json")
OUT_PNG = os.path.join(ROOT, "rarity_plot.png")

CURVE_RE = re.compile(r"^(all|inc|ent|hyb)_c(\d+)$")
OPERATING_POINTS = {"all": "all_c300", "ent": "ent_c300", "inc": "inc_c525", "hyb": "hyb_c300"}
METHOD_LABEL = {"all": "all-tokens", "inc": "incremental", "ent": "entropy", "hyb": "hybrid (ent_inc)"}
METHOD_COLOR = {"all": "#4C72B0", "inc": "#DD8452", "ent": "#55A868", "hyb": "#8172B3"}


def load_latest_rows(path):
    latest = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            latest[d["tag"]] = d  # last occurrence wins
    return latest


def load_hard_prompts(split_path, base_row):
    hard_prompts = None
    if os.path.exists(split_path):
        try:
            with open(split_path) as f:
                split = json.load(f)
            hp = split.get("hard_prompts")
            if hp and isinstance(hp, list) and len(hp) == 5:
                hard_prompts = hp
        except Exception:
            hard_prompts = None
    used_fallback = False
    if hard_prompts is None:
        used_fallback = True
        pp = sorted(base_row["per_prompt"], key=lambda x: x["r_mean"])
        hard_prompts = [p["prompt"] for p in pp[:5]]
    return hard_prompts, used_fallback


def pooled_samples(row, prompts):
    """Pool r_samples across the given prompt strings for a row."""
    by_prompt = {p["prompt"]: p for p in row["per_prompt"]}
    pooled = []
    for p in prompts:
        entry = by_prompt.get(p)
        if entry is None or "r_samples" not in entry:
            return None
        pooled.extend(entry["r_samples"])
    return pooled


def main():
    latest = load_latest_rows(EVALS_PATH)
    base = latest["base"]

    hard_prompts, hard_fallback = load_hard_prompts(SPLIT_PATH, base)
    all_prompts = [p["prompt"] for p in base["per_prompt"]]

    notes = []
    if hard_fallback:
        notes.append(
            "rarity_split.json hard_prompts unusable -> recomputed hard-5 "
            "from base per_prompt r_mean."
        )

    # --- eta computation ---
    base_hard_samples = pooled_samples(base, hard_prompts)
    eta_fallback_used = False
    if base_hard_samples is not None:
        eta = float(np.percentile(base_hard_samples, 75))
    else:
        eta_fallback_used = True
        by_prompt = {p["prompt"]: p for p in base["per_prompt"]}
        hard_means = [by_prompt[p]["r_mean"] for p in hard_prompts]
        eta = float(np.percentile(hard_means, 75))
        notes.append(
            "base row has NO r_samples on any per_prompt entry (confirmed "
            "for both raw occurrences of tag=='base' in evals.jsonl). "
            "eta was instead computed as the 75th percentile of the base "
            "row's 5 hard-prompt r_mean values (prompt-level, not "
            "sample-level)."
        )

    # --- curve tags ---
    curve_tags = [t for t in latest if CURVE_RE.match(t)]
    skipped = []
    results = []  # list of dicts: method, ckpt, tag, r_mean, hard_rare_frac, overall_frac

    for tag in curve_tags:
        row = latest[tag]
        m = CURVE_RE.match(tag)
        method, ckpt = m.group(1), int(m.group(2))

        hard_pooled = pooled_samples(row, hard_prompts)
        all_pooled = pooled_samples(row, all_prompts)
        if hard_pooled is None or all_pooled is None:
            skipped.append(tag)
            continue

        hard_rare_frac = float(np.mean(np.array(hard_pooled) >= eta))
        overall_frac = float(np.mean(np.array(all_pooled) >= eta))

        results.append(
            {
                "method": method,
                "ckpt": ckpt,
                "tag": tag,
                "r_mean": row["r_mean"],
                "hard_rare_frac": hard_rare_frac,
                "overall_frac": overall_frac,
            }
        )

    results.sort(key=lambda r: (r["method"], r["ckpt"]))

    # --- base row approximate rarity (prompt-level, since no r_samples) ---
    by_prompt_base = {p["prompt"]: p for p in base["per_prompt"]}
    base_hard_means = np.array([by_prompt_base[p]["r_mean"] for p in hard_prompts])
    base_all_means = np.array([p["r_mean"] for p in base["per_prompt"]])
    base_hard_rare_frac_approx = float(np.mean(base_hard_means >= eta))
    base_overall_frac_approx = float(np.mean(base_all_means >= eta))
    notes.append(
        "base operating-point rarity is an APPROXIMATION: fraction of "
        "per-prompt r_mean values >= eta (5 hard / 15 overall), not a "
        "true per-sample fraction, since base has no r_samples."
    )

    # --- stdout table ---
    print(f"eta (75th pct of base pooled hard-prompt r_samples) = {eta:.6f}")
    if eta_fallback_used:
        print("  [FALLBACK] base r_samples missing -> eta computed from base per-prompt r_mean (5 hard prompts).")
    print()
    header = f"{'method':<10}{'ckpt':>8}{'r_mean':>12}{'hard_rare_frac':>16}{'overall_frac':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['method']:<10}{r['ckpt']:>8}{r['r_mean']:>12.4f}"
            f"{r['hard_rare_frac']:>16.4f}{r['overall_frac']:>14.4f}"
        )
    if skipped:
        print()
        print("Skipped tags (missing r_samples):", skipped)

    print()
    print("Operating-point rarity comparison (hard_rare_frac):")
    op_results = {}
    for method, tag in OPERATING_POINTS.items():
        match = next((r for r in results if r["tag"] == tag), None)
        if match:
            op_results[method] = match
            print(f"  {method:<10} ({tag:<8}) hard_rare_frac={match['hard_rare_frac']:.4f}  r_mean={match['r_mean']:.4f}")
        else:
            print(f"  {method:<10} ({tag:<8}) MISSING")
    print(
        f"  {'base':<10} ({'base':<8}) hard_rare_frac(approx, prompt-level)="
        f"{base_hard_rare_frac_approx:.4f}  r_mean={base['r_mean']:.4f}"
    )

    # --- save JSON ---
    out = {
        "eta": eta,
        "eta_fallback_used": eta_fallback_used,
        "hard_prompts": hard_prompts,
        "hard_prompts_fallback_used": hard_fallback,
        "notes": notes,
        "skipped_tags": skipped,
        "curves": results,
        "base": {
            "r_mean": base["r_mean"],
            "hard_rare_frac_approx": base_hard_rare_frac_approx,
            "overall_frac_approx": base_overall_frac_approx,
        },
        "operating_points": {
            "all": "all_c300",
            "ent": "ent_c300",
            "inc": "inc_c525",
            "hyb": "hyb_c300",
        },
        "operating_point_results": {
            method: {
                "tag": r["tag"],
                "hard_rare_frac": r["hard_rare_frac"],
                "overall_frac": r["overall_frac"],
                "r_mean": r["r_mean"],
            }
            for method, r in op_results.items()
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved JSON -> {OUT_JSON}")

    # --- plot ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.2))

    for method in ["all", "ent", "inc", "hyb"]:
        pts = sorted(
            [r for r in results if r["method"] == method], key=lambda r: r["ckpt"]
        )
        if not pts:
            continue
        xs = [p["r_mean"] for p in pts]
        ys = [p["hard_rare_frac"] for p in pts]
        ax_left.plot(
            xs, ys, marker="o", color=METHOD_COLOR[method],
            label=METHOD_LABEL[method], linewidth=2, markersize=6,
        )

    ax_left.scatter(
        [base["r_mean"]], [base_hard_rare_frac_approx],
        marker="*", s=350, color="black", zorder=5,
        label="base (approx.)",
    )
    ax_left.set_xlabel("mean reward")
    ax_left.set_ylabel(r"hard-prompt rare-capture  $P(R \geq \eta)$")
    ax_left.set_title("Rare-capture vs. reward (curve checkpoints)")
    ax_left.annotate(
        "higher & more right\nis better", xy=(0.97, 0.03), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="gray",
        arrowprops=None,
    )
    ax_left.legend(frameon=False, fontsize=9)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    op_labels = []
    op_vals = []
    op_colors = []
    for method in ["all", "ent", "inc", "hyb"]:
        if method in op_results:
            op_labels.append(f"{METHOD_LABEL[method]}\n({op_results[method]['tag']})")
            op_vals.append(op_results[method]["hard_rare_frac"])
            op_colors.append(METHOD_COLOR[method])
    op_labels.append("base")
    op_vals.append(base_hard_rare_frac_approx)
    op_colors.append("black")

    bars = ax_right.bar(op_labels, op_vals, color=op_colors)
    ax_right.set_ylabel(r"hard-prompt rare-capture  $P(R \geq \eta)$")
    ax_right.set_title("Operating-point comparison")
    for b, v in zip(bars, op_vals):
        ax_right.text(
            b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax_right.annotate(
        "higher is better ↑", xy=(0.5, 1.05), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=9, color="gray",
    )
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.suptitle(
        f"Rarity capture on hard prompts (η={eta:.3f})"
        + ("  [η via r_mean fallback]" if eta_fallback_used else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
