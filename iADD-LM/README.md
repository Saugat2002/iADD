# iADD-LM: diffusion-LM extension of iADD

This folder is a snapshot of the working codebase from `ajad:/home/ajad/dllm/iadd-lm/`,
pulled in for pickup by someone starting on a fresh server. It sits alongside the
original iADD (diffusion-policy image) codebase at the repo root; the two are
independent projects that share a name and a research lineage, not a Python
dependency.

## Overview

iADD-LM extends the (rejected-from-ECCV) iADD paper's ideas from image diffusion
policies to discrete diffusion language models (dLLMs). Two ideas are ported:

1. **Sparse-incremental / entropy step-selection curriculum for trace-GRPO
   training.** Instead of computing the PPO-clip loss over every denoising
   step of a trace, select a subset `S` of steps by a rule (`all`, `early`,
   `late`, `any`, `entropy`, `incremental`, or the hybrid `ent_inc`) and train
   only on those. This is the RL-training-time idea.
2. **Feynman-Kac branch/resample with a TELESCOPING diff potential**, used at
   **inference time** (no training) to steer generation toward high-reward,
   diverse completions via particle branching/resampling mid-trajectory. The
   potential comes in two flavors, `max` and `diff`; **`diff` is the one that
   works** — see Known gotchas.

**Status:** Phase 1 is done, on MDLM-OWT (170M-parameter discrete diffusion
LM, OpenWebText-pretrained, `kuleshov-group/mdlm-owt`). Phase 2 targets
LLaDA-8B and is scaffolded but not yet run — see `phase2-8b/`.

## Environment

- Conda env name on ajad: `dllm`. Python 3.10.
- PyTorch 2.4.1+cu121.
- `transformers`, plus `sentence-transformers`/MiniLM for the diversity
  metric (falls back to a plain `transformers` load of the MiniLM checkpoint
  if `sentence-transformers` isn't installed — see `eval.py`).
- **No flash-attn required.** The ajad GPU is pre-Ampere (sm75, no flash-attn
  support), so the base MDLM code was patched to fall back to PyTorch SDPA
  attention. Those patches live in `patches/sm75/` in this folder — see
  `patches/sm75/NOTES.md`. On an Ampere/Hopper box (A100/H100, sm80+),
  flash-attn works unpatched and you can use the upstream files instead.
- Base model: `kuleshov-group/mdlm-owt` from Hugging Face (~0.7GB), loaded
  from a local clone at `~/dllm/mdlm-owt-local` on ajad (the patched
  `modeling_mdlm.py` lives inside that local clone; the patch file mirrored
  here is `patches/sm75/modeling_mdlm.py`).
- **Upstream dependency:** these scripts are *not* self-contained — they
  `sys.path.insert` and `os.chdir` into a sibling checkout of
  **Fk-Diffusion-Steering** (the Feynman-Kac steering codebase this work
  builds on), expected at `~/dllm/Fk-Diffusion-Steering/discrete_diffusion`
  on the training/eval host, and referenced in-code as `FK_DIR`:

  ```python
  FK_DIR = os.path.expanduser('~/dllm/Fk-Diffusion-Steering/discrete_diffusion')
  sys.path.insert(0, FK_DIR)
  sys.path.insert(0, os.path.join(FK_DIR, 'mdlm'))
  os.chdir(FK_DIR)  # hydra searchpath file://mdlm/configs is cwd-relative
  ```

  So the expected layout on a fresh box is:

  ```
  ~/dllm/
    iadd-lm/                          <- this folder's contents go here
    Fk-Diffusion-Steering/            <- clone of the FKD steering repo (sibling)
      discrete_diffusion/
        mdlm/...
        configs/...
        evaluation/pplm_discrim_prompts_orig.jsonl
    mdlm-owt-local/                   <- local clone of kuleshov-group/mdlm-owt
      modeling_mdlm.py                <- patched for sm75 (patches/sm75/modeling_mdlm.py)
  ```

  Every script `chdir`s into `Fk-Diffusion-Steering/discrete_diffusion` at
  import time (hydra's searchpath is cwd-relative), so run everything from
  that directory or just invoke scripts with an absolute path from anywhere
  — the `chdir` happens inside the script itself.

## File-by-file guide

- `trace_grpo.py` — trace-based GRPO trainer. Records the full denoising
  trace (which tokens committed at which step, with exact categorical
  log-probs) and runs PPO-clip on a selectable subset of steps.
  `--select {all,early,late,any,entropy,incremental,ent_inc}`.
- `trace_grpo_fast.py` — a faster variant of the same trainer (see
  `BENCH.md` for the benchmark harness comparing the two; adds
  `--prompts-per-iter`, `--amp`).
- `fk_trace_grpo.py` — Feynman-Kac particle trainer, and (via `--eval-only`)
  the **inference-time steering evaluator** — same script, two modes. Key
  flags: `--potential {max,diff}` (**use `diff`** — `max` is broken in the
  discrete regime, see gotchas), `--k-particles`, `--branch-frac`,
  `--resample-mode {multinomial,...}`, `--resample-fracs`, `--lmbda`,
  `--init-ckpt` (resume/base a run off a saved checkpoint), `--eval-only N`
  (skip training, just run N eval episodes with FK steering), `--eval-save-texts`
  (dump generated text into the eval record), `--adv-mode {group,...}`.
  Env var `IADD_KEEP_ALL_CKPTS=1` disables the default checkpoint-pruning
  behavior (needed for curve/AUC runs that need every checkpoint on disk).
- `eval.py` — reward + diversity (semantic, via MiniLM embeddings) + rarity
  eval of a checkpoint (or `--ckpt none` for the base model); appends one
  JSON record per run to `evals.jsonl`.
- `auc_analysis.py` — turns `evals.jsonl` curves into AUC numbers, common-
  support operating points, and Pareto fronts; produces `auc_report.json`
  and `tradeoff_curves.png`.
- `rare_analysis.py` — rarity-regime analysis (rare-token/rare-completion
  success rates) from run directories; produces `rare_plot.png`.
- `rarity_curves.py` — builds rarity curves across sparse/eta-thresholded
  runs.
- `rarity_split.py` — splits prompts/completions into rare vs. common
  buckets, feeding the rarity metrics above.
- `calibrate_eta.py` — calibrates the reward threshold `eta` for a target
  positive rate (`--target-rate`), writes `eta_star.json`; feeds the
  `--reward-threshold` sparse/rare-success training mode.
- `analyze_fkinfer.py` — analysis helper for FK-inference-only runs.
- `d1_style.py` / `dm_style.py` — baseline reimplementations (non-trace-GRPO
  RL baselines) used as comparison points.
- `reach_probe.py` — the Prop-1 reachability probe.
- `verify.py` — 3 sanity checks, run with `--check {logprob,noop,resample,all}`:
  (1) `logprob`: recorded trace log-probs exactly recompute from the model;
  (2) `noop`: constant reward -> zero advantage -> no weight update;
  (3) `resample`: `lmbda -> 0` FK resampling is uniform.
- `leaderboard.py` — builds a leaderboard table/plot (`leaderboard.md`/`.png`,
  not copied here — regenerate by rerunning against `evals.jsonl`).
- `run_sweep.sh` / `run_rare_exp_chain.sh` / `fkdiff_curve2_watch.sh` /
  `fkdiff_ext_watch.sh` — shell drivers for sweeps and long-running watch
  loops used during Phase 1 experimentation.
- `hard_prompts.jsonl` — 5 rare-regime prompts used for targeted rarity
  evals.
- `evals.jsonl` — **the Phase-1 results database.** Every reported number in
  this README's results table is reproducible by filtering/aggregating this
  file (via `auc_analysis.py` / `rare_analysis.py`). Treat it as the source
  of truth over anything written in prose.
- `auc_report.json`, `reach.json` — cached outputs of `auc_analysis.py` /
  `reach_probe.py` at the time of the snapshot.
- `tradeoff_curves.png`, `rare_plot.png` — the corresponding plots.
- `BENCH.md` — a not-yet-run benchmark plan comparing `trace_grpo.py` vs
  `trace_grpo_fast.py` wall-clock (explicitly marked "do NOT run yet" at
  snapshot time — check before relying on it).

Not copied here (excluded per pickup scope): `runs/` (training run
directories — potentially large, includes logs/checkpoints),
`*.pt` checkpoint files, `__pycache__/`, `evals.jsonl.bak*` backup files,
and a handful of one-off logs/plots from ajad (`*.log`, `*.out`,
`leaderboard.md`/`.png`, `rarity_split.*`, `rarity_curves.json`,
`rarity_plot.png`, `chain.log`, `eta_star.json`) that are regenerable from
the scripts above and not needed to reproduce the headline numbers.

## Reproduce Phase 1

Run all of these from `~/dllm/Fk-Diffusion-Steering/discrete_diffusion`
(or let the script's own `os.chdir` handle it), with the `dllm` conda env
active and `HF_HOME` pointed at a cache with the base model downloaded.
**Check each script's `argparse` block before trusting these flags on a
newer/renamed checkout — the code is ground truth, not this file.**

### (a) A curriculum training run (entropy+incremental hybrid)

```bash
python ~/dllm/iadd-lm/trace_grpo.py \
    --reward sentiment --iters 600 --group 8 \
    --select ent_inc --select-n 16 --micro-bs 4 \
    --seed 1234 --save-every 75
```

Swap `--select` for `all` / `early` / `late` / `any` / `entropy` /
`incremental` to reproduce the other curriculum baselines in the AUC table.

### (b) FK particle training (train-time branch/resample)

```bash
python ~/dllm/iadd-lm/fk_trace_grpo.py \
    --reward sentiment --iters 600 --group 8 \
    --k-particles 8 --potential diff --branch-frac 0.2 \
    --resample-mode multinomial --resample-fracs 0.2 0.4 0.6 0.8 \
    --lmbda 2.0 --select ent_inc --select-n 16 --micro-bs 4 \
    --seed 1234 --save-every 75
```

Set `IADD_KEEP_ALL_CKPTS=1` in the environment first if you'll later run
`auc_analysis.py` against this run's full checkpoint curve.

### (c) FK-diff inference-time steering, eval-only (no training)

This is the training-free result — steer the **base** model with FK
branch/resample at inference and eval directly, no RL training involved:

```bash
python ~/dllm/iadd-lm/fk_trace_grpo.py \
    --eval-only 120 --k-particles 8 --potential diff \
    --branch-frac 0.0 --resample-mode multinomial \
    --resample-fracs 0.2 0.4 0.6 0.8 --lmbda 2.0 \
    --reward sentiment --eval-save-texts
```

(`--branch-frac 0.0` with `--eval-only` set means: no training-time
branching config needed since we never train; only the eval-time FK
resampling loop runs.)

### (d) Curve evals + AUC analysis

Run `eval.py` against a sweep of checkpoints from a curve run (with
`IADD_KEEP_ALL_CKPTS=1` set during training so all checkpoints survive
pruning):

```bash
python ~/dllm/iadd-lm/eval.py --ckpt /absolute/path/to/backbone_it300.pt \
    --reward sentiment --group 8 --tag ent_inc_it300
```

`--ckpt` must be an **absolute path**; `eval.py` does not expand `~` or
resolve relative paths reliably for checkpoint loading. Repeat across the
checkpoint sweep, then:

```bash
python ~/dllm/iadd-lm/auc_analysis.py
```

which reads `evals.jsonl`, computes five-way common-support AUC and Pareto
operating points, and writes `auc_report.json` + `tradeoff_curves.png`.

### (e) Rarity pipeline

```bash
# 1. calibrate the sparse-reward threshold for a 5% positive rate
python ~/dllm/iadd-lm/calibrate_eta.py --target-rate 0.05

# 2. sparse-reward training runs, using the calibrated eta as
#    --reward-threshold (read it out of eta_star.json)
python ~/dllm/iadd-lm/trace_grpo.py \
    --reward sentiment --iters 600 --group 8 --select ent_inc \
    --select-n 16 --micro-bs 4 --reward-threshold <eta_star>

# 3. rarity analysis over the resulting run directories
python ~/dllm/iadd-lm/rare_analysis.py --runs-root ~/dllm/iadd-lm/runs
```

## Phase-1 headline results

(Numbers pulled from `evals.jsonl` / `auc_report.json` / `reach.json` at
snapshot time; treat `evals.jsonl` as ground truth if these ever disagree.)

**Five-way common-support AUC** (higher = better reward/diversity
trade-off across the operating-point sweep):

| Variant | AUC |
|---|---|
| hybrid (`ent_inc`) | 90.8 |
| entropy (`entropy`) | 89.5 |
| incremental (`incremental`) | 88.2 |
| FK-train (`fk_trace_grpo.py`, training-time branch/resample) | 87.2 |
| all-steps baseline (`select=all`) | 84.2 |

**FK-diff inference-time steering on the base (untrained) model:**
reward ≈ −0.32, semantic diversity ≈ 0.724 — this **beats every trained
operating point above, training-free**, just by steering the base model at
inference.

**Rarity (rare-completion success rate):**

| Setting | Rate |
|---|---|
| Base model | 0.025 |
| FK steering (base model, inference-time) | 0.258 (~10x) |
| FK-diff, 8 particles, select-best | 0.525 |
| Best-of-8 (no FK, just resampling) baseline | 0.350 |
| FK-max (broken potential) | 0.000 |

Full write-ups (PDF reports, not stored in this repo): referenced
externally as the Phase-1 report and the rarity-regime addendum. Ask
whoever ran the ajad experiments for the PDFs if needed — they were not
part of the `iadd-lm/` working directory and so aren't in this snapshot.

## Known gotchas

- **`--potential max` is broken in the discrete regime** — it collapses to
  degenerate/always-zero behavior (see the FK-max rarity rate of 0.000
  above). Always use `--potential diff` (the telescoping diff potential).
- **Checkpoint pruning is on by default** during training; if you're going
  to run a curve eval / AUC analysis over a run afterward, set
  `IADD_KEEP_ALL_CKPTS=1` in the environment *before* starting that training
  run, or you'll only have the last checkpoint to work with.
- `z_traj` (the denoising trajectory tensor) is stored as `int32`; watch for
  silent dtype mismatches if you extend the trace-recording code.
- The entropy step-selection rule had an off-by-one bug that was fixed
  during Phase 1 — if you're diffing against an older snapshot of
  `trace_grpo.py`, make sure you have the fixed version.
- `eval.py --ckpt` needs an **absolute path** — passing a relative or
  `~`-prefixed path silently fails to load the intended checkpoint (or
  loads the base model instead if `--ckpt none`-like fallback logic kicks
  in). Always pass the full resolved path.
- All scripts assume they're either run from, or will `chdir` into,
  `Fk-Diffusion-Steering/discrete_diffusion` — hydra's config searchpath is
  cwd-relative. If you see hydra "config not found" errors, this is why.

## Directory contents in this snapshot

```
iADD-LM/
  README.md                  <- this file
  BENCH.md
  trace_grpo.py
  trace_grpo_fast.py
  fk_trace_grpo.py
  eval.py
  auc_analysis.py
  rare_analysis.py
  rarity_curves.py
  calibrate_eta.py
  analyze_fkinfer.py
  d1_style.py
  dm_style.py
  reach_probe.py
  verify.py
  leaderboard.py
  rarity_split.py
  run_sweep.sh
  run_rare_exp_chain.sh
  fkdiff_curve2_watch.sh
  fkdiff_ext_watch.sh
  hard_prompts.jsonl
  evals.jsonl
  auc_report.json
  reach.json
  tradeoff_curves.png
  rare_plot.png
  patches/sm75/            <- flash-attn -> SDPA fallback patches, see NOTES.md
    dit.py
    modeling_mdlm.py
    NOTES.md
  phase2-8b/                <- Phase-2 (LLaDA-8B) setup, not yet run
    SETUP.md
    requirements.txt
    smoke_llada.py
    port_notes.md
```
