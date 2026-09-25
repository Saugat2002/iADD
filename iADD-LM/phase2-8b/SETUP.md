# Phase 2: LLaDA-8B setup

Phase 1 validated both ideas (trace-GRPO step-selection curriculum, and
FK-diff inference-time steering) on MDLM-OWT (170M params). Phase 2 scales
to **LLaDA-8B**, a real 8B-parameter diffusion LLM, to see whether the same
effects hold at scale and on a harder, verifiable task (math, not sentiment).

This file is a setup plan, written before any Phase-2 GPU time was spent —
treat every number here as a budget estimate, not a measured result.

## Target model

- **`GSAI-ML/LLaDA-8B-Instruct`** on Hugging Face. An 8B-parameter masked
  discrete diffusion LLM (LLaDA = Large Language Diffusion with mAsking),
  instruction-tuned variant. Load via `transformers` with
  `trust_remote_code=True` (LLaDA ships custom modeling code, not a stock
  `AutoModelForCausalLM` architecture).

## Hardware

- **1x A100-80GB** is the target box for both LoRA training and FK
  inference-time steering, in bf16. 80GB headroom is needed for: the 8B
  model in bf16 (~16GB weights) + optimizer state for LoRA adapters (small)
  + activations for `k_particles`-way batched generation (FK steering runs
  several particles in parallel, which multiplies activation memory) +
  KV/diffusion-state caches during multi-step denoising.
- Workflow: rent on **vast.ai**, spin up, run the day-1 smoke test
  (`smoke_llada.py`) before committing to a multi-hour run.

## Environment

- Python 3.10+
- `torch>=2.4` (cu121 build, matching the Phase-1 environment)
- `transformers` (recent enough to have stable `trust_remote_code` support —
  pin a version once you've confirmed LLaDA loads with it)
- `peft` (LoRA)
- `accelerate`
- `math-verify` (or an equivalent GSM8K/math answer-checking library — used
  for scoring math correctness; anything that can pull a final numeric
  answer out of a completion and compare to ground truth works, `math-verify`
  is a reasonable off-the-shelf choice)

See `requirements.txt` in this folder.

## Budget

Rough GPU-hour / dollar budget, assuming ~$1.50-2.00/hr for a rented
A100-80GB on vast.ai (spot pricing varies — check at rent time):

| Phase | Est. GPU-h | Est. cost |
|---|---|---|
| FK inference-time steering study (training-free, run first) | 30-50 | $50-90 |
| d1-style anchor reproduction (**Gate B**, see below) | ~40-60 | $70-110 |
| TraceRL-style baseline reproduction | ~40-60 | $70-110 |
| iADD-LM trainer (trace-GRPO + FK-train) on LLaDA-8B | ~80-120 | $150-220 |
| **Total** | **~250-350** | **~$500-750** |

## Experiment order

1. **Day 1 smoke test** — `smoke_llada.py`: load the model, generate one
   completion with LLaDA's native diffusion sampler, confirm it runs and
   produces coherent text. This is a correctness check on the rented box's
   environment, not a research result.
2. **FK steering study first** (training-free, cheapest, highest
   information-per-dollar): repeat the Phase-1 FK-diff inference-time
   steering experiment on the base LLaDA-8B-Instruct model, on GSM8K-style
   math prompts, with a **continuous** reward signal (see caveat below). If
   FK-diff steering doesn't move the needle at 8B scale, that's a cheap,
   fast signal before spending the rest of the budget on training runs.
3. **Gate B — d1-style anchor.** Before trusting any new training result at
   8B scale, first reproduce the d1-style baseline's published/expected
   GSM8K accuracy (~82%) on your setup. This validates the eval harness,
   tokenization, and decoding loop against a known number. **Do not proceed
   to the iADD-LM trainer runs until this gate passes** — a silent eval bug
   at 8B scale is expensive to discover late.
4. TraceRL-style baseline reproduction (comparison point).
5. iADD-LM trainer (trace-GRPO curriculum + FK-train branch/resample),
   ported per `port_notes.md`.

## Binary-reward caveat (important — read before running FK on 8B)

Phase-1 FK potentials (`fk_trace_grpo.py --potential diff`) implicitly rely
on the reward signal varying smoothly enough mid-trajectory to be a useful
resampling weight before generation is complete (sentiment classifier score
on a partial completion is a reasonable continuous proxy). **GSM8K
correctness is binary (0/1) and only knowable at the very end** (once a
final numeric answer is extracted) — a 0/1 reward gives the FK potential
*no signal* to resample on mid-trajectory, since every incomplete particle
looks identical (reward undefined/zero) until the last step.

**Do not port `--potential diff` straight onto raw 0/1 correctness.** Use a
**continuous mid-trajectory signal** instead:
- partial-credit scoring (e.g. reward shaped by how much of the correct
  reasoning chain / intermediate steps are present),
- a self-verification / self-consistency score (e.g. the model's own
  confidence that the (partial) chain-of-thought is on track, or agreement
  across a small number of parallel continuations),
- or a learned/lightweight process-reward-model style score.

The **final** correctness (0/1, math-verify) is still the metric you report
and gate on (Gate B above uses it); it's specifically the **FK potential's**
input that needs to be continuous, since that's what drives resampling
*during* generation, not just the final eval score.
