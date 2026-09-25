# Porting fk_trace_grpo.py: MDLM-170M -> LLaDA-8B

Notes on what has to change (and what should NOT change) when moving the
Phase-1 trainer (`../fk_trace_grpo.py`, built on `../trace_grpo.py`) from
MDLM-OWT (170M) to LLaDA-8B. Written during Phase-2 planning, before any
porting work has actually started — treat as a checklist, not a diff.

## What changes

- **Tokenizer / vocab.** MDLM-OWT uses GPT-2 BPE (~50k vocab) with a single
  `[MASK]` token id baked into the schedule math in `trace_grpo.py`
  (`build_config`, `dataloader.get_tokenizer`). LLaDA-8B has its own
  tokenizer and mask-token id — every place `trace_grpo.py` /
  `fk_trace_grpo.py` hardcodes or infers the mask id (search for
  `mask_index` / `mask_token_id`-equivalent) needs to be re-derived from
  LLaDA's tokenizer, not assumed.

- **Decode loop: block/confidence unmasking, not ddpm-style.** MDLM uses a
  continuous-time ddpm-style reverse process (see `trace_grpo.py`'s module
  docstring: `P(commit v) = p_x0(v)*(mct-mcs)/mct`, evaluated at
  `torch.linspace(1, eps, steps+1)` timesteps). LLaDA-8B instead decodes in
  **fixed-size blocks**, unmasking within each block by **confidence order**
  (highest-confidence-first, a discrete number-of-steps-per-block schedule,
  not a continuous SDE). This is a materially different decode loop, not a
  parameter change:
  - The trace-recording logic in `TraceGRPO.rollout` (which token committed
    at which step, with its exact log-prob, used to exactly recompute
    log-probs for the PPO-clip ratio) needs to be rewritten against LLaDA's
    block/confidence loop, recording *which position was unmasked in which
    block-step* instead of *which position transitioned under the ddpm
    reverse kernel*.
  - The log-prob recomputation check (`verify.py`'s `check_logprob`, and
    the corresponding exact-recompute invariant in `TraceGRPO`) needs an
    equivalent for the new loop: log p(token | context at the step it was
    unmasked), which should still be well-defined and recomputable, but the
    formula in `trace_grpo.py`'s docstring is MDLM-specific and doesn't
    directly transfer.
  - Step-selection rules (`--select {all,early,late,any,entropy,incremental,
    ent_inc}`) operate on "steps" — decide whether "step" now means
    "block" or "within-block unmasking step" for LLaDA, and update the
    entropy/incremental selection logic (which currently indexes into a
    flat `steps`-length trace) accordingly.

- **Full fine-tuning -> LoRA via `peft`.** Phase-1 trains the full 170M
  backbone (`torch.optim.AdamW(model.backbone.parameters(), ...)` in
  `trace_grpo.py`). At 8B, full fine-tuning is not the plan — wrap the
  backbone with `peft.get_peft_model(..., LoraConfig(...))` and optimize
  only the LoRA adapter parameters. This changes:
  - checkpoint save/load (`--init-ckpt`, `--out`, the `backbone_itN.pt`
    naming convention) — LoRA adapters save as a small adapter dir, not a
    full backbone state dict; update the checkpoint I/O accordingly rather
    than assuming a monolithic `.pt` file.
  - the `IADD_KEEP_ALL_CKPTS` pruning logic can stay conceptually the same
    (still want every checkpoint for curve evals) but adapter checkpoints
    are much smaller, so the disk-space motivation for pruning is weaker —
    consider defaulting to keep-all at 8B scale.

- **Reward = verifiable math score, not sentiment.** Swap the
  `REWARD_NAMES` / `--reward sentiment` sentiment classifier for a GSM8K (or
  similar) correctness scorer (`math-verify`-style: extract final numeric
  answer, compare to ground truth, 0/1). This reward is what Gate B (the
  d1-style ~82% GSM8K anchor, see `SETUP.md`) validates the eval harness
  against.

- **Potential = partial credit, not raw reward.** See `SETUP.md`'s binary-
  reward caveat: the FK potential (`fk_potentials` / `--potential diff`)
  needs a *continuous* signal at intermediate steps, which the final 0/1
  correctness reward cannot provide. Introduce a separate
  partial-credit / self-verification scoring function specifically for the
  potential, decoupled from the final reward used for the RL advantage and
  for the reported metric.

## What should NOT change

- **The diff potential itself.** `--potential diff` (the telescoping
  difference potential) is the one Phase-1 result that clearly worked (see
  the main README's Known Gotchas — `max` is broken, `diff` isn't). Keep
  the potential *formula* (telescoping diff of scores between resample
  points) unchanged; only the *input signal* to it changes (partial-credit
  score instead of sentiment score).
- **Keep-distinct / lineage machinery.** The particle bookkeeping that
  keeps resampled particles distinct and tracks lineage across
  branch/resample events (so you can attribute a final completion back to
  its ancestry for diversity metrics) is decode-loop-agnostic — it operates
  on particle indices and trace records, not on the specific unmasking
  schedule. Port this machinery as-is; it should plug into the new
  block/confidence loop the same way it plugged into the ddpm loop, as long
  as the new loop still produces one trace record per particle per
  resample-eligible point.
- **Resample modes / fractions / lambda.** `--resample-mode multinomial`,
  `--resample-fracs`, `--lmbda` are all schedule-agnostic hyperparameters of
  the FK resampling step itself, not of the underlying diffusion decode
  loop — carry the Phase-1 values over as starting points for Phase-2
  sweeps, don't assume they need rederivation.

## Suggested porting order

1. Get `smoke_llada.py`'s native-sampler generation working first (day-1
   sanity check, no training).
2. Reimplement just the trace-recording + exact-recompute half of
   `TraceGRPO.rollout` against LLaDA's block/confidence loop, and validate
   it with a LLaDA-adapted version of `verify.py`'s `check_logprob` before
   writing a single line of RL training code.
3. Wire up LoRA (`peft`) in place of full backbone optimization.
4. Add the GSM8K reward + partial-credit potential-scoring function.
5. Only then re-enable `--select` step-selection and FK branch/resample —
   at that point it's the same trainer loop as Phase 1, just pointed at a
   different decode loop, reward, and potential.
