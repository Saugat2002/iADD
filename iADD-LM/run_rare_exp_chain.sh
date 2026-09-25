#!/bin/bash
# Sequential chain: baseline sparse trace-GRPO, then FK sparse trace-GRPO.
# Touches /tmp/rare_exp_done when both finish (or one fails -- check logs).
set -x
export HF_HOME=~/dllm/hf_cache
cd ~/dllm/iadd-lm
PY=~/miniconda3/envs/dllm/bin/python
ETA=$(python3 -c "import json; print(json.load(open('eta_star.json'))['eta_star'])")
echo "Using eta_star=$ETA"

echo "=== [1/2] baseline trace_grpo.py (sparse) ==="
$PY trace_grpo.py --reward sentiment --reward-threshold "$ETA" \
    --iters 300 --group 8 --steps 128 --select incremental --select-n 16 \
    --micro-bs 4 --seed 1234 --save-every 100 \
    --prompt-file ~/dllm/iadd-lm/hard_prompts.jsonl \
    > ~/dllm/iadd-lm/baseline_sparse.log 2>&1
echo "baseline exit code: $?"

echo "=== [2/2] FK fk_trace_grpo.py (sparse) ==="
# fk_trace_grpo.py has no --group flag (group size == --k-particles);
# all other flags mirror the baseline exactly.
$PY fk_trace_grpo.py --reward sentiment --reward-threshold "$ETA" \
    --iters 300 --steps 128 --select incremental --select-n 16 \
    --micro-bs 4 --seed 1234 --save-every 100 \
    --prompt-file ~/dllm/iadd-lm/hard_prompts.jsonl \
    --k-particles 8 --branch-frac 0.0 --resample-mode keep-distinct \
    --adv-mode pair --resample-fracs 0.6 \
    > ~/dllm/iadd-lm/fk_sparse.log 2>&1
echo "fk exit code: $?"

touch /tmp/rare_exp_done
echo "chain done"
