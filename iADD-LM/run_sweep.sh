#!/bin/bash
# iADD-LM Phase 1, Step 2: step-selection sweep (sequential, nohup-safe).
#   nohup bash ~/dllm/iadd-lm/run_sweep.sh &
# Runs base-model eval once, then for each selection rule: 300-iter
# trace-GRPO + eval of the final checkpoint. Waits for >8000MiB free GPU
# memory before each run.

source ~/miniconda3/etc/profile.d/conda.sh && conda activate dllm
export HF_HOME=~/dllm/hf_cache

cd ~/dllm/iadd-lm
mkdir -p runs
LOG=~/dllm/iadd-lm/runs/sweep_$(date +%m%d-%H%M).log
echo "=== sweep started $(date) ===" >> "$LOG"

wait_for_gpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -gt 8000 ]; then break; fi
    echo "$(date +%H:%M) gpu busy (free=${free}MiB), waiting" >> "$LOG"
    sleep 120
  done
}

# baseline: base model eval (no training)
wait_for_gpu
echo "=== $(date) eval base model ===" >> "$LOG"
python eval.py --ckpt none --reward sentiment --group 8 --tag base >> "$LOG" 2>&1

for select in all early late any entropy incremental; do
  wait_for_gpu
  echo "=== $(date) training select=$select ===" >> "$LOG"
  python trace_grpo.py --reward sentiment --iters 300 --group 8 --steps 128 \
      --select "$select" --select-n 16 --micro-bs 4 >> "$LOG" 2>&1
  status=$?
  echo "=== select=$select train exit=$status ===" >> "$LOG"

  ck=$(ls -t runs/sentiment_${select}16_g8_*/backbone_it*.pt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    wait_for_gpu
    echo "=== $(date) eval $ck ===" >> "$LOG"
    python eval.py --ckpt "$ck" --reward sentiment --group 8 --tag "$select" >> "$LOG" 2>&1
  else
    echo "WARN: no checkpoint found for select=$select" >> "$LOG"
  fi
done

echo "=== sweep done $(date) ===" >> "$LOG"
