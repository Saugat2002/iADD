#!/bin/bash
# Watcher for fkdiff_dense_ext run: waits for completion then runs eval sweep + auc_analysis.
set -u
export HF_HOME=~/dllm/hf_cache
cd ~/dllm/iadd-lm || exit 1

RUN_DIR=~/dllm/iadd-lm/runs/fkdiff_dense_ext_0922-1426
LOG=$RUN_DIR/log.jsonl
HEARTBEAT=/tmp/fkdiff_ext_watch.log
PY=~/miniconda3/envs/dllm/bin/python

echo "$(date) watcher started, run_dir=$RUN_DIR" >> "$HEARTBEAT"

while true; do
    n_lines=0
    if [ -f "$LOG" ]; then
        n_lines=$(wc -l < "$LOG")
    fi
    proc_alive=0
    if pgrep -af "[f]k_trace_grpo.py" | grep -q "run-tag fkdiff_dense_ext"; then
        proc_alive=1
    fi
    echo "$(date) poll n_lines=$n_lines proc_alive=$proc_alive" >> "$HEARTBEAT"

    if [ "$n_lines" -ge 595 ] || [ "$proc_alive" -eq 0 ]; then
        echo "$(date) run finished (n_lines=$n_lines proc_alive=$proc_alive), starting eval sweep" >> "$HEARTBEAT"
        break
    fi
    sleep 600
done

for it in 75 150 225 300 375 450 525 600; do
    ckpt="$RUN_DIR/backbone_it${it}.pt"
    tag="fk_c$((600+it))"
    echo "$(date) evaluating $ckpt tag=$tag" >> "$HEARTBEAT"
    "$PY" eval.py --ckpt "$ckpt" --tag "$tag" --reward sentiment >> /tmp/fkdiff_ext_evals.log 2>&1
done

"$PY" auc_analysis.py > /tmp/auc5_ext.txt 2>&1
echo FKDIFF_EXT_DONE >> /tmp/fkdiff_ext_evals.log
echo "$(date) watcher done" >> "$HEARTBEAT"
