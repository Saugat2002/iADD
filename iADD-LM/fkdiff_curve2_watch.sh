#!/bin/bash
# Watcher for fkdiff_curve run: waits for completion then runs full eval curve + auc_analysis.
set -u
export HF_HOME=~/dllm/hf_cache
cd ~/dllm/iadd-lm || exit 1

RUN_DIR=~/dllm/iadd-lm/runs/fkdiff_curve_0922-1609
LOG=$RUN_DIR/log.jsonl
HEARTBEAT=/tmp/fkdiff_curve2_watch.log
PY=~/miniconda3/envs/dllm/bin/python

echo "$(date) watcher started, run_dir=$RUN_DIR" >> "$HEARTBEAT"

while true; do
    n_lines=0
    if [ -f "$LOG" ]; then
        n_lines=$(wc -l < "$LOG")
    fi
    proc_alive=0
    if pgrep -af "[f]k_trace_grpo.py" | grep -q "run-tag fkdiff_curve"; then
        proc_alive=1
    fi
    echo "$(date) poll n_lines=$n_lines proc_alive=$proc_alive" >> "$HEARTBEAT"

    if [ "$n_lines" -ge 1195 ] || [ "$proc_alive" -eq 0 ]; then
        echo "$(date) run finished (n_lines=$n_lines proc_alive=$proc_alive), starting eval sweep" >> "$HEARTBEAT"
        break
    fi
    sleep 600
done

for it in 75 150 225 300 375 450 525 600 675 750 825 900 975 1050 1125 1200; do
    ckpt="$RUN_DIR/backbone_it${it}.pt"
    tag="fk_c$it"
    if [ -f "$ckpt" ]; then
        echo "$(date) evaluating $ckpt tag=$tag" >> "$HEARTBEAT"
        "$PY" eval.py --ckpt "$ckpt" --tag "$tag" --reward sentiment >> /tmp/fkdiff_curve2_evals.log 2>&1
        if [ "$it" -ne 1200 ]; then
            realpath "$ckpt" >> /tmp/deletable_ckpts.txt
        fi
    else
        echo "$(date) missing ckpt $ckpt, skipping" >> "$HEARTBEAT"
    fi
done

"$PY" auc_analysis.py > /tmp/auc5_final.txt 2>&1
echo FKDIFF_CURVE2_DONE >> /tmp/fkdiff_curve2_evals.log
echo "$(date) watcher done" >> "$HEARTBEAT"
