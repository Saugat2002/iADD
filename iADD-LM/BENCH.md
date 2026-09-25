# Benchmark: trace_grpo.py vs trace_grpo_fast.py

Run when the GPU is free (both commands run from
`~/dllm/Fk-Diffusion-Steering/discrete_diffusion`, same as the existing
scripts require). 5 iterations each is enough to compare per-iteration
wall-clock in `log.jsonl` -- do NOT run this yet, the GPU is busy with an
overnight sweep.

```bash
cd ~/dllm/Fk-Diffusion-Steering/discrete_diffusion && \
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dllm && \
export HF_HOME=~/dllm/hf_cache && \
python ~/dllm/iadd-lm/trace_grpo.py --iters 5 --group 8 --steps 128 \
    --select all --out ~/dllm/iadd-lm/runs_bench \
  && python ~/dllm/iadd-lm/trace_grpo_fast.py --iters 5 --group 8 --steps 128 \
    --select all --prompts-per-iter 2 --micro-bs 8 --amp \
    --out ~/dllm/iadd-lm/runs_bench
```

Then compare the `sec` field per iteration (skip iter 0 of each if it looks
like an outlier -- CUDA context / cudnn autotune warmup):

```bash
echo old:  && tail -n +2 ~/dllm/iadd-lm/runs_bench/<old_run_name>/log.jsonl | python3 -c "import sys,json; [print(json.loads(l).get('sec')) for l in sys.stdin]"
echo fast: && tail -n +2 ~/dllm/iadd-lm/runs_bench/<fast_run_name>/log.jsonl | python3 -c "import sys,json; [print(json.loads(l).get('sec')) for l in sys.stdin]"
```

(`<old_run_name>` / `<fast_run_name>` are the timestamped directories
`ls -t ~/dllm/iadd-lm/runs_bench` prints newest-first; the fast one is
prefixed `fast_`.) Compare mean `sec` over iters 1-4 (drop iter 0) between
the two runs -- that ratio is the realized speedup.
