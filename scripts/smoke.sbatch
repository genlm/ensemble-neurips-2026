#!/bin/bash
#SBATCH -J ens-smoke
#SBATCH -n 1 --cpus-per-task=8 --gpus=2 --time=06:00:00 --mem-per-cpu=8g
#SBATCH -o _smoke/smoke-%j.out
# Smoke test: every task, token level (within-model) and byte level (cross-model), 3 examples,
# 3 particles, two operators. Submit from the repository root:
#   sbatch $SBATCH_EXTRA --gres=$GPU_MEM_SPEC scripts/smoke.sbatch
# Site settings come from scripts/env.sh (see there).
set -uo pipefail
. "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"
W=$ENSEMBLE_BENCH_ROOT
mkdir -p $W/_smoke/out && cd $W/_smoke
job_header
MODEL_1=${MODEL_1:-meta-llama/Llama-3.1-8B-Instruct}
MODEL_2=${MODEL_2:-Qwen/Qwen2.5-7B-Instruct}
COMMON="--n 3 --particles 3 --ops min sum --max-tokens 48 --log-every 1 --out-dir $W/_smoke/out"
fail=()
for task in bbh mmlu json_schema gsm8k spider; do
  echo "=================== token / $task ==================="
  timeout 60m $ENSEMBLE_PYTHON -m ensemble_bench.smc --task $task --level token --model-1 $MODEL_1 $COMMON 2>&1 | grep -vE "$LOG_FILTER" | tail -40
  test ${PIPESTATUS[0]} -eq 0 || fail+=("token/$task")
done
for task in bbh mmlu json_schema gsm8k spider; do
  echo "=================== byte / $task (cross-model) ==================="
  timeout 60m $ENSEMBLE_PYTHON -m ensemble_bench.smc --task $task --level byte --model-1 $MODEL_1 --model-2 $MODEL_2 $COMMON 2>&1 | grep -vE "$LOG_FILTER" | tail -40
  test ${PIPESTATUS[0]} -eq 0 || fail+=("byte/$task")
done
# per-example errors are caught by the runner (exit code 0); treat any as a failure
for f in $W/_smoke/out/*-metrics.jsonl; do
  grep -q '"n_errors": [1-9]' "$f" && fail+=("errors in $(basename $f | cut -d- -f2-5)")
done
echo "=================== SMOKE SUMMARY ==================="
ls $W/_smoke/out | sed 's/^/  /'
if [ ${#fail[@]} -eq 0 ]; then echo "ALL 10 SMOKE RUNS PASSED"; else echo "FAILED: ${fail[*]}"; fi
