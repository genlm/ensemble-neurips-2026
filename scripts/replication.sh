#!/bin/bash
# Replication grid, single seed:
#   token, within-model  Llama-3.1-8B x Llama-3.1-8B  (paper prompt pair), 5 tasks x 4 ops  -> 5 jobs, 1 x 24 GB each
#   byte,  cross-model   Llama-3.1-8B x Qwen2.5-7B    (same prompt both),  5 tasks x 4 ops  -> 20 jobs (one per op), 2 x 24 GB each
#
#   scripts/replication.sh [out_dir] [seed]        (site settings: scripts/env.sh; ONLY=token|byte, TASKS=..., OPS=... to submit a subset)
set -euo pipefail
. "$(dirname "$0")/env.sh"
W=$ENSEMBLE_BENCH_ROOT
OUT=${1:-$W/results/replication}
SEED=${2:-42}
LLAMA=${MODEL_1:-meta-llama/Llama-3.1-8B-Instruct}
QWEN=${MODEL_2:-Qwen/Qwen2.5-7B-Instruct}
mkdir -p "$OUT/logs"

MEM_TOKEN=8g; MEM_BYTE=${MEM_BYTE:-20g}   # per CPU (8 CPUs): byte level keeps up to ~130 GB of beams on the host
submit() {  # name, gpus, time, args...
  local name=$1 gpus=$2 time=$3; shift 3
  local mem=$([ "$gpus" = 2 ] && echo $MEM_BYTE || echo $MEM_TOKEN)
  printf '#!/bin/bash\n. %s/scripts/env.sh; cd %s; job_header; %s\n' "$W" "$W" \
    "$ENSEMBLE_PYTHON -m ensemble_bench.smc $* --seed $SEED --out-dir $OUT/$(echo $name | cut -d- -f1-2) 2>&1 | grep -vE '$LOG_FILTER'" |
  sbatch --parsable -J "$name" -n 1 --cpus-per-task=8 --gpus=$gpus --gres=$GPU_MEM_SPEC --time=$time --mem-per-cpu=$mem \
    ${SBATCH_EXTRA:-} -o "$OUT/logs/$name-%j.out"
}

# cross-model byte prompt: the paper's rule (Tab. 5) -- the prompt with the better average base-model
# accuracy across Qwen/Phi/Llama (Tab. 3/4): bbh P1 26.7 vs 22.2, spider P1 34.2 vs P2 41.5,
# json P1 88.5 vs 85.9, gsm8k P1 (few_shot_cot) 52.7 vs P2 (precision_cot) 54.4, mmlu P1 63.3 vs P2 67.7.
declare -A BYTE_T=( [bbh]=instructional [spider]=mid_1 [json_schema]=default [gsm8k]=precision_cot [mmlu]=template2 )
ONLY=${ONLY:-}; TASKS=${TASKS:-bbh spider mmlu json_schema gsm8k}; OPS=${OPS:-min prod sum max}   # subsets for relaunches
for task in $TASKS; do
  if [ "$ONLY" != byte ]; then
    j=$(submit "token-$task" 1 12:00:00 --task $task --level token --model-1 $LLAMA --n 100)
    echo "token/$task  job $j"
  fi
  [ "$ONLY" = token ] && continue
  t=${BYTE_T[$task]}
  for op in $OPS; do   # one byte op per job: 5-13 h each on 2 x 24 GB, so four do not fit a 24 h allocation
    j=$(submit "byte-$task-$op" 2 24:00:00 --task $task --level byte --ops $op --model-1 $LLAMA --model-2 $QWEN --template-1 $t --template-2 $t --n 100)
    echo "byte/$task/$op  job $j"
  done
done | tee -a "$OUT/jobs.txt"
