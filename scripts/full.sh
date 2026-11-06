#!/bin/bash
# Full-reproduction launcher, one block at a time (see README "Reproducing the paper"):
#   scripts/full.sh <block> [seed ...]        seeds default to 42 43 44 45 46
#
#   A  token SMC, within-model: 3 models x 5 tasks x 4 ops                (Tab. 3-5, Fig. 2)
#   B  token local, M = 1: 3 models x {bbh, spider, json} x 4 ops          (Tab. 5 "Token Local")
#   C  base models: 3 models x 5 tasks x 2 prompts, 10 samples/example      (Prompt 1/2 rows, majority voting, Fig. 5)
#   D  byte SMC, cross-model: 3 pairs x {bbh, spider, json} x 4 ops, one job per op   (Tab. 5, Fig. 3a)
#   E  particle study: 3 models x {bbh, spider, json} x M in {2, 5, 25, 50, 75}       (Fig. 4, 6-11, Tab. 6; M = 1 is B, M = 10 is A)
#   F  UniTE / GaC / GaC-key: 3 pairs x {bbh, spider, json}               (Fig. 3a)
#   H  ESS ablation: 3 models x {bbh, spider, json} x ESS in {0.0, 0.3, 0.6}   (0.9 is A)
#   I  K-expert ablation, min: 3 models x {bbh, spider, json} x K in {3, 4, 5, 6}
#
# Resumable: a (cell, op, seed) whose *-smc.jsonl already has N lines is skipped. Jobs loop over
# seeds unless a single seed is long (D, and E with M >= 25). Site settings: scripts/env.sh.
set -euo pipefail
. "$(dirname "$0")/env.sh"
W=$ENSEMBLE_BENCH_ROOT
BLOCK=${1:?block A|B|C|D|E|F|H|I}; shift
SEEDS=${*:-42 43 44 45 46}
OUT=${FULL_OUT:-$W/results/full}/$BLOCK
N=${N_QUERY:-100}
LLAMA=meta-llama/Llama-3.1-8B-Instruct; QWEN=Qwen/Qwen2.5-7B-Instruct; PHI=microsoft/phi-4
MODELS=${MODELS:-"$LLAMA $QWEN $PHI"}                        # override to submit a subset, e.g. MODELS="$LLAMA"
PAIRS=${PAIRS:-"$LLAMA,$QWEN $LLAMA,$PHI $QWEN,$PHI"}
TASKS5=${TASKS:-"bbh spider mmlu json_schema gsm8k"}; TASKS3=${TASKS:-"bbh spider json_schema"}   # TASKS=... to submit a subset
# cross-model prompt: the better single prompt of the pair (Tab. 5 rule), as in scripts/replication.sh
declare -A BYTE_T=( [bbh]=instructional [spider]=mid_1 [json_schema]=default [gsm8k]=precision_cot [mmlu]=template2 )
# K-expert template orders (the paper's nexpert runners)
declare -A KT=( [bbh]="instructional example_based minimal detailed technical persona"
                [spider]="bad_1 bad_2 mid_1 mid_2 good_1 good_2"
                [json_schema]="default technical detailed concise minimal persona" )
mkdir -p "$OUT/logs"
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null) || { echo "squeue failed: not submitting (an empty queue view would resubmit everything)" >&2; exit 1; }   # job names already queued or running
short() { echo "${1##*/}"; }
# phi-4 (14B) does not fit a 24 GB card: by default it runs on one large card (PHI_GPU_MEM, e.g. the
# 96 GB RTX Pro 6000 nodes), where vLLM needs no tensor parallelism. PHI_GPU_MEM=24g falls back to two
# 24 GB cards: vLLM TP = 2 for base runs, the HF backend (sharded) for SMC, whose logits hook does not
# survive TP.
PHI_GPU_MEM=${PHI_GPU_MEM:-96g}
big_phi() { [ "$PHI_GPU_MEM" != 24g ]; }
gpus() { case "$1" in *phi*) big_phi && echo 1 || echo 2;; *) echo 1;; esac; }
gres() { case "$*" in *phi*) big_phi && echo "gpumem:$PHI_GPU_MEM" || echo $GPU_MEM_SPEC;; *) echo $GPU_MEM_SPEC;; esac; }
tp() { case "$1" in *phi*) big_phi || echo "--tensor-parallel 2";; *) echo "";; esac; }
be() { case "$1" in *phi*) big_phi || echo "--backend hf";; *) echo "";; esac; }

submit() {  # name gpus time 'shell command' -- the GPU memory class follows the models named in the job
  local name=$1 g=$2 t=$3 cmd=$4 gr=$(gres "$4")
  # phi on a large card runs at vLLM speed: 4 h jobs are backfilled into the busy 96 GB pool far sooner
  # than 24 h ones, and the seed loops resume where they stopped (not D: a byte cell can exceed 4 h)
  case "$gr" in gpumem:$PHI_GPU_MEM) [ "$BLOCK" != D ] && big_phi && t=${PHI_TIME:-04:00:00};; esac
  # single-GPU seed-loop jobs (E, I) as 4 h jobs: backfilled within hours on a busy cluster, where a
  # 24 h job waits days; the auto-resume resubmits timeouts and finished (op, seed) cells are skipped
  case "$BLOCK" in E|I) [ "$g" = 1 ] && t=${SHORT_TIME:-04:00:00};; esac
  local mem=$([ "$BLOCK" = D ] && echo ${MEM_BYTE:-6g} || echo 8g)
  case "$name" in *phi-4*-sh[0-9]*of*) [ "$BLOCK" != D ] && mem=4g;; esac   # phi token shards peak at ~15 GB host memory   # byte level: ~40 GB peak measured (model load + beams, HF memo cleared every 4 examples)
  # the script goes in on stdin with an explicit bash shebang (--wrap may run under /bin/sh)
  printf '#!/bin/bash\n. %s/scripts/env.sh; cd %s; job_header; %s\n' "$W" "$W" "$cmd" |
  sbatch --parsable -J "$name" -n 1 --cpus-per-task=8 --gpus=$g --gres=$gr --time=$t --mem-per-cpu=$mem \
    ${SBATCH_EXTRA:-} -o "$OUT/logs/$name-%j.out" | sed "s/^/$name  job /"
}
# shell snippet (runs inside the job) that skips a (op, seed) whose output is complete
SKIP='done_ops() { local all=1; for op in $OPS; do m=0; for f in '"$OUT"'/*-$TAG-$op-$SUF*-seed$s-smc.jsonl; do [ -f "$f" ] && [ $(wc -l < "$f") -gt $m ] && m=$(wc -l < "$f"); done; [ $m -ge '"$N"' ] || all=0; done; [ $all = 1 ]; }'
loop() {  # tag ops seeds 'python args' [suffix after op, e.g. K3-] -> command looping over seeds, skipping finished ones
  local tag=$1 ops=$2 seeds=$3 args=$4 suf=${5:-}
  echo "OPS='$ops'; TAG='$tag'; SUF='$suf'; $SKIP; for s in $seeds; do done_ops && { echo \"seed \$s done, skipping\"; continue; }; $ENSEMBLE_PYTHON -m ensemble_bench.smc $args --ops $ops --seed \$s --out-dir $OUT 2>&1 | grep --line-buffered -vE '$LOG_FILTER'; done"
}
OPS4=${OPS:-"min prod sum max"}   # OPS=... to submit a subset

# most lines in any of the given output files (several runs of one cell can coexist: an
# interrupted one and a complete rerun), 0 if none exists
maxlines() { local f m=0 n; for f in "$@"; do [ -f "$f" ] || continue; n=$(wc -l < "$f"); [ $n -gt $m ] && m=$n; done; echo $m; }

covered() {  # tag-op-suffix seed lo hi -> true if complete shard files of any split cover examples [lo, hi)
  local f j k a b
  for f in $(ls $OUT/*-$1*-seed$2-shard*of*-smc.jsonl 2>/dev/null || true); do
    j=$(echo "$f" | sed -E 's/.*-shard([0-9]+)of([0-9]+)-smc.jsonl/\1/'); k=$(echo "$f" | sed -E 's/.*-shard([0-9]+)of([0-9]+)-smc.jsonl/\2/')
    a=$(( N * j / k )); b=$(( N * (j + 1) / k ))
    [ $a -le $3 ] && [ $b -ge $4 ] && [ $(wc -l < "$f") -ge $(( b - a )) ] && return 0
  done
  return 1
}

sharded() {  # name-prefix gpus tag op seed 'python args' -> SHARDS jobs of N/SHARDS examples (4 h queue);
             # skips shards (or the whole cell) already complete, and shards already queued
  local pre=$1 g=$2 tag=$3 op=$4 s=$5 args=$6 suf=${7:-} full f i
  [ "$(maxlines $OUT/*-$tag-$op-$suf*-seed$s-smc.jsonl)" -ge $N ] && return 0
  for i in $(seq 0 $((SHARDS - 1))); do
    [ "$(maxlines $OUT/*-$tag-$op-$suf*-seed$s-shard${i}of$SHARDS-smc.jsonl)" -ge $(( N / SHARDS )) ] && continue
    covered "$tag-$op-$suf" $s $(( N * i / SHARDS )) $(( N * (i + 1) / SHARDS )) && continue   # done under another split
    grep -qx "$pre-$op-s$s-sh${i}of$SHARDS" <<< "$QUEUED" && continue
    submit "$pre-$op-s$s-sh${i}of$SHARDS" $g ${SHARD_TIME:-04:00:00} \
      "$ENSEMBLE_PYTHON -m ensemble_bench.smc $args --ops $op --seed $s --shard $i/$SHARDS --out-dir $OUT 2>&1 | grep --line-buffered -vE '$LOG_FILTER'"
  done
}

case $BLOCK in
A) for m in $MODELS; do for task in $TASKS5; do
     tag="$task-$(short $m)-$(short $m)-token-p10-b0.9-$N"; args="--task $task --level token --model-1 $m $(be $m) --n $N"
     # phi GSM8K on the HF backend: the per-particle decomposition re-scores 1024-token CoTs (diagnostic only)
     case "$m-$task" in *phi*-gsm8k) big_phi || args="$args --no-decompose";; esac
     if [ "${SHARDS:-1}" = 1 ]; then
       submit "A-$(short $m)-$task" $(gpus $m) 24:00:00 "$(loop "$tag" "$OPS4" "$SEEDS" "$args")"
     else   # e.g. phi GSM8K on the HF backend: ~9 min per example, a 100-example cell exceeds any queue
       for op in $OPS4; do for s in $SEEDS; do sharded "A-$(short $m)-$task" $(gpus $m) "$tag" $op $s "$args"; done; done
     fi
   done; done ;;
B) for m in $MODELS; do for task in $TASKS3; do
     submit "B-$(short $m)-$task" $(gpus $m) 12:00:00 "$(loop "$task-$(short $m)-$(short $m)-token-p1-b0.9-$N" "$OPS4" "$SEEDS" "--task $task --level token --model-1 $m $(be $m) --particles 1 --n $N")"
   done; done ;;
C) for m in $MODELS; do for task in $TASKS5; do
     submit "C-$(short $m)-$task" $(gpus $m) 12:00:00 "$ENSEMBLE_PYTHON -m ensemble_bench.base --task $task --model $m $(tp $m) --samples 10 --seeds $SEEDS --n $N --out-dir $OUT 2>&1 | grep --line-buffered -vE '$LOG_FILTER'"
   done; done ;;
D) for pair in $PAIRS; do m1=${pair%,*}; m2=${pair#*,}; g=$(( $(gpus $m1) + $(gpus $m2) )); case $pair in *phi*) big_phi && g=1;; esac; for task in $TASKS3; do t=${BYTE_T[$task]}; for op in $OPS4; do for s in $SEEDS; do
     tag="$task-$(short $m1)-$(short $m2)-byte-p10-b0.9-$N"; args="--task $task --level byte --model-1 $m1 --model-2 $m2 --template-1 $t --template-2 $t --n $N"
     if [ "${SHARDS:-1}" = 1 ]; then
       submit "D-$(short $m1)-$(short $m2)-$task-$op-s$s" $g 24:00:00 "$(loop "$tag" "$op" "$s" "$args")"
     else
       sharded "D-$(short $m1)-$(short $m2)-$task" $g "$tag" $op $s "$args"
     fi
   done; done; done; done ;;
E) for m in $MODELS; do for task in $TASKS3; do
     for M in ${EMS_SMALL-2 5}; do   # EMS_SMALL / EMS_LARGE select particle counts (e.g. EMS_LARGE="" to submit only M = 2, 5)
       if [ "${SHARDS:-1}" != 1 ]; then for op in $OPS4; do for s in $SEEDS; do
         sharded "E-$(short $m)-$task-M$M" $(gpus $m) "$task-$(short $m)-$(short $m)-token-p$M-b0.9-$N" $op $s "--task $task --level token --model-1 $m $(be $m) --particles $M --n $N"
       done; done; continue; fi
       submit "E-$(short $m)-$task-M$M" $(gpus $m) 24:00:00 "$(loop "$task-$(short $m)-$(short $m)-token-p$M-b0.9-$N" "$OPS4" "$SEEDS" "--task $task --level token --model-1 $m $(be $m) --particles $M --n $N")"
     done
     for M in ${EMS_LARGE-25 50 75}; do for s in $SEEDS; do
       tag="$task-$(short $m)-$(short $m)-token-p$M-b0.9-$N"; args="--task $task --level token --model-1 $m $(be $m) --particles $M --n $N"
       if [ "${SHARDS:-1}" = 1 ]; then submit "E-$(short $m)-$task-M$M-s$s" $(gpus $m) 24:00:00 "$(loop "$tag" "$OPS4" "$s" "$args")"
       else for op in $OPS4; do sharded "E-$(short $m)-$task-M$M" $(gpus $m) "$tag" $op $s "$args"; done; fi   # e.g. phi: a 4-op seed at M = 75 exceeds 4 h
     done; done
   done; done ;;
F) for pair in $PAIRS; do m1=${pair%,*}; m2=${pair#*,}; g=$(( $(gpus $m1) + $(gpus $m2) )); case $pair in *phi*) big_phi && g=1;; esac; for task in $TASKS3; do t=${BYTE_T[$task]}
     submit "F-$(short $m1)-$(short $m2)-$task" $g 24:00:00 "$(loop "$task-$(short $m1)-$(short $m2)-token-p10-b0.9-$N" "unite gac gac_key" "$SEEDS" "--task $task --level token --backend hf --model-1 $m1 --model-2 $m2 --template-1 $t --template-2 $t --n $N")"
   done; done ;;
H) for m in $MODELS; do for task in $TASKS3; do for ess in ${ESS:-0.0 0.3 0.6}; do
     tag="$task-$(short $m)-$(short $m)-token-p10-b$ess-$N"; args="--task $task --level token --model-1 $m $(be $m) --ess $ess --n $N"
     if [ "${SHARDS:-1}" = 1 ]; then submit "H-$(short $m)-$task-ess$ess" $(gpus $m) 24:00:00 "$(loop "$tag" "$OPS4" "$SEEDS" "$args")"
     else for op in $OPS4; do for s in $SEEDS; do sharded "H-$(short $m)-$task-ess$ess" $(gpus $m) "$tag" $op $s "$args"; done; done; fi
   done; done; done ;;
I) for m in $MODELS; do for task in $TASKS3; do for K in ${KS:-3 4 5 6}; do
     T=$(echo ${KT[$task]} | cut -d' ' -f1-$K)
     tag="$task-$(short $m)-$(short $m)-token-p10-b0.9-$N"; args="--task $task --level token --model-1 $m $(be $m) --templates $T --n $N"
     if [ "${SHARDS:-1}" = 1 ]; then submit "I-$(short $m)-$task-K$K" $(gpus $m) 24:00:00 "$(loop "$tag" "min" "$SEEDS" "$args" "K$K-")"
     else for s in $SEEDS; do sharded "I-$(short $m)-$task-K$K" $(gpus $m) "$tag" min $s "$args" "K$K-"; done; fi
   done; done; done ;;
*) echo "unknown block $BLOCK" >&2; exit 1 ;;
esac | tee -a "$OUT/jobs.txt"
