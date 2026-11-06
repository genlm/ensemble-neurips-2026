# Environment for the Slurm scripts. Everything site-specific is an environment variable;
# put your values in scripts/env.local.sh (git-ignored) or export them before submitting.
#
#   ENSEMBLE_PYTHON      python interpreter with this package and genlm-control installed   (default: python)
#   SPIDER_DIR           Spider release dir: tables.json, dev.json, train_spider.json, database/  (spider task only)
#   HF_HOME              Hugging Face cache; set HF_HUB_OFFLINE=1 on nodes without internet
#   ENSEMBLE_CACHE_DIR   scratch dir for vLLM / Triton / Inductor compile caches             (default: $TMPDIR or /tmp)
#   SBATCH_EXTRA         extra sbatch flags, e.g. --exclude=<nodes> to avoid pre-Ampere GPUs (bf16 is ~2x slower there)
#   GPU_MEM_SPEC         --gres value for a 24 GB card on your cluster                      (default: gpumem:24g)

ENSEMBLE_BENCH_ROOT="${ENSEMBLE_BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$ENSEMBLE_BENCH_ROOT/scripts/env.local.sh" ] && . "$ENSEMBLE_BENCH_ROOT/scripts/env.local.sh"

export ENSEMBLE_PYTHON="${ENSEMBLE_PYTHON:-python}"
export PYTHONPATH="$ENSEMBLE_BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU_MEM_SPEC="${GPU_MEM_SPEC:-gpumem:24g}"

# per-job compile caches on scratch (never in $HOME, which is small and shared on most clusters)
_cache="${ENSEMBLE_CACHE_DIR:-${TMPDIR:-/tmp}}/ensemble-cache-${SLURM_JOB_ID:-$$}"
export XDG_CACHE_HOME="$_cache" VLLM_CACHE_ROOT="$_cache/vllm" TRITON_CACHE_DIR="$_cache/triton" TORCHINDUCTOR_CACHE_DIR="$_cache/inductor"
mkdir -p "$_cache"

# drop the asyncio/backend chatter that the HF byte backend emits per step
LOG_FILTER='Task was destroyed|^task: <Task|NVML|raw_cnt|import_utils'

job_header() {
  local lib; lib=$($ENSEMBLE_PYTHON -c "import genlm.control,os,subprocess as s;d=os.path.dirname(genlm.control.__file__);print(s.run(['git','-C',d,'log','-1','--format=%h'],capture_output=True,text=True).stdout.strip() or __import__('importlib.metadata').metadata.version('genlm-control'))" 2>/dev/null)
  echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',') harness=$(git -C "$ENSEMBLE_BENCH_ROOT" log -1 --format=%h 2>/dev/null) genlm-control=${lib:-?}"
}
