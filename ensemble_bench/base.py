"""Base-model runs: one model, one prompt template, n samples per example.

    python -m ensemble_bench.base --task bbh --model meta-llama/Llama-3.1-8B-Instruct --seeds 42 43 44 45 46

Produces the "Prompt 1 / Prompt 2" rows (single-model accuracy), the per-example samples
behind the overlap analysis, and the inputs for majority voting. Prompts are built exactly as
for SMC (same ``tasks/*.py``, same ``tokenizer.encode``), so the base model sees the same
token ids the ensemble components see. Sampling is plain vLLM at the task's temperature.

Per example and template the record holds every sample with its correctness, plus three
summaries: ``accuracy`` (first sample), ``expected_accuracy`` (mean over samples) and
``majority_correct`` (plurality vote over sample texts, evaluated once).
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import transformers

from ensemble_bench.smc import make_task
from ensemble_bench.tasks import NAMES
from ensemble_bench.util import bootstrap_ci, set_seed, short

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%m/%d %H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
logger = logging.getLogger("ensemble_bench.base")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=NAMES)
    p.add_argument("--model", required=True)
    p.add_argument("--templates", nargs="+", default=None, help="prompt templates (default: the paper's pair)")
    p.add_argument("--samples", type=int, default=10, help="samples per example")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--tensor-parallel", type=int, default=1, help="2 for phi-4 on 24 GB cards")
    p.add_argument("--out-dir", default=".")
    # task-specific (same as smc.py)
    p.add_argument("--spider-dir", default=None)
    p.add_argument("--json-split", default="val")
    p.add_argument("--json-subtasks", nargs="+", default=["Glaiveai2K"])
    p.add_argument("--gsm8k-split", default="test")
    p.add_argument("--bbh-few-shot", type=int, default=0)
    return p


def majority(texts):
    """Plurality over exact sample texts; ties resolved by first occurrence."""
    counts = Counter(texts)
    best = max(counts.values())
    return next(t for t in texts if counts[t] == best)


def main():
    from vllm import LLM, SamplingParams

    args = build_arg_parser().parse_args()
    task = make_task(args)
    templates = args.templates or list(task.templates)
    max_tokens = args.max_tokens or task.max_tokens["token"]
    temperature = args.temperature if args.temperature is not None else task.temperature
    logger.info(f"task={args.task} model={args.model} templates={templates} samples={args.samples} seeds={args.seeds} "
                f"n={len(task.items)} max_tokens={max_tokens} T={temperature}")

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    engine_kw = dict(model=args.model, dtype="half", max_model_len=args.max_model_len or task.max_model_len,
                     gpu_memory_utilization=args.gpu_util, tensor_parallel_size=args.tensor_parallel, seed=args.base_seed)
    if "phi" in args.model.lower():
        engine_kw["disable_sliding_window"] = True
    if args.tensor_parallel > 1:
        engine_kw["enforce_eager"] = True
    llm = LLM(**engine_kw)
    logger.info("engine ready")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%d%H%M%S%f")
    for template in templates:
        # the same token ids PromptedLLM.set_prompt_from_str would produce
        prompt_ids = [tok.encode(task.prompt(item, template, tok, slot=1)) for item in task.items]
        for seed in args.seeds:
            set_seed(seed)
            task.score_template = template   # MMLU scores letter or option text depending on the template
            stem = f"{ts}-{args.task}-{short(args.model)}-base-{template}-n{args.samples}-{len(task.items)}-seed{seed}"
            t0 = time.time()
            outs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids],
                                SamplingParams(n=args.samples, temperature=temperature, max_tokens=max_tokens, seed=seed),
                                use_tqdm=False)
            acc, exp, maj = [], [], []
            with open(out_dir / f"{stem}-samples.jsonl", "w") as f:
                for item, out in zip(task.items, outs):
                    samples = []
                    for o in out.outputs:
                        ok, info = task.score(item, o.text)
                        samples.append(dict(text=o.text, correct=bool(ok), result=info, finished=o.finish_reason == "stop",
                                            n_tokens=len(o.token_ids)))
                    voted = majority([s["text"] for s in samples])
                    rec = dict(task=args.task, model=args.model, template=template, seed=seed, **task.record(item),
                               accuracy=samples[0]["correct"], expected_accuracy=float(np.mean([s["correct"] for s in samples])),
                               majority_text=voted, majority_correct=bool(task.score(item, voted)[0]), samples=samples)
                    print(json.dumps(rec), file=f)
                    acc.append(rec["accuracy"]); exp.append(rec["expected_accuracy"]); maj.append(rec["majority_correct"])
            m, lo, hi = bootstrap_ci(exp, seed=seed)
            with open(out_dir / f"{stem}-metrics.jsonl", "w") as f:
                print(json.dumps(dict(model=args.model, template=template, seed=seed, n=len(exp), samples=args.samples,
                                      accuracy=float(np.mean(acc)), expected_accuracy=m, ci_95={"lower": lo, "upper": hi},
                                      majority_accuracy=float(np.mean(maj)), elapsed_s=round(time.time() - t0, 1),
                                      args=vars(args))), file=f)
            logger.info(f"[{template}] seed {seed}: accuracy {np.mean(acc):.3f}  expected {m:.3f} [{lo:.3f},{hi:.3f}]  "
                        f"majority {np.mean(maj):.3f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
