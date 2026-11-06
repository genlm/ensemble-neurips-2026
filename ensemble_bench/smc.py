"""The SMC runner: an f-ensemble of two prompted LMs, sampled with SMC, at token or byte level.

    python -m ensemble_bench.smc --task bbh --level token --model-1 meta-llama/Llama-3.1-8B-Instruct
    python -m ensemble_bench.smc --task spider --level byte \
        --model-1 meta-llama/Llama-3.1-8B-Instruct --model-2 Qwen/Qwen2.5-7B-Instruct --template-2 mid_1

Writes one ``*-smc.jsonl`` per operator (one line per example, with all particles) and one
``*-metrics.jsonl`` per run (bootstrap CI of the expected accuracy per operator).
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import transformers
from genlm.control import Ensemble, direct_token_sampler
from genlm.control.constant import EndOfSequence

from ensemble_bench import models
from ensemble_bench.baselines import BASELINES
from ensemble_bench.experts import k_expert_ensemble, expert_logws
from ensemble_bench.tasks import NAMES, get_task_class
from ensemble_bench.util import bootstrap_ci, resolve_op, safe_decode, set_seed, short

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%m/%d %H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
logger = logging.getLogger("ensemble_bench.smc")

OPS = ["min", "prod", "sum", "max"]


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=NAMES)
    p.add_argument("--level", default="token", choices=["token", "byte"])
    p.add_argument("--ops", nargs="+", default=OPS,
                   help="min prod sum max harmonic, p<k> / p-<k> for power means; unite gac gac_key for the baselines")
    p.add_argument("--model-1", required=True)
    p.add_argument("--model-2", default=None, help="defaults to --model-1 (within-model ensembling)")
    p.add_argument("--template-1", default=None, help="prompt template for model 1 (default: the paper's)")
    p.add_argument("--template-2", default=None, help="prompt template for model 2 (default: the paper's)")
    p.add_argument("--templates", nargs="+", default=None,
                   help="K >= 3 prompt templates of --model-1 for the K-expert ablation (token level, min / prod)")
    p.add_argument("--particles", type=int, default=10)
    p.add_argument("--ess", type=float, default=0.9, help="ESS threshold")
    p.add_argument("--weight", type=float, default=0.5, help="weight a of model 1 (1-a for model 2)")
    p.add_argument("--seed", type=int, default=42, help="generation seed")
    p.add_argument("--base-seed", type=int, default=42, help="data / few-shot selection seed")
    p.add_argument("--n", type=int, default=100, help="number of examples")
    p.add_argument("--shard", default=None, help="i/k: run only the i-th of k contiguous slices of the examples "
                   "(example indices and per-example seeds stay those of the full run)")
    p.add_argument("--max-tokens", type=int, default=None, help="generation budget (default: the task's)")
    p.add_argument("--max-model-len", type=int, default=None, help="vLLM context (default: the task's)")
    p.add_argument("--temperature", type=float, default=None, help="default: the task's (1.0; 0.7 for gsm8k)")
    p.add_argument("--backend", default=None, choices=["vllm", "hf"], help="token level only; default vllm, hf for cross-model")
    p.add_argument("--gpu-util", type=float, default=0.95)
    p.add_argument("--tensor-parallel", type=int, default=1, help="vLLM tensor parallelism (2 for phi-4 on 24 GB cards)")
    p.add_argument("--beam-k", type=int, default=models.BEAM_K)
    p.add_argument("--prune", type=float, default=models.PRUNE_THRESHOLD)
    p.add_argument("--byte-batch", type=int, default=20, help="HF backend auto-batching cap (queries per forward pass)")
    p.add_argument("--byte-cache", type=int, default=models.BEAM_CACHE, help="beams kept per ByteLLM (host memory; results unaffected)")
    p.add_argument("--no-decompose", action="store_true",
                   help="skip the per-model log weights of each particle (diagnostic only; re-scoring long "
                        "generations under each model dominates the cost of phi-4 GSM8K on the HF backend)")
    p.add_argument("--clear-every", type=int, default=4,
                   help="HF backend: clear its per-prefix memo every N examples (it grows ~1 GB per byte-level example; clearing re-scores the prompt)")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--log-every", type=int, default=10)
    # task-specific
    p.add_argument("--spider-dir", default=None, help="Spider release directory (or $SPIDER_DIR)")
    p.add_argument("--json-split", default="val")
    p.add_argument("--json-subtasks", nargs="+", default=["Glaiveai2K"])
    p.add_argument("--gsm8k-split", default="test")
    p.add_argument("--bbh-few-shot", type=int, default=0)
    return p


def make_task(args):
    cls = get_task_class(args.task)
    kw = {}
    if args.task == "spider":
        kw["spider_dir"] = args.spider_dir
    elif args.task == "json_schema":
        kw.update(subtasks=tuple(args.json_subtasks), split=args.json_split)
    elif args.task == "gsm8k":
        kw["split"] = args.gsm8k_split
    elif args.task == "bbh":
        kw["few_shot"] = args.bbh_few_shot
    return cls(n_query=args.n, base_seed=args.base_seed, **kw)


def decode(context, level):
    toks = [t for t in context if not isinstance(t, EndOfSequence)]
    if level == "token":
        return "".join(safe_decode(t) for t in toks)  # per-token, as the paper's runners did
    return b"".join(toks).decode("utf-8", errors="ignore")


async def component_logws(ensemble, ctx, retries=2):
    """Per-model prefix log-weights of a finished particle (K entries), or ``None`` if the backend fails.

    ``Ensemble.component_logws`` scores the whole string under each model, which for a
    long generation means one engine row per prefix position; the vLLM backend has been
    seen to stall on such batches ("engine made no progress on pending rows"). The
    decomposition is diagnostic only -- the SMC result and the expected accuracy do not
    depend on it -- so a failure here must not discard the example.
    """
    for attempt in range(retries + 1):
        try:
            if isinstance(ensemble, Ensemble) and isinstance(ensemble.p1, Ensemble):
                return [float(w) for w in await expert_logws(ensemble, ctx)]
            w1, w2 = await ensemble.component_logws(ctx)
            return [float(w1), float(w2)]
        except (RuntimeError, ValueError) as e:
            if attempt == retries:
                logger.warning(f"component_logws failed after {retries + 1} attempts: {type(e).__name__}: {str(e)[:120]}")
                return None
            await asyncio.sleep(1.0)


async def collect_particles(sequences, ensemble, level, max_tokens, decompose=True):
    out = []
    for ctx, lw in zip(sequences.contexts, sequences.log_weights):
        ctx = list(ctx)
        finished = bool(ctx) and isinstance(ctx[-1], EndOfSequence)
        if level == "token" and finished and len(ctx) == max_tokens:
            # The library forces EOS when the budget is hit; the paper's runs counted
            # these as unfinished and excluded them from the expected accuracy.
            finished, ctx = False, ctx[:-1]
        ws = await component_logws(ensemble, ctx) if decompose else None
        rec = dict(text=decode(ctx, level), log_weight=float(lw), log_weight_1=ws and ws[0],
                   log_weight_2=ws and ws[1], finished=finished)
        if ws and len(ws) > 2:
            rec["log_weights"] = ws
        out.append(rec)
    return out


def choose_and_score(task, item, particles, example_seed):
    """Posterior-sample one finished particle for ``pred``; expected accuracy over all finished ones."""
    finished = [p for p in particles if p["finished"]]
    chosen = particles[0] if particles else None
    if finished:
        lw = np.array([p["log_weight"] for p in finished])
        if np.isfinite(lw.max()):
            w = np.exp(lw - lw.max())
            chosen = finished[np.random.default_rng(example_seed).choice(len(finished), p=w / w.sum())]
        else:
            chosen = finished[0]
    if chosen is None:
        return None, False, None, 0.0
    is_correct, info = task.score(item, chosen["text"])

    groups = defaultdict(float)
    for p in finished:
        groups[p["text"]] += float(np.exp(p["log_weight"]))
    total = sum(groups.values())
    expected = 0.0
    if total > 0:
        expected = sum(wt * float(task.score(item, text)[0]) for text, wt in groups.items()) / total
    return chosen, bool(is_correct), info, float(expected)


async def run(args):
    args.model_2 = args.model_2 or args.model_1
    task = make_task(args)
    templates = args.templates or [args.template_1 or task.templates[0], args.template_2 or task.templates[1]]
    task.score_template = templates[0]   # MMLU: letter or text scoring follows the (shared) output format
    t1, t2, K = templates[0], templates[1], len(templates)
    if K > 2 and (args.model_2 != args.model_1 or args.level != "token"):
        raise ValueError("K >= 3 experts: within-model, token level only")
    max_tokens = args.max_tokens or task.max_tokens[args.level]
    temperature = args.temperature if args.temperature is not None else task.temperature
    cross = args.model_1 != args.model_2
    logger.info(f"task={args.task} level={args.level} ops={args.ops} models=({args.model_1}, {args.model_2}) "
                f"templates={tuple(templates)} M={args.particles} ess={args.ess} a={args.weight} seed={args.seed} "
                f"n={len(task.items)} max_tokens={max_tokens} T={temperature}")

    tok1 = transformers.AutoTokenizer.from_pretrained(args.model_1)
    tok2 = tok1 if not cross else transformers.AutoTokenizer.from_pretrained(args.model_2)
    if args.level == "token":
        backend = args.backend or ("hf" if cross else "vllm")
        llm1, llm2 = models.load_token_llms(args.model_1, args.model_2, backend=backend,
                                            max_model_len=args.max_model_len or task.max_model_len,
                                            gpu_memory_utilization=args.gpu_util, temperature=temperature,
                                            tensor_parallel=args.tensor_parallel)
        pots = [llm1, llm2] + [llm1.spawn() for _ in range(K - 2)]
        bos = [None] * K
    else:
        b1, b2, raw1, raw2 = models.load_byte_llms(args.model_1, args.model_2, beam_k=args.beam_k, prune_threshold=args.prune,
                                                   batch_size=args.byte_batch, cache_size=args.byte_cache)
        pots = [b1, b2]
        bos = [models.QWEN_BOS if raw1.tokenizer.bos_token == models.QWEN_BOS else None,
               models.QWEN_BOS if raw2.tokenizer.bos_token == models.QWEN_BOS else None]
    logger.info("models loaded")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%d%H%M%S%f")

    shard_i, shard_k = map(int, args.shard.split("/")) if args.shard else (0, 1)
    lo = len(task.items) * shard_i // shard_k
    hi = len(task.items) * (shard_i + 1) // shard_k
    items = list(enumerate(task.items))[lo:hi]

    def stem(op):
        return (f"{ts}-{args.task}-{short(args.model_1)}-{short(args.model_2)}-{args.level}-p{args.particles}"
                f"-b{args.ess}-{len(task.items)}-{op}-{'K%d-' % K if K > 2 else ''}{'_'.join(templates)}-{args.weight}-seed{args.seed}"
                f"{'-shard%dof%d' % (shard_i, shard_k) if args.shard else ''}")

    metrics_path = out_dir / f"{stem('all')}-metrics.jsonl"
    metrics = open(metrics_path, "w")
    print(json.dumps({"args": vars(args), "templates": templates, "max_tokens": max_tokens, "temperature": temperature}),
          file=metrics, flush=True)

    for op in args.ops:
        out_path = out_dir / f"{stem(op)}-smc.jsonl"
        logger.info(f"[{op}] writing {out_path.name}")
        expected_all, n_correct, n_err, t_op = [], 0, 0, time.time()
        with open(out_path, "w") as out:
            for done, (i, item) in enumerate(items):
                for k, (pot, t) in enumerate(zip(pots, templates)):
                    prompt = task.prompt(item, t, tok2 if k == 1 else tok1, slot=min(k + 1, 2))
                    pot.set_prompt_from_str(prompt if args.level == "token" else models.strip_bos(prompt, bos[k]))

                example_seed = args.seed + i
                set_seed(example_seed)
                t0, error = time.time(), None
                try:
                    if op in BASELINES:
                        ensemble = BASELINES[op](pots[0], pots[1], args.weight)
                    elif K == 2:
                        ensemble = Ensemble(pots[0], pots[1], resolve_op(op), args.weight)
                    else:
                        ensemble = k_expert_ensemble(pots, op)
                    seqs = await direct_token_sampler(ensemble).smc(
                        n_particles=args.particles, ess_threshold=args.ess, max_tokens=max_tokens)
                    particles = await collect_particles(seqs, ensemble, args.level, max_tokens, decompose=not args.no_decompose)
                except (RuntimeError, ValueError, KeyError) as e:
                    error = f"{type(e).__name__}: {str(e)[:300]}"
                    logger.error(f"[{op}] example {i}: {error}", exc_info=True)
                    models.release_gpu_after_error(e)
                    particles, n_err = [], n_err + 1

                if (done + 1) % args.clear_every == 0:  # HF backend: drop the per-prefix memo (host memory)
                    if args.level == "byte":
                        models.clear_backend_caches(raw1, raw2)
                    elif backend == "hf":
                        models.clear_backend_caches(*(p.model for p in pots))
                chosen, is_correct, info, expected = choose_and_score(task, item, particles, example_seed)
                rec = dict(task=args.task, level=args.level, op=op, seed=args.seed, example_seed=example_seed,
                           example_idx=i, **task.record(item),
                           pred=chosen["text"] if chosen else "", is_correct=is_correct, result=info,
                           finished=bool(chosen["finished"]) if chosen else False, weighting_param=args.weight,
                           particles=particles, expected_accuracy=expected, elapsed_s=round(time.time() - t0, 2),
                           n_particles_without_decomposition=sum(1 for p in particles if p["log_weight_1"] is None),
                           error=error)
                print(json.dumps(rec), file=out, flush=True)
                expected_all.append(expected)
                n_correct += int(is_correct)
                if (done + 1) % args.log_every == 0 or done + 1 == len(items):
                    logger.info(f"[{op}] {done + 1}/{len(items)}  correct={n_correct}  "
                                f"mean_exp_acc={np.mean(expected_all):.4f}  errors={n_err}  "
                                f"{(time.time() - t_op) / (done + 1):.1f}s/ex")

        mean, lo, hi = bootstrap_ci(expected_all, seed=args.seed)
        print(json.dumps({"model": "ensemble", "level": args.level, "method": op, "seed": args.seed, "n": len(expected_all),
                          "accuracy": mean, "ci_95": {"lower": lo, "upper": hi}, "n_correct": n_correct,
                          "n_errors": n_err, "elapsed_s": round(time.time() - t_op, 1)}), file=metrics, flush=True)
        logger.info(f"[{op}] seed {args.seed} - correct: {n_correct}/{len(expected_all)}, "
                    f"mean_acc: {mean:.4f} [{lo:.4f}, {hi:.4f}], errors: {n_err}")
    metrics.close()
    logger.info(f"done; metrics in {metrics_path}")


def main():
    asyncio.run(run(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
