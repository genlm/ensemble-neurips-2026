#!/usr/bin/env python3
"""Paired per-example bootstrap: token SMC vs the local ensemble (M = 1) and vs the best single prompt.

    python scripts/paired.py results/full [-o results/full/paired.md] [--boot 10000]

Every method is evaluated on the same examples with the same seeds, so the comparison is paired:
for each example the per-seed scores are averaged over the seeds all methods share, the
difference SMC - other is taken per example, and examples are resampled (percentile bootstrap)
for a 95 % CI of the mean difference. Done for both metrics: expected accuracy and accuracy.
"Best prompt" is the base prompt with the higher mean expected accuracy on these same examples
(so the comparison slightly favours the baseline). Examples are matched by input text, question, or JSON instance id.
"""

import argparse
import glob
import json
import os
import statistics as st

import numpy as np
from collections import defaultdict

from aggregate import BASE, MODELS, OPS, PROMPTS, SMC, TASKS, load


def ex_id(r):
    """Task-independent example identity: the input text (BBH), question (Spider, GSM8K, MMLU) or instance id (JSON)."""
    for k in ("instance_id", "input", "question"):
        if r.get(k) is not None:
            return str(r[k]) + ("|" + r["db_name"] if "db_name" in r else "")
    raise KeyError("record has no example identity")


def collect(root):
    """(kind, ...) -> seed -> input -> (expected, accuracy); SMC shards merged, clean reruns preferred."""
    out = defaultdict(lambda: defaultdict(dict))
    errored = defaultdict(lambda: defaultdict(set))
    for path in glob.glob(f"{root}/**/*.jsonl", recursive=True):
        if "/stale/" in path:
            continue
        name = os.path.basename(path)
        if m := SMC.match(name):
            d = m.groupdict()
            if d["level"] != "token" or d["m1"] != d["m2"] or int(d["K"] or 2) != 2 or float(d["ess"]) != 0.9:
                continue
            key = ("smc", d["task"], MODELS[d["m1"]], int(d["M"]), d["op"])
            seed = int(d["seed"])
            for r in load(path):
                x = ex_id(r)
                if x in out[key][seed] and not (x in errored[key][seed] and not r.get("error")):
                    continue
                exp = r.get("expected_accuracy")
                out[key][seed][x] = (float(exp) if exp is not None else 0.0, float(bool(r["is_correct"])))
                (errored[key][seed].add if r.get("error") else errored[key][seed].discard)(x)
        elif m := BASE.match(name):
            d = m.groupdict()
            key = ("base", d["task"], MODELS[d["model"]], d["template"])
            for r in load(path):
                out[key][int(d["seed"])][ex_id(r)] = (float(r["expected_accuracy"]), float(bool(r["accuracy"])))
    return out


def per_example(a, b):
    """Two seed -> input -> scores maps -> lists of per-example seed-averaged scores on shared (seed, input)."""
    seeds = sorted(set(a) & set(b))
    xs = sorted(set.intersection(*[set(a[s]) & set(b[s]) for s in seeds])) if seeds else []
    av = [[st.fmean(a[s][x][j] for s in seeds) for x in xs] for j in (0, 1)]
    bv = [[st.fmean(b[s][x][j] for s in seeds) for x in xs] for j in (0, 1)]
    return seeds, xs, av, bv


def boot(d, n, rng):
    d = np.asarray(d)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(m, lo, hi):
    s = f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"**{s}**" if lo > 0 else s   # CI excludes 0 in SMC's favour


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    data = collect(args.root)
    rng = np.random.default_rng(0)
    out = [f"# Paired per-example bootstrap: `{args.root}`\n",
           "Mean difference SMC (M = 10) - other in percentage points, 95 % percentile-bootstrap CI over examples "
           "(per-example scores averaged over the shared seeds). Bold: CI excludes 0 in SMC's favour. "
           "Best prompt = base prompt with the higher mean expected accuracy on the same examples.\n",
           "| task | model | op | seeds | n | SMC exp. acc | vs local: exp. acc | vs local: acc | vs best prompt | vs best prompt: exp. acc | vs best prompt: acc |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for model in MODELS.values():
            bases = {t: data[("base", task, model, t)] for t in PROMPTS[task] if ("base", task, model, t) in data}
            for op in OPS:
                smc, loc = data.get(("smc", task, model, 10, op)), data.get(("smc", task, model, 1, op))
                if not smc:
                    continue
                row = [task, model, op]
                cells = []
                if loc:
                    seeds, xs, a, b = per_example(smc, loc)
                    row += [str(len(seeds)), str(len(xs)), f"{100 * st.fmean(a[0]):.1f}"]
                    cells += [fmt(*boot([p - q for p, q in zip(a[j], b[j])], args.boot, rng)) for j in (0, 1)]
                else:   # no local run (B covers BBH, Spider, JSON): seeds / n from the base comparison below
                    row += [None, None, None]
                    cells += ["--", "--"]
                best = None
                for t, bd in bases.items():
                    _, xs_, a_, b_ = per_example(smc, bd)
                    if xs_ and (best is None or st.fmean(b_[0]) > best[0]):
                        best = (st.fmean(b_[0]), t, a_, b_)
                if best:
                    _, t, a, b = best
                    if row[3] is None:
                        seeds = sorted(set(smc) & set(bases[t]))
                        row[3:6] = [str(len(seeds)), str(len(a[0])), f"{100 * st.fmean(a[0]):.1f}"]
                    cells += [t] + [fmt(*boot([p - q for p, q in zip(a[j], b[j])], args.boot, rng)) for j in (0, 1)]
                else:
                    cells += ["--", "--", "--"]
                row = [c if c is not None else "" for c in row]
                out.append("| " + " | ".join(row + cells) + " |")
    text = "\n".join(out) + "\n"
    dst = args.out or os.path.join(args.root, "paired.md")
    open(dst, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
