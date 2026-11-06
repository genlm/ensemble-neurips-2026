#!/usr/bin/env python3
"""Aggregate every run under a results tree into the paper's tables (both metrics, mean +- std over seeds).

    python scripts/aggregate.py results/full [-o results/full/tables.md]

Reads the runners' ``*-smc.jsonl`` (SMC, local, baselines, ablations) and ``*-samples.jsonl``
(base models). Every cell reports **expected accuracy** (weight-averaged over finished particles,
or the mean over the base model's samples) and **accuracy** (the posterior-sampled particle, or
the base model's first sample), each as the mean over seeds +- the std over seeds, with the seed
count. Runtime per example (Tab. 2) is the mean of ``elapsed_s``. Also writes a flat CSV of
per-seed means next to the Markdown.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

MODELS = {"Llama-3.1-8B-Instruct": "Llama", "Qwen2.5-7B-Instruct": "Qwen", "phi-4": "Phi"}
_M = "|".join(re.escape(m) for m in MODELS)
SMC = re.compile(rf"^\d+-(?P<task>\w+?)-(?P<m1>{_M})-(?P<m2>{_M})-(?P<level>token|byte)-p(?P<M>\d+)-b(?P<ess>[\d.]+)-(?P<n>\d+)"
                 rf"-(?P<op>[^-]+)-(?:K(?P<K>\d+)-)?(?P<templates>.+)-(?P<a>[\d.]+)-seed(?P<seed>\d+)(?:-shard\d+of\d+)?-smc\.jsonl$")
BASE = re.compile(rf"^\d+-(?P<task>\w+?)-(?P<model>{_M})-base-(?P<template>.+)-n(?P<samples>\d+)-(?P<n>\d+)-seed(?P<seed>\d+)-samples\.jsonl$")
TASKS = ["bbh", "spider", "json_schema", "gsm8k", "mmlu"]
PROMPTS = {"bbh": ["instructional", "example_based"], "spider": ["bad_1", "mid_1"], "json_schema": ["default", "technical"],
           "gsm8k": ["few_shot_cot", "precision_cot"], "mmlu": ["template1_text_with_answers", "template2_text_with_answers"]}
OPS = ["min", "prod", "sum", "max"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def smc_cell(recs):
    exp = [r["expected_accuracy"] for r in recs if r.get("expected_accuracy") is not None]
    return dict(expected=st.fmean(exp) if exp else math.nan, accuracy=st.fmean(bool(r["is_correct"]) for r in recs),
                n=len(recs), errors=sum(1 for r in recs if r.get("error")),
                sec_per_ex=st.fmean(r["elapsed_s"] for r in recs if "elapsed_s" in r) if recs else math.nan)


def base_cell(recs):
    return dict(expected=st.fmean(r["expected_accuracy"] for r in recs), accuracy=st.fmean(bool(r["accuracy"]) for r in recs),
                majority=st.fmean(bool(r["majority_correct"]) for r in recs), n=len(recs), errors=0, sec_per_ex=math.nan)


def collect(root):
    """key -> seed -> per-seed cell. Keys are tuples describing the run."""
    cells = defaultdict(dict)
    smc_recs = defaultdict(dict)   # key -> seed -> {example_idx: record}; merges shards, de-duplicates reruns
    for path in glob.glob(f"{root}/**/*.jsonl", recursive=True):
        if "/stale/" in path:  # outputs of cancelled or superseded jobs, kept for reference
            continue
        name = os.path.basename(path)
        if m := SMC.match(name):
            d = m.groupdict()
            key = ("smc", d["task"], MODELS[d["m1"]], MODELS[d["m2"]], d["level"], int(d["M"]), float(d["ess"]),
                   d["op"], int(d["K"] or 2), d["templates"])
            per_ex = smc_recs[key].setdefault(int(d["seed"]), {})
            for r in load(path):
                i = r.get("example_idx", len(per_ex))
                if i not in per_ex or (per_ex[i].get("error") and not r.get("error")):   # a clean rerun replaces an errored record
                    per_ex[i] = r
        elif m := BASE.match(name):
            d = m.groupdict()
            key = ("base", d["task"], MODELS[d["model"]], d["template"])
            cells[key][int(d["seed"])] = base_cell(load(path))
    for key, seeds in smc_recs.items():
        for seed, per_ex in seeds.items():
            if per_ex:   # files of jobs that are still starting are empty
                cells[key][seed] = smc_cell(list(per_ex.values()))
    return cells


def agg(seeds, field):
    vals = [c[field] for c in seeds.values() if not math.isnan(c[field])]
    if not vals:
        return "--"
    return f"{100 * st.fmean(vals):.1f} ± {100 * (st.stdev(vals) if len(vals) > 1 else 0):.1f} ({len(vals)})"


def fmt(seeds):
    return f"{agg(seeds, 'expected')} / {agg(seeds, 'accuracy')}"


def table(out, title, rows, cols, lookup):
    """rows: [(label, rowkey)], cols: [(label, colkey)], lookup(rowkey, colkey) -> seeds dict or None."""
    out.append(f"\n### {title}\n\nexpected accuracy / accuracy, mean ± std over seeds (n seeds)\n")
    out.append("| | " + " | ".join(c for c, _ in cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for rl, rk in rows:
        vals = [lookup(rk, ck) for _, ck in cols]
        if all(v is None for v in vals):
            continue
        out.append(f"| {rl} | " + " | ".join(fmt(v) if v else "--" for v in vals) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("-o", "--out", default=None, help="Markdown output (default: <root>/tables.md)")
    args = ap.parse_args()
    cells = collect(args.root)
    out = [f"# Aggregated results: `{args.root}`\n"]
    models = [m for m in MODELS.values() if any(k[2] == m for k in cells)]

    def smc(task, m1, m2, level, M, ess, op, K=2):
        hits = [v for k, v in cells.items() if k[0] == "smc" and k[1:9] == (task, m1, m2, level, M, ess, op, K)]
        return hits[0] if hits else None

    def base(task, model, template):
        return cells.get(("base", task, model, template))

    # Tab. 3/4: within-model, per task
    for task in TASKS:
        cols = [("Prompt 1", ("base", 0)), ("Prompt 2", ("base", 1)), ("Majority vote", ("maj", 0))] + \
               [(f"Token SMC {op}", ("smc", op)) for op in OPS]
        found = {k[3] for k in cells if k[0] == "base" and k[1] == task}
        templates = [t for t in PROMPTS[task] if t in found] + sorted(found - set(PROMPTS[task]))  # the paper's Prompt 1 / 2 first

        def lk(model, col, task=task, templates=templates):
            kind, x = col
            if kind == "base":
                return base(task, model, templates[x]) if len(templates) > x else None
            if kind == "maj":
                b = base(task, model, templates[0]) if templates else None
                return {s: dict(c, expected=c["majority"], accuracy=c["majority"]) for s, c in b.items()} if b else None
            return smc(task, model, model, "token", 10, 0.9, x)
        table(out, f"{task}: within-model (prompts {templates})", [(m, m) for m in models], cols, lk)

    # Tab. 5: within-model token SMC / local, cross-model byte SMC and baselines
    pairs = [("Llama", "Qwen"), ("Llama", "Phi"), ("Qwen", "Phi")]
    for task in ["bbh", "spider", "json_schema"]:
        rows = [(f"{m} Token SMC (M=10)", ("w", m, 10)) for m in models] + [(f"{m} Token Local (M=1)", ("w", m, 1)) for m in models] + \
               [(f"{a}×{b} Byte SMC", ("x", a, b)) for a, b in pairs]
        cols = [(op, op) for op in OPS] + [("UniTE", "unite"), ("GaC", "gac"), ("GaC-key", "gac_key")]

        def lk(row, op, task=task):
            if row[0] == "w":
                return smc(task, row[1], row[1], "token", row[2], 0.9, op)
            return smc(task, row[1], row[2], "byte", 10, 0.9, op) or smc(task, row[1], row[2], "token", 10, 0.9, op)
        table(out, f"{task}: Tab. 5 rows and baselines", rows, cols, lk)

    # Particle study, ESS, K
    for task in ["bbh", "spider", "json_schema"]:
        Ms = sorted({k[5] for k in cells if k[0] == "smc" and k[1] == task and k[4] == "token"})
        table(out, f"{task}: particles M (ESS 0.9)", [(f"{m} {op}", (m, op)) for m in models for op in OPS],
              [(f"M={M}", M) for M in Ms], lambda r, M, task=task: smc(task, r[0], r[0], "token", M, 0.9, r[1]))
        esss = sorted({k[6] for k in cells if k[0] == "smc" and k[1] == task and k[5] == 10})
        table(out, f"{task}: ESS threshold (M=10)", [(f"{m} {op}", (m, op)) for m in models for op in OPS],
              [(f"ESS={e}", e) for e in esss], lambda r, e, task=task: smc(task, r[0], r[0], "token", 10, e, r[1]))
        Ks = sorted({k[8] for k in cells if k[0] == "smc" and k[1] == task})
        table(out, f"{task}: K experts (min, M=10)", [(m, m) for m in models], [(f"K={K}", K) for K in Ks],
              lambda m, K, task=task: smc(task, m, m, "token", 10, 0.9, "min", K))

    # Tab. 2: seconds per example
    out.append("\n### Runtime: seconds per example (mean over seeds)\n\n| | " + " | ".join(TASKS) + " |\n|---|" + "---|" * len(TASKS))
    for level in ["token", "byte"]:
        for m1, m2 in [(m, m) for m in models] + pairs:
            row = []
            for task in TASKS:
                c = smc(task, m1, m2, level, 10, 0.9, "min")
                row.append(f"{st.fmean(v['sec_per_ex'] for v in c.values()):.0f}" if c else "--")
            if any(r != "--" for r in row):
                out.append(f"| {level} {m1}×{m2} | " + " | ".join(row) + " |")

    path = args.out or os.path.join(args.root, "tables.md")
    open(path, "w").write("\n".join(out) + "\n")
    with open(path.replace(".md", ".csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["kind", "task", "model_1", "model_2", "level", "M", "ess", "op", "K", "templates", "seed", "expected", "accuracy", "majority", "n", "errors", "sec_per_ex"])
        for key, seeds in sorted(cells.items(), key=str):
            for seed, c in sorted(seeds.items()):
                k = list(key) + [""] * (10 - len(key))
                w.writerow(k + [seed, c["expected"], c["accuracy"], c.get("majority", ""), c["n"], c["errors"], c["sec_per_ex"]])
    print(f"{len(cells)} cells -> {path} and {path.replace('.md', '.csv')}")


if __name__ == "__main__":
    main()
