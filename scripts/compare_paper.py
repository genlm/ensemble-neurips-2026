#!/usr/bin/env python3
"""Reproduced vs paper: reads aggregate.py's CSV (per-seed means) and scripts/paper_reference.json.

    python scripts/compare_paper.py results/full/tables.csv

Prints, per (table, task, model, op), the reproduced mean expected accuracy (n seeds), the paper
value and the difference, flagging |diff| > 3 pp. Byte cross-model cells are listed with all three
printed paper columns, since the PDF text does not label which pair each column is.
"""
import csv, json, os, sys, statistics as st
from collections import defaultdict

ref = json.load(open(os.path.join(os.path.dirname(__file__), "paper_reference.json")))
OPS = ref["_ops"]
rows = list(csv.DictReader(open(sys.argv[1])))
cell = defaultdict(list)
for r in rows:
    try:
        v = float(r["expected"])
    except ValueError:
        continue
    if int(r["n"]) < 100:   # a seed counts only once all 100 examples are in (shards merged)
        continue
    if r["kind"] == "base":
        cell[("base", r["task"], r["model_1"], r["model_2"])].append(v)   # model_2 column holds the template
    else:
        if r["K"] not in ("2", "") or r["ess"] != "0.9":
            continue
        cell[(r["level"], r["task"], r["model_1"], r["model_2"], r["M"], r["op"])].append(v)

def m(vals):
    return (100 * st.fmean(vals), len(vals)) if vals else (None, 0)

PROMPTS = {"bbh": ["instructional", "example_based"], "spider": ["bad_1", "mid_1"], "json_schema": ["default", "technical"],
           "gsm8k": ["few_shot_cot", "precision_cot"], "mmlu": ["template1_text_with_answers", "template2_text_with_answers"]}
out, flagged, n_cmp = [], 0, 0
def line(name, ours, n, paper):
    global flagged, n_cmp
    if ours is None:
        out.append(f"{name:44s} {'--':>8s}        paper {paper:5.1f}"); return
    d = ours - paper; n_cmp += 1; flag = "  <-- |diff| > 3" if abs(d) > 3 else ""; flagged += bool(flag)
    out.append(f"{name:44s} {ours:6.1f} ({n})  paper {paper:5.1f}  diff {d:+5.1f}{flag}")

for task, models in ref["base"].items():
    for model, vals in models.items():
        for k, p in enumerate(vals):
            ours, n = m(cell[("base", task, model, PROMPTS[task][k])])
            line(f"base  {task:11s} {model:5s} prompt{k + 1}", ours, n, p)
for table, M in (("token_smc", "10"), ("token_local", "1")):
    for task, models in ref[table].items():
        for model, vals in models.items():
            for op, p in zip(OPS, vals):
                ours, n = m(cell[("token", task, model, model, M, op)])
                line(f"{table:11s} {task:11s} {model:5s} {op:4s}", ours, n, p)
out.append("")
for task, per_op in ref["byte_cross_raw"].items():
    if task.startswith("_"):
        continue
    for op, cols in zip(OPS, per_op):
        # column order Q x P, L x P, L x Q: identified from the BBH and Spider cells, where every
        # reproduced pair matches exactly one printed column within ~2 pp
        for (a, b), p in zip((("Qwen", "Phi"), ("Llama", "Phi"), ("Llama", "Qwen")), cols):
            ours, n = m(cell[("byte", task, a, b, "10", op)])
            line(f"byte_cross  {task:11s} {a[0]}x{b[0]}  {op:4s}", ours, n, p)
print("\n".join(out))
print(f"\n{n_cmp} cells compared, {flagged} differ by more than 3 pp")
