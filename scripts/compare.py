#!/usr/bin/env python3
"""Compare replication outputs with the validation run (and, where they exist, the paper's files).

    python scripts/compare.py results/replication --ref <dir with the validation run's out/ tree>

For every ``*-smc.jsonl`` under the results dir, prints expected accuracy (all records and
clean records only), n_correct, particle retention, and -- for the (task, level, op) cells the
validation run covered -- the same numbers from that run and the delta.
"""

import argparse
import glob
import json
import math
import os
import re
import statistics as st

NAME = re.compile(r"-(?P<task>bbh|spider|mmlu|json_schema|gsm8k)-.*-(?P<level>token|byte)-p(?P<M>\d+)-b[\d.]+-(?P<n>\d+)-(?P<op>[^-]+)-")


def summarize(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    ea = [r["expected_accuracy"] for r in recs if isinstance(r.get("expected_accuracy"), (int, float)) and math.isfinite(r["expected_accuracy"])]
    clean = [r["expected_accuracy"] for r in recs if len(r.get("particles", [])) == 10 and not r.get("error")]
    n_part = sum(len(r.get("particles", [])) for r in recs)
    return dict(n=len(recs), acc=st.fmean(ea) if ea else float("nan"), clean=st.fmean(clean) if clean else float("nan"),
                n_clean=len(clean), n_correct=sum(1 for r in recs if r.get("is_correct") or (isinstance(r.get("result"), list) and r["result"] and r["result"][0])),
                particles=n_part, errors=sum(1 for r in recs if r.get("error")))


def validation_cells(ref_root):
    """The validation run's files: token under out/token-*/, byte under out/byte-grid/{ds}-{op}/."""
    cells = {}
    for p in glob.glob(f"{ref_root}/out/**/*-smc.jsonl", recursive=True):
        if "stale" in p:
            continue
        base = os.path.basename(p)
        task = next((t for t in ("bbh", "spider", "mmlu") if f"-{t}-" in base or base.startswith(f"{t}-")), None)
        if task is None and ("mid_1" in base or "bad_1" in base):
            task = "spider"
        level = "byte" if "-byte-" in base else "token"
        # the op follows the example count: ...-p10-b0.9-100-sum-...  ("p10" is the particle count, not an op)
        m = re.search(r"-\d+-(min|prod|sum|max|harmonic|p-?\d+(?:\.\d+)?)-", base)
        if not (task and m):
            continue
        key = (task, level, m.group(1))
        if key not in cells or os.path.getmtime(p) > os.path.getmtime(cells[key]):
            cells[key] = p
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ref", default=os.environ.get("ENSEMBLE_REF_DIR"), help="directory holding the reference run (out/**/*-smc.jsonl); or $ENSEMBLE_REF_DIR")
    args = ap.parse_args()
    ref = validation_cells(args.ref) if args.ref else {}
    rows = []
    for p in sorted(glob.glob(f"{args.results}/**/*-smc.jsonl", recursive=True)):
        if "/stale/" in p:  # outputs of cancelled or superseded jobs
            continue
        m = NAME.search(os.path.basename(p))
        if not m:
            continue
        s = summarize(p)
        key = (m["task"], m["level"], m["op"])
        r = summarize(ref[key]) if key in ref else None
        rows.append((key, s, r))
    print("| task | level | op | n | exp_acc | clean | n_corr | particles | errors | validation exp_acc | delta |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (task, level, op), s, r in sorted(rows, key=lambda row: row[0]):
        va = f"{r['acc']*100:.1f}%" if r else "—"
        d = f"{(s['clean'] - r['clean'])*100:+.1f}" if r else "—"
        print(f"| {task} | {level} | {op} | {s['n']} | {s['acc']*100:.1f}% | **{s['clean']*100:.1f}%** | {s['n_correct']} | "
              f"{s['particles']} | {s['errors']} | {va} | {d} |")


if __name__ == "__main__":
    main()
