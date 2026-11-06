#!/usr/bin/env python3
"""Tab. 6 / Fig. 4: correlation between the SMC estimate of the log marginal likelihood, log Z-hat,
and expected accuracy, across particle counts and seeds.

    python scripts/logz_corr.py results/full [-o results/full/logz_corr.md]

log Z-hat of a run on an example = logsumexp(final particle log weights) - log M (the standard SMC
estimate; after resampling every particle carries the average weight, so the formula still holds).
Per example, the points are all (M, seed) runs of a (task, model, op) cell with M in {1, 2, 5, 10, 25,
50, 75} (blocks B, E, A); we report
  * per-sample Spearman r averaged over examples +- standard error, t-test against 0 (Tab. 6), and
  * the pooled Pearson rho of per-example z-scored values (Fig. 4, 9-11).
Examples where either quantity is constant across runs are skipped for the per-sample correlation.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

MODELS = {"Llama-3.1-8B-Instruct": "Llama", "Qwen2.5-7B-Instruct": "Qwen", "phi-4": "Phi"}
_M = "|".join(re.escape(m) for m in MODELS)
SMC = re.compile(rf"^\d+-(?P<task>\w+?)-(?P<m1>{_M})-(?P=m1)-token-p(?P<M>\d+)-b0\.9-100-(?P<op>min|prod|sum|max)-(?P<t>[a-z_0-9]+)-[\d.]+-seed(?P<seed>\d+)(?:-shard\d+of\d+)?-smc\.jsonl$")
PAPER_T = {"bbh": "instructional_example_based", "spider": "bad_1_mid_1", "json_schema": "default_technical"}


def logz(parts):
    lw = np.array([p["log_weight"] for p in parts], dtype=float)
    if len(lw) == 0 or not np.isfinite(lw).any():
        return None
    m = lw[np.isfinite(lw)].max()
    return float(m + math.log(np.exp(lw - m).sum()) - math.log(len(lw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    pts = defaultdict(lambda: defaultdict(dict))   # (task, model, op) -> example -> (M, seed) -> (logz, exp)
    for p in glob.glob(f"{args.root}/[ABE]/*-smc.jsonl"):
        m = SMC.match(os.path.basename(p))
        if not m or m["t"] != PAPER_T.get(m["task"]):
            continue
        key = (m["task"], MODELS[m["m1"]], m["op"])
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") or r.get("expected_accuracy") is None:
                continue
            z = logz(r.get("particles") or [])
            if z is None:
                continue
            pts[key][r["example_idx"]][(int(m["M"]), int(m["seed"]))] = (z, float(r["expected_accuracy"]))
    out = ["# log Z-hat vs expected accuracy (Tab. 6, Fig. 4 analogue)\n",
           "Per-sample Spearman r (mean +- s.e. over examples, one-sample t-test vs 0) | pooled Pearson rho of per-example z-scores | runs per example\n",
           "| task | model | op | per-sample r | pooled rho (z-scored) | examples | runs/example | particle counts |", "|---|---|---|---|---|---|---|---|"]
    star = lambda p: "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
    for key in sorted(pts):
        rs, zx, zy, n_runs, Ms = [], [], [], [], set()
        for ex, runs in pts[key].items():
            v = np.array(list(runs.values()))
            Ms |= {k[0] for k in runs}
            n_runs.append(len(v))
            if len(v) < 3 or v[:, 0].std() == 0 or v[:, 1].std() == 0:
                continue
            rs.append(stats.spearmanr(v[:, 0], v[:, 1]).correlation)
            zx += list((v[:, 0] - v[:, 0].mean()) / v[:, 0].std())
            zy += list((v[:, 1] - v[:, 1].mean()) / v[:, 1].std())
        if not rs:
            continue
        rs = np.array(rs)
        t = stats.ttest_1samp(rs, 0.0)
        rho, prho = stats.pearsonr(zx, zy)
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {rs.mean():+.2f} ± {rs.std(ddof=1)/math.sqrt(len(rs)):.2f}{star(t.pvalue)} | "
                   f"{rho:+.3f}{star(prho)} | {len(rs)} | {np.median(n_runs):.0f} | {','.join(map(str, sorted(Ms)))} |")
    text = "\n".join(out) + "\n"
    open(args.out or os.path.join(args.root, "logz_corr.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
