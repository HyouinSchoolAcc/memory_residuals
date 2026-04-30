#!/usr/bin/env python3
"""Bootstrap confidence intervals for any eval_chain.py output.

Reads ``eval_chain.py`` JSON output (which has ``per_chain`` records with
per-chain ``ce_mem``, ``ce_nomem``, ``ce_shuffle``, ``ce_oracle``,
optionally ``ce_rag``) and produces 95% CIs on every aggregate ? via
non-parametric bootstrap over chains.

Usage:

    python tools/bootstrap_ci.py \
      --input  results/eval/chain_v3_softparity_full_eval.json \
      --output results/eval/chain_v3_softparity_full_ci.json \
      --n_resamples 1000 --seed 42

The output is a JSON with both the original metrics and 2.5%/97.5%
percentiles for every ?. Designed for the paper's results table.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def bootstrap_delta(
    per_chain: list[dict],
    field_a: str,
    field_b: str,
    n_resamples: int,
    rng: random.Random,
) -> dict:
    """Bootstrap CI for (mean_a - mean_b) over chains.

    Skips chains where either field is missing or NaN.
    """
    rows = []
    for r in per_chain:
        a = r.get(field_a, None)
        b = r.get(field_b, None)
        if a is None or b is None:
            continue
        if isinstance(a, float) and a != a:    # NaN
            continue
        if isinstance(b, float) and b != b:
            continue
        rows.append((a, b))
    if not rows:
        return {
            "n": 0, "point": float("nan"),
            "ci_lo_95": float("nan"), "ci_hi_95": float("nan"),
            "ci_lo_99": float("nan"), "ci_hi_99": float("nan"),
            "se": float("nan"),
        }
    a_mean = sum(a for a, _ in rows) / len(rows)
    b_mean = sum(b for _, b in rows) / len(rows)
    point = a_mean - b_mean
    n = len(rows)

    samples: list[float] = []
    for _ in range(n_resamples):
        a_sum = 0.0
        b_sum = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            a_sum += rows[i][0]
            b_sum += rows[i][1]
        samples.append(a_sum / n - b_sum / n)
    samples.sort()
    var = sum((s - point) ** 2 for s in samples) / max(n_resamples - 1, 1)
    return {
        "n": n,
        "point": point,
        "ci_lo_95": percentile(samples, 2.5),
        "ci_hi_95": percentile(samples, 97.5),
        "ci_lo_99": percentile(samples, 0.5),
        "ci_hi_99": percentile(samples, 99.5),
        "se": math.sqrt(var),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="eval_chain.py JSON")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input) as f:
        eval_blob = json.load(f)

    # eval_chain.py packs results per corpus name; for each corpus we have a
    # ``per_chain`` list. Bootstrap each corpus independently.
    rng = random.Random(args.seed)
    out: dict = {"input": args.input, "n_resamples": args.n_resamples,
                 "seed": args.seed, "corpora": {}}

    # Two possible JSON layouts: a flat single-corpus dict, or a dict of corpora.
    if "per_chain" in eval_blob:
        corpora = {"_default": eval_blob}
    elif "corpora" in eval_blob:
        corpora = eval_blob["corpora"]
    else:
        corpora = eval_blob       # assume top-level dict of name -> result

    pairs = [
        ("delta_nm_m",      "ce_nomem",   "ce_mem"),
        ("delta_sh_m",      "ce_shuffle", "ce_mem"),
        ("delta_or_m",      "ce_oracle",  "ce_mem"),
        ("delta_rag_m",     "ce_rag",     "ce_mem"),
    ]

    for cname, cresult in corpora.items():
        per_chain = cresult.get("per_chain", [])
        if not per_chain:
            print(f"[bootstrap] skipping corpus '{cname}' -- no per_chain records")
            continue
        out["corpora"][cname] = {}
        for label, fa, fb in pairs:
            stats = bootstrap_delta(per_chain, fa, fb, args.n_resamples, rng)
            out["corpora"][cname][label] = stats
            print(f"[{cname}] {label:<14}  n={stats['n']:3d}  "
                  f"point={stats['point']:+.4f}  "
                  f"95% CI=[{stats['ci_lo_95']:+.4f}, {stats['ci_hi_95']:+.4f}]  "
                  f"se={stats['se']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[bootstrap] wrote {args.output}")


if __name__ == "__main__":
    main()
