#!/usr/bin/env python3
"""Bootstrap 95% CI over routing-mass traces (Paper B Table 4).

Reads existing JSON dumps from `tools/routing_trace.py` and produces:

  * `mean_alpha_mem_mem`     — mean alpha_mem under the chain's own M_c
  * `mean_alpha_mem_shuffle` — mean under another chain's M_c
  * `gap_pct`                — (mem - shuffle) / shuffle * 100
  * `top_k_sublayers`        — sublayers ranked by mem-vs-shuffle gap
  * `init_floor`             — exp(-8) / N (analytic init floor for ±4 bias)
  * `init_floor_multiple`    — mem_alpha / init_floor

Each is reported with a cluster-bootstrap 95% CI where the cluster is the
chain ID (so CIs respect within-chain correlation across (sublayer, depth)
samples). Uses the per-(chain, depth, condition) `raw_layer_means_sample`
that `routing_trace.py` writes; `--input` may be either a single JSON or
two JSONs (one per memres mode).

Output is JSON to `--output`; when `--output` is `-`, prints to stdout
in human-readable form too.

Why this script, not a re-run of the trace: the original JSONs already
carry the per-(chain, depth, condition) layer means we need to bootstrap
over. Re-running the trace is unnecessary and would consume GPU time we
need for the headline ablations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _flatten_raw(raw: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (chain_id_int, depth, condition_idx, alpha_per_sublayer[n_sublayers])."""
    chain_id_str = []
    depth = []
    condition = []  # 0 = mem, 1 = shuffle
    alphas = []
    for row in raw:
        chain_id_str.append(row["chain_id"])
        depth.append(int(row["depth"]))
        condition.append(0 if row["condition"] == "mem" else 1)
        alphas.append(np.asarray(row["alpha_per_sublayer"], dtype=np.float64))

    if not alphas:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 0), dtype=np.float64),
        )

    n_sublayers = max(a.shape[0] for a in alphas)
    A = np.full((len(alphas), n_sublayers), np.nan, dtype=np.float64)
    for i, a in enumerate(alphas):
        A[i, : a.shape[0]] = a

    chain_uniq = sorted(set(chain_id_str))
    chain_to_idx = {c: i for i, c in enumerate(chain_uniq)}
    cid = np.asarray([chain_to_idx[c] for c in chain_id_str], dtype=np.int64)
    return (
        cid,
        np.asarray(depth, dtype=np.int64),
        np.asarray(condition, dtype=np.int64),
        A,
    )


def _bootstrap_chain(
    cid: np.ndarray,
    cond: np.ndarray,
    A: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Cluster-bootstrap by chain id; report mean alpha (per condition) and gap."""
    n = cid.shape[0]
    if n == 0 or A.size == 0:
        return {
            "mean_mem": float("nan"),
            "mean_shuffle": float("nan"),
            "gap_pct": float("nan"),
            "ci_mem": (float("nan"), float("nan")),
            "ci_shuffle": (float("nan"), float("nan")),
            "ci_gap_pct": (float("nan"), float("nan")),
            "n_chains": 0,
            "n_samples_mem": 0,
            "n_samples_shuffle": 0,
            "n_boot": n_boot,
        }

    chain_ids = np.unique(cid)
    n_chains = chain_ids.shape[0]

    def _means(sample_cid_set: np.ndarray) -> tuple[float, float, float]:
        mask = np.isin(cid, sample_cid_set)
        if not mask.any():
            return float("nan"), float("nan"), float("nan")
        # Per-row mean alpha across sublayers (ignoring NaN slots).
        per_row = np.nanmean(A[mask], axis=1)
        cond_sub = cond[mask]
        mem_vals = per_row[cond_sub == 0]
        sh_vals = per_row[cond_sub == 1]
        if mem_vals.size == 0 or sh_vals.size == 0:
            return float("nan"), float("nan"), float("nan")
        m = float(np.mean(mem_vals))
        s = float(np.mean(sh_vals))
        gap = (m - s) / s * 100.0 if s != 0 else float("nan")
        return m, s, gap

    point_m, point_s, point_g = _means(chain_ids)

    boot_m = np.empty(n_boot, dtype=np.float64)
    boot_s = np.empty(n_boot, dtype=np.float64)
    boot_g = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(chain_ids, size=n_chains, replace=True)
        m, s, g = _means(sample)
        boot_m[b] = m
        boot_s[b] = s
        boot_g[b] = g

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        valid = arr[~np.isnan(arr)]
        if valid.size < 10:
            return float("nan"), float("nan")
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return float(lo), float(hi)

    return {
        "mean_mem": point_m,
        "mean_shuffle": point_s,
        "gap_pct": point_g,
        "ci_mem": _ci(boot_m),
        "ci_shuffle": _ci(boot_s),
        "ci_gap_pct": _ci(boot_g),
        "n_chains": int(n_chains),
        "n_samples_mem": int(((cond == 0)).sum()),
        "n_samples_shuffle": int(((cond == 1)).sum()),
        "n_boot": int(n_boot),
    }


def _per_sublayer_gap(cid: np.ndarray, cond: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Per-sublayer mem - shuffle gap in absolute alpha units."""
    if A.size == 0:
        return np.zeros(0, dtype=np.float64)
    mem_mask = cond == 0
    sh_mask = cond == 1
    if not mem_mask.any() or not sh_mask.any():
        return np.zeros(A.shape[1], dtype=np.float64)
    mem_per_sublayer = np.nanmean(A[mem_mask], axis=0)
    sh_per_sublayer = np.nanmean(A[sh_mask], axis=0)
    return mem_per_sublayer - sh_per_sublayer


def process_one(
    in_path: Path,
    n_boot: int,
    seed: int,
    top_k: int,
) -> dict:
    blob = json.loads(in_path.read_text())
    raw = blob.get("raw_layer_means_sample") or blob.get("raw_layer_means") or []
    cid, depth, cond, A = _flatten_raw(raw)

    rng = np.random.default_rng(seed)
    boot = _bootstrap_chain(cid, cond, A, n_boot, rng)

    init_floor = math.exp(-8) / max(1, blob.get("n_sublayers", A.shape[1] if A.size else 1))
    boot["init_floor"] = init_floor
    boot["init_floor_multiple_mem"] = (
        boot["mean_mem"] / init_floor if init_floor and not math.isnan(boot["mean_mem"]) else float("nan")
    )

    sub_gap = _per_sublayer_gap(cid, cond, A)
    if sub_gap.size > 0:
        order = np.argsort(-sub_gap)
        boot["top_k_sublayers"] = [
            {"sublayer": int(i), "gap_alpha": float(sub_gap[i])}
            for i in order[:top_k]
        ]
        boot["max_sublayer_gap_alpha"] = float(sub_gap.max())
        boot["max_sublayer_alpha_mem"] = float(
            np.nanmean(A[cond == 0], axis=0).max() if (cond == 0).any() else float("nan")
        )
    else:
        boot["top_k_sublayers"] = []
        boot["max_sublayer_gap_alpha"] = float("nan")
        boot["max_sublayer_alpha_mem"] = float("nan")

    return {
        "input": str(in_path),
        "memres_mode": blob.get("memres_mode"),
        "n_chains_corpus": blob.get("n_chains"),
        "n_score_positions_corpus": blob.get("n_score_positions"),
        "n_sublayers": blob.get("n_sublayers"),
        "bootstrap": boot,
    }


def _fmt(v):
    return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:+.3e}" if abs(v) < 1e-2 else f"{v:+.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more routing_*.json files produced by tools/routing_trace.py.",
    )
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--output", default="-")
    a = ap.parse_args()

    results = [process_one(Path(p), a.n_boot, a.seed, a.top_k) for p in a.input]
    out = {"args": vars(a), "results": results}

    if a.output == "-":
        print(json.dumps(out, indent=2))
    else:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(out, indent=2))
        print(f"saved -> {a.output}")

    # Always print a quick human summary to stderr so the run is self-documenting.
    print("\n=== Bootstrap CIs (cluster=chain) ===", file=sys.stderr)
    for r in results:
        b = r["bootstrap"]
        ci_m = b["ci_mem"]
        ci_s = b["ci_shuffle"]
        ci_g = b["ci_gap_pct"]
        print(
            f"\n[{r['memres_mode']}]  {Path(r['input']).name}",
            file=sys.stderr,
        )
        print(
            f"  n_chains={b['n_chains']}  n_mem={b['n_samples_mem']}  "
            f"n_shuffle={b['n_samples_shuffle']}  n_boot={b['n_boot']}",
            file=sys.stderr,
        )
        print(
            f"  mean alpha_mem (mem)     = {_fmt(b['mean_mem'])}  "
            f"95% CI [{_fmt(ci_m[0])}, {_fmt(ci_m[1])}]",
            file=sys.stderr,
        )
        print(
            f"  mean alpha_mem (shuffle) = {_fmt(b['mean_shuffle'])}  "
            f"95% CI [{_fmt(ci_s[0])}, {_fmt(ci_s[1])}]",
            file=sys.stderr,
        )
        print(
            f"  mem-vs-shuffle gap       = {_fmt(b['gap_pct'])} %  "
            f"95% CI [{_fmt(ci_g[0])}, {_fmt(ci_g[1])}] %",
            file=sys.stderr,
        )
        if not math.isnan(b["init_floor_multiple_mem"]):
            print(
                f"  init floor x{b['init_floor_multiple_mem']:.2f}  "
                f"(analytic floor = exp(-8)/N = {b['init_floor']:.2e})",
                file=sys.stderr,
            )
        if b["top_k_sublayers"]:
            tops = ", ".join(
                f"L{x['sublayer']}={_fmt(x['gap_alpha'])}" for x in b["top_k_sublayers"]
            )
            print(f"  top-{a.top_k} sublayer gaps: {tops}", file=sys.stderr)
            print(
                f"  max single-sublayer alpha_mem = {_fmt(b['max_sublayer_alpha_mem'])}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
