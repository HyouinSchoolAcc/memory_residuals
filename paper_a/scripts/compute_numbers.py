#!/usr/bin/env python3
"""Compute the headline numbers for Paper A.

Pulls per-cell summary stats from:
  - results/eval_v25_seed_pack_evpos/v27b_no_probe_seed{1..4}_*lme_val_evpos.json
  - results/eval_v25_seed_pack_evpos/v28[a-d]_no_probe_seed{1..4}_*lme_val_evpos.json
  - results/rag_baseline/qwen3_{0p6b,1p7b}_{nomem,bm25_top{1,3},dense_top3,oracle_top3}.json

Writes:
  - paper_a/figures/p_a_numbers.json   (machine-readable summary)
  - paper_a/figures/p_a_headline_table.tex  (LaTeX-ready headline table)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "results" / "eval_v25_seed_pack_evpos"
RAG_DIR = ROOT / "results" / "rag_baseline"
OUT_DIR = ROOT / "paper_a" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _per_chain_dnm(blob: dict) -> list[float]:
    """Extract per-chain pa_cb_dnm = ce_nomem - ce_mem (callback only)."""
    inner = next(iter(blob.values())) if "lme_val" in blob else blob
    pc = inner.get("per_chain", [])
    out = []
    for r in pc:
        ce_m = r.get("ce_mem")
        ce_n = r.get("ce_nomem")
        if ce_m is not None and ce_n is not None and not math.isnan(ce_m) and not math.isnan(ce_n):
            out.append(ce_n - ce_m)
    return out


def _per_chain_drag(blob: dict) -> list[float]:
    """Extract per-chain pa_cb_drag = ce_nomem - ce_rag for RAG cells."""
    pc = blob.get("per_chain", [])
    out = []
    for r in pc:
        ce_r = r.get("ce_rag")
        ce_n = r.get("ce_nomem")
        if ce_r is None or ce_n is None:
            continue
        if isinstance(ce_r, float) and math.isnan(ce_r):
            continue
        if isinstance(ce_n, float) and math.isnan(ce_n):
            continue
        out.append(ce_n - ce_r)
    return out


def _bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boots = np.empty(n_boot, dtype=np.float64)
    n = arr.shape[0]
    for b in range(n_boot):
        boots[b] = rng.choice(arr, size=n, replace=True).mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _summary(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "sem": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "sem": float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    out: dict = {}

    # === Memres seeds, by size ===
    memres_groups = {
        "memres_0p6b_v27b": [
            # seed 1 was the first cell to land and uses the un-suffixed name
            "v27b_no_probe_final_lme_val_evpos.json",
            "v27b_no_probe_seed2_final_lme_val_evpos.json",
            "v27b_no_probe_seed3_final_lme_val_evpos.json",
            "v27b_no_probe_seed4_final_lme_val_evpos.json",
        ],
        "memres_1p7b_v28": [
            "v28a_no_probe_seed1_final_lme_val_evpos.json",
            "v28b_no_probe_seed2_final_lme_val_evpos.json",
            "v28c_no_probe_seed3_final_lme_val_evpos.json",
            "v28d_no_probe_seed4_final_lme_val_evpos.json",
        ],
    }

    for grp, fnames in memres_groups.items():
        seed_means_dnm = []
        seed_means_dsh = []
        seed_means_evlift = []
        all_per_chain_dnm: list[float] = []
        seed_files_present = []
        for fn in fnames:
            p = EVAL_DIR / fn
            blob = _load(p)
            if blob is None:
                continue
            inner = next(iter(blob.values()))
            seed_means_dnm.append(inner["pa_cb_dnm"])
            seed_means_dsh.append(inner["pa_cb_dsh"])
            seed_means_evlift.append(inner.get("pa_cb_evidence_lift", float("nan")))
            seed_files_present.append(fn)
            pc = _per_chain_dnm(blob)
            all_per_chain_dnm.extend(pc)
        out[grp] = {
            "files": seed_files_present,
            "n_seeds": len(seed_means_dnm),
            "dnm_seed_mean": _summary(seed_means_dnm),
            "dsh_seed_mean": _summary(seed_means_dsh),
            "evlift_seed_mean": _summary(seed_means_evlift),
            "n_per_chain_residuals_pooled": len(all_per_chain_dnm),
            "per_chain_pooled_bootstrap_95ci": _bootstrap_ci(all_per_chain_dnm),
            "per_chain_pooled_mean": float(np.mean(all_per_chain_dnm)) if all_per_chain_dnm else float("nan"),
            "per_chain_pooled_median": float(np.median(all_per_chain_dnm)) if all_per_chain_dnm else float("nan"),
            "per_chain_pooled_pos_count": sum(1 for x in all_per_chain_dnm if x > 0),
            "per_chain_pooled_total": len(all_per_chain_dnm),
        }

    # === RAG cells, by size ===
    rag_cells = {
        "rag_0p6b": [
            ("nomem",       "qwen3_0p6b_nomem.json"),
            ("bm25_top1",   "qwen3_0p6b_bm25_top1.json"),
            ("bm25_top3",   "qwen3_0p6b_bm25_top3.json"),
            ("dense_top3",  "qwen3_0p6b_dense_top3.json"),
            ("oracle_top3", "qwen3_0p6b_oracle_top3.json"),
        ],
        "rag_1p7b": [
            ("nomem",       "qwen3_1p7b_nomem.json"),
            ("bm25_top3",   "qwen3_1p7b_bm25_top3.json"),
            ("dense_top3",  "qwen3_1p7b_dense_top3.json"),
            ("oracle_top3", "qwen3_1p7b_oracle_top3.json"),
        ],
    }
    for grp, cells in rag_cells.items():
        sub = {}
        for label, fn in cells:
            blob = _load(RAG_DIR / fn)
            if blob is None:
                continue
            pc_drag = _per_chain_drag(blob)
            ci = _bootstrap_ci(pc_drag) if pc_drag else (float("nan"), float("nan"))
            sub[label] = {
                "ce_rag": blob.get("ce_rag"),
                "ce_nomem": blob.get("ce_nomem"),
                "pa_cb_drag": blob.get("pa_cb_drag"),
                "pa_cb_dsh": blob.get("pa_cb_dsh"),
                "stdev_per_chain_drag": blob.get("stdev_per_chain_drag"),
                "n_chains_per_chain_positive": blob.get("n_chains_per_chain_positive"),
                "n_chains_per_chain": blob.get("n_chains_per_chain"),
                "per_chain_drag_pooled_bootstrap_95ci": ci,
                "n_per_chain": len(pc_drag),
            }
        out[grp] = sub

    out_path = OUT_DIR / "p_a_numbers.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")

    # Pretty summary
    print("\n=== Memres ===")
    for k in ("memres_0p6b_v27b", "memres_1p7b_v28"):
        d = out.get(k, {})
        if not d.get("n_seeds"):
            print(f"{k}: NO SEEDS YET"); continue
        s = d["dnm_seed_mean"]
        ci = d["per_chain_pooled_bootstrap_95ci"]
        pos = d["per_chain_pooled_pos_count"]
        tot = d["per_chain_pooled_total"]
        print(
            f"{k}: n_seeds={d['n_seeds']}  Δ_dnm seed-mean = {s['mean']:+.3f} ± {s['std']:.3f}  "
            f"per-chain pooled CI95 = [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
            f"per-chain positive = {pos}/{tot}"
        )
    print("\n=== RAG ===")
    for grp in ("rag_0p6b", "rag_1p7b"):
        sub = out.get(grp, {})
        for label, info in sub.items():
            ci = info.get("per_chain_drag_pooled_bootstrap_95ci", (float("nan"), float("nan")))
            print(
                f"{grp}.{label:12s}  Δ_drag = {info.get('pa_cb_drag', float('nan')):+.3f}  "
                f"per-chain CI95 = [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
                f"chains_pos = {info.get('n_chains_per_chain_positive')}/{info.get('n_chains_per_chain')}"
            )


if __name__ == "__main__":
    main()
