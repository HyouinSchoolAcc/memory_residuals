#!/usr/bin/env python3
"""Cross-corpus eval numbers for Paper A.

Reads:
  results/eval_v27_v28_cross_corpus/<tag>_cb.json     (callback-aware on synthd4/d4v2/d5)
  results/eval_v27_v28_cross_corpus/<tag>_fullce.json (eval_chain on locomo + msc_test)

Tags are v27b_seed{1..4} (0.6B) and v28a_seed1, v28b_seed2, v28c_seed3 (1.7B).
v28d_seed4 (1.7B) is added when its tag JSONs land.

Writes paper/xcorpus_numbers.tex with macros consumed by main.tex.
For every (size, corpus) cell we emit:
  - seed-mean Δ_dnm (callback CE for cb corpora; chain-mean CE for fullce corpora)
  - seed std
  - per-corpus pooled-over-seeds 95% bootstrap CI of per-chain Δ
  - per-corpus seed-mean Δ_sh
  - per-corpus seed-mean evidence_lift (cb only)
  - per-chain positive count out of (n_chains * n_seeds)

Macro naming: \\xcS<size><corpus><field>, e.g. \\xcSsixsynthdfvCBdnm = "+0.45 \\pm 0.08".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "eval_v27_v28_cross_corpus"
OUT = ROOT / "paper" / "xcorpus_numbers.tex"

SEEDS = {
    "0p6b": ["v27b_seed1", "v27b_seed2", "v27b_seed3", "v27b_seed4"],
    "1p7b": ["v28a_seed1", "v28b_seed2", "v28c_seed3", "v28d_seed4"],
}
CB_CORPORA = ["synthd4_val", "synthd4v2_val", "synthd5_val"]
FC_CORPORA = ["locomo", "msc_test"]


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def boot_ci(values: list[float], n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).any():
        return float("nan"), float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boots[b] = rng.choice(arr, size=arr.size, replace=True).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def seed_summary(cell_values: list[float]) -> dict:
    arr = np.asarray([v for v in cell_values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max())}


def cb_per_corpus(size: str, corpus: str) -> dict:
    """Aggregate callback eval across seeds for one (size, corpus)."""
    seed_dnm: list[float] = []
    seed_dsh: list[float] = []
    seed_evl: list[float] = []
    pooled_per_chain: list[float] = []
    n_pos = 0
    n_total = 0
    n_seeds_present = 0
    for tag in SEEDS[size]:
        blob = load(DATA / f"{tag}_cb.json")
        if blob is None or corpus not in blob:
            continue
        sub = blob[corpus]
        if "pa_cb_dnm" not in sub:
            continue
        n_seeds_present += 1
        seed_dnm.append(sub.get("pa_cb_dnm", float("nan")))
        seed_dsh.append(sub.get("pa_cb_dsh", float("nan")))
        seed_evl.append(sub.get("pa_cb_evidence_lift", float("nan")))
        for r in sub.get("per_chain", []):
            ce_m, ce_n = r.get("ce_mem"), r.get("ce_nomem")
            if ce_m is None or ce_n is None:
                continue
            d = ce_n - ce_m
            if not math.isnan(d):
                pooled_per_chain.append(d)
                n_total += 1
                if d > 0:
                    n_pos += 1
    return {
        "n_seeds": n_seeds_present,
        "dnm": seed_summary(seed_dnm),
        "dsh": seed_summary(seed_dsh),
        "evl": seed_summary(seed_evl),
        "ci95": boot_ci(pooled_per_chain),
        "pos": n_pos,
        "total": n_total,
    }


def fc_per_corpus(size: str, corpus: str) -> dict:
    """Aggregate full-CE eval across seeds for one (size, corpus)."""
    seed_dnm: list[float] = []
    seed_dsh: list[float] = []
    pooled_per_chain: list[float] = []
    n_pos = 0
    n_total = 0
    n_seeds_present = 0
    for tag in SEEDS[size]:
        blob = load(DATA / f"{tag}_fullce.json")
        if blob is None or corpus not in blob:
            continue
        sub = blob[corpus]
        if "delta_nomem_minus_mem" not in sub:
            continue
        n_seeds_present += 1
        seed_dnm.append(sub.get("delta_nomem_minus_mem", float("nan")))
        seed_dsh.append(sub.get("delta_shuffle_minus_mem", float("nan")))
        for r in sub.get("per_chain", []):
            ce_m, ce_n = r.get("ce_mem"), r.get("ce_nomem")
            if ce_m is None or ce_n is None:
                continue
            d = ce_n - ce_m
            if not math.isnan(d):
                pooled_per_chain.append(d)
                n_total += 1
                if d > 0:
                    n_pos += 1
    return {
        "n_seeds": n_seeds_present,
        "dnm": seed_summary(seed_dnm),
        "dsh": seed_summary(seed_dsh),
        "ci95": boot_ci(pooled_per_chain),
        "pos": n_pos,
        "total": n_total,
    }


def fmt_mean_std(s: dict) -> str:
    # Math-mode body, no $...$ wrapping (user wraps with $...$ at use site).
    if s["n"] == 0:
        return "\\text{(pending)}"
    if s["n"] == 1:
        return f"{s['mean']:+.3f}"
    return f"{s['mean']:+.3f} \\pm {s['std']:.3f}"


def fmt_signed(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "\\text{(pending)}"
    return f"{x:+.3f}"


def fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "\\text{(pending)}"
    return f"[{lo:+.3f},\\,{hi:+.3f}]"


def fmt_pos(d: dict) -> str:
    if d["total"] == 0:
        return "(pending)"
    return f"{d['pos']}/{d['total']}"


def main() -> None:
    lines: list[str] = []
    L = lines.append
    L("% Auto-generated by paper/scripts/compute_xcorpus_numbers.py")
    L("% Source: results/eval_v27_v28_cross_corpus/*.json")
    L("% DO NOT EDIT BY HAND -- re-run the script.")
    L("")

    # Compute everything.
    summary: dict = {}
    for size in SEEDS:
        summary[size] = {"cb": {}, "fc": {}}
        for corpus in CB_CORPORA:
            summary[size]["cb"][corpus] = cb_per_corpus(size, corpus)
        for corpus in FC_CORPORA:
            summary[size]["fc"][corpus] = fc_per_corpus(size, corpus)

    # Emit a count of seeds available so the section can adapt verdict text.
    n06_seeds = max((summary["0p6b"]["cb"][c]["n_seeds"] for c in CB_CORPORA), default=0)
    n17_seeds = max((summary["1p7b"]["cb"][c]["n_seeds"] for c in CB_CORPORA), default=0)
    L(f"\\newcommand{{\\xcSnsix}}{{{n06_seeds}}}")
    L(f"\\newcommand{{\\xcSnseventeen}}{{{n17_seeds}}}")
    L("")

    # Per-cell macros. Naming: \xcS<size><corpus><field>
    # size: Six | Seventeen
    # corpus: Dfour | DfourV | Dfive | Locomo | Msc
    # field: Dnm | Dsh | Evl | Ci | Pos
    SIZE_MAP = {"0p6b": "Six", "1p7b": "Seventeen"}
    CB_MAP = {"synthd4_val": "Dfour", "synthd4v2_val": "DfourV", "synthd5_val": "Dfive"}
    FC_MAP = {"locomo": "Locomo", "msc_test": "Msc"}

    for size, ssz in SIZE_MAP.items():
        for corpus, sc in CB_MAP.items():
            d = summary[size]["cb"][corpus]
            base = f"\\xcS{ssz}{sc}"
            L(f"\\newcommand{{{base}Dnm}}{{{fmt_mean_std(d['dnm'])}}}")
            L(f"\\newcommand{{{base}Dsh}}{{{fmt_signed(d['dsh']['mean'])}}}")
            L(f"\\newcommand{{{base}Evl}}{{{fmt_signed(d['evl']['mean'])}}}")
            L(f"\\newcommand{{{base}Ci}}{{{fmt_ci(d['ci95'])}}}")
            L(f"\\newcommand{{{base}Pos}}{{{fmt_pos(d)}}}")
            L(f"\\newcommand{{{base}Nseeds}}{{{d['n_seeds']}}}")
        for corpus, sc in FC_MAP.items():
            d = summary[size]["fc"][corpus]
            base = f"\\xcS{ssz}{sc}"
            L(f"\\newcommand{{{base}Dnm}}{{{fmt_mean_std(d['dnm'])}}}")
            L(f"\\newcommand{{{base}Dsh}}{{{fmt_signed(d['dsh']['mean'])}}}")
            L(f"\\newcommand{{{base}Ci}}{{{fmt_ci(d['ci95'])}}}")
            L(f"\\newcommand{{{base}Pos}}{{{fmt_pos(d)}}}")
            L(f"\\newcommand{{{base}Nseeds}}{{{d['n_seeds']}}}")
        L("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"saved -> {OUT}")
    print(f"  0.6B seeds present (max across cb corpora): {n06_seeds}/4")
    print(f"  1.7B seeds present (max across cb corpora): {n17_seeds}/4")
    for size, ssz in SIZE_MAP.items():
        print(f"--- {size} ---")
        for corpus in CB_CORPORA + FC_CORPORA:
            kind = "cb" if corpus in CB_MAP else "fc"
            d = summary[size][kind][corpus]
            ds_str = "(no data)"
            if d["n_seeds"]:
                m = d["dnm"]
                ds_str = f"Δ={m['mean']:+.3f}" + (f" ± {m['std']:.3f}" if m["n"] > 1 else "") + f"  n_seeds={d['n_seeds']}  pos={d['pos']}/{d['total']}"
            print(f"  {corpus:18s} {ds_str}")


if __name__ == "__main__":
    main()
