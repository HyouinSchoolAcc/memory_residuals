#!/usr/bin/env python3
"""Mechanism comparison figure for Paper A.

Two side-by-side panels showing how memres and RAG differ in *what kind of
information* they encode:

  Left panel  — Shuffle confound (Δ_dsh = ce_shuffle - ce_method).
                A small / negative value means substituting in another
                chain's memory (or another chain's retrieved sessions)
                does NOT degrade the prediction. RAG bars are large
                positive (chain-specific surface evidence). Memres bars
                are statistically zero (chain-conditional context).

  Right panel — Evidence lift (memres only).
                evidence_lift = ce_mem_floor - ce_mem, where the floor
                rebuilds M_c with gold evidence sessions redacted. ≈0
                across every cell shows that the memory benefit is
                *not* literal evidence recall.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "results" / "eval_v25_seed_pack_evpos"
RAG_DIR = ROOT / "results" / "rag_baseline"
NUMS_PATH = ROOT / "paper" / "figures" / "p_a_numbers.json"
OUT_PATH = ROOT / "paper" / "figures" / "p_a_mechanism.pdf"

COLOR_MEMRES = "#d97706"
COLOR_RAG    = "#94a3b8"
COLOR_ORACLE = "#475569"


def main() -> None:
    nums = json.loads(NUMS_PATH.read_text())

    # === Left panel data: Δ_shuffle by method ===
    # RAG shuffle = pa_cb_dsh from each cell (positive = chain-specific evidence)
    # Memres shuffle = pa_cb_dsh from seed-mean (close to zero = chain-conditional)
    rag_06 = nums.get("rag_0p6b", {})
    rag_17 = nums.get("rag_1p7b", {})
    mem_06 = nums.get("memres_0p6b_v27b", {})
    mem_17 = nums.get("memres_1p7b_v28", {})

    rows: list[tuple[str, str, float, str]] = []  # (size_label, method, dsh, color)
    cells = [
        ("0.6B", "BM25\ntop-3",   rag_06.get("bm25_top3",   {}).get("pa_cb_dsh"),  COLOR_RAG),
        ("0.6B", "Dense\ntop-3",  rag_06.get("dense_top3",  {}).get("pa_cb_dsh"),  COLOR_RAG),
        ("0.6B", "MemRes",        mem_06.get("dsh_seed_mean", {}).get("mean"),     COLOR_MEMRES),
        ("1.7B", "BM25\ntop-3",   rag_17.get("bm25_top3",   {}).get("pa_cb_dsh"),  COLOR_RAG),
        ("1.7B", "Dense\ntop-3",  rag_17.get("dense_top3",  {}).get("pa_cb_dsh"),  COLOR_RAG),
        ("1.7B", "MemRes",        mem_17.get("dsh_seed_mean", {}).get("mean"),     COLOR_MEMRES),
    ]
    rows = [(s, m, v, c) for (s, m, v, c) in cells if v is not None and not math.isnan(v)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [3, 2]})

    # === Left panel: Δ_shuffle ===
    ax = axes[0]
    n_cells = len(rows)
    xs = np.arange(n_cells)
    points = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    bars = ax.bar(xs, points, color=colors, edgecolor="black", linewidth=0.7, width=0.7)

    # Annotation: numeric value above each bar
    for i, (sz, lbl, v, c) in enumerate(rows):
        ax.text(i, v + (0.012 if v >= 0 else -0.022), f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[1] for r in rows], fontsize=8)
    # Group separators
    ax.axvline(2.5, color="#cbd5e1", linewidth=0.7, linestyle=":")
    ax.text(1, ax.get_ylim()[1] * 1.02 if ax.get_ylim()[1] > 0 else 0.55, "0.6B",
            ha="center", fontsize=9, color="#1f2937", transform=ax.get_xaxis_transform())
    ax.text(4, ax.get_ylim()[1] * 1.02 if ax.get_ylim()[1] > 0 else 0.55, "1.7B",
            ha="center", fontsize=9, color="#1f2937", transform=ax.get_xaxis_transform())
    ax.set_ylabel(r"$\Delta_{\mathrm{sh}}$ = CE$_{\mathrm{shuffle}}$ − CE$_{\mathrm{method}}$  [nats]", fontsize=9)
    ax.set_title("Chain-shuffle confound", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.05, max(0.55, max(points) * 1.18))

    # === Right panel: evidence_lift (memres only) ===
    ax2 = axes[1]
    el_06 = mem_06.get("evlift_seed_mean", {})
    el_17 = mem_17.get("evlift_seed_mean", {})
    rows2 = []
    if el_06.get("n"):
        rows2.append(("0.6B\n(n=4)", el_06["mean"], el_06.get("std", 0.0)))
    if el_17.get("n"):
        rows2.append((f"1.7B\n(n={el_17['n']})", el_17["mean"], el_17.get("std", 0.0)))

    xs2 = np.arange(len(rows2))
    pts2 = [r[1] for r in rows2]
    errs2 = [r[2] for r in rows2]
    ax2.bar(xs2, pts2, yerr=errs2, color=COLOR_MEMRES, edgecolor="black", linewidth=0.7,
            width=0.55, capsize=5, error_kw={"linewidth": 0.7, "ecolor": "#1f2937"})
    for i, (lbl, v, e) in enumerate(rows2):
        ax2.text(i, v + e + 0.008 if v >= 0 else v - e - 0.018, f"{v:+.3f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([r[0] for r in rows2], fontsize=8.5)
    ax2.set_ylabel(r"$\mathrm{evidence\_lift}$ = CE$_{\mathrm{mem,floor}}$ − CE$_{\mathrm{mem}}$  [nats]", fontsize=9)
    ax2.set_title("Evidence redaction (memres)", fontsize=10)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_ylim(-0.05, 0.05)

    fig.suptitle(
        "RAG localises evidence (large $\\Delta_{\\mathrm{sh}}$); "
        "MemRes encodes chain-conditional context "
        "($\\Delta_{\\mathrm{sh}}\\approx 0$, $\\mathrm{evidence\\_lift}\\approx 0$)",
        fontsize=9.5, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
