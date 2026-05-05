#!/usr/bin/env python3
"""Headline bar chart for Paper A (memres vs RAG, by model size).

Two panels (0.6B / 1.7B). Bars = pa_cb_drag (RAG) or pa_cb_dnm (memres).
Error bars: 95% bootstrap-over-chains CI for both arms (RAG: per-chain pa_cb_drag;
memres: per-chain pa_cb_dnm pooled across seeds).

Reads from paper/figures/p_a_numbers.json. Writes
paper/figures/p_a_headline.pdf.
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
NUMS_PATH = ROOT / "paper" / "figures" / "p_a_numbers.json"
OUT_PATH = ROOT / "paper" / "figures" / "p_a_headline.pdf"


# Memres = warm orange; RAG cells = cool grey-blue. Oracle = darker (gold standard).
COLOR_MEMRES = "#d97706"
COLOR_RAG    = "#94a3b8"
COLOR_ORACLE = "#475569"


def _ci_to_yerr(point: float, ci: tuple[float, float]) -> tuple[float, float]:
    """Convert (lo, hi) CI endpoints to (down, up) error-bar offsets from point."""
    lo, hi = ci
    return (max(0.0, point - lo), max(0.0, hi - point))


def _bar_panel(ax, size_label: str, nums: dict) -> None:
    """One panel of the figure (one model size)."""
    bars = []  # list of (label, point, yerr_lo, yerr_hi, color, n_pos_str)

    # RAG cells (left side)
    rag_grp = nums.get(f"rag_{size_label}", {})
    cell_order_0p6b = ["bm25_top1", "bm25_top3", "dense_top3", "oracle_top3"]
    cell_order_1p7b = ["bm25_top3", "dense_top3", "oracle_top3"]
    cell_order = cell_order_0p6b if "0p6b" in size_label else cell_order_1p7b

    for cell in cell_order:
        if cell not in rag_grp:
            continue
        info = rag_grp[cell]
        point = info.get("pa_cb_drag")
        if point is None:
            continue
        ci = info.get("per_chain_drag_pooled_bootstrap_95ci", (point, point))
        n_pos = info.get("n_chains_per_chain_positive")
        n_total = 50  # known from corpus
        pretty = {
            "bm25_top1": "BM25\ntop-1",
            "bm25_top3": "BM25\ntop-3",
            "dense_top3": "Dense\ntop-3",
            "oracle_top3": "Oracle\ntop-3",
        }[cell]
        color = COLOR_ORACLE if cell.startswith("oracle") else COLOR_RAG
        yerr = _ci_to_yerr(point, ci)
        bars.append((pretty, point, yerr, color, f"{n_pos}/{n_total}"))

    # Memres (right side) — final entry, prominent
    mem_grp_key = f"memres_{size_label}_v27b" if "0p6b" in size_label else f"memres_{size_label}_v28"
    m = nums.get(mem_grp_key, {})
    if m.get("n_seeds"):
        point = m["dnm_seed_mean"]["mean"]
        ci = m["per_chain_pooled_bootstrap_95ci"]
        n_seeds = m["n_seeds"]
        n_pos_total = m["per_chain_pooled_total"]
        n_pos_count = m["per_chain_pooled_pos_count"]
        # Per-chain count is pooled across all seeds; divide for per-seed mean.
        per_seed_pos = round(n_pos_count / max(1, n_seeds))
        per_seed_total = round(n_pos_total / max(1, n_seeds))
        yerr = _ci_to_yerr(point, ci)
        bars.append((
            f"MemRes\n(n={n_seeds})", point, yerr, COLOR_MEMRES,
            f"{per_seed_pos}/{per_seed_total}",
        ))

    # Plot
    xs = np.arange(len(bars))
    points = [b[1] for b in bars]
    yerr_arr = np.array([[b[2][0] for b in bars], [b[2][1] for b in bars]])
    colors = [b[3] for b in bars]

    bar_objs = ax.bar(xs, points, yerr=yerr_arr, color=colors, edgecolor="black",
                      linewidth=0.8, capsize=4, error_kw={"linewidth": 0.8, "ecolor": "#1f2937"})

    # Per-bar annotation: numeric Δ above bar; chain-positive count below x-axis
    for i, (label, point, yerr, color, frac) in enumerate(bars):
        # Δ value above bar
        annot_y = (point + yerr[1] + 0.06) if point >= 0 else 0.06
        ax.text(i, annot_y, f"{point:+.2f}", ha="center", va="bottom",
                fontsize=8, color="black")
        # n positive chains label
        ax.text(i, -0.15, f"{frac}", ha="center", va="top",
                fontsize=7, color="#475569", style="italic")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel(r"$\Delta_{\mathrm{cb}}$ (nats, callback CE)", fontsize=9)
    ax.set_title(f"Qwen3-{'0.6B' if '0p6b' in size_label else '1.7B'}", fontsize=10)
    ax.set_ylim(-0.4, max(2.1, max(points) + 0.5))
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    nums = json.loads(NUMS_PATH.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=False)
    _bar_panel(axes[0], "0p6b", nums)
    _bar_panel(axes[1], "1p7b", nums)

    # Single shared legend at the bottom
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_RAG,    edgecolor="black", linewidth=0.8, label="RAG (BM25 / dense)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ORACLE, edgecolor="black", linewidth=0.8, label="Oracle RAG (gold evidence)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MEMRES, edgecolor="black", linewidth=0.8, label="Memory Residuals (this work)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=8.5)
    fig.text(0.5, 0.07, "italic numbers below x-axis: per-chain positive count out of 50",
             ha="center", fontsize=7, color="#475569", style="italic")
    fig.suptitle(
        "Memory Residuals beat strong RAG including oracle-evidence on LongMemEval-S callback CE",
        fontsize=10, y=1.00,
    )
    plt.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
