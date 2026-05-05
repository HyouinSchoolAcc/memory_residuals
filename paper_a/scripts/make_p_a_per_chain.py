#!/usr/bin/env python3
"""Per-chain robustness scatter for Paper A.

Plot Δ_callback_memres (per chain, averaged over 4 v27b 0.6B seeds) on the
y-axis vs Δ_callback_oracle_RAG (per chain) on the x-axis. Each dot is one
of the 50 LongMemEval-S validation chains.

Marginal histograms on each axis show:
  - top: oracle-RAG distribution (predominantly near zero / wide tails)
  - right: memres distribution (predominantly positive, narrower)
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
OUT_PATH = ROOT / "paper_a" / "figures" / "p_a_per_chain.pdf"


def _per_chain_dnm(blob: dict) -> dict[str, float]:
    inner = next(iter(blob.values())) if "lme_val" in blob else blob
    out: dict[str, float] = {}
    for r in inner.get("per_chain", []):
        cid = r.get("chain_id")
        ce_m, ce_n = r.get("ce_mem"), r.get("ce_nomem")
        if cid is not None and ce_m is not None and ce_n is not None:
            out[cid] = ce_n - ce_m
    return out


def _per_chain_drag(blob: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in blob.get("per_chain", []):
        cid = r.get("chain_id")
        ce_r, ce_n = r.get("ce_rag"), r.get("ce_nomem")
        if cid is None or ce_r is None or ce_n is None:
            continue
        if isinstance(ce_r, float) and math.isnan(ce_r):
            continue
        if isinstance(ce_n, float) and math.isnan(ce_n):
            continue
        out[cid] = ce_n - ce_r
    return out


def main() -> None:
    # Load 4 memres seeds and average per chain
    memres_seed_files = [
        EVAL_DIR / "v27b_no_probe_final_lme_val_evpos.json",
        EVAL_DIR / "v27b_no_probe_seed2_final_lme_val_evpos.json",
        EVAL_DIR / "v27b_no_probe_seed3_final_lme_val_evpos.json",
        EVAL_DIR / "v27b_no_probe_seed4_final_lme_val_evpos.json",
    ]
    memres_per_chain: dict[str, list[float]] = {}
    for fp in memres_seed_files:
        if not fp.exists():
            continue
        seed_pc = _per_chain_dnm(json.loads(fp.read_text()))
        for cid, v in seed_pc.items():
            memres_per_chain.setdefault(cid, []).append(v)
    memres_chain = {cid: float(np.mean(vs)) for cid, vs in memres_per_chain.items()}

    # Load oracle-RAG (the strongest RAG cell)
    rag_blob = json.loads((RAG_DIR / "qwen3_0p6b_oracle_top3.json").read_text())
    rag_chain = _per_chain_drag(rag_blob)

    common = sorted(set(memres_chain.keys()) & set(rag_chain.keys()))
    xs = np.asarray([rag_chain[c] for c in common], dtype=np.float64)
    ys = np.asarray([memres_chain[c] for c in common], dtype=np.float64)
    n = xs.shape[0]

    # Stats for the caption
    n_mem_positive = int((ys > 0).sum())
    n_rag_positive = int((xs > 0).sum())
    n_both = int(((xs > 0) & (ys > 0)).sum())
    n_mem_only = int(((xs <= 0) & (ys > 0)).sum())
    n_rag_only = int(((xs > 0) & (ys <= 0)).sum())
    n_neither = int(((xs <= 0) & (ys <= 0)).sum())

    # Layout: scatter + marginals
    fig = plt.figure(figsize=(5.6, 5.2))
    gs = fig.add_gridspec(
        4, 4, wspace=0.05, hspace=0.05,
        left=0.12, right=0.97, top=0.93, bottom=0.10,
    )
    ax = fig.add_subplot(gs[1:, :3])
    ax_top = fig.add_subplot(gs[0, :3], sharex=ax)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax)

    # Quadrant shading for "memres wins"
    ax.axhspan(0, 100, color="#fef3c7", alpha=0.35, zorder=0)
    ax.axvspan(0, 100, color="#dbeafe", alpha=0.20, zorder=0)
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.axvline(0, color="black", linewidth=0.6, zorder=1)
    # y = x diagonal as reference
    lim = float(max(abs(xs).max() if n else 1, abs(ys).max() if n else 1)) * 1.05
    diag = np.linspace(-lim, lim, 100)
    ax.plot(diag, diag, "--", color="#94a3b8", linewidth=0.6, zorder=1, label="$y = x$")

    # Scatter — colour by whether memres wins
    win_color = "#d97706"  # memres > oracle
    lose_color = "#475569"  # oracle >= memres
    win_mask = ys > xs
    ax.scatter(xs[win_mask], ys[win_mask], color=win_color, edgecolor="black",
               linewidth=0.4, s=22, alpha=0.85, label=f"memres wins (n={int(win_mask.sum())})",
               zorder=3)
    ax.scatter(xs[~win_mask], ys[~win_mask], color=lose_color, edgecolor="black",
               linewidth=0.4, s=22, alpha=0.85, label=f"oracle wins (n={int((~win_mask).sum())})",
               zorder=3)

    ax.set_xlabel(r"$\Delta_{\mathrm{cb}}$ (oracle-RAG, top-3)  [nats]", fontsize=9)
    ax.set_ylabel(r"$\Delta_{\mathrm{cb}}$ (memres, mean over 4 seeds)  [nats]", fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Marginals
    bins = np.linspace(-lim, lim, 24)
    ax_top.hist(xs, bins=bins, color=lose_color, alpha=0.55, edgecolor="black", linewidth=0.4)
    ax_top.axvline(0, color="black", linewidth=0.6)
    ax_top.set_ylabel("oracle\n# chains", fontsize=8)
    ax_top.tick_params(labelsize=7, labelbottom=False)
    ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)

    ax_right.hist(ys, bins=bins, color=win_color, alpha=0.55, orientation="horizontal",
                  edgecolor="black", linewidth=0.4)
    ax_right.axhline(0, color="black", linewidth=0.6)
    ax_right.set_xlabel("memres\n# chains", fontsize=8)
    ax_right.tick_params(labelsize=7, labelleft=False)
    ax_right.spines["top"].set_visible(False); ax_right.spines["right"].set_visible(False)

    fig.suptitle(
        f"Per-chain robustness ({n} chains).  Memres positive on {n_mem_positive}/{n}; oracle-RAG positive on {n_rag_positive}/{n}.",
        fontsize=9, y=0.99,
    )
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    print(f"saved -> {OUT_PATH}")
    print(f"  n_chains: {n}")
    print(f"  memres positive on {n_mem_positive}/{n}; oracle-RAG positive on {n_rag_positive}/{n}")
    print(f"  both positive: {n_both}; memres only: {n_mem_only}; oracle only: {n_rag_only}; neither: {n_neither}")
    print(f"  memres wins on {int(win_mask.sum())}/{n} chains")


if __name__ == "__main__":
    main()
