#!/usr/bin/env python3
"""Build the figures used in paper/memory_residuals.tex.

Inputs:
  - logs/chain_v3_softparity_full.log
  - logs/chain_v3_attentionbase_full.log
  - logs/chain_v2_abl_residual_mode.log  (kept for the historical simple_gate plateau)
  - paper_artifacts/eval/*_eval.json
  - paper_artifacts/eval/*_horizon.json
  - paper_artifacts/eval/init_parity_test.json

Outputs (paper/figures/):
  - trajectory.pdf       ?_sh-m and ?_nm-m vs step for the three modes
  - horizon_pg19_test.pdf ?_sh-m vs chain length bucket
  - gate_profile.pdf     per-sublayer gate magnitude (simple_gate) vs
                         per-sublayer router weight on b_{-1} (attention_parity)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EVAL_LINE_RE = re.compile(
    r"EVAL @ step\s+(\d+):\s+n=(\d+)\s+"
    r"mem=([0-9.]+)\s+nomem=([0-9.]+)\s+shuffle=([0-9.]+)\s+oracle=([0-9.]+)\s+"
    r"\u0394nm-m=([+\-0-9.]+)\s+\u0394sh-m=([+\-0-9.]+)\s+\u0394or-m=([+\-0-9.]+)"
)


def parse_eval_lines(log_path: Path) -> list[dict]:
    rows = []
    if not log_path.exists():
        return rows
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    for m in EVAL_LINE_RE.finditer(txt):
        rows.append({
            "step": int(m.group(1)),
            "n": int(m.group(2)),
            "ce_mem": float(m.group(3)),
            "ce_nomem": float(m.group(4)),
            "ce_shuffle": float(m.group(5)),
            "ce_oracle": float(m.group(6)),
            "delta_nm_m": float(m.group(7)),
            "delta_sh_m": float(m.group(8)),
            "delta_or_m": float(m.group(9)),
        })
    return rows


def figure_trajectory(out_path: Path, args: argparse.Namespace) -> None:
    """?_nm-m and ?_sh-m vs step for all three routing modes (in-trainer eval)."""
    runs = []
    for label, log, color, dash in [
        ("attention_parity (soft)", args.softparity_log, "C0", "-"),
        ("attention_base", args.attentionbase_log, "C1", "-"),
        ("simple_gate", args.simplegate_log, "C2", "-"),
    ]:
        rows = parse_eval_lines(Path(log))
        runs.append((label, rows, color, dash))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=False)
    for label, rows, color, dash in runs:
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        d_nm = [r["delta_nm_m"] for r in rows]
        d_sh = [r["delta_sh_m"] for r in rows]
        axes[0].plot(steps, d_nm, color=color, linestyle=dash,
                     label=label, linewidth=1.4)
        axes[1].plot(steps, d_sh, color=color, linestyle=dash,
                     label=label, linewidth=1.4)
    for ax, ttl, ylab in [
        (axes[0], r"$\Delta_{nm-m}$ (memory help)",
         r"$\mathrm{CE}_{\text{nomem}} - \mathrm{CE}_{\text{mem}}$ (nats)"),
        (axes[1], r"$\Delta_{sh-m}$ (history specificity)",
         r"$\mathrm{CE}_{\text{shuffle}} - \mathrm{CE}_{\text{mem}}$ (nats)"),
    ]:
        ax.set_xlabel("training step")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.axhline(0, color="0.5", linewidth=0.6, linestyle=":")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def figure_horizon(out_path: Path, args: argparse.Namespace) -> None:
    """Per-bucket ?_sh-m on PG-19 test for the three modes."""
    sources = []
    for label, json_path, color in [
        ("attention_parity (soft)", args.softparity_horizon, "C0"),
        ("attention_base", args.attentionbase_horizon, "C1"),
        ("simple_gate", args.simplegate_horizon, "C2"),
    ]:
        p = Path(json_path)
        if p.exists():
            blob = json.loads(p.read_text(encoding="utf-8"))
            buckets = blob.get("pg19_test", {}).get("buckets") or \
                      blob.get("pg19_validation", {}).get("buckets") or []
            sources.append((label, buckets, color))
    if not sources:
        print("  (skipping horizon figure: no horizon JSONs found)")
        return

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    n_modes = len(sources)
    width = 0.8 / max(n_modes, 1)
    bucket_labels = [b["label"] for b in sources[0][1]]
    x = np.arange(len(bucket_labels))

    for i, (label, buckets, color) in enumerate(sources):
        vals = [b.get("delta_shuffle_minus_mem", float("nan")) for b in buckets]
        ax.bar(x + (i - (n_modes - 1) / 2) * width, vals, width,
               label=label, color=color, alpha=0.85)
    ax.axhline(0, color="0.3", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("chain length (sessions)")
    ax.set_ylabel(r"$\Delta_{sh-m}$ (nats)")
    ax.set_title(r"History-specific memory benefit by horizon (PG-19 test)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def figure_gate_profile(out_path: Path, args: argparse.Namespace) -> None:
    """Per-sublayer learned routing weight on the memory source.

    Two panels, side-by-side, to keep the very different y-axis scales
    of simple_gate's |g_l| and attention_parity's mem_bias readable.
    """
    import torch
    from safetensors.torch import load_file

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))

    # Left: simple_gate |g_l| (small numbers, ReZero scalar)
    if args.simplegate_ckpt and Path(args.simplegate_ckpt).exists():
        st = load_file(str(Path(args.simplegate_ckpt) / "model.safetensors"))
        if "model.memory_gate.gate" in st:
            gates = st["model.memory_gate.gate"].float().abs().cpu().numpy()
            axes[0].plot(np.arange(len(gates)), gates, "o-",
                         color="C2", label="$|g_\\ell|$",
                         markersize=4, linewidth=1.2)
            axes[0].axhline(0, color="0.6", linewidth=0.5, linestyle=":")
    axes[0].set_xlabel("sublayer index $\\ell$")
    axes[0].set_ylabel("$|g_\\ell|$  (ReZero gate magnitude)")
    axes[0].set_title("\\textsc{simple\\_gate}", fontsize=10)
    axes[0].grid(alpha=0.25)

    # Right: attention_parity / attention_base recent_bias - mem_bias
    # (i.e. effective log-odds of recent vs memory at init, larger = more
    #  recent-favoured; smaller / negative = more memory-favoured)
    def _plot_router(path: Path, color: str, label: str, marker: str):
        if not path.exists():
            return
        st = load_file(str(path / "model.safetensors"))
        if "model.depth_router.mem_bias" in st and \
                "model.depth_router.recent_bias" in st:
            mem = st["model.depth_router.mem_bias"].float().cpu().numpy()
            rec = st["model.depth_router.recent_bias"].float().cpu().numpy()
            log_odds = rec - mem
            axes[1].plot(np.arange(len(mem)), log_odds, marker + "-",
                         color=color, label=label,
                         markersize=4, linewidth=1.2)

    if args.softparity_ckpt:
        _plot_router(Path(args.softparity_ckpt), "C0",
                     r"\textsc{attn\_parity}", "s")
    if args.attentionbase_ckpt:
        _plot_router(Path(args.attentionbase_ckpt), "C1",
                     r"\textsc{attn\_base}", "d")

    axes[1].axhline(8, color="0.6", linewidth=0.5, linestyle=":")
    axes[1].set_xlabel("sublayer index $\\ell$")
    axes[1].set_ylabel(r"$\mathrm{recent\_bias}_\ell - \mathrm{mem\_bias}_\ell$")
    axes[1].set_title("Block AttnRes routers", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False, loc="best")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--softparity_log",
                   default="logs/chain_v3_softparity_full.log")
    p.add_argument("--attentionbase_log",
                   default="logs/chain_v3_attentionbase_full.log")
    p.add_argument("--simplegate_log",
                   default="logs/chain_v2_abl_residual_mode.log")
    p.add_argument("--softparity_horizon",
                   default="paper_artifacts/eval/chain_v3_softparity_full_eval_horizon.json")
    p.add_argument("--attentionbase_horizon",
                   default="paper_artifacts/eval/chain_v3_attentionbase_full_eval_horizon.json")
    p.add_argument("--simplegate_horizon",
                   default="paper_artifacts/eval/chain_v2_abl_residual_mode_step5200_eval_horizon.json")
    p.add_argument("--softparity_ckpt",
                   default="output/chain_v3_softparity_full/best")
    p.add_argument("--attentionbase_ckpt",
                   default="output/chain_v3_attentionbase_full/best")
    p.add_argument("--simplegate_ckpt",
                   default="output/chain_v2_abl_residual_mode/best")
    p.add_argument("--out_dir", default="paper/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_trajectory(out_dir / "trajectory.pdf", args)
    figure_horizon(out_dir / "horizon_pg19_test.pdf", args)
    figure_gate_profile(out_dir / "gate_profile.pdf", args)


if __name__ == "__main__":
    main()
