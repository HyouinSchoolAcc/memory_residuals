"""
Visualize per-layer memory routing mass α for a MemRes model.

Builds paired callback / filler prompts (same compressed history) and plots:
  - Bar chart: mean α per MemRes site, callback vs filler
  - Heatmap:   α per site per sample (callback minus filler)

Usage:
    python visualize_memres.py --model_path output/smoke_memres/final \\
        --data_path data/friends_scripts.jsonl --num_samples 16
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probe_memres import RoutingProbe


class MemResVisualizer:
    def __init__(self, probe: RoutingProbe):
        self.probe = probe

    def collect(self, samples):
        callback_traces, filler_traces = self.probe.run(samples)
        return np.array(callback_traces), np.array(filler_traces)

    @staticmethod
    def plot(cb: np.ndarray, fl: np.ndarray, model_name: str, output_path: str):
        if cb.size == 0:
            raise RuntimeError("No valid samples collected.")

        n_samples, n_sites = cb.shape
        cb_mean = cb.mean(axis=0)
        fl_mean = fl.mean(axis=0)
        delta = cb - fl

        fig, (ax_bar, ax_heat) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]}
        )
        fig.subplots_adjust(left=0.07, right=0.96, top=0.9, bottom=0.12, wspace=0.3)

        x = np.arange(n_sites)
        width = 0.4
        ax_bar.bar(x - width / 2, cb_mean, width, label="callback", color="#4CAF50")
        ax_bar.bar(x + width / 2, fl_mean, width, label="filler", color="#FF9800")
        ax_bar.set_xlabel("Routing layer (depth-wise)")
        ax_bar.set_ylabel(r"mean $\alpha_{b_{-1} \to l}$ (memory mass)")
        ax_bar.set_title("Routing mass per site")
        ax_bar.set_xticks(x)
        ax_bar.legend()
        ax_bar.grid(axis="y", alpha=0.3)

        vmax = float(np.abs(delta).max()) or 1e-6
        im = ax_heat.imshow(
            delta,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_heat.set_xlabel("Routing layer (depth-wise)")
        ax_heat.set_ylabel("sample index")
        ax_heat.set_title(r"$\Delta\alpha$ = callback - filler")
        ax_heat.set_xticks(x)
        cbar = plt.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
        cbar.set_label(r"$\Delta\alpha$")

        fig.suptitle(
            f"Memory Residuals depth-wise routing — {model_name}  "
            f"(n={n_samples}, layers={n_sites})",
            fontsize=12,
            fontweight="bold",
        )
        plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved visualization to {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--history_len", type=int, default=512)
    p.add_argument("--probe_len", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--eval_start", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    probe = RoutingProbe(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        device=torch.device(args.device),
        history_len=args.history_len,
        probe_len=args.probe_len,
    )

    with open(args.data_path) as f:
        lines = f.readlines()
    samples = [
        json.loads(line)
        for line in lines[args.eval_start : args.eval_start + args.num_samples]
    ]

    viz = MemResVisualizer(probe)
    cb, fl = viz.collect(samples)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    output_path = args.output or f"memres_routing_{model_name}.png"
    viz.plot(cb, fl, model_name, output_path)


if __name__ == "__main__":
    main()
