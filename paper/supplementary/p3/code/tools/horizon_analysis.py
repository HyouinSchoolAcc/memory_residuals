#!/usr/bin/env python3
"""Horizon-bucketed analysis of eval_chain.py output.

The standard ``eval_chain.py`` aggregates per-token NLLs across every
chain in the corpus.  Reviewers will (correctly) ask whether the memory
benefit decays gracefully or collapses past some horizon — PITFALLS §5
calls this out as the most architecturally serious failure mode.

This script reads one or more eval_chain JSONs, buckets each per-chain
result by chain length, and writes:

    - ``<basename>_horizon.json`` — bucket-level aggregates
    - ``<basename>_horizon.md``  — Markdown table for the paper

Bucket edges default to {1..7, 8..15, 16..31, 32..63, 64..127, 128+},
which roughly correspond to short / medium / long / very-long /
LRMT-stress / extreme regimes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_BUCKETS = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 10**9)]


def bucket_label(lo: int, hi: int) -> str:
    if hi >= 10**9:
        return f"{lo}+"
    return f"{lo}-{hi}"


def aggregate_bucket(chains: list[dict], lo: int, hi: int) -> dict:
    matched = [c for c in chains if lo <= int(c.get("length", 0)) <= hi]
    n = len(matched)
    if n == 0:
        return {"n_chains": 0}

    def mean_field(name: str) -> float | None:
        vals = [c[name] for c in matched
                if name in c and c[name] is not None
                and not (isinstance(c[name], float) and c[name] != c[name])]
        return float(sum(vals) / len(vals)) if vals else None

    out = {
        "n_chains": n,
        "mean_chain_length": float(
            sum(int(c["length"]) for c in matched) / n
        ),
        "ce_mem": mean_field("ce_mem"),
        "ce_nomem": mean_field("ce_nomem"),
        "ce_shuffle": mean_field("ce_shuffle"),
        "ce_oracle": mean_field("ce_oracle"),
        "ce_rag": mean_field("ce_rag"),
    }
    if out["ce_mem"] is not None and out["ce_nomem"] is not None:
        out["delta_nomem_minus_mem"] = out["ce_nomem"] - out["ce_mem"]
    if out["ce_mem"] is not None and out["ce_shuffle"] is not None:
        out["delta_shuffle_minus_mem"] = out["ce_shuffle"] - out["ce_mem"]
    if out["ce_mem"] is not None and out["ce_oracle"] is not None:
        out["delta_oracle_minus_mem"] = out["ce_oracle"] - out["ce_mem"]
    return out


def render_markdown(corpus_name: str, buckets: list[tuple[int, int]],
                    rows: list[dict]) -> str:
    lines = []
    lines.append(f"### {corpus_name} — horizon-bucketed memory benefit")
    lines.append("")
    lines.append("| horizon | n | mean len | CE_mem | CE_nomem | CE_shuffle | CE_oracle "
                 "| Δ_nm-m | Δ_sh-m | Δ_or-m |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (lo, hi), r in zip(buckets, rows):
        if r["n_chains"] == 0:
            continue
        def fmt(v):
            if v is None:
                return "—"
            return f"{v:.4f}"
        def fmt_d(v):
            if v is None:
                return "—"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.4f}"
        lines.append(
            f"| {bucket_label(lo, hi)} | {r['n_chains']} | "
            f"{r['mean_chain_length']:.1f} | "
            f"{fmt(r['ce_mem'])} | {fmt(r['ce_nomem'])} | "
            f"{fmt(r['ce_shuffle'])} | {fmt(r['ce_oracle'])} | "
            f"{fmt_d(r.get('delta_nomem_minus_mem'))} | "
            f"{fmt_d(r.get('delta_shuffle_minus_mem'))} | "
            f"{fmt_d(r.get('delta_oracle_minus_mem'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True,
                   help="One or more eval_chain JSONs.")
    p.add_argument(
        "--buckets", type=str, default=None,
        help="Comma-separated bucket bounds, e.g. '7,15,31,63,127'. "
             "Default: 7,15,31,63,127 -> {1-7,8-15,16-31,32-63,64-127,128+}."
    )
    p.add_argument("--out_dir", type=Path,
                   default=Path("results/eval"))
    args = p.parse_args()

    if args.buckets:
        edges = [int(x) for x in args.buckets.split(",")]
        buckets = []
        prev = 0
        for e in edges:
            buckets.append((prev + (1 if prev else 0), e))
            prev = e
        buckets.append((prev + 1, 10**9))
    else:
        buckets = DEFAULT_BUCKETS

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.inputs:
        blob = json.loads(src.read_text(encoding="utf-8"))
        # eval_chain.py emits {corpus_name: {... per_chain: [...]} } at top-level.
        all_md: list[str] = []
        all_json: dict = {}
        for corpus_name, payload in blob.items():
            chains = payload.get("per_chain", [])
            rows = [aggregate_bucket(chains, lo, hi) for (lo, hi) in buckets]
            all_md.append(render_markdown(corpus_name, buckets, rows))
            all_json[corpus_name] = {
                "n_chains_total": len(chains),
                "buckets": [
                    {"lo": lo, "hi": hi, "label": bucket_label(lo, hi), **r}
                    for (lo, hi), r in zip(buckets, rows)
                ],
            }
        out_json = args.out_dir / (src.stem + "_horizon.json")
        out_md = args.out_dir / (src.stem + "_horizon.md")
        out_json.write_text(json.dumps(all_json, indent=2), encoding="utf-8")
        out_md.write_text("\n".join(all_md), encoding="utf-8")
        print(f"  {src.name} -> {out_json.name}, {out_md.name}")


if __name__ == "__main__":
    main()
