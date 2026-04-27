#!/usr/bin/env python3
"""Aggregate pilot experiment metrics into paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, default=Path("paper_artifacts/eval"))
    p.add_argument("--out-csv", type=Path, default=Path("paper_artifacts/results_summary.csv"))
    p.add_argument("--out-md", type=Path, default=Path("paper_artifacts/results_summary.md"))
    args = p.parse_args()

    rows = []
    for path in sorted(args.eval_dir.glob("qwen3-*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        mem = obj["memres"]
        base = obj.get("base", {})
        rows.append(
            {
                "preset": path.stem,
                "n": mem.get("n"),
                "base_ce": base.get("base_ce"),
                "with_memory_ce": mem.get("with_memory_ce"),
                "no_memory_ce": mem.get("no_memory_ce"),
                "delta_no_memory_minus_memory": mem.get("delta_no_memory_minus_memory"),
                "shuffle_history_ce": mem.get("shuffle_history_ce"),
                "delta_shuffle_minus_memory": mem.get("delta_shuffle_minus_memory"),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Pilot Results Summary",
        "",
        "| Preset | n | Base CE | MemRes CE | No-memory CE | NoMem-Mem | Shuffled CE | Shuffle-Mem |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(
            f"| `{r['preset']}` | {r['n']} | {fmt(r['base_ce'])} | "
            f"{fmt(r['with_memory_ce'])} | {fmt(r['no_memory_ce'])} | "
            f"{fmt(r['delta_no_memory_minus_memory'])} | {fmt(r['shuffle_history_ce'])} | "
            f"{fmt(r['delta_shuffle_minus_memory'])} |"
        )
    rag_path = args.eval_dir / "rag_qwen3-0.6b.json"
    if rag_path.exists():
        rag = json.loads(rag_path.read_text(encoding="utf-8"))
        md.extend(
            [
                "",
                "## RAG Baseline",
                "",
                f"- Qwen3-0.6B dense-retrieval baseline: CE {rag['rag_ce']:.4f} "
                f"on n={rag['n']} with top-k={rag['top_k']}.",
            ]
        )
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
