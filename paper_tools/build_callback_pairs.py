#!/usr/bin/env python3
"""Build a small held-out pair dataset for callback_probe.py from PG-19 test
and LoCoMo chains.

The pair dataset is what callback_probe.py consumes: each row is
{"history": str, "current": str, "source": str, "chain_id": str}.
We construct each pair by:
  - choosing a chain in the source corpus,
  - taking the last 4 sessions as history (concatenated as plain text), and
  - the next session (if it exists) as current.

For PG-19 we use the raw .text field; for LoCoMo we use the same; for
TV stage1 we use the .text field (rendered transcripts).

The small held-out pair set is used to compute the per-callback-token
help ratio (?_callback / ?_filler), which is the rhetorically-most-
powerful diagnostic in the paper.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def iter_jsonl(p: Path) -> Iterable[dict]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def chain_to_pair(rows: list[dict], history_sessions: int) -> dict | None:
    rows.sort(key=lambda r: int(r.get("session_index", 0)))
    rows = [r for r in rows if r.get("text")]
    if len(rows) < history_sessions + 1:
        return None
    pivot = max(history_sessions, len(rows) - 1)
    h_rows = rows[max(0, pivot - history_sessions):pivot]
    cur = rows[pivot]
    history = "\n\n".join(r["text"] for r in h_rows)
    current = cur["text"]
    if len(history) < 256 or len(current) < 128:
        return None
    return {
        "source": cur.get("source") or "unknown",
        "chain_id": cur.get("conv_id") or cur.get("book_id") or cur.get("show_id"),
        "history": history,
        "current": current,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=Path("../memory_residuals_data"))
    p.add_argument("--out_dir", type=Path,
                   default=Path("paper_artifacts/pairs_eval"))
    p.add_argument("--history_sessions", type=int, default=4)
    p.add_argument("--max_per_split", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = random.Random(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("pg19_test", a.data_root / "stage1" / "pg19" / "test", "*.jsonl"),
        ("locomo",    Path("paper_artifacts/locomo_chains"),     "*.jsonl"),
    ]

    counts = {}
    for split_name, root, glob in sources:
        if not root.exists():
            print(f"[pairs] skipping {split_name}: {root} missing")
            continue
        files = sorted(root.glob(glob))
        rng.shuffle(files)
        out_path = a.out_dir / f"{split_name}.jsonl"
        n = 0
        with out_path.open("w", encoding="utf-8") as fout:
            for f in files:
                rows = list(iter_jsonl(f))
                pair = chain_to_pair(rows, a.history_sessions)
                if pair is None:
                    continue
                fout.write(json.dumps(pair) + "\n")
                n += 1
                if n >= a.max_per_split:
                    break
        counts[split_name] = n
        print(f"[pairs] wrote {n} pairs -> {out_path}")

    (a.out_dir / "manifest.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
