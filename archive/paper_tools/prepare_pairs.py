#!/usr/bin/env python3
"""Convert staged Memory Residuals chain corpora into history/current pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def render_stage2(row: dict[str, Any]) -> str:
    rendered = []
    for turn in row.get("turns", []):
        speaker = turn.get("speaker") or "Narration"
        text = turn.get("text") or ""
        if text:
            rendered.append(f"{speaker}: {text}")
    return "\n".join(rendered)


def load_chain(path: Path, text_field: str) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    rows.sort(key=lambda r: int(r.get("session_index", r.get("episode_index", 0))))
    if text_field == "stage2_turns":
        for row in rows:
            row["_pair_text"] = render_stage2(row)
    else:
        for row in rows:
            row["_pair_text"] = row.get(text_field, "")
    return [r for r in rows if r.get("_pair_text")]


def chain_pairs(
    path: Path,
    source: str,
    text_field: str,
    history_sessions: int,
    min_history_chars: int,
    min_current_chars: int,
) -> Iterable[dict[str, Any]]:
    rows = load_chain(path, text_field)
    for idx in range(1, len(rows)):
        start = max(0, idx - history_sessions)
        history_rows = rows[start:idx]
        current = rows[idx]["_pair_text"]
        history = "\n\n".join(r["_pair_text"] for r in history_rows)
        if len(history) < min_history_chars or len(current) < min_current_chars:
            continue
        yield {
            "source": source,
            "chain_id": rows[idx].get("book_id") or rows[idx].get("show_id") or path.stem,
            "current_session_index": rows[idx].get("session_index", idx),
            "history_session_start": rows[start].get("session_index", start),
            "history_session_end": rows[idx - 1].get("session_index", idx - 1),
            "history": history,
            "current": current,
        }


def collect_stage1_pg19(data_root: Path, split: str, args) -> list[dict[str, Any]]:
    root = data_root / "stage1" / "pg19" / split
    files = sorted(root.glob("*.jsonl"))
    if args.max_pg19_books is not None:
        files = files[: args.max_pg19_books]
    rows: list[dict[str, Any]] = []
    for path in tqdm(files, desc=f"pairs pg19/{split}"):
        rows.extend(
            chain_pairs(
                path,
                source=f"pg19_{split}",
                text_field="text",
                history_sessions=args.history_sessions,
                min_history_chars=args.min_history_chars,
                min_current_chars=args.min_current_chars,
            )
        )
    return rows


def collect_stage1_tv(data_root: Path, args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in tqdm(sorted((data_root / "stage1" / "tv").glob("*.jsonl")), desc="pairs stage1/tv"):
        rows.extend(
            chain_pairs(
                path,
                source="tv_stage1",
                text_field="text",
                history_sessions=args.history_sessions,
                min_history_chars=args.min_history_chars,
                min_current_chars=args.min_current_chars,
            )
        )
    return rows


def collect_stage2_tv(data_root: Path, args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in tqdm(sorted((data_root / "stage2").glob("*.jsonl")), desc="pairs stage2/tv"):
        rows.extend(
            chain_pairs(
                path,
                source="tv_stage2",
                text_field="stage2_turns",
                history_sessions=args.history_sessions,
                min_history_chars=args.min_history_chars,
                min_current_chars=args.min_current_chars,
            )
        )
    return rows


def cap_and_shuffle(rows: list[dict[str, Any]], max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("../memory_residuals_data"))
    p.add_argument("--out-dir", type=Path, default=Path("paper_artifacts/pairs"))
    p.add_argument("--history-sessions", type=int, default=4)
    p.add_argument("--min-history-chars", type=int, default=256)
    p.add_argument("--min-current-chars", type=int, default=128)
    p.add_argument("--max-pg19-books", type=int, default=None)
    p.add_argument("--max-stage1-train", type=int, default=200_000)
    p.add_argument("--max-stage1-eval", type=int, default=5_000)
    p.add_argument("--max-stage2-train", type=int, default=50_000)
    p.add_argument("--max-stage2-eval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train = collect_stage1_pg19(args.data_root, "train", args)
    train.extend(collect_stage1_tv(args.data_root, args))
    train = cap_and_shuffle(train, args.max_stage1_train, args.seed)

    validation = collect_stage1_pg19(args.data_root, "validation", args)
    validation = cap_and_shuffle(validation, args.max_stage1_eval, args.seed + 1)

    test = collect_stage1_pg19(args.data_root, "test", args)
    test = cap_and_shuffle(test, args.max_stage1_eval, args.seed + 2)

    stage2 = cap_and_shuffle(collect_stage2_tv(args.data_root, args), args.max_stage2_train, args.seed + 3)
    stage2_eval = stage2[: min(len(stage2), args.max_stage2_eval or len(stage2))]

    counts = {
        "stage1_train": write_jsonl(args.out_dir / "stage1_train.jsonl", train),
        "stage1_validation": write_jsonl(args.out_dir / "stage1_validation.jsonl", validation),
        "stage1_test": write_jsonl(args.out_dir / "stage1_test.jsonl", test),
        "stage2_train": write_jsonl(args.out_dir / "stage2_train.jsonl", stage2),
        "stage2_eval": write_jsonl(args.out_dir / "stage2_eval.jsonl", stage2_eval),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
