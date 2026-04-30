#!/usr/bin/env python3
"""Audit Memory Residuals corpora and write paper-ready data statistics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"count": 0, "mean": 0, "median": 0, "p90": 0, "max": 0}
    ordered = sorted(lengths)
    p90_idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "count": len(lengths),
        "mean": round(statistics.fmean(lengths), 2),
        "median": int(statistics.median(lengths)),
        "p90": int(ordered[p90_idx]),
        "max": max(lengths),
    }


def encode_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def audit_stage1_pg19(data_root: Path, tokenizer, max_train_books: int | None) -> dict[str, Any]:
    root = data_root / "stage1" / "pg19"
    out: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        files = sorted((root / split).glob("*.jsonl"))
        if split == "train" and max_train_books is not None:
            files = files[:max_train_books]
        chars = 0
        tokens = 0
        sessions = 0
        books = set()
        boundaries: dict[str, int] = {}
        session_token_lengths: list[int] = []
        for path in tqdm(files, desc=f"stage1/pg19/{split}"):
            for row in iter_jsonl(path):
                text = row.get("text", "")
                n_tok = encode_len(tokenizer, text)
                chars += len(text)
                tokens += n_tok
                sessions += 1
                books.add(row.get("book_id", path.stem))
                boundaries[row.get("boundary", "unknown")] = boundaries.get(row.get("boundary", "unknown"), 0) + 1
                session_token_lengths.append(n_tok)
        out[split] = {
            "books": len(books),
            "sessions": sessions,
            "chars": chars,
            "qwen_tokens": tokens,
            "boundaries": boundaries,
            "session_tokens": summarize_lengths(session_token_lengths),
            "is_train_subset": split == "train" and max_train_books is not None,
        }
    return out


def audit_stage1_tv(data_root: Path, tokenizer) -> dict[str, Any]:
    root = data_root / "stage1" / "tv"
    shows = []
    totals = {"shows": 0, "sessions": 0, "chars": 0, "qwen_tokens": 0}
    for path in tqdm(sorted(root.glob("*.jsonl")), desc="stage1/tv"):
        chars = 0
        tokens = 0
        sessions = 0
        lengths: list[int] = []
        for row in iter_jsonl(path):
            text = row.get("text", "")
            n_tok = encode_len(tokenizer, text)
            chars += len(text)
            tokens += n_tok
            sessions += 1
            lengths.append(n_tok)
        shows.append(
            {
                "show_id": path.stem,
                "sessions": sessions,
                "chars": chars,
                "qwen_tokens": tokens,
                "session_tokens": summarize_lengths(lengths),
            }
        )
        totals["shows"] += 1
        totals["sessions"] += sessions
        totals["chars"] += chars
        totals["qwen_tokens"] += tokens
    return {"totals": totals, "shows": shows}


def render_stage2_turns(row: dict[str, Any]) -> str:
    parts = []
    for turn in row.get("turns", []):
        speaker = turn.get("speaker") or "Narration"
        text = turn.get("text") or ""
        if text:
            parts.append(f"{speaker}: {text}")
    return "\n".join(parts)


def sequence_values(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(x) for x in value]


def audit_stage2(data_root: Path, tokenizer) -> dict[str, Any]:
    root = data_root / "stage2"
    chain_manifest = {}
    chain_manifest_path = data_root / "tv_continuity_chains" / "manifest.json"
    if chain_manifest_path.exists():
        chain_manifest = json.loads(chain_manifest_path.read_text(encoding="utf-8"))
    speaker_ratios = {
        s["show_id"]: s.get("avg_speaker_label_ratio")
        for s in chain_manifest.get("shows", [])
    }

    shows = []
    totals = {"shows": 0, "sessions": 0, "turns": 0, "chars": 0, "qwen_tokens": 0}
    all_ratios = []
    for path in tqdm(sorted(root.glob("*.jsonl")), desc="stage2"):
        chars = 0
        tokens = 0
        sessions = 0
        turns = 0
        lengths: list[int] = []
        for row in iter_jsonl(path):
            text = render_stage2_turns(row)
            n_tok = encode_len(tokenizer, text)
            row_turns = len(row.get("turns", []))
            chars += len(text)
            tokens += n_tok
            sessions += 1
            turns += row_turns
            lengths.append(n_tok)
        ratio = speaker_ratios.get(path.stem)
        if ratio is not None:
            all_ratios.append(float(ratio))
        shows.append(
            {
                "show_id": path.stem,
                "sessions": sessions,
                "turns": turns,
                "chars": chars,
                "qwen_tokens": tokens,
                "avg_speaker_label_ratio": ratio,
                "session_tokens": summarize_lengths(lengths),
            }
        )
        totals["shows"] += 1
        totals["sessions"] += sessions
        totals["turns"] += turns
        totals["chars"] += chars
        totals["qwen_tokens"] += tokens
    return {
        "totals": totals,
        "speaker_label_ratio": summarize_lengths([int(r * 1000) for r in all_ratios]),
        "shows": shows,
    }


def audit_eval_sets(data_root: Path, tokenizer) -> dict[str, Any]:
    out: dict[str, Any] = {}

    msc_root = data_root / "hf_corpora" / "msc" / "data"
    msc = {}
    for split in ("train", "validation", "test"):
        rows = []
        for path in sorted(msc_root.glob(f"{split}-*.parquet")):
            rows.append(pd.read_parquet(path))
        if rows:
            df = pd.concat(rows, ignore_index=True)
            chars = 0
            toks = 0
            turns = 0
            for dialogue in tqdm(df["dialogue"], desc=f"msc/{split}"):
                values = sequence_values(dialogue)
                text = "\n".join(values)
                chars += len(text)
                toks += encode_len(tokenizer, text)
                turns += len(values)
            msc[split] = {"rows": len(df), "turns": turns, "chars": chars, "qwen_tokens": toks}
    out["msc"] = msc

    locomo_path = data_root / "hf_corpora" / "locomo" / "transformed" / "locomo_mc10_with_name.json"
    if locomo_path.exists():
        raw = locomo_path.read_text(encoding="utf-8")
        try:
            obj = json.loads(raw)
            items = obj if isinstance(obj, list) else obj.get("data", [])
        except json.JSONDecodeError:
            items = [json.loads(line) for line in raw.splitlines() if line.strip()]
        type_counts: dict[str, int] = {}
        sessions = []
        for item in items:
            qtype = item.get("question_type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            if "num_sessions" in item:
                sessions.append(int(item["num_sessions"]))
        out["locomo_mc10"] = {
            "items": len(items),
            "question_types": type_counts,
            "num_sessions": summarize_lengths(sessions),
        }
    return out


def write_markdown(stats: dict[str, Any], output_md: Path) -> None:
    pg = stats["stage1_pg19"]
    tv1 = stats["stage1_tv"]["totals"]
    tv2 = stats["stage2"]["totals"]
    msc = stats["eval_sets"].get("msc", {})
    locomo = stats["eval_sets"].get("locomo_mc10", {})
    lines = [
        "# Memory Residuals Data Audit",
        "",
        f"Tokenizer: `{stats['tokenizer']}`",
        "",
        "## Stage 1",
        "",
        "| Source | Split | Books/Shows | Sessions | Chars | Qwen tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in pg.items():
        lines.append(
            f"| PG-19 | {split} | {row['books']:,} | {row['sessions']:,} | "
            f"{row['chars']:,} | {row['qwen_tokens']:,} |"
        )
    lines.append(
        f"| TV | train | {tv1['shows']:,} | {tv1['sessions']:,} | "
        f"{tv1['chars']:,} | {tv1['qwen_tokens']:,} |"
    )
    lines.extend(
        [
            "",
            "## Stage 2",
            "",
            "| Source | Shows | Episodes | Turns | Chars | Qwen tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| TV SFT | {tv2['shows']:,} | {tv2['sessions']:,} | {tv2['turns']:,} | "
            f"{tv2['chars']:,} | {tv2['qwen_tokens']:,} |",
            "",
            "## Eval Sets",
            "",
            "| Benchmark | Split | Rows/items | Turns | Chars | Qwen tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, row in msc.items():
        lines.append(
            f"| MSC | {split} | {row['rows']:,} | {row['turns']:,} | "
            f"{row['chars']:,} | {row['qwen_tokens']:,} |"
        )
    if locomo:
        lines.append(
            f"| LoCoMo-MC10 | all | {locomo['items']:,} | - | - | - |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("../memory_residuals_data"))
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-json", type=Path, default=Path("paper_artifacts/data_audit.json"))
    p.add_argument("--output-md", type=Path, default=Path("paper_artifacts/data_audit.md"))
    p.add_argument("--max-pg19-train-books", type=int, default=None)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stats = {
        "tokenizer": args.tokenizer,
        "stage1_pg19": audit_stage1_pg19(args.data_root, tokenizer, args.max_pg19_train_books),
        "stage1_tv": audit_stage1_tv(args.data_root, tokenizer),
        "stage2": audit_stage2(args.data_root, tokenizer),
        "eval_sets": audit_eval_sets(args.data_root, tokenizer),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_markdown(stats, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
