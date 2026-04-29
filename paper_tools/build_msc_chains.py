#!/usr/bin/env python3
"""Convert the Multi-Session Chat (MSC) parquet dataset into MemRes chain JSONLs.

MSC v2 is a 4-5 session per-dialogue conversational corpus from FAIR.
Each row in the source parquet is one (dialogue, session) tuple with
the schema::

    dataset       : str        # always "msc"
    dialoug_id    : int64      # FAIR's misspelling preserved by the HF dump
    session_id    : int64      # 0..4 (typically), the session index in the dialogue
    persona1      : list[str]  # facts about speaker_1
    persona2      : list[str]  # facts about speaker_2
    dialogue      : list[str]  # the utterances, in order
    speaker       : list[str]  # speaker label per utterance ("Speaker 1" / "Speaker 2")

We emit one JSONL file per dialogue (chain), one row per session.  Each
emitted row has the chain-corpus format that ``pretokenize_chains.py``
expects::

    {"source": "msc",
     "chain_id": int,
     "session_index": int,
     "session_id": "ses<i>",
     "title": "...",
     "boundary": "natural",
     "turns": [{"speaker": "Speaker 1", "text": "..."}, ...]}

The first session of each chain gets a short *system prefix* turn that
declares the personas (using FAIR's persona facts); subsequent sessions
do not -- this makes the dataset look like a chatbot whose system prompt
is established once at chain start.  The model must remember persona
content across sessions on its own (the whole point of MemRes on this
corpus).

Usage::

    python paper_tools/build_msc_chains.py \
        --in_parquet /path/to/msc/data/train-*.parquet \
        --out_dir    /path/to/stage1/msc/train \
        --max_chains 20000 \
        --min_sessions 3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def render_persona_prefix(p1: list[str], p2: list[str]) -> str:
    p1_lines = [s.strip() for s in (p1 or []) if s and s.strip()]
    p2_lines = [s.strip() for s in (p2 or []) if s and s.strip()]
    parts = []
    if p1_lines:
        parts.append("Speaker 1 persona: " + " ".join(p1_lines))
    if p2_lines:
        parts.append("Speaker 2 persona: " + " ".join(p2_lines))
    return "\n".join(parts)


def build_session_row(
    *,
    chain_id: int,
    session_index: int,
    persona1: list[str],
    persona2: list[str],
    dialogue: list[str],
    speakers: list[str],
    include_persona_prefix: bool,
) -> dict | None:
    """Materialise one session as a chain-corpus row."""
    turns: list[dict] = []
    if include_persona_prefix:
        prefix_text = render_persona_prefix(persona1, persona2)
        if prefix_text:
            turns.append({"speaker": "System", "text": prefix_text})

    if dialogue is None or speakers is None:
        return None
    if len(dialogue) != len(speakers):
        # Defensive: drop misaligned sessions rather than scramble speaker labels.
        return None

    for spk, utt in zip(speakers, dialogue):
        utt_str = (utt or "").strip()
        spk_str = (spk or "Speaker").strip() or "Speaker"
        if not utt_str:
            continue
        turns.append({"speaker": spk_str, "text": utt_str})

    if not any(t["speaker"] != "System" for t in turns):
        return None

    return {
        "source": "msc",
        "chain_id": int(chain_id),
        "session_index": int(session_index),
        "session_id": f"ses{int(session_index)}",
        "title": f"msc_dialog_{int(chain_id)}_session_{int(session_index)}",
        "boundary": "natural",
        "turns": turns,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_parquet", required=True, type=Path,
        help="Path to a MSC parquet split (e.g. train-00000-of-00001-*.parquet).",
    )
    ap.add_argument(
        "--out_dir", required=True, type=Path,
        help="Directory to write per-chain *.jsonl files into.",
    )
    ap.add_argument(
        "--max_chains", type=int, default=None,
        help="Cap on number of dialogues to emit (sorted by dialoug_id).",
    )
    ap.add_argument(
        "--min_sessions", type=int, default=2,
        help="Drop chains with fewer than this many sessions after parsing.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[msc] reading {args.in_parquet} ...")
    df = pd.read_parquet(args.in_parquet)
    print(f"[msc]   {len(df):,} rows")

    def _to_list(x) -> list:
        # Pandas/NumPy give us object arrays for `sequence` HF features; we
        # can't use Python's truthiness on those.  None -> [], otherwise
        # coerce to a python list of strings.
        if x is None:
            return []
        try:
            return [str(s) for s in list(x)]
        except TypeError:
            return []

    # Group by dialogue id and sort by session id.
    grouped: dict[int, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        grouped[int(row["dialoug_id"])].append({
            "session_id": int(row["session_id"]),
            "persona1": _to_list(row.get("persona1")),
            "persona2": _to_list(row.get("persona2")),
            "dialogue": _to_list(row.get("dialogue")),
            "speaker": _to_list(row.get("speaker")),
        })

    chain_ids = sorted(grouped.keys())
    if args.max_chains is not None:
        chain_ids = chain_ids[: args.max_chains]
    print(f"[msc] {len(chain_ids):,} dialogues considered "
          f"(after --max_chains cap)")

    written = 0
    dropped_short = 0
    total_sessions = 0
    for cid in tqdm(chain_ids, desc="msc chains"):
        sessions = sorted(grouped[cid], key=lambda s: s["session_id"])
        if len(sessions) < args.min_sessions:
            dropped_short += 1
            continue

        rows: list[dict] = []
        for i, sess in enumerate(sessions):
            row = build_session_row(
                chain_id=cid,
                session_index=i,
                persona1=sess["persona1"],
                persona2=sess["persona2"],
                dialogue=sess["dialogue"],
                speakers=sess["speaker"],
                include_persona_prefix=(i == 0),
            )
            if row is not None:
                rows.append(row)

        if len(rows) < args.min_sessions:
            dropped_short += 1
            continue

        out_path = args.out_dir / f"msc_{cid:07d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        written += 1
        total_sessions += len(rows)

    print(f"[msc] wrote {written:,} chains "
          f"({total_sessions:,} sessions; "
          f"dropped {dropped_short:,} short)")
    print(f"[msc] -> {args.out_dir}")


if __name__ == "__main__":
    main()
