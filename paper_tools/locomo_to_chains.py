#!/usr/bin/env python3
"""Convert the LoCoMo benchmark into our chain JSONL format.

LoCoMo is a public 10-conversation, 19-32-sessions-each long-conversation
memory benchmark released by the Mem0 team.  Each conversation has two
named speakers and a long arc of dated sessions.  We render each session as

    Speaker: text
    Speaker: text
    ...

and emit one JSONL file per conversation, with one row per session, sharing
the same shape that ``paper_tools/pretokenize_chains.py`` consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_session(turns: list[dict]) -> str:
    """Render a LoCoMo session list-of-turns as Speaker: text per line."""
    parts = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker") or "Unknown"
        text = turn.get("text") or ""
        if text:
            parts.append(f"{speaker}: {text}")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in_path",
        default="../memory_residuals_data/hf_corpora/locomo/raw/locomo10.json",
        type=Path,
    )
    p.add_argument(
        "--out_dir",
        default="paper_artifacts/locomo_chains",
        type=Path,
    )
    a = p.parse_args()

    data = json.loads(a.in_path.read_text(encoding="utf-8"))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for conv in data:
        cid = conv["sample_id"]
        c = conv["conversation"]
        # Find session keys in order, ignoring session_N_date_time entries.
        session_keys = sorted(
            (k for k in c if k.startswith("session_") and not k.endswith("date_time")),
            key=lambda k: int(k.split("_")[1]),
        )
        out_path = a.out_dir / f"{cid}.jsonl"
        rows = []
        for idx, k in enumerate(session_keys):
            turns = c[k]
            text = render_session(turns) if isinstance(turns, list) else str(turns or "")
            if not text.strip():
                continue
            rows.append({
                "source": "locomo",
                "conv_id": cid,
                "session_index": idx,
                "session_key": k,
                "speaker_a": c.get("speaker_a"),
                "speaker_b": c.get("speaker_b"),
                "text": text,
            })
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary.append({"conv_id": cid, "n_sessions": len(rows)})
        print(f"{cid}: wrote {len(rows)} sessions -> {out_path}")

    (a.out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Manifest -> {a.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
