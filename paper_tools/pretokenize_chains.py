#!/usr/bin/env python3
"""Pre-tokenize Memory Residuals *chain* corpora into a packed tensor file.

Each chain is one PG-19 book (sequence of chapter sessions) or one TV show
(sequence of episode sessions).  The model's recurrent memory $M_c$ must
persist across all sessions of a chain in order, so the on-disk format is

    {
      "session_ids": LongTensor (N_total_sessions, S),  # one row per session
      "session_chain_id": LongTensor (N_total_sessions,),  # which chain
      "session_index": LongTensor (N_total_sessions,),  # position within chain
      "chain_starts": LongTensor (N_chains,),  # first session row per chain
      "chain_lengths": LongTensor (N_chains,),  # # sessions per chain
      "chain_names": list[str],  # debug, optional
      "session_len": int S,
      "tokenizer": str,
    }

Right-truncation: each session is encoded with at most ``session_len`` tokens
and padded with EOS up to ``session_len``.  Sessions that tokenize to fewer
than ``min_tokens`` are dropped (typically credits-only TV episodes).

Use chain-aware sampling at training time: for a window of k=4 sessions,
pick a random chain, pick a random valid start index s, slice
``session_ids[chain_starts[c] + s : chain_starts[c] + s + k]``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def init_worker(tokenizer_name: str):
    global _TOK, _CFG
    _TOK = AutoTokenizer.from_pretrained(tokenizer_name)


def encode_session(args: tuple[str, int, int, int]):
    """Tokenise a single session text to a fixed-length id list.

    Returns ``None`` if the session is too short to be useful.
    """
    text, session_len, min_tokens, eos_id = args
    if not text:
        return None
    ids = _TOK.encode(text, add_special_tokens=False)[:session_len]
    if len(ids) < min_tokens:
        return None
    if len(ids) < session_len:
        ids = ids + [eos_id] * (session_len - len(ids))
    return ids


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def render_session(row: dict) -> str:
    """Materialise a session text from the JSONL row."""
    if "text" in row and isinstance(row["text"], str):
        return row["text"]
    if "turns" in row:
        # Render dialogue rows (TV-style) as Speaker: line\n.
        rendered = []
        for turn in row["turns"]:
            speaker = turn.get("speaker") or "Narration"
            text = turn.get("text") or ""
            if text:
                rendered.append(f"{speaker}: {text}")
        return "\n".join(rendered)
    return ""


def chain_sessions(path: Path) -> list[str]:
    rows = list(iter_jsonl(path))
    rows.sort(
        key=lambda r: int(r.get("session_index", r.get("episode_index", 0)))
    )
    return [render_session(r) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", action="append", type=Path, required=True,
                    help="Repeatable; each is a directory of <chain>.jsonl files.")
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--session_len", type=int, default=512)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--min_sessions_per_chain", type=int, default=2)
    ap.add_argument("--max_chains", type=int, default=None)
    ap.add_argument("--max_sessions_per_chain", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id

    # Gather chain files.  We cap *per-directory* so a TV/PG-19 mix doesn't
    # accidentally drop one source entirely when --max_chains is small.
    chain_files: list[Path] = []
    for d in args.in_dir:
        per_dir = sorted(Path(d).glob("*.jsonl"))
        if args.max_chains is not None:
            per_dir = per_dir[: args.max_chains]
        chain_files.extend(per_dir)
        print(f"  {d}: {len(per_dir)} chains kept (after per-dir cap)")
    print(f"Discovered {len(chain_files)} chain files across {len(args.in_dir)} dirs")

    chain_sessions_list: list[list[str]] = []
    chain_names: list[str] = []
    for path in tqdm(chain_files, desc="loading chains"):
        sessions = chain_sessions(path)
        if args.max_sessions_per_chain is not None:
            sessions = sessions[: args.max_sessions_per_chain]
        if len(sessions) >= args.min_sessions_per_chain:
            chain_sessions_list.append(sessions)
            chain_names.append(path.stem)

    print(
        f"Kept {len(chain_sessions_list)} chains "
        f"({sum(len(c) for c in chain_sessions_list)} sessions total)"
    )

    # Tokenise sessions in parallel, preserving chain order.
    flat_jobs = [
        (text, args.session_len, args.min_tokens, eos_id)
        for sessions in chain_sessions_list
        for text in sessions
    ]
    flat_lengths = [len(c) for c in chain_sessions_list]

    with mp.Pool(
        args.workers, initializer=init_worker, initargs=(args.tokenizer,)
    ) as pool:
        encoded_flat = list(
            tqdm(
                pool.imap(encode_session, flat_jobs, chunksize=64),
                total=len(flat_jobs),
                desc="tokenize sessions",
            )
        )

    # Re-stitch into chains, dropping sessions that tokenized to < min_tokens.
    session_rows: list[list[int]] = []
    session_chain_id: list[int] = []
    session_position: list[int] = []
    chain_starts: list[int] = []
    chain_lengths: list[int] = []

    cursor = 0
    kept_chain_id = 0
    new_chain_names: list[str] = []
    for chain_id, n in enumerate(flat_lengths):
        chain_slice = encoded_flat[cursor : cursor + n]
        cursor += n
        kept = [(p, ids) for p, ids in enumerate(chain_slice) if ids is not None]
        if len(kept) < args.min_sessions_per_chain:
            continue
        start_row = len(session_rows)
        for new_pos, (orig_pos, ids) in enumerate(kept):
            session_rows.append(ids)
            session_chain_id.append(kept_chain_id)
            session_position.append(new_pos)
        chain_starts.append(start_row)
        chain_lengths.append(len(kept))
        new_chain_names.append(chain_names[chain_id])
        kept_chain_id += 1

    if not session_rows:
        raise RuntimeError("No usable chains after tokenisation")

    session_ids = torch.tensor(session_rows, dtype=torch.int32)
    session_chain_id_t = torch.tensor(session_chain_id, dtype=torch.int32)
    session_position_t = torch.tensor(session_position, dtype=torch.int32)
    chain_starts_t = torch.tensor(chain_starts, dtype=torch.int64)
    chain_lengths_t = torch.tensor(chain_lengths, dtype=torch.int64)

    print(
        f"Encoded {session_ids.shape[0]} sessions "
        f"across {len(new_chain_names)} chains -> {session_ids.shape}"
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "session_ids": session_ids,
            "session_chain_id": session_chain_id_t,
            "session_position": session_position_t,
            "chain_starts": chain_starts_t,
            "chain_lengths": chain_lengths_t,
            "chain_names": new_chain_names,
            "session_len": args.session_len,
            "tokenizer": args.tokenizer,
        },
        args.out_path,
    )
    print(f"Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
