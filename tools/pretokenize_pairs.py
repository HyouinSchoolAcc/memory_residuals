#!/usr/bin/env python3
"""Pre-tokenize a (history, current) pair JSONL into a packed tensor file.

Output format (.pt file):
    {
      "history_ids": LongTensor (N, H),
      "current_ids": LongTensor (N, C+1),
      "history_len": int,
      "current_len": int,
      "tokenizer": str,
    }

Tokenization is left-truncated for history (keep the most recent context)
and right-truncated for current.  Pads with eos to fixed length.
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
    global _TOK
    _TOK = AutoTokenizer.from_pretrained(tokenizer_name)


def encode_pair(args: tuple[dict, int, int, int]):
    sample, history_len, current_len, eos_id = args
    tok = _TOK
    history = sample.get("history") or ""
    current = sample.get("current") or ""
    if not history or not current:
        return None
    h_ids = tok.encode(history, add_special_tokens=False)[-history_len:]
    c_ids = tok.encode(current, add_special_tokens=False)[: current_len + 1]
    if len(h_ids) < 16 or len(c_ids) < 2:
        return None
    h_ids = h_ids + [eos_id] * (history_len - len(h_ids))
    c_ids = c_ids + [eos_id] * (current_len + 1 - len(c_ids))
    return (h_ids[:history_len], c_ids[: current_len + 1])


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--history_len", type=int, default=1024)
    parser.add_argument("--current_len", type=int, default=512)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id

    samples = list(iter_jsonl(args.in_path))
    if args.max_rows is not None:
        samples = samples[: args.max_rows]
    print(f"Loaded {len(samples)} samples from {args.in_path}")

    # Build (sample, history_len, current_len, eos_id) tuples.
    work = [(s, args.history_len, args.current_len, eos_id) for s in samples]

    history_buf = []
    current_buf = []
    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        for enc in tqdm(
            pool.imap_unordered(encode_pair, work, chunksize=64),
            total=len(work),
            desc="tokenize",
        ):
            if enc is None:
                continue
            h_ids, c_ids = enc
            history_buf.append(h_ids)
            current_buf.append(c_ids)

    if not history_buf:
        raise RuntimeError("No usable samples after filtering")

    history = torch.tensor(history_buf, dtype=torch.long)
    current = torch.tensor(current_buf, dtype=torch.long)
    print(f"Encoded {history.shape[0]} pairs -> H {history.shape}, C {current.shape}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "history_ids": history,
            "current_ids": current,
            "history_len": args.history_len,
            "current_len": args.current_len,
            "tokenizer": args.tokenizer,
        },
        args.out_path,
    )
    print(f"Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
