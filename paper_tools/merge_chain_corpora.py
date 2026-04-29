#!/usr/bin/env python3
"""Concatenate multiple pretokenised chain corpora into a single .pt file.

Each input must have been produced by ``pretokenize_chains.py`` and have
the schema documented there.  The merged output preserves chain
boundaries and remaps chain ids so they are unique across sources.

Usage::

    python paper_tools/merge_chain_corpora.py \
        --in paper_artifacts/chains/stage1_train_s512.pt \
        --in paper_artifacts/chains/msc_train_s512.pt \
        --out paper_artifacts/chains/stage1_msc_train_s512.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", type=Path, required=True,
                    help="Repeatable; each is a pretokenised chain .pt to merge.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = [torch.load(p, weights_only=False) for p in args.inputs]
    if not parts:
        raise SystemExit("no inputs")

    session_len = parts[0]["session_len"]
    tokenizer = parts[0]["tokenizer"]
    for i, p in enumerate(parts[1:], start=1):
        if p["session_len"] != session_len:
            raise SystemExit(
                f"session_len mismatch: parts[0]={session_len}, parts[{i}]={p['session_len']}"
            )
        if p["tokenizer"] != tokenizer:
            raise SystemExit(
                f"tokenizer mismatch: parts[0]={tokenizer}, parts[{i}]={p['tokenizer']}"
            )

    session_ids: list[torch.Tensor] = []
    session_chain_id: list[torch.Tensor] = []
    session_position: list[torch.Tensor] = []
    chain_starts: list[int] = []
    chain_lengths: list[torch.Tensor] = []
    chain_names: list[str] = []

    row_offset = 0
    chain_offset = 0
    for src_idx, p in enumerate(parts):
        n_rows = int(p["session_ids"].shape[0])
        n_chains = int(p["chain_starts"].shape[0])

        session_ids.append(p["session_ids"])
        session_position.append(p["session_position"])
        cid = p["session_chain_id"].clone().to(torch.int32) + chain_offset
        session_chain_id.append(cid)

        for cs in p["chain_starts"].tolist():
            chain_starts.append(cs + row_offset)
        chain_lengths.append(p["chain_lengths"])
        chain_names.extend(p["chain_names"])

        print(f"  part {src_idx}: {args.inputs[src_idx]} "
              f"({n_chains} chains, {n_rows} sessions)")
        row_offset += n_rows
        chain_offset += n_chains

    merged = {
        "session_ids": torch.cat(session_ids, dim=0),
        "session_chain_id": torch.cat(session_chain_id, dim=0),
        "session_position": torch.cat(session_position, dim=0),
        "chain_starts": torch.tensor(chain_starts, dtype=torch.int64),
        "chain_lengths": torch.cat(chain_lengths, dim=0),
        "chain_names": chain_names,
        "session_len": session_len,
        "tokenizer": tokenizer,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(f"Merged -> {args.out} "
          f"({merged['chain_starts'].shape[0]} chains, "
          f"{merged['session_ids'].shape[0]} sessions)")


if __name__ == "__main__":
    main()
