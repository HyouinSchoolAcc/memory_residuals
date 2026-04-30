#!/usr/bin/env python3
"""Concatenate multiple pretokenised chain corpora into a single .pt file.

Each input must have been produced by ``pretokenize_chains.py`` and have
the schema documented there.  The merged output preserves chain
boundaries and remaps chain ids so they are unique across sources.

Usage::

    python tools/merge_chain_corpora.py \
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
    # Optional callback fields; if any input has them, we propagate them
    # (zero-filling the parts that don't).
    has_callback = any("session_callback_mask" in p for p in parts)
    has_evidence = any("chain_evidence_positions" in p for p in parts)
    session_callback_mask: list[torch.Tensor] = []
    chain_callback_position: list[torch.Tensor] = []
    chain_evidence_positions: list[list[int]] = []

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

        if has_callback:
            if "session_callback_mask" in p:
                session_callback_mask.append(
                    p["session_callback_mask"].to(torch.int8)
                )
            else:
                session_callback_mask.append(
                    torch.zeros_like(p["session_ids"], dtype=torch.int8)
                )
            if "chain_callback_position" in p:
                chain_callback_position.append(
                    p["chain_callback_position"].to(torch.int32)
                )
            else:
                chain_callback_position.append(
                    torch.full((n_chains,), -1, dtype=torch.int32)
                )

        if has_evidence:
            ev = p.get("chain_evidence_positions")
            if ev is None:
                # Sentinel: this part has no evidence labels. Empty list
                # per chain so the sampler falls through to uniform.
                ev = [[] for _ in range(n_chains)]
            chain_evidence_positions.extend(ev)

        print(f"  part {src_idx}: {args.inputs[src_idx]} "
              f"({n_chains} chains, {n_rows} sessions"
              + (f", cb={int((p.get('session_callback_mask', torch.zeros(0))).sum())}"
                 if has_callback else "") + ")")
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
    if has_callback:
        merged["session_callback_mask"] = torch.cat(session_callback_mask, dim=0)
        merged["chain_callback_position"] = torch.cat(chain_callback_position, dim=0)
    if has_evidence:
        merged["chain_evidence_positions"] = chain_evidence_positions

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(f"Merged -> {args.out} "
          f"({merged['chain_starts'].shape[0]} chains, "
          f"{merged['session_ids'].shape[0]} sessions)")


if __name__ == "__main__":
    main()
