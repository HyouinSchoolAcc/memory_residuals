#!/usr/bin/env python3
"""Patch a pretokenized LongMemEval chain blob with chain_evidence_positions.

The pretokenizer dropped the v11 ``meta["evidence_positions"]`` field
when emitting the .pt blob, so the callback evaluator's redact-evidence
floor falls back to a single random-distractor session and
``evidence_lift`` collapses to ~0 nats. This tool rebuilds the field
from the original ``longmemeval_s_cleaned.json`` (which carries
``answer_session_ids`` and ``haystack_session_ids``) and writes a new
blob with ``chain_evidence_positions`` populated.

Mirrors the post-empty-session-drop indexing used in
``tools/build_conversational_callback_chains.py::load_longmemeval``.

Usage:
    python tools/patch_lme_evidence_positions.py \\
        --in_blob paper_artifacts/chains/lme_val_s512.pt \\
        --raw_json /path/to/longmemeval_s_cleaned.json \\
        --out_blob paper_artifacts/chains/lme_val_s512_evpos.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def question_id_from_chain_name(name: str) -> str:
    if name.startswith("longmemeval_"):
        return name[len("longmemeval_") :]
    return name


def evidence_positions_for_sample(sample: dict) -> list[int]:
    """Mirror load_longmemeval: walk haystack, drop empty sessions, mark
    positions of those whose haystack session id is in answer_session_ids."""
    haystack = sample["haystack_sessions"]
    haystack_ids = list(sample.get("haystack_session_ids", []))
    answer_ids = set(sample.get("answer_session_ids", []))
    kept = 0
    positions: list[int] = []
    for raw_idx, sess in enumerate(haystack):
        non_empty = any((turn.get("content") or "").strip() for turn in sess)
        if not non_empty:
            continue
        sid = haystack_ids[raw_idx] if raw_idx < len(haystack_ids) else None
        if sid is not None and sid in answer_ids:
            positions.append(kept)
        kept += 1
    return positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_blob", type=Path, required=True)
    ap.add_argument("--raw_json", type=Path, required=True)
    ap.add_argument("--out_blob", type=Path, required=True)
    a = ap.parse_args()

    blob = torch.load(a.in_blob, map_location="cpu", weights_only=False)
    with a.raw_json.open() as f:
        raw = json.load(f)
    by_qid = {s["question_id"]: s for s in raw}

    chain_names = list(blob["chain_names"])
    chain_lengths = blob["chain_lengths"]
    chain_callback_position = blob["chain_callback_position"]

    chain_evidence_positions: list[list[int]] = []
    n_with_evidence, n_unmatched, n_truncated = 0, 0, 0
    for ci, name in enumerate(chain_names):
        qid = question_id_from_chain_name(name)
        sample = by_qid.get(qid)
        if sample is None:
            chain_evidence_positions.append([])
            n_unmatched += 1
            continue
        positions = evidence_positions_for_sample(sample)
        cb_pos = int(chain_callback_position[ci])
        chain_len = int(chain_lengths[ci])
        # Evidence positions are within the haystack (i.e. < cb_pos);
        # if any landed >= chain_len they were truncated by pretokenize
        # and we drop them.
        in_range = [int(p) for p in positions if 0 <= int(p) < cb_pos and int(p) < chain_len]
        if len(in_range) < len(positions):
            n_truncated += 1
        chain_evidence_positions.append(in_range)
        if in_range:
            n_with_evidence += 1

    blob["chain_evidence_positions"] = chain_evidence_positions
    a.out_blob.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, a.out_blob)

    n = len(chain_names)
    avg = sum(len(p) for p in chain_evidence_positions) / max(1, n)
    print(f"chains:          {n}")
    print(f"with evidence:   {n_with_evidence}")
    print(f"unmatched qid:   {n_unmatched}")
    print(f"truncated rows:  {n_truncated}")
    print(f"avg n_evidence:  {avg:.2f}")
    print(f"saved -> {a.out_blob}")


if __name__ == "__main__":
    main()
