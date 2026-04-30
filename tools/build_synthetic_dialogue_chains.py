#!/usr/bin/env python3
"""Download and convert synthetic / roleplay dialogue datasets into the
Memory Residuals chain-JSONL format expected by
``tools/pretokenize_chains.py``.

Layout produced (one file per chain, rows = sessions)::

    <out_dir>/<source>/<source>_<id>.jsonl
        {"session_index": 0, "turns": [{"speaker": "Alice", "text": "..."}, ...]}
        {"session_index": 1, "turns": [...]}
        ...

Each source treats one HF example as a single chain, then chunks the
turn sequence into ``--turns_per_session`` sessions so the recurrent
judge has a non-trivial number of sessions per chain.  Sessions that
would have fewer than ``--min_turns_per_session`` turns are merged
into their predecessor (keeps chains dense).

Supported sources (``--sources`` accepts any subset)::

    ultrachat      : HuggingFaceH4/ultrachat_200k  (~200k 2-party multi-turn)
    pippa          : PygmalionAI/PIPPA             (~20k persona / roleplay)
    soda           : allenai/soda                  (~1.5M synthetic social)
    lmsys          : lmsys/lmsys-chat-1m           (1M real user-assistant, gated)
    oasst1         : OpenAssistant/oasst1          (~80k tree conversations, flattened)
    no_robots     : HuggingFaceH4/no_robots        (~10k high-quality multi-turn)
    narrativeqa    : deepmind/narrativeqa          (long-narrative Q/A -> memory-critical)
    writingprompts : euclaise/writingprompts       (~300k prompt->long-story pairs)
    hh_rlhf        : Anthropic/hh-rlhf             (multi-turn helpful/harmless)

All sources emit the same chain JSONL schema; ``_detect_source`` in
``train_chain.py`` maps the filename prefix to the source tag used by
``--source_weights`` so you can up- or down-weight each stream
individually at training time.  For memory-requiring data, the default
weights upweight ``narrativeqa`` and ``writingprompts`` slightly.

We deliberately do NOT emit a ``session_callback_mask`` - these corpora
do not have annotated callback spans, so ``merge_chain_corpora.py``
will zero-fill the mask for these chains and ``chain_callback_position``
will be -1.  That is the correct regime for generic-LM data in the
mega corpus: the competition / evidence curriculum branches still only
fire on LongMemEval chains (cb_pos >= 0 required), and these chains
contribute contiguous LM gradient that regularises the readout against
over-injection on callback-less distributions.

Typical usage on the GH200 (where disk / bandwidth is plentiful)::

    python tools/build_synthetic_dialogue_chains.py \
        --sources ultrachat pippa soda \
        --out_dir ../memory_residuals_data/mega_stage \
        --max_chains_per_source 40000 \
        --turns_per_session 4 \
        --min_turns_per_session 2

The conversion is cheap relative to pretokenization; expect < 20 min
for ~120k chains on the GH200 with streaming downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Iterator


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)[:64]


def _write_chain(
    out_dir: Path,
    source: str,
    name_prefix: str,
    chain_id: str,
    sessions: list[list[dict]],
) -> bool:
    """Write a single chain as JSONL (one row per session).

    ``source`` is the on-disk subdirectory (one per HF source), while
    ``name_prefix`` is the filename prefix -- which the trainer's
    ``_detect_source`` bucket maps each chain into (e.g. narrativeqa
    goes into the 'synthdlg' bucket because it shares
    ``narrativeqa_<id>.jsonl``-flavoured content with other generic-LM
    dialogue sources). Per-source weight overrides should be provided
    via ``--source_weights`` at train time.

    Returns True on success (>= 2 sessions with >=1 turn each), else
    False and does not write anything.
    """
    kept: list[list[dict]] = []
    for sess in sessions:
        sess = [t for t in sess if t.get("text", "").strip()]
        if sess:
            kept.append(sess)
    if len(kept) < 2:
        return False
    chain_path = out_dir / source / f"{name_prefix}_{_slug(chain_id)}.jsonl"
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    with chain_path.open("w", encoding="utf-8") as f:
        for i, sess in enumerate(kept):
            row = {"session_index": i, "turns": sess}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def _chunk_turns(
    turns: list[dict], turns_per_session: int, min_turns_per_session: int,
) -> list[list[dict]]:
    """Pack a flat dialogue into ``turns_per_session``-sized sessions,
    merging trailing undersized chunks into the previous session."""
    sessions: list[list[dict]] = []
    for i in range(0, len(turns), turns_per_session):
        chunk = turns[i : i + turns_per_session]
        if (
            sessions
            and len(chunk) < min_turns_per_session
        ):
            sessions[-1].extend(chunk)
        else:
            sessions.append(list(chunk))
    return sessions


# ----------------------------------------------------------------------
# Per-source iterators.  Each yields (chain_id, list[turn_dict]).
# ----------------------------------------------------------------------

def iter_ultrachat(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
    )
    seen = 0
    for ex in ds:
        if max_chains is not None and seen >= max_chains:
            return
        turns_raw = ex.get("messages") or []
        turns: list[dict] = []
        for m in turns_raw:
            role = m.get("role", "") or ""
            text = (m.get("content") or "").strip()
            if not text:
                continue
            speaker = "User" if role == "user" else "Assistant"
            turns.append({"speaker": speaker, "text": text})
        if len(turns) < 4:
            continue
        cid = ex.get("prompt_id") or f"ex{seen}"
        yield str(cid), turns
        seen += 1


def iter_pippa(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    """PygmalionAI/PIPPA ships as a dataset-script repo which is no
    longer supported by `datasets` >=3. Bypass `load_dataset` and
    stream the raw JSONL file via `huggingface_hub`."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="PygmalionAI/PIPPA",
            filename="pippa_deduped.jsonl",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  [skip pippa] hub download failed: {e}", flush=True)
        return
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_chains is not None and seen >= max_chains:
                return
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = ex.get("conversation") or []
            if not conv:
                continue
            bot_name = ex.get("bot_name", "Bot") or "Bot"
            turns: list[dict] = []
            for m in conv:
                text = (m.get("message") or "").strip()
                if not text:
                    continue
                is_human = bool(m.get("is_human", False))
                speaker = "User" if is_human else bot_name
                turns.append({"speaker": speaker, "text": text})
            if len(turns) < 4:
                continue
            cid = (
                ex.get("submission_timestamp")
                or ex.get("bot_id")
                or f"ex{seen}"
            )
            yield str(cid), turns
            seen += 1


def iter_soda(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    from datasets import load_dataset
    ds = load_dataset("allenai/soda", split="train", streaming=True)
    seen = 0
    for ex in ds:
        if max_chains is not None and seen >= max_chains:
            return
        speakers = ex.get("speakers") or []
        utterances = ex.get("dialogue") or []
        if len(speakers) != len(utterances) or len(utterances) < 4:
            continue
        turns = [
            {"speaker": str(s), "text": str(u).strip()}
            for s, u in zip(speakers, utterances)
            if str(u).strip()
        ]
        if len(turns) < 4:
            continue
        cid = ex.get("head") or f"ex{seen}"
        yield str(cid)[:32], turns
        seen += 1


def iter_lmsys(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    """lmsys-chat-1m is gated on HF.  If the token isn't present the
    caller will get an error; silently skip if load_dataset can't
    authenticate."""
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    except Exception as e:
        print(f"  [skip lmsys] {e}", flush=True)
        return
    seen = 0
    for ex in ds:
        if max_chains is not None and seen >= max_chains:
            return
        conv = ex.get("conversation") or []
        turns: list[dict] = []
        for m in conv:
            text = (m.get("content") or "").strip()
            if not text:
                continue
            role = m.get("role", "") or ""
            speaker = "User" if role == "user" else "Assistant"
            turns.append({"speaker": speaker, "text": text})
        if len(turns) < 4:
            continue
        cid = ex.get("conversation_id") or f"ex{seen}"
        yield str(cid), turns
        seen += 1


def iter_oasst1(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    """Flatten OASST1 message tree into per-conversation linear chains by
    DFS-walking the top-scoring reply at each node."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    # Group messages by conversation tree (message_tree_id -> list).
    trees: dict[str, list[dict]] = {}
    for ex in ds:
        trees.setdefault(ex["message_tree_id"], []).append(dict(ex))
    seen = 0
    for tree_id, msgs in trees.items():
        if max_chains is not None and seen >= max_chains:
            return
        by_id = {m["message_id"]: m for m in msgs}
        children: dict[str | None, list[dict]] = {}
        for m in msgs:
            children.setdefault(m.get("parent_id"), []).append(m)
        roots = children.get(None, [])
        if not roots:
            continue
        # Walk from the single root following highest-rank child each hop.
        linear = [roots[0]]
        cur = roots[0]
        while children.get(cur["message_id"]):
            kids = sorted(
                children[cur["message_id"]],
                key=lambda c: -(c.get("rank") or 0),
            )
            cur = kids[0]
            linear.append(cur)
        turns = []
        for m in linear:
            role = (m.get("role") or "").lower()
            text = (m.get("text") or "").strip()
            if not text:
                continue
            speaker = "User" if role == "prompter" else "Assistant"
            turns.append({"speaker": speaker, "text": text})
        if len(turns) < 4:
            continue
        yield tree_id, turns
        seen += 1


def iter_no_robots(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
    seen = 0
    for ex in ds:
        if max_chains is not None and seen >= max_chains:
            return
        msgs = ex.get("messages") or []
        turns: list[dict] = []
        for m in msgs:
            text = (m.get("content") or "").strip()
            role = m.get("role", "") or ""
            if not text:
                continue
            if role == "system":
                speaker = "System"
            elif role == "user":
                speaker = "User"
            else:
                speaker = "Assistant"
            turns.append({"speaker": speaker, "text": text})
        if len(turns) < 3:
            continue
        cid = ex.get("prompt_id") or f"ex{seen}"
        yield str(cid), turns
        seen += 1


def iter_narrativeqa(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    """Each NarrativeQA example: (document_summary / text, question, answer).
    We build a chain per document that stacks (document, Q1->A1, Q2->A2, ...)
    so the model has to remember the document to answer later questions."""
    from datasets import load_dataset
    try:
        ds = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
    except Exception as e:
        print(f"  [skip narrativeqa] {e}", flush=True)
        return
    # Group by document id on the fly; streaming means we buffer per-doc
    # in a dict with a small eviction policy.
    by_doc: dict[str, dict] = {}
    seen = 0
    flushed = 0
    MAX_BUFFER = 2000  # cap memory
    for ex in ds:
        if max_chains is not None and flushed >= max_chains:
            return
        doc_id = ex.get("document", {}).get("id") or ex.get("document_id")
        if not doc_id:
            continue
        doc = by_doc.setdefault(doc_id, {
            "summary": ex.get("document", {}).get("summary", {}).get("text")
                       or ex.get("document", {}).get("text") or "",
            "qas": [],
        })
        q = (ex.get("question", {}).get("text") or "").strip()
        a_list = ex.get("answers") or []
        a = ""
        for cand in a_list:
            a = (cand.get("text") or "").strip()
            if a:
                break
        if q and a:
            doc["qas"].append((q, a))
        seen += 1
        # Flush saturated docs.
        if len(by_doc) > MAX_BUFFER:
            # Flush the oldest document with >=2 QA pairs.
            for flush_id, d in list(by_doc.items()):
                if len(d["qas"]) >= 2:
                    turns = []
                    if d["summary"]:
                        turns.append({"speaker": "Narrator",
                                      "text": d["summary"][:4000]})
                    for q, a in d["qas"]:
                        turns.append({"speaker": "User", "text": q})
                        turns.append({"speaker": "Assistant", "text": a})
                    if len(turns) >= 4:
                        yield flush_id, turns
                        flushed += 1
                    del by_doc[flush_id]
                    break
    # Final flush.
    for flush_id, d in by_doc.items():
        if max_chains is not None and flushed >= max_chains:
            break
        if len(d["qas"]) < 2:
            continue
        turns = []
        if d["summary"]:
            turns.append({"speaker": "Narrator", "text": d["summary"][:4000]})
        for q, a in d["qas"]:
            turns.append({"speaker": "User", "text": q})
            turns.append({"speaker": "Assistant", "text": a})
        if len(turns) >= 4:
            yield flush_id, turns
            flushed += 1


def iter_writingprompts(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    """Each WritingPrompts row: prompt + long story. We split the story
    into paragraphs, treat each paragraph as an 'author' turn to produce
    a chain of 6-20 sessions of continuous narrative (classic
    memory-requiring data, analogous to PG-19)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)
    except Exception as e:
        print(f"  [skip writingprompts] {e}", flush=True)
        return
    seen = 0
    for ex in ds:
        if max_chains is not None and seen >= max_chains:
            return
        prompt = (ex.get("prompt") or "").strip()
        story = (ex.get("story") or "").strip()
        if not story:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", story) if p.strip()]
        if len(paragraphs) < 4:
            continue
        turns = [{"speaker": "Prompt", "text": prompt}] if prompt else []
        for p in paragraphs:
            turns.append({"speaker": "Author", "text": p})
        if len(turns) < 4:
            continue
        cid = ex.get("id") or f"ex{seen}"
        yield str(cid), turns
        seen += 1


def iter_hh_rlhf(max_chains: int | None) -> Iterator[tuple[str, list[dict]]]:
    """Anthropic HH-RLHF 'chosen' responses, flattened into multi-turn
    (user/assistant) chains. Conservative multi-turn source; few callbacks
    but trains the writer on dialogue distribution."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    except Exception as e:
        print(f"  [skip hh_rlhf] {e}", flush=True)
        return
    seen = 0
    for ex in ds:
        if max_chains is not None and seen >= max_chains:
            return
        text = ex.get("chosen") or ""
        if not text:
            continue
        turns: list[dict] = []
        for block in re.split(r"\n\n(?=(?:Human:|Assistant:))", text.strip()):
            block = block.strip()
            if block.startswith("Human:"):
                turns.append({"speaker": "User",
                              "text": block[len("Human:"):].strip()})
            elif block.startswith("Assistant:"):
                turns.append({"speaker": "Assistant",
                              "text": block[len("Assistant:"):].strip()})
        if len(turns) < 4:
            continue
        yield f"ex{seen}", turns
        seen += 1


SOURCE_ITERATORS = {
    "ultrachat":      iter_ultrachat,
    "pippa":          iter_pippa,
    "soda":           iter_soda,
    "lmsys":          iter_lmsys,
    "oasst1":         iter_oasst1,
    "no_robots":      iter_no_robots,
    "narrativeqa":    iter_narrativeqa,
    "writingprompts": iter_writingprompts,
    "hh_rlhf":        iter_hh_rlhf,
}


# Each non-canonical prefix in filenames maps to a _detect_source()
# bucket. Defaults below keep names short and prefix-detectable.
SOURCE_FILENAME_PREFIX = {
    "ultrachat":      "ultrachat",
    "pippa":          "pippa",
    "soda":           "soda",
    "lmsys":          "lmsys",
    # these don't have a dedicated bucket; reuse synthdlg catch-all.
    "oasst1":         "synthdlg",
    "no_robots":      "synthdlg",
    "narrativeqa":    "synthdlg",
    "writingprompts": "synthdlg",
    "hh_rlhf":        "synthdlg",
}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources", nargs="+", required=True,
        choices=sorted(SOURCE_ITERATORS),
        help="Which HF sources to download and convert.",
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--turns_per_session", type=int, default=4)
    ap.add_argument("--min_turns_per_session", type=int, default=2)
    ap.add_argument(
        "--max_chains_per_source", type=int, default=40000,
        help="Cap on chains written per source.  None or 0 = unlimited.",
    )
    ap.add_argument(
        "--hf_home", default=None,
        help="Override HF_HOME for the download cache (e.g. a large "
             "SSD).  Defaults to the existing HF_HOME / ~/.cache/huggingface.",
    )
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    cap = None if args.max_chains_per_source in (0, None) else int(
        args.max_chains_per_source
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}

    for source in args.sources:
        prefix = SOURCE_FILENAME_PREFIX.get(source, source)
        written = 0
        try:
            it = SOURCE_ITERATORS[source](cap)
            for cid, turns in it:
                sessions = _chunk_turns(
                    turns, args.turns_per_session, args.min_turns_per_session,
                )
                if _write_chain(args.out_dir, source, prefix, cid, sessions):
                    written += 1
                    if written % 1000 == 0:
                        print(f"  [{source}] {written} chains written", flush=True)
        except Exception as e:
            print(f"[{source}] FAILED mid-stream at {written} chains: "
                  f"{type(e).__name__}: {e}", flush=True)
        totals[source] = written
        print(f"[{source}] DONE: {written} chains -> "
              f"{args.out_dir / source}", flush=True)

    print("\nSummary:")
    for s, n in totals.items():
        print(f"  {s}: {n} chains")


if __name__ == "__main__":
    main()
