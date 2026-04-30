#!/usr/bin/env python3
"""Build a conversational long-horizon chain corpus with callback supervision.

This is the v6 corpus builder. It replaces the earlier
build_longhorizon_chains.py (synthetic passkey on PG-19), which was the
wrong domain (books, not conversation) for this paper's claim
("conversational memory recall"). The user's note: "books might not be
fit for a conversational dataset; I'm still highly speculative on its
usefulness in providing a conversationally rich recall method."

We support three source formats:

  --source msc          : per-chain JSONL files at $MSC_DIR/{train,val}/msc_*.jsonl
                          Each file is a sequence of session rows (already
                          our chain format), with persona facts in the
                          System turn of session 0. We append a synthetic
                          callback Q+A turn at the end of the last session
                          asking about a persona fact; the answer tokens
                          are flagged as callback span. Each chain remains
                          5 sessions long; window_k=5 sees the entire chain.

  --source longmemeval  : a single longmemeval_oracle.json or
                          longmemeval_s_cleaned.json. Each sample has
                          haystack_sessions (list of sessions, each a list
                          of turns with role/content/has_answer) plus an
                          explicit question and answer. We render the
                          chat history as N sessions, then append one
                          synthetic "User asks `question` / Assistant
                          answers `answer`" session at the end. The answer
                          tokens are the callback span. Chain length is
                          (N + 1).

  --source realtalk     : the danny911kr/REALTALK GitHub repo's data dir
                          with Chat_*_X_Y.json files. Each chat has up to
                          36 sessions of real human-human messaging. We
                          do not synthesise callbacks here (the dataset
                          provides its own qa annotations); REALTALK is
                          held out as eval.

Output format (extends pretokenize_chains.py with a callback mask):

  {
    "session_ids":              (N_sessions, S) int32
    "session_callback_mask":    (N_sessions, S) int8     # 0/1
    "session_chain_id":         (N_sessions,) int32
    "session_position":         (N_sessions,) int32
    "chain_starts":             (N_chains,)  int64
    "chain_lengths":            (N_chains,)  int64
    "chain_names":              list[str]
    "chain_callback_position":  (N_chains,)  int32      # which session has the callback
    "chain_evidence_positions": list[list[int]]         # per-chain list of session
                                                        # indices that actually contain
                                                        # the answer (LongMemEval only;
                                                        # empty list for sources w/o
                                                        # ground-truth labels). v11+
    "chain_meta":               list[dict]              # debug: question, answer, persona-fact
    "session_len":              int (S)
    "tokenizer":                str
    "source":                   str (msc | longmemeval | realtalk | merged)
  }

v11 additions (2026-04-30, post-v10-audit):
  - LongMemEval ships per-sample ``answer_session_ids`` (list of haystack
    session IDs that actually contain the answer text). Earlier corpus
    builders silently dropped this field. We now preserve it as a list
    of *position indices within the haystack* so downstream training
    can use it for evidence-aware curriculum sampling. Without this
    fix, ``ChainSampler.sample_window`` picks "evidence" uniformly from
    [0, cb_pos), where only ~3.6% of sessions actually contain the
    answer -- 96%+ of training samples then build M_c from irrelevant
    sessions, and the LM loss correctly converges to "ignore memory".
  - For sources without per-chain evidence labels (MSC / PG-19 / TV /
    REALTALK / synthdlg) we emit an empty list so the sampler falls
    through to its existing uniform branch.

Notes on tokenisation:
  - We tokenise each "chunk" (a turn or persona prefix or callback span)
    separately and concatenate the resulting id lists, so the callback
    mask is a contiguous, exactly-aligned span in token space.
  - Sessions are right-truncated to session_len and right-padded with EOS.
  - Callback spans that get truncated (because the session content
    overflowed S) are dropped (mask is empty for that session).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------
# A "chunk" is (text, is_callback). Sessions are lists of chunks; chains are
# lists of sessions. The builder is purely lexical until the final tokenise
# step.
Chunk = tuple[str, bool]
Session = list[Chunk]
Chain = list[Session]


# ---------------------------------------------------------------------------
# MSC loader
# ---------------------------------------------------------------------------

# Matches each persona fact line ("I/My ..." pattern). MSC's System turn:
#   "Speaker 1 persona: I'm in school. I have a cat. ...
#    Speaker 2 persona: I love photography. ..."
# We split on sentence boundaries within each persona to get individual facts.
_PERSONA_PREFIX_RE = re.compile(
    r"^Speaker\s+(\d+)\s+persona:\s*(.*?)$",
    flags=re.MULTILINE | re.DOTALL,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_msc_personas(system_text: str) -> dict[str, list[str]]:
    """Return {speaker_label: [persona_facts]} from the System persona prefix."""
    out: dict[str, list[str]] = {}
    # Split into Speaker 1 / Speaker 2 sections by walking the regex.
    sections = re.split(r"(Speaker\s+\d+\s+persona:)", system_text)
    # sections is like ["", "Speaker 1 persona:", "I'm in school. ...",
    #                   "Speaker 2 persona:", "I love photography. ..."]
    cur_label = None
    cur_text_parts: list[str] = []
    for chunk in sections:
        m = re.match(r"Speaker\s+(\d+)\s+persona:", chunk.strip())
        if m:
            if cur_label is not None:
                out[cur_label] = _split_persona_facts(" ".join(cur_text_parts))
            cur_label = f"Speaker {m.group(1)}"
            cur_text_parts = []
        else:
            cur_text_parts.append(chunk.strip())
    if cur_label is not None:
        out[cur_label] = _split_persona_facts(" ".join(cur_text_parts))
    return out


def _split_persona_facts(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]
    return [s for s in sents if len(s) >= 8 and len(s) <= 200]


# Question templates for MSC callback. The choice depends on the fact's
# leading words. We only emit a callback if the fact has a clear topical
# anchor (job/family/hobby/like/work/live/study). We DO NOT make up facts;
# the answer is always the persona fact verbatim, so the callback is a
# faithful retrieval target.
def _msc_callback_template(speaker: str, fact: str) -> tuple[str, str] | None:
    """Return (question, answer) for one persona fact, or None if no template fits.

    The "answer" is the persona fact restated; the trainer will mark
    those answer tokens as the callback span.
    """
    fact = fact.strip().rstrip(".")
    if not fact:
        return None
    fl = fact.lower()
    # Map the fact's initial pattern to a question. Keep the answer
    # verbatim so the model is rewarded for *recalling* the fact, not
    # paraphrasing it.
    if fl.startswith("i'm a ") or fl.startswith("i am a "):
        q = f"By the way, what do you do for a living again, {speaker}?"
    elif fl.startswith("i work as ") or fl.startswith("i work at "):
        q = f"Remind me what you do, {speaker}?"
    elif fl.startswith("i live in ") or fl.startswith("i'm from "):
        q = f"Where did you say you live, {speaker}?"
    elif fl.startswith("i love ") or fl.startswith("i like ") or fl.startswith("i enjoy "):
        q = f"What do you enjoy doing in your free time, {speaker}?"
    elif fl.startswith("i have a ") and ("cat" in fl or "dog" in fl or "pet" in fl):
        q = f"Tell me again about your pet, {speaker}?"
    elif fl.startswith("i have ") and ("daughter" in fl or "son" in fl or "child" in fl):
        q = f"How is your family doing, {speaker}?"
    elif fl.startswith("my favorite ") or fl.startswith("my favourite "):
        q = f"What was your favourite thing again, {speaker}?"
    elif fl.startswith("i'm in school") or fl.startswith("i am in school") or fl.startswith("i study "):
        q = f"What are you studying again, {speaker}?"
    else:
        # Generic fallback only if the fact is short (otherwise too vague).
        if len(fact) > 80:
            return None
        q = f"Remind me, {speaker}, what was that thing you mentioned about yourself?"
    return q, fact + "."


def load_msc_chain_file(path: Path, rng: random.Random,
                       max_callbacks_per_chain: int = 1) -> Chain | None:
    """Convert one MSC chain JSONL into our (sessions of chunks) representation.

    Returns None if the chain has no usable persona facts to test.
    """
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    rows.sort(key=lambda r: int(r.get("session_index", 0)))
    if len(rows) < 2:
        return None
    # Find persona facts from session 0's System turn.
    sys_text = ""
    for t in rows[0].get("turns", []):
        if t.get("speaker") == "System":
            sys_text = t.get("text", "") or ""
            break
    persona_map = parse_msc_personas(sys_text) if sys_text else {}
    # Pick a persona fact for the callback. Prefer Speaker 1 (asker) so
    # the assistant is asking Speaker 1 / Speaker 2 about themselves
    # in either direction. Try a few facts until one matches a template.
    callback_pair: tuple[str, str] | None = None
    callback_speaker_label: str | None = None
    candidates: list[tuple[str, str]] = []
    for spk_label, facts in persona_map.items():
        for f in facts:
            candidates.append((spk_label, f))
    rng.shuffle(candidates)
    for spk_label, f in candidates:
        cb = _msc_callback_template(spk_label, f)
        if cb is not None:
            callback_pair = cb
            callback_speaker_label = spk_label
            break
    if callback_pair is None:
        return None  # no usable persona fact; skip this chain
    cb_question, cb_answer = callback_pair

    # Render sessions as lists of chunks. Speaker labels and turn separators
    # are preserved verbatim.
    chain: Chain = []
    for sess_row in rows:
        session: Session = []
        for turn in sess_row.get("turns", []):
            spk = (turn.get("speaker") or "").strip() or "Speaker"
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            if spk == "System":
                session.append((text + "\n", False))
            else:
                session.append((f"{spk}: {text}\n", False))
        chain.append(session)

    # Append the callback Q+A as the final two turns of the last session.
    # Question speaker = the OTHER speaker (the one asking), answer speaker
    # = the persona owner (the one being asked about themselves).
    asker = "Speaker 1" if callback_speaker_label == "Speaker 2" else "Speaker 2"
    answerer = callback_speaker_label
    last_session = chain[-1]
    last_session.append((f"{asker}: {cb_question}\n", False))
    # Split the answer turn into prefix (boilerplate) + callback span (the
    # actual fact). The fact tokens are the only ones we want to upweight.
    answer_prefix = f"{answerer}: "
    last_session.append((answer_prefix, False))
    last_session.append((cb_answer + "\n", True))

    return chain


# ---------------------------------------------------------------------------
# LongMemEval loader
# ---------------------------------------------------------------------------

def load_longmemeval(path: Path, rng: random.Random,
                     max_chains: int | None = None) -> Iterable[tuple[str, Chain, dict]]:
    """Yield (chain_name, chain, meta) for each LongMemEval sample.

    v11 (2026-04-30): preserves ``answer_session_ids`` as
    ``meta["evidence_positions"]`` -- a list of *post-filter haystack
    indices* (after the empty-session drop loop) of sessions whose ID
    is in ``answer_session_ids``. Empty sessions (every turn was
    whitespace) are skipped during the haystack walk, so we have to
    track the index into the *kept* session list rather than the raw
    haystack index.

    Additionally marks per-turn ``has_answer`` tokens *inside the
    haystack* with cb=True chunk flag. These tokens are the actual
    answer text in their original position, not the synthesized
    callback span -- when the trainer applies callback_loss_weight
    to them, the writer is incentivised to compactly preserve the
    answer span as it streams past, which is the inductive bias we
    want for episodic memory. The synthesized callback at the end
    keeps its own cb=True span (the sole cb on chains where no
    haystack turn has has_answer, which is rare).
    """
    with path.open() as f:
        data = json.load(f)
    if max_chains is not None:
        data = data[: max_chains]
    for sample in data:
        qid = sample["question_id"]
        question = sample["question"]
        answer = sample["answer"]
        haystack = sample["haystack_sessions"]
        haystack_ids: list[str] = list(sample.get("haystack_session_ids", []))
        answer_ids: set[str] = set(sample.get("answer_session_ids", []))
        chain: Chain = []
        evidence_positions: list[int] = []
        for raw_idx, sess in enumerate(haystack):
            session: Session = []
            sid = haystack_ids[raw_idx] if raw_idx < len(haystack_ids) else None
            is_evidence_session = sid in answer_ids if sid is not None else False
            for turn in sess:
                role = turn["role"]
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                spk = "User" if role == "user" else "Assistant"
                # If this turn is flagged has_answer, mark it as cb=True so
                # the trainer's callback_loss_weight upweights NLL on the
                # actual answer-bearing tokens *in their original haystack
                # position* (not just the synthesized recall span at the
                # end). This is the v11 P0 fix: the writer must learn to
                # compactly preserve these tokens as it streams past.
                has_answer = bool(turn.get("has_answer"))
                session.append((f"{spk}: {content}\n", has_answer))
            if session:
                # Position within the *kept* session list = the index the
                # downstream trainer will see in the chain tensor.
                if is_evidence_session:
                    evidence_positions.append(len(chain))
                chain.append(session)
        # Append synthetic callback session: question + answer. The answer
        # tokens here are the recall span (always cb=True).
        callback_session: Session = [
            (f"User: {question}\n", False),
            ("Assistant: ", False),
            (str(answer).strip() + "\n", True),
        ]
        chain.append(callback_session)
        yield qid, chain, {
            "question": question,
            "answer": answer,
            "question_type": sample.get("question_type"),
            # v11: Position indices in the post-filter chain (NOT the raw
            # haystack indices) of sessions that actually contain the
            # answer. The synthesized callback session lives at chain[-1]
            # and is NOT in this list.
            "evidence_positions": evidence_positions,
        }


# ---------------------------------------------------------------------------
# REALTALK loader (eval-only; no callback supervision in train mode)
# ---------------------------------------------------------------------------

def load_realtalk(path: Path) -> Iterable[tuple[str, Chain, dict]]:
    for chat_path in sorted(path.glob("Chat_*.json")):
        with chat_path.open() as f:
            d = json.load(f)
        names = d.get("name", {})
        speaker_map = {
            "speaker_1": names.get("speaker_1", "Speaker1"),
            "speaker_2": names.get("speaker_2", "Speaker2"),
        }
        # Only keep "session_<int>" keys (skip "session_<i>_date_time" string fields).
        session_keys = sorted(
            (k for k in d.keys() if re.fullmatch(r"session_\d+", k)),
            key=lambda s: int(s.split("_")[1]),
        )
        chain: Chain = []
        for sk in session_keys:
            session: Session = []
            for turn in d[sk]:
                spk = turn.get("speaker") or "Speaker"
                text = (turn.get("clean_text") or turn.get("text") or "").strip()
                if not text:
                    continue
                session.append((f"{spk}: {text}\n", False))
            if session:
                chain.append(session)
        if chain:
            yield chat_path.stem, chain, {"speakers": speaker_map}


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenise_session(
    session: Session, tokenizer, session_len: int, eos_id: int,
) -> tuple[list[int], list[int]]:
    """Tokenise one session of chunks; return (ids, mask) of length session_len.

    Right-truncated to session_len; right-padded with eos_id (mask 0).
    Callback tokens that get truncated past session_len are dropped from
    the mask (we cannot supervise tokens we cannot see).
    """
    ids: list[int] = []
    mask: list[int] = []
    for text, is_cb in session:
        chunk_ids = tokenizer.encode(text, add_special_tokens=False)
        for tid in chunk_ids:
            if len(ids) >= session_len:
                break
            ids.append(tid)
            mask.append(1 if is_cb else 0)
        if len(ids) >= session_len:
            break
    while len(ids) < session_len:
        ids.append(eos_id)
        mask.append(0)
    return ids, mask


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["msc", "longmemeval", "realtalk"])
    ap.add_argument("--in_path", required=True, type=Path,
                    help="MSC: directory of msc_*.jsonl files. "
                         "LongMemEval: path to longmemeval_*.json. "
                         "REALTALK: path to data/ directory.")
    ap.add_argument("--out_path", required=True, type=Path)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--session_len", type=int, default=512)
    ap.add_argument("--min_sessions", type=int, default=2,
                    help="Drop chains with fewer sessions than this AFTER "
                         "appending the synthetic callback (so for MSC "
                         "with 5 source sessions + 1 callback append, "
                         "min_sessions=5 keeps everything).")
    ap.add_argument("--max_chains", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = int(tokenizer.eos_token_id)

    # Iterate (chain_name, chain_chunks, meta) over the source.
    if args.source == "msc":
        files = sorted(args.in_path.glob("*.jsonl"))
        if args.max_chains is not None:
            files = files[: args.max_chains]
        chains_iter = ((f.stem, load_msc_chain_file(f, rng), {}) for f in files)
    elif args.source == "longmemeval":
        chains_iter = load_longmemeval(args.in_path, rng, args.max_chains)
    elif args.source == "realtalk":
        chains_iter = load_realtalk(args.in_path)
    else:
        raise ValueError(args.source)

    out_session_ids: list[list[int]] = []
    out_callback_mask: list[list[int]] = []
    out_session_chain_id: list[int] = []
    out_session_position: list[int] = []
    out_chain_starts: list[int] = []
    out_chain_lengths: list[int] = []
    out_chain_names: list[str] = []
    out_chain_callback_position: list[int] = []
    out_chain_evidence_positions: list[list[int]] = []  # v11
    out_chain_meta: list[dict] = []

    n_in = 0
    n_skipped_short = 0
    n_skipped_no_cb_tokens = 0
    n_kept = 0
    cursor = 0

    pbar = tqdm(chains_iter, desc=f"build {args.source}")
    for name, chain, meta in pbar:
        n_in += 1
        if chain is None or len(chain) < args.min_sessions:
            n_skipped_short += 1
            continue
        # Tokenise each session. If a chain has any callback chunk that
        # gets entirely truncated, we drop the chain (no supervision).
        sess_ids_list = []
        sess_mask_list = []
        for sess in chain:
            ids, mask = tokenise_session(sess, tokenizer, args.session_len, eos_id)
            sess_ids_list.append(ids)
            sess_mask_list.append(mask)
        # Locate callback session (the LAST session that has any cb=1 token).
        cb_position = -1
        for si in range(len(sess_mask_list) - 1, -1, -1):
            if any(sess_mask_list[si]):
                cb_position = si
                break
        # For sources that synthesise callbacks (msc, longmemeval), require
        # the callback span to survive truncation. For realtalk (no cb),
        # accept any chain with min_sessions.
        if args.source in ("msc", "longmemeval"):
            if cb_position == -1 or sum(sess_mask_list[cb_position]) == 0:
                n_skipped_no_cb_tokens += 1
                continue
        # v11: Drop evidence positions whose answer span got entirely
        # truncated past session_len (the callback mask check above only
        # protects the synthesized recall span; if a haystack
        # has_answer turn is *all* past S we should not pretend it
        # bears evidence in the M_c the writer can build).
        # Conservatively keep the positions even when the intra-session
        # cb mask got truncated -- the session may still contain the
        # answer-relevant context that *would* have been marked, just
        # not the exact answer tokens. We will revisit if this proves
        # an issue.
        evidence_positions_meta: list[int] = list(meta.get("evidence_positions") or [])
        # Filter out any evidence position that's >= cb_pos (the
        # synthesized callback session is appended at the end; no
        # haystack evidence should ever land there but defend anyway)
        # AND any position outside the kept-session range.
        evidence_positions_meta = [
            p for p in evidence_positions_meta
            if 0 <= p < cb_position
        ]
        out_chain_starts.append(cursor)
        out_chain_lengths.append(len(chain))
        out_chain_names.append(f"{args.source}_{name}")
        out_chain_callback_position.append(cb_position)
        out_chain_evidence_positions.append(evidence_positions_meta)
        out_chain_meta.append(meta)
        for sp, (ids, mask) in enumerate(zip(sess_ids_list, sess_mask_list)):
            out_session_ids.append(ids)
            out_callback_mask.append(mask)
            out_session_chain_id.append(n_kept)
            out_session_position.append(sp)
        cursor += len(chain)
        n_kept += 1
        if n_kept % 200 == 0:
            pbar.set_postfix(kept=n_kept, skipped=n_skipped_short + n_skipped_no_cb_tokens)

    print(
        f"\n[{args.source}] total={n_in} kept={n_kept} "
        f"skipped_short={n_skipped_short} skipped_no_cb={n_skipped_no_cb_tokens}"
    )
    if not out_session_ids:
        raise RuntimeError("No usable chains")

    session_ids = torch.tensor(out_session_ids, dtype=torch.int32)
    callback_mask = torch.tensor(out_callback_mask, dtype=torch.int8)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "session_ids": session_ids,
            "session_callback_mask": callback_mask,
            "session_chain_id": torch.tensor(out_session_chain_id, dtype=torch.int32),
            "session_position": torch.tensor(out_session_position, dtype=torch.int32),
            "chain_starts": torch.tensor(out_chain_starts, dtype=torch.int64),
            "chain_lengths": torch.tensor(out_chain_lengths, dtype=torch.int64),
            "chain_callback_position": torch.tensor(out_chain_callback_position, dtype=torch.int32),
            # v11: list[list[int]] of evidence-session indices per chain
            # (empty list for sources with no ground-truth labels).
            "chain_evidence_positions": out_chain_evidence_positions,
            "chain_meta": out_chain_meta,
            "chain_names": out_chain_names,
            "session_len": args.session_len,
            "tokenizer": args.tokenizer,
            "source": args.source,
        },
        args.out_path,
    )
    n_with_ev = sum(1 for ep in out_chain_evidence_positions if ep)
    n_total_ev = sum(len(ep) for ep in out_chain_evidence_positions)
    print(f"Saved -> {args.out_path}")
    print(
        f"  session_ids={tuple(session_ids.shape)}, "
        f"callback_mask non-zero positions={int(callback_mask.sum())}, "
        f"chains_with_evidence_labels={n_with_ev}/{len(out_chain_evidence_positions)}, "
        f"mean_evidence_positions={n_total_ev / max(1, n_with_ev):.2f}"
    )


if __name__ == "__main__":
    main()
