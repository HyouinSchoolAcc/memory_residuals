#!/usr/bin/env python3
"""Build a long-horizon chain corpus with planted passkey + callback supervision.

Takes an existing pre-tokenised chain corpus (e.g. stage1_train_s512.pt
which is PG-19 + TV books) and produces a new corpus where each chain has:

  - At session 0 (the "anchor" session): the last K_anchor tokens of the
    session are overwritten with " Remember the passkey is <PASSKEY>."
    where PASSKEY is a 5-word random token sequence drawn from a fixed
    word pool, unique per chain.
  - At session N-1 (the "callback" session, where N is the chain's
    truncated length): the last K_callback tokens are overwritten with
    " The passkey was <PASSKEY>." with the same PASSKEY.
  - A new tensor `session_callback_mask` of shape (N_sessions, S) marks
    the PASSKEY token positions in the callback session as 1 and
    everything else as 0. The trainer multiplies the loss at those
    positions by (1 + callback_loss_weight) so the model gets dense
    gradient on memory-relevant tokens.

The passkey itself is the only memory-dependent prediction in the chain
because:

  - Without M_c, the model has no context to predict it -- the callback
    sentence " The passkey was" appears at the very end of session N-1
    and the model must look back to session 0 (which was written into
    M_c on the recurrent update boundary) to recover it.
  - The callback mask focuses gradient on those 5 word tokens (not on
    the surrounding boilerplate or the rest of the session), so the
    supervision signal is concentrated where memory matters.

We also build a separate `horizon_eval_*.pt` corpus where the callback
session is varied in {2, 4, 8, 16, 24} to produce a horizon-decay curve
post-training. Each horizon depth lives in a separate "chain" so the
eval can compute Δ_sh-m at fixed depth.

Output schema (extends pretokenize_chains.py):

  {
    "session_ids":          (N_sessions, S) int32,
    "session_callback_mask": (N_sessions, S) int8,   # NEW, 0/1
    "session_chain_id":     (N_sessions,)   int32,
    "session_position":     (N_sessions,)   int32,
    "chain_starts":         (N_chains,)     int64,
    "chain_lengths":        (N_chains,)     int64,
    "chain_callback_position": (N_chains,)  int32,   # NEW, which session has the callback
    "chain_passkey_token_ids": list[list[int]],     # NEW, debug: the actual passkey ids per chain
    "chain_names":          list[str],
    "session_len":          int,
    "tokenizer":            str,
    "passkey_word_pool":    list[str],              # NEW, debug
  }
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

# A small, deterministic pool of common single-token English words.
# Tokenisation under Qwen3 tokenizer: each of these is a single BPE token
# (verified at build time below; we filter out any multi-token entries).
DEFAULT_WORD_POOL = [
    "banana", "apple", "orange", "lemon", "grape", "cherry", "peach", "plum",
    "ocean", "river", "mountain", "valley", "desert", "forest", "meadow", "canyon",
    "purple", "yellow", "silver", "golden", "crimson", "azure", "scarlet", "emerald",
    "tiger", "eagle", "wolf", "falcon", "badger", "otter", "raven", "salmon",
    "violin", "drum", "piano", "guitar", "harp", "flute", "trumpet", "cello",
    "north", "south", "east", "west", "summer", "winter", "spring", "autumn",
    "iron", "copper", "marble", "granite", "amber", "crystal", "velvet", "linen",
    "thunder", "lightning", "shadow", "twilight", "dawn", "dusk", "ember", "frost",
]

# Number of words drawn (without replacement) per chain.
PASSKEY_LEN = 5

# Templates. Tokenised once and used as fixed token sequences.
ANCHOR_TEMPLATE = " Remember the passkey is {passkey}."
CALLBACK_TEMPLATE = " The passkey was {passkey}."


def filter_word_pool(tokenizer, words: list[str]) -> list[str]:
    """Keep only words that BPE-encode to a single token (with leading space).

    This makes the passkey length deterministic in tokens, so the
    callback mask is a fixed-length contiguous span at the end of the
    callback session.
    """
    kept = []
    for w in words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            kept.append(w)
    return kept


def build_passkey_pairs(
    tokenizer,
    passkey_words: list[str],
) -> tuple[list[int], list[int], int]:
    """Return (anchor_token_ids, callback_token_ids, passkey_span_len).

    The callback_token_ids has the form [..., space, w1, w2, w3, w4, w5, "."].
    The passkey span (length PASSKEY_LEN) is the slice
        callback_token_ids[-(1 + PASSKEY_LEN):-1]
    which is what we want the loss to focus on. The trailing "." period
    is excluded so the loss sees the punctuation as standard NLL only.
    """
    passkey_str = " ".join(passkey_words)
    anchor_text = ANCHOR_TEMPLATE.format(passkey=passkey_str)
    callback_text = CALLBACK_TEMPLATE.format(passkey=passkey_str)
    anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
    callback_ids = tokenizer.encode(callback_text, add_special_tokens=False)
    return anchor_ids, callback_ids, PASSKEY_LEN


def overwrite_tail(session: torch.Tensor, replacement_ids: list[int],
                   eos_id: int) -> torch.Tensor:
    """Overwrite the *content tail* of a session with replacement_ids.

    Sessions are right-padded with eos_id to S. We:
      - find content_len = number of non-EOS tokens at the start
      - overwrite session[content_len - len(replacement_ids) : content_len]
        with replacement_ids
      - leave the EOS-padding tail intact
    If content_len < len(replacement_ids), we overwrite from position 0
    and accept that the entire session content is replaced.
    """
    out = session.clone()
    S = out.shape[0]
    is_pad = out == eos_id
    # content_len is the index of the first padding position
    if is_pad.any():
        content_len = int(torch.argmax(is_pad.int()))
    else:
        content_len = S
    rl = len(replacement_ids)
    if content_len <= 0:
        # Empty session -- shouldn't happen post-pretokenisation, but
        # be defensive: write replacement at the start, pad rest.
        out[:rl] = torch.tensor(replacement_ids, dtype=out.dtype)
        out[rl:] = eos_id
        return out
    if rl >= content_len:
        # Replacement is at least as long as the content; clobber the
        # whole content area.
        out[:rl] = torch.tensor(replacement_ids, dtype=out.dtype)
        out[rl:] = eos_id
    else:
        start = content_len - rl
        out[start:content_len] = torch.tensor(replacement_ids, dtype=out.dtype)
    return out


def callback_mask_for_session(
    session: torch.Tensor, callback_ids: list[int],
    passkey_span_len: int, eos_id: int,
) -> torch.Tensor:
    """Build a (S,) int8 mask with 1's at the passkey positions of session.

    Assumes session was just overwritten with callback_ids via overwrite_tail.
    The passkey span occupies the last (1 + passkey_span_len) tokens of
    callback_ids minus the trailing period -- i.e. positions
        [content_len - (1 + passkey_span_len) : content_len - 1]
    in the session (content_len = end of the callback content).
    """
    S = session.shape[0]
    mask = torch.zeros(S, dtype=torch.int8)
    is_pad = session == eos_id
    if is_pad.any():
        content_len = int(torch.argmax(is_pad.int()))
    else:
        content_len = S
    if content_len < passkey_span_len + 1:
        return mask
    end = content_len - 1  # exclude the trailing "."
    start = end - passkey_span_len
    mask[start:end] = 1
    return mask


def build_chain(
    sessions: torch.Tensor,
    chain_len: int,
    callback_position: int,
    rng: random.Random,
    word_pool: list[str],
    tokenizer,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Build one long-horizon chain.

    Args:
      sessions: (L, S) original chain sessions, L >= chain_len.
      chain_len: # sessions to keep (slicing 0..chain_len-1).
      callback_position: index in [1, chain_len-1] where the callback goes.
      rng: per-chain RNG for passkey selection.
      word_pool: filtered single-token words.

    Returns:
      session_ids: (chain_len, S) int32
      callback_mask: (chain_len, S) int8
      passkey_words: the 5 words drawn for this chain
    """
    if chain_len > sessions.shape[0]:
        raise ValueError(
            f"chain_len={chain_len} > available sessions={sessions.shape[0]}"
        )
    if callback_position <= 0 or callback_position >= chain_len:
        raise ValueError(
            f"callback_position must be in [1, chain_len-1]; "
            f"got {callback_position} for chain_len={chain_len}"
        )

    # Slice and clone so we don't mutate the source corpus tensor.
    chain_ids = sessions[:chain_len].clone()

    # Draw passkey
    passkey_words = rng.sample(word_pool, PASSKEY_LEN)
    anchor_ids, callback_ids, passkey_span_len = build_passkey_pairs(
        tokenizer, passkey_words
    )

    # Overwrite session 0 with anchor
    chain_ids[0] = overwrite_tail(chain_ids[0], anchor_ids, eos_id)

    # Overwrite the callback session
    chain_ids[callback_position] = overwrite_tail(
        chain_ids[callback_position], callback_ids, eos_id
    )

    # Build mask
    callback_mask = torch.zeros_like(chain_ids, dtype=torch.int8)
    callback_mask[callback_position] = callback_mask_for_session(
        chain_ids[callback_position], callback_ids, passkey_span_len, eos_id
    )
    return chain_ids, callback_mask, passkey_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=Path, required=True,
                    help="Existing pretokenised chain .pt to use as base.")
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument("--chain_len", type=int, default=16,
                    help="Number of sessions per output chain.")
    ap.add_argument("--callback_position", type=int, default=None,
                    help="Fixed callback session index. If None, randomised "
                         "in [chain_len//2, chain_len-1] per chain.")
    ap.add_argument("--callback_position_set", type=str, default=None,
                    help="Comma-separated callback positions to fan out per "
                         "input chain (one output chain per position). "
                         "E.g. '2,4,8,16'. Mutually exclusive with "
                         "--callback_position.")
    ap.add_argument("--max_chains", type=int, default=None,
                    help="Cap output chains (after eligibility filter).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    blob = torch.load(args.in_path, map_location="cpu", weights_only=False)
    src_sessions = blob["session_ids"]
    src_starts = blob["chain_starts"]
    src_lengths = blob["chain_lengths"]
    src_names = blob["chain_names"]
    session_len = int(blob["session_len"])
    tokenizer_name = blob["tokenizer"]

    if args.tokenizer != tokenizer_name:
        print(
            f"WARNING: --tokenizer {args.tokenizer!r} differs from corpus "
            f"tokenizer {tokenizer_name!r}; using corpus tokenizer."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = int(tokenizer.eos_token_id)

    word_pool = filter_word_pool(tokenizer, DEFAULT_WORD_POOL)
    if len(word_pool) < PASSKEY_LEN * 2:
        raise RuntimeError(
            f"Single-token word pool too small after filtering: "
            f"{len(word_pool)} words, need >= {PASSKEY_LEN * 2}."
        )
    print(f"Word pool: {len(word_pool)} single-token words")

    # Resolve callback position(s)
    if args.callback_position_set is not None and args.callback_position is not None:
        raise ValueError("--callback_position and --callback_position_set are exclusive")
    if args.callback_position_set is not None:
        cb_positions = [int(x) for x in args.callback_position_set.split(",")]
    elif args.callback_position is not None:
        cb_positions = [args.callback_position]
    else:
        cb_positions = None  # signal: randomise per chain

    # Iterate source chains, keep those with chain_lengths >= args.chain_len
    out_session_ids: list[torch.Tensor] = []
    out_callback_masks: list[torch.Tensor] = []
    out_chain_starts: list[int] = []
    out_chain_lengths: list[int] = []
    out_chain_names: list[str] = []
    out_callback_positions: list[int] = []
    out_passkey_token_ids: list[list[int]] = []
    out_session_chain_id: list[int] = []
    out_session_position: list[int] = []

    cursor = 0
    n_eligible = 0
    n_kept = 0
    for ci in range(len(src_starts)):
        L = int(src_lengths[ci])
        if L < args.chain_len:
            continue
        n_eligible += 1
        s = int(src_starts[ci])
        chain_sessions = src_sessions[s : s + L]
        # If max_chains set, randomly subsample (deterministic via seed)
        if args.max_chains is not None and n_kept >= args.max_chains:
            # Continue counting eligible but don't emit
            continue

        # Resolve callback positions for this chain
        if cb_positions is None:
            cb_pos = rng.randint(args.chain_len // 2, args.chain_len - 1)
            chain_cb_positions = [cb_pos]
        else:
            chain_cb_positions = [
                p for p in cb_positions if p < args.chain_len
            ]
            if not chain_cb_positions:
                continue

        for cb_pos in chain_cb_positions:
            chain_ids, mask, passkey_words = build_chain(
                chain_sessions, args.chain_len, cb_pos, rng,
                word_pool, tokenizer, eos_id,
            )
            out_session_ids.append(chain_ids)
            out_callback_masks.append(mask)
            out_chain_starts.append(cursor)
            out_chain_lengths.append(args.chain_len)
            base_name = src_names[ci]
            out_chain_names.append(f"longhorizon_{base_name}_cb{cb_pos}")
            out_callback_positions.append(cb_pos)
            passkey_str = " " + " ".join(passkey_words)
            out_passkey_token_ids.append(
                tokenizer.encode(passkey_str, add_special_tokens=False)
            )
            for sp in range(args.chain_len):
                out_session_chain_id.append(n_kept)
                out_session_position.append(sp)
            cursor += args.chain_len
            n_kept += 1

    print(
        f"Source chains: {len(src_starts)} | "
        f"eligible (>= {args.chain_len} sessions): {n_eligible} | "
        f"emitted: {n_kept}"
    )

    if not out_session_ids:
        raise RuntimeError("No chains emitted; check --chain_len vs source corpus")

    session_ids = torch.cat([s.unsqueeze(0) if s.dim() == 1 else s for s in out_session_ids], dim=0)
    callback_mask = torch.cat([m.unsqueeze(0) if m.dim() == 1 else m for m in out_callback_masks], dim=0)
    session_ids = session_ids.to(torch.int32)
    callback_mask = callback_mask.to(torch.int8)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "session_ids": session_ids,
            "session_callback_mask": callback_mask,
            "session_chain_id": torch.tensor(out_session_chain_id, dtype=torch.int32),
            "session_position": torch.tensor(out_session_position, dtype=torch.int32),
            "chain_starts": torch.tensor(out_chain_starts, dtype=torch.int64),
            "chain_lengths": torch.tensor(out_chain_lengths, dtype=torch.int64),
            "chain_callback_position": torch.tensor(out_callback_positions, dtype=torch.int32),
            "chain_passkey_token_ids": out_passkey_token_ids,
            "chain_names": out_chain_names,
            "session_len": session_len,
            "tokenizer": tokenizer_name,
            "passkey_word_pool": word_pool,
            "passkey_len_tokens": PASSKEY_LEN,
        },
        args.out_path,
    )
    print(f"Saved -> {args.out_path}")
    print(
        f"  shape: session_ids {tuple(session_ids.shape)}, "
        f"callback_mask {tuple(callback_mask.shape)}, "
        f"non-zero mask positions: {int(callback_mask.sum())}"
    )


if __name__ == "__main__":
    main()
