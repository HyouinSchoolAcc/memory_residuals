#!/usr/bin/env python3
"""Synthetic random-code corpus: D5 in the diagnostic harness.

This corpus is designed to make `evidence_lift` (and not `pa_cb_dnm`) the
load-bearing diagnostic. The D4v2 corpus that v14/v15 trained on has a
32-item-per-category closed answer set with the binding template
`Your favorite {category} is {item}` echoed in the assistant turn of the
evidence session. The full v14k+v15 ledger shows that under that
corpus the writer/readout can win `pa_cb_dnm` by encoding the dataset's
per-category marginal as a content-blind prior tensor; the corresponding
`pa_cb_dnm_floor` (M_c built from a non-evidence filler session) carries
~70-95% of the apparent "memory benefit", and `pa_cb_evidence_lift` -- the
gap between memory-with-evidence and memory-without-evidence -- has been
0.05-0.15 nats since v14g. That is not memory retrieval; it is a learnt
output-side prior. See ``results/exp2_chain_recipe/runs.md`` v16 entry.

D5's design forces evidence retrieval to be the only viable strategy:

  * **Per-chain unique random alphanumeric IDs** (default: 5 character
    `[A-Z0-9]`, ~60M unique strings) replace the 32-item closed set as
    the answer span. Across 5000 training chains x 2 evidence facts the
    expected ID collision rate is <1e-3, so no chain's ID is recoverable
    from a per-category dataset marginal. The LM-loss-optimal policy
    when memory is uninformative is (per-token) uniform over the BPE
    vocabulary, which is ~10 nats per ID-character. Memory retrieval,
    if it works, drives this to ~0 nats. The headroom for actual
    retrieval is therefore ~10x the D4v2 headroom.
  * **No assistant echo of the binding inside the evidence session.**
    D4v2's evidence text was
        "User: My favorite color is red.
         Assistant: Got it, your favorite color is red. I'll remember that."
    -- so the LM head was directly supervised on the binding template
    inside the evidence session, where the answer was trivially
    predictable from the user turn 1-2 sentences earlier. D5's evidence
    is
        "User: my new locker code is K7M2X.
         Assistant: Got it, I'll remember that."
    -- the assistant acknowledges without echoing the code, so the only
    place in the chain that the binding [category->code] is required by
    the LM loss is the *callback* session's answer span. Combined with
    high-entropy IDs this means the LM head cannot learn a chain-blind
    template that helps the callback (because there is no chain-blind
    template to learn).
  * **Multiple distinct categories per chain** so the readout has to
    discriminate which slot to retrieve from at callback time (the
    callback asks about exactly one of the 2 evidence facts). This is
    inherited from D4v2 v15 (n_evidence_sessions=2).

Output schema is identical to ``synthd4v2_persona_callback_*.pt`` so the
chain trainer / phase-aligned eval / eval_callback consume it without
changes:

    session_ids                    (n_sessions, S)  long
    session_callback_mask          (n_sessions, S)  int8
    chain_starts                   (n_chains,)      long
    chain_lengths                  (n_chains,)      long
    chain_callback_position        (n_chains,)      long
    chain_evidence_positions       list[list[int]]  per-chain ev positions
    chain_names                    list[str]
    session_len                    int
    tokenizer                      str

Two new fields are added (used only by the v16 trainer flag
``--mask_evidence_session_loss`` and by audit tooling):

    session_evidence_mask          (n_sessions, S)  int8
        1 on tokens of the EVIDENCE-session ID span; 0 elsewhere.
        Lets the trainer mask the binding token from LM loss on the
        evidence session if requested, so the LM head cannot be
        directly supervised on the random ID.
    session_role                   (n_sessions,)    int8
        0 = filler, 1 = evidence, 2 = callback. Trainer-side hooks can
        use this to e.g. fully zero LM loss on evidence sessions for the
        cleanest "memory is the only path" stress test.

Why this is the right diagnostic for the vision:
  * The PDF's framing -- "compress past sessions into M_c, query during
    the live session via depth-wise routing" -- predicts pa_cb_dnm to be
    dominated by evidence_lift, not by pa_cb_dnm_floor.
  * Under v15 D4v2 the prediction empirically inverts. Under D5, if the
    memory pathway is doing its job, the prediction should hold.
  * If the architecture *still* shows evidence_lift << pa_cb_dnm_floor on
    D5, then the writer/readout pathway has a deeper problem that
    high-entropy answers expose, and the position paper's vision needs
    to be empirically retrenched, not just rephrased.
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import torch
from transformers import AutoTokenizer


# 8 fact categories with naturalistic phrasings. Each renders to:
#   "User: <user_phrasing> {code}. \nAssistant: Got it, I'll remember that."
# The callback uses:
#   "User: Quick question, <callback_phrasing>?\nAssistant: <answer_phrasing> {code}."
# The category cue (e.g. "locker code") is identical across user_phrasing
# and answer_phrasing so the readout can in principle discriminate which
# evidence fact the callback is asking about (and so the in-trainer
# `chain_evidence_positions` semantics carry over from D4v2).
CATEGORIES: dict[str, dict[str, str]] = {
    "locker_code": {
        "user": "My new locker code is",
        "callback_q": "what's my locker code again",
        "answer": "Your locker code is",
    },
    "pin": {
        "user": "I set my new PIN to",
        "callback_q": "what was my new PIN",
        "answer": "Your PIN is",
    },
    "employee_id": {
        "user": "They issued me an employee ID,",
        "callback_q": "what's my employee ID",
        "answer": "Your employee ID is",
    },
    "apartment": {
        "user": "I just moved into apartment",
        "callback_q": "what apartment did I move into",
        "answer": "You moved into apartment",
    },
    "flight": {
        "user": "I'm booked on flight",
        "callback_q": "what flight am I on",
        "answer": "You're on flight",
    },
    "confirmation": {
        "user": "My booking confirmation number is",
        "callback_q": "what was my confirmation number",
        "answer": "Your confirmation number is",
    },
    "tracking": {
        "user": "The package tracking number is",
        "callback_q": "what's the package tracking number",
        "answer": "The tracking number is",
    },
    "voucher": {
        "user": "The voucher code I got is",
        "callback_q": "what's the voucher code",
        "answer": "The voucher code is",
    },
}


# Same 50 generic fillers as build_synthetic_persona_callback (D4v2). Care
# was taken in the original list to avoid mentioning any closed-set items;
# these are equally safe for D5 (they don't mention any random alphanumeric
# strings either, which is the new constraint).
FILLERS: list[str] = [
    "User: How was your day?\nAssistant: It was pretty quiet, mostly catching up on emails.",
    "User: Got any plans for the weekend?\nAssistant: Probably staying in. You?",
    "User: The weather has been wild this week.\nAssistant: I know, I keep forgetting to bring an umbrella.",
    "User: I think I'm coming down with something.\nAssistant: Take it easy, get some rest.",
    "User: Any good news today?\nAssistant: Nothing major, just the usual ups and downs.",
    "User: Did you watch the game last night?\nAssistant: Caught the second half. Pretty intense ending.",
    "User: I'm thinking about taking a class.\nAssistant: That sounds great, what kind?",
    "User: Traffic was terrible this morning.\nAssistant: Yeah, I heard there was construction on the highway.",
    "User: I tried a new recipe yesterday.\nAssistant: How did it turn out?",
    "User: My phone keeps acting up.\nAssistant: Have you tried restarting it?",
    "User: I read an interesting article today.\nAssistant: Oh? What was it about?",
    "User: I'm trying to drink more water.\nAssistant: That's a good goal. I should too.",
    "User: Work has been a lot lately.\nAssistant: Hang in there. It will calm down.",
    "User: I keep forgetting to call my mom back.\nAssistant: Set a reminder, she'll appreciate the call.",
    "User: I miss the long summer evenings.\nAssistant: Same here. The days feel so short now.",
    "User: My neighbor was making noise all night.\nAssistant: That's annoying. Did you say anything?",
    "User: I think I left my keys somewhere.\nAssistant: Retrace your steps, they will turn up.",
    "User: I'm trying to read more this year.\nAssistant: Any genre in particular?",
    "User: My back has been bothering me.\nAssistant: Maybe it is time to switch up your chair.",
    "User: I had the strangest dream last night.\nAssistant: Tell me about it.",
    "User: Have you ever been to the mountains?\nAssistant: Once, years ago. It was beautiful.",
    "User: I keep meaning to start journaling.\nAssistant: Even five minutes a day is plenty.",
    "User: I'm thinking about getting a haircut.\nAssistant: Going for a big change?",
    "User: My commute is wearing me out.\nAssistant: Could you work from home some days?",
    "User: I love the smell of fresh laundry.\nAssistant: Same. It is the small things.",
    "User: I need to do laundry tonight.\nAssistant: Same boat over here.",
    "User: How's your morning going?\nAssistant: Slow, but the coffee is helping.",
    "User: I haven't had time to clean in days.\nAssistant: A quick fifteen minutes can make a big difference.",
    "User: I tried meditating this morning.\nAssistant: How did you find it?",
    "User: I'm going to bed early tonight.\nAssistant: Good idea, sleep is underrated.",
    "User: My back to back meetings ran long.\nAssistant: Hopefully you got a break in there.",
    "User: I think I need new shoes.\nAssistant: They do wear out faster than you'd think.",
    "User: I had a really productive afternoon.\nAssistant: Nice, what did you get done?",
    "User: I'm going to try to be more patient.\nAssistant: That's a worthwhile goal.",
    "User: I keep losing track of my time.\nAssistant: Maybe try blocking your calendar more.",
    "User: I want to plan a small trip soon.\nAssistant: Where are you thinking?",
    "User: I tried to call earlier but missed you.\nAssistant: Sorry, I was on another call.",
    "User: How's the family doing?\nAssistant: Everyone is doing well, thanks for asking.",
    "User: I'm thinking about reorganizing my office.\nAssistant: A fresh layout can help your focus.",
    "User: I had a really good cup of coffee today.\nAssistant: Where from?",
    "User: I forgot to charge my phone last night.\nAssistant: That's a rough way to start the day.",
    "User: I love a quiet evening at home.\nAssistant: Couldn't agree more.",
    "User: I'm trying to be better about replying to emails.\nAssistant: It's a never ending battle.",
    "User: I caught up on some old shows last night.\nAssistant: Anything good?",
    "User: I think I drank too much coffee.\nAssistant: Switch to water for the rest of the day.",
    "User: My printer is on the fritz again.\nAssistant: Always at the worst time, isn't it.",
    "User: I tried a new route to work today.\nAssistant: Any faster?",
    "User: My friend just got a new puppy.\nAssistant: Oh, what kind?",
    "User: I'm thinking about repainting my room.\nAssistant: Fresh paint is always a nice change.",
    "User: I've been feeling pretty motivated this week.\nAssistant: Ride the wave!",
]


_ALPHA = string.ascii_uppercase + string.digits  # 36 symbols


def gen_code(rng: random.Random, length: int = 5) -> str:
    """Generate a random alphanumeric code. Default 5 chars -> ~60M space."""
    return "".join(rng.choice(_ALPHA) for _ in range(length))


def render_evidence(category_phrasings: dict, code: str) -> str:
    """User states the binding once. Assistant ACK without echoing the code.

    This is the load-bearing design choice vs D4v2: by NOT having the
    assistant echo "your locker code is K7M2X" inside the evidence
    session, the LM is never directly supervised on the binding template
    inside that session. The only place the LM head's loss has direct
    pressure to predict the code is the callback session's answer span,
    where the code is not in the local context -- so memory is the only
    pathway that can drive that prediction.
    """
    return (
        f"User: {category_phrasings['user']} {code}.\n"
        f"Assistant: Got it, I'll remember that."
    )


def render_callback(category_phrasings: dict, code: str) -> str:
    return (
        f"User: Quick question, {category_phrasings['callback_q']}?\n"
        f"Assistant: {category_phrasings['answer']} {code}."
    )


def encode_session(tok, text: str, session_len: int, eos_id: int) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)[:session_len]
    if len(ids) < session_len:
        ids = ids + [eos_id] * (session_len - len(ids))
    return ids


def find_code_positions(
    tok,
    text: str,
    code: str,
    session_len: int,
    *,
    last_occurrence: bool,
) -> list[int]:
    """Return BPE-token indices of the code's *occurrence* in the encoded text.

    For evidence text the code appears exactly once (in the user turn).
    For callback text the code appears exactly once (in the assistant
    turn). ``last_occurrence`` controls which one we anchor on if the
    same code appeared multiple times -- always True here for the
    callback (we want the answer span, not any earlier mention).
    """
    full_ids = tok.encode(text, add_special_tokens=False)[:session_len]
    candidates = []
    # Codes are uppercase alphanumeric; the leading-space variant is
    # the typical BPE merge target inside running text.
    for prefix in (" ", ""):
        cand = tok.encode(prefix + code, add_special_tokens=False)
        if cand:
            candidates.append(cand)
    matches: list[list[int]] = []
    for cand in candidates:
        n = len(cand)
        for start in range(len(full_ids) - n + 1):
            if full_ids[start : start + n] == cand:
                matches.append(list(range(start, start + n)))
        if matches:
            break
    if not matches:
        return []
    return matches[-1] if last_occurrence else matches[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_chains", type=int, default=5000)
    p.add_argument("--n_evidence_sessions", type=int, default=2)
    p.add_argument("--n_filler_sessions", type=int, default=7)
    p.add_argument("--n_prefix_sessions", type=int, default=0)
    p.add_argument(
        "--code_len", type=int, default=5,
        help="Length of the random alphanumeric code. 5 chars over a "
             "36-symbol alphabet gives ~60M unique IDs and ~3 BPE tokens "
             "per ID under the Qwen3 tokenizer.",
    )
    p.add_argument("--session_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--audit_tokenisation", action="store_true",
        help="Print per-code token-id lengths for a sample of items, "
             "first chain's evidence/callback masks, then exit.",
    )
    a = p.parse_args()

    if a.n_evidence_sessions < 1:
        raise SystemExit("--n_evidence_sessions must be >= 1.")
    if a.n_evidence_sessions > len(CATEGORIES):
        raise SystemExit(
            f"--n_evidence_sessions={a.n_evidence_sessions} exceeds the "
            f"number of distinct categories ({len(CATEGORIES)})."
        )

    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("tokeniser has no eos_token_id; cannot pad.")
    S = a.session_len
    chain_len = (
        a.n_prefix_sessions + a.n_evidence_sessions + a.n_filler_sessions + 1
    )

    if a.audit_tokenisation:
        print(f"alphabet size: {len(_ALPHA)}; code_len={a.code_len}; "
              f"unique-code space: {len(_ALPHA) ** a.code_len:,}")
        print(f"categories: {list(CATEGORIES.keys())}")
        sample = [gen_code(rng, a.code_len) for _ in range(8)]
        for c in sample:
            ids = tok.encode(" " + c, add_special_tokens=False)
            print(f"  code {c!r}  -> {len(ids)} BPE tokens  {ids}  "
                  f"-> {[tok.decode([i]) for i in ids]}")
        cat_name, cat_p = next(iter(CATEGORIES.items()))
        code = sample[0]
        ev_text = render_evidence(cat_p, code)
        cb_text = render_callback(cat_p, code)
        print(f"\nsample evidence text:\n  {ev_text}")
        print(f"sample callback text:\n  {cb_text}")
        ev_pos = find_code_positions(tok, ev_text, code, S, last_occurrence=True)
        cb_pos = find_code_positions(tok, cb_text, code, S, last_occurrence=True)
        print(f"  evidence-mask BPE positions: {ev_pos}")
        print(f"  callback-mask BPE positions: {cb_pos}")
        return

    cats = list(CATEGORIES.keys())

    all_sessions: list[list[int]] = []
    cb_masks: list[list[int]] = []
    ev_masks: list[list[int]] = []
    sess_roles: list[int] = []  # 0=filler, 1=evidence, 2=callback
    chain_starts: list[int] = []
    chain_lengths: list[int] = []
    chain_callback_position: list[int] = []
    chain_evidence_positions: list[list[int]] = []
    chain_names: list[str] = []
    n_skipped_no_answer = 0

    body_len = a.n_evidence_sessions + a.n_filler_sessions

    for ci in range(a.n_chains):
        chosen_cats = rng.sample(cats, a.n_evidence_sessions)
        codes = [gen_code(rng, a.code_len) for _ in range(a.n_evidence_sessions)]
        callback_idx = rng.randrange(a.n_evidence_sessions)
        callback_cat = chosen_cats[callback_idx]
        callback_code = codes[callback_idx]
        chain_start = len(all_sessions)

        body_positions = list(range(body_len))
        evidence_body_positions = sorted(
            rng.sample(body_positions, a.n_evidence_sessions)
        )
        chain_evidence_position = [
            a.n_prefix_sessions + p for p in evidence_body_positions
        ]

        sessions_text: list[str] = []
        roles_for_chain: list[int] = []
        evidence_codes_at_session: dict[int, tuple[str, str]] = {}
        for _ in range(a.n_prefix_sessions):
            sessions_text.append(rng.choice(FILLERS))
            roles_for_chain.append(0)
        evidence_at_body_pos: dict[int, tuple[str, str]] = {
            p: (chosen_cats[i], codes[i])
            for i, p in enumerate(evidence_body_positions)
        }
        for p in range(body_len):
            session_idx_in_chain = a.n_prefix_sessions + p
            if p in evidence_at_body_pos:
                cat_p, code_p = evidence_at_body_pos[p]
                sessions_text.append(render_evidence(CATEGORIES[cat_p], code_p))
                roles_for_chain.append(1)
                evidence_codes_at_session[session_idx_in_chain] = (cat_p, code_p)
            else:
                sessions_text.append(rng.choice(FILLERS))
                roles_for_chain.append(0)
        sessions_text.append(render_callback(CATEGORIES[callback_cat], callback_code))
        roles_for_chain.append(2)

        assert len(sessions_text) == chain_len

        chain_session_ids: list[list[int]] = []
        chain_cb_masks: list[list[int]] = []
        chain_ev_masks: list[list[int]] = []
        cb_answer_positions: list[int] = []

        for si, (txt, role) in enumerate(zip(sessions_text, roles_for_chain)):
            ids = encode_session(tok, txt, S, eos_id)
            cb_mask = [0] * S
            ev_mask = [0] * S
            if si == chain_len - 1:
                positions = find_code_positions(
                    tok, txt, callback_code, S, last_occurrence=True
                )
                if not positions:
                    cb_answer_positions = []
                    break
                cb_answer_positions = positions
                for pos in positions:
                    if 0 <= pos < S:
                        cb_mask[pos] = 1
            elif si in evidence_codes_at_session:
                _cat_e, code_e = evidence_codes_at_session[si]
                positions = find_code_positions(
                    tok, txt, code_e, S, last_occurrence=False
                )
                for pos in positions:
                    if 0 <= pos < S:
                        ev_mask[pos] = 1
            chain_session_ids.append(ids)
            chain_cb_masks.append(cb_mask)
            chain_ev_masks.append(ev_mask)

        if not cb_answer_positions:
            n_skipped_no_answer += 1
            continue

        all_sessions.extend(chain_session_ids)
        cb_masks.extend(chain_cb_masks)
        ev_masks.extend(chain_ev_masks)
        sess_roles.extend(roles_for_chain)
        chain_starts.append(chain_start)
        chain_lengths.append(chain_len)
        chain_callback_position.append(chain_len - 1)
        chain_evidence_positions.append(chain_evidence_position)
        chain_names.append(
            f"synthetic_random_codes_{ci:05d}"
            f"_{callback_cat}_{callback_code}"
            f"_n{a.n_evidence_sessions}ev"
        )

    blob = {
        "session_ids": torch.tensor(all_sessions, dtype=torch.long),
        "session_callback_mask": torch.tensor(cb_masks, dtype=torch.int8),
        "session_evidence_mask": torch.tensor(ev_masks, dtype=torch.int8),
        "session_role": torch.tensor(sess_roles, dtype=torch.int8),
        "chain_starts": torch.tensor(chain_starts, dtype=torch.long),
        "chain_lengths": torch.tensor(chain_lengths, dtype=torch.long),
        "chain_callback_position": torch.tensor(chain_callback_position, dtype=torch.long),
        "chain_evidence_positions": chain_evidence_positions,
        "chain_names": chain_names,
        "session_len": S,
        "tokenizer": a.tokenizer,
        "corpus_kind": "synthd5_random_codes",
        "corpus_version": "v16",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    n_chains_kept = len(chain_starts)
    n_sessions_total = len(all_sessions)
    n_cb_tokens = int(torch.tensor(cb_masks, dtype=torch.int32).sum().item())
    n_ev_tokens = int(torch.tensor(ev_masks, dtype=torch.int32).sum().item())
    print(
        f"Wrote {n_chains_kept} chains "
        f"({n_sessions_total} sessions, "
        f"{n_cb_tokens} cb tokens, "
        f"{n_ev_tokens} evidence-id tokens, "
        f"avg {n_cb_tokens / max(1, n_chains_kept):.2f} cb-tokens/chain) "
        f"-> {out}"
    )
    if n_skipped_no_answer:
        print(f"  (skipped {n_skipped_no_answer} chains where the answer "
              f"span couldn't be located in the encoded callback session)")


if __name__ == "__main__":
    main()
