#!/usr/bin/env python3
"""Synthetic persona-callback corpus: D4 in the diagnostic harness.

Builds a closed-set, ground-truth-known training corpus where memory is
*necessary* to solve the task.  Used to test whether the Memory
Residuals architecture can plausibly do its job at all, before
committing more compute to natural-data tuning that bakes in dozens of
confounds (filler distribution, callback density, evidence labels,
backbone co-evolution, ...).

Structure of one chain (defaults: ``n_evidence_sessions=2``,
``n_filler_sessions=7``, ``n_prefix_sessions=0`` -> chain length 10):

  session 0  EVIDENCE_1  "User: My favorite COLOR is red.
                          Assistant: ... your favorite color is red."
  session 1  FILLER      random innocuous chit-chat, never names items
  session 2  FILLER
  session 3  EVIDENCE_2  "User: My favorite TOOL is whisk.
                          Assistant: ... your favorite tool is whisk."
  session 4..n  FILLER
  session 9  CALLBACK    "User: Quick question, what was my favorite
                          {color OR tool} again?
                          Assistant: Your favorite {category} is {item}."

The corpus draws ``n_evidence_sessions`` (category, item) pairs per
chain, each from a *distinct category*, and shuffles them into random
positions inside the chain (after any prepended sessions, before the
callback).  The callback uniformly samples ONE of the evidence pairs
to ask about; the assistant turn answers with the corresponding item,
and ``session_callback_mask`` is set on those token positions.

WHY MULTIPLE EVIDENCE SESSIONS (v15 fix, 2026-05-02).
The original (n_evidence_sessions=1) variant gave the writer a trivial
optimal policy: write evidence at session 0, then KEEP M_c unchanged
through every distractor session because M_c_prev is always the right
answer.  Under that policy the writer never has to integrate signal
across multiple sessions, so the judge's keep/write decisioning is
never exercised, and the failure mode of a saturated write_gate (g~0
keep-everything) is rewarded by the LM loss.  With
n_evidence_sessions>=2 the writer must (a) capture evidence_1, (b)
preserve it across fillers, (c) *integrate* evidence_2 into M_c
without overwriting evidence_1, and (d) hold both across remaining
fillers; the readout in turn must discriminate which evidence the
callback asks about.  This exercises every memory primitive (store /
preserve / merge / discriminate) that a working memory subsystem must
implement, and it is no longer trivially solvable by a content-blind
keep-everything writer.

WHY PREPENDED SESSIONS.  The optional ``n_prefix_sessions`` flag
inserts that many filler sessions before the first evidence, lengthening
the chain without adding evidence and stressing the writer's robustness
to longer histories before the relevant content appears.  Defaults to
0 to preserve the original budget; raise to 2-4 for stress sweeps.

ITEM_X is drawn uniformly from a closed set of N_ITEMS distinct
single-string items (default 256: 32 each of colors, fruits, animals,
objects, sports, tools, instruments, hobbies) over 8 disjoint
categories.  Evidence sessions in a single chain use distinct
categories, so the callback's "what was my favorite {category}" is
unambiguous.  The callback session's ``session_callback_mask`` is set
on the token positions of the answer item in the assistant turn -- so
the trainer's ``--callback_loss_weight`` surfaces explicit gradient on
those tokens.

Phase-aligned eval reads ``chain_evidence_positions`` (now a list of
*all* evidence positions per chain) and picks one or all as evidence;
``chain_callback_position`` points at the callback session.  Both
fields are written here exactly the way the v11+ LongMemEval corpus
expects.

WHY THIS IS DIAGNOSTIC.  Under any architecture that *can* learn to
keep ITEM_X alive across distractor sessions, the per-token NLL on
the callback's ITEM_X tokens should drop close to 0.  Under an
architecture that cannot (because the writer doesn't actually write,
or because the readout cannot retrieve, or because the judge softmax
has flattened), the NLL stays at log(N_ITEMS_in_vocab) - log(prior).
The closed-set design means we *know* the floor: with 256 items, a
correct-on-task model has CE ~ log(1) = 0 nats on the answer token
(modulo tokenization noise; some items split into 2 BPE tokens, where
the second token is conditioned on the first and so still has very
low CE under a model that has learned the binding).

Output format matches ``ChainCorpus`` in train_chain.py exactly:
``chain_evidence_positions[ci]`` is a list of session indices for the
evidence (length = ``n_evidence_sessions``) and
``chain_callback_position[ci]`` is the session index of the callback.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer


# 256-item closed set: 32 each of 8 disjoint categories.  All items are
# common single English words; they tokenise to 1-3 BPE tokens under
# the Qwen3 tokenizer (verified at corpus-build time).
CLOSED_SET: dict[str, list[str]] = {
    "color": [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
        "black", "white", "gray", "cyan", "magenta", "violet", "indigo", "teal",
        "maroon", "navy", "olive", "lime", "scarlet", "amber", "coral", "ivory",
        "crimson", "azure", "beige", "khaki", "salmon", "turquoise", "lavender", "mint",
    ],
    "fruit": [
        "apple", "banana", "cherry", "grape", "orange", "peach", "pear", "plum",
        "strawberry", "watermelon", "pineapple", "mango", "kiwi", "papaya", "guava", "lemon",
        "lime", "blueberry", "raspberry", "blackberry", "cranberry", "apricot", "fig", "pomegranate",
        "coconut", "avocado", "lychee", "passionfruit", "tangerine", "grapefruit", "nectarine", "persimmon",
    ],
    "animal": [
        "dog", "cat", "horse", "cow", "sheep", "pig", "chicken", "duck",
        "rabbit", "hamster", "fox", "wolf", "bear", "deer", "elk", "moose",
        "lion", "tiger", "leopard", "cheetah", "elephant", "giraffe", "zebra", "rhino",
        "monkey", "gorilla", "panda", "koala", "kangaroo", "penguin", "dolphin", "whale",
    ],
    "object": [
        "chair", "table", "lamp", "couch", "bed", "desk", "shelf", "mirror",
        "clock", "vase", "rug", "curtain", "pillow", "blanket", "basket", "bucket",
        "ladder", "broom", "shovel", "hammer", "saw", "wrench", "drill", "nail",
        "candle", "torch", "telescope", "microscope", "compass", "globe", "atlas", "map",
    ],
    "sport": [
        "soccer", "basketball", "baseball", "football", "tennis", "golf", "hockey", "rugby",
        "cricket", "volleyball", "swimming", "cycling", "running", "skiing", "snowboarding", "surfing",
        "boxing", "wrestling", "judo", "karate", "fencing", "archery", "rowing", "sailing",
        "climbing", "hiking", "skating", "dancing", "yoga", "pilates", "diving", "kayaking",
    ],
    "tool": [
        "scissors", "knife", "spoon", "fork", "spatula", "whisk", "tongs", "ladle",
        "grater", "peeler", "rolling_pin", "strainer", "thermometer", "scale", "timer", "blender",
        "toaster", "kettle", "fryer", "mixer", "grinder", "sharpener", "stapler", "ruler",
        "calculator", "tape", "pen", "pencil", "marker", "eraser", "notebook", "calendar",
    ],
    "instrument": [
        "piano", "guitar", "violin", "drums", "flute", "trumpet", "saxophone", "clarinet",
        "trombone", "harp", "cello", "viola", "oboe", "bassoon", "accordion", "harmonica",
        "ukulele", "banjo", "mandolin", "sitar", "marimba", "xylophone", "tuba", "tambourine",
        "bagpipes", "didgeridoo", "kazoo", "ocarina", "recorder", "synthesizer", "organ", "keyboard",
    ],
    "hobby": [
        "reading", "writing", "drawing", "painting", "sculpting", "knitting", "sewing", "crochet",
        "gardening", "cooking", "baking", "fishing", "hunting", "camping", "birdwatching", "stargazing",
        "photography", "filmmaking", "podcasting", "blogging", "vlogging", "gaming", "puzzling", "origami",
        "calligraphy", "pottery", "woodworking", "metalworking", "beekeeping", "brewing", "winemaking", "collecting",
    ],
}

ITEM_TYPE_PHRASE: dict[str, str] = {
    "color": "color",
    "fruit": "fruit",
    "animal": "animal",
    "object": "thing",
    "sport": "sport",
    "tool": "tool",
    "instrument": "instrument",
    "hobby": "hobby",
}


# 50 generic fillers; each ~30-70 tokens.  Care taken to never mention
# any of the closed-set items by name.  Topics: weather, work, plans,
# food (without naming foods from the closed set), travel, family,
# small chit-chat, etc.
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


def _flatten_closed_set() -> list[tuple[str, str]]:
    """Return [(category, item)] for the entire closed set, deduped."""
    seen = set()
    pairs = []
    for cat, items in CLOSED_SET.items():
        for it in items:
            key = it.lower()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((cat, it))
    return pairs


def render_persona(category: str, item: str) -> str:
    type_phrase = ITEM_TYPE_PHRASE[category]
    return (
        f"User: My favorite {type_phrase} in the world is {item}.\n"
        f"Assistant: Got it, your favorite {type_phrase} is {item}. "
        f"I'll remember that."
    )


def render_callback(category: str, item: str) -> str:
    """Produce the callback session.

    Important: the answer string ITEM appears in the assistant turn so
    the LM-loss is non-trivially predicting it given prior context.
    The session_callback_mask will be set on the token positions of the
    *answer* span only.
    """
    type_phrase = ITEM_TYPE_PHRASE[category]
    return (
        f"User: Quick question, what was my favorite {type_phrase} again?\n"
        f"Assistant: Your favorite {type_phrase} is {item}."
    )


def encode_session(tok, text: str, session_len: int, eos_id: int) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)[:session_len]
    if len(ids) < session_len:
        ids = ids + [eos_id] * (session_len - len(ids))
    return ids


def find_answer_positions(
    tok,
    callback_text: str,
    item: str,
    session_len: int,
) -> list[int]:
    """Return the indices in the encoded callback session that
    correspond to the *final* occurrence of ``item`` in
    ``callback_text``.  We pick the final occurrence so we mask the
    answer span (Assistant: ... is ITEM.) rather than the question's
    earlier mention.
    """
    full_ids = tok.encode(callback_text, add_special_tokens=False)[:session_len]

    # Try a few tokenisation variants; the leading-space variant is
    # the one that fires inside running text in BPE tokenisers.
    candidates = []
    for prefix in (" ", ""):
        cand = tok.encode(prefix + item, add_special_tokens=False)
        if cand:
            candidates.append(cand)

    last_match: list[int] = []
    for cand in candidates:
        n = len(cand)
        for start in range(len(full_ids) - n + 1):
            if full_ids[start : start + n] == cand:
                last_match = list(range(start, start + n))
        if last_match:
            return last_match
    return []


def _sample_evidence(
    rng: random.Random,
    by_category: dict[str, list[tuple[str, str]]],
    n_evidence: int,
) -> list[tuple[str, str]]:
    """Sample ``n_evidence`` (category, item) pairs from distinct
    categories.  Returns a list of length ``n_evidence`` with no two
    entries sharing a category, so the callback's "what was my
    favorite {category}" is always unambiguous.
    """
    cats = list(by_category.keys())
    if n_evidence > len(cats):
        raise SystemExit(
            f"--n_evidence_sessions={n_evidence} exceeds the number of "
            f"distinct categories ({len(cats)}); cannot sample without "
            f"category collision."
        )
    chosen_cats = rng.sample(cats, n_evidence)
    out: list[tuple[str, str]] = []
    for c in chosen_cats:
        # by_category[c] is a list of (cat, item) tuples; pick one.
        _cat, item = rng.choice(by_category[c])
        out.append((c, item))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output .pt path.")
    parser.add_argument("--n_chains", type=int, default=5000)
    parser.add_argument(
        "--n_filler_sessions", type=int, default=7,
        help="Distractor sessions interspersed between evidence and "
             "before the callback. Default 7.",
    )
    parser.add_argument(
        "--n_evidence_sessions", type=int, default=2,
        help="v15 (2026-05-02): Number of distinct evidence sessions "
             "per chain, each introducing a fact in a distinct "
             "category. The callback uniformly samples one evidence "
             "to ask about. Set to 1 for the legacy single-fact corpus. "
             "Default 2 forces the writer to merge two facts and "
             "discriminate at readout time.",
    )
    parser.add_argument(
        "--n_prefix_sessions", type=int, default=0,
        help="v15 (2026-05-02): Number of filler sessions inserted at "
             "the start of each chain before the first evidence. "
             "Stresses the writer's robustness to long pre-evidence "
             "history. Default 0.",
    )
    parser.add_argument("--session_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--audit_tokenisation", action="store_true",
        help="Print per-item token-id lengths and first chain's "
             "callback-mask diagnostic, then exit.",
    )
    args = parser.parse_args()

    if args.n_evidence_sessions < 1:
        raise SystemExit("--n_evidence_sessions must be >= 1.")
    if args.n_filler_sessions < 0:
        raise SystemExit("--n_filler_sessions must be >= 0.")
    if args.n_prefix_sessions < 0:
        raise SystemExit("--n_prefix_sessions must be >= 0.")

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("tokeniser has no eos_token_id; cannot pad.")
    S = args.session_len
    chain_len = (
        args.n_prefix_sessions
        + args.n_evidence_sessions
        + args.n_filler_sessions
        + 1  # callback
    )

    items_pool = _flatten_closed_set()
    by_category: dict[str, list[tuple[str, str]]] = {}
    for cat, it in items_pool:
        by_category.setdefault(cat, []).append((cat, it))
    if args.audit_tokenisation:
        print(f"closed set size: {len(items_pool)}")
        print(f"categories: {list(by_category.keys())} "
              f"(n={len(by_category)})")
        token_lens = []
        for cat, it in items_pool:
            ids = tok.encode(" " + it, add_special_tokens=False)
            token_lens.append((cat, it, len(ids), ids))
        token_lens.sort(key=lambda x: -x[2])
        print(f"longest 10 (token-count, cat, item, ids):")
        for cat, it, n, ids in token_lens[:10]:
            print(f"  {n} {cat:<10} {it:<14} {ids}")
        print(f"shortest 10:")
        for cat, it, n, ids in sorted(token_lens, key=lambda x: x[2])[:10]:
            print(f"  {n} {cat:<10} {it:<14} {ids}")
        # First-chain audit.
        cat, it = items_pool[0]
        cb_text = render_callback(cat, it)
        positions = find_answer_positions(tok, cb_text, it, S)
        print(f"\nfirst chain audit: cat={cat} item={it!r}")
        print(f"  callback_text: {cb_text}")
        print(f"  answer-span positions: {positions}")
        print(f"  answer-span tokens: {[tok.decode([tok.encode(cb_text, add_special_tokens=False)[p]]) for p in positions]}")
        # v15: also audit evidence sampling for n_evidence > 1.
        if args.n_evidence_sessions > 1:
            ev = _sample_evidence(rng, by_category, args.n_evidence_sessions)
            print(f"\nsample evidence draw "
                  f"(n_evidence={args.n_evidence_sessions}):")
            for i, (c, ip) in enumerate(ev):
                print(f"  evidence_{i}: cat={c} item={ip[1]!r}")
            print(f"chain_len = {chain_len} = "
                  f"{args.n_prefix_sessions} prefix + "
                  f"{args.n_evidence_sessions} evidence + "
                  f"{args.n_filler_sessions} filler + 1 callback")
        return

    all_sessions: list[list[int]] = []
    cb_masks: list[list[int]] = []
    chain_starts: list[int] = []
    chain_lengths: list[int] = []
    chain_callback_position: list[int] = []
    chain_evidence_positions: list[list[int]] = []
    chain_names: list[str] = []
    n_skipped_no_answer = 0

    # Total non-prefix non-callback "body" length where evidence and
    # filler get interleaved at random positions.
    body_len = args.n_evidence_sessions + args.n_filler_sessions

    for ci in range(args.n_chains):
        # 1) Sample distinct-category evidence facts.
        evidence_pairs = _sample_evidence(
            rng, by_category, args.n_evidence_sessions
        )
        # 2) Choose which evidence the callback asks about.
        callback_idx = rng.randrange(args.n_evidence_sessions)
        callback_cat, callback_item = evidence_pairs[callback_idx]
        chain_start = len(all_sessions)

        # 3) Choose evidence positions within the body. The body has
        #    `body_len` slots; evidence occupies `n_evidence_sessions`
        #    of them at random distinct positions, fillers fill the
        #    rest. Body positions are then mapped to chain positions
        #    by adding `n_prefix_sessions`.
        body_positions = list(range(body_len))
        evidence_body_positions = sorted(
            rng.sample(body_positions, args.n_evidence_sessions)
        )
        chain_evidence_position = [
            args.n_prefix_sessions + p for p in evidence_body_positions
        ]

        # 4) Build session texts in order.
        sessions_text: list[str] = []
        # 4a) Prefix fillers.
        for _ in range(args.n_prefix_sessions):
            sessions_text.append(rng.choice(FILLERS))
        # 4b) Body: place each evidence at its chosen position; fillers
        #     in the gaps.
        evidence_at_body_pos: dict[int, tuple[str, str]] = {
            p: evidence_pairs[i]
            for i, p in enumerate(evidence_body_positions)
        }
        for p in range(body_len):
            if p in evidence_at_body_pos:
                cat_p, item_p = evidence_at_body_pos[p]
                sessions_text.append(render_persona(cat_p, item_p))
            else:
                sessions_text.append(rng.choice(FILLERS))
        # 4c) Callback (asks about evidence_pairs[callback_idx]).
        sessions_text.append(render_callback(callback_cat, callback_item))

        assert len(sessions_text) == chain_len, (
            f"len(sessions_text)={len(sessions_text)} != "
            f"chain_len={chain_len}"
        )

        # 5) Encode + build callback mask. Mask is non-zero ONLY in the
        #    final session, on the token positions of the answer-span
        #    (the *callback's* item).
        chain_session_ids: list[list[int]] = []
        chain_cb_masks: list[list[int]] = []
        answer_positions: list[int] = []
        for si, txt in enumerate(sessions_text):
            ids = encode_session(tok, txt, S, eos_id)
            mask = [0] * S
            if si == chain_len - 1:
                positions = find_answer_positions(
                    tok, txt, callback_item, S
                )
                if not positions:
                    answer_positions = []
                    break
                answer_positions = positions
                for pos in positions:
                    if 0 <= pos < S:
                        mask[pos] = 1
            chain_session_ids.append(ids)
            chain_cb_masks.append(mask)

        if not answer_positions:
            n_skipped_no_answer += 1
            continue

        all_sessions.extend(chain_session_ids)
        cb_masks.extend(chain_cb_masks)
        chain_starts.append(chain_start)
        chain_lengths.append(chain_len)
        chain_callback_position.append(chain_len - 1)
        chain_evidence_positions.append(chain_evidence_position)
        # Name encodes the asked-about (cat, item); the other evidence
        # facts are still in M_c at callback time but the LM head only
        # sees one being queried.
        chain_names.append(
            f"synthetic_persona_callback_{ci:05d}"
            f"_{callback_cat}_{callback_item}"
            f"_n{args.n_evidence_sessions}ev"
        )

    blob = {
        "session_ids": torch.tensor(all_sessions, dtype=torch.long),
        "session_callback_mask": torch.tensor(cb_masks, dtype=torch.int8),
        "chain_starts": torch.tensor(chain_starts, dtype=torch.long),
        "chain_lengths": torch.tensor(chain_lengths, dtype=torch.long),
        "chain_callback_position": torch.tensor(chain_callback_position, dtype=torch.long),
        "chain_evidence_positions": chain_evidence_positions,
        "chain_names": chain_names,
        "session_len": S,
        "tokenizer": args.tokenizer,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    n_chains_kept = len(chain_starts)
    n_sessions_total = len(all_sessions)
    n_cb_tokens = int(torch.tensor(cb_masks, dtype=torch.int32).sum().item())
    print(
        f"Wrote {n_chains_kept} chains "
        f"({n_sessions_total} sessions, "
        f"{n_cb_tokens} callback tokens, "
        f"avg {n_cb_tokens / max(1, n_chains_kept):.2f} cb-tokens/chain) "
        f"-> {out}"
    )
    if n_skipped_no_answer:
        print(f"  (skipped {n_skipped_no_answer} chains where the answer "
              f"span couldn't be located in the encoded callback session)")


if __name__ == "__main__":
    main()
