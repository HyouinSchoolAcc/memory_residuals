#!/usr/bin/env python3
"""Needle-in-a-haystack eval for Memory Residuals.

We construct synthetic chains of ``session_len=512`` sessions, place a
short, distinctive "needle" code into one of them, fill the remaining
slots with PG-19 filler sessions, and measure how well the model recalls
the needle after the chain has been compressed into ``M_c`` session by
session.

Concretely, for each ``(depth, position, seed)`` triple:

    needle_session  = template containing CODE  (e.g. "MX7-RT9-LB2")
    filler_sessions = PG-19 sessions sampled from the chain corpus
    recall_session  = template ending in the same CODE token-by-token

We score next-token NLL on the CODE tokens of the recall session under
three conditions:

    mem      : M_c built from the full chain prefix (the needle is in there)
    nomem    : M_c = None
    shuffle  : M_c built from a different chain's prefix (no needle present)

A working memory should give

    nll_nomem  >>  nll_mem               (memory helped)
    nll_shuffle >  nll_mem               (memory was *this* chain's, not generic style)

Output: a JSON with per-(depth, position) grids of these NLLs, plus the
hyper-parameters used. Pair with ``bootstrap_ci.py`` for confidence
intervals.

Usage:

    python tools/niah_eval.py \
      --model_path output/chain_v3_softparity_full/best \
      --filler_corpus paper_artifacts/chains/stage1_validation_s512.pt \
      --depths 1,5,10,20,30 \
      --positions 0.1,0.5,0.9 \
      --n_seeds 6 \
      --output results/eval/niah_v3_softparity.json
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


# -----------------------------------------------------------------------------
# Needle generation
# -----------------------------------------------------------------------------


def random_code(rng: random.Random, n_segments: int = 3, seg_len: int = 3) -> str:
    """Return a code like 'MX7-RT9-LB2' that's unlikely to be in pretraining."""
    alphabet = string.ascii_uppercase + string.digits
    segs = [
        "".join(rng.choice(alphabet) for _ in range(seg_len)) for _ in range(n_segments)
    ]
    return "-".join(segs)


SETUP_TEMPLATES = [
    'A small brass medallion lay on the table, engraved with the code {code}. The professor noted it carefully.',
    'In the dim attic, the children found an old envelope marked with the cipher {code}, sealed and waiting.',
    'The radio operator transmitted a single phrase that night: "the verification code is {code}, repeat {code}."',
    'Among the curiosities of the museum was a stone tablet inscribed only with the symbol {code}.',
    'On the back of the photograph, in faded ink, was written the inventory number {code}.',
]


RECALL_TEMPLATES = [
    'The professor reached into his pocket and produced once more the medallion bearing the code {code}',
    'The same envelope from the attic, marked with the cipher {code}',
    'The verification code, as we had heard transmitted, was {code}',
    'The stone tablet they had seen at the museum displayed only the symbol {code}',
    'On the photograph, in the same faded ink, the inventory number {code}',
]


# -----------------------------------------------------------------------------
# Chain construction
# -----------------------------------------------------------------------------


def session_at(blob: dict, chain_idx: int, position: int) -> torch.Tensor:
    s = int(blob["chain_starts"][chain_idx])
    return blob["session_ids"][s + position].long()


def pad_session(ids: list[int], session_len: int, pad_id: int) -> list[int]:
    if len(ids) >= session_len:
        return ids[:session_len]
    return ids + [pad_id] * (session_len - len(ids))


def build_needle_session(
    tokenizer,
    code: str,
    session_len: int,
    rng: random.Random,
    pad_id: int,
) -> tuple[torch.Tensor, list[int]]:
    """Returns (session_ids, code_token_ids)."""
    template = rng.choice(SETUP_TEMPLATES).format(code=code)
    # Pad with random English filler so the session is full-length and
    # doesn't look anomalously short to the compression module.
    filler = (
        " The day passed slowly, with the usual chores and minor errands, "
        "and nothing else of any importance occurred until much later."
    ) * 8
    text = template + " " + filler
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = pad_session(ids, session_len, pad_id)
    sess = torch.tensor(ids, dtype=torch.long)
    code_ids = tokenizer.encode(code, add_special_tokens=False)
    return sess, code_ids


def build_recall_session(
    tokenizer,
    code: str,
    session_len: int,
    rng: random.Random,
    pad_id: int,
    template_idx: int,
) -> tuple[torch.Tensor, slice]:
    """Returns (session_ids, span_for_code) where span_for_code indexes the
    *code* tokens within the session_ids tensor.

    We craft the recall text so the code lands toward the *middle* of the
    session -- early enough that memory has time to influence the activations,
    late enough that the model has a clear textual cue to predict the code.
    """
    intro = (
        "Many days later, the events of that earlier moment came back to mind. "
        "It happened almost without warning: a recollection, vivid and exact. "
    )
    template = RECALL_TEMPLATES[template_idx]
    # split the template at "{code}" so we can insert and locate it precisely
    before, after = template.split("{code}")
    full_text_pre_code = intro + before
    code_ids = tokenizer.encode(code, add_special_tokens=False)
    pre_ids = tokenizer.encode(full_text_pre_code, add_special_tokens=False)
    after_ids = tokenizer.encode(after + ".", add_special_tokens=False)
    tail_filler = (
        " The afternoon stretched on, slow and full of small distractions, "
        "as it always did when one was waiting for something to happen."
    ) * 4
    tail_ids = tokenizer.encode(tail_filler, add_special_tokens=False)

    ids = pre_ids + code_ids + after_ids + tail_ids
    ids = pad_session(ids, session_len, pad_id)
    sess = torch.tensor(ids, dtype=torch.long)
    # Code span within the session (within input_ids[:-1] for next-token labels)
    code_start = len(pre_ids)
    code_end = code_start + len(code_ids)
    return sess, slice(code_start, code_end)


def sample_filler_sessions(
    blob: dict,
    n_filler: int,
    rng: random.Random,
) -> list[torch.Tensor]:
    """Draw ``n_filler`` random sessions from random chains, with replacement."""
    n_chains = int(blob["chain_starts"].shape[0])
    chain_lens = blob["chain_lengths"]
    out = []
    while len(out) < n_filler:
        ci = rng.randint(0, n_chains - 1)
        L = int(chain_lens[ci])
        if L <= 0:
            continue
        pos = rng.randint(0, L - 1)
        out.append(session_at(blob, ci, pos))
    return out


# -----------------------------------------------------------------------------
# Memory state plumbing
# -----------------------------------------------------------------------------


@torch.no_grad()
def build_M_from_sessions(
    model: Qwen3MemResForCausalLM,
    sessions: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    for sess in sessions:
        ids = sess.to(device).unsqueeze(0)
        # Drop the last token so input_len == session_len-1 like the trainer.
        C = model.model.embed_tokens(ids[:, :-1])
        M_c = model.model.compress_session(C, M_c)
    return M_c


@torch.no_grad()
def score_code_nll(
    model: Qwen3MemResForCausalLM,
    recall_session: torch.Tensor,
    code_span: slice,
    M_c: torch.Tensor | None,
    device: torch.device,
) -> float:
    """Mean per-token NLL over the code span, given the recall session and M_c."""
    # We feed the full session to the model; labels mask all but the code span.
    ids = recall_session.to(device).unsqueeze(0)        # (1, S)
    input_ids = ids[:, :-1]
    targets = ids[:, 1:].clone()
    mask = torch.full_like(targets, fill_value=-100)
    # Predict token at position t+1 from input position t. The model emits
    # logits at position t for token at t+1, so the labels we want at
    # output-positions (code_span.start - 1) .. (code_span.stop - 2) are
    # exactly the code tokens themselves.
    out_lo = max(code_span.start - 1, 0)
    out_hi = max(code_span.stop - 1, 0)
    if out_hi <= out_lo:
        return float("nan")
    mask[:, out_lo:out_hi] = targets[:, out_lo:out_hi]
    out = model(input_ids=input_ids, labels=mask, M_c=M_c)
    # ``out.loss`` is the mean over valid positions (those with label != -100),
    # which is exactly the per-code-token NLL since we masked everything else.
    return float(out.loss.item())


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def parse_csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="path to a saved best/ ckpt dir")
    ap.add_argument("--filler_corpus", required=True,
                    help="pre-tokenized chain corpus to sample filler sessions from")
    ap.add_argument("--depths", default="1,5,10,20,30",
                    help="comma-separated needle depths (#sessions of prefix before recall)")
    ap.add_argument("--positions", default="0.1,0.5,0.9",
                    help="comma-separated relative needle positions in [0,1]")
    ap.add_argument("--n_seeds", type=int, default=6,
                    help="number of (code, filler-draw, template) random seeds per cell")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    depths = parse_csv_ints(args.depths)
    positions = parse_csv_floats(args.positions)

    print(f"[niah] loading model from {args.model_path}", flush=True)
    model = Qwen3MemResForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    cfg = model.config
    session_len = int(getattr(cfg, "memres_session_len", 512))
    if session_len <= 0:
        session_len = 512
    print(f"[niah] session_len={session_len}, K={cfg.memres_num_vectors}, "
          f"d={cfg.hidden_size}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    print(f"[niah] loading filler corpus from {args.filler_corpus}", flush=True)
    blob = torch.load(args.filler_corpus, map_location="cpu", weights_only=False)
    n_filler_chains = int(blob["chain_starts"].shape[0])
    print(f"[niah]  filler corpus has {n_filler_chains} chains", flush=True)

    rng = random.Random(args.seed)
    device = torch.device(args.device)

    grid: dict[str, list[dict]] = {}
    pbar = tqdm(total=len(depths) * len(positions) * args.n_seeds,
                desc="niah cells")

    for depth in depths:
        for rel_pos in positions:
            cell = []
            # Place the needle at int(rel_pos * (depth - 1)) (0..depth-1).
            # depth==1 forces position 0 (needle is the only prefix session).
            needle_pos = max(0, int(rel_pos * max(depth - 1, 0)))
            for s in range(args.n_seeds):
                cell_rng = random.Random(rng.random() + 1e9 * (depth * 100 + s))
                code = random_code(cell_rng)
                template_idx = cell_rng.randrange(len(SETUP_TEMPLATES))
                # Build needle and recall sessions
                needle_session, _ = build_needle_session(
                    tokenizer, code, session_len, cell_rng, pad_id,
                )
                recall_session, code_span = build_recall_session(
                    tokenizer, code, session_len, cell_rng, pad_id, template_idx,
                )
                # Build the prefix: sample (depth - 1) filler sessions and
                # splice the needle at needle_pos.
                fillers = sample_filler_sessions(blob, max(depth - 1, 0), cell_rng)
                prefix = list(fillers)
                prefix.insert(needle_pos, needle_session)
                prefix = prefix[:depth]                 # exact length

                # Shuffle prefix: same length, all random fillers, no needle.
                shuffle_prefix = sample_filler_sessions(blob, depth, cell_rng)

                # Score under three conditions.
                M_match = build_M_from_sessions(model, prefix, device)
                M_shuffle = build_M_from_sessions(model, shuffle_prefix, device)
                nll_mem = score_code_nll(model, recall_session, code_span, M_match, device)
                nll_no = score_code_nll(model, recall_session, code_span, None, device)
                nll_sh = score_code_nll(model, recall_session, code_span, M_shuffle, device)

                cell.append({
                    "seed": s,
                    "code": code,
                    "needle_position": needle_pos,
                    "depth": depth,
                    "rel_position": rel_pos,
                    "nll_mem": nll_mem,
                    "nll_nomem": nll_no,
                    "nll_shuffle": nll_sh,
                    "delta_nm_m": nll_no - nll_mem,
                    "delta_sh_m": nll_sh - nll_mem,
                })
                pbar.update(1)

            key = f"depth{depth}_rp{rel_pos:.2f}"
            grid[key] = cell

    pbar.close()

    # Aggregate per-cell summaries.
    summary = {}
    for key, cell in grid.items():
        n = len(cell)
        def _mean(field: str) -> float:
            xs = [c[field] for c in cell if c[field] == c[field]]   # filter NaN
            return float(sum(xs) / len(xs)) if xs else float("nan")
        summary[key] = {
            "n": n,
            "depth": cell[0]["depth"],
            "rel_position": cell[0]["rel_position"],
            "needle_position": cell[0]["needle_position"],
            "mean_nll_mem":     _mean("nll_mem"),
            "mean_nll_nomem":   _mean("nll_nomem"),
            "mean_nll_shuffle": _mean("nll_shuffle"),
            "mean_delta_nm_m":  _mean("delta_nm_m"),
            "mean_delta_sh_m":  _mean("delta_sh_m"),
        }

    out = {
        "model_path": args.model_path,
        "filler_corpus": args.filler_corpus,
        "session_len": session_len,
        "n_seeds": args.n_seeds,
        "depths": depths,
        "positions": positions,
        "summary": summary,
        "raw": grid,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[niah] wrote {args.output}", flush=True)
    # Print a tiny summary table to stdout.
    print(f"\n{'cell':<22} {'mem':>8} {'nomem':>8} {'shuffle':>8} {'?nm-m':>8} {'?sh-m':>8}")
    for key, s in summary.items():
        print(f"{key:<22} {s['mean_nll_mem']:8.4f} {s['mean_nll_nomem']:8.4f} "
              f"{s['mean_nll_shuffle']:8.4f} {s['mean_delta_nm_m']:+8.4f} "
              f"{s['mean_delta_sh_m']:+8.4f}")


if __name__ == "__main__":
    main()
