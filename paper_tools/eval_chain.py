#!/usr/bin/env python3
"""Comprehensive chain-aware evaluator for Memory Residuals.

For every chain in the input corpus, sequentially compress the prefix into
$M_c$ session by session (the same recurrent unroll the trainer uses) and
score the *last* ``--score_window`` sessions under several configurations:

    - **mem**:        $M_c$ built from all preceding sessions.
    - **nomem**:      $M_c = \\varnothing$ (no memory).
    - **shuffle**:    $M_c$ built from another chain's preceding sessions
                       (same length).
    - **oracle**:     concatenate the last ``--oracle_window`` raw prior
                       sessions into the input context, no memory.  This is
                       the upper-bound on what any compression scheme could
                       extract from the prior sessions.
    - **rag**:        for each scored token, retrieve the top-k dense chunks
                       from the chain prefix via MiniLM and prepend them to
                       the current session, no memory.  This is the
                       comparison against textual retrieval.

Outputs a single JSON with aggregate CE and the per-token-bucket help
ratios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def load_chain_corpus(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def chain_session_at(blob: dict, chain_idx: int, position: int) -> torch.Tensor:
    s = int(blob["chain_starts"][chain_idx])
    return blob["session_ids"][s + position].long()


def chain_window(blob: dict, chain_idx: int, start: int, k: int) -> torch.Tensor:
    s = int(blob["chain_starts"][chain_idx])
    return blob["session_ids"][s + start : s + start + k].long()


@torch.no_grad()
def build_M_c(
    model: Qwen3MemResForCausalLM,
    blob: dict,
    chain_idx: int,
    end: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Build M_c from chain[chain_idx][:end], session by session."""
    if end <= 0:
        return None
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    for j in range(end):
        sess = chain_session_at(blob, chain_idx, j).to(device).unsqueeze(0)
        C = model.model.embed_tokens(sess[:, :-1])
        M_c = model.model.compress_session(C, M_c)
    return M_c


@torch.no_grad()
def chain_loss(
    model: Qwen3MemResForCausalLM,
    input_ids: torch.Tensor,
    M_c: torch.Tensor | None,
    labels_mask: torch.Tensor | None = None,
) -> float:
    labels = input_ids.clone() if labels_mask is None else labels_mask
    out = model(input_ids=input_ids, labels=labels, M_c=M_c)
    return float(out.loss.item())


@torch.no_grad()
def evaluate_corpus(
    model: Qwen3MemResForCausalLM,
    blob: dict,
    device: torch.device,
    score_window: int,
    oracle_window: int,
    do_rag: bool = False,
    rag_top_k: int = 3,
    rag_chunk_chars: int = 1200,
    rag_prefix_len: int = 1024,
    embedder=None,
    tokenizer=None,
) -> dict:
    """Evaluate every chain in ``blob`` and return aggregate metrics."""
    n_chains = int(blob["chain_starts"].shape[0])
    chain_lengths = blob["chain_lengths"]
    chain_names = blob["chain_names"]

    ce_mem, ce_nomem, ce_shuffle, ce_oracle, ce_rag = [], [], [], [], []
    per_chain = []
    for ci in tqdm(range(n_chains), desc="eval chains"):
        length = int(chain_lengths[ci])
        if length < score_window + 1:
            continue

        # Build M_c by walking the *full* prefix once; cache it at every
        # boundary so we can reuse for the score positions.
        cfg = model.config
        K, d = cfg.memres_num_vectors, cfg.hidden_size
        M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
        prefix_M = [M_c.clone()]   # M_c at *position 0* (no sessions seen yet)
        for end in range(length):
            sess = chain_session_at(blob, ci, end).to(device).unsqueeze(0)
            C = model.model.embed_tokens(sess[:, :-1])
            M_c = model.model.compress_session(C, M_c)
            prefix_M.append(M_c.clone())   # M_c after seeing sessions 0..end

        # Pre-build the shuffled chain's M_c trajectory (same length as ci).
        shuffle_idx = (ci + 1) % n_chains
        shuffle_len = int(chain_lengths[shuffle_idx])
        shuffle_M_at: dict[int, torch.Tensor] = {}
        if shuffle_len >= 1:
            M_sh = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
            shuffle_M_at[0] = M_sh.clone()
            for j in range(min(length, shuffle_len)):
                ssess = chain_session_at(blob, shuffle_idx, j).to(device).unsqueeze(0)
                C_sh = model.model.embed_tokens(ssess[:, :-1])
                M_sh = model.model.compress_session(C_sh, M_sh)
                shuffle_M_at[j + 1] = M_sh.clone()

        chain_buckets = {
            "mem": [], "nomem": [], "shuffle": [], "oracle": [], "rag": [],
        }
        score_starts = range(length - score_window, length)
        for end in score_starts:
            sess = chain_session_at(blob, ci, end).to(device).unsqueeze(0)
            input_ids = sess[:, :-1]

            # Memory on
            ce_m = chain_loss(model, input_ids, prefix_M[end])
            ce_mem.append(ce_m)
            chain_buckets["mem"].append(ce_m)
            # Memory off
            ce_n = chain_loss(model, input_ids, None)
            ce_nomem.append(ce_n)
            chain_buckets["nomem"].append(ce_n)
            # Shuffled memory
            if end <= max(shuffle_M_at):
                ce_s = chain_loss(model, input_ids, shuffle_M_at[end])
                ce_shuffle.append(ce_s)
                chain_buckets["shuffle"].append(ce_s)
            # Oracle: concat last oracle_window prior sessions raw.
            if end > 0:
                start = max(0, end - oracle_window)
                if start < end:
                    prior = chain_window(blob, ci, start, end - start).to(device)
                    prior_flat = prior.flatten().unsqueeze(0)
                    full = torch.cat([prior_flat, input_ids], dim=1)
                    labels_o = full.clone()
                    labels_o[:, : prior_flat.shape[1]] = -100
                    out_or = model(input_ids=full, labels=labels_o, M_c=None)
                    ce_oracle.append(float(out_or.loss.item()))
                    chain_buckets["oracle"].append(float(out_or.loss.item()))

            # RAG over chain prefix sessions (decoded), if requested.
            if do_rag and end > 0:
                # Decode the prior sessions to raw text using the saved tokenizer.
                decoded_chunks: list[str] = []
                for j in range(end):
                    osess = chain_session_at(blob, ci, j).tolist()
                    text = tokenizer.decode(osess, skip_special_tokens=True)
                    # Split text into chunks
                    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    if not parts:
                        parts = [text[:rag_chunk_chars]]
                    for p in parts:
                        if len(p) <= rag_chunk_chars:
                            decoded_chunks.append(p)
                        else:
                            for i in range(0, len(p), rag_chunk_chars):
                                decoded_chunks.append(p[i : i + rag_chunk_chars])
                if decoded_chunks:
                    cur_text = tokenizer.decode(input_ids[0].tolist()[:rag_chunk_chars],
                                                skip_special_tokens=True)
                    emb = embedder.encode(decoded_chunks, convert_to_tensor=True,
                                          normalize_embeddings=True)
                    q = embedder.encode([cur_text], convert_to_tensor=True,
                                        normalize_embeddings=True)
                    scores = (q @ emb.T).squeeze(0)
                    top = torch.topk(scores, k=min(rag_top_k, len(decoded_chunks))).indices.tolist()
                    retrieved = "\n\n".join(decoded_chunks[i] for i in top)
                    prefix_ids = tokenizer.encode(retrieved, add_special_tokens=False)[-rag_prefix_len:]
                    full_ids = prefix_ids + input_ids[0].tolist()
                    full = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
                    labels_r = full.clone()
                    labels_r[:, : len(prefix_ids)] = -100
                    out_r = model(input_ids=full, labels=labels_r, M_c=None)
                    ce_rag.append(float(out_r.loss.item()))
                    chain_buckets["rag"].append(float(out_r.loss.item()))

        per_chain.append({
            "chain_id": chain_names[ci],
            "length": length,
            "ce_mem": mean(chain_buckets["mem"]),
            "ce_nomem": mean(chain_buckets["nomem"]),
            "ce_shuffle": mean(chain_buckets["shuffle"]),
            "ce_oracle": mean(chain_buckets["oracle"]),
            "ce_rag": mean(chain_buckets["rag"]) if do_rag else None,
        })

    out = {
        "n_chains_scored": len(per_chain),
        "n_score_positions": len(ce_mem),
        "ce_mem": mean(ce_mem),
        "ce_nomem": mean(ce_nomem),
        "ce_shuffle": mean(ce_shuffle),
        "ce_oracle_concat": mean(ce_oracle),
        "ce_rag": mean(ce_rag) if do_rag else None,
        "delta_nomem_minus_mem": mean(ce_nomem) - mean(ce_mem),
        "delta_shuffle_minus_mem": mean(ce_shuffle) - mean(ce_mem),
        "delta_oracle_minus_mem": mean(ce_oracle) - mean(ce_mem),
        "delta_rag_minus_mem": (mean(ce_rag) - mean(ce_mem)) if do_rag else None,
        # Useful normalised metrics:
        "memory_capture_ratio": (
            (mean(ce_nomem) - mean(ce_mem))
            / max(1e-6, mean(ce_nomem) - mean(ce_oracle))
        ) if mean(ce_oracle) < mean(ce_nomem) else None,
        "per_chain": per_chain,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="One or more pre-tokenized chain .pt files")
    ap.add_argument("--names", nargs="+", default=None,
                    help="Optional names matching --corpora")
    ap.add_argument("--score_window", type=int, default=4)
    ap.add_argument("--oracle_window", type=int, default=4)
    ap.add_argument("--do_rag", action="store_true")
    ap.add_argument("--rag_top_k", type=int, default=3)
    ap.add_argument("--rag_prefix_len", type=int, default=1024)
    ap.add_argument("--rag_embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    device = torch.device(a.device)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(a.model_path)

    embedder = None
    if a.do_rag:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(a.rag_embedder, device=device)

    if a.names is None:
        names = [Path(p).stem for p in a.corpora]
    else:
        names = a.names
    if len(names) != len(a.corpora):
        raise ValueError("--names must match --corpora length")

    overall = {}
    for path, name in zip(a.corpora, names):
        print(f"\n--- Eval on {name} ({path}) ---")
        blob = load_chain_corpus(Path(path))
        metrics = evaluate_corpus(
            model, blob, device,
            score_window=a.score_window,
            oracle_window=a.oracle_window,
            do_rag=a.do_rag,
            rag_top_k=a.rag_top_k,
            rag_prefix_len=a.rag_prefix_len,
            embedder=embedder,
            tokenizer=tokenizer,
        )
        # Print summary.
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_chain"}, indent=2))
        overall[name] = metrics

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"\nSaved -> {a.output}")


if __name__ == "__main__":
    main()
