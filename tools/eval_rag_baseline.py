#!/usr/bin/env python3
"""Simple RAG baseline against the same callback-CE metric as eval_callback.py.

For each chain in the eval corpus:
    * decode every prior (filler / evidence) session to plain text
    * decode the *question* from the callback session (text before "\nAssistant:")
    * retrieve top-k prior sessions with one of:
        - bm25     : rank_bm25 over whitespace-tokenised text
        - dense    : MiniLM (sentence-transformers/all-MiniLM-L6-v2) cosine
        - oracle   : the ground-truth chain_evidence_positions
        - shuffle  : retrieve from a *different* chain (chain-specificity probe)
    * concat the retrieved sessions' token ids in front of the callback
      session's token ids
    * forward a *bare* (non-memres) base LM on the concat sequence
    * compute CE only over the callback-mask positions of the callback session

Metric definitions (mirror eval_callback.py):
    pa_cb_drag = ce_nomem - ce_rag
    pa_cb_dsh  = ce_shuffle - ce_rag      (shuffle: retrieve from another chain)

Comparison row of interest:
    memres v27b 0.6B (n=4 mean): pa_cb_dnm = +1.32 ± 0.53 nats
    memres v28  1.7B (n=2 mean): pa_cb_dnm = +0.93 nats
    nomem baseline ce ≈ 4.09 nats (will be reproduced by --rag_method none).

Usage examples:
    python tools/eval_rag_baseline.py \
        --base_model Qwen/Qwen3-0.6B \
        --corpus paper_artifacts/chains/lme_val_s512_evpos.pt \
        --rag_method bm25 --top_k 3 \
        --output results/rag_baseline/qwen3_0p6b_bm25_top3.json

    python tools/eval_rag_baseline.py \
        --base_model Qwen/Qwen3-0.6B \
        --corpus paper_artifacts/chains/lme_val_s512_evpos.pt \
        --rag_method dense --top_k 3 \
        --encoder sentence-transformers/all-MiniLM-L6-v2 \
        --output results/rag_baseline/qwen3_0p6b_dense_top3.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# --------------------------------------------------------------------------- #
# corpus helpers (mirror eval_callback.py)
# --------------------------------------------------------------------------- #

def load_blob(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def chain_session(blob: dict, ci: int, j: int) -> torch.Tensor:
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + j].long()


def chain_session_mask(blob: dict, ci: int, j: int) -> torch.Tensor | None:
    if "session_callback_mask" not in blob:
        return None
    s = int(blob["chain_starts"][ci])
    return blob["session_callback_mask"][s + j].bool()


def strip_pad(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Drop trailing pad tokens. Keeps any in-content tokens (rare)."""
    if ids.numel() == 0:
        return ids
    nonpad = (ids != pad_id).long()
    if nonpad.sum() == 0:
        return ids[:0]
    last = int(nonpad.cumsum(0).argmax().item())
    return ids[: last + 1]


def decode_sessions(
    blob: dict, ci: int, end: int, tok, pad_id: int
) -> tuple[list[torch.Tensor], list[str]]:
    """Return (token_id_tensors, decoded_texts) for sessions 0..end of chain ci."""
    id_list, text_list = [], []
    for j in range(end):
        ids = chain_session(blob, ci, j)
        ids = strip_pad(ids, pad_id)
        id_list.append(ids)
        text_list.append(tok.decode(ids.tolist()))
    return id_list, text_list


def extract_question_text(callback_text: str) -> str:
    """Grab the user-question portion of the callback session.

    Each session is rendered as "User: ...\\nAssistant: ...\\n<|im_end|>".
    The question is the User: line up to the first \\nAssistant:.
    """
    m = re.search(r"User:\s*(.*?)\nAssistant:", callback_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: first 256 chars
    return callback_text[:256]


# --------------------------------------------------------------------------- #
# retrievers
# --------------------------------------------------------------------------- #

def bm25_retrieve(corpus_texts: list[str], query: str, top_k: int) -> list[int]:
    from rank_bm25 import BM25Okapi
    if not corpus_texts:
        return []
    tokenised_corpus = [re.findall(r"\w+", t.lower()) for t in corpus_texts]
    tokenised_query = re.findall(r"\w+", query.lower())
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(tokenised_query)
    order = scores.argsort()[::-1].tolist()
    return order[:top_k]


class DenseEncoder:
    def __init__(self, name: str, device: torch.device):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=torch.float32).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            h = self.model(**enc).last_hidden_state
            mask = enc.attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.cpu())
        return torch.cat(out, dim=0)


def dense_retrieve(
    encoder: DenseEncoder, corpus_texts: list[str], query: str, top_k: int
) -> list[int]:
    if not corpus_texts:
        return []
    corpus_emb = encoder.embed(corpus_texts)
    query_emb = encoder.embed([query])
    sims = (corpus_emb @ query_emb.T).squeeze(-1)
    order = sims.argsort(descending=True).tolist()
    return order[:top_k]


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #

@torch.no_grad()
def callback_loss_concat(
    model,
    prefix_ids: torch.Tensor,           # 1D tensor of pre-callback context tokens (may be empty)
    callback_ids: torch.Tensor,          # 1D tensor of unpadded callback session tokens
    callback_mask_full: torch.Tensor,    # 1D bool, len(callback_ids), True at callback target positions
    device,
    max_total_len: int = 8192,
) -> float:
    """CE averaged over callback positions in callback session (target = next-token).

    The eval_callback.py reference computes:
        target = input_ids[:, 1:]
        pred   = logits[:, :-1, :]
        nll[mask[1:]].mean()
    so the callback mask is shifted by one (target predicts next token at masked
    position-1). We replicate that: build full input = prefix + callback, build a
    full mask that is False over the prefix and equal to callback_mask_full over
    the callback portion, then evaluate nll[full_mask[1:]].mean() exactly.
    """
    if prefix_ids is None:
        prefix_ids = torch.zeros(0, dtype=callback_ids.dtype)

    full = torch.cat([prefix_ids, callback_ids], dim=0)
    full_mask = torch.cat(
        [torch.zeros(prefix_ids.numel(), dtype=torch.bool), callback_mask_full],
        dim=0,
    )
    if full.numel() > max_total_len:
        # truncate from the LEFT of the prefix; never drop callback content
        keep = max_total_len
        cb_len = callback_ids.numel()
        keep_pref = max(0, keep - cb_len)
        full = torch.cat([prefix_ids[-keep_pref:], callback_ids], dim=0)
        full_mask = torch.cat(
            [torch.zeros(min(keep_pref, prefix_ids.numel()), dtype=torch.bool), callback_mask_full],
            dim=0,
        )

    input_ids = full.unsqueeze(0).to(device)
    out = model(input_ids=input_ids)
    logits = out.logits  # (1, S, V)
    target = input_ids[:, 1:]
    pred = logits[:, :-1, :]
    mask = full_mask[1:].to(device)
    if mask.sum() == 0:
        return float("nan")
    log_probs = torch.nn.functional.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return float(nll[0][mask].mean().item())


# --------------------------------------------------------------------------- #
# main eval loop
# --------------------------------------------------------------------------- #

def evaluate(
    model,
    blob,
    tok,
    rag_method: str,
    top_k: int,
    device,
    max_total_len: int = 8192,
    n_chains_max: int | None = None,
    encoder: DenseEncoder | None = None,
    score_nomem: bool = True,
    score_shuffle: bool = True,
    pad_id: int | None = None,
):
    if pad_id is None:
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id

    n_chains = int(blob["chain_starts"].shape[0])
    if n_chains_max is not None:
        n_chains = min(n_chains_max, n_chains)

    has_callback_pos = "chain_callback_position" in blob
    has_callback_mask = "session_callback_mask" in blob
    if not (has_callback_pos and has_callback_mask):
        raise RuntimeError("corpus lacks callback annotations; cannot eval")

    # pre-cache decoded prior sessions per chain (tokens + text)
    cache_ids: dict[int, list[torch.Tensor]] = {}
    cache_text: dict[int, list[str]] = {}
    cache_cb_question: dict[int, str] = {}
    cache_cb_ids: dict[int, torch.Tensor] = {}
    cache_cb_mask: dict[int, torch.Tensor] = {}

    def _prepare_chain(ci):
        if ci in cache_ids:
            return
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
            cache_ids[ci] = []
            cache_text[ci] = []
            cache_cb_question[ci] = ""
            cache_cb_ids[ci] = torch.zeros(0, dtype=torch.long)
            cache_cb_mask[ci] = torch.zeros(0, dtype=torch.bool)
            return
        cb_ids_full = chain_session(blob, ci, cb_pos)
        cb_mask_full = chain_session_mask(blob, ci, cb_pos)
        cb_ids = strip_pad(cb_ids_full, pad_id)
        # mask aligned to cb_ids (truncated tail removed alongside pad)
        if cb_mask_full is not None:
            cb_mask = cb_mask_full[: cb_ids.numel()].clone()
        else:
            cb_mask = torch.zeros(cb_ids.numel(), dtype=torch.bool)
        ids_list, text_list = decode_sessions(blob, ci, cb_pos, tok, pad_id)
        cache_ids[ci] = ids_list
        cache_text[ci] = text_list
        cache_cb_question[ci] = extract_question_text(tok.decode(cb_ids.tolist()))
        cache_cb_ids[ci] = cb_ids
        cache_cb_mask[ci] = cb_mask

    # pre-embed all priors per chain for dense retrieval (cheap for 50 chains)
    chain_dense: dict[int, torch.Tensor] = {}

    per_chain = []
    ce_rag, ce_nomem, ce_shuffle = [], [], []
    for ci in tqdm(range(n_chains), desc=f"rag={rag_method} k={top_k}"):
        _prepare_chain(ci)
        cb_ids = cache_cb_ids[ci]
        cb_mask = cache_cb_mask[ci]
        if cb_ids.numel() == 0 or cb_mask.sum() == 0:
            continue

        priors_ids = cache_ids[ci]
        priors_text = cache_text[ci]
        question = cache_cb_question[ci]

        if rag_method == "none":
            retrieved_idx: list[int] = []
        elif rag_method == "bm25":
            retrieved_idx = bm25_retrieve(priors_text, question, top_k)
        elif rag_method == "dense":
            assert encoder is not None
            if ci not in chain_dense:
                chain_dense[ci] = (
                    encoder.embed(priors_text)
                    if priors_text else torch.zeros(0, encoder.model.config.hidden_size)
                )
            corpus_emb = chain_dense[ci]
            if corpus_emb.shape[0] == 0:
                retrieved_idx = []
            else:
                qe = encoder.embed([question])
                sims = (corpus_emb @ qe.T).squeeze(-1)
                retrieved_idx = sims.argsort(descending=True).tolist()[:top_k]
        elif rag_method == "oracle":
            ev = blob.get("chain_evidence_positions", [None] * n_chains)
            cb_pos = int(blob["chain_callback_position"][ci])
            if ev[ci] is None:
                retrieved_idx = []
            else:
                evp = sorted({int(p) for p in ev[ci] if 0 <= int(p) < cb_pos})
                # if more than top_k evidence, take the first top_k (chronological)
                retrieved_idx = evp[:top_k]
        else:
            raise ValueError(f"unknown rag_method {rag_method}")

        # build prefix in chronological order
        retrieved_idx_chron = sorted(retrieved_idx)
        prefix_pieces = [priors_ids[j] for j in retrieved_idx_chron]
        prefix_ids = (
            torch.cat(prefix_pieces, dim=0)
            if prefix_pieces else torch.zeros(0, dtype=torch.long)
        )

        ce_r = callback_loss_concat(
            model, prefix_ids, cb_ids, cb_mask, device, max_total_len=max_total_len
        )

        if score_nomem:
            ce_n = callback_loss_concat(
                model,
                torch.zeros(0, dtype=torch.long),
                cb_ids,
                cb_mask,
                device,
                max_total_len=max_total_len,
            )
        else:
            ce_n = float("nan")

        if score_shuffle and rag_method not in ("none", "oracle"):
            shuffle_ci = (ci + 1) % n_chains
            _prepare_chain(shuffle_ci)
            shuf_priors_ids = cache_ids[shuffle_ci]
            shuf_priors_text = cache_text[shuffle_ci]
            if rag_method == "bm25":
                shuf_idx = bm25_retrieve(shuf_priors_text, question, top_k)
            elif rag_method == "dense":
                if shuffle_ci not in chain_dense:
                    chain_dense[shuffle_ci] = (
                        encoder.embed(shuf_priors_text)
                        if shuf_priors_text
                        else torch.zeros(0, encoder.model.config.hidden_size)
                    )
                corpus_emb = chain_dense[shuffle_ci]
                if corpus_emb.shape[0] == 0:
                    shuf_idx = []
                else:
                    qe = encoder.embed([question])
                    sims = (corpus_emb @ qe.T).squeeze(-1)
                    shuf_idx = sims.argsort(descending=True).tolist()[:top_k]
            else:
                shuf_idx = []
            shuf_idx_chron = sorted(shuf_idx)
            shuf_pieces = [shuf_priors_ids[j] for j in shuf_idx_chron]
            shuf_prefix = (
                torch.cat(shuf_pieces, dim=0)
                if shuf_pieces else torch.zeros(0, dtype=torch.long)
            )
            ce_s = callback_loss_concat(
                model, shuf_prefix, cb_ids, cb_mask, device, max_total_len=max_total_len
            )
        else:
            ce_s = float("nan")

        ce_rag.append(ce_r)
        if not math.isnan(ce_n):
            ce_nomem.append(ce_n)
        if not math.isnan(ce_s):
            ce_shuffle.append(ce_s)
        per_chain.append({
            "chain_id": blob["chain_names"][ci] if "chain_names" in blob else f"c{ci}",
            "callback_pos": int(blob["chain_callback_position"][ci]),
            "n_callback_tok": int(cb_mask.sum().item()),
            "retrieved": [int(x) for x in retrieved_idx_chron],
            "prefix_tokens": int(prefix_ids.numel()),
            "ce_rag": ce_r,
            "ce_nomem": ce_n,
            "ce_shuffle": ce_s,
            "evidence_positions": list(blob["chain_evidence_positions"][ci])
                if "chain_evidence_positions" in blob else [],
        })

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def _stdev(xs):
        return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0

    summary = {
        "n_chains_scored": len(per_chain),
        "ce_rag": _mean(ce_rag),
        "ce_nomem": _mean(ce_nomem),
        "ce_shuffle": _mean(ce_shuffle),
        "pa_cb_drag": _mean(ce_nomem) - _mean(ce_rag),
        "pa_cb_dsh": _mean(ce_shuffle) - _mean(ce_rag),
        "stdev_per_chain_drag": _stdev(
            [pc["ce_nomem"] - pc["ce_rag"] for pc in per_chain
             if not math.isnan(pc["ce_nomem"]) and not math.isnan(pc["ce_rag"])]
        ),
        "n_chains_per_chain_positive": sum(
            1 for pc in per_chain
            if not math.isnan(pc["ce_nomem"])
            and not math.isnan(pc["ce_rag"])
            and pc["ce_nomem"] > pc["ce_rag"]
        ),
    }
    return {**summary, "per_chain": per_chain}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--corpus", required=True)
    p.add_argument("--rag_method", choices=["none", "bm25", "dense", "oracle"], default="bm25")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max_total_len", type=int, default=8192)
    p.add_argument("--n_chains_max", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    device = torch.device(a.device)

    tok = AutoTokenizer.from_pretrained(a.base_model)
    print(f"[load] base LM = {a.base_model} dtype=bfloat16")
    model = (
        AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    encoder = None
    if a.rag_method == "dense":
        print(f"[load] dense encoder = {a.encoder}")
        encoder = DenseEncoder(a.encoder, device)

    blob = load_blob(Path(a.corpus))
    out = evaluate(
        model,
        blob,
        tok,
        a.rag_method,
        a.top_k,
        device,
        max_total_len=a.max_total_len,
        n_chains_max=a.n_chains_max,
        encoder=encoder,
    )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({
        "config": {
            "base_model": a.base_model,
            "corpus": a.corpus,
            "rag_method": a.rag_method,
            "top_k": a.top_k,
            "encoder": a.encoder if a.rag_method == "dense" else None,
            "max_total_len": a.max_total_len,
            "n_chains_max": a.n_chains_max,
        },
        **out,
    }, indent=2))

    print()
    print(f"=== {a.rag_method} top_k={a.top_k} on {Path(a.corpus).name} ===")
    print(f"  n={out['n_chains_scored']} ce_rag={out['ce_rag']:.4f} "
          f"ce_nomem={out['ce_nomem']:.4f} ce_shuf={out['ce_shuffle']:.4f}")
    print(f"  pa_cb_drag={out['pa_cb_drag']:+.4f}  "
          f"pa_cb_dsh={out['pa_cb_dsh']:+.4f}  "
          f"per-chain RAG-positive={out['n_chains_per_chain_positive']}/{out['n_chains_scored']}")
    print(f"  saved -> {a.output}")


if __name__ == "__main__":
    main()
