#!/usr/bin/env python3
"""Patent RAG-fairness ablation — trained (LoRA) RAG baseline.

The reviewer critique: the patent's RAG comparison is asymmetric — a
*fitted* method (41.5M trained memory-side params) vs an *unfitted*
zero-shot RAG pipeline. This tool closes the gap by training a
LoRA-RAG baseline under a matched budget and identical recipe:

  * same backbone (frozen Qwen3-0.6B + LoRA r=64 on all attn/MLP
    projections ~= 40.4M trainable params vs the method's 41.5M),
  * same training data (LongMemEval-S train split, 450 chains),
  * same loss (LM CE on the callback session, callback tokens
    weighted 1+5.0),
  * same optimizer/schedule (AdamW lr 1e-4, 200 warmup, cosine,
    1000 steps, effective batch 8, grad-clip 1.0),
  * input format = the RAG deployment format: top-k retrieved prefix
    sessions (dense all-MiniLM-L6-v2, the strongest deployable
    retriever from the zero-shot sweep) concatenated before the
    callback session.

Eval mirrors results/rag_baseline/*.json: CE (and greedy exact-match
accuracy) over the callback tokens of the callback session,
conditioned on the retrieved context; Δ is reported against the same
frozen-base no-context reference used by every row of patent table 2.
The adapter can also be evaluated with oracle (evidence-annotated)
retrieval and on a transfer corpus (RealTalk).

Usage:
    # train
    python tools/patent_rag_trained.py train \
        --out_dir output/patent_lora_rag_seed1 --seed 1

    # eval (dense + oracle on LME val; add --corpus for transfer)
    python tools/patent_rag_trained.py eval \
        --adapter output/patent_lora_rag_seed1/adapter \
        --corpus paper_artifacts/chains/lme_val_s512_evpos.pt \
        --output results/patent_experiments/lora_rag_seed1_lmeval.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-0.6B"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------- data
def load_blob(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def chain_session_idx(blob, ci, j):
    return int(blob["chain_starts"][ci]) + j


def decode_all_sessions(blob, tok):
    ids = blob["session_ids"]
    texts = []
    for i in tqdm(range(ids.shape[0]), desc="decode sessions"):
        texts.append(tok.decode(ids[i].tolist(), skip_special_tokens=True))
    return texts


@torch.no_grad()
def dense_retrieval(blob, top_k, device):
    """For every chain with a callback session, return the top_k prefix
    session positions (chronological order) ranked by cosine similarity
    between the callback session and each prefix session (MiniLM)."""
    from sentence_transformers import SentenceTransformer
    tok = AutoTokenizer.from_pretrained(BASE)
    texts = decode_all_sessions(blob, tok)
    enc = SentenceTransformer(ENCODER, device=str(device))
    emb = enc.encode(texts, batch_size=256, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=True)
    retrieved = {}
    n_chains = int(blob["chain_starts"].shape[0])
    for ci in range(n_chains):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0:
            continue
        q = emb[chain_session_idx(blob, ci, cb_pos)]
        prefix = emb[chain_session_idx(blob, ci, 0):
                     chain_session_idx(blob, ci, cb_pos)]
        sims = prefix @ q
        k = min(top_k, cb_pos)
        top = torch.topk(sims, k).indices.tolist()
        retrieved[ci] = sorted(top)
    del enc, emb
    torch.cuda.empty_cache()
    return retrieved


def oracle_retrieval(blob, top_k):
    if "chain_evidence_positions" not in blob:
        return None
    retrieved = {}
    for ci in range(int(blob["chain_starts"].shape[0])):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0:
            continue
        ev = sorted({int(p) for p in blob["chain_evidence_positions"][ci]
                     if int(p) < cb_pos})
        retrieved[ci] = ev[:top_k] if ev else []
    return retrieved


def build_sample(blob, ci, retrieved_js, device, cb_weight=5.0):
    """input_ids (1,L); train per-token loss weights (1,L); eval mask (L,)
    over callback tokens of the callback session. Context tokens carry
    zero loss weight — they are conditioning only, like the retrieved
    prompt in deployment."""
    cb_pos = int(blob["chain_callback_position"][ci])
    parts = [blob["session_ids"][chain_session_idx(blob, ci, j)].long()
             for j in retrieved_js]
    cb_ids = blob["session_ids"][chain_session_idx(blob, ci, cb_pos)].long()
    cb_mask = blob["session_callback_mask"][
        chain_session_idx(blob, ci, cb_pos)].bool()
    ids = torch.cat(parts + [cb_ids]) if parts else cb_ids
    L, S = ids.numel(), cb_ids.numel()
    w = torch.zeros(L)
    w[L - S:] = 1.0
    w[L - S:] += cb_weight * cb_mask.float()
    m = torch.zeros(L, dtype=torch.bool)
    m[L - S:] = cb_mask
    return (ids.unsqueeze(0).to(device), w.unsqueeze(0).to(device),
            m.to(device))


# ------------------------------------------------------------------- train
def weighted_lm_loss(model, ids, w):
    out = model(input_ids=ids)
    pred = out.logits[:, :-1, :].float()
    target = ids[:, 1:]
    logp = torch.nn.functional.log_softmax(pred, dim=-1)
    nll = -logp.gather(2, target.unsqueeze(-1)).squeeze(-1)
    wt = w[:, 1:]
    return (nll * wt).sum() / wt.sum().clamp(min=1.0)


def cmd_train(a):
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    blob = load_blob(a.train_chains)
    cache = Path(a.retrieval_cache or
                 (ROOT / "results/patent_experiments/"
                  f"rag_retrieval_train_top{a.top_k}.json"))
    if cache.exists():
        retrieved = {int(k): v for k, v in
                     json.loads(cache.read_text()).items()}
        print(f"retrieval cache hit: {cache}")
    else:
        retrieved = dense_retrieval(blob, a.top_k, device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(retrieved))
        print(f"retrieval cached -> {cache}")
    chain_ids = sorted(retrieved.keys())

    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16).to(device)
    model = get_peft_model(base, LoraConfig(
        r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M (method budget ~41.5M)")
    model.train()

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=a.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / max(1, a.warmup)
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * t))

    g = torch.Generator().manual_seed(a.seed)
    micro = a.batch_size * a.grad_accum
    for step in tqdm(range(1, a.steps + 1), desc="LoRA-RAG train"):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        picks = [chain_ids[int(torch.randint(0, len(chain_ids), (1,),
                                             generator=g).item())]
                 for _ in range(micro)]
        for ci in picks:
            ids, w, _ = build_sample(blob, ci, retrieved[ci], device,
                                     cb_weight=a.callback_loss_weight)
            loss = weighted_lm_loss(model, ids, w) / micro
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), a.max_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            tqdm.write(f"step {step} lr {lr_at(step):.2e} "
                       f"last-loss {loss.item() * micro:.4f}")

    out = Path(a.out_dir) / "adapter"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    (Path(a.out_dir) / "train_config.json").write_text(json.dumps({
        "base": BASE, "encoder": ENCODER, "lora_r": a.lora_r,
        "trainable_params": n_train, "top_k": a.top_k,
        "steps": a.steps, "lr": a.lr, "warmup": a.warmup,
        "batch": [a.batch_size, a.grad_accum],
        "callback_loss_weight": a.callback_loss_weight,
        "seed": a.seed, "train_chains": str(a.train_chains),
    }, indent=2))
    print(f"adapter -> {out}")


# -------------------------------------------------------------------- eval
@torch.no_grad()
def callback_scores(model, ids, mask):
    out = model(input_ids=ids)
    pred = out.logits[:, :-1, :]
    target = ids[:, 1:]
    m = mask[1:]
    if m.sum() == 0:
        return float("nan"), float("nan")
    logp = torch.nn.functional.log_softmax(pred.float(), dim=-1)
    nll = -logp.gather(2, target.unsqueeze(-1)).squeeze(-1)
    correct = (pred.argmax(dim=-1) == target).float()
    return float(nll[0][m].mean()), float(correct[0][m].mean())


def cmd_eval(a):
    device = torch.device(a.device)
    blob = load_blob(a.corpus)
    dense = dense_retrieval(blob, a.top_k, device)
    oracle = oracle_retrieval(blob, a.top_k)

    base = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16).to(device).eval()
    model = base
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, a.adapter).eval()

    per_chain = []
    for ci in tqdm(sorted(dense.keys()), desc="eval"):
        cb_pos = int(blob["chain_callback_position"][ci])
        row = {"chain_id": (blob["chain_names"][ci]
                            if "chain_names" in blob else f"c{ci}"),
               "cb_pos": cb_pos, "dense_js": dense[ci]}

        ids, _, m = build_sample(blob, ci, dense[ci], device)
        row["ce_rag_dense"], row["acc_rag_dense"] = \
            callback_scores(model, ids, m)

        if oracle is not None and oracle.get(ci):
            ids, _, m = build_sample(blob, ci, oracle[ci], device)
            row["oracle_js"] = oracle[ci]
            row["ce_rag_oracle"], row["acc_rag_oracle"] = \
                callback_scores(model, ids, m)

        ids, _, m = build_sample(blob, ci, [], device)
        row["ce_nomem_model"], row["acc_nomem_model"] = \
            callback_scores(model, ids, m)
        if a.adapter:
            with model.disable_adapter():
                row["ce_nomem_base"], row["acc_nomem_base"] = \
                    callback_scores(model, ids, m)
        else:
            row["ce_nomem_base"] = row["ce_nomem_model"]
            row["acc_nomem_base"] = row["acc_nomem_model"]
        per_chain.append(row)

    def mean(key):
        xs = [r[key] for r in per_chain if key in r and r[key] == r[key]]
        return float(sum(xs) / len(xs)) if xs else None

    summary = {
        "config": {"base": BASE, "adapter": a.adapter, "corpus": str(a.corpus),
                   "encoder": ENCODER, "top_k": a.top_k},
        "n_chains_scored": len(per_chain),
        "ce_rag_dense": mean("ce_rag_dense"),
        "ce_rag_oracle": mean("ce_rag_oracle"),
        "ce_nomem_model": mean("ce_nomem_model"),
        "ce_nomem_base": mean("ce_nomem_base"),
        "acc_rag_dense": mean("acc_rag_dense"),
        "acc_rag_oracle": mean("acc_rag_oracle"),
        "acc_nomem_model": mean("acc_nomem_model"),
        "acc_nomem_base": mean("acc_nomem_base"),
        "per_chain": per_chain,
    }
    for key in ("ce_rag_dense", "ce_rag_oracle", "ce_nomem_model"):
        if summary[key] is not None:
            summary[f"delta_{key.removeprefix('ce_')}_vs_base_nomem"] = \
                summary["ce_nomem_base"] - summary[key]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chain"},
                     indent=2))
    print(f"Saved -> {a.output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train_chains",
                   default="paper_artifacts/chains/lme_train_s512.pt")
    t.add_argument("--out_dir", required=True)
    t.add_argument("--retrieval_cache", default=None)
    t.add_argument("--top_k", type=int, default=3)
    t.add_argument("--lora_r", type=int, default=64)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--grad_accum", type=int, default=2)
    t.add_argument("--max_norm", type=float, default=1.0)
    t.add_argument("--callback_loss_weight", type=float, default=5.0)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--device", default="cuda")

    e = sub.add_parser("eval")
    e.add_argument("--adapter", default=None)
    e.add_argument("--corpus",
                   default="paper_artifacts/chains/lme_val_s512_evpos.pt")
    e.add_argument("--top_k", type=int, default=3)
    e.add_argument("--device", default="cuda")
    e.add_argument("--output", type=Path, required=True)

    a = p.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    else:
        cmd_eval(a)


if __name__ == "__main__":
    main()
