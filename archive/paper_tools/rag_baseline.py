#!/usr/bin/env python3
"""Dense-retrieval baseline over past-session chunks for pair JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_samples(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def chunks(text: str, max_chars: int) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for part in parts:
        if len(part) <= max_chars:
            out.append(part)
        else:
            out.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
    return out or [text[:max_chars]]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--num-eval", type=int, default=128)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--chunk-chars", type=int, default=1200)
    p.add_argument("--current-len", type=int, default=512)
    p.add_argument("--prefix-len", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("paper_artifacts/rag_metrics.json"))
    args = p.parse_args()

    samples = read_samples(args.data_path, args.num_eval)
    embedder = SentenceTransformer(args.embedder, device=args.device)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    lm = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(args.device).eval()
    losses = []
    for sample in tqdm(samples, desc="rag eval"):
        corpus = chunks(sample.get("history", ""), args.chunk_chars)
        if not corpus or not sample.get("current"):
            continue
        emb = embedder.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
        q = embedder.encode([sample["current"][: args.chunk_chars]], convert_to_tensor=True, normalize_embeddings=True)
        scores = (q @ emb.T).squeeze(0)
        top = torch.topk(scores, k=min(args.top_k, len(corpus))).indices.tolist()
        retrieved = "\n\n".join(corpus[i] for i in top)
        prefix_ids = tok.encode(retrieved, add_special_tokens=False)[-args.prefix_len :]
        current_ids = tok.encode(sample["current"], add_special_tokens=False)[: args.current_len + 1]
        if len(current_ids) < 2:
            continue
        full = torch.tensor(prefix_ids + current_ids, dtype=torch.long, device=args.device).unsqueeze(0)
        labels = full.clone()
        if prefix_ids:
            labels[:, : len(prefix_ids)] = -100
        losses.append(lm(input_ids=full, labels=labels).loss.item())
    metrics = {
        "n": len(losses),
        "rag_ce": sum(losses) / len(losses) if losses else float("nan"),
        "top_k": args.top_k,
        "embedder": args.embedder,
        "base_model": args.base_model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
