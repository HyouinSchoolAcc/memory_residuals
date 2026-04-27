#!/usr/bin/env python3
"""Dense-retrieval baseline using a *finetuned* MemRes backbone (memory off).

Same as ``rag_baseline.py`` but loads a Memory Residuals checkpoint and
runs the model with ``M_c=None`` so only the backbone path is exercised.
This gives the apples-to-apples comparison: same backbone weights, with
either RAG-retrieved context or a learned memory matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


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
    p.add_argument("--model-path", required=True, help="MemRes checkpoint dir")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--num-eval", type=int, default=256)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--chunk-chars", type=int, default=1200)
    p.add_argument("--current-len", type=int, default=512)
    p.add_argument("--prefix-len", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output", type=Path, default=Path("paper_artifacts/eval/rag_finetuned.json")
    )
    args = p.parse_args()

    samples = read_samples(args.data_path, args.num_eval)
    embedder = SentenceTransformer(args.embedder, device=args.device)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    lm = (
        Qwen3MemResForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    losses = []
    for sample in tqdm(samples, desc="rag-finetuned eval"):
        corpus = chunks(sample.get("history", ""), args.chunk_chars)
        if not corpus or not sample.get("current"):
            continue
        emb = embedder.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
        q = embedder.encode(
            [sample["current"][: args.chunk_chars]],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        scores = (q @ emb.T).squeeze(0)
        top = torch.topk(scores, k=min(args.top_k, len(corpus))).indices.tolist()
        retrieved = "\n\n".join(corpus[i] for i in top)
        prefix_ids = tok.encode(retrieved, add_special_tokens=False)[-args.prefix_len :]
        current_ids = tok.encode(sample["current"], add_special_tokens=False)[
            : args.current_len + 1
        ]
        if len(current_ids) < 2:
            continue
        full = (
            torch.tensor(prefix_ids + current_ids, dtype=torch.long, device=args.device)
            .unsqueeze(0)
        )
        labels = full.clone()
        if prefix_ids:
            labels[:, : len(prefix_ids)] = -100
        # Memory off — pure RAG comparison on the same finetuned backbone.
        out = lm(input_ids=full, labels=labels, M_c=None)
        losses.append(out.loss.item())
    metrics = {
        "n": len(losses),
        "rag_finetuned_ce": sum(losses) / len(losses) if losses else float("nan"),
        "top_k": args.top_k,
        "embedder": args.embedder,
        "model_path": args.model_path,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
