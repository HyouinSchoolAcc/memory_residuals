#!/usr/bin/env python3
"""Structured paper evaluation for Memory Residuals checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_memres import Qwen3MemResForCausalLM


def read_samples(path: Path, limit: int | None, offset: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def encode_pair(tokenizer, sample: dict, history_len: int, current_len: int):
    h = tokenizer.encode(sample.get("history", ""), add_special_tokens=False)[-history_len:]
    c = tokenizer.encode(sample.get("current", ""), add_special_tokens=False)[: current_len + 1]
    if len(h) < 16 or len(c) < 2:
        return None
    h = h + [tokenizer.eos_token_id] * (history_len - len(h))
    return (
        torch.tensor(h[:history_len], dtype=torch.long),
        torch.tensor(c[: current_len + 1], dtype=torch.long),
    )


@torch.no_grad()
def eval_memres(args, samples: list[dict]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = Qwen3MemResForCausalLM.from_pretrained(args.model_path).to(args.device).eval()
    encoded = [encode_pair(tokenizer, s, args.history_len, args.current_len) for s in samples]
    encoded = [x for x in encoded if x is not None]
    losses_mem, losses_nomem, losses_shuffle = [], [], []
    for i, (h_ids, c_ids) in enumerate(tqdm(encoded, desc="memres eval")):
        h_ids = h_ids.unsqueeze(0).to(args.device)
        c_ids = c_ids.unsqueeze(0).to(args.device)
        input_ids = c_ids[:, :-1]
        labels = input_ids.clone()
        M_c = model.model.compute_memory(h_ids)
        losses_mem.append(model(input_ids=input_ids, labels=labels, M_c=M_c).loss.item())
        losses_nomem.append(model(input_ids=input_ids, labels=labels, M_c=None).loss.item())
        if len(encoded) > 1:
            h_other = encoded[(i + 1) % len(encoded)][0].unsqueeze(0).to(args.device)
            M_other = model.model.compute_memory(h_other)
            losses_shuffle.append(model(input_ids=input_ids, labels=labels, M_c=M_other).loss.item())
    out = {
        "n": len(encoded),
        "with_memory_ce": mean(losses_mem),
        "no_memory_ce": mean(losses_nomem),
        "delta_no_memory_minus_memory": mean(losses_nomem) - mean(losses_mem),
        "shuffle_history_ce": mean(losses_shuffle),
        "delta_shuffle_minus_memory": mean(losses_shuffle) - mean(losses_mem),
    }
    return out


@torch.no_grad()
def eval_base(args, samples: list[dict]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(args.device).eval()
    losses = []
    for sample in tqdm(samples, desc="base eval"):
        c = tokenizer.encode(sample.get("current", ""), add_special_tokens=False)[: args.current_len + 1]
        if len(c) < 2:
            continue
        c_ids = torch.tensor(c, dtype=torch.long, device=args.device).unsqueeze(0)
        input_ids = c_ids[:, :-1]
        losses.append(model(input_ids=input_ids, labels=input_ids).loss.item())
    return {"n": len(losses), "base_ce": mean(losses)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--history-len", type=int, default=1024)
    p.add_argument("--current-len", type=int, default=512)
    p.add_argument("--num-eval", type=int, default=128)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("paper_artifacts/eval_metrics.json"))
    p.add_argument("--skip-base", action="store_true")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_path

    samples = read_samples(args.data_path, args.num_eval, args.offset)
    metrics = {
        "model_path": args.model_path,
        "data_path": str(args.data_path),
        "memres": eval_memres(args, samples),
    }
    if not args.skip_base:
        metrics["base"] = eval_base(args, samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
