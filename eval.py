"""
Evaluate from-scratch models on held-out data and simple benchmarks.

Usage:
    python eval.py --model_path output/scratch-baseline-d512-L12-20k/final --mode baseline
    python eval.py --model_path output/scratch-block-d512-L12-20k/final --mode block
    python eval.py --model_path output/scratch-full-d512-L12-20k/final --mode full
    python eval.py --model_path output/scratch-memory-d512-L12-20k/final --mode memory
"""

import argparse
import math
import os
import sys

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from modeling_attnres import Qwen3AttnResConfig, Qwen3AttnResForCausalLM
from modeling_memory_residuals import MemResForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument(
        "--mode", required=True, choices=["baseline", "block", "full", "memory"]
    )
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument(
        "--num_samples", type=int, default=200, help="Number of evaluation samples"
    )
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_model(model_path, mode, device):
    if mode == "baseline":
        model = Qwen3ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    elif mode == "memory":
        model = MemResForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    else:
        model = Qwen3AttnResForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    model.eval()
    return model


def eval_perplexity(
    model,
    tokenizer,
    seq_len,
    num_samples,
    device,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
):
    """Compute perplexity on a dataset."""
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    total_tokens = 0

    # Sliding window evaluation
    max_pos = min(input_ids.size(1), num_samples * seq_len)
    for begin in tqdm(range(0, max_pos, seq_len), desc="Evaluating"):
        end = min(begin + seq_len, input_ids.size(1))
        chunk = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(input_ids=chunk)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="sum")
        nll = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(nll.item())
        total_tokens += shift_labels.numel()

        if total_tokens >= num_samples * seq_len:
            break

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    return avg_nll, ppl, total_tokens


def eval_lambada(model, tokenizer, device, max_samples=500):
    """Evaluate on LAMBADA (last word prediction accuracy)."""
    ds = load_dataset("lambada", split="test")
    correct = 0
    total = 0

    for sample in tqdm(ds.select(range(min(max_samples, len(ds)))), desc="LAMBADA"):
        text = sample["text"]
        # Split into context and last word
        words = text.strip().split()
        if len(words) < 2:
            continue
        last_word = words[-1]
        context = " ".join(words[:-1])

        input_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
        target_ids = tokenizer(" " + last_word, add_special_tokens=False)["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            predicted_id = next_token_logits.argmax().item()

        if len(target_ids) > 0 and predicted_id == target_ids[0]:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    return acc, correct, total


def eval_hellaswag(model, tokenizer, device, max_samples=200):
    """Evaluate on HellaSwag (commonsense completion)."""
    ds = load_dataset("Rowan/hellaswag", split="validation")

    correct = 0
    total = 0

    for sample in tqdm(ds.select(range(min(max_samples, len(ds)))), desc="HellaSwag"):
        ctx = sample["ctx"]
        endings = sample["endings"]
        label = int(sample["label"])

        scores = []
        for ending in endings:
            text = ctx + " " + ending
            input_ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            # Score = avg log-prob of the ending tokens
            ctx_ids = tokenizer(ctx, return_tensors="pt")["input_ids"]
            ctx_len = ctx_ids.size(1)

            shift_logits = logits[:, ctx_len - 1 : -1, :]
            shift_labels = input_ids[:, ctx_len:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(
                -1
            )
            score = token_log_probs.mean().item()
            scores.append(score)

        if scores.index(max(scores)) == label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    return acc, correct, total


def eval_memory_recall(
    model, tokenizer, device, max_samples: int = 100
) -> tuple[float, int, int]:
    """
    Multi-session memory recall evaluation.

    Tests whether the model can answer factual questions about information
    introduced in a previous session.  Each probe consists of:
      - history: "User: My [attribute] is [value]."
      - query:   "User: What is my [attribute]? Assistant:"
      - target:  the [value] token(s)

    The model runs with history_ids → M_c → query, and we check whether the
    top-1 predicted next token matches the first token of [value].
    """
    probes = [
        ("My name is Alice.", "What is my name? Assistant:", "Alice"),
        (
            "My favourite colour is blue.",
            "What is my favourite colour? Assistant:",
            "blue",
        ),
        ("I live in Paris.", "Where do I live? Assistant:", "Paris"),
        ("My job is a doctor.", "What is my job? Assistant:", "doctor"),
        ("My cat is called Whiskers.", "What is my cat called? Assistant:", "Whiskers"),
        ("I was born in 1990.", "What year was I born? Assistant:", "1990"),
        ("My hobby is painting.", "What is my hobby? Assistant:", "painting"),
        ("I drive a Toyota.", "What car do I drive? Assistant:", "Toyota"),
    ]

    correct = 0
    total = 0

    for history_text, query_text, target_word in probes[:max_samples]:
        history_ids = tokenizer("User: " + history_text, return_tensors="pt")[
            "input_ids"
        ].to(device)
        query_ids = tokenizer("User: " + query_text, return_tensors="pt")[
            "input_ids"
        ].to(device)
        target_ids = tokenizer(" " + target_word, add_special_tokens=False)["input_ids"]

        if len(target_ids) == 0:
            continue

        with torch.no_grad():
            out = model(input_ids=query_ids, history_ids=history_ids)
            predicted = out.logits[0, -1, :].argmax().item()

        if predicted == target_ids[0]:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def main():
    args = parse_args()

    print(f"Loading {args.mode} model from {args.model_path}...")
    model = load_model(args.model_path, args.mode, args.device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params | mode={args.mode}")
    print()

    # 1. Perplexity on WikiText-2
    print("=" * 50)
    print("WikiText-2 Perplexity")
    print("=" * 50)
    nll, ppl, n_tokens = eval_perplexity(
        model, tokenizer, args.seq_len, args.num_samples, args.device
    )
    print(f"  Loss: {nll:.4f} | PPL: {ppl:.2f} | Tokens: {n_tokens}")
    print()

    # 2. LAMBADA accuracy
    print("=" * 50)
    print("LAMBADA (last word prediction)")
    print("=" * 50)
    acc, correct, total = eval_lambada(model, tokenizer, args.device)
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    print()

    # 3. HellaSwag
    print("=" * 50)
    print("HellaSwag (commonsense)")
    print("=" * 50)
    acc_hs, correct_hs, total_hs = eval_hellaswag(model, tokenizer, args.device)
    print(f"  Accuracy: {acc_hs:.4f} ({correct_hs}/{total_hs})")
    print()

    # 4. Memory recall (memory mode only — tests cross-session information retrieval)
    acc_mem, correct_mem, total_mem = 0.0, 0, 0
    if args.mode == "memory":
        print("=" * 50)
        print("Memory Recall (multi-session, PAPER.md eval)")
        print("=" * 50)
        acc_mem, correct_mem, total_mem = eval_memory_recall(
            model, tokenizer, args.device
        )
        print(f"  Accuracy: {acc_mem:.4f} ({correct_mem}/{total_mem})")
        print()

    # Summary
    print("=" * 50)
    print(f"SUMMARY ({args.mode})")
    print("=" * 50)
    print(f"  WikiText-2 PPL: {ppl:.2f}")
    print(f"  LAMBADA Acc:    {acc:.4f}")
    print(f"  HellaSwag Acc:  {acc_hs:.4f}")
    if args.mode == "memory":
        print(f"  Memory Recall:  {acc_mem:.4f} ({correct_mem}/{total_mem})")


if __name__ == "__main__":
    main()
