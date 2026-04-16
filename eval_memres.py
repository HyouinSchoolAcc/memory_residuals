"""
Evaluate Memory Residuals on held-out (history, current) pairs.

Reports mean next-token cross-entropy on the `current` session:
  - with_memory:  model conditioned on compress_history(history_ids)
  - no_memory:    model run on current_ids alone
  - delta:        no_memory - with_memory  (positive => memory helps)

Usage:
    python eval_memres.py --model_path output/smoke_memres/final \\
        --data_path data/friends_scripts.jsonl --num_eval 32
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from modeling_memres import Qwen3MemResForCausalLM


class MemoryEvaluator:
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        device: torch.device,
        history_len: int,
        current_len: int,
    ):
        self.device = device
        self.history_len = history_len
        self.current_len = current_len
        self.model = (
            Qwen3MemResForCausalLM.from_pretrained(model_path).to(device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @torch.no_grad()
    def encode_pair(self, sample: dict):
        tok = self.tokenizer
        h_ids = tok.encode(sample["history"], add_special_tokens=False)[
            -self.history_len :
        ]
        c_ids = tok.encode(sample["current"], add_special_tokens=False)[
            : self.current_len + 1
        ]
        if len(h_ids) < 16 or len(c_ids) < 2:
            return None
        h_ids = h_ids + [tok.eos_token_id] * (self.history_len - len(h_ids))
        return (
            torch.tensor(h_ids[: self.history_len], dtype=torch.long),
            torch.tensor(c_ids[: self.current_len + 1], dtype=torch.long),
        )

    @torch.no_grad()
    def lm_loss(self, input_ids, labels, memory_state=None) -> float:
        return self.model(
            input_ids=input_ids, labels=labels, memory_state=memory_state
        ).loss.item()

    @torch.no_grad()
    def compress(self, history_ids: torch.Tensor) -> torch.Tensor:
        h_hidden = self.model.model(input_ids=history_ids).last_hidden_state
        return self.model.model.compress_history(h_hidden)

    def run(self, samples):
        losses_mem, losses_nomem = [], []
        for sample in samples:
            if not sample.get("history") or not sample.get("current"):
                continue
            enc = self.encode_pair(sample)
            if enc is None:
                continue
            h_ids, c_ids = enc
            h_ids = h_ids.unsqueeze(0).to(self.device)
            c_ids = c_ids.unsqueeze(0).to(self.device)

            input_ids = c_ids[:, :-1]
            labels = input_ids.clone()

            m_c = self.compress(h_ids)
            losses_mem.append(self.lm_loss(input_ids, labels, memory_state=m_c))
            losses_nomem.append(self.lm_loss(input_ids, labels, memory_state=None))
        return losses_mem, losses_nomem


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--history_len", type=int, default=1024)
    p.add_argument("--current_len", type=int, default=512)
    p.add_argument("--num_eval", type=int, default=32, help="held-out samples")
    p.add_argument(
        "--eval_start",
        type=int,
        default=200,
        help="index where eval split begins (train uses [0:eval_start))",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    evaluator = MemoryEvaluator(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        device=torch.device(args.device),
        history_len=args.history_len,
        current_len=args.current_len,
    )

    with open(args.data_path) as f:
        lines = f.readlines()
    held_out = [
        json.loads(line)
        for line in lines[args.eval_start : args.eval_start + args.num_eval]
    ]

    losses_mem, losses_nomem = evaluator.run(held_out)

    if not losses_mem:
        print("No valid eval samples.")
        return

    mean_mem = sum(losses_mem) / len(losses_mem)
    mean_nomem = sum(losses_nomem) / len(losses_nomem)
    print(f"n = {len(losses_mem)}")
    print(f"mean CE with_memory : {mean_mem:.4f}")
    print(f"mean CE no_memory   : {mean_nomem:.4f}")
    print(
        f"delta (nomem - mem) : {mean_nomem - mean_mem:+.4f}  "
        f"(positive => memory helps)"
    )


if __name__ == "__main__":
    main()
