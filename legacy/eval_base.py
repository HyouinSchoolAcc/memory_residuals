"""Baseline CE on `current` with the untrained base Qwen (no memory)."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--data_path", required=True)
    p.add_argument("--current_len", type=int, default=256)
    p.add_argument("--num_eval", type=int, default=32)
    p.add_argument("--eval_start", type=int, default=200)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    a = parse_args()
    device = torch.device(a.device)

    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForCausalLM.from_pretrained(a.base_model).to(device).eval()

    with open(a.data_path) as f:
        samples = [json.loads(l) for l in f.readlines()[a.eval_start : a.eval_start + a.num_eval]]

    losses = []
    for s in samples:
        if not s.get("current"):
            continue
        c_ids = tok.encode(s["current"], add_special_tokens=False)[: a.current_len + 1]
        if len(c_ids) < 2:
            continue
        c = torch.tensor(c_ids, dtype=torch.long, device=device).unsqueeze(0)
        input_ids = c[:, :-1]
        labels = input_ids.clone()
        losses.append(model(input_ids=input_ids, labels=labels).loss.item())

    print(f"n = {len(losses)}")
    print(f"mean CE base (no memory, no finetune): {sum(losses) / len(losses):.4f}")


if __name__ == "__main__":
    main()
