"""
Memory Residual probe: measure α-mass routed to M_c.

Builds paired prompts from a history:
  - "callback":  explicit callback to the history ("Remember when we...")
  - "filler":    generic non-callback continuation ("The quick brown fox...")

Both share the same `memory_state = compress_history(history)`. We run the
model with `collect_alpha_trace=True` and report mean α-mass on M_c
(per layer × sublayer) for each prompt type. Positive callback-vs-filler Δ
means the model routes more mass to memory when the prompt calls back to
the history.

Usage:
    python probe_memres.py --model_path output/smoke_memres/final \\
        --data_path data/friends_scripts.jsonl --num_samples 16
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from modeling_memres import Qwen3MemResForCausalLM


class RoutingProbe:
    CALLBACK_SUFFIXES = [
        "\n\n[Later that day.] Remember what we just talked about? Well, ",
        "\n\nOkay, so going back to what you just said earlier — ",
        "\n\nThinking about that whole thing we discussed, I realized ",
        "\n\nRachel: About that story you told me a minute ago — ",
    ]

    FILLER_SUFFIXES = [
        "\n\nThe quick brown fox jumps over the lazy dog. ",
        "\n\nIn geometry, a triangle is a three-sided polygon. ",
        "\n\nThe capital of France is Paris. The capital of Spain is Madrid. ",
        "\n\nWater boils at one hundred degrees celsius at sea level. ",
    ]

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        device: torch.device,
        history_len: int,
        probe_len: int,
    ):
        self.device = device
        self.history_len = history_len
        self.probe_len = probe_len
        self.model = (
            Qwen3MemResForCausalLM.from_pretrained(model_path).to(device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @torch.no_grad()
    def compress(self, history_text: str):
        tok = self.tokenizer
        ids = tok.encode(history_text, add_special_tokens=False)[-self.history_len :]
        if len(ids) < 16:
            return None
        ids = ids + [tok.eos_token_id] * (self.history_len - len(ids))
        ids_t = torch.tensor(
            ids[: self.history_len], dtype=torch.long, device=self.device
        ).unsqueeze(0)
        h = self.model.model(input_ids=ids_t).last_hidden_state
        return self.model.model.compress_history(h)

    @torch.no_grad()
    def alpha_mass_for_suffix(self, suffix_text: str, memory_state: torch.Tensor):
        ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)[
            : self.probe_len
        ]
        if len(ids) < 2:
            return None
        ids_t = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model(
            input_ids=ids_t, memory_state=memory_state, collect_alpha_trace=True
        )
        return [a.squeeze(0).mean().item() for a in out.alpha_trace]

    def run(self, samples):
        callback_traces, filler_traces = [], []
        for i, sample in enumerate(samples):
            if not sample.get("history"):
                continue
            m_c = self.compress(sample["history"])
            if m_c is None:
                continue

            cb_suffix = self.CALLBACK_SUFFIXES[i % len(self.CALLBACK_SUFFIXES)]
            fl_suffix = self.FILLER_SUFFIXES[i % len(self.FILLER_SUFFIXES)]

            cb = self.alpha_mass_for_suffix(cb_suffix, m_c)
            fl = self.alpha_mass_for_suffix(fl_suffix, m_c)
            if cb is None or fl is None:
                continue
            callback_traces.append(cb)
            filler_traces.append(fl)
        return callback_traces, filler_traces


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--history_len", type=int, default=512)
    p.add_argument(
        "--probe_len",
        type=int,
        default=32,
        help="token length of the suffix prompt to probe",
    )
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--eval_start", type=int, default=200)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def report(callback_traces, filler_traces):
    if not callback_traces:
        print("No valid samples.")
        return

    n = len(callback_traces)
    n_sites = len(callback_traces[0])
    cb_per_site = [sum(t[s] for t in callback_traces) / n for s in range(n_sites)]
    fl_per_site = [sum(t[s] for t in filler_traces) / n for s in range(n_sites)]

    print(f"n samples = {n}, n MemRes sites = {n_sites}")
    print(f"{'site':>6}  {'callback α→M_c':>15}  {'filler α→M_c':>15}  {'Δ':>8}")
    for s in range(n_sites):
        delta = cb_per_site[s] - fl_per_site[s]
        print(
            f"{s:>6}  {cb_per_site[s]:>15.4f}  {fl_per_site[s]:>15.4f}  {delta:>+8.4f}"
        )

    mean_cb = sum(cb_per_site) / n_sites
    mean_fl = sum(fl_per_site) / n_sites
    print(
        f"\nOverall: callback={mean_cb:.4f}  filler={mean_fl:.4f}  "
        f"Δ={mean_cb - mean_fl:+.4f}"
    )
    print("(positive Δ => model routes MORE mass to memory on callback prompts)")


def main():
    args = parse_args()

    probe = RoutingProbe(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        device=torch.device(args.device),
        history_len=args.history_len,
        probe_len=args.probe_len,
    )

    with open(args.data_path) as f:
        lines = f.readlines()
    samples = [
        json.loads(line)
        for line in lines[args.eval_start : args.eval_start + args.num_samples]
    ]

    callback_traces, filler_traces = probe.run(samples)
    report(callback_traces, filler_traces)


if __name__ == "__main__":
    main()
