#!/usr/bin/env python3
"""Patent Experiment 1 — O(1) inference cost vs. extended-context baseline.

Compares, as a function of total conversation-history length H (tokens):

  Baseline ("extended context window"): bare Qwen3 backbone that carries
  the entire history in its context. Per inference call it must
  (a) prefill H history tokens + the 512-token current session, and
  (b) hold a KV cache proportional to H while decoding.

  MemRes (this invention): frozen Qwen3 + fixed K x d memory M_c.
  History is absorbed into M_c at session boundaries (constant cost per
  session); per inference call it prefills only the 512-token current
  session + reads the constant-size M_c.

Measured per H:
  - prefill latency (ms)
  - mean per-token decode latency over 64 greedy steps (ms)
  - peak VRAM during the call (GiB)
  - persistent state size (KV-cache bytes vs M_c bytes)
  - MemRes only: per-session write cost (extract+judge, ms) — constant.

Real dialogue tokens are used (sessions concatenated from the LME val
corpus). Each measurement is repeated `--reps` times after a warmup; the
median is reported.

Usage:
    python tools/patent_o1_bench.py \
        --memres_path runs/chain_v27b_.../final \
        --output results/patent_experiments/o1_bench_0p6b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

SESSION_LEN = 512
DECODE_STEPS = 64


def sync():
    torch.cuda.synchronize()


def timed(fn, reps: int) -> float:
    """Median wall time (ms) of fn over reps, after one warmup call."""
    fn()
    sync()
    outs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        outs.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(outs))


def history_tokens(blob, n_tokens: int, device) -> torch.Tensor:
    """Concatenate real LME sessions (cycling chains) to n_tokens length."""
    ids = []
    total = 0
    n_sessions = blob["session_ids"].shape[0]
    j = 0
    while total < n_tokens:
        ids.append(blob["session_ids"][j % n_sessions].long())
        total += ids[-1].numel()
        j += 1
    return torch.cat(ids)[:n_tokens].unsqueeze(0).to(device)


def cache_nbytes(cache) -> int:
    total = 0
    for layer in cache:
        for t in layer:
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def bench_prefill_decode(model, prompt_ids, extra_kwargs, reps):
    """Returns (prefill_ms, per_token_decode_ms, peak_gib, kv_bytes)."""
    device = prompt_ids.device

    def prefill():
        return model(input_ids=prompt_ids, use_cache=True,
                     logits_to_keep=1, **extra_kwargs)

    prefill_ms = timed(prefill, reps)

    torch.cuda.reset_peak_memory_stats(device)
    out = prefill()
    kv_bytes = cache_nbytes(out.past_key_values)

    # greedy decode DECODE_STEPS tokens with the live cache
    cache = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    pos = prompt_ids.shape[1]
    sync()
    t0 = time.perf_counter()
    for i in range(DECODE_STEPS):
        step = model(
            input_ids=nxt, use_cache=True, past_key_values=cache,
            cache_position=torch.tensor([pos + i], device=device),
            **extra_kwargs,
        )
        nxt = step.logits[:, -1:].argmax(-1)
    sync()
    decode_ms = (time.perf_counter() - t0) * 1e3 / DECODE_STEPS
    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
    del cache, out
    torch.cuda.empty_cache()
    return prefill_ms, decode_ms, peak_gib, kv_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memres_path", required=True)
    p.add_argument("--baseline", default="Qwen/Qwen3-0.6B")
    p.add_argument("--corpus",
                   default="paper_artifacts/chains/lme_val_s512_evpos.pt")
    p.add_argument("--history_lens", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    device = torch.device(a.device)
    blob = torch.load(a.corpus, map_location="cpu", weights_only=False)
    cur_session = history_tokens(blob, SESSION_LEN, device)

    results = {"baseline": [], "memres": [], "meta": {
        "session_len": SESSION_LEN, "decode_steps": DECODE_STEPS,
        "reps": a.reps, "baseline_model": a.baseline,
        "memres_path": a.memres_path,
        "gpu": torch.cuda.get_device_name(device),
    }}

    # ---------------- baseline: full history in context -------------------
    base = AutoModelForCausalLM.from_pretrained(
        a.baseline, dtype=torch.bfloat16).to(device).eval()
    for H in a.history_lens:
        prompt = torch.cat(
            [history_tokens(blob, H, device), cur_session], dim=1)
        pre, dec, peak, kv = bench_prefill_decode(base, prompt, {}, a.reps)
        row = {"H": H, "prompt_len": int(prompt.shape[1]),
               "prefill_ms": pre, "decode_ms_per_tok": dec,
               "peak_vram_gib": peak, "kv_cache_bytes": kv}
        results["baseline"].append(row)
        print("baseline", json.dumps(row))
    del base
    torch.cuda.empty_cache()

    # ---------------- MemRes: constant window + M_c -----------------------
    mem = Qwen3MemResForCausalLM.from_pretrained(
        a.memres_path, dtype=torch.bfloat16).to(device).eval()
    K, d = mem.config.memres_num_vectors, mem.config.hidden_size
    mc_bytes = K * d * 2  # bf16

    with torch.no_grad():
        for H in a.history_lens:
            hist = history_tokens(blob, H, device)
            n_sessions = H // SESSION_LEN
            # session-boundary write cost (constant per session)
            M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
            per_session_ms = []
            sync()
            for j in range(n_sessions):
                sess = hist[:, j * SESSION_LEN:(j + 1) * SESSION_LEN]
                t0 = time.perf_counter()
                C = mem.model.extract_source(sess)
                M_c = mem.model.compress_session(C, M_c)
                sync()
                per_session_ms.append((time.perf_counter() - t0) * 1e3)
            kwargs = {"M_c": M_c}
            pre, dec, peak, kv = bench_prefill_decode(
                mem, cur_session, kwargs, a.reps)
            row = {"H": H, "prompt_len": SESSION_LEN,
                   "prefill_ms": pre, "decode_ms_per_tok": dec,
                   "peak_vram_gib": peak, "kv_cache_bytes": kv,
                   "mc_bytes": mc_bytes, "n_sessions_written": n_sessions,
                   "write_ms_per_session_median":
                       float(statistics.median(per_session_ms)),
                   }
            results["memres"].append(row)
            print("memres  ", json.dumps(row))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {a.output}")


if __name__ == "__main__":
    main()
