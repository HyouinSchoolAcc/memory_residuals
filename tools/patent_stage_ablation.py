#!/usr/bin/env python3
"""Patent Experiment 4 — three-stage ablation ladder.

Evaluates callback CE on LME val under an eval-time stage ladder using a
trained MemRes checkpoint. Conditions (per chain):

  nomem       : no memory at all (M_c=None) — S1/S2/S3 all absent.
  s1_zero     : S1 injection active but memory never written (M_c = 0)
                — "stage 1 alone".
  s1s2_last   : S1 + S2 without S3 — per-session extraction is performed
                but there is no old-vs-new competition: the memory is
                naively overwritten by the latest session's candidate
                (M_c := RMSNorm(E_t)).
  s1s2_mean   : S1 + S2 without S3, pooling variant — the memory is the
                running mean of all sessions' normalised candidates
                (no competition, uniform blending).
  full        : S1 + S2 + S3 — the canonical recurrent competitive update
                (M_c := judge(M_c_prev, E_t)); identical to eval_callback.py
                ce_mem.
  shuffle     : full pipeline but with another chain's M_c (content
                specificity control).

The extract pass (stage 2's E_t) is computed once per session and shared
across s1s2_last / s1s2_mean / full so all conditions see bit-identical
extraction outputs.

Usage:
    python tools/patent_stage_ablation.py \
        --model_path runs/chain_v27b_.../final \
        --corpus paper_artifacts/chains/lme_val_s512_evpos.pt \
        --output results/patent_experiments/stage_ablation_<seed>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def chain_session(blob, ci, j):
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + j].long()


def chain_session_mask(blob, ci, j):
    s = int(blob["chain_starts"][ci])
    return blob["session_callback_mask"][s + j].bool()


@torch.no_grad()
def build_all_memories(model, blob, ci, end, device):
    """One pass over chain[ci][:end] producing the three write-side variants.

    Returns (M_full, M_last, M_mean): each (1, K, d) bf16.
    """
    mb = model.model.memory_block
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_full = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    M_last = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    mean_acc = torch.zeros(1, K, d, device=device, dtype=torch.float32)
    for j in range(end):
        sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
        C = model.model.extract_source(sess[:, :-1])
        E = mb.extract(C)                       # stage-2 candidate (B, K, d)
        E_norm = mb.judge_norm(E)               # scale-matched, no competition
        M_full = mb.judge(M_full, E)            # stage-3 competitive update
        M_last = E_norm
        mean_acc += E_norm.float()
    M_mean = (mean_acc / max(1, end)).to(torch.bfloat16)
    return M_full, M_last, M_mean


@torch.no_grad()
def callback_loss(model, input_ids, M_c, callback_mask):
    out = model(input_ids=input_ids, M_c=M_c)
    target = input_ids[:, 1:]
    pred = out.logits[:, :-1, :]
    mask = callback_mask[1:].to(input_ids.device)
    if mask.sum() == 0:
        return float("nan")
    log_probs = torch.nn.functional.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return float(nll[0][mask].mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--n_chains_max", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    device = torch.device(a.device)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size

    blob = torch.load(a.corpus, map_location="cpu", weights_only=False)
    n_chains = int(blob["chain_starts"].shape[0])
    if a.n_chains_max is not None:
        n_chains = min(n_chains, a.n_chains_max)

    conds = ["nomem", "s1_zero", "s1s2_last", "s1s2_mean", "full", "shuffle"]
    ce = {c: [] for c in conds}
    per_chain = []

    # Pre-build the "full" memory of every chain once so the shuffle
    # condition can reuse the neighbour's memory (as eval_callback.py does).
    mem_cache: dict[int, tuple] = {}
    for ci in tqdm(range(n_chains), desc="build memories"):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
            mem_cache[ci] = None
            continue
        mem_cache[ci] = build_all_memories(model, blob, ci, cb_pos, device)

    M_zero = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)

    for ci in tqdm(range(n_chains), desc="score callbacks"):
        if mem_cache.get(ci) is None:
            continue
        cb_pos = int(blob["chain_callback_position"][ci])
        sess = chain_session(blob, ci, cb_pos).to(device).unsqueeze(0)
        cb_mask = chain_session_mask(blob, ci, cb_pos)
        if cb_mask is None or cb_mask.sum() == 0:
            continue
        M_full, M_last, M_mean = mem_cache[ci]
        # neighbour chain's full memory (same convention as eval_callback.py)
        shuf_idx = (ci + 1) % n_chains
        M_shuf = mem_cache[shuf_idx][0] if mem_cache.get(shuf_idx) else M_zero

        row = {
            "nomem": callback_loss(model, sess, None, cb_mask),
            "s1_zero": callback_loss(model, sess, M_zero, cb_mask),
            "s1s2_last": callback_loss(model, sess, M_last, cb_mask),
            "s1s2_mean": callback_loss(model, sess, M_mean, cb_mask),
            "full": callback_loss(model, sess, M_full, cb_mask),
            "shuffle": callback_loss(model, sess, M_shuf, cb_mask),
        }
        for c in conds:
            ce[c].append(row[c])
        row["chain_id"] = (
            blob["chain_names"][ci] if "chain_names" in blob else f"c{ci}"
        )
        per_chain.append(row)

    ce_mean = {c: mean(ce[c]) for c in conds}
    summary = {
        "model_path": a.model_path,
        "corpus": a.corpus,
        "n_chains_scored": len(per_chain),
        "ce_mean": ce_mean,
        "delta_vs_nomem": {
            c: ce_mean["nomem"] - ce_mean[c] for c in conds if c != "nomem"
        },
        "per_chain": per_chain,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chain"},
                     indent=2))
    print(f"Saved -> {a.output}")


if __name__ == "__main__":
    main()
