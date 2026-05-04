#!/usr/bin/env python3
"""Audit A3: template-prior decomposition for v15 callback CE.

Builds the empirical category-conditional answer prior P(item | category)
from the D4v2 train blob's chain_names and scores per-token CE under that
prior on the val callback answer-spans, alongside `pa_cb_ce_mem` from each
checkpoint (memory enabled, full-prefix M_c, mirroring tools/eval_callback.py).

Per-chain CE under the template prior is

    CE_template_prior(chain) = -log P(item | category) / n_answer_tokens

i.e. the per-token surprisal of the chain's answer span under the train
marginal of items conditional on the category cue. This is mathematically
equivalent to "draw item ~ P(.|cat), tokenise, average per-token nll",
since for a single chain the per-token decomposition sums back to
-log P(item | cat).

The mean of CE_template_prior across chains is the template-prior baseline
that the joint backbone needs to BEAT to demonstrate that memory carries
content beyond what the LM pathway can learn from the cue-template alone.

Usage::

    CUDA_VISIBLE_DEVICES=1 python memory_residuals/tools/audit_a3_template_prior.py \\
        --n_chains 128 \\
        --out_json memory_residuals/results/exp2_chain_recipe/audit_a3_data.json \\
        --ckpts <path1> <path2> ... \\
        --ckpt_tags v15a_best v15b_best ...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402
from build_synthetic_persona_callback import (  # noqa: E402
    CLOSED_SET,
    ITEM_TYPE_PHRASE,
)


CHAIN_NAME_RE = re.compile(
    r"^synthetic_persona_callback_(\d+)_([a-zA-Z]+)_(.+)_n(\d+)ev$"
)


def parse_chain_name(name: str) -> tuple[str, str]:
    m = CHAIN_NAME_RE.match(name)
    if not m:
        raise ValueError(f"unparseable chain name: {name!r}")
    return m.group(2), m.group(3)


def build_prior(
    train_blob: dict, alpha: float = 0.5
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """Empirical P(item | category) from train chain_names with α-smoothing."""
    counts: dict[str, dict[str, int]] = {
        c: {it: 0 for it in CLOSED_SET[c]} for c in CLOSED_SET
    }
    for name in train_blob["chain_names"]:
        cat, item = parse_chain_name(name)
        if cat not in counts:
            raise ValueError(f"unknown category {cat!r} in chain {name!r}")
        if item not in counts[cat]:
            counts[cat][item] = 0
        counts[cat][item] += 1
    probs: dict[str, dict[str, float]] = {}
    for cat, ic in counts.items():
        items = list(CLOSED_SET[cat])
        c = torch.tensor([ic[it] + alpha for it in items], dtype=torch.float64)
        c /= c.sum()
        probs[cat] = dict(zip(items, c.tolist()))
    return probs, counts


@torch.no_grad()
def callback_loss(model, input_ids, M_c, callback_mask):
    out = model(input_ids=input_ids, M_c=M_c)
    logits = out.logits  # (1, S, V)
    target = input_ids[:, 1:]
    pred = logits[:, :-1, :]
    mask = callback_mask[1:].to(input_ids.device)
    if mask.sum() == 0:
        return float("nan"), 0
    log_probs = F.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    n = int(mask.sum().item())
    return float(nll[0][mask].sum().item() / n), n


@torch.no_grad()
def build_Mc(model, blob, ci, end, device):
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    starts = blob["chain_starts"]
    sess = blob["session_ids"]
    if end <= 0:
        return M_c
    for j in range(end):
        ids = sess[int(starts[ci]) + j].to(device).unsqueeze(0).long()
        C = model.model.extract_source(ids[:, :-1])
        M_c = model.model.compress_session(C, M_c)
    return M_c


def evaluate_ckpt(
    ckpt_path: Path,
    val_blob: dict,
    n_chains: int,
    device,
    prior: dict[str, dict[str, float]],
):
    print(f"  loading {ckpt_path}", flush=True)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(
            str(ckpt_path), dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    starts = val_blob["chain_starts"].tolist()
    cb_pos_all = val_blob["chain_callback_position"].tolist()
    cb_mask_all = val_blob["session_callback_mask"]
    sess = val_blob["session_ids"]
    names = val_blob["chain_names"]

    ce_mem_list: list[float] = []
    ce_prior_list: list[float] = []
    n_tok_list: list[int] = []
    multi_tok = 0
    n = min(n_chains, len(starts))
    for ci in tqdm(range(n), desc=str(ckpt_path)[-50:]):
        st = starts[ci]
        cb_pos = cb_pos_all[ci]
        cb_session = sess[st + cb_pos].to(device).unsqueeze(0).long()
        msk = cb_mask_all[st + cb_pos].bool()
        n_tok = int(msk.sum().item())
        if n_tok == 0:
            continue
        Mc = build_Mc(model, val_blob, ci, cb_pos, device)
        ce_mem, _ = callback_loss(model, cb_session, Mc, msk)
        ce_mem_list.append(ce_mem)
        cat, item = parse_chain_name(names[ci])
        # If the val blob ever picks an item outside the train prior's
        # support, fall back to α/total -- but with α-smoothing on the
        # closed set already this branch never fires for D4v2.
        p = prior.get(cat, {}).get(item)
        if p is None or p <= 0:
            # Tiny floor instead of -inf.
            p = 1e-12
        ce_prior_list.append(-math.log(p) / n_tok)
        n_tok_list.append(n_tok)
        if n_tok > 1:
            multi_tok += 1

    del model
    torch.cuda.empty_cache()
    return {
        "n_scored": len(ce_mem_list),
        "multi_tok": multi_tok,
        "avg_ans_toks": (
            float(sum(n_tok_list) / len(n_tok_list)) if n_tok_list else float("nan")
        ),
        "ce_mem_per_chain": ce_mem_list,
        "ce_prior_per_chain": ce_prior_list,
        "n_ans_tokens_per_chain": n_tok_list,
    }


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_chains", type=int, default=128)
    p.add_argument("--out_json", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--ckpt_tags", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=0.5)
    a = p.parse_args()

    if len(a.ckpts) != len(a.ckpt_tags):
        raise SystemExit("--ckpts and --ckpt_tags must have the same length")

    train_path = ROOT / "paper_artifacts/chains/synthd4v2_persona_callback_train_s512.pt"
    val_path = ROOT / "paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt"
    print(f"loading train blob {train_path}", flush=True)
    train_blob = torch.load(str(train_path), map_location="cpu", weights_only=False)
    print(f"loading val blob   {val_path}", flush=True)
    val_blob = torch.load(str(val_path), map_location="cpu", weights_only=False)

    prior, counts = build_prior(train_blob, alpha=a.alpha)

    print("=== prior summary (train counts per category, α={}) ===".format(a.alpha))
    for cat, ic in counts.items():
        total = sum(ic.values())
        max_item = max(ic.items(), key=lambda x: x[1])
        min_item = min(ic.items(), key=lambda x: x[1])
        print(
            f"  {cat:>10}  total={total:5d}  uniform=1/32={1/32:.4f}  "
            f"max={max_item[0]} ({max_item[1]/total:.4f})  "
            f"min={min_item[0]} ({min_item[1]/total:.4f})"
        )

    device = torch.device(a.device)
    out = {
        "alpha_smoothing": a.alpha,
        "n_chains_max": a.n_chains,
        "train_blob": str(train_path),
        "val_blob": str(val_path),
        "prior_counts": counts,
        "prior_probs": prior,
        "ckpts": [],
    }

    for tag, ckpt in zip(a.ckpt_tags, a.ckpts):
        print(f"\n=== eval {tag} ({ckpt}) ===", flush=True)
        try:
            r = evaluate_ckpt(Path(ckpt), val_blob, a.n_chains, device, prior)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            out["ckpts"].append({
                "tag": tag, "path": ckpt, "error": str(e),
            })
            torch.cuda.empty_cache()
            continue
        ce_mem_mean = mean(r["ce_mem_per_chain"])
        ce_prior_mean = mean(r["ce_prior_per_chain"])
        gap = ce_mem_mean - ce_prior_mean
        print(
            f"  n={r['n_scored']}  multi_tok={r['multi_tok']}  "
            f"avg_ans_toks={r['avg_ans_toks']:.2f}\n"
            f"  pa_cb_ce_mem    = {ce_mem_mean:.4f}\n"
            f"  CE_template_prior= {ce_prior_mean:.4f}\n"
            f"  Δ = pa_cb_ce_mem - CE_template_prior = {gap:+.4f}"
        )
        out["ckpts"].append({
            "tag": tag,
            "path": ckpt,
            "n_scored": r["n_scored"],
            "multi_tok": r["multi_tok"],
            "avg_ans_toks": r["avg_ans_toks"],
            "pa_cb_ce_mem": ce_mem_mean,
            "ce_template_prior": ce_prior_mean,
            "gap": gap,
            "ce_mem_per_chain": r["ce_mem_per_chain"],
            "ce_prior_per_chain": r["ce_prior_per_chain"],
            "n_ans_tokens_per_chain": r["n_ans_tokens_per_chain"],
        })

    out_path = Path(a.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
