#!/usr/bin/env python3
"""Audit A1 (v15 OPEN AUDIT, candidate leak #1).

(1) Window-collision counts: for k in {1,2,3,4,5}, count chains where any
    evidence session falls inside [callback_pos - k + 1, callback_pos].
    Reports per-rank and offset breakdown.
(2) Frozen-base CE on callback tokens with no memory at all (single
    callback session, no chain prefix, no fine-tuning). Mirrors
    tools/audit_base_prior.py but with n=128 by default.

Outputs JSON to memory_residuals/results/exp2_chain_recipe/audit_a1_window_leakage.json
plus pretty stdout for the markdown report.

Usage:
    CUDA_VISIBLE_DEVICES=0 python memory_residuals/tools/audit_a1_window_leakage.py \
        --n_chains 128 --models 0.6B
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "exp2_chain_recipe"
CHAINS_DIR = ROOT / "paper_artifacts" / "chains"

MODEL_REGISTRY = {
    "0.6B": ("Qwen/Qwen3-0.6B", "0.6B-base"),
    "1.7B": ("Qwen/Qwen3-1.7B", "1.7B-base"),
}


def load_blob(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def collision_stats(blob: dict, ks: list[int]) -> dict:
    """Compute window-collision statistics for a single blob."""
    cb_pos_all = blob["chain_callback_position"].tolist()
    ev_pos_all = blob["chain_evidence_positions"]
    chain_lengths = blob["chain_lengths"].tolist() if "chain_lengths" in blob else None
    n_chains = len(cb_pos_all)

    out: dict = {
        "n_chains": n_chains,
        "n_evidence_per_chain": (
            int(min(len(p) for p in ev_pos_all))
            if ev_pos_all else 0
        ),
        "callback_pos_distribution": dict(Counter(cb_pos_all)),
        "chain_length_distribution": (
            dict(Counter(chain_lengths)) if chain_lengths else None
        ),
        "by_k": {},
    }

    # Bucket of (rank_within_chain_ev, offset_from_callback) for chains
    # that DO leak (any rank). Computed once at max(k).
    per_chain_offsets: list[list[int]] = []
    per_chain_ranks: list[list[int]] = []
    for ci in range(n_chains):
        cb = int(cb_pos_all[ci])
        ev = [int(p) for p in ev_pos_all[ci]]
        offsets = [cb - p for p in ev]  # >=1 always
        per_chain_offsets.append(offsets)
        per_chain_ranks.append(list(range(len(ev))))

    for k in ks:
        leak_any = 0
        leak_first_only = 0
        leak_second_only = 0
        leak_both = 0
        rank_counts = Counter()
        offset_hist = Counter()
        for ci in range(n_chains):
            offsets = per_chain_offsets[ci]
            n_ev = len(offsets)
            in_window = [(0 < off < k) for off in offsets]  # offset>=1, < k means within last k incl callback
            n_in = sum(in_window)
            if n_in > 0:
                leak_any += 1
                for r, hit in enumerate(in_window):
                    if hit:
                        rank_counts[r] += 1
                        offset_hist[offsets[r]] += 1
                if n_ev == 2:
                    if in_window[0] and in_window[1]:
                        leak_both += 1
                    elif in_window[0]:
                        leak_first_only += 1
                    elif in_window[1]:
                        leak_second_only += 1
        out["by_k"][k] = {
            "k": k,
            "leak_any_count": leak_any,
            "leak_any_frac": leak_any / max(1, n_chains),
            "leak_first_only_count": leak_first_only,
            "leak_second_only_count": leak_second_only,
            "leak_both_count": leak_both,
            "rank_hits": dict(rank_counts),
            "offset_histogram": dict(offset_hist),
        }
    return out


@torch.no_grad()
def base_ce_on_callback(
    blob: dict,
    n_chains: int,
    ckpt: str,
    tag: str,
    device: str = "cuda:0",
) -> dict:
    sess = blob["session_ids"]
    starts = blob["chain_starts"].tolist()
    cb_pos_all = blob["chain_callback_position"].tolist()
    cb_mask = blob["session_callback_mask"]

    print(f"-- loading {tag} ({ckpt}) --", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16
    ).to(device).eval()

    nlls: list[float] = []
    item_lens: list[int] = []
    multi_tok = 0
    n_eval = min(n_chains, len(starts))
    for ci in range(n_eval):
        st = starts[ci]
        cb_pos = cb_pos_all[ci]
        cb_session = sess[st + cb_pos].to(device).unsqueeze(0)
        msk = cb_mask[st + cb_pos].to(device)
        n_msk = int(msk.sum().item())
        item_lens.append(n_msk)
        if n_msk > 1:
            multi_tok += 1
        ids = cb_session[:, :-1]
        out_lm = m(ids)
        logits = out_lm.logits
        labels = cb_session[:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            reduction="none",
        ).reshape(1, -1)
        msk1 = msk[1:].float()
        n = msk1.sum().item()
        if n > 0:
            nlls.append((ce * msk1).sum().item() / n)

    del m
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "ckpt": ckpt,
        "n_eval": len(nlls),
        "mean_ce_nats": statistics.mean(nlls),
        "median_ce_nats": statistics.median(nlls),
        "stdev_ce_nats": statistics.pstdev(nlls) if len(nlls) > 1 else 0.0,
        "min_ce_nats": min(nlls),
        "max_ce_nats": max(nlls),
        "multi_token_items": multi_tok,
        "avg_answer_tokens": statistics.mean(item_lens),
    }


def fmt_table_collisions(name: str, stats: dict) -> str:
    rows = []
    rows.append(f"### {name} (n_chains={stats['n_chains']}, n_evidence/chain={stats['n_evidence_per_chain']})")
    rows.append("")
    rows.append("| window_k | any-leak count | any-leak frac | first-only | second-only | both |")
    rows.append("|---:|---:|---:|---:|---:|---:|")
    for k in sorted(stats["by_k"].keys()):
        b = stats["by_k"][k]
        rows.append(
            f"| {k} | {b['leak_any_count']} | {b['leak_any_frac']:.3f} | "
            f"{b['leak_first_only_count']} | {b['leak_second_only_count']} | {b['leak_both_count']} |"
        )
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_chains", type=int, default=128)
    ap.add_argument(
        "--models", nargs="+", default=["0.6B"],
        choices=list(MODEL_REGISTRY.keys()),
    )
    ap.add_argument(
        "--train_blob",
        default=str(CHAINS_DIR / "synthd4v2_persona_callback_train_s512.pt"),
    )
    ap.add_argument(
        "--val_blob",
        default=str(CHAINS_DIR / "synthd4v2_persona_callback_val_s512.pt"),
    )
    ap.add_argument(
        "--ks", nargs="+", type=int, default=[1, 2, 3, 4, 5],
    )
    ap.add_argument(
        "--out_json",
        default=str(RESULTS_DIR / "audit_a1_window_leakage.json"),
    )
    ap.add_argument(
        "--no_ce", action="store_true",
        help="Skip the base-CE step (collision stats only).",
    )
    args = ap.parse_args()

    print(f"== Audit A1: window leakage + base-prior CE ==", flush=True)
    print(f"train_blob: {args.train_blob}", flush=True)
    print(f"val_blob:   {args.val_blob}", flush=True)
    print(f"ks: {args.ks}  n_chains_for_CE: {args.n_chains}  models: {args.models}",
          flush=True)
    print()

    train_blob = load_blob(Path(args.train_blob))
    val_blob = load_blob(Path(args.val_blob))

    train_stats = collision_stats(train_blob, args.ks)
    val_stats = collision_stats(val_blob, args.ks)

    print(fmt_table_collisions("TRAIN", train_stats))
    print()
    print(fmt_table_collisions("VAL", val_stats))
    print()

    # Per-chain offset histograms (val).
    print("=== VAL offset histograms (cb_pos - evidence_pos) per leaking chain at each k ===")
    for k in args.ks:
        b = val_stats["by_k"][k]
        if b["offset_histogram"]:
            hist = sorted(b["offset_histogram"].items())
            print(f"k={k}  rank_hits={b['rank_hits']}  offsets={hist}")
    print()

    ce_results: dict = {}
    if not args.no_ce:
        for tag in args.models:
            ckpt, label = MODEL_REGISTRY[tag]
            res = base_ce_on_callback(val_blob, args.n_chains, ckpt, label)
            ce_results[label] = res
            print(
                f"  {label}: mean_ce={res['mean_ce_nats']:.4f}  "
                f"median={res['median_ce_nats']:.4f}  "
                f"stdev={res['stdev_ce_nats']:.4f}  "
                f"min/max=[{res['min_ce_nats']:.4f}, {res['max_ce_nats']:.4f}]  "
                f"n={res['n_eval']}  multi_tok_items={res['multi_token_items']}  "
                f"avg_ans_toks={res['avg_answer_tokens']:.2f}",
                flush=True,
            )

    print()
    print("=== theoretical floors ===")
    print(f"log(256) = {math.log(256):.4f}  (uniform over all 256 items)")
    print(f"log(32)  = {math.log(32):.4f}  (uniform within cued category)")

    out = {
        "train": train_stats,
        "val": val_stats,
        "base_ce": ce_results,
        "n_chains_for_ce": args.n_chains,
        "ks": args.ks,
        "theoretical": {
            "log_256": math.log(256),
            "log_32": math.log(32),
        },
        "train_blob": args.train_blob,
        "val_blob": args.val_blob,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
