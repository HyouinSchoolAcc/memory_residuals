#!/usr/bin/env python3
"""Tier-1 (3) within-category Δ_sh + Tier-4 (10) per-category Δ_sh breakdown.

Diagnostic sibling of ``tools/eval_callback.py``. Same forward path, same
M_c construction, same callback-aware CE; the only differences are:

1) The chain-shuffle confound is computed **within callback category**
   (using ``chain_meta[ci]['question_type']`` from LongMemEval) instead
   of taking ``ci+1 mod N``. This addresses the friend's Tier-1 (3)
   diagnostic: "if Δ_sh is still ~0 with same-category negatives, the
   writer is encoding 'I am callback-category-K' and nothing finer; if
   it's mildly positive, you've found a hidden signal the random-shuffle
   metric was washing out."

2) Per-category breakdown is reported alongside the corpus mean
   (Tier-4 (10)). Knowledge-update / temporal-reasoning / multi-session
   are categories where chain-specificity should actually matter; the
   single-session-* / preference categories are precisely where a
   chain-class prior is correct. A recipe that lifts Δ_sh on
   knowledge-update but not on preference is doing the right thing.

3) (Bonus) For each chain we also emit the *random*-shuffle Δ_sh from
   the canonical eval (ci+1 mod N), so the within-category and random
   metrics are computed under one model load and one M_c-build pass.

This script is **read-only** — it does not touch the canonical eval
JSONs in ``results/eval_v25_seed_pack_evpos/`` and writes its own
output to ``results/eval_per_category/``.

Usage:
    python tools/eval_callback_categories.py \\
        --model_path runs/chain_v27b_v24a_no_probe_seed3_0p6b_frozen_local/final \\
        --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \\
        --names lme_val \\
        --output results/eval_per_category/v27b_seed3_lme_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(var ** 0.5)


def load_blob(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def chain_session(blob: dict, ci: int, j: int) -> torch.Tensor:
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + j].long()


def chain_session_mask(blob: dict, ci: int, j: int) -> torch.Tensor | None:
    if "session_callback_mask" not in blob:
        return None
    s = int(blob["chain_starts"][ci])
    return blob["session_callback_mask"][s + j].bool()


def get_category(blob: dict, ci: int) -> str | None:
    """Return chain ci's callback category, or None if no metadata."""
    meta = blob.get("chain_meta")
    if meta is None or ci >= len(meta):
        return None
    m = meta[ci]
    if not isinstance(m, dict):
        return None
    return m.get("question_type") or m.get("category") or m.get("task_type")


@torch.no_grad()
def build_Mc(model, blob, ci, end, device):
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    if end <= 0:
        return M_c
    for j in range(end):
        sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
        C = model.model.extract_source(sess[:, :-1])
        M_c = model.model.compress_session(C, M_c)
    return M_c


@torch.no_grad()
def callback_loss(model, input_ids, M_c, callback_mask):
    out = model(input_ids=input_ids, M_c=M_c)
    logits = out.logits
    target = input_ids[:, 1:]
    pred = logits[:, :-1, :]
    mask = callback_mask[1:].to(input_ids.device)
    if mask.sum() == 0:
        return float("nan")
    log_probs = torch.nn.functional.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return float(nll[0][mask].mean().item())


def pick_within_category_negative(
    ci: int, n_chains: int, category_to_chains: dict[str, list[int]],
    chain_to_category: dict[int, str | None],
) -> int:
    """Pick a different chain index from the same callback category.

    Falls back to the random-shuffle ``(ci+1) mod n_chains`` rule if
    chain ci has no category or the category has no other chains.
    """
    cat = chain_to_category.get(ci)
    if cat is None:
        return (ci + 1) % n_chains
    pool = [j for j in category_to_chains.get(cat, []) if j != ci]
    if not pool:
        return (ci + 1) % n_chains
    # Deterministic pick: rotate by 1 within the same-category pool.
    pool.sort()
    pos = pool.index(ci) if ci in pool else 0
    return pool[(pos + 1) % len(pool)]


def evaluate_corpus(
    model,
    blob,
    device,
    n_chains_max: int | None = None,
):
    n_chains = int(blob["chain_starts"].shape[0])
    if n_chains_max is not None:
        n_chains = min(n_chains_max, n_chains)
    if "chain_callback_position" not in blob or "session_callback_mask" not in blob:
        raise RuntimeError("corpus lacks callback annotations")

    chain_to_category: dict[int, str | None] = {
        ci: get_category(blob, ci) for ci in range(n_chains)
    }
    category_to_chains: dict[str, list[int]] = defaultdict(list)
    for ci, cat in chain_to_category.items():
        if cat is not None:
            category_to_chains[cat].append(ci)

    have_categories = sum(1 for c in chain_to_category.values() if c is not None) > 0
    cat_counts = {k: len(v) for k, v in category_to_chains.items()}
    print(f"  category counts: {cat_counts}" if have_categories
          else "  WARN: corpus has no chain_meta; per-category breakdown N/A")

    per_chain = []
    for ci in tqdm(range(n_chains), desc="per-cat eval"):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
            continue
        sess = chain_session(blob, ci, cb_pos).to(device).unsqueeze(0)
        cb_mask = chain_session_mask(blob, ci, cb_pos)
        if cb_mask is None or cb_mask.sum() == 0:
            continue

        Mc_match = build_Mc(model, blob, ci, cb_pos, device)

        # Random-shuffle (canonical eval rule).
        rand_idx = (ci + 1) % n_chains
        Mc_rand = build_Mc(model, blob, rand_idx, cb_pos, device)

        # Within-category shuffle (Tier-1 (3)).
        sc_idx = pick_within_category_negative(
            ci, n_chains, category_to_chains, chain_to_category,
        )
        if sc_idx == rand_idx:
            Mc_sc = Mc_rand  # same neg as random fallback
        else:
            Mc_sc = build_Mc(model, blob, sc_idx, cb_pos, device)

        ce_m = callback_loss(model, sess, Mc_match, cb_mask)
        ce_n = callback_loss(model, sess, None, cb_mask)
        ce_rand = callback_loss(model, sess, Mc_rand, cb_mask)
        ce_sc = callback_loss(model, sess, Mc_sc, cb_mask)

        per_chain.append({
            "chain_id": blob["chain_names"][ci] if "chain_names" in blob else f"c{ci}",
            "category": chain_to_category.get(ci),
            "rand_neg_idx": rand_idx,
            "samecat_neg_idx": sc_idx,
            "samecat_fallback_to_random": (sc_idx == rand_idx),
            "callback_pos": cb_pos,
            "n_callback_tok": int(cb_mask.sum().item()),
            "ce_mem": ce_m,
            "ce_nomem": ce_n,
            "ce_shuffle_random": ce_rand,
            "ce_shuffle_samecat": ce_sc,
            "pa_cb_dnm": ce_n - ce_m,
            "pa_cb_dsh_random": ce_rand - ce_m,
            "pa_cb_dsh_samecat": ce_sc - ce_m,
        })

    # Aggregate.
    overall = {
        "n_chains_scored": len(per_chain),
        "ce_mem": mean([r["ce_mem"] for r in per_chain]),
        "ce_nomem": mean([r["ce_nomem"] for r in per_chain]),
        "ce_shuffle_random": mean([r["ce_shuffle_random"] for r in per_chain]),
        "ce_shuffle_samecat": mean([r["ce_shuffle_samecat"] for r in per_chain]),
        "pa_cb_dnm": mean([r["pa_cb_dnm"] for r in per_chain]),
        "pa_cb_dsh_random": mean([r["pa_cb_dsh_random"] for r in per_chain]),
        "pa_cb_dsh_samecat": mean([r["pa_cb_dsh_samecat"] for r in per_chain]),
        "pa_cb_dsh_random_std": stdev([r["pa_cb_dsh_random"] for r in per_chain]),
        "pa_cb_dsh_samecat_std": stdev([r["pa_cb_dsh_samecat"] for r in per_chain]),
    }

    by_cat: dict[str, dict] = {}
    for cat in sorted(set(r["category"] for r in per_chain if r["category"])):
        rows = [r for r in per_chain if r["category"] == cat]
        if not rows:
            continue
        n_fallback = sum(1 for r in rows if r["samecat_fallback_to_random"])
        by_cat[cat] = {
            "n_chains": len(rows),
            "n_samecat_fallback_to_random": n_fallback,
            "pa_cb_dnm": mean([r["pa_cb_dnm"] for r in rows]),
            "pa_cb_dsh_random": mean([r["pa_cb_dsh_random"] for r in rows]),
            "pa_cb_dsh_samecat": mean([r["pa_cb_dsh_samecat"] for r in rows]),
            "pa_cb_dsh_random_std": stdev([r["pa_cb_dsh_random"] for r in rows]),
            "pa_cb_dsh_samecat_std": stdev([r["pa_cb_dsh_samecat"] for r in rows]),
        }

    return {**overall, "per_category": by_cat, "per_chain": per_chain}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--corpora", nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)
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

    names = a.names if a.names is not None else [Path(c).stem for c in a.corpora]
    out: dict = {}
    for path, name in zip(a.corpora, names):
        print(f"\n--- per-cat eval on {name} ({path}) ---")
        blob = load_blob(Path(path))
        if "chain_callback_position" not in blob:
            print(f"  SKIP: {name} lacks chain_callback_position")
            out[name] = {"skipped": True, "reason": "no callback positions"}
            continue
        m = evaluate_corpus(model, blob, device, n_chains_max=a.n_chains_max)
        out[name] = m
        print(
            f"  n={m['n_chains_scored']} "
            f"pa_cb_dnm={m['pa_cb_dnm']:+.4f} "
            f"pa_cb_dsh_random={m['pa_cb_dsh_random']:+.4f} "
            f"pa_cb_dsh_samecat={m['pa_cb_dsh_samecat']:+.4f}"
        )
        for cat, c in m["per_category"].items():
            print(
                f"    [{cat:<28s}] n={c['n_chains']:>2d} "
                f"dnm={c['pa_cb_dnm']:+.4f} "
                f"dsh_rand={c['pa_cb_dsh_random']:+.4f} "
                f"dsh_samecat={c['pa_cb_dsh_samecat']:+.4f}"
            )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {a.output}")


if __name__ == "__main__":
    main()
