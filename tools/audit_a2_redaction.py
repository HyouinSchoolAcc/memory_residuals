#!/usr/bin/env python3
"""Audit A2: probe whether the standard ``evidence_redact`` floor leaks
evidence information through the rolling memory state.

Runs four floor variants in one pass per chain so we don't pay the
forward cost twice:

  - mem            : full-prefix M_c with original sessions (control).
  - nomem          : M_c = None (control).
  - shuffle        : M_c built from a different chain's full prefix.
  - floor_default  : current eval_callback.py behaviour -- replace
                     evidence-position sessions with cb-1 sessions
                     drawn from the first ``redact_pool_size*4`` chains
                     of the corpus.
  - floor_cross    : replace evidence-position sessions with sessions
                     drawn from NON-evidence positions of a different
                     chain (sampled per-evidence with a deterministic
                     RNG so re-runs are reproducible).
  - floor_skip     : skip evidence-position sessions entirely (do NOT
                     compress them into M_c). The rolling state still
                     has the non-evidence prefix sessions, but never
                     sees anything in the evidence slot.
  - floor_zero     : replace evidence-position sessions with an
                     all-EOS token id sequence (truly content-free
                     input -- the writer GRU still gets a forward, but
                     the source tokens carry no information).

The decisive comparison is ``floor_default`` vs ``floor_cross`` /
``floor_zero``: if they're within ~0.05 nats the standard redaction is
sound; a gap > 0.2 nats means the stock floor is leaking evidence
content (or template content correlated with evidence).

This script DOES NOT modify ``tools/eval_callback.py``. It re-uses the
same ``build_Mc`` semantics by parameterising the per-position session
selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def chain_session(blob, ci, j):
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + j].long()


def chain_session_mask(blob, ci, j):
    s = int(blob["chain_starts"][ci])
    return blob["session_callback_mask"][s + j].bool()


@torch.no_grad()
def build_Mc_variant(
    model, blob, ci, end, device,
    *, mode: str = "match",
    distractor_pool=None, distractor_offset: int = 0,
    cross_chain_picks=None,
    pad_token_id: int = 0,
):
    """Build M_c with one of several evidence-redaction strategies.

    mode:
      - "match"        : original sessions (control).
      - "default_pool" : evidence sessions replaced by ``distractor_pool``
                         entries starting at ``distractor_offset`` (this
                         is exactly what ``eval_callback.py`` does).
      - "cross_chain"  : evidence sessions replaced by entries from
                         ``cross_chain_picks`` (a list of S-token tensors
                         drawn from non-evidence positions of OTHER
                         chains).
      - "skip"         : do NOT compress evidence sessions at all.
      - "zero"         : compress an all-pad-token session in their
                         place.
    """
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    if end <= 0:
        return M_c

    ev_positions: set = set()
    if mode != "match":
        try:
            ev_positions = {
                int(p) for p in blob["chain_evidence_positions"][ci]
                if int(p) < end
            }
        except (TypeError, KeyError):
            ev_positions = set()

    pool_iter = None
    if mode == "default_pool" and distractor_pool is not None:
        pool_iter = iter(
            distractor_pool[distractor_offset:]
            + distractor_pool[:distractor_offset]
        )
    cross_iter = (
        iter(cross_chain_picks) if cross_chain_picks is not None else None
    )

    sample_zero = None
    for j in range(end):
        if j in ev_positions:
            if mode == "skip":
                continue
            if mode == "zero":
                if sample_zero is None:
                    S_tok = blob["session_ids"][0].shape[-1]
                    sample_zero = torch.full(
                        (1, S_tok), pad_token_id,
                        device=device, dtype=torch.long,
                    )
                sess = sample_zero
            elif mode == "default_pool" and pool_iter is not None:
                try:
                    sess = next(pool_iter).to(device).unsqueeze(0).long()
                except StopIteration:
                    sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
            elif mode == "cross_chain" and cross_iter is not None:
                try:
                    sess = next(cross_iter).to(device).unsqueeze(0).long()
                except StopIteration:
                    sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
            else:
                sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
        else:
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


def eval_one(model, blob, device, n_chains_max, redact_pool_size=8, seed=0):
    n_chains = int(blob["chain_starts"].shape[0])
    if n_chains_max is not None:
        n_chains = min(n_chains_max, n_chains)

    # Same default-pool construction as eval_callback.py.
    distractor_pool = []
    pool_chain_count = min(redact_pool_size * 4, n_chains)
    for ci in range(pool_chain_count):
        cb_pos = int(blob["chain_callback_position"][ci])
        last_filler_pos = max(0, cb_pos - 1)
        distractor_pool.append(chain_session(blob, ci, last_filler_pos))

    rng = random.Random(seed)
    cb_mem, cb_no, cb_shuf = [], [], []
    cb_floor_default, cb_floor_cross, cb_floor_skip, cb_floor_zero = [], [], [], []
    per_chain = []

    for ci in tqdm(range(n_chains), desc="audit_a2 eval"):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
            continue
        sess = chain_session(blob, ci, cb_pos).to(device).unsqueeze(0)
        cb_mask = chain_session_mask(blob, ci, cb_pos)
        if cb_mask is None or cb_mask.sum() == 0:
            continue
        ev_positions = list(blob["chain_evidence_positions"][ci])
        n_ev = len(ev_positions)

        # Cross-chain pickset: for each evidence slot, pick a session
        # from a DIFFERENT chain at a NON-evidence (and non-callback)
        # position. Deterministic per (seed, ci).
        cross_picks = []
        for k in range(max(1, n_ev)):
            for _ in range(20):
                other = rng.randrange(n_chains)
                if other == ci:
                    continue
                o_cb = int(blob["chain_callback_position"][other])
                o_len = int(blob["chain_lengths"][other])
                if o_cb < 1 or o_len < 2:
                    continue
                o_ev = set(int(p) for p in blob["chain_evidence_positions"][other])
                candidates = [
                    p for p in range(o_len)
                    if p != o_cb and p not in o_ev
                ]
                if not candidates:
                    continue
                pick_pos = rng.choice(candidates)
                cross_picks.append(chain_session(blob, other, pick_pos))
                break

        offset = (ci * 7) % max(1, len(distractor_pool) - max(1, n_ev))

        Mc_match = build_Mc_variant(model, blob, ci, cb_pos, device, mode="match")
        Mc_shuf = build_Mc_variant(
            model, blob, (ci + 1) % n_chains, cb_pos, device, mode="match"
        )
        Mc_def = build_Mc_variant(
            model, blob, ci, cb_pos, device,
            mode="default_pool",
            distractor_pool=distractor_pool, distractor_offset=offset,
        )
        Mc_cross = build_Mc_variant(
            model, blob, ci, cb_pos, device,
            mode="cross_chain", cross_chain_picks=cross_picks,
        )
        Mc_skip = build_Mc_variant(
            model, blob, ci, cb_pos, device, mode="skip",
        )
        Mc_zero = build_Mc_variant(
            model, blob, ci, cb_pos, device,
            mode="zero", pad_token_id=0,
        )

        ce_m = callback_loss(model, sess, Mc_match, cb_mask)
        ce_n = callback_loss(model, sess, None, cb_mask)
        ce_s = callback_loss(model, sess, Mc_shuf, cb_mask)
        ce_d = callback_loss(model, sess, Mc_def, cb_mask)
        ce_c = callback_loss(model, sess, Mc_cross, cb_mask)
        ce_k = callback_loss(model, sess, Mc_skip, cb_mask)
        ce_z = callback_loss(model, sess, Mc_zero, cb_mask)

        cb_mem.append(ce_m); cb_no.append(ce_n); cb_shuf.append(ce_s)
        cb_floor_default.append(ce_d)
        cb_floor_cross.append(ce_c)
        cb_floor_skip.append(ce_k)
        cb_floor_zero.append(ce_z)
        per_chain.append(dict(
            chain_id=blob["chain_names"][ci] if "chain_names" in blob else f"c{ci}",
            cb_pos=cb_pos, n_ev=n_ev,
            ce_mem=ce_m, ce_nomem=ce_n, ce_shuffle=ce_s,
            ce_floor_default=ce_d, ce_floor_cross=ce_c,
            ce_floor_skip=ce_k, ce_floor_zero=ce_z,
        ))

    out = {
        "n_scored": len(cb_mem),
        "ce_mem": mean(cb_mem),
        "ce_nomem": mean(cb_no),
        "ce_shuffle": mean(cb_shuf),
        "ce_floor_default": mean(cb_floor_default),
        "ce_floor_cross": mean(cb_floor_cross),
        "ce_floor_skip": mean(cb_floor_skip),
        "ce_floor_zero": mean(cb_floor_zero),
    }
    out["pa_cb_dnm"] = out["ce_nomem"] - out["ce_mem"]
    out["pa_cb_dsh"] = out["ce_shuffle"] - out["ce_mem"]
    out["dnm_floor_default"] = out["ce_nomem"] - out["ce_floor_default"]
    out["dnm_floor_cross"] = out["ce_nomem"] - out["ce_floor_cross"]
    out["dnm_floor_skip"] = out["ce_nomem"] - out["ce_floor_skip"]
    out["dnm_floor_zero"] = out["ce_nomem"] - out["ce_floor_zero"]
    out["evidence_lift_default"] = out["pa_cb_dnm"] - out["dnm_floor_default"]
    out["evidence_lift_cross"] = out["pa_cb_dnm"] - out["dnm_floor_cross"]
    out["evidence_lift_skip"] = out["pa_cb_dnm"] - out["dnm_floor_skip"]
    out["evidence_lift_zero"] = out["pa_cb_dnm"] - out["dnm_floor_zero"]
    out["per_chain"] = per_chain
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n_chains_max", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    device = torch.device(a.device)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
        .to(device).eval()
    )
    blob = torch.load(a.corpus, map_location="cpu", weights_only=False)
    res = eval_one(model, blob, device, a.n_chains_max, seed=a.seed)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_chain"}, indent=2))
    print(f"\n-> {a.output}")


if __name__ == "__main__":
    main()
