#!/usr/bin/env python3
"""Callback-aware standalone evaluator.

Mirrors the in-trainer ``pa_cb_*`` family of metrics:
    pa_cb_dnm        = CE_nomem_callback - CE_mem_callback
    pa_cb_dsh        = CE_shuffle_callback - CE_mem_callback
    evidence_lift    = pa_cb_dnm - pa_cb_dnm_floor

where the loss is computed only over the *callback* tokens of the
*callback* session (using ``session_callback_mask`` and
``chain_callback_position`` from the chain blob), and the prefix Mc
is built from the full chain prefix the same way as in
``tools/eval_chain.py``.

The "evidence floor" is computed by running the model on a chain
where the *evidence* sessions have been replaced with random
distractor sessions from the same corpus, with their callback masks
zeroed. This gives the no-evidence baseline that the trainer reports
as ``pa_cb_dnm_floor`` -- the gap between dnm (memory CE - nomem CE)
and dnm_floor (same with evidence redacted) is the lift attributable
specifically to evidence retrieval.

Usage:
    python tools/eval_callback.py \
        --model_path output/<run>/best \
        --corpora paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
        --output results/eval_v14v15/<run>_callback_aware.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


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


@torch.no_grad()
def build_Mc(model, blob, ci, end, device, *, evidence_redact: bool = False,
             redact_with: torch.Tensor | None = None):
    """Build M_c from chain[ci][:end].

    If ``evidence_redact`` is True, replace evidence sessions
    (positions in ``blob['chain_evidence_positions'][ci]``) with
    ``redact_with`` rows (a (n_evidence, S) tensor of distractor
    session ids).
    """
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    if end <= 0:
        return M_c

    evidence_positions: set[int] = set()
    if evidence_redact and "chain_evidence_positions" in blob:
        try:
            evidence_positions = {
                int(p) for p in blob["chain_evidence_positions"][ci]
                if int(p) < end
            }
        except (TypeError, KeyError):
            evidence_positions = set()

    redact_iter = iter(redact_with) if redact_with is not None else None
    for j in range(end):
        if j in evidence_positions and redact_iter is not None:
            try:
                sess = next(redact_iter).to(device).unsqueeze(0).long()
            except StopIteration:
                sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
        else:
            sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
        C = model.model.extract_source(sess[:, :-1])
        M_c = model.model.compress_session(C, M_c)
    return M_c


@torch.no_grad()
def callback_loss(model, input_ids, M_c, callback_mask):
    """CE averaged only over callback positions."""
    out = model(input_ids=input_ids, M_c=M_c)
    logits = out.logits  # (1, S, V)
    target = input_ids[:, 1:]
    pred = logits[:, :-1, :]
    mask = callback_mask[1:].to(input_ids.device)
    if mask.sum() == 0:
        return float("nan")
    log_probs = torch.nn.functional.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return float(nll[0][mask].mean().item())


def evaluate_corpus(model, blob, device, n_chains_max: int | None = None,
                    redact_pool_size: int = 8):
    n_chains = int(blob["chain_starts"].shape[0])
    if n_chains_max is not None:
        n_chains = min(n_chains_max, n_chains)
    has_callback_pos = "chain_callback_position" in blob
    has_callback_mask = "session_callback_mask" in blob

    if not has_callback_pos or not has_callback_mask:
        raise RuntimeError(
            "Corpus lacks chain_callback_position / session_callback_mask "
            "fields; cannot run callback-aware eval. Use eval_chain.py."
        )

    ce_mem, ce_nomem, ce_shuffle, ce_mem_floor, ce_nomem_floor = [], [], [], [], []
    per_chain = []

    # Pool of distractor sessions for evidence redaction: take the
    # last session of every chain (callback session is rich in
    # template structure but devoid of evidence text).
    distractor_pool: list[torch.Tensor] = []
    for ci in range(min(redact_pool_size * 4, n_chains)):
        cb_pos = int(blob["chain_callback_position"][ci])
        last_filler_pos = max(0, cb_pos - 1)
        distractor_pool.append(chain_session(blob, ci, last_filler_pos))

    for ci in tqdm(range(n_chains), desc="callback eval"):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
            continue
        sess = chain_session(blob, ci, cb_pos).to(device).unsqueeze(0)
        cb_mask = chain_session_mask(blob, ci, cb_pos)
        if cb_mask is None or cb_mask.sum() == 0:
            continue

        Mc_match = build_Mc(model, blob, ci, cb_pos, device)

        shuffle_idx = (ci + 1) % n_chains
        Mc_shuf = build_Mc(model, blob, shuffle_idx, cb_pos, device)

        ev_positions_blob = blob.get("chain_evidence_positions")
        if ev_positions_blob is not None and ci < len(ev_positions_blob):
            n_evidence = len(ev_positions_blob[ci])
        else:
            # Corpora without explicit evidence-position annotation (LME,
            # MSC, real-content): treat as 1 evidence session for the
            # redact-based floor baseline; the redaction picks one prefix
            # session at random.
            n_evidence = 1
        offset = (ci * 7) % max(1, len(distractor_pool) - n_evidence)
        redact_rows = torch.stack(
            distractor_pool[offset : offset + max(1, n_evidence)]
        ) if distractor_pool else None
        Mc_floor = build_Mc(
            model, blob, ci, cb_pos, device,
            evidence_redact=True, redact_with=redact_rows,
        )

        ce_m = callback_loss(model, sess, Mc_match, cb_mask)
        ce_n = callback_loss(model, sess, None,    cb_mask)
        ce_s = callback_loss(model, sess, Mc_shuf, cb_mask)
        ce_mf = callback_loss(model, sess, Mc_floor, cb_mask)
        ce_nf = ce_n

        ce_mem.append(ce_m)
        ce_nomem.append(ce_n)
        ce_shuffle.append(ce_s)
        ce_mem_floor.append(ce_mf)
        ce_nomem_floor.append(ce_nf)

        per_chain.append({
            "chain_id": blob["chain_names"][ci] if "chain_names" in blob else f"c{ci}",
            "callback_pos": cb_pos,
            "n_callback_tok": int(cb_mask.sum().item()),
            "ce_mem": ce_m, "ce_nomem": ce_n, "ce_shuffle": ce_s,
            "ce_mem_floor": ce_mf,
        })

    return {
        "n_chains_scored": len(per_chain),
        "ce_mem": mean(ce_mem),
        "ce_nomem": mean(ce_nomem),
        "ce_shuffle": mean(ce_shuffle),
        "ce_mem_floor": mean(ce_mem_floor),
        "pa_cb_dnm": mean(ce_nomem) - mean(ce_mem),
        "pa_cb_dsh": mean(ce_shuffle) - mean(ce_mem),
        "pa_cb_dnm_floor": mean(ce_nomem) - mean(ce_mem_floor),
        "pa_cb_evidence_lift": (
            (mean(ce_nomem) - mean(ce_mem))
            - (mean(ce_nomem) - mean(ce_mem_floor))
        ),
        "per_chain": per_chain,
    }


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

    if a.names is None:
        names = [Path(c).stem for c in a.corpora]
    else:
        names = a.names

    out = {}
    for path, name in zip(a.corpora, names):
        print(f"\n--- callback-aware eval on {name} ({path}) ---")
        blob = load_blob(Path(path))
        if "chain_callback_position" not in blob:
            print(f"  SKIP: {name} has no chain_callback_position field")
            out[name] = {"skipped": True, "reason": "no callback positions"}
            continue
        m = evaluate_corpus(model, blob, device, n_chains_max=a.n_chains_max)
        out[name] = m
        print(
            f"  n={m['n_chains_scored']} "
            f"ce_mem={m['ce_mem']:.4f} ce_nomem={m['ce_nomem']:.4f} "
            f"ce_shuf={m['ce_shuffle']:.4f} ce_floor={m['ce_mem_floor']:.4f}"
        )
        print(
            f"  pa_cb_dnm={m['pa_cb_dnm']:+.4f} "
            f"pa_cb_dsh={m['pa_cb_dsh']:+.4f} "
            f"evidence_lift={m['pa_cb_evidence_lift']:+.4f}"
        )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {a.output}")


if __name__ == "__main__":
    main()
