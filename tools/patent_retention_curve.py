#!/usr/bin/env python3
"""Patent Experiment 3 (MemRes side) — retention / adoption over updates.

For every chain in the corpus, the memory M_c is updated session by
session (the two-stage write of the invention), and after EVERY update t
we score the callback-session CE. This produces a per-chain curve

    ce(t) = callback CE after sessions 1..t have been written,
            t = 0 .. cb_pos   (t=0 -> zero memory)

from which we report:

  adoption  : the CE drop achieved immediately once the last
              evidence-bearing session has been written
              (ce(ev_last+1) vs ce(ev_last)) and the total drop vs t=0.
  retention : how much of the gain achieved at t = ev_last+1 is still
              present at t = cb_pos, i.e. after (cb_pos - ev_last - 1)
              further competitive updates have overwritten slots
              ("old-information keep-rate under continued writes").

Output JSON has per-chain curves plus aggregates, ready for plotting.

Usage:
    python tools/patent_retention_curve.py \
        --model_path runs/chain_v27b_.../final \
        --output results/patent_experiments/retention_0p6b_seed3.json
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
    p.add_argument("--corpus",
                   default="paper_artifacts/chains/lme_val_s512_evpos.pt")
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
    K, d = model.config.memres_num_vectors, model.config.hidden_size

    blob = torch.load(a.corpus, map_location="cpu", weights_only=False)
    n_chains = int(blob["chain_starts"].shape[0])
    if a.n_chains_max is not None:
        n_chains = min(n_chains, a.n_chains_max)

    per_chain = []
    with torch.no_grad():
        for ci in tqdm(range(n_chains), desc="retention curves"):
            cb_pos = int(blob["chain_callback_position"][ci])
            if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
                continue
            cb_sess = chain_session(blob, ci, cb_pos).to(device).unsqueeze(0)
            cb_mask = chain_session_mask(blob, ci, cb_pos)
            if cb_mask is None or cb_mask.sum() == 0:
                continue
            try:
                ev = sorted(int(x) for x in
                            blob["chain_evidence_positions"][ci] if int(x) < cb_pos)
            except (KeyError, TypeError):
                ev = []

            M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
            curve = [callback_loss(model, cb_sess, M_c, cb_mask)]  # t=0
            for t in range(cb_pos):
                sess = chain_session(blob, ci, t).to(device).unsqueeze(0)
                C = model.model.extract_source(sess[:, :-1])
                M_c = model.model.compress_session(C, M_c)
                curve.append(callback_loss(model, cb_sess, M_c, cb_mask))

            ev_last = ev[-1] if ev else None
            row = {
                "chain_id": (blob["chain_names"][ci]
                             if "chain_names" in blob else f"c{ci}"),
                "cb_pos": cb_pos,
                "evidence_positions": ev,
                "curve": curve,          # length cb_pos+1, index = t
            }
            if ev_last is not None:
                ce_before_ev = curve[ev_last]           # memory w/o evidence
                ce_after_ev = curve[ev_last + 1]        # right after write
                ce_final = curve[cb_pos]                # after all writes
                gain_at_write = ce_before_ev - ce_after_ev
                gain_final = ce_before_ev - ce_final
                row.update({
                    "n_updates_after_evidence": cb_pos - ev_last - 1,
                    "ce_before_ev": ce_before_ev,
                    "ce_after_ev": ce_after_ev,
                    "ce_final": ce_final,
                    "gain_at_write": gain_at_write,
                    "gain_final": gain_final,
                    "keep_rate": (gain_final / gain_at_write
                                  if abs(gain_at_write) > 1e-6 else None),
                })
            per_chain.append(row)

    # aggregates
    total_drop = [r["curve"][0] - r["curve"][-1] for r in per_chain]
    keep = [r["keep_rate"] for r in per_chain
            if r.get("keep_rate") is not None and r.get("gain_at_write", 0) > 0.05]
    summary = {
        "model_path": a.model_path,
        "n_chains": len(per_chain),
        "mean_ce_t0": mean([r["curve"][0] for r in per_chain]),
        "mean_ce_final": mean([r["curve"][-1] for r in per_chain]),
        "mean_total_drop": mean(total_drop),
        "mean_gain_at_write": mean([r.get("gain_at_write", float("nan"))
                                    for r in per_chain]),
        "mean_gain_final": mean([r.get("gain_final", float("nan"))
                                 for r in per_chain]),
        "mean_keep_rate_pos_gain": mean(keep),
        "n_keep_rate_chains": len(keep),
        "per_chain": per_chain,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chain"},
                     indent=2))
    print(f"Saved -> {a.output}")


if __name__ == "__main__":
    main()
