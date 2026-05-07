#!/usr/bin/env python3
"""Counterfactual sensitivity: does perturbing a single prior session move
the loss on the current session?

Protocol:

For every chain c with length L >= depth+2:

    score_session  = session L-1 (the held-out scoring session)
    normal_prefix  = sessions 0..L-2          (true history of c)
    perturbed_pref = sessions 0..L-2 with
                     position (L-1-depth) replaced by a session randomly
                     sampled from a *different* chain's history.

We rebuild M_c from each prefix, score the same `score_session`, and report

    delta(depth) = NLL_perturbed - NLL_normal   (positive = the model
                                                  used the perturbed slot)

Aggregating across chains gives the recall-decay curve: how far back can the
model "see" through the memory state?

We also record:

    ce_normal      mean NLL with the true memory
    ce_perturbed   mean NLL with the perturbed memory
    ce_nomem       mean NLL with no memory (bound: how much memory helps total)

For comparison the script also runs the trivial "perturb = no memory at all"
case as `depth=ALL`.

Output JSON shape::

    {
      "model_path": str,
      "corpus": str,
      "depths": [1, 2, 4, 8],
      "results": {
        "depth_1": {
          "n":             int,
          "ce_normal":     float,
          "ce_perturbed":  float,
          "delta_mean":    float,
          "delta_std":     float
        },
        ...,
        "depth_ALL": {...}        # memory wiped entirely
      },
      "n_chains_eligible": int,
      "n_chains_used":     int,
      "per_chain": [...]
    }
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def _load_blob(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _session_at(blob: dict, ci: int, pos: int) -> torch.Tensor:
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + pos].long()


@torch.no_grad()
def _build_M_from_sessions(
    model: Qwen3MemResForCausalLM,
    sessions: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    for sess in sessions:
        sess = sess.to(device).unsqueeze(0)
        C = model.model.extract_source(sess[:, :-1])
        M_c = model.model.compress_session(C, M_c)
    return M_c


@torch.no_grad()
def _score(
    model: Qwen3MemResForCausalLM,
    input_ids: torch.Tensor,
    M_c: torch.Tensor | None,
) -> float:
    out = model(input_ids=input_ids, labels=input_ids, M_c=M_c)
    return float(out.loss.item())


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max_chains", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)

    device = torch.device(a.device)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    print(f"model loaded; memres_mode={getattr(model.config, 'memres_mode', '?')}",
          flush=True)

    blob = _load_blob(Path(a.corpus))
    n_chains = int(blob["chain_starts"].shape[0])
    chain_lengths = blob["chain_lengths"]
    chain_names = blob["chain_names"]

    # Eligibility: chain must have length >= max(depths)+2 so we can perturb a
    # slot at depth max(depths) AND still have a scoring session.
    max_depth = max(a.depths)
    eligible = [
        ci for ci in range(n_chains)
        if int(chain_lengths[ci]) >= max_depth + 2
    ]
    rng.shuffle(eligible)
    used = eligible[: a.max_chains]

    # For perturbation we need a pool of donor sessions from OTHER chains.
    donor_pool: list[tuple[int, int]] = []
    for ci in range(n_chains):
        L = int(chain_lengths[ci])
        for j in range(L):
            donor_pool.append((ci, j))

    def _donor_session(exclude_chain: int) -> torch.Tensor:
        for _ in range(20):
            ci, j = rng.choice(donor_pool)
            if ci != exclude_chain:
                return _session_at(blob, ci, j)
        # Fallback: pick any session even from the same chain.
        ci, j = rng.choice(donor_pool)
        return _session_at(blob, ci, j)

    results: dict[str, dict] = {}
    deltas_by_depth: dict[int, list[float]] = {d: [] for d in a.depths}
    ce_normal_by_depth: dict[int, list[float]] = {d: [] for d in a.depths}
    ce_perturb_by_depth: dict[int, list[float]] = {d: [] for d in a.depths}
    ce_normal_all = []
    ce_nomem_all = []
    per_chain: list[dict] = []

    for ci in tqdm(used, desc="counterfactual chains"):
        L = int(chain_lengths[ci])
        # Score session is session L-1; prefix is 0..L-2.
        score_sess = _session_at(blob, ci, L - 1).to(device).unsqueeze(0)
        score_input = score_sess[:, :-1]

        prefix = [_session_at(blob, ci, j) for j in range(L - 1)]
        try:
            M_normal = _build_M_from_sessions(model, prefix, device)
        except Exception as e:
            print(f"chain {ci} normal M build failed: {e}", flush=True)
            continue
        ce_normal = _score(model, score_input, M_normal)
        ce_nomem = _score(model, score_input, None)
        ce_normal_all.append(ce_normal)
        ce_nomem_all.append(ce_nomem)

        chain_record = {
            "chain_id": str(chain_names[ci]),
            "length": L,
            "ce_normal": ce_normal,
            "ce_nomem": ce_nomem,
            "by_depth": {},
        }

        for depth in a.depths:
            slot = (L - 1) - depth          # position in prefix to perturb
            if slot < 0 or slot >= L - 1:
                continue
            perturbed = list(prefix)
            perturbed[slot] = _donor_session(exclude_chain=ci)
            try:
                M_perturbed = _build_M_from_sessions(model, perturbed, device)
            except Exception as e:
                print(f"chain {ci} depth {depth} perturb failed: {e}", flush=True)
                continue
            ce_p = _score(model, score_input, M_perturbed)
            delta = ce_p - ce_normal
            deltas_by_depth[depth].append(delta)
            ce_normal_by_depth[depth].append(ce_normal)
            ce_perturb_by_depth[depth].append(ce_p)
            chain_record["by_depth"][str(depth)] = {
                "ce_perturbed": ce_p,
                "delta": delta,
            }

        per_chain.append(chain_record)

    for depth in a.depths:
        m, s = _mean_std(deltas_by_depth[depth])
        m_n, _ = _mean_std(ce_normal_by_depth[depth])
        m_p, _ = _mean_std(ce_perturb_by_depth[depth])
        results[f"depth_{depth}"] = {
            "n": len(deltas_by_depth[depth]),
            "ce_normal": m_n,
            "ce_perturbed": m_p,
            "delta_mean": m,
            "delta_std": s,
        }

    # depth_ALL: memory is wiped (delta = ce_nomem - ce_normal)
    deltas_all = [n - m for n, m in zip(ce_nomem_all, ce_normal_all)]
    m_all, s_all = _mean_std(deltas_all)
    m_n_all, _ = _mean_std(ce_normal_all)
    m_p_all, _ = _mean_std(ce_nomem_all)
    results["depth_ALL"] = {
        "n": len(deltas_all),
        "ce_normal": m_n_all,
        "ce_perturbed": m_p_all,
        "delta_mean": m_all,
        "delta_std": s_all,
    }

    out = {
        "model_path": str(a.model_path),
        "corpus": str(a.corpus),
        "depths": a.depths,
        "results": results,
        "n_chains_eligible": len(eligible),
        "n_chains_used": len(used),
        "per_chain": per_chain,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {a.output}", flush=True)
    print(json.dumps({k: results[k] for k in results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
