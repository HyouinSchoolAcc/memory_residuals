#!/usr/bin/env python3
"""Routing-trace analysis: where does the memory mass go?

For every chain in a corpus, build M_c by walking the prefix session-by-session
(same recurrent unroll as eval_chain.py), then score each of the last
``--score_window`` sessions with ``collect_alpha_trace=True``.  This collects
alpha_mem -- the per-token softmax mass on the memory source b_{-1} -- at
every routing sublayer.

We aggregate alpha_mem across (sublayer, position-decile, chain-depth) buckets
and dump a JSON suitable for plotting a heatmap in the paper.

Two routing conditions are recorded:

    - ``mem``:     alpha_mem under the chain's own M_c.
    - ``shuffle``: alpha_mem under another chain's M_c.

If the model truly USES its memory (not just routes mass to it as a no-op),
mem and shuffle alpha values should differ either in magnitude or in their
positional distribution.  In particular, mem alpha should rise at "callback"
positions where the model would benefit from prior context.

Output JSON shape::

    {
      "model_path": str,
      "memres_mode": "attention_parity" | "attention_base" | "simple_gate",
      "n_sublayers": int,
      "n_chains": int,
      "n_score_positions": int,
      "alpha_mem_by_sublayer": {
        "mem":     [n_sublayers]  # mean over all (chain, position, token)
        "shuffle": [n_sublayers]
      },
      "alpha_mem_by_sublayer_position_decile": {
        "mem":     [n_sublayers][10]
        "shuffle": [n_sublayers][10]
      },
      "alpha_mem_by_chain_depth": {
        "mem":     [score_window]  # last K sessions, oldest..newest
        "shuffle": [score_window]
      },
      "raw_layer_means": {                # convenience: per-(chain,depth) means
        ...
      }
    }
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


def _load_blob(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _session_at(blob: dict, ci: int, pos: int) -> torch.Tensor:
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + pos].long()


@torch.no_grad()
def _build_M_at_each_step(
    model: Qwen3MemResForCausalLM,
    blob: dict,
    ci: int,
    length: int,
    device: torch.device,
) -> list[torch.Tensor]:
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    out = [M_c.clone()]
    for j in range(length):
        sess = _session_at(blob, ci, j).to(device).unsqueeze(0)
        C = model.model.extract_source(sess[:, :-1])
        M_c = model.model.compress_session(C, M_c)
        out.append(M_c.clone())
    return out


@torch.no_grad()
def _alpha_trace_for(
    model: Qwen3MemResForCausalLM,
    input_ids: torch.Tensor,
    M_c: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Return list of (B, S) alpha_mem tensors, one per routing sublayer."""
    out = model(input_ids=input_ids, M_c=M_c, collect_alpha_trace=True)
    trace = getattr(out, "alpha_trace", None)
    if not trace:
        return []
    # Each entry is (B, S); detach and move to cpu fp32 for aggregation.
    return [t.detach().to(torch.float32).cpu() for t in trace]


def _decile_buckets(seq_len: int) -> list[tuple[int, int]]:
    """Return 10 (lo, hi) index bands across [0, seq_len)."""
    buckets = []
    for d in range(10):
        lo = (d * seq_len) // 10
        hi = ((d + 1) * seq_len) // 10
        if hi <= lo:
            hi = min(seq_len, lo + 1)
        buckets.append((lo, hi))
    return buckets


def _accumulate(
    accum_sum: list[torch.Tensor],
    accum_cnt: list[torch.Tensor],
    trace: list[torch.Tensor],
) -> None:
    """Sum and count alpha values per sublayer, broadcasting across (B,S)."""
    while len(accum_sum) < len(trace):
        accum_sum.append(torch.zeros(1, dtype=torch.float64))
        accum_cnt.append(torch.zeros(1, dtype=torch.float64))
    for li, t in enumerate(trace):
        accum_sum[li] += t.double().sum()
        accum_cnt[li] += float(t.numel())


def _accumulate_position_decile(
    accum_sum: list[list[float]],
    accum_cnt: list[list[float]],
    trace: list[torch.Tensor],
) -> None:
    """Per (sublayer, position-decile) sum and count."""
    if not trace:
        return
    seq_len = trace[0].shape[1]
    buckets = _decile_buckets(seq_len)
    while len(accum_sum) < len(trace):
        accum_sum.append([0.0] * 10)
        accum_cnt.append([0.0] * 10)
    for li, t in enumerate(trace):
        flat = t.double()  # (B, S)
        for di, (lo, hi) in enumerate(buckets):
            seg = flat[:, lo:hi]
            accum_sum[li][di] += float(seg.sum())
            accum_cnt[li][di] += float(seg.numel())


def _per_layer_mean(trace: list[torch.Tensor]) -> list[float]:
    return [float(t.double().mean()) for t in trace]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--corpus", required=True, help="One pre-tokenized chain .pt file")
    ap.add_argument("--score_window", type=int, default=4)
    ap.add_argument("--max_chains", type=int, default=80,
                    help="Cap chain count for tractability.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    device = torch.device(a.device)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    cfg = model.config
    memres_mode = getattr(cfg, "memres_mode", "simple_gate")
    parity_init = getattr(cfg, "block_attnres_parity_init", False)
    if memres_mode in ("attention_base",) and parity_init:
        memres_mode_eff = "attention_parity"
    else:
        memres_mode_eff = memres_mode
    print(f"model loaded; memres_mode={memres_mode_eff}", flush=True)

    blob = _load_blob(Path(a.corpus))
    n_chains = int(blob["chain_starts"].shape[0])
    chain_lengths = blob["chain_lengths"]
    chain_names = blob["chain_names"]
    n_chains_eff = min(n_chains, a.max_chains)

    sum_per_layer: dict[str, list[torch.Tensor]] = {"mem": [], "shuffle": []}
    cnt_per_layer: dict[str, list[torch.Tensor]] = {"mem": [], "shuffle": []}
    sum_per_decile: dict[str, list[list[float]]] = {"mem": [], "shuffle": []}
    cnt_per_decile: dict[str, list[list[float]]] = {"mem": [], "shuffle": []}
    sum_by_chain_depth: dict[str, dict[int, list[float]]] = {
        "mem": {k: [] for k in range(a.score_window)},
        "shuffle": {k: [] for k in range(a.score_window)},
    }
    n_score_positions = 0
    raw_layer_means: list[dict] = []

    for ci in tqdm(range(n_chains_eff), desc="routing chains"):
        length = int(chain_lengths[ci])
        if length < a.score_window + 1:
            continue
        try:
            prefix_M = _build_M_at_each_step(model, blob, ci, length, device)
        except Exception as e:
            print(f"chain {ci} build failed: {e}", flush=True)
            continue

        shuffle_idx = (ci + 1) % n_chains
        shuffle_len = int(chain_lengths[shuffle_idx])
        try:
            prefix_M_sh = _build_M_at_each_step(
                model, blob, shuffle_idx, min(length, shuffle_len), device
            )
        except Exception as e:
            print(f"chain {ci} shuffle build failed: {e}", flush=True)
            prefix_M_sh = []

        # Last `score_window` sessions: depth k = score_window - 1 - rel
        # (rel=0 means oldest of the scored window).
        score_starts = list(range(length - a.score_window, length))
        for rel, end in enumerate(score_starts):
            sess = _session_at(blob, ci, end).to(device).unsqueeze(0)
            input_ids = sess[:, :-1]

            # mem
            try:
                trace_mem = _alpha_trace_for(model, input_ids, prefix_M[end])
            except Exception as e:
                print(f"chain {ci} pos {end} mem trace failed: {e}", flush=True)
                continue
            if not trace_mem:
                # Model returned no alphas (e.g. M_c is None at the very start);
                # skip silently.
                continue
            n_score_positions += 1

            _accumulate(sum_per_layer["mem"], cnt_per_layer["mem"], trace_mem)
            _accumulate_position_decile(
                sum_per_decile["mem"], cnt_per_decile["mem"], trace_mem
            )
            sum_by_chain_depth["mem"][rel].append(
                float(sum(t.double().mean() for t in trace_mem) / max(1, len(trace_mem)))
            )
            raw_layer_means.append({
                "chain_id": str(chain_names[ci]),
                "depth": int(rel),
                "condition": "mem",
                "alpha_per_sublayer": _per_layer_mean(trace_mem),
            })

            # shuffle
            if end < len(prefix_M_sh):
                try:
                    trace_sh = _alpha_trace_for(model, input_ids, prefix_M_sh[end])
                except Exception as e:
                    print(f"chain {ci} pos {end} shuffle trace failed: {e}", flush=True)
                    trace_sh = []
                if trace_sh:
                    _accumulate(sum_per_layer["shuffle"], cnt_per_layer["shuffle"], trace_sh)
                    _accumulate_position_decile(
                        sum_per_decile["shuffle"], cnt_per_decile["shuffle"], trace_sh
                    )
                    sum_by_chain_depth["shuffle"][rel].append(
                        float(sum(t.double().mean() for t in trace_sh) / max(1, len(trace_sh)))
                    )
                    raw_layer_means.append({
                        "chain_id": str(chain_names[ci]),
                        "depth": int(rel),
                        "condition": "shuffle",
                        "alpha_per_sublayer": _per_layer_mean(trace_sh),
                    })

    def _by_sublayer_mean(condition: str) -> list[float]:
        out = []
        for s, c in zip(sum_per_layer[condition], cnt_per_layer[condition]):
            denom = float(c.item())
            out.append(float((s / max(1.0, denom)).item()) if denom > 0 else float("nan"))
        return out

    def _by_decile_mean(condition: str) -> list[list[float]]:
        out = []
        for s_row, c_row in zip(sum_per_decile[condition], cnt_per_decile[condition]):
            row = []
            for s, c in zip(s_row, c_row):
                row.append(float(s) / float(c) if c > 0 else float("nan"))
            out.append(row)
        return out

    def _depth_mean(condition: str) -> list[float]:
        return [
            (sum(sum_by_chain_depth[condition][k]) / len(sum_by_chain_depth[condition][k]))
            if sum_by_chain_depth[condition][k] else float("nan")
            for k in range(a.score_window)
        ]

    out = {
        "model_path": str(a.model_path),
        "corpus": str(a.corpus),
        "memres_mode": memres_mode_eff,
        "n_sublayers": len(sum_per_layer["mem"]),
        "n_chains": n_chains_eff,
        "n_score_positions": int(n_score_positions),
        "score_window": int(a.score_window),
        "alpha_mem_by_sublayer": {
            "mem": _by_sublayer_mean("mem"),
            "shuffle": _by_sublayer_mean("shuffle"),
        },
        "alpha_mem_by_sublayer_position_decile": {
            "mem": _by_decile_mean("mem"),
            "shuffle": _by_decile_mean("shuffle"),
        },
        "alpha_mem_by_chain_depth": {
            "mem": _depth_mean("mem"),
            "shuffle": _depth_mean("shuffle"),
        },
        "raw_layer_means_sample": raw_layer_means[:200],
    }

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {a.output}", flush=True)
    print(json.dumps({
        "n_sublayers": out["n_sublayers"],
        "n_score_positions": out["n_score_positions"],
        "alpha_mem_by_sublayer.mem.mean": (
            sum(out["alpha_mem_by_sublayer"]["mem"]) / max(1, len(out["alpha_mem_by_sublayer"]["mem"]))
        ),
        "alpha_mem_by_sublayer.shuffle.mean": (
            sum(out["alpha_mem_by_sublayer"]["shuffle"]) / max(1, len(out["alpha_mem_by_sublayer"]["shuffle"]))
        ),
        "alpha_mem_by_chain_depth.mem": out["alpha_mem_by_chain_depth"]["mem"],
        "alpha_mem_by_chain_depth.shuffle": out["alpha_mem_by_chain_depth"]["shuffle"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
