#!/usr/bin/env python3
"""Run the v8+ recruitment + phase-aligned diagnostics on any checkpoint.

This tool exists so that the diagnostic lens introduced for the v8
training loop (see ``Trainer._phase_aligned_eval``,
``Trainer._routing_recruitment_summary``,
``Trainer._readout_magnitude_diag`` in ``train_chain.py``) can be applied
*retroactively* to any saved Memory-Residuals checkpoint without
re-running the full trainer.  Use it to:

  - Demonstrate "the standard eval said memory is poison; the
    phase-aligned eval says the callback-token Δ_sh-m is +X".
  - Inspect a routing-collapsed (attention_parity, gate_max=0)
    checkpoint and confirm that the readout magnitude is non-trivial
    and the per-sublayer α_mem trace really is content-blind.
  - Compare two checkpoints (e.g. simple_gate vs attention_parity)
    on the same eval corpus with the same evidence/callback pairs.

Outputs a JSON to ``--output`` and prints a one-line summary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling_memres import (  # noqa: E402
    Qwen3MemResForCausalLM,
    _normalise_memres_mode,
)


def load_corpus(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def chain_session_at(blob: dict, ci: int, pos: int) -> torch.Tensor:
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + pos].long()


def session_callback_mask_at(blob: dict, ci: int, pos: int) -> torch.Tensor:
    s = int(blob["chain_starts"][ci])
    if "session_callback_mask" not in blob:
        return torch.zeros_like(blob["session_ids"][s + pos], dtype=torch.int8)
    return blob["session_callback_mask"][s + pos].to(torch.int8)


def chain_callback_position(blob: dict, ci: int) -> int:
    if "chain_callback_position" not in blob:
        return -1
    return int(blob["chain_callback_position"][ci])


@torch.no_grad()
def per_position_nll(
    model: Qwen3MemResForCausalLM,
    input_ids: torch.Tensor,
    M_c: torch.Tensor | None,
) -> torch.Tensor:
    out = model(input_ids=input_ids, M_c=M_c)
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.squeeze(0)  # (S-1,)


@torch.no_grad()
def phase_aligned_eval(
    model: Qwen3MemResForCausalLM,
    blob: dict,
    device: torch.device,
    eos_id: int,
    n_chains: int,
    seed: int,
) -> dict:
    """Phase-aligned eval: 1 fresh evidence + 1 callback session per chain.

    Mirror of ``Trainer._phase_aligned_eval`` in ``train_chain.py``.
    """
    rng = random.Random(seed)
    n_total = int(blob["chain_starts"].shape[0])
    eligible = [
        i for i in range(n_total)
        if chain_callback_position(blob, i) >= 1
    ]
    if len(eligible) < 2:
        return {"n_pa_scored": 0, "n_pa_cb_scored": 0}
    rng.shuffle(eligible)
    sample = eligible[:n_chains]

    ws_mem, ws_no, ws_sh = [], [], []
    cb_mem, cb_no, cb_sh = [], [], []
    for ci in sample:
        cb_pos = chain_callback_position(blob, ci)
        e = rng.randint(0, cb_pos - 1)
        evidence = chain_session_at(blob, ci, e).to(device).unsqueeze(0)
        callback = chain_session_at(blob, ci, cb_pos).to(device).unsqueeze(0)
        cb_mask = session_callback_mask_at(blob, ci, cb_pos).to(device)

        C_e = model.model.extract_source(evidence[:, :-1])
        M_c = model.model.compress_session(C_e, None)

        others = [j for j in eligible if j != ci]
        if not others:
            M_sh = None
        else:
            other = rng.choice(others)
            o_cb = chain_callback_position(blob, other)
            o_e = rng.randint(0, o_cb - 1)
            o_ev = chain_session_at(blob, other, o_e).to(device).unsqueeze(0)
            C_o = model.model.extract_source(o_ev[:, :-1])
            M_sh = model.model.compress_session(C_o, None)

        input_ids = callback[:, :-1]
        valid = (input_ids[0, :] != eos_id).float()
        cb_mask_in = cb_mask[: input_ids.shape[1]].float() * valid
        valid_sh = valid[1:]
        cb_mask_sh = cb_mask_in[1:]

        nll_mem = per_position_nll(model, input_ids, M_c)
        nll_no = per_position_nll(model, input_ids, None)
        nll_sh = per_position_nll(model, input_ids, M_sh) if M_sh is not None else None

        v_sum = float(valid_sh.sum().item())
        if v_sum > 0:
            ws_mem.append(float((nll_mem * valid_sh).sum().item()) / v_sum)
            ws_no.append(float((nll_no * valid_sh).sum().item()) / v_sum)
            if nll_sh is not None:
                ws_sh.append(float((nll_sh * valid_sh).sum().item()) / v_sum)

        cb_sum = float(cb_mask_sh.sum().item())
        if cb_sum > 0:
            cb_mem.append(float((nll_mem * cb_mask_sh).sum().item()) / cb_sum)
            cb_no.append(float((nll_no * cb_mask_sh).sum().item()) / cb_sum)
            if nll_sh is not None:
                cb_sh.append(float((nll_sh * cb_mask_sh).sum().item()) / cb_sum)

    def m(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    out = {
        "n_pa_scored": len(ws_mem),
        "n_pa_cb_scored": len(cb_mem),
        "pa_ws_ce_mem": m(ws_mem),
        "pa_ws_ce_nomem": m(ws_no),
        "pa_ws_ce_shuffle": m(ws_sh),
        "pa_cb_ce_mem": m(cb_mem),
        "pa_cb_ce_nomem": m(cb_no),
        "pa_cb_ce_shuffle": m(cb_sh),
    }
    out["pa_ws_dnm"] = (
        out["pa_ws_ce_nomem"] - out["pa_ws_ce_mem"]
        if ws_no and ws_mem else float("nan")
    )
    out["pa_ws_dsh"] = (
        out["pa_ws_ce_shuffle"] - out["pa_ws_ce_mem"]
        if ws_sh and ws_mem else float("nan")
    )
    out["pa_cb_dnm"] = (
        out["pa_cb_ce_nomem"] - out["pa_cb_ce_mem"]
        if cb_no and cb_mem else float("nan")
    )
    out["pa_cb_dsh"] = (
        out["pa_cb_ce_shuffle"] - out["pa_cb_ce_mem"]
        if cb_sh and cb_mem else float("nan")
    )
    return out


@torch.no_grad()
def routing_recruitment_summary(
    model: Qwen3MemResForCausalLM,
    blob: dict,
    device: torch.device,
    n_chains: int,
) -> dict:
    mode = _normalise_memres_mode(
        getattr(model.config, "memres_mode", "simple_gate"),
        getattr(model.config, "block_attnres_parity_init", None),
    )
    if mode == "simple_gate":
        gate = model.model.memory_gate.gate.float().detach().cpu()
        absg = gate.abs()
        n = absg.numel()
        k = min(3, n)
        top_idx = torch.topk(absg, k=k).indices.tolist()
        top = [(int(i), float(gate[i])) for i in top_idx]
        return {
            "rec_mode": "simple_gate",
            "rec_gate_max_abs": float(absg.max()),
            "rec_gate_mean_abs": float(absg.mean()),
            "rec_gate_top": top,
            "rec_frac_open": float((absg > 1e-3).float().mean()),
        }

    eligible = [
        i for i in range(int(blob["chain_starts"].shape[0]))
        if chain_callback_position(blob, i) >= 1
    ][:n_chains]
    if not eligible:
        return {"rec_mode": mode, "rec_alpha_mem_max": 0.0}
    per_sublayer_acc = []
    for ci in eligible:
        cb_pos = chain_callback_position(blob, ci)
        e = max(0, cb_pos - 1)
        evidence = chain_session_at(blob, ci, e).to(device).unsqueeze(0)
        callback = chain_session_at(blob, ci, cb_pos).to(device).unsqueeze(0)
        C_e = model.model.extract_source(evidence[:, :-1])
        M_c = model.model.compress_session(C_e, None)
        out = model(input_ids=callback[:, :-1], M_c=M_c, collect_alpha_trace=True)
        trace = getattr(out, "alpha_trace", None)
        if not trace:
            continue
        per_sublayer_acc.append([float(a.float().mean().item()) for a in trace])
    if not per_sublayer_acc:
        return {"rec_mode": mode, "rec_alpha_mem_max": 0.0}
    n_sub = len(per_sublayer_acc[0])
    mean_per_sub = [
        sum(row[i] for row in per_sublayer_acc) / len(per_sublayer_acc)
        for i in range(n_sub)
    ]
    arr = torch.tensor(mean_per_sub)
    k = min(3, n_sub)
    top_idx = torch.topk(arr, k=k).indices.tolist()
    top = [(int(i), float(arr[i])) for i in top_idx]
    return {
        "rec_mode": mode,
        "rec_alpha_mem_max": float(arr.max()),
        "rec_alpha_mem_mean": float(arr.mean()),
        "rec_alpha_mem_top": top,
        "rec_frac_open": float((arr > 0.05).float().mean()),
        "rec_per_sublayer_alpha_mem": mean_per_sub,
    }


@torch.no_grad()
def readout_magnitude_diag(
    model: Qwen3MemResForCausalLM,
    blob: dict,
    device: torch.device,
    n_chains: int,
) -> dict:
    eligible = [
        i for i in range(int(blob["chain_starts"].shape[0]))
        if chain_callback_position(blob, i) >= 1
    ][:n_chains]
    if not eligible:
        return {"mt_norm_ratio_mean": float("nan")}
    ratios = []
    for ci in eligible:
        cb_pos = chain_callback_position(blob, ci)
        e = max(0, cb_pos - 1)
        evidence = chain_session_at(blob, ci, e).to(device).unsqueeze(0)
        callback = chain_session_at(blob, ci, cb_pos).to(device).unsqueeze(0)
        C_e = model.model.extract_source(evidence[:, :-1])
        M_c = model.model.compress_session(C_e, None)
        X = model.model.embed_tokens(callback[:, :-1])
        m_t = model.model.memory_readout(X, M_c)
        mt_n = m_t.float().norm(dim=-1).mean().item()
        h_n = X.float().norm(dim=-1).mean().item()
        ratios.append(mt_n / max(h_n, 1e-6))
    return {
        "mt_norm_ratio_mean": float(sum(ratios) / len(ratios)),
        "mt_norm_ratio_max": float(max(ratios)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--corpus", required=True,
                   help="Pre-tokenised chain corpus .pt with "
                        "session_callback_mask + chain_callback_position "
                        "(LME-style).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_pa_chains", type=int, default=48)
    p.add_argument("--n_diag_chains", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"loading model: {args.model_path}", flush=True)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.model_path)
    eos_id = int(tok.eos_token_id)
    print(f"loading corpus: {args.corpus}", flush=True)
    blob = load_corpus(Path(args.corpus))
    print(
        f"  {int(blob['chain_starts'].shape[0])} chains, "
        f"session_len={int(blob['session_len'])}",
        flush=True,
    )

    print("running phase-aligned eval ...", flush=True)
    pa = phase_aligned_eval(
        model, blob, device, eos_id, args.n_pa_chains, args.seed
    )
    print("running routing recruitment ...", flush=True)
    rec = routing_recruitment_summary(model, blob, device, args.n_diag_chains)
    print("running readout magnitude ...", flush=True)
    mt = readout_magnitude_diag(model, blob, device, args.n_diag_chains)

    out = {
        "model_path": str(args.model_path),
        "corpus": str(args.corpus),
        **pa, **rec, **mt,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote -> {args.output}")

    print("\n=== Summary ===")
    print(
        f"PA-EVAL: n={pa['n_pa_scored']} cb_n={pa['n_pa_cb_scored']} | "
        f"WS Δnm-m={pa['pa_ws_dnm']:+.4f} Δsh-m={pa['pa_ws_dsh']:+.4f} | "
        f"CB Δnm-m={pa['pa_cb_dnm']:+.4f} Δsh-m={pa['pa_cb_dsh']:+.4f}"
    )
    if rec.get("rec_mode") == "simple_gate":
        print(
            f"ROUTE: simple_gate |gate|_max={rec['rec_gate_max_abs']:.4f} "
            f"|gate|_mean={rec['rec_gate_mean_abs']:.4f} "
            f"frac_open={rec['rec_frac_open']:.2f}"
        )
    else:
        print(
            f"ROUTE: {rec['rec_mode']} α_mem_max={rec.get('rec_alpha_mem_max', 0.0):.4f} "
            f"α_mem_mean={rec.get('rec_alpha_mem_mean', 0.0):.4f} "
            f"frac_open={rec.get('rec_frac_open', 0.0):.2f}"
        )
    print(
        f"READOUT: ||m^t||/||embed|| mean={mt.get('mt_norm_ratio_mean', float('nan')):.3f} "
        f"max={mt.get('mt_norm_ratio_max', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
