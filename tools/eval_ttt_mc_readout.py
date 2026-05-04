#!/usr/bin/env python3
"""Extended TTT-on-(M_c, W_V_read, W_K_read, W_Q_read) capacity probe.

§5 from v17_wildcards (`tools/eval_ttt_mc.py`) returned 6/6 NEG with
*only* M_c TTT-able and the entire read-side frozen.  That falsifies
"readout has capacity for chain-specific recall **as currently
configured**" -- it does not separate (a) the readout architecture
lacks capacity from (b) joint training drove the readout into a
content-blind local minimum that no M_c can escape.

This probe decouples the two by adding the MemoryReadout's W_Q/W_K/W_V
to the TTT-able parameter set.  Three modes via --readout_unfreeze:

  ``v_only``   -- TTT on (M_c, W_V_read).  Cheapest expansion.
  ``qkv``      -- TTT on (M_c, W_Q_read, W_K_read, W_V_read).
  ``qkv_reset`` -- as ``qkv`` but the readout Q/K/V are re-initialised
                  from N(0, d**-0.5) before TTT.  Tests intrinsic
                  capacity from a clean init -- if this also fails the
                  readout architecture itself is the bottleneck.

Decision rule (mirrors §5 schema):

  ttt_lift_vs_floor > +0.30  ->  read-side has capacity; joint training
                                 collapsed it.  Read-side probe loss
                                 (v18) is the right intervention.
  in (0, +0.30]              ->  partial; read-side probe loss is
                                 necessary but a deeper architectural
                                 change (e.g. multi-layer readout
                                 depth) may also be required.
  <= 0                       ->  read-side architecture itself cannot
                                 represent chain-specific recall.
                                 Multi-layer readout depth (Perceiver-
                                 style stack) becomes load-bearing on
                                 top of probe loss.

Compute: ~30s/chain at K=80 SGD steps on a frozen 0.6B; full sweep of
3 modes x 2 ckpts x 16 chains ~ 25 min on a single H100.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402
from eval_callback import (  # noqa: E402
    chain_session,
    chain_session_mask,
    build_Mc,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_corpus", required=True)
    p.add_argument("--n_chains", type=int, default=16)
    p.add_argument("--ttt_steps", type=int, default=80)
    p.add_argument("--ttt_lr_mc", type=float, default=1e-2)
    p.add_argument("--ttt_lr_readout", type=float, default=1e-3)
    p.add_argument(
        "--readout_unfreeze",
        choices=["v_only", "qkv", "qkv_reset"],
        default="qkv",
    )
    p.add_argument("--init_mode", choices=["writer", "iid"], default="writer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    return p.parse_args()


def evidence_session_loss(model, evidence_ids, M_c):
    out = model(input_ids=evidence_ids, M_c=M_c)
    logits = out.logits[:, :-1, :].float()
    target = evidence_ids[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
    )


@torch.no_grad()
def callback_ce_local(model, callback_ids, callback_mask, M_c):
    out = model(input_ids=callback_ids, M_c=M_c)
    logits = out.logits[:, :-1, :].float()
    target = callback_ids[:, 1:]
    mask = callback_mask[1:].to(logits.device)
    if mask.sum() == 0:
        return float("nan")
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return float(nll[0][mask].mean().item())


def _readout_module(model):
    return model.model.memory_readout


def _snapshot_readout(readout):
    return {
        "W_Q": readout.W_Q.weight.detach().clone(),
        "W_K": readout.W_K.weight.detach().clone(),
        "W_V": readout.W_V.weight.detach().clone(),
    }


def _restore_readout(readout, snap):
    with torch.no_grad():
        readout.W_Q.weight.copy_(snap["W_Q"])
        readout.W_K.weight.copy_(snap["W_K"])
        readout.W_V.weight.copy_(snap["W_V"])


def _set_readout_trainable(readout, mode: str, hidden_size: int):
    """Make selected readout projections trainable for TTT.

    Returns the list of parameters that should be added to the optimiser.
    Parameters not returned remain `requires_grad_(False)` (already set
    by the caller).
    """
    params = []
    if mode == "v_only":
        readout.W_V.weight.requires_grad_(True)
        params.append(readout.W_V.weight)
    elif mode in ("qkv", "qkv_reset"):
        for w in (readout.W_Q.weight, readout.W_K.weight, readout.W_V.weight):
            w.requires_grad_(True)
            params.append(w)
        if mode == "qkv_reset":
            std = hidden_size ** -0.5
            with torch.no_grad():
                for w in (readout.W_Q.weight, readout.W_K.weight,
                          readout.W_V.weight):
                    w.data.normal_(mean=0.0, std=std)
    else:
        raise ValueError(f"unknown readout_unfreeze={mode!r}")
    return params


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"loading {args.ckpt}", flush=True)
    model = Qwen3MemResForCausalLM.from_pretrained(
        args.ckpt, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    print(f"loading {args.eval_corpus}", flush=True)
    blob = torch.load(args.eval_corpus, weights_only=False, map_location="cpu")
    n_chains = min(args.n_chains, int(blob["chain_starts"].shape[0]))

    K = model.config.memres_num_vectors
    d = model.config.hidden_size

    distractor_pool: list[torch.Tensor] = []
    pool_target = min(64, n_chains)
    for ci in range(pool_target):
        cb_pos = int(blob["chain_callback_position"][ci])
        last_filler_pos = max(0, cb_pos - 1)
        distractor_pool.append(chain_session(blob, ci, last_filler_pos))

    readout = _readout_module(model)
    snap = _snapshot_readout(readout)
    print(
        f"readout snapshot: ||W_Q||={snap['W_Q'].float().norm().item():.3f} "
        f"||W_K||={snap['W_K'].float().norm().item():.3f} "
        f"||W_V||={snap['W_V'].float().norm().item():.3f}",
        flush=True,
    )

    per_chain = []
    t0 = time.time()
    for ci in range(n_chains):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
            continue

        ev_pos_list = list(blob.get("chain_evidence_positions", [[]])[ci])
        if not ev_pos_list:
            continue
        ev_pos = int(ev_pos_list[0])
        ev_ids = chain_session(blob, ci, ev_pos).to(args.device).unsqueeze(0)

        cb_ids = chain_session(blob, ci, cb_pos).to(args.device).unsqueeze(0)
        cb_mask = chain_session_mask(blob, ci, cb_pos)
        if cb_mask is None or cb_mask.sum() == 0:
            continue

        n_evidence = len(ev_pos_list)
        offset = (ci * 7) % max(1, len(distractor_pool) - n_evidence)
        redact_with = torch.stack(
            distractor_pool[offset:offset + n_evidence]
        ) if distractor_pool else None

        # Reset readout to the trained checkpoint state before each chain
        # so TTT lifts are independent (chain c's TTT does not bleed into
        # chain c+1).
        _restore_readout(readout, snap)

        # Snapshot ce_writer + ce_floor under the *original* (frozen)
        # readout, before re-enabling readout grads.
        with torch.no_grad():
            Mc_writer = build_Mc(model, blob, ci, cb_pos, args.device)
            Mc_floor = build_Mc(
                model, blob, ci, cb_pos, args.device,
                evidence_redact=True,
                redact_with=redact_with,
            )
        ce_writer = callback_ce_local(model, cb_ids, cb_mask, Mc_writer)
        ce_floor = callback_ce_local(model, cb_ids, cb_mask, Mc_floor)

        # Initialise M_c (TTT-able)
        if args.init_mode == "writer":
            Mc_init = Mc_writer.detach().clone()
        else:
            Mc_init = (
                torch.randn(1, K, d, device=args.device, dtype=torch.bfloat16)
                * (d ** -0.5)
            )

        Mc_ttt = Mc_init.detach().clone().float().requires_grad_(True)

        # Set up readout TTT-able params
        readout_params = _set_readout_trainable(readout, args.readout_unfreeze, d)

        # Two param groups: M_c at higher LR (i.i.d. shape, fewer params),
        # readout projections at smaller LR (millions of params, easy to
        # destabilise).
        opt = torch.optim.Adam(
            [
                {"params": [Mc_ttt], "lr": args.ttt_lr_mc},
                {"params": readout_params, "lr": args.ttt_lr_readout},
            ]
        )
        ev_loss_trace = []
        for step in range(args.ttt_steps):
            opt.zero_grad()
            loss = evidence_session_loss(model, ev_ids, Mc_ttt.bfloat16())
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
            if step in (0, 9, 24, 49, 79) or step == args.ttt_steps - 1:
                ev_loss_trace.append((step, float(loss.item())))

        Mc_ttt_eval = Mc_ttt.detach().bfloat16()
        ce_ttt = callback_ce_local(model, cb_ids, cb_mask, Mc_ttt_eval)

        # Tear down readout grads + restore for the next chain.
        for w in (readout.W_Q.weight, readout.W_K.weight, readout.W_V.weight):
            w.requires_grad_(False)
        _restore_readout(readout, snap)

        per_chain.append({
            "chain_idx": ci,
            "cb_pos": cb_pos,
            "ev_pos": ev_pos,
            "n_callback_tok": int(cb_mask.sum().item()),
            "ce_writer": ce_writer,
            "ce_floor": ce_floor,
            "ce_ttt": ce_ttt,
            "ev_loss_trace": ev_loss_trace,
        })
        if (ci + 1) % 2 == 0 or ci == n_chains - 1:
            ev_first = ev_loss_trace[0][1] if ev_loss_trace else float("nan")
            ev_last = ev_loss_trace[-1][1] if ev_loss_trace else float("nan")
            print(
                f"  chain {ci+1}/{n_chains} | mode={args.readout_unfreeze} | "
                f"ev_loss {ev_first:.3f} -> {ev_last:.3f} | "
                f"cb_ce writer={ce_writer:.3f} floor={ce_floor:.3f} "
                f"ttt={ce_ttt:.3f} | "
                f"lift_floor-ttt={ce_floor - ce_ttt:+.3f} | "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )

    if not per_chain:
        print("ERROR: no chains scored", flush=True)
        return

    ce_writer = statistics.mean(c["ce_writer"] for c in per_chain)
    ce_floor = statistics.mean(c["ce_floor"] for c in per_chain)
    ce_ttt = statistics.mean(c["ce_ttt"] for c in per_chain)
    ttt_lift_vs_floor = ce_floor - ce_ttt
    ttt_lift_vs_writer = ce_writer - ce_ttt

    if ttt_lift_vs_floor > 0.30:
        verdict = (
            "POSITIVE: read-side has capacity; joint training collapsed it. "
            "Read-side probe loss (v18) is the right intervention."
        )
    elif ttt_lift_vs_floor > 0.0:
        verdict = (
            "MIXED: partial read-side capacity. Read-side probe loss "
            "necessary; multi-layer readout depth (Perceiver) likely also."
        )
    else:
        verdict = (
            "NEGATIVE: read-side architecture itself cannot represent "
            "chain-specific recall. Multi-layer readout depth becomes "
            "load-bearing on top of probe loss."
        )

    summary = {
        "ckpt": args.ckpt,
        "eval_corpus": args.eval_corpus,
        "init_mode": args.init_mode,
        "readout_unfreeze": args.readout_unfreeze,
        "ttt_steps": args.ttt_steps,
        "ttt_lr_mc": args.ttt_lr_mc,
        "ttt_lr_readout": args.ttt_lr_readout,
        "n_chains_scored": len(per_chain),
        "ce_writer_mean": ce_writer,
        "ce_floor_mean": ce_floor,
        "ce_ttt_mean": ce_ttt,
        "ttt_lift_vs_floor": ttt_lift_vs_floor,
        "ttt_lift_vs_writer": ttt_lift_vs_writer,
        "verdict": verdict,
        "per_chain": per_chain,
    }
    print()
    print("=" * 72)
    print(
        f"RESULT: TTT-on-(M_c, readout {args.readout_unfreeze}) "
        f"init={args.init_mode} K={args.ttt_steps}"
    )
    print(f"  ckpt          : {args.ckpt}")
    print(f"  n_chains      : {len(per_chain)}")
    print(f"  ce_writer     : {ce_writer:.4f}")
    print(f"  ce_floor      : {ce_floor:.4f}")
    print(f"  ce_ttt        : {ce_ttt:.4f}")
    print(f"  TTT lift vs floor : {ttt_lift_vs_floor:+.4f}  <-- DECISION")
    print(f"  TTT lift vs writer: {ttt_lift_vs_writer:+.4f}")
    print(f"  VERDICT       : {verdict}")
    print("=" * 72)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote summary -> {out}", flush=True)


if __name__ == "__main__":
    main()
