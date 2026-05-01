#!/usr/bin/env python3
"""D5: Test-Time-Training on the readout only.

The architecture audit (2026-05-01) raised two competing hypotheses
for why the v11 callback-token Δ_sh-m signal stays small:

  W: the WRITER (extract + judge + M_c update) is content-blind.  M_c
     does not actually contain the information needed to answer the
     callback, so no readout can recover it.  Architecturally fatal.

  R: the WRITER does encode the information; the READOUT (single-shot
     cross-attention from input embeddings to M_c) is too shallow to
     decode it.  Architecturally fixable (deeper readout, layer-wise
     readouts, ...).

Both produce the same empirical signature in normal training (small
Δ_sh-m).  This script disambiguates by *freezing every parameter
except the readout's W_Q / W_K / W_V*, then doing a small number of
gradient steps on a held-out training split, supervising callback
tokens only.

Decision rule:

  callback CE drops substantially (e.g., >= 30%) after a few hundred
  TTT steps  =>  the writer ENCODED the information; readout was
                 the bottleneck.  Path R.

  callback CE does not drop (or even rises)  =>  the writer did NOT
                 encode the information; no readout can save us.  Path
                 W.  Re-architect the writer.

Run-cost: a few thousand callback tokens, single GPU, < 10 minutes
on a ~0.6B model.

Usage:
  python tools/d5_ttt_readout.py \\
      --ckpt   results/exp2_chain_recipe/v11g/best \\
      --train_corpus paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \\
      --eval_corpus  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \\
      --steps 300 --batch_size 4 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_chain import ChainCorpus  # noqa: E402
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to a save_pretrained() checkpoint dir.")
    p.add_argument("--train_corpus", required=True,
                   help="ChainCorpus .pt for TTT training split.")
    p.add_argument("--eval_corpus", required=True,
                   help="ChainCorpus .pt for held-out scoring; never "
                        "shown to the optimizer.")
    p.add_argument("--steps", type=int, default=300,
                   help="Number of optimizer steps on the readout.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate; high relative to a typical "
                        "training run because we are tuning ~3 small "
                        "matrices.")
    p.add_argument("--max_burn_in", type=int, default=12,
                   help="Cap on number of pre-callback sessions to "
                        "feed into the writer per chain.  Mirrors "
                        "the trainer's burn_in_max convention.")
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--n_eval_chains", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None,
                   help="Optional path to dump the JSON metric trace.")
    return p.parse_args()


def _build_window_for_chain(
    corpus: ChainCorpus,
    chain_idx: int,
    max_burn_in: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Return (burn_in_ids, callback_ids, callback_mask) for a chain.

    burn_in_ids: (B<=max_burn_in, S) on device.  None if cb_pos == 0
    callback_ids: (1, S) on device.
    callback_mask: (1, S) on device, int8 bitmask of supervisable
        positions in the callback session.  If no positions are
        marked, returns an empty mask and the caller should skip.
    """
    cb_pos = int(corpus.chain_callback_position[chain_idx])
    if cb_pos < 0:
        cb_pos = int(corpus.chain_lengths[chain_idx]) - 1
    chain_len = int(corpus.chain_lengths[chain_idx])
    cb_pos = min(cb_pos, chain_len - 1)

    if cb_pos == 0:
        burn_in = None
    else:
        start = max(0, cb_pos - max_burn_in)
        burn_in = corpus.chain_window(chain_idx, start, cb_pos - start)
        burn_in = burn_in.to(device)
    cb_session = corpus.chain_session_at(chain_idx, cb_pos).to(device).unsqueeze(0)
    cb_mask = corpus.chain_window_callback_mask(chain_idx, cb_pos, 1).to(device)
    return burn_in, cb_session, cb_mask


@torch.no_grad()
def _build_M_c(
    model: Qwen3MemResForCausalLM,
    burn_in_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the writer recurrently over burn_in_ids; return M_c or None."""
    if burn_in_ids is None or burn_in_ids.numel() == 0:
        return None
    inner = model.model  # Qwen3MemResModel
    M_c = None
    for t in range(burn_in_ids.size(0)):
        sess = burn_in_ids[t : t + 1]
        C = inner.extract_source(sess)  # (1, N, d)
        if M_c is None:
            M_c_prev = torch.zeros(
                1, model.config.memres_num_vectors,
                model.config.hidden_size,
                device=sess.device, dtype=C.dtype,
            )
        else:
            M_c_prev = M_c
        M_c = inner.memory_block(C, M_c_prev)
    return M_c


def _callback_token_ce(
    model: Qwen3MemResForCausalLM,
    cb_session: torch.Tensor,
    cb_mask: torch.Tensor,
    M_c: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_callback_ce, n_supervised_tokens).

    Cross-entropy on shift-positions (predict input_ids[1:] from
    input_ids[:-1]) where cb_mask[:, 1:] == 1.  The mask aligns with
    *targets*, not inputs, so the supervisory bit is whether the
    predicted token is callback-supervised.
    """
    input_ids = cb_session[:, :-1]
    targets = cb_session[:, 1:]
    out = model(input_ids=input_ids, M_c=M_c)
    logits = out.logits  # (1, S-1, V)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, S-1)
    target_mask = cb_mask[:, 1:].to(nll.dtype)
    n = target_mask.sum()
    if n.item() < 1:
        return torch.tensor(0.0, device=nll.device), n
    return (nll * target_mask).sum() / n, n


def _eval_callback_ce(
    model: Qwen3MemResForCausalLM,
    corpus: ChainCorpus,
    n_chains: int,
    max_burn_in: int,
    device: torch.device,
) -> dict:
    """Eval helper: mean callback-token CE over up to ``n_chains``."""
    model.eval()
    total_nll = 0.0
    total_n = 0.0
    n_eligible = 0
    for ci in range(min(n_chains, len(corpus))):
        burn_in, cb, mask = _build_window_for_chain(
            corpus, ci, max_burn_in, device
        )
        if mask.sum().item() < 1:
            continue
        with torch.no_grad():
            M_c = _build_M_c(model, burn_in)
            ce, n = _callback_token_ce(model, cb, mask, M_c)
        if n.item() < 1:
            continue
        total_nll += float(ce.item()) * float(n.item())
        total_n += float(n.item())
        n_eligible += 1
    if total_n < 1:
        return {"callback_ce": float("nan"), "n_chains": 0, "n_tokens": 0}
    return {
        "callback_ce": total_nll / total_n,
        "n_chains": n_eligible,
        "n_tokens": int(total_n),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model = Qwen3MemResForCausalLM.from_pretrained(args.ckpt)
    model.to(args.device)

    train_corpus = ChainCorpus(Path(args.train_corpus))
    eval_corpus = ChainCorpus(Path(args.eval_corpus))
    print(f"Train corpus: {len(train_corpus)} chains, "
          f"session_len={train_corpus.session_len}", flush=True)
    print(f"Eval corpus:  {len(eval_corpus)} chains, "
          f"session_len={eval_corpus.session_len}", flush=True)

    readout_params = list(model.model.memory_readout.parameters())  # noqa
    n_readout = sum(p.numel() for p in readout_params)
    print(f"Readout params (W_Q+W_K+W_V+norm): {n_readout:,}", flush=True)

    for name, p in model.named_parameters():
        if p in set(readout_params):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen all but readout: {n_trainable:,} params trainable.",
          flush=True)

    optim = torch.optim.AdamW(readout_params, lr=args.lr, weight_decay=0.0)

    # Baseline eval.
    base = _eval_callback_ce(
        model, eval_corpus, args.n_eval_chains, args.max_burn_in,
        torch.device(args.device),
    )
    print(f"BASELINE  callback_ce={base['callback_ce']:.4f}  "
          f"n_chains={base['n_chains']}  n_tokens={base['n_tokens']}",
          flush=True)
    history = [{"step": 0, **base}]

    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    n_train = len(train_corpus)
    t_start = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        # Freezing requires_grad doesn't put backbone modules in
        # eval-only mode -- the dropout + norm running stats are still
        # active during model.train().  This is fine: we only update
        # readout params, but the forward path uses train-mode
        # backbone exactly the way the original training did.
        optim.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_n = 0.0
        attempts = 0
        kept = 0
        while kept < args.batch_size and attempts < args.batch_size * 4:
            attempts += 1
            ci = int(torch.randint(
                0, n_train, (1,), generator=rng
            ).item())
            burn_in, cb, mask = _build_window_for_chain(
                train_corpus, ci, args.max_burn_in,
                torch.device(args.device),
            )
            if mask.sum().item() < 1:
                continue
            M_c = _build_M_c(model, burn_in)
            # Re-enable grad on M_c with respect to the readout *only*
            # by detaching the writer output -- writer params are
            # frozen but their forward graph holds activations we
            # don't need to backprop through.
            if M_c is not None:
                M_c = M_c.detach()
            ce, n = _callback_token_ce(model, cb, mask, M_c)
            if n.item() < 1:
                continue
            (ce / args.batch_size).backward()
            batch_loss += float(ce.item()) * float(n.item())
            batch_n += float(n.item())
            kept += 1

        if kept == 0:
            print(f"step {step}: no usable chains in batch, skipping",
                  flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(readout_params, 1.0)
        optim.step()

        if step % 10 == 0:
            avg_ce = batch_loss / max(1.0, batch_n)
            elapsed = time.time() - t_start
            print(f"step {step:4d} | train_callback_ce={avg_ce:.4f} | "
                  f"batch_n={int(batch_n)} | "
                  f"elapsed {elapsed:.1f}s",
                  flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            ev = _eval_callback_ce(
                model, eval_corpus, args.n_eval_chains, args.max_burn_in,
                torch.device(args.device),
            )
            delta = ev["callback_ce"] - base["callback_ce"]
            rel = (delta / max(base["callback_ce"], 1e-8)) * 100.0
            print(
                f"  EVAL @ step {step}: callback_ce={ev['callback_ce']:.4f} "
                f"(Δ vs baseline = {delta:+.4f}, {rel:+.1f}%) "
                f"n_chains={ev['n_chains']} n_tokens={ev['n_tokens']}",
                flush=True,
            )
            history.append({"step": step, **ev})

    final = history[-1]
    delta = final["callback_ce"] - base["callback_ce"]
    rel = (delta / max(base["callback_ce"], 1e-8)) * 100.0
    print()
    print("=" * 60)
    print(f"D5 RESULT: TTT on readout only")
    print(f"  baseline callback_ce: {base['callback_ce']:.4f}")
    print(f"  final    callback_ce: {final['callback_ce']:.4f}")
    print(f"  Δ                   : {delta:+.4f}  ({rel:+.1f}%)")
    if rel <= -30:
        verdict = ("LIKELY R: the writer encoded the information; "
                   "the readout was the bottleneck.")
    elif rel <= -10:
        verdict = ("MIXED: readout helps somewhat but a non-trivial "
                   "fraction of the gap remains.  The writer is "
                   "partially content-blind.")
    elif rel <= 0:
        verdict = ("LIKELY W: TTT moves callback CE only marginally; "
                   "the writer is the bottleneck.")
    else:
        verdict = ("PATHOLOGICAL: TTT made callback CE WORSE.  "
                   "Either the readout was already at a local optimum "
                   "or the writer's M_c is anti-correlated with the "
                   "task.")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({
                "ckpt": args.ckpt,
                "train_corpus": args.train_corpus,
                "eval_corpus": args.eval_corpus,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_burn_in": args.max_burn_in,
                "history": history,
                "baseline": base,
                "final": final,
                "delta": delta,
                "delta_pct": rel,
                "verdict": verdict,
            }, f, indent=2)
        print(f"Wrote trace -> {out}")


if __name__ == "__main__":
    main()
