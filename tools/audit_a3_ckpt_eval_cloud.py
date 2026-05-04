#!/usr/bin/env python3
"""AUDIT A3-CLOUD step 3: compute pa_cb_ce_mem on a Qwen3MemRes checkpoint
over n=128 val chains from synthd4v2_persona_callback_val_s512.pt.

Scoring mirrors tools/audit_base_prior.py / tools/eval_callback.py: build
M_c from the chain prefix (positions [0, cb_pos)), run the callback
session with M_c, average CE over the callback-mask positions per chain,
then mean over chains. No shuffle / floor / no-mem terms - just mem-CE.

Also reports per-token CE for (a) the first callback-mask token ("is ")
and (b) the first item-subword (second callback-mask token), and mean
over subsequent subwords. Per the build script the callback mask covers
the FINAL occurrence of the item tokens in the answer string
"...favorite <type> is <item>." – position 0 of the mask is the first
answer-item subword, not the "is " token. We therefore report:
  - ce_first_item_tok: CE on the 1st masked position (1st item subword)
  - ce_subsequent_item_tok: mean CE on 2nd.. masked positions

Usage:
    python tools/audit_a3_ckpt_eval_cloud.py --ckpt <path> --tag <tag> [--n 128]
"""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


def load_blob(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def chain_session(blob, ci, j):
    s = int(blob["chain_starts"][ci])
    return blob["session_ids"][s + j].long()


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
def score_one(model, blob, ci, device):
    cb_pos = int(blob["chain_callback_position"][ci])
    if cb_pos <= 0 or cb_pos >= int(blob["chain_lengths"][ci]):
        return None
    sess = chain_session(blob, ci, cb_pos).to(device).unsqueeze(0)
    cb_mask = blob["session_callback_mask"][int(blob["chain_starts"][ci]) + cb_pos].bool()
    if int(cb_mask.sum().item()) == 0:
        return None
    M_c = build_Mc(model, blob, ci, cb_pos, device)
    out = model(input_ids=sess, M_c=M_c)
    logits = out.logits  # (1, S, V)
    target = sess[:, 1:]
    pred = logits[:, :-1, :]
    log_probs = F.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)[0]  # (S-1,)
    msk1 = cb_mask[1:].to(device)
    n = int(msk1.sum().item())
    if n == 0:
        return None
    per_tok_nll = nll[msk1].tolist()
    return {
        "ce_mem": float(sum(per_tok_nll) / n),
        "n_tok": n,
        "ce_first_item_tok": per_tok_nll[0],
        "ce_subsequent_item_tok": (float(sum(per_tok_nll[1:]) / (n - 1))
                                   if n > 1 else None),
        "per_tok": per_tok_nll,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out_dir", default=str(ROOT / "results/exp2_chain_recipe"))
    a = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    print(f"[{a.tag}] loading {a.ckpt} ...", flush=True)
    model = Qwen3MemResForCausalLM.from_pretrained(
        a.ckpt, dtype=torch.bfloat16
    ).to(device).eval()

    val = load_blob(ROOT / "paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt")
    n_avail = int(val["chain_starts"].shape[0])
    n = min(a.n, n_avail)
    print(f"[{a.tag}] scoring {n}/{n_avail} chains", flush=True)

    ce_mem_list, first_list, sub_list, tok_lens = [], [], [], []
    per_chain = []
    for ci in range(n):
        r = score_one(model, val, ci, device)
        if r is None:
            continue
        ce_mem_list.append(r["ce_mem"])
        first_list.append(r["ce_first_item_tok"])
        if r["ce_subsequent_item_tok"] is not None:
            sub_list.append(r["ce_subsequent_item_tok"])
        tok_lens.append(r["n_tok"])
        per_chain.append({
            "chain_idx": ci,
            "ce_mem": r["ce_mem"],
            "n_tok": r["n_tok"],
            "ce_first_item_tok": r["ce_first_item_tok"],
            "ce_subsequent_item_tok": r["ce_subsequent_item_tok"],
        })
        if (ci + 1) % 16 == 0:
            print(f"  [{a.tag}] {ci+1}/{n}  running mean pa_cb_ce_mem="
                  f"{statistics.mean(ce_mem_list):.4f}", flush=True)

    result = {
        "tag": a.tag,
        "ckpt": a.ckpt,
        "n_chains_scored": len(ce_mem_list),
        "n_chains_requested": n,
        "pa_cb_ce_mem": statistics.mean(ce_mem_list) if ce_mem_list else float("nan"),
        "ce_first_item_tok_mean": statistics.mean(first_list) if first_list else float("nan"),
        "ce_subsequent_item_tok_mean": statistics.mean(sub_list) if sub_list else float("nan"),
        "avg_item_tok_len": statistics.mean(tok_lens) if tok_lens else float("nan"),
        "log_32": math.log(32),
        "log_256": math.log(256),
        "per_chain": per_chain,
    }
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"audit_a3_{a.tag}_cloud.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{a.tag}] SAVED -> {out_path}")
    print(f"[{a.tag}] pa_cb_ce_mem = {result['pa_cb_ce_mem']:.4f} nats "
          f"over n={result['n_chains_scored']}")
    print(f"[{a.tag}] ce_first_item_tok = {result['ce_first_item_tok_mean']:.4f} nats")
    print(f"[{a.tag}] ce_subsequent_item_tok = "
          f"{result['ce_subsequent_item_tok_mean']:.4f} nats")

    # Free GPU
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
