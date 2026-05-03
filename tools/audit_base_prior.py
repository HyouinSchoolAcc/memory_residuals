#!/usr/bin/env python3
"""Audit: measure base-model CE on D4v2 callback-mask tokens with NO memory,
NO finetuning. This sets the floor for what the backbone can predict from
priors alone (category cue + pretraining knowledge of common words).

Run:
    python tools/audit_base_prior.py
"""
from __future__ import annotations
import sys, math, statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
device = "cuda:0"
torch.set_grad_enabled(False)

MODELS = [("Qwen/Qwen3-0.6B", "0.6B-base"), ("Qwen/Qwen3-1.7B", "1.7B-base")]
N_CHAINS = 64

va = torch.load(
    ROOT / "paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt",
    weights_only=False, map_location="cpu",
)
sess = va["session_ids"]
starts = va["chain_starts"].tolist()
cb_pos_all = va["chain_callback_position"].tolist()
cb_mask = va["session_callback_mask"]

results = {}
for ckpt, tag in MODELS:
    print(f"-- loading {tag} ({ckpt}) --", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nlls, multi_tok = [], 0
    item_lens = []
    for ci in range(min(N_CHAINS, len(starts))):
        st = starts[ci]; cb_pos = cb_pos_all[ci]
        cb_session = sess[st + cb_pos].to(device).unsqueeze(0)
        msk = cb_mask[st + cb_pos].to(device)
        item_lens.append(int(msk.sum().item()))
        if int(msk.sum().item()) > 1: multi_tok += 1
        ids = cb_session[:, :-1]
        out = m(ids)
        logits = out.logits
        labels = cb_session[:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1), reduction="none"
        ).reshape(1, -1)
        msk1 = msk[1:].float()
        n = msk1.sum().item()
        if n > 0: nlls.append((ce * msk1).sum().item() / n)
    results[tag] = (statistics.mean(nlls), len(nlls), multi_tok, statistics.mean(item_lens))
    print(f"  {tag} mean CE = {results[tag][0]:.4f} nats over n={len(nlls)} chains; "
          f"multi-token-items={multi_tok}/{len(nlls)}; avg_ans_toks={results[tag][3]:.2f}", flush=True)
    del m; torch.cuda.empty_cache()

print()
print("=== theoretical floors ===")
print(f"log(256) = {math.log(256):.4f}  (uniform over all items)")
print(f"log(32)  = {math.log(32):.4f}  (uniform per category cue)")
print()
print("=== base-model CEs (no memory, no FT) ===")
for tag, (ce, n, mt, al) in results.items():
    print(f"  {tag:>10}: {ce:.4f} nats")
