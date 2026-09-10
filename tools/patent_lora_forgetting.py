#!/usr/bin/env python3
"""Patent Experiment 3 (baseline side) — per-user LoRA fine-tuning.

Simulates the "user-level fine-tuning" prior art: for each chain (= one
user's conversation history), a LoRA adapter on the bare Qwen3 backbone is
fine-tuned *sequentially*, session by session, with the LM loss on that
session's tokens. After every session we measure:

  callback CE : CE on the callback-session's callback tokens (does the
                fine-tuned model recall the user's information?)
  general CE  : CE on a fixed, held-out set of generic dialogue sessions
                from a DIFFERENT corpus (MSC test) — drift here is
                catastrophic forgetting of general ability.

Also reported: adapter storage size per user (bytes).

The MemRes numbers to compare against come from
tools/patent_retention_curve.py (callback CE) and, by construction,
general CE drift is exactly zero for MemRes because the backbone is
frozen and unmodified (M_c only conditions the forward pass; with
M_c = None the forward is bit-identical to bare Qwen3).

Usage:
    python tools/patent_lora_forgetting.py \
        --n_chains 8 --steps_per_session 8 \
        --output results/patent_experiments/lora_forgetting_0p6b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM


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
def masked_ce(model, input_ids, mask=None):
    out = model(input_ids=input_ids)
    target = input_ids[:, 1:]
    pred = out.logits[:, :-1, :]
    log_probs = torch.nn.functional.log_softmax(pred.float(), dim=-1)
    nll = -log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)[0]
    if mask is not None:
        m = mask[1:].to(input_ids.device)
        if m.sum() == 0:
            return float("nan")
        return float(nll[m].mean().item())
    return float(nll.mean().item())


@torch.no_grad()
def general_ce(model, heldout_ids):
    return mean([masked_ce(model, s) for s in heldout_ids])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-0.6B")
    p.add_argument("--corpus",
                   default="paper_artifacts/chains/lme_val_s512_evpos.pt")
    p.add_argument("--general_corpus",
                   default="paper_artifacts/chains/msc_test_s512.pt")
    p.add_argument("--general_source", choices=["corpus", "fineweb"],
                   default="corpus",
                   help="'fineweb' streams out-of-domain English prose "
                        "(fineweb-edu) instead of the dialogue corpus, so "
                        "the general-ability probe is disjoint from the "
                        "fine-tuning domain.")
    p.add_argument("--n_chains", type=int, default=8)
    p.add_argument("--n_general", type=int, default=16)
    p.add_argument("--steps_per_session", type=int, default=8)
    p.add_argument("--mode", choices=["sequential", "joint"],
                   default="sequential",
                   help="'sequential' fine-tunes session-by-session in "
                        "chain order (the continual-update setting). "
                        "'joint' spends the same total step budget sampling "
                        "sessions uniformly at random from the whole "
                        "history (the strongest offline per-user FT "
                        "variant, free of ordering effects).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--measure_every", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    device = torch.device(a.device)
    blob = torch.load(a.corpus, map_location="cpu", weights_only=False)
    if a.general_source == "fineweb":
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.base)
        stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)
        heldout = []
        for doc in stream:
            ids = tok(doc["text"], return_tensors="pt").input_ids[0]
            if ids.numel() >= 512:
                heldout.append(ids[:512].unsqueeze(0).to(device))
            if len(heldout) >= a.n_general:
                break
    else:
        gblob = torch.load(a.general_corpus, map_location="cpu",
                           weights_only=False)
        heldout = [gblob["session_ids"][i].long().unsqueeze(0).to(device)
                   for i in range(a.n_general)]

    lora_cfg = LoraConfig(
        r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    per_chain = []
    for ci in range(a.n_chains):
        cb_pos = int(blob["chain_callback_position"][ci])
        if cb_pos <= 0:
            continue
        cb_sess = chain_session(blob, ci, cb_pos).to(device).unsqueeze(0)
        cb_mask = chain_session_mask(blob, ci, cb_pos)

        base = AutoModelForCausalLM.from_pretrained(
            a.base, dtype=torch.bfloat16).to(device)
        model = get_peft_model(base, lora_cfg)
        model.train()
        adapter_bytes = sum(
            p.numel() * p.element_size()
            for n, p in model.named_parameters() if "lora" in n)
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=a.lr)

        model.eval()
        trace = [{
            "t": 0,
            "callback_ce": masked_ce(model, cb_sess, cb_mask),
            "general_ce": general_ce(model, heldout),
        }]
        model.train()

        if a.mode == "sequential":
            for t in tqdm(range(cb_pos), desc=f"chain {ci} LoRA seq"):
                sess = chain_session(blob, ci, t).to(device).unsqueeze(0)
                for _ in range(a.steps_per_session):
                    out = model(input_ids=sess, labels=sess)
                    out.loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if (t + 1) % a.measure_every == 0 or t == cb_pos - 1:
                    model.eval()
                    trace.append({
                        "t": t + 1,
                        "callback_ce": masked_ce(model, cb_sess, cb_mask),
                        "general_ce": general_ce(model, heldout),
                    })
                    model.train()
        else:  # joint: same step budget, uniform random session order
            g = torch.Generator().manual_seed(ci)
            total_steps = a.steps_per_session * cb_pos
            measure_at = {int(total_steps * f) for f in (0.25, 0.5, 0.75, 1.0)}
            for step in tqdm(range(1, total_steps + 1),
                             desc=f"chain {ci} LoRA joint"):
                t = int(torch.randint(0, cb_pos, (1,), generator=g).item())
                sess = chain_session(blob, ci, t).to(device).unsqueeze(0)
                out = model(input_ids=sess, labels=sess)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step in measure_at:
                    model.eval()
                    trace.append({
                        "t": step / a.steps_per_session,
                        "callback_ce": masked_ce(model, cb_sess, cb_mask),
                        "general_ce": general_ce(model, heldout),
                    })
                    model.train()

        try:
            ev = sorted(int(x) for x in
                        blob["chain_evidence_positions"][ci] if int(x) < cb_pos)
        except (KeyError, TypeError):
            ev = []
        per_chain.append({
            "chain_id": (blob["chain_names"][ci]
                         if "chain_names" in blob else f"c{ci}"),
            "cb_pos": cb_pos,
            "evidence_positions": ev,
            "adapter_bytes": adapter_bytes,
            "trace": trace,
        })
        print(json.dumps(per_chain[-1]["trace"][-1]))

        del model, base, opt
        torch.cuda.empty_cache()

    final_cb = [r["trace"][-1]["callback_ce"] for r in per_chain]
    init_cb = [r["trace"][0]["callback_ce"] for r in per_chain]
    final_gen = [r["trace"][-1]["general_ce"] for r in per_chain]
    init_gen = [r["trace"][0]["general_ce"] for r in per_chain]
    summary = {
        "base": a.base,
        "mode": a.mode,
        "lr": a.lr,
        "n_chains": len(per_chain),
        "steps_per_session": a.steps_per_session,
        "lora_r": a.lora_r,
        "adapter_bytes": per_chain[0]["adapter_bytes"] if per_chain else None,
        "mean_callback_ce_init": mean(init_cb),
        "mean_callback_ce_final": mean(final_cb),
        "mean_callback_gain": mean(init_cb) - mean(final_cb),
        "mean_general_ce_init": mean(init_gen),
        "mean_general_ce_final": mean(final_gen),
        "mean_general_drift": mean(final_gen) - mean(init_gen),
        "per_chain": per_chain,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chain"},
                     indent=2))
    print(f"Saved -> {a.output}")


if __name__ == "__main__":
    main()
