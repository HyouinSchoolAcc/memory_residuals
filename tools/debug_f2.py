#!/usr/bin/env python3
"""Debug F2 probe NaN.

Reproduces the v17 probe-NaN signature in a minimal harness:
  load qwen3-0.6b-large with writer_probe_enabled, build M_c from a
  single evidence session, run probe(M_c), report finiteness at every
  step.  Run with `--steps 5` to isolate which iteration is the
  source of the NaN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_chain import ChainCorpus  # noqa: E402
from modeling_memres import (  # noqa: E402
    Qwen3MemResConfig,
    Qwen3MemResForCausalLM,
)


def main() -> None:
    device = "cuda:0"
    torch.manual_seed(42)

    print(f"loading config + 0.6b model...", flush=True)
    base_cfg = Qwen3MemResConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg_dict = dict(base_cfg.to_dict())
    cfg_dict.update({
        "memres_num_vectors": 128,
        "memres_extraction_depth": 0,
        "memres_num_blocks": 8,
        "memres_mode": "attention_parity",
        "router_recent_bias_init": 4.0,
        "router_mem_bias_init": 0.0,
        "memres_update_mode": "gated",
        "memres_extract_source": "hidden_14",
        "memres_extract_input_norm": True,
        "memres_gate_init": 0.0,
        "memres_readout_norm_init": 0.05,
        "memres_writer_kind": "slot_attention",
        "memres_slot_attention_iters": 3,
        "memres_queries_init": "orthogonal",
        "memres_slot_positional": True,
        "memres_judge_qk_layernorm": True,
        "writer_probe_enabled": True,
        "writer_probe_n_queries": 1,
    })
    cfg = Qwen3MemResConfig(**cfg_dict)
    model = Qwen3MemResForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", config=cfg, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print("model loaded; freezing backbone, training writer + probe.")

    # Freeze backbone
    bb_params = [p for n, p in model.named_parameters()
                 if "memory_block" not in n and "memory_readout" not in n
                 and "depth_router" not in n and "memory_gate" not in n
                 and "writer_probe_head" not in n]
    for p in bb_params:
        p.requires_grad_(False)

    train_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in train_params)/1e6:.1f}M",
          flush=True)
    opt = torch.optim.AdamW(train_params, lr=1e-4, weight_decay=0.0)

    print("loading synthd5 train corpus...", flush=True)
    train_corpus = ChainCorpus(
        ROOT / "paper_artifacts/chains/synthd5_random_codes_train_s512.pt"
    )
    n_chains = int(train_corpus.chain_starts.shape[0])
    print(f"  {n_chains} chains", flush=True)

    K = cfg.memres_num_vectors
    d = cfg.hidden_size
    BSZ = 4

    def check(name, t):
        ok = bool(torch.isfinite(t).all().item())
        nm = float(t.float().abs().max().item())
        nn_ = int((~torch.isfinite(t)).sum().item())
        print(f"    {name}: shape={tuple(t.shape)} max|.|={nm:.3e} "
              f"finite={ok} nonfin={nn_}")
        return ok

    # Inspect probe head weights once
    ph = model.writer_probe_head
    print(f"\n=== probe head init ===")
    print(f"  query        : {tuple(ph.query.shape)} dtype={ph.query.dtype} max|.|={ph.query.float().abs().max():.3e}")
    print(f"  W_K.weight   : {tuple(ph.W_K.weight.shape)} dtype={ph.W_K.weight.dtype} max|.|={ph.W_K.weight.float().abs().max():.3e}")
    print(f"  W_V.weight   : {tuple(ph.W_V.weight.shape)} dtype={ph.W_V.weight.dtype} max|.|={ph.W_V.weight.float().abs().max():.3e}")
    print(f"  norm.weight  : {tuple(ph.norm.weight.shape)} dtype={ph.norm.weight.dtype} max|.|={ph.norm.weight.float().abs().max():.3e}")
    print(f"  proj.weight  : {tuple(ph.proj.weight.shape)} dtype={ph.proj.weight.dtype} max|.|={ph.proj.weight.float().abs().max():.3e}")
    print(f"  proj.weight  finite={bool(torch.isfinite(ph.proj.weight).all())}")
    print(f"  proj.weight  nonfin={int((~torch.isfinite(ph.proj.weight)).sum())}")
    print(f"  proj.weight  zeros={int((ph.proj.weight == 0).sum())}")
    print(f"  query        : {tuple(ph.query.shape)} dtype={ph.query.dtype} "
          f"finite={bool(torch.isfinite(ph.query).all())} "
          f"nonfin={int((~torch.isfinite(ph.query)).sum())} "
          f"max|.|={float(ph.query.float().abs().max()):.3e}")
    print(f"  W_K.weight   : {tuple(ph.W_K.weight.shape)} dtype={ph.W_K.weight.dtype} "
          f"finite={bool(torch.isfinite(ph.W_K.weight).all())} "
          f"nonfin={int((~torch.isfinite(ph.W_K.weight)).sum())} "
          f"max|.|={float(ph.W_K.weight.float().abs().max()):.3e}")
    print(f"  W_V.weight   : {tuple(ph.W_V.weight.shape)} dtype={ph.W_V.weight.dtype} "
          f"finite={bool(torch.isfinite(ph.W_V.weight).all())} "
          f"nonfin={int((~torch.isfinite(ph.W_V.weight)).sum())} "
          f"max|.|={float(ph.W_V.weight.float().abs().max()):.3e}")

    for step in range(1, 4):
        print(f"\n=== step {step} ===", flush=True)
        # Sample one chain, evidence session 0, callback session.
        ev_ids_list = []
        cb_ids_list = []
        cb_mask_list = []
        for _ in range(BSZ):
            ci = torch.randint(0, n_chains, (1,)).item()
            ev_pos = train_corpus.chain_evidence_positions[ci][0]
            cb_pos = int(train_corpus.chain_callback_position[ci].item())
            st = int(train_corpus.chain_starts[ci].item())
            ev_ids = train_corpus.session_ids[st + ev_pos]
            cb_ids = train_corpus.session_ids[st + cb_pos]
            cb_mask = train_corpus.session_callback_mask[st + cb_pos]
            ev_ids_list.append(ev_ids)
            cb_ids_list.append(cb_ids)
            cb_mask_list.append(cb_mask)
        ev_t = torch.stack(ev_ids_list).to(device)
        cb_t = torch.stack(cb_ids_list).to(device)
        cb_mask_t = torch.stack(cb_mask_list).to(device)
        # First answer-token id per row
        first_ids = []
        for b in range(BSZ):
            mask = cb_mask_t[b].bool()
            idx = mask.nonzero(as_tuple=False).flatten()[0].item()
            tid = cb_t[b, idx].item()
            first_ids.append(tid)
        target_t = torch.tensor(first_ids, dtype=torch.long, device=device)

        # Build M_c from evidence session.
        with torch.enable_grad():
            C_t = model.model.extract_source(ev_t[:, :-1])
            M_c = model.model.compress_session(C_t, None)
            check("M_c", M_c)
            # Trace probe internals manually.
            ph = model.writer_probe_head
            K_proj = ph.W_K(M_c); check("K_proj", K_proj)
            V_proj = ph.W_V(M_c); check("V_proj", V_proj)
            q = ph.query.to(M_c.dtype).unsqueeze(0).expand(M_c.shape[0], -1, -1)
            scores = torch.matmul(q, K_proj.transpose(-2, -1)) * ph.scale
            check("scores", scores)
            attn = F.softmax(scores.float(), dim=-1).to(M_c.dtype)
            check("attn", attn)
            v0 = torch.matmul(attn, V_proj); check("attn@V", v0)
            v1 = v0.mean(dim=1); check("v_mean", v1)
            v2 = ph.norm(v1); check("norm(v)", v2)
            v3 = ph.proj(v2.to(ph.proj.weight.dtype))
            check("proj(v)", v3)
            probe_logits = v3.float()
            check("probe_logits", probe_logits)
            loss_probe = F.cross_entropy(probe_logits, target_t)
            print(f"    loss_probe = {float(loss_probe):.4f}",
                  flush=True)

            # Backward + step.
            loss_probe.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            print(f"    grad_norm = {float(grad_norm):.3f}", flush=True)
            opt.step()
            opt.zero_grad()


if __name__ == "__main__":
    main()
