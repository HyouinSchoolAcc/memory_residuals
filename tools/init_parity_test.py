"""
Init-parity diagnostic for Memory Residuals.

Verifies, on a fixed Qwen3 backbone, that attaching MemRes does (or does
not) leave the forward pass bit-equivalent to the bare backbone at step 0.

Six cases are measured:

    case                                                  | expected
    ------------------------------------------------------+----------
    memres_mode="simple_gate",       no memory injection  | parity (0 backbone perturbation)
    memres_mode="simple_gate",       memory injection on  | parity (gate g_l = 0)
    memres_mode="attention_base",    no memory            | broken (1/N uniform mix)
    memres_mode="attention_base",    memory on            | broken
    memres_mode="attention_parity",  no memory            | parity (one-hot recent + cumulative pool)
    memres_mode="attention_parity",  memory on            | parity (mem_bias -> -inf)

For each case we report the max absolute logit difference vs the bare
backbone on a fixed text prompt, and a pass/fail at a 1e-3 tolerance
(loose enough to absorb bf16 rounding from intermediate reductions
through the depth router).

Run:
    python tools/init_parity_test.py \\
        --pretrained Qwen/Qwen3-0.6B \\
        --out results/eval/init_parity_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modeling_memres import Qwen3MemResConfig, Qwen3MemResForCausalLM  # noqa: E402

PROMPT = (
    "The Memory Residuals architecture extends a pretrained Qwen3 "
    "backbone with a fixed-size recurrent memory matrix and an off-"
    "sequence depth-wise readout."
)
MEM_PROMPT = (
    "Earlier in this conversation, the user mentioned that they live "
    "in Reykjavik and work as a marine biologist studying narwhals."
)


def build_memres(
    pretrained: str,
    base_state_dict: dict[str, torch.Tensor],
    memres_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3MemResForCausalLM:
    """Build a fresh MemRes model on top of the pretrained Qwen3 weights.

    The pretrained Qwen3 weights are loaded into the backbone via
    `from_pretrained`; the memory-specific parameters (MemoryBlock,
    MemoryReadout, depth_router, memory_gate) are initialised by the
    custom `_init_memres_params` invoked during `post_init()`.
    """
    base_cfg = AutoConfig.from_pretrained(pretrained)
    cfg = Qwen3MemResConfig(
        **base_cfg.to_dict(),
        memres_num_vectors=128,
        memres_extraction_depth=0,
        memres_num_blocks=8,
        memres_mode=memres_mode,
    )
    model = Qwen3MemResForCausalLM.from_pretrained(
        pretrained, config=cfg, dtype=dtype
    ).to(device).eval()
    # Sanity: shared backbone weights should match the donor exactly.
    msd = model.state_dict()
    backbone_keys = [k for k in base_state_dict if k in msd]
    for k in backbone_keys[:8]:  # quick spot-check
        assert torch.equal(
            base_state_dict[k].to(msd[k].device).to(msd[k].dtype), msd[k]
        ), f"backbone weight drift on {k}"
    return model


@torch.no_grad()
def base_logits(
    pretrained: str, device: torch.device, dtype: torch.dtype, ids: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    base_cfg = AutoConfig.from_pretrained(pretrained)
    base = AutoModelForCausalLM.from_pretrained(
        pretrained, dtype=dtype
    ).to(device).eval()
    out = base(input_ids=ids).logits
    sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    del base
    torch.cuda.empty_cache()
    return out, sd


@torch.no_grad()
def memres_logits(
    model: Qwen3MemResForCausalLM,
    ids: torch.Tensor,
    history_ids: torch.Tensor | None,
) -> torch.Tensor:
    return model(input_ids=ids, history_ids=history_ids).logits


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rms": float(d.pow(2).mean().sqrt().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--out", default="results/eval/init_parity_test.json"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Max-abs-logit threshold at which a case is reported as PARITY.",
    )
    a = ap.parse_args()

    device = torch.device(a.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[a.dtype]

    tok = AutoTokenizer.from_pretrained(a.pretrained)
    enc = tok(PROMPT, return_tensors="pt").to(device)
    history = tok(MEM_PROMPT, return_tensors="pt").to(device)
    ids = enc.input_ids
    history_ids = history.input_ids

    print(f"[parity] backbone: {a.pretrained}  device={device}  dtype={dtype}")
    print(f"[parity] current  tokens: {ids.shape[1]}")
    print(f"[parity] history  tokens: {history_ids.shape[1]}")

    base_out, base_sd = base_logits(a.pretrained, device, dtype, ids)
    print(f"[parity] base logits: shape={tuple(base_out.shape)}  "
          f"dtype={base_out.dtype}")

    cases = [
        # (label, memres_mode, attach_memory, expect_parity)
        ("simple_gate_no_mem",          "simple_gate",      False, True),
        ("simple_gate_with_mem",        "simple_gate",      True,  True),
        ("attention_base_no_mem",       "attention_base",   False, False),
        ("attention_base_with_mem",     "attention_base",   True,  False),
        ("attention_parity_no_mem",     "attention_parity", False, True),
        ("attention_parity_with_mem",   "attention_parity", True,  True),
    ]

    results: list[dict] = []
    for label, mode, attach_mem, expect_parity in cases:
        torch.manual_seed(0)  # MemoryReadout.W_V is normal-init -- pin RNG.
        model = build_memres(a.pretrained, base_sd, mode, device, dtype)
        h = history_ids if attach_mem else None
        # Probe the router's per-step bias and gate inits (helps diagnose
        # why parity does or doesn't hold).
        router = model.model.depth_router
        gate = model.model.memory_gate
        diag = {
            "depth_router.mem_bias[0]": float(router.mem_bias[0].detach().item()),
            "depth_router.recent_bias[0]": float(
                router.recent_bias[0].detach().item()
            ),
            "depth_router.w[0].abs.max": float(router.w[0].detach().abs().max().item()),
            "memory_gate.gate[0]": float(gate.gate[0].detach().item()),
        }
        out = memres_logits(model, ids, h)
        stats = diff_stats(out, base_out)
        passed = stats["max_abs"] <= a.tolerance
        verdict = "PARITY" if passed else "PERTURBED"
        ok = "OK " if passed == expect_parity else "MISMATCH"
        print(
            f"[parity] {label:<32s}  mode={mode:<18s} mem={attach_mem}  "
            f"max|Δ|={stats['max_abs']:.3e}  "
            f"mean|Δ|={stats['mean_abs']:.3e}  -> {verdict:<9s} ({ok})"
        )
        results.append(
            {
                "label": label,
                "memres_mode": mode,
                "memory_attached": attach_mem,
                "expect_parity": expect_parity,
                "logit_diff_vs_base": stats,
                "init_diagnostics": diag,
                "verdict": verdict,
                "matches_expectation": passed == expect_parity,
            }
        )
        del model
        torch.cuda.empty_cache()

    summary = {
        "pretrained": a.pretrained,
        "dtype": a.dtype,
        "device": str(device),
        "current_tokens": int(ids.shape[1]),
        "history_tokens": int(history_ids.shape[1]),
        "tolerance": a.tolerance,
        "cases": results,
        "all_match_expectation": all(c["matches_expectation"] for c in results),
    }

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[parity] wrote {out_path}")
    print(
        f"[parity] all cases match expectation: "
        f"{summary['all_match_expectation']}"
    )


if __name__ == "__main__":
    main()
