#!/usr/bin/env python3
"""Recurrent chain trainer for Memory Residuals (Phase 1, the real one).

This is the trainer that actually exercises the recurrent state $M_c$.  Each
training step:

    1.  Sample a chain (a PG-19 book or a TV show) from the pre-tokenised
        chain corpus produced by ``tools/pretokenize_chains.py``.
    2.  Sample a window of $k$ consecutive sessions inside that chain.
    3.  Initialise $M_c \\leftarrow 0$ (or carry the detached $M_c$ from the
        previous window if --carry_state is set).
    4.  Iterate the window left-to-right.  At session $t$:
            (a) Forward with the *current* $M_c$, compute next-token NLL on
                session $t$.  This is the loss the optimizer actually sees.
            (b) Update memory: $M_c \\leftarrow$ judge($M_c$, extract($C_t$)).
                The judge step compares the *previous* memory against the new
                candidate so old and new compete for slots, exactly per the
                paper.
    5.  Backprop $\\sum_t L_t / k$ through all $k$ forward passes and all $k$
        recurrent judge steps.

This is *not* the ``train_phase1.py`` setup, where the four-session history
was concatenated into a single text field and compressed once per step.
That setup is a useful warm-up because it teaches the readout/extraction
parameters to be useful at all, but it does not train the recurrent
competition.  This trainer does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoTokenizer

from modeling_memres import (
    Qwen3MemResConfig,
    Qwen3MemResForCausalLM,
    _normalise_memres_mode,
)
from presets import PRESETS, apply_preset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument(
        "--memres_mode",
        # Canonical names: simple_gate / attention_base / attention_parity.
        # Legacy names "residual" / "block_attnres" are accepted and folded
        # into the canonical set after parsing (see _normalise_memres_args)
        # so old shell scripts and saved checkpoints keep working.
        choices=(
            "simple_gate", "attention_base", "attention_parity",
            "residual", "block_attnres",
        ),
        default="attention_parity",
        help="simple_gate = ReZero-style scalar-gate injection (toy/baseline); "
             "attention_base = full Block AttnRes routing pool, delta sources, "
             "no init parity; "
             "attention_parity = full Block AttnRes pool, cumulative sources, "
             "step-0 logits bit-identical to the bare backbone (default). "
             "Legacy: 'residual' -> simple_gate, "
             "'block_attnres' -> attention_parity (or attention_base when "
             "paired with --no-block_attnres_parity_init).",
    )
    p.add_argument(
        "--block_attnres_parity_init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="DEPRECATED -- use --memres_mode attention_base / "
        "attention_parity directly. Only honoured when --memres_mode is the "
        "legacy 'block_attnres' string; ignored otherwise.",
    )
    p.add_argument("--init_from", default=None,
                   help="Optional: warm-start from a Phase-0 (pair-trained) checkpoint")

    # Router init-bias overrides.  Default (None / None) -> mode-derived
    # defaults: attention_parity uses (-32, +32) for bit-exact init parity
    # with a saturated softmax; attention_base / simple_gate uses (-8, 0).
    # Setting these to less extreme magnitudes (e.g. -4 / +4) softens the
    # parity init: step-0 logits drift slightly from bare Qwen3, but the
    # softmax has non-trivial gradient on the pseudo-queries from step 0
    # and the router can actually learn to recruit memory.
    p.add_argument(
        "--router_mem_bias_init", type=float, default=None,
        help="Override BlockAttnResRouter.mem_bias init value. "
             "Default: -32 (attention_parity) / -8 (other modes).",
    )
    p.add_argument(
        "--router_recent_bias_init", type=float, default=None,
        help="Override BlockAttnResRouter.recent_bias init value. "
             "Default: +32 (attention_parity) / 0 (other modes).",
    )

    # MemRes hyper-params (overridden by --preset)
    p.add_argument("--memres_num_vectors", type=int, default=128)
    p.add_argument("--memres_extraction_depth", type=int, default=0)
    p.add_argument("--memres_num_blocks", type=int, default=8)
    p.add_argument(
        "--memres_extract_source", default="embed",
        help="Source for the per-token representation C_t fed to the "
             "extraction stack. 'embed' = bag-of-token-embeddings (legacy "
             "default); 'hidden_<L>' = bare-backbone hidden state at "
             "layer L (no_grad, detached) -- contextualised representation "
             "that carries syntax/anaphora/entity binding into M_in's "
             "cross-attention. 'hidden_14' is the recommended default for "
             "a 28-layer Qwen3 backbone (mid-stack, semantic).",
    )
    p.add_argument(
        "--memres_update_mode", default="competitive",
        choices=["competitive", "gated"],
        help="MemoryBlock recurrent update mode. 'competitive' (default, "
             "v3-v5) uses the zero-sum softmax over [M_c^{t-1} || M_new] "
             "where each slot is fully replaced or fully kept. 'gated' (v6+) "
             "adds a per-slot sigmoid write gate so the model can KEEP an "
             "existing slot rather than overwrite it -- useful for long "
             "horizons where a fact written at session 1 must survive "
             "through sessions 2..N without being clobbered by intervening "
             "session content. Init bias -1.0 -> g ~ 0.27 (modest writes).",
    )
    # v11 bootstrap-fix knobs (README "Stop everything" P1 + P2).
    p.add_argument(
        "--memres_gate_init", type=float, default=0.0,
        help="Initial value for ``MemoryGate.gate`` (per-sublayer scalar "
             "gate used in simple_gate routing). Default 0.0 (historical: "
             "augmented model behaviourally identical to bare backbone at "
             "step 0). v11 fix: a small positive value (e.g. 0.005) gives "
             "the readout nonzero forward influence at step 0, so the LM "
             "gradient can flow back to W_V^read / writer / judge from "
             "step 1 -- breaks the gate/readout/writer chicken-and-egg.",
    )
    p.add_argument(
        "--memres_readout_norm_init", type=float, default=1.0,
        help="Initial value for ``MemoryReadout.out_norm.weight`` (the "
             "RMSNorm scale on the readout output m^t). Default 1.0 "
             "produces ||m^t||/||embed|| ~ 73 at d=1024, which forces the "
             "gate to operate in [0, ~0.014] -- too narrow for AdamW's "
             "natural step size. v11 fix: 0.05 produces "
             "||m^t||/||embed|| ~ 3.6 and gate operating range [0, ~0.3], "
             "well within AdamW step. The parameter remains learnable, "
             "so the model can still grow the readout magnitude during "
             "training as needed.",
    )
    # v12 (2026-05-01) writer-subsystem architectural switch.
    p.add_argument(
        "--memres_writer_kind", default="original",
        choices=["original", "slot_attention", "slot_attention_full"],
        help="Implementation of the writer subsystem (MemoryBlock). "
             "'original' (default, v1-v11) uses Eq. 1-2/5 with a "
             "single-pass cross-attention judge whose softmax is over "
             "the *inputs* axis -- D2 audit on v11g showed this "
             "degenerates to row_entropy/log(2K) = 0.999 (uniform) and "
             "effective rank 1.02 across 4000 steps, i.e. the judge "
             "never actually competes. "
             "'slot_attention' (v12a) replaces the Stage-2 judge with "
             "an iterative Slot Attention writer (Locatello et al. 2020): "
             "softmax over SLOTS, weighted-mean renormalisation, GRU "
             "keep/write per slot. The slot-axis softmax forces slots "
             "to specialise on disjoint pieces of the input -- the "
             "PDF's 'zero-sum competition' realised structurally. "
             "'slot_attention_full' (v12b) additionally replaces the "
             "Stage-1 extraction stack with the same primitive over "
             "raw session tokens C_t.",
    )
    p.add_argument(
        "--memres_slot_attention_iters", type=int, default=3,
        help="Number of iterations inside the Stage-2 SlotAttentionWriter "
             "(when --memres_writer_kind != original). Canonical Slot "
             "Attention default is 3. Stage 1 (when "
             "writer_kind=slot_attention_full) uses 1 + "
             "--memres_extraction_depth iterations to subsume the L_E "
             "Perceiver-refinement layer count.",
    )

    # Callback supervision (v6+ long-horizon recipe)
    p.add_argument(
        "--callback_loss_weight", type=float, default=0.0,
        help="When > 0, tokens flagged in the corpus's session_callback_mask "
             "get an additional NLL weight of (1 + callback_loss_weight). "
             "The mask is loaded from the .pt file as 'session_callback_mask' "
             "(shape == session_ids.shape, int8 0/1). Tokens with mask=1 "
             "are typically the answer span of a synthesised or curated "
             "callback question (e.g. LongMemEval), so this concentrates "
             "gradient where memory retrieval actually matters. Set to 0 "
             "(default) to recover the legacy uniform NLL.",
    )
    p.add_argument(
        "--callback_window_bias", type=float, default=0.0,
        help="When > 0, with this probability sample a chain window aligned "
             "so that the callback session is INSIDE the window (and is "
             "the last session in the window). 0 = uniform sampling "
             "(default; the callback is rarely seen on long chains "
             "because it sits at position L-1 and only one window covers "
             "it). 1.0 = always include the callback. 0.5-0.9 is a "
             "reasonable training mix.",
    )
    p.add_argument(
        "--curriculum_competition_bias", type=float, default=0.0,
        help="When > 0, with this probability build a JUDGE-COMPETITION "
             "window: a 3-session window structured as either "
             "[evidence, distractor, callback] (50%%, 'keep-prev' sample: "
             "M_c after evidence is the relevant memory; the judge step at "
             "the distractor must keep prev memory and reject new content) "
             "OR [noise, evidence, callback] (50%%, 'write-new' sample: M_c "
             "after noise is irrelevant; the judge step at evidence must "
             "write the new content and discard prev memory). Both samples "
             "score the same callback session, so the gradient signal "
             "directly trains the write_gate / judge to be content-aware "
             "about when to KEEP vs WRITE. This isolates the judge sub-"
             "problem from the writer / readout sub-problem (which is what "
             "--curriculum_evidence_bias trains; the P0 evidence+callback "
             "window has M_c=0 at the judge step, so the judge degenerates "
             "to a no-competition aggregate). Requires --window_k >= 3. "
             "On chains with cb_pos < 3 falls through to subsequent "
             "branches. Composes with --curriculum_evidence_bias and "
             "--callback_window_bias: competition is tried first, then "
             "evidence-callback curriculum, then callback alignment, then "
             "uniform. See results/exp2_chain_recipe/runs.md "
             "v9 section for the curriculum-decomposition rationale.",
    )
    p.add_argument(
        "--curriculum_evidence_bias", type=float, default=0.0,
        help="When > 0, with this probability build a CURRICULUM window of "
             "the form [evidence, ...intermediates, callback] instead of a "
             "contiguous slice. Concretely: pick a random evidence position "
             "i in [0, callback_pos), pick (window_k - 2) random intermediate "
             "positions strictly between i and callback_pos, and stack them "
             "in chronological order ending in the callback session. This "
             "shortens the credit-assignment path between an early-session "
             "fact and the callback that depends on it -- the missing piece "
             "for the chain trainer per the v3-v6 routing-collapse diagnosis. "
             "The user controls difficulty by setting --window_k: window_k=2 "
             "is pure evidence+callback (P0), window_k=3 adds 1 distractor "
             "(P1), window_k=5 adds 3 (P2), window_k=8 is full-distractor (P3, "
             "equivalent to --callback_window_bias). Falls back to contiguous "
             "sampling on chains where callback_pos < window_k - 1. Composes "
             "with --callback_window_bias: curriculum is tried first, then "
             "callback alignment, then uniform.",
    )

    # Data. Defaults point at the active v6 LongMemEval-S corpora.
    # Pre-v6 corpora (stage1_*, msc_*, tv_*) live in archive/datasets/ if
    # you need to reproduce historical runs.
    p.add_argument("--train_chains",
                   default="paper_artifacts/chains/lme_train_s512.pt")
    p.add_argument("--eval_chains",
                   default="paper_artifacts/chains/lme_val_s512.pt")
    p.add_argument("--session_len", type=int, default=512,
                   help="Must match the .pt file's session_len.")
    p.add_argument("--window_k", type=int, default=4,
                   help="TBPTT window size: # sessions backpropped through "
                        "the recurrent judge stack per step.")
    p.add_argument("--source_weights",
                   default=None,
                   help="Optional JSON mapping source -> upweight, e.g. "
                        "'{\"pg19\":1.0,\"tv\":4.0,\"msc\":3.0}'.  "
                        "Source is detected from chain_name prefix: "
                        "'msc_' -> 'msc', leading digit -> 'pg19', "
                        "anything else -> 'tv'.  Default weights are "
                        "{pg19:1.0, tv:4.0, msc:3.0}.")

    # Optimization
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Number of independent chains processed in parallel "
                        "per step (NOT effective batch).")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_backbone", type=float, default=3e-6)
    p.add_argument("--lr_min_ratio", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)

    # Regularisers
    p.add_argument("--memory_dropout", type=float, default=0.10,
                   help="Per-session probability of zeroing M_c on the "
                        "*read* path (the judge update still runs).")
    p.add_argument("--context_dropout", type=float, default=0.30,
                   help="Per-session probability of masking the prefix of "
                        "the current session, forcing memory reliance.")
    p.add_argument("--carry_state", action="store_true",
                   help="If set, persist M_c across windows (detached). "
                        "Otherwise each window starts from zero.")
    p.add_argument("--neg_chain_weight", type=float, default=0.0,
                   help="Per-step weight on the negative-chain replay "
                        "auxiliary loss.  At each step we draw a *different* "
                        "chain's first session, build a shuffle M_c from it, "
                        "score the matched chain's last TBPTT session under "
                        "that shuffle memory, and add  alpha * max(0, "
                        "L_match - L_shuffle + margin)  to the total loss. "
                        "This is the PITFALLS.md §3 prescription against "
                        "style-only / shortcut-learning collapse.  0 disables. "
                        "When --neg_chain_warmup_steps > 0 this is the FINAL "
                        "weight, ramped linearly from --neg_chain_initial_weight."
                   )
    p.add_argument("--neg_chain_initial_weight", type=float, default=None,
                   help="Initial value of neg_chain_weight at step 0; ramps "
                        "linearly to --neg_chain_weight over "
                        "--neg_chain_warmup_steps. Default: equal to "
                        "--neg_chain_weight (no ramp).")
    p.add_argument("--neg_chain_warmup_steps", type=int, default=0,
                   help="Number of steps over which to linearly ramp "
                        "neg_chain_weight from initial -> final. The "
                        "COMPREHENSIVE.md §4.1 Phase B curriculum uses "
                        "0.05 -> 0.2 -> 0.5 in three phases; in this "
                        "trainer we fold that into a single linear ramp.")
    p.add_argument("--neg_chain_margin", type=float, default=0.05,
                   help="Margin for the negative-chain auxiliary loss.")

    # Intra-chain perturbation contrastive loss.  A *strictly stronger*
    # version of the inter-chain neg_chain_weight loss:
    #
    # - neg_chain_weight: contrasts same-chain last session under
    #   different-chain M_c.  Catches "memory learned style" but not
    #   "memory drops salient facts under interference from sessions
    #   between fact and recall".
    # - in_chain_contrast_weight: contrasts same-chain last session
    #   under same-chain M_c with one earlier session swapped for a
    #   random distractor.  The contrast concentrates on fact-recall
    #   tokens because most of the last-session NLL is invariant to
    #   the swap.  Gradient flows through the judge step at every
    #   intermediate session, putting direct pressure on the two-stage
    #   QKV competition to *preserve* fact-bearing channels even while
    #   sessions in between are pushing irrelevant updates.
    p.add_argument("--in_chain_contrast_weight", type=float, default=0.0,
                   help="Weight on the intra-chain perturbation "
                        "contrastive loss.  0 disables.  See block "
                        "comment in the trainer for the full spec.")
    p.add_argument("--in_chain_contrast_initial_weight", type=float, default=None,
                   help="Initial value of in_chain_contrast_weight at "
                        "step 0; ramps linearly to "
                        "--in_chain_contrast_weight over "
                        "--in_chain_contrast_warmup_steps. "
                        "Default: equal to --in_chain_contrast_weight (no ramp).")
    p.add_argument("--in_chain_contrast_warmup_steps", type=int, default=0,
                   help="Linear warmup for in_chain_contrast_weight.")
    p.add_argument("--in_chain_contrast_margin", type=float, default=0.05,
                   help="Margin for the in-chain contrastive loss.")
    p.add_argument("--in_chain_perturb_strategy",
                   choices=("session_zero", "random_earlier"),
                   default="random_earlier",
                   help="Which session in the window to swap for the "
                        "perturbed M_c. session_zero: always perturb "
                        "session 0 (the persona prefix in MSC). "
                        "random_earlier (default): perturb a random "
                        "session in [0, window_k-2] each step, so "
                        "every position gets pressured over training.")

    # Multi-negative InfoNCE contrastive loss.  A *batched* upgrade of
    # the existing single-negative --neg_chain_weight hinge:
    #
    # - --neg_chain_weight: hinge `max(0, L_match - L_shuffle + margin)`
    #   with ONE random out-of-batch chain as negative.  Hard-zero
    #   gradient when the gap exceeds margin.
    # - --contrastive_infonce_weight: for each batch element i, score
    #   last session i under EVERY batch element j's M_c (B*B forward),
    #   then cross-entropy with diagonal=positive.  Smooth gradient,
    #   B-1 in-batch negatives per positive, gradient flows through
    #   M_c[j] for all j (so off-diagonal pressure pushes M_c[j] AWAY
    #   from chain i's content -- direct attack on Δ_sh-m ≈ 0).
    p.add_argument("--contrastive_infonce_weight", type=float, default=0.0,
                   help="Weight on multi-negative InfoNCE contrastive "
                        "loss.  For each batch element i, scores last "
                        "session i under all B in-batch M_c's "
                        "(diagonal=positive, off-diagonal=negative) and "
                        "minimises cross-entropy.  Cost is +B*B forwards "
                        "on the last session only (the rest of the TBPTT "
                        "chain is reused).  At B=4 ~ +30-100% per-step "
                        "overhead.  0 disables.")
    p.add_argument("--contrastive_infonce_temperature", type=float, default=1.0,
                   help="Temperature for the InfoNCE softmax.  Larger = "
                        "softer (weaker gradient on small NLL gaps); "
                        "smaller = sharper.  NLL values are typically in "
                        "[1, 4] nats so T=1.0 gives reasonable softmax "
                        "dynamics; lower T (e.g. 0.5) when the gap is "
                        "small and we want the contrast to bite harder.")
    p.add_argument("--contrastive_infonce_initial_weight", type=float, default=None,
                   help="Initial value of contrastive_infonce_weight at "
                        "step 0; ramps linearly to "
                        "--contrastive_infonce_weight over "
                        "--contrastive_infonce_warmup_steps.  Default: "
                        "equal to --contrastive_infonce_weight (no ramp).")
    p.add_argument("--contrastive_infonce_warmup_steps", type=int, default=0,
                   help="Linear warmup for contrastive_infonce_weight, "
                        "ramping from initial -> final over this many "
                        "steps.  Lets the LM-loss alone shape M_c for "
                        "the first N steps before the contrastive "
                        "objective starts pulling on the readout.")
    p.add_argument("--contrastive_infonce_callback_only",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="If true (default), score per-pair NLL only on "
                        "callback-supervision tokens of the last "
                        "session, falling back to all valid tokens "
                        "when no callback span is present.  Concentrates "
                        "the contrastive signal on the tokens that "
                        "actually require memory.  Use "
                        "--no-contrastive_infonce_callback_only to score "
                        "all valid tokens.")

    # ------------------------------------------------------------------
    # Diagnostic toolkit (D1 / D2 / D3 from the architecture audit
    # 2026-05-01).  All three are *lightweight* ablations that surface
    # mechanism-level signal about whether the writer / judge / readout
    # are doing their nominal jobs.  Off by default; CLI-flagged on for
    # diagnostic runs.
    # ------------------------------------------------------------------
    p.add_argument("--diagnose_grad_groups",
                   action="store_true", default=False,
                   help="D1: log per-module gradient L2 norms each "
                        "log_every step (extract / M_in / judge / "
                        "M_judge / readout / router / backbone).  "
                        "If a memory-subsystem group's norm is "
                        "<<1%% of the backbone's, gradient is starving "
                        "that module and the architecture cannot "
                        "learn it from the LM loss alone -- which is "
                        "exactly the failure mode we are auditing.")
    p.add_argument("--diagnose_memory_dynamics",
                   action="store_true", default=False,
                   help="D2/D3: at every eval, run a no-grad "
                        "diagnostic pass over a few held-out chains "
                        "and log (a) per-row entropy of the judge "
                        "softmax (D2 -- decision-less judge ~ flat), "
                        "(b) keep/write mass split (D2), "
                        "(c) effective rank of M_c (D3), "
                        "(d) per-step ||M_c^t - M_c^{t-1}||_F (D3 "
                        "stability) and (e) pairwise Frobenius "
                        "distance between distinct chains' M_c "
                        "(D3 distinguishability).")
    p.add_argument("--diagnose_memory_dynamics_n_chains", type=int,
                   default=8,
                   help="Number of held-out chains to use for the "
                        "D2/D3 diagnostic.  Up to chain_lengths[ci] "
                        "session-boundary judge calls are made per "
                        "chain so this is also the cost knob.")

    # ------------------------------------------------------------------
    # Architectural fix A: scaffolded readout warmup (2026-05-01).
    # ------------------------------------------------------------------
    # The D5 audit on chain_v11g_ap_baseline_gh200/best showed that
    # the writer DOES encode chain-specific content -- with 300 steps
    # of TTT on the readout alone (frozen writer + router + LM head),
    # callback CE drops 48% on the synthetic D4 task.  But under the
    # standard joint-optimisation training, the router closes the
    # memory pathway (alpha_mem ~ 0.01) before the readout has a
    # chance to learn to decode, locking in a memory-disabled
    # solution.  The fix:
    #
    #   Phase 1 (steps 0 .. readout_warmup_steps):
    #     - Force the routing toward memory (mem_bias raised) so the
    #       readout's gradient is large.
    #     - Freeze the writer, the router, the backbone, and the LM
    #       head; only the readout (W_Q, W_K, W_V, out_norm) trains.
    #     - The readout learns to decode whatever M_c the writer is
    #       already producing, in isolation from co-optimisation
    #       pressure to close routing.
    #   Phase 2 (steps readout_warmup_steps .. end):
    #     - Unfreeze everything.
    #     - Anneal the routing bias back toward its natural init over
    #       readout_warmup_anneal_steps; the LM and the writer can
    #       now co-evolve, but the readout starts from a competent
    #       state instead of from random projections.
    p.add_argument("--readout_warmup_steps", type=int, default=0,
                   help="Architectural fix A: initial steps where "
                        "all parameters except the memory_readout "
                        "are frozen and the router's mem_bias is "
                        "force-set high.  0 disables the schedule "
                        "(legacy training).  Recommended: "
                        "200-500 for a 4000-step run.")
    p.add_argument("--readout_warmup_router_bias", type=float, default=4.0,
                   help="The mem_bias value held during phase 1.  "
                        "softmax((+4, recent_init=4.0)) ~ uniform "
                        "between memory and recent on a "
                        "0.6B-attention_parity router; that is enough "
                        "memory-channel routing to give the readout "
                        "strong LM gradient.")
    p.add_argument("--readout_warmup_anneal_steps", type=int, default=200,
                   help="After phase 1, linearly anneal mem_bias "
                        "from --readout_warmup_router_bias back to "
                        "--router_mem_bias_init over this many steps.")

    # v13 (2026-05-01): writer warmup.  Extends the readout_warmup
    # idea to the writer: during phase 1, freeze ONLY the backbone
    # and LM head/embed; train the FULL memory subsystem (M_in,
    # extract, M_judge, judge, judge_norm, write_gate, readout,
    # router, memres_gate) under forced-open routing.  The backbone
    # produces stable hidden states for C_t, and the memory
    # subsystem receives the full LM gradient signal with every
    # softmax multiplier already at a useful operating point.
    # This is the single highest-leverage fix identified by the
    # 2026-05-01 audit (problems.md §5): it directly attacks
    # §2a (gradient starvation of M_in/M_judge: 10^-8 of backbone)
    # and §2b (permutation-equivariant uniform fixed point of the
    # judge softmax) by giving the writer 5+ orders of magnitude
    # more effective gradient than it receives under joint LM-only
    # training.
    p.add_argument("--writer_warmup_steps", type=int, default=0,
                   help="v13 architectural fix: phase 1 length where "
                        "backbone + LM head + embed are frozen and the "
                        "entire memres subsystem (writer + readout + "
                        "router + gate) trains under forced-open "
                        "routing (see --writer_warmup_router_bias). 0 "
                        "disables the schedule (legacy training). "
                        "Recommended: 500-1000 for a 4000-step run. "
                        "Supersedes --readout_warmup_steps if both are "
                        "set (writer_warmup trains a strict superset of "
                        "parameters).")
    p.add_argument("--writer_warmup_router_bias", type=float, default=4.0,
                   help="The mem_bias value held during writer warmup "
                        "phase 1. With recent_bias_init=4.0 the router "
                        "softmax gives memory ~50% mass on "
                        "attention_parity, so LM gradient through m^t "
                        "is dominant.")
    p.add_argument("--writer_warmup_anneal_steps", type=int, default=200,
                   help="After writer warmup phase 1, linearly anneal "
                        "mem_bias back to --router_mem_bias_init over "
                        "this many steps. Backbone also unfreezes at "
                        "the end of phase 1 (unless --freeze_backbone "
                        "is also set).")
    p.add_argument("--writer_warmup_keep_backbone_frozen",
                   action="store_true",
                   help="If set, the backbone stays frozen even after "
                        "writer warmup phase 1 ends.  Equivalent to "
                        "--freeze_backbone.  Useful when composing "
                        "writer warmup with a frozen-backbone recipe "
                        "(v11l / v12d_frozen / v13 headline).")
    # v13 symmetry-break init knobs (threaded to Qwen3MemResConfig).
    p.add_argument("--memres_queries_init",
                   choices=("normal", "orthogonal"),
                   default="normal",
                   help="Init for M_in and M_judge writer-query seeds. "
                        "'orthogonal' breaks the permutation-"
                        "equivariant uniform fixed point observed in "
                        "v11g/v12a D2 audits (row_entropy/log(2K)="
                        "0.999, eff_rank=1.02).")
    p.add_argument("--memres_slot_positional", action="store_true",
                   help="Add a learnable Fourier per-slot positional "
                        "embedding to M_in and M_judge before the "
                        "Q/K/V projections.  Gives each slot a "
                        "permanent address the symmetry-break survives "
                        "adversarial drift.")
    # v14 architectural knobs (threaded to Qwen3MemResConfig).
    p.add_argument("--memres_judge_qk_layernorm", action="store_true",
                   help="Apply post-projection RMSNorm to the Q and K "
                        "of the Stage-2 judge (MemoryBlock.judging).  "
                        "Decouples attention-logit magnitude from "
                        "W_Q/W_K spectral norm so the judge softmax "
                        "can find sharper distributions.  v14 fix for "
                        "the D2-JUDGE row_entropy / log(2K) > 0.95 "
                        "uniform-attention pathology measured on v13r "
                        "@ step 10000 (Zhai et al. 2023 entropy-"
                        "collapse direction; inverse pathology here).")
    # v14 auxiliary-loss knobs.  These are read during fit() and
    # applied only when their weight is > 0; defaults preserve
    # pre-v14 behaviour exactly.
    p.add_argument("--alpha_mem_floor_aux_weight", type=float, default=0.0,
                   help="Weight on an MoE-style load-balance auxiliary "
                        "loss that penalises the depth router's "
                        "per-sublayer alpha_mem dropping below "
                        "--alpha_mem_floor_target.  Computed as "
                        "mean_l relu(target - mean(alpha_mem_l)) "
                        "over all routed sublayers l, averaged over "
                        "positions.  Targets the v13 finding that the "
                        "router actively collapses memory usage to "
                        "alpha_mem_mean ~ 0.001 (textbook MoE expert "
                        "collapse).  0.0 disables.")
    p.add_argument("--alpha_mem_floor_target", type=float, default=0.05,
                   help="Target floor for per-sublayer mean(alpha_mem). "
                        "0.05 matches the manuscript's empirical "
                        "'mild' recruitment band and leaves plenty of "
                        "mass on the non-memory sources during content-"
                        "rich turns.  Only active when "
                        "--alpha_mem_floor_aux_weight > 0.")

    p.add_argument("--burn_in_max", type=int, default=12,
                   help="Maximum number of sessions to unroll under "
                        "no_grad before the TBPTT window starts.  This lets "
                        "the model see M_c states from realistic recurrence "
                        "depths during training; needed because eval-time "
                        "chains are 20+ sessions long while TBPTT is only "
                        "k sessions deep. With --burn_in_resample, the actual "
                        "burn-in is sampled per-step in {0..burn_in_max} "
                        "(LRMT failure-mode defense, COMPREHENSIVE §4.1).")
    p.add_argument("--burn_in_resample", action="store_true",
                   help="If set, resample burn-in per chain in {0,4,8,...,burn_in_max} "
                        "so the model sees M_c states from a range of recurrence "
                        "depths during training.")
    p.add_argument("--save_best_metric",
                   choices=("ce_mem", "composite", "phase_aligned"),
                   default="phase_aligned",
                   help="ce_mem: legacy; minimised. "
                        "composite: maximise standard-eval "
                        "(delta_nomem_minus_mem + 2*delta_shuffle_minus_mem). "
                        "phase_aligned (default; v8+): maximise "
                        "(pa_cb_dsh + 0.5 * pa_ws_dsh) on the phase-aligned "
                        "eval that matches the curriculum training "
                        "distribution. The phase-aligned callback-token "
                        "delta_sh-m is the only metric that actually "
                        "measures whether the readout is content- "
                        "discriminative on the tokens we care about, "
                        "without being polluted by the train/eval "
                        "distribution mismatch on M_c statistics.")
    p.add_argument("--phase_aligned_eval_n_chains", type=int, default=48,
                   help="Number of LME chains (with cb_pos >= 1) to use "
                        "for the phase-aligned eval. The eval picks one "
                        "random evidence position per chain and one "
                        "shuffle partner, so cost is roughly 4 forwards "
                        "per chain (mem readout + nomem + shuffle + "
                        "compress * 2).")
    p.add_argument("--diag_routing_n_chains", type=int, default=8,
                   help="Number of chains to use for the per-sublayer "
                        "routing-recruitment diagnostic (alpha_mem "
                        "trace in attention_parity mode; gate snapshot "
                        "in simple_gate mode -- the latter doesn't need "
                        "data forwards). Cheap, default 8.")
    p.add_argument("--eval_only", action="store_true",
                   help="If set, skip training and run the phase-aligned "
                        "eval (plus routing + readout diagnostics) N "
                        "times on the loaded checkpoint, then exit. "
                        "Intended for reading the signal off a saved "
                        "best with a bigger eval sample and multiple "
                        "seeds. N is controlled by --eval_seeds.")
    p.add_argument("--eval_seeds", type=int, default=5,
                   help="Number of independent PA-eval seeds to run in "
                        "--eval_only mode. Mean + std + 95%% CI are "
                        "reported for all PA metrics.")
    p.add_argument("--eval_out", default=None,
                   help="If set in --eval_only mode, write the eval "
                        "JSON to this path (default stdout only).")

    # Conversational-pipeline knobs
    p.add_argument("--mask_padding_loss", action="store_true",
                   help="Mask EOS-padding positions from the LM loss. "
                        "Essential for conversational corpora (MSC) where "
                        "sessions are ~150 tokens padded with EOS to "
                        "session_len; without masking ~70%% of the loss "
                        "is EOS-on-EOS noise. Detected by the first EOS "
                        "in each row of input_ids -- safe iff the "
                        "tokeniser only emits EOS as trailing padding "
                        "(which is what pretokenize_chains.py does).")
    p.add_argument("--score_tail_frac", type=float, default=1.0,
                   help="Score only the last fraction of each session's "
                        "*non-padding* content.  E.g. 0.5 means only the "
                        "second half of each session contributes to the "
                        "LM loss; the first half is treated as context. "
                        "Concentrates gradient on the response tail "
                        "where memory should matter most.  Default 1.0 "
                        "= legacy (score the entire content).")

    # IO + logging
    p.add_argument("--out_dir", default="output/chain_run")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_n_chains", type=int, default=24,
                   help="# eval chains; we use the *full* prefix per chain "
                        "(up to a clamp) to score the last few sessions.")
    p.add_argument("--eval_window", type=int, default=4,
                   help="Score the last N sessions of each eval chain "
                        "(M_c built sequentially over all preceding sessions "
                        "regardless of this clamp).")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--freeze_backbone", action="store_true",
        help="If set, every non-memres backbone parameter has requires_grad=False "
             "so the optimiser only sees memres params (memory_block, "
             "memory_readout, memory_gate, depth_router). Cuts optimiser-state "
             "memory by ~(backbone/total)x -- REQUIRED for qwen3-8b-xlarge "
             "(L_E=10) on a single 96 GB GPU under default PyTorch AdamW "
             "without bitsandbytes. Forward/backward still run through the "
             "backbone; only the optimiser update is suppressed.",
    )
    p.add_argument(
        "--use_adam8bit", action="store_true",
        help="Use bitsandbytes.optim.AdamW8bit instead of torch.optim.AdamW. "
             "Cuts optimiser state by ~4x (1 byte/param for m and v in "
             "block-wise quantisation). Allows full backbone+memres training "
             "on single 96 GB GPU for 8B models. Requires `pip install "
             "bitsandbytes`; silently falls through to standard AdamW if the "
             "import fails.",
    )

    a = p.parse_args()
    if a.preset is not None:
        apply_preset(a, a.preset)
    a.memres_mode = _normalise_memres_mode(
        a.memres_mode, a.block_attnres_parity_init
    )
    a.block_attnres_parity_init = a.memres_mode == "attention_parity"
    return a


def cosine_with_warmup(step: int, warmup: int, total: int, lr_min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


# ---------------------------------------------------------------------------
# Chain dataset
# ---------------------------------------------------------------------------


class ChainCorpus:
    """Stores one big (N_sessions, S) tensor and per-chain metadata.

    The training sampler picks a chain (with optional source up-weighting)
    and a random window inside it.
    """

    def __init__(self, path: Path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.session_ids: torch.Tensor = blob["session_ids"].long()
        self.chain_starts: torch.Tensor = blob["chain_starts"]
        self.chain_lengths: torch.Tensor = blob["chain_lengths"]
        self.chain_names: list[str] = blob["chain_names"]
        self.session_len: int = int(blob["session_len"])
        self.tokenizer: str = blob["tokenizer"]
        # Optional fields (v6+ conversational callback corpora).
        # Existing pretokenize_chains.py output (PG-19/TV/MSC) lacks these;
        # we fall back to an all-zeros callback mask and chain_callback_position
        # sentinel of -1 so the trainer is corpus-format agnostic.
        if "session_callback_mask" in blob:
            self.session_callback_mask: torch.Tensor = (
                blob["session_callback_mask"].to(torch.int8)
            )
        else:
            self.session_callback_mask = torch.zeros_like(
                self.session_ids, dtype=torch.int8
            )
        if "chain_callback_position" in blob:
            self.chain_callback_position: torch.Tensor = (
                blob["chain_callback_position"].long()
            )
        else:
            # Sentinel: -1 means "no callback session in this chain".
            self.chain_callback_position = torch.full(
                (len(self.chain_starts),), -1, dtype=torch.long
            )
        # v11 (2026-04-30): per-chain list of session indices that
        # actually contain the answer text (LongMemEval ships these as
        # ``answer_session_ids``; the v11 corpus builder preserves
        # them). Empty list per chain = "no ground-truth labels"
        # (non-LME sources, or pre-v11 LME corpora). The training
        # curriculum_competition branch and the phase-aligned eval
        # use this to sample meaningful evidence sessions instead of
        # picking uniformly over the haystack (which has ~3.6%
        # evidence prior on a 48-session chain -- the v9/v10 root
        # cause documented in README "Stop everything").
        ev_raw = blob.get("chain_evidence_positions")
        n_chains = len(self.chain_starts)
        if ev_raw is None:
            self.chain_evidence_positions: list[list[int]] = [[] for _ in range(n_chains)]
        else:
            # Defensive copy & normalisation: cast to ints, drop any
            # out-of-range positions (cb_pos >= length, or negative).
            normalised: list[list[int]] = []
            for ci in range(n_chains):
                raw = ev_raw[ci] if ci < len(ev_raw) else []
                cb = int(self.chain_callback_position[ci])
                length = int(self.chain_lengths[ci])
                cleaned = [
                    int(p) for p in raw
                    if 0 <= int(p) < length and (cb < 0 or int(p) < cb)
                ]
                normalised.append(cleaned)
            self.chain_evidence_positions = normalised

    def __len__(self) -> int:
        return len(self.chain_starts)

    def chain_session_at(self, chain_idx: int, position: int) -> torch.Tensor:
        start = int(self.chain_starts[chain_idx])
        return self.session_ids[start + position]

    def chain_window(self, chain_idx: int, start: int, k: int) -> torch.Tensor:
        s = int(self.chain_starts[chain_idx])
        return self.session_ids[s + start : s + start + k]

    def chain_window_callback_mask(
        self, chain_idx: int, start: int, k: int,
    ) -> torch.Tensor:
        """Same slice as chain_window but on the callback mask."""
        s = int(self.chain_starts[chain_idx])
        return self.session_callback_mask[s + start : s + start + k]

    def chain_curriculum_window(
        self,
        chain_idx: int,
        positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack a non-contiguous window of sessions in the order given.

        ``positions`` is a list of session indices within the chain (each
        in ``[0, chain_lengths[chain_idx])``); the result is a tensor of
        shape ``(len(positions), session_len)`` with sessions in the
        listed order, plus the matching callback mask of the same shape.
        Used by the curriculum sampler to construct
        ``[evidence, ...intermediates, callback]`` windows that collapse
        the credit-assignment path between an early-session fact and a
        late-session callback.
        """
        s = int(self.chain_starts[chain_idx])
        idxs = torch.tensor([s + p for p in positions], dtype=torch.long)
        return (
            self.session_ids.index_select(0, idxs),
            self.session_callback_mask.index_select(0, idxs),
        )

    def chain_prefix(self, chain_idx: int, end: int) -> torch.Tensor:
        s = int(self.chain_starts[chain_idx])
        return self.session_ids[s : s + end]


def _detect_source(name: str) -> str:
    """Detect the corpus source of a chain from its name.

    'msc_<id>'          -> 'msc'         (multi-session chat dialogues)
    'longmemeval_<id>'  -> 'longmemeval' (LongMemEval-S, callback-supervised)
    'realtalk_<id>'     -> 'realtalk'    (REALTALK 21-day messaging)
    'ultrachat_<id>'    -> 'ultrachat'   (v10 mega corpus: HF ultrachat_200k)
    'pippa_<id>'        -> 'pippa'       (v10 mega corpus: PygmalionAI/PIPPA character chats)
    'soda_<id>'         -> 'soda'        (v10 mega corpus: allenai/soda synthetic social)
    'lmsys_<id>'        -> 'lmsys'       (v10 mega corpus: lmsys-chat-1m real user-assistant)
    'synthdlg_<id>'     -> 'synthdlg'    (v10 mega corpus: generic synthetic dialogue fallback)
    leading digit       -> 'pg19'        (PG-19 books are named '<book_id>')
    anything else       -> 'tv'          (TV episode chains have show-name prefixes)
    """
    if name.startswith("msc_"):
        return "msc"
    if name.startswith("longmemeval_"):
        return "longmemeval"
    if name.startswith("realtalk_"):
        return "realtalk"
    if name.startswith("ultrachat_"):
        return "ultrachat"
    if name.startswith("pippa_"):
        return "pippa"
    if name.startswith("soda_"):
        return "soda"
    if name.startswith("lmsys_"):
        return "lmsys"
    if name.startswith("synthdlg_"):
        return "synthdlg"
    if name[:1].isdigit():
        return "pg19"
    return "tv"


class ChainSampler:
    def __init__(
        self,
        corpus: ChainCorpus,
        rank: int,
        world_size: int,
        seed: int,
        window_k: int,
        source_weights: dict[str, float] | None = None,
        callback_window_bias: float = 0.0,
        curriculum_evidence_bias: float = 0.0,
        curriculum_competition_bias: float = 0.0,
    ):
        self.corpus = corpus
        self.rank = rank
        self.world_size = world_size
        self.window_k = window_k
        self.rng = random.Random(seed + rank * 7919)
        # Probability of sampling a window that ENDS at the callback session
        # (i.e. start = max(0, cb_pos - window_k + 1)). Only fires for
        # chains whose chain_callback_position >= 0. Lets the trainer
        # actually see the callback supervision on long chains where
        # uniform sampling would only hit it ~1/L of the time.
        self.callback_window_bias = float(callback_window_bias)
        # Probability of building a CURRICULUM window
        # [evidence, ...intermediates, callback] -- used to collapse the
        # credit-assignment path between an early-session fact and the
        # callback that depends on it. See --curriculum_evidence_bias
        # CLI help for the rationale.
        self.curriculum_evidence_bias = float(curriculum_evidence_bias)
        # Probability of building a JUDGE-COMPETITION pair window
        # [(evidence|noise), (distractor|evidence), callback] specifically
        # designed to train the judge layer's keep-vs-write decision in
        # isolation from the writer/readout sub-problem. See
        # --curriculum_competition_bias CLI help for the rationale and the
        # paired sample structure.
        self.curriculum_competition_bias = float(curriculum_competition_bias)

        # Default source up-weights.  PG-19 has ~218k chains so it dominates
        # by token count; TV (~30 chains) and MSC (~4k chains) get
        # multiplicative up-weight on per-chain length so the sampler
        # actually visits them.
        # LongMemEval (450 chains, callback-supervised) is upweighted by
        # default because it's the only corpus with explicit memory
        # supervision; mixing in MSC for distribution otherwise dilutes
        # the gradient signal on memory tokens.
        default_w = {
            "pg19": 1.0, "tv": 4.0, "msc": 3.0,
            "longmemeval": 4.0, "realtalk": 1.0,
            "ultrachat": 2.0, "pippa": 2.0, "soda": 1.5,
            "lmsys": 2.0, "synthdlg": 1.5,
        }
        if source_weights:
            default_w.update(source_weights)

        weights: list[float] = []
        source_counts: dict[str, int] = {}
        for ci, name in enumerate(corpus.chain_names):
            length = int(corpus.chain_lengths[ci])
            src = _detect_source(name)
            source_counts[src] = source_counts.get(src, 0) + 1
            # Eligibility: a chain of length L can produce a window of
            # window_k sessions iff L >= window_k (start=0 is always
            # legal even when no burn-in prefix is available).  This is
            # critical for short conversational chains (MSC): 3-session
            # dialogues are eligible at window_k=3 with start=0.
            if length < window_k:
                weights.append(0.0)
                continue
            w = default_w.get(src, 1.0)
            weights.append(w * length)
        self._source_counts = source_counts
        self._effective_weights = default_w
        self.weights = weights
        self.cum = []
        running = 0.0
        for w in weights:
            running += w
            self.cum.append(running)
        self.total = running
        self.eligible = sum(1 for w in weights if w > 0)
        if self.total <= 0:
            raise RuntimeError("No eligible chains for sampling")

    def sample_window(
        self,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Sample a (chain_idx, start, window, burn_in, callback_mask) tuple.

        ``start`` is the position of the FIRST included session in the chain.
        For contiguous windows it indexes ``window`` exactly. For curriculum
        windows (non-contiguous), ``start`` is the evidence position and the
        intermediate / callback positions are not exposed; the precomputed
        ``callback_mask`` is then non-None and the trainer must use it
        instead of re-slicing the corpus mask via (start, window_k).
        """
        r = self.rng.random() * self.total
        # binary search
        lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= self.cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        chain_idx = lo
        length = int(self.corpus.chain_lengths[chain_idx])
        max_start = length - self.window_k
        cb_pos = int(self.corpus.chain_callback_position[chain_idx])

        # Branch 0: judge-competition pair (NEW in v9). Builds a 3-session
        # window that ISOLATES the judge layer's keep-vs-write decision.
        # Two paired structures, sampled 50/50:
        #
        #   Sample A (KEEP-PREV, judge must keep evidence in memory):
        #     window = [evidence, distractor, callback]
        #     -- step 0: M = compress(extract(evidence), 0)  (writer + extractor)
        #     -- step 1: M = compress(extract(distractor), M_prev)  -- THE JUDGE STEP
        #                  judge sees prev_M = M_after(evidence) (relevant)
        #                  judge sees C_t  = extract(distractor)   (irrelevant)
        #                  correct behaviour: gate small, KEEP M_prev
        #     -- step 2: forward(callback, M_c=M); CB tokens scored
        #
        #   Sample B (WRITE-NEW, judge must overwrite irrelevant memory):
        #     window = [noise, evidence, callback]
        #     -- step 0: M = compress(extract(noise), 0)
        #     -- step 1: M = compress(extract(evidence), M_prev)  -- THE JUDGE STEP
        #                  judge sees prev_M = M_after(noise)     (irrelevant)
        #                  judge sees C_t  = extract(evidence)    (relevant)
        #                  correct behaviour: gate large, WRITE C_t
        #     -- step 2: forward(callback, M_c=M); CB tokens scored
        #
        # Both samples score the same callback session, so the gradient
        # directly trains the write_gate / judge weights to be content-
        # aware. Requires window_k >= 3 and cb_pos >= 2 (Sample A) or
        # cb_pos >= 2 with evidence_pos >= 1 (Sample B).
        # Note on label noise: the LME corpus does NOT annotate which
        # earlier session contains the actual referenced fact -- only
        # that cb_pos is the session that CONTAINS the callback
        # question. The "evidence" in Sample A and "noise" in Sample B
        # are uniform random picks from [0, cb_pos).
        #
        # This is INTENTIONAL and correct, not a corpus shortcoming:
        # at deployment we will not have ground-truth evidence
        # annotations either. The model's job is to compact arbitrary
        # incoming sessions into M_c such that whatever might be
        # callback-relevant survives, and to write_gate / judge based
        # on observed content alignment with downstream demand. The
        # training curriculum therefore reflects the deployment
        # regime: the model sees a stream of sessions, must compact
        # them, and gets graded on callback prediction quality. Across
        # many samples the gradient signal averages to "the
        # writer + judge should preserve content that LOOKS like it
        # could be referenced later" -- which is exactly the
        # generalisable inductive bias we want.
        #
        # ---------------------------------------------------------------
        # v11 (2026-04-30) UPDATE -- evidence-aware sampling.
        # ---------------------------------------------------------------
        # The pre-v11 trainer sampled "evidence" and "noise" uniformly
        # from [0, cb_pos). LongMemEval-S has mean ~48 haystack
        # sessions and mean ~1.9 ground-truth evidence sessions per
        # chain, so the prior on any random session being evidence is
        # ~3.97%. That meant 96%+ of training samples built M_c from
        # sessions that demonstrably did not contain the answer, the
        # gradient said "ignore memory" because that's the LM-loss-
        # optimal policy on the sampled distribution, and the writer/
        # judge/readout converged to "do nothing useful". See
        # README "Stop everything" P0 for the full diagnosis.
        #
        # The v11 corpus preserves ``answer_session_ids`` as
        # ``ChainCorpus.chain_evidence_positions[ci]``. When
        # evidence_positions is non-empty we sample the "evidence"
        # slot only from that list; the "distractor" / "noise" slot
        # is sampled from the *complement* (non-evidence positions in
        # the appropriate range) so the judge's keep-vs-write decision
        # is grounded in real content asymmetry. When evidence_positions
        # is empty (non-LME chains) we fall back to the original
        # uniform behaviour. This is NOT an oracle leak at inference:
        # at deployment M_c is built sequentially over all haystack
        # sessions, so it always contains the answer; we simply align
        # training-time distribution with that property.
        if (
            self.curriculum_competition_bias > 0.0
            and self.window_k >= 3
            and cb_pos >= 2
            and self.rng.random() < self.curriculum_competition_bias
        ):
            ev_positions = self.corpus.chain_evidence_positions[chain_idx]
            # Filter to the legal range (defensive; ChainCorpus already
            # normalises but cb_pos >= 2 guard means we further want
            # ev positions in [0, cb_pos-1]).
            ev_in_range = [p for p in ev_positions if 0 <= p < cb_pos]

            keep_prev = self.rng.random() < 0.5
            if keep_prev:
                # Sample A: [evidence, distractor, callback]
                # Evidence: prefer a *true* evidence position (v11). Need
                # evidence_pos <= cb_pos - 2 so that a distractor slot fits.
                ev_kp = [p for p in ev_in_range if p <= cb_pos - 2]
                if ev_kp:
                    evidence_pos = self.rng.choice(ev_kp)
                else:
                    # Fallback: uniform (legacy v9/v10 behaviour).
                    evidence_pos = self.rng.randint(0, cb_pos - 2)
                # Distractor: uniform from the *non-evidence* gap
                # (evidence_pos, cb_pos). Falls back to uniform if all
                # candidate slots happen to be evidence (rare).
                gap = list(range(evidence_pos + 1, cb_pos))
                non_ev_gap = [p for p in gap if p not in ev_in_range]
                if non_ev_gap:
                    distractor_pos = self.rng.choice(non_ev_gap)
                elif gap:
                    distractor_pos = self.rng.choice(gap)
                else:
                    distractor_pos = evidence_pos + 1  # safety fallback
                positions = [evidence_pos, distractor_pos, cb_pos]
                anchor = evidence_pos
            else:
                # Sample B: [noise, evidence, callback]
                # Evidence: prefer a *true* evidence position with
                # evidence_pos >= 1 so a noise slot fits before it.
                ev_wn = [p for p in ev_in_range if p >= 1]
                if ev_wn:
                    evidence_pos = self.rng.choice(ev_wn)
                else:
                    evidence_pos = self.rng.randint(1, cb_pos - 1)
                # Noise: uniform from the *non-evidence* prefix
                # [0, evidence_pos). Fallback to uniform if no non-
                # evidence option exists.
                pre = list(range(0, evidence_pos))
                non_ev_pre = [p for p in pre if p not in ev_in_range]
                if non_ev_pre:
                    noise_pos = self.rng.choice(non_ev_pre)
                elif pre:
                    noise_pos = self.rng.choice(pre)
                else:
                    noise_pos = 0  # safety fallback (shouldn't trigger)
                positions = [noise_pos, evidence_pos, cb_pos]
                anchor = noise_pos
            if positions:
                window, cb_mask = self.corpus.chain_curriculum_window(
                    chain_idx, positions
                )
                # Pad to window_k if window_k > 3 by re-using the first
                # session as additional context. This is rare and only
                # happens when a cell mixes competition_bias with a
                # window_k > 3 setting; the typical v9 cell uses
                # window_k=3 which is the natural fit.
                if self.window_k > 3:
                    pad_n = self.window_k - 3
                    pad_pos = [positions[0]] * pad_n + positions
                    window, cb_mask = self.corpus.chain_curriculum_window(
                        chain_idx, pad_pos
                    )
                return chain_idx, anchor, window, None, cb_mask

        # Branch 1: curriculum [evidence, ...intermediates, callback]. Requires
        # an annotated callback position and enough room to fit window_k - 1
        # earlier sessions. window_k=2 needs cb_pos >= 1 (one evidence slot);
        # window_k=k needs cb_pos >= k-1 (one evidence + k-2 intermediates).
        if (
            self.curriculum_evidence_bias > 0.0
            and cb_pos >= self.window_k - 1
            and self.rng.random() < self.curriculum_evidence_bias
        ):
            evidence_pos = self.rng.randint(0, cb_pos - 1)
            n_intermediate = self.window_k - 2
            if n_intermediate > 0:
                # Sample intermediates strictly between evidence and callback.
                # If the gap is too small (rare given the eligibility check),
                # fall through to contiguous below.
                gap = list(range(evidence_pos + 1, cb_pos))
                if len(gap) >= n_intermediate:
                    intermediates = sorted(self.rng.sample(gap, n_intermediate))
                    positions = [evidence_pos, *intermediates, cb_pos]
                    window, cb_mask = self.corpus.chain_curriculum_window(
                        chain_idx, positions
                    )
                    # No burn-in for curriculum: M_c starts fresh at evidence
                    # so the credit-assignment chain is exactly [evidence -> ... -> callback].
                    return chain_idx, evidence_pos, window, None, cb_mask
            else:
                # window_k == 2: just [evidence, callback]
                positions = [evidence_pos, cb_pos]
                window, cb_mask = self.corpus.chain_curriculum_window(
                    chain_idx, positions
                )
                return chain_idx, evidence_pos, window, None, cb_mask

        # Branch 2: callback alignment (existing behavior). Window is
        # contiguous and ends at cb_pos.
        if (
            self.callback_window_bias > 0.0
            and cb_pos >= 0
            and cb_pos < length
            and self.rng.random() < self.callback_window_bias
        ):
            start = max(0, cb_pos - self.window_k + 1)
            start = min(start, max_start)
        else:
            start = self.rng.randint(0, max_start)
        window = self.corpus.chain_window(chain_idx, start, self.window_k)
        burn_in = (
            self.corpus.chain_window(chain_idx, 0, start) if start > 0 else None
        )
        return chain_idx, start, window, burn_in, None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._init_distributed()
        self._resolve_names()
        torch.manual_seed(args.seed + self.rank)
        random.seed(args.seed + self.rank)

        self.is_main = self.rank == 0
        self.use_wandb = self._init_wandb()
        self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path())
        # EOS token id used by pretokenize_chains.py to right-pad sessions
        # to session_len.  Conversational corpora (MSC) have sessions
        # ~150 tokens padded to 512, so masking padding from the LM loss
        # is essential -- otherwise ~70%% of the loss is the easy
        # EOS-on-EOS prediction and gradient signal on real content
        # collapses.
        self.eos_id = int(self.tokenizer.eos_token_id)

        if self.is_main:
            print(self._banner())

        self.model = self._build_model()
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._apply_freeze()
        self.wrapped = self._wrap_model()
        self.optimizer, self.scheduler = self._build_optimizer()

        if self.is_main:
            print(f"  loading train chains: {args.train_chains}", flush=True)
        self.train_corpus = ChainCorpus(Path(args.train_chains))
        sw = None
        if args.source_weights:
            try:
                sw = json.loads(args.source_weights)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"--source_weights must be valid JSON; got {args.source_weights!r}"
                ) from e
        self.train_sampler = ChainSampler(
            self.train_corpus, self.rank, self.world_size, args.seed,
            args.window_k, source_weights=sw,
            callback_window_bias=args.callback_window_bias,
            curriculum_evidence_bias=args.curriculum_evidence_bias,
            curriculum_competition_bias=args.curriculum_competition_bias,
        )
        if self.is_main:
            print(
                f"  train: {len(self.train_corpus)} chains, "
                f"{self.train_sampler.eligible} eligible "
                f"(window_k={args.window_k}); "
                f"source counts={self.train_sampler._source_counts}; "
                f"weights={self.train_sampler._effective_weights}",
                flush=True,
            )

        if self.is_main:
            print(f"  loading eval chains: {args.eval_chains}", flush=True)
        self.eval_corpus = ChainCorpus(Path(args.eval_chains))
        if self.is_main:
            print(
                f"  eval: {len(self.eval_corpus)} chains, "
                f"sessions: {self.eval_corpus.session_ids.shape[0]}",
                flush=True,
            )

        os.makedirs(args.out_dir, exist_ok=True)
        self.best_eval_ce = float("inf")
        self.global_step = 0

        # Telemetry for the InfoNCE contrastive loss.  Updated per step
        # in _train_step when --contrastive_infonce_weight > 0; printed
        # on the throttled training-step log line and pushed to wandb.
        self._last_contrastive_loss = 0.0
        self._last_contrastive_diag = 0.0
        self._last_contrastive_offdiag = 0.0
        self._last_contrastive_gap = 0.0

        # D1 telemetry: pre-compute per-module gradient-group buckets
        # once.  Buckets are mutually exclusive (each grad-tracked param
        # lands in exactly one group), so the sum of squared norms equals
        # the global ``grad_norm`` reported pre-clip.  See
        # ``--diagnose_grad_groups`` in build_arg_parser for rationale.
        if getattr(args, "diagnose_grad_groups", False):
            self._param_groups_for_telemetry = self._build_telemetry_groups()
        else:
            self._param_groups_for_telemetry = {}
        self._last_grad_norms: dict[str, float] = {}

        if args.init_from:
            if self.is_main:
                print(f"  warm-start (memres params only) from {args.init_from}",
                      flush=True)
            self._load_memres_warm_start(args.init_from)

    # ------------------------------------------------------------------
    # Distributed / model setup
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        if dist.is_available() and "WORLD_SIZE" in os.environ:
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

    def _resolve_names(self) -> None:
        a = self.args
        if a.run_name is None:
            tag = a.preset or "scratch"
            a.run_name = f"chain-{tag}-K{a.memres_num_vectors}-k{a.window_k}-{a.steps // 1000}k"

    def _tokenizer_path(self) -> str:
        if self.args.pretrained:
            return self.args.pretrained
        return "Qwen/Qwen3-0.6B"

    def _banner(self) -> str:
        a = self.args
        return (
            f"\n=== Memory Residuals -- recurrent chain trainer ===\n"
            f"  run_name        : {a.run_name}\n"
            f"  preset          : {a.preset}\n"
            f"  pretrained      : {a.pretrained}\n"
            f"  K, L_E, N       : {a.memres_num_vectors}, "
            f"{a.memres_extraction_depth}, {a.memres_num_blocks}\n"
            f"  session_len     : {a.session_len}, window_k={a.window_k}\n"
            f"  steps           : {a.steps}, bs={a.batch_size}, "
            f"grad_accum={a.grad_accum}\n"
            f"  lr              : memres={a.lr}, backbone={a.lr_backbone}\n"
            f"  reg             : mem_drop={a.memory_dropout}, "
            f"ctx_drop={a.context_dropout}, carry_state={a.carry_state}\n"
            f"  world_size      : {self.world_size}\n"
            f"  out_dir         : {a.out_dir}"
        )

    def _init_wandb(self) -> bool:
        if not self.is_main or not self.args.wandb_project:
            return False
        try:
            import wandb
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.run_name,
                config=vars(self.args),
            )
            return True
        except Exception as exc:
            print(f"W&B init failed ({exc})")
            return False

    def _build_model(self) -> Qwen3MemResForCausalLM:
        a = self.args
        memres_kwargs = dict(
            memres_num_vectors=a.memres_num_vectors,
            memres_extraction_depth=a.memres_extraction_depth,
            memres_num_blocks=a.memres_num_blocks,
            memres_mode=a.memres_mode,
            block_attnres_parity_init=a.block_attnres_parity_init,
            router_mem_bias_init=a.router_mem_bias_init,
            router_recent_bias_init=a.router_recent_bias_init,
            memres_extract_source=a.memres_extract_source,
            memres_update_mode=a.memres_update_mode,
            memres_gate_init=a.memres_gate_init,
            memres_readout_norm_init=a.memres_readout_norm_init,
            memres_writer_kind=a.memres_writer_kind,
            memres_slot_attention_iters=a.memres_slot_attention_iters,
            memres_queries_init=a.memres_queries_init,
            memres_slot_positional=a.memres_slot_positional,
            memres_judge_qk_layernorm=a.memres_judge_qk_layernorm,
        )
        if a.pretrained:
            # A memres checkpoint saves model_type="qwen3_memres", which
            # AutoConfig does not know about; try loading it directly
            # before falling back to AutoConfig (which handles the base
            # Qwen3 / HF preset path).
            #
            # BUGFIX 2026-05-01 (v13 campaign): ``Qwen3MemResConfig`` is a
            # subclass of ``Qwen3Config``, so ``.from_pretrained`` ALSO
            # succeeds on a plain Qwen3 config.json.  Using the try/except
            # success as the "is this a memres checkpoint?" detector is
            # therefore a false-positive on every non-memres base model
            # (e.g. ``Qwen/Qwen3-0.6B``), which silently falls through to
            # the overridable-subset restriction below and DROPS CLI
            # overrides of ``--memres_mode``, ``--memres_writer_kind``,
            # ``--memres_slot_positional``, ``--memres_update_mode``,
            # ``--memres_extract_source``, ``--memres_num_vectors``,
            # ``--memres_extraction_depth``, ``--memres_num_blocks``,
            # ``--memres_slot_attention_iters``.  Root cause of v13a/b/c
            # running in ``attention_parity`` + ``original`` writer mode
            # despite requesting ``simple_gate`` + ``slot_attention`` in
            # the launch script.  Fix: detect by the raw ``model_type``
            # field loaded from config.json -- a real memres checkpoint
            # has ``"qwen3_memres"``, a plain Qwen3 base has ``"qwen3"``.
            from_memres_ckpt = False
            try:
                base_cfg = Qwen3MemResConfig.from_pretrained(a.pretrained)
                if getattr(base_cfg, "model_type", None) == "qwen3_memres":
                    from_memres_ckpt = True
            except Exception:
                base_cfg = AutoConfig.from_pretrained(a.pretrained)
            # When loading from a memres checkpoint, preserve the
            # architecture fields that were baked into the ckpt (K,
            # L_E, N, routing mode, extract source, update mode) --
            # otherwise CLI defaults (L_E=0 when no preset is given)
            # silently reshape the model and leave 4/5 of the
            # extraction stack randomly initialised.  Only allow CLI
            # override of router *init* biases (those are training
            # knobs, not architecture shape).
            if from_memres_ckpt:
                overridable = {
                    "router_mem_bias_init",
                    "router_recent_bias_init",
                    "block_attnres_parity_init",
                    # v11 (2026-04-30): the bootstrap-fix init knobs
                    # are *training* knobs (they affect parameter
                    # init values, not architecture shape), so we
                    # allow CLI override even when warm-starting from
                    # a memres checkpoint. Without this the trainer
                    # silently uses the *config*'s default (0.0 / 1.0)
                    # and the v11 fix becomes a no-op -- root cause
                    # of the v11 first-launch smoke-test failure.
                    "memres_gate_init",
                    "memres_readout_norm_init",
                    # v13 symmetry-break knobs: also architecture-shape
                    # (they only affect the init values of M_in, M_judge,
                    # and add / don't add M_in_pos / M_judge_pos
                    # parameters).  When warm-starting from a memres
                    # checkpoint, shape-changing flags must match the
                    # checkpoint, so ``memres_slot_positional`` is NOT
                    # in this overridable set — changing it would
                    # silently drop or create parameters.  The init
                    # kind IS overridable because it only affects
                    # values at construction time, which are immediately
                    # overwritten by the loaded checkpoint anyway.
                    "memres_queries_init",
                }
                merged = dict(base_cfg.to_dict())
                for k, v in memres_kwargs.items():
                    if k in overridable:
                        merged[k] = v
                cfg = Qwen3MemResConfig(**merged)
            else:
                cfg = Qwen3MemResConfig(
                    **{**base_cfg.to_dict(), **memres_kwargs}
                )
            return Qwen3MemResForCausalLM.from_pretrained(
                a.pretrained, config=cfg, dtype=torch.bfloat16
            ).to(self.device)
        raise ValueError("Chain trainer requires --pretrained or --preset")

    def _load_memres_warm_start(self, path: str) -> None:
        from safetensors.torch import load_file
        ckpt = Path(path) / "model.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        state = load_file(str(ckpt))
        memres_keys = [k for k in state if any(
            m in k for m in ("memory_block", "memory_readout",
                              "memory_gate", "depth_router")
        )]
        target = self.model.state_dict()
        loaded = 0
        for k in memres_keys:
            if k in target and target[k].shape == state[k].shape:
                target[k] = state[k].to(target[k].dtype).to(target[k].device)
                loaded += 1
        self.model.load_state_dict(target)
        if self.is_main:
            print(f"  warm-started {loaded}/{len(memres_keys)} memres params")

    def _apply_freeze(self) -> None:
        # In simple_gate mode the depth_router is constructed (so checkpoint
        # shapes stay stable across modes) but unused on the forward path,
        # so its parameters never receive gradient and we freeze them
        # explicitly to keep the optimiser / DDP graph tight.
        if self.args.memres_mode == "simple_gate":
            for name, p in self.model.named_parameters():
                if "depth_router" in name:
                    p.requires_grad = False

        # v10: --freeze_backbone zeroes grad on every non-memres param so
        # the optimiser only tracks the MemoryBlock / MemoryReadout /
        # MemoryGate / DepthRouter stack. Forward and backward still run
        # through the backbone (we still need gradient to flow *through*
        # it to reach the memres params), but the backbone's own params
        # don't update. This is the single-GPU-fit recipe for
        # qwen3-8b-xlarge (~700M memres vs ~8B frozen backbone).
        if getattr(self.args, "freeze_backbone", False):
            markers = ("memory_block", "memory_readout",
                       "memory_gate", "depth_router")
            frozen = 0
            trained = 0
            for name, p in self.model.named_parameters():
                if any(m in name for m in markers):
                    trained += p.numel()
                    continue
                p.requires_grad = False
                frozen += p.numel()
            if self.is_main:
                print(
                    f"  freeze_backbone : frozen {frozen/1e6:.1f}M params, "
                    f"training {trained/1e6:.1f}M memres params",
                    flush=True,
                )

    def _wrap_model(self):
        if self.world_size <= 1:
            return self.model
        # ``static_graph=True`` is required for chain TBPTT: the same memres
        # parameters (M_in, M_judge, gates, readout W_*) participate in
        # multiple forward passes inside a single backward, which makes DDP's
        # default ``find_unused_parameters`` machinery throw a "marked as
        # ready twice" error.
        ddp = DDP(
            self.model,
            device_ids=[self.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
        )
        return ddp

    def _model(self) -> Qwen3MemResForCausalLM:
        return self.wrapped.module if hasattr(self.wrapped, "module") else self.wrapped

    # ------------------------------------------------------------------
    # D1: per-module gradient-norm telemetry
    # ------------------------------------------------------------------
    def _build_telemetry_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Bucket every grad-tracked parameter into one mutually-exclusive
        named group.  Group names are diagnostic-stable (logged to wandb
        under ``grad/<group>``) so cross-run comparisons line up.

        Priority order matters: the *first* matching prefix wins.  This
        is checked at init time (assert sum of group sizes == total
        param count).  See _compute_grad_norms for the consumer.
        """
        groups: dict[str, list[torch.nn.Parameter]] = {
            "M_in":       [],
            "extract":    [],
            "M_judge":    [],
            "judge":      [],
            "judge_norm": [],
            "write_gate": [],
            "readout":    [],
            "router":     [],
            "memres_gate": [],
            "lm_head":    [],
            "embed":      [],
            "backbone":   [],
            "other":      [],
        }
        n_total = 0
        for name, p in self._model().named_parameters():
            if not p.requires_grad:
                continue
            n_total += 1
            if "memory_block.M_in" in name and ".extraction_layers" not in name:
                bucket = "M_in"
            elif "memory_block.extraction_layers" in name:
                bucket = "extract"
            elif "memory_block.M_judge" in name:
                bucket = "M_judge"
            elif "memory_block.judging" in name:
                bucket = "judge"
            elif "memory_block.judge_norm" in name:
                bucket = "judge_norm"
            elif "memory_block.write_gate" in name:
                bucket = "write_gate"
            elif "memory_readout" in name:
                bucket = "readout"
            elif "depth_router" in name:
                bucket = "router"
            elif "memory_gate" in name:
                bucket = "memres_gate"
            elif "lm_head" in name:
                bucket = "lm_head"
            elif "embed_tokens" in name:
                bucket = "embed"
            elif (".layers." in name or "model.norm" in name
                  or "rotary_emb" in name):
                bucket = "backbone"
            else:
                bucket = "other"
            groups[bucket].append(p)
        if self.is_main:
            sizes = {k: len(v) for k, v in groups.items() if v}
            n_grouped = sum(sizes.values())
            print(f"  [D1] gradient-group telemetry: "
                  f"{n_grouped}/{n_total} params bucketed "
                  f"(sizes={sizes})", flush=True)
        return groups

    @torch.no_grad()
    def _compute_grad_norms(self) -> dict[str, float]:
        """L2 norm per group, computed AFTER backward and BEFORE
        clip_grad_norm_ so the values reflect the raw learning signal
        the optimiser would see in the absence of global clipping.

        DDP handles all-reduce of grads on the underlying tensors, so
        this is safe to call on any rank; we compute locally on rank 0
        only (called from the main rank's logging branch).
        """
        out: dict[str, float] = {}
        for name, params in self._param_groups_for_telemetry.items():
            sq = 0.0
            for p in params:
                if p.grad is not None:
                    sq += float(p.grad.detach().float().pow(2).sum().item())
            if params:
                out[name] = sq ** 0.5
        return out

    def _build_optimizer(self):
        a = self.args
        markers = ("memory_block", "memory_readout", "depth_router", "memory_gate")
        memres_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(m in name for m in markers):
                memres_params.append(p)
            else:
                backbone_params.append(p)
        groups = []
        if memres_params:
            groups.append({"params": memres_params, "lr": a.lr, "name": "memres"})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": a.lr_backbone,
                           "name": "backbone"})
        if self.is_main:
            print(
                f"  param groups    : memres={sum(p.numel() for p in memres_params)/1e6:.1f}M "
                f"@ {a.lr}, backbone={sum(p.numel() for p in backbone_params)/1e6:.1f}M "
                f"@ {a.lr_backbone}"
            )
        opt_cls = AdamW
        if getattr(a, "use_adam8bit", False):
            try:
                import bitsandbytes as bnb
                opt_cls = bnb.optim.AdamW8bit
                if self.is_main:
                    print(
                        f"  optimizer       : bitsandbytes AdamW8bit "
                        f"(v{bnb.__version__})",
                        flush=True,
                    )
            except Exception as e:
                if self.is_main:
                    print(
                        f"  [warn] --use_adam8bit set but bitsandbytes "
                        f"unavailable ({e}); falling back to torch AdamW",
                        flush=True,
                    )
        opt = opt_cls(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.weight_decay)
        sched = LambdaLR(
            opt,
            lr_lambda=lambda s: cosine_with_warmup(s, a.warmup, a.steps, a.lr_min_ratio),
        )
        return opt, sched

    # ------------------------------------------------------------------
    # Recurrent training step
    # ------------------------------------------------------------------

    def _make_attn_mask(self, seq_len: int, cutoff: int) -> dict | None:
        if cutoff <= 0:
            return None
        neg_inf = torch.finfo(torch.bfloat16).min
        mask = torch.zeros(1, 1, seq_len, seq_len, device=self.device,
                           dtype=torch.bfloat16)
        upper = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device), diagonal=1
        ).bool()
        mask[:, :, upper] = neg_inf
        mask[:, :, cutoff:, :cutoff] = neg_inf
        return {"full_attention": mask}

    def _weighted_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        callback_mask: torch.Tensor,
        callback_loss_weight: float,
    ) -> torch.Tensor:
        """Per-position weighted causal LM NLL.

        - logits:        (B, S, V), unshifted (matches HF Qwen3 forward).
        - labels:        (B, S), values in [0, V) or -100 for ignore (built by
                         _build_labels with --mask_padding_loss / --score_tail_frac).
        - callback_mask: (B, S), 0/1 marking callback-span tokens.
        - callback_loss_weight: scalar; final per-token weight is
                                (1 + callback_loss_weight * mask).

        We do the standard HF causal-LM shift (predict tokens [1..S) from
        positions [0..S-1)) and reduce as a *weighted mean* over valid
        positions, so the loss magnitude stays comparable to the legacy
        uniform-NLL even when callback positions are sparse.
        """
        shift_logits = logits[..., :-1, :].contiguous()      # (B, S-1, V)
        shift_labels = labels[..., 1:].contiguous()          # (B, S-1)
        shift_mask = callback_mask[..., 1:].contiguous().to(shift_logits.dtype)
        nll = F.cross_entropy(
            shift_logits.flatten(0, 1),
            shift_labels.flatten(),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)                              # (B, S-1)
        valid = (shift_labels != -100).to(shift_logits.dtype)
        weights = 1.0 + callback_loss_weight * shift_mask
        weighted = nll * weights * valid
        denom = (weights * valid).sum().clamp_min(1.0)
        return weighted.sum() / denom

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Construct loss labels honouring --mask_padding_loss / --score_tail_frac.

        Logic:
        - If neither knob is set, labels == input_ids (legacy behaviour).
        - With --mask_padding_loss, every position where input_ids == eos_id
          is set to -100 so it does not contribute to the LM loss.  This
          assumes EOS only appears as trailing right-padding (true for
          pretokenize_chains.py output).
        - With --score_tail_frac F < 1.0, the first (1-F) fraction of each
          row's *non-padding* content is also masked to -100, so only the
          tail F contributes to loss.  Computed per-row from the row's
          actual content length.
        """
        a = self.args
        if not a.mask_padding_loss and a.score_tail_frac >= 1.0:
            return input_ids
        labels = input_ids.clone()
        is_pad = input_ids == self.eos_id  # (B, S) bool
        if a.mask_padding_loss:
            labels = labels.masked_fill(is_pad, -100)
        if a.score_tail_frac < 1.0:
            # content_len[bi] = number of non-padding tokens in row bi.
            # When mask_padding_loss is also set, this matches the number
            # of positions still contributing to loss after masking.
            # When it's not set, we still use the same definition so the
            # tail fraction is over real content (not over padded length).
            content_len = (~is_pad).long().sum(dim=1)
            B = input_ids.shape[0]
            for bi in range(B):
                cl = int(content_len[bi].item())
                if cl <= 0:
                    continue
                head_cut = int(cl * (1.0 - a.score_tail_frac))
                if head_cut > 0:
                    labels[bi, :head_cut] = -100
        return labels

    def _train_step(self, rng: random.Random,
                    carry: torch.Tensor | None) -> tuple[float, torch.Tensor | None]:
        a = self.args
        # Sample one chain window per micro-batch element.  Cap the burn-in
        # length at ``--burn_in_max`` for memory; longer prefixes are simply
        # tail-truncated (we keep the most recent burn_in_max sessions).
        windows = []
        callback_masks: list[torch.Tensor] = []
        burn_ins: list[torch.Tensor | None] = []
        # Long-horizon training protocol: when --burn_in_resample is set, draw
        # a per-chain burn-in length from {0, 4, 8, ..., burn_in_max} so the
        # judge step sees M_c at a range of recurrence depths each batch.
        # Otherwise, behave as before: cap at burn_in_max, full prefix.
        for _ in range(a.batch_size):
            ci, st, w, b, cb_mask = self.train_sampler.sample_window()
            if a.burn_in_resample and a.burn_in_max > 0 and b is not None:
                step = max(4, a.burn_in_max // 5)
                choices = [0] + list(range(step, a.burn_in_max + 1, step))
                pick = rng.choice(choices)
                if pick == 0 or b.shape[0] == 0:
                    b = None
                else:
                    b = b[-pick:]
            elif b is not None and a.burn_in_max > 0 and b.shape[0] > a.burn_in_max:
                b = b[-a.burn_in_max:]
            elif a.burn_in_max == 0:
                b = None
            windows.append(w)
            # For curriculum windows the sampler hands back a precomputed
            # mask aligned to the non-contiguous slice; for legacy contiguous
            # windows we slice it ourselves from (st, window_k).
            if cb_mask is None:
                cb_mask = self.train_corpus.chain_window_callback_mask(
                    ci, st, a.window_k
                )
            callback_masks.append(cb_mask)
            burn_ins.append(b)
        # (B, k, S)
        window = torch.stack(windows).to(self.device)
        # (B, k, S) int8 with 1's at callback supervision positions.
        callback_mask_window = torch.stack(callback_masks).to(self.device)

        model = self._model()
        cfg = model.config
        K, d = cfg.memres_num_vectors, cfg.hidden_size
        B = window.shape[0]

        # Initial memory: zero tensor (NOT None) so the graph topology is
        # constant across iterations -- DDP static_graph=True requires this.
        if carry is not None and a.carry_state:
            M_c = carry.detach()
        else:
            M_c = torch.zeros(B, K, d, device=self.device, dtype=torch.bfloat16)

        # Snapshot M_c at the START of the TBPTT window (after burn-in,
        # after carry-state).  The intra-chain contrastive loss
        # re-builds a perturbed M_c starting from this exact state, so
        # positive and perturbed only differ in one swapped session.
        # Captured here before the main loop mutates M_c.
        M_c_window_start: torch.Tensor | None = None

        # Burn-in: process each batch element's prefix under no_grad, building
        # a per-element M_c at the right recurrence depth, then stack.
        # Non-uniform burn-in lengths require we process each element separately.
        if any(b is not None for b in burn_ins):
            with torch.no_grad():
                burn_M = []
                for bi in range(B):
                    m = M_c[bi : bi + 1]
                    bp = burn_ins[bi]
                    if bp is not None:
                        bp = bp.to(self.device)
                        for j in range(bp.shape[0]):
                            sess_j = bp[j].unsqueeze(0)
                            C_j = model.model.extract_source(sess_j[:, :-1])
                            m = model.model.compress_session(C_j, m)
                    burn_M.append(m)
                M_c = torch.cat(burn_M, dim=0).detach()
        M_c_window_start = M_c.detach().clone()

        # Pre-compute a trivial causal mask we can always pass; we'll
        # optionally OR-in a context-dropout block.  The MemRes backbone
        # accepts a dict {"full_attention": (1, 1, S, S)} additive bias.
        seq = window.shape[2] - 1   # we feed [:, :-1]
        neg_inf = torch.finfo(torch.bfloat16).min
        causal_only = torch.zeros(
            1, 1, seq, seq, device=self.device, dtype=torch.bfloat16
        )
        upper = torch.triu(
            torch.ones(seq, seq, device=self.device), diagonal=1
        ).bool()
        causal_only[:, :, upper] = neg_inf
        causal_mask_dict = {"full_attention": causal_only}

        losses = []
        n_drop_mem = 0
        n_drop_ctx = 0
        last_input_ids = None
        last_labels = None
        last_M_c_pre = None  # M_c BEFORE the last session's update, for contrast.
        last_cb_mask: torch.Tensor | None = None  # last session's CB mask (B, S-1).
        # Pre-update M_c snapshot for the InfoNCE contrastive forward.
        # Distinct from last_M_c_pre (which carries memory-dropout
        # zero-outs); we want the *un-dropped* M_c here so the
        # diagonal of the B*B contrastive matrix is informative.
        M_c_for_contrast: torch.Tensor | None = None
        # v14 alpha_mem floor aux loss: we collect per-layer alpha_mem
        # from every forward that has non-zero memory (drop_mem=False)
        # and compute a per-sublayer mean-over-positions floor
        # penalty.  Keeping the tensors (with grad) in this list and
        # reducing at the bottom of the loss-construction block keeps
        # the forward graph small.
        alpha_floor_traces: list[list[torch.Tensor]] = []
        want_alpha_floor = a.alpha_mem_floor_aux_weight > 0.0
        for t in range(a.window_k):
            session = window[:, t]                         # (B, S)
            input_ids = session[:, :-1]
            labels = self._build_labels(input_ids)
            # Per-session callback mask (B, S-1) aligned with input_ids.
            cb_mask_t = callback_mask_window[:, t, :-1]

            # Memory dropout: zero out M_c (don't change topology).
            drop_mem = rng.random() < a.memory_dropout
            if drop_mem:
                read_M = torch.zeros_like(M_c)
                n_drop_mem += 1
            else:
                read_M = M_c

            # Context dropout: always pass a mask dict; sometimes block prefix.
            drop_ctx = rng.random() < a.context_dropout
            if drop_ctx and seq >= 4:
                cutoff = rng.randint(1, seq // 2)
                m = causal_only.clone()
                m[:, :, cutoff:, :cutoff] = neg_inf
                attn_mask = {"full_attention": m}
                n_drop_ctx += 1
            else:
                attn_mask = causal_mask_dict

            # When callback supervision is on AND this session has any
            # callback tokens, compute the loss manually so we can
            # multiply the NLL on callback positions by (1 + weight).
            # Otherwise, fall back to the model's default reduction.
            use_cb_loss = (
                a.callback_loss_weight > 0.0
                and bool(cb_mask_t.any().item())
            )
            # v14: collect alpha_trace only on forwards that carried
            # memory (drop_mem=False); on dropped-memory forwards the
            # router has no memory source to route to so the floor is
            # not well-defined.
            collect_alpha = want_alpha_floor and not drop_mem
            if use_cb_loss:
                out = self.wrapped(
                    input_ids=input_ids,
                    labels=None,
                    M_c=read_M,
                    attention_mask=attn_mask,
                    collect_alpha_trace=collect_alpha,
                )
                logits = out.logits  # (B, S-1, V)
                loss_t = self._weighted_lm_loss(
                    logits, labels, cb_mask_t, a.callback_loss_weight,
                )
                losses.append(loss_t)
            else:
                out = self.wrapped(
                    input_ids=input_ids,
                    labels=labels,
                    M_c=read_M,
                    attention_mask=attn_mask,
                    collect_alpha_trace=collect_alpha,
                )
                losses.append(out.loss)
            if collect_alpha:
                tr = getattr(out, "alpha_trace", None)
                if tr is not None and len(tr) > 0:
                    alpha_floor_traces.append(tr)

            # Snapshot M_c right before the *last* session's update so we can
            # use it (and a paired shuffle M_c) for the negative-chain loss.
            if t == a.window_k - 1:
                last_input_ids = input_ids
                last_labels = labels
                last_M_c_pre = read_M  # what the loss above conditioned on
                last_cb_mask = cb_mask_t
                # Un-dropped M_c (reference, not clone -- keeps grad).
                # Used by the InfoNCE contrastive forward below.
                M_c_for_contrast = M_c

            # Recurrent memory update: M_c <- judge(M_c, extract(C_t)).
            # extract_source honours config.memres_extract_source: 'embed'
            # for legacy bag-of-token-embeddings, 'hidden_<L>' for the
            # contextualised mid-layer hidden state path.
            C_t = model.model.extract_source(input_ids)
            M_c = model.model.compress_session(C_t, M_c)

        loss_match = torch.stack(losses).mean()
        total_loss = loss_match

        # v14 alpha_mem floor auxiliary loss.  MoE-style load-balance:
        # penalise per-sublayer mean(alpha_mem) dropping below
        # ``alpha_mem_floor_target``.  The router is then obligated to
        # keep sampling the memory source even when local features
        # would dominate -- giving the writer / readout a persistent
        # downstream gradient signal during joint training.  Motivated
        # by v13r ROUTE @ step 10000: alpha_mem_mean=0.0011, top l5=
        # 0.0136 (router collapsed to effectively ignoring memory),
        # which is textbook MoE expert collapse and the standard fix
        # is a load-balance auxiliary loss (Switch Transformer, Fedus
        # et al. 2021).  Zero-cost when weight=0 because the
        # alpha_floor_traces list stays empty.
        if want_alpha_floor and alpha_floor_traces:
            target = float(a.alpha_mem_floor_target)
            aux_terms = []
            for trace in alpha_floor_traces:
                for a_mem in trace:
                    # a_mem: (B, S) depth-wise attention mass on b_{-1}.
                    # Mean over batch & position for this sublayer.
                    m = a_mem.float().mean()
                    # relu(target - m): penalises only the downward
                    # violation; when mean(alpha_mem) >= target the aux
                    # gradient is zero and the main loss is unimpeded.
                    aux_terms.append(F.relu(target - m))
            loss_alpha_floor = torch.stack(aux_terms).mean()
            total_loss = total_loss + a.alpha_mem_floor_aux_weight * loss_alpha_floor
            self._last_alpha_floor_loss = float(loss_alpha_floor.detach().item())
            with torch.no_grad():
                all_means = [
                    a_mem.float().mean().detach().item()
                    for trace in alpha_floor_traces for a_mem in trace
                ]
                self._last_alpha_mem_batch_mean = (
                    sum(all_means) / max(1, len(all_means))
                )
        else:
            self._last_alpha_floor_loss = 0.0
            self._last_alpha_mem_batch_mean = 0.0

        # Negative-chain contrastive auxiliary loss (PITFALLS §3).  Build a
        # shuffle M_c from a randomly-chosen *different* chain (any chain,
        # any random window) and score the same matched-chain last session
        # under it.  We push the matched loss to be at least `margin` below
        # the shuffle loss; otherwise we add the gap to the total loss.
        if a.neg_chain_warmup_steps > 0:
            init_w = a.neg_chain_initial_weight
            if init_w is None:
                init_w = a.neg_chain_weight
            ramp = min(1.0, self.global_step / max(1, a.neg_chain_warmup_steps))
            cur_neg_weight = init_w + (a.neg_chain_weight - init_w) * ramp
        else:
            cur_neg_weight = a.neg_chain_weight
        if cur_neg_weight > 0.0:
            with torch.no_grad():
                shuffle_windows = []
                for _ in range(a.batch_size):
                    _, _, sw, sb, _ = self.train_sampler.sample_window()
                    if sb is not None and a.burn_in_max > 0 and sb.shape[0] > a.burn_in_max:
                        sb = sb[-a.burn_in_max:]
                    elif a.burn_in_max == 0:
                        sb = None
                    shuffle_windows.append((sw, sb))
                # Build shuffle M_c per batch element.
                M_sh_list = []
                for bi, (sw, sb) in enumerate(shuffle_windows):
                    m = torch.zeros(1, K, d, device=self.device, dtype=torch.bfloat16)
                    if sb is not None:
                        sb = sb.to(self.device)
                        for j in range(sb.shape[0]):
                            sj = sb[j].unsqueeze(0)
                            Cj = model.model.extract_source(sj[:, :-1])
                            m = model.model.compress_session(Cj, m)
                    sw = sw.to(self.device)
                    for j in range(sw.shape[0]):
                        sj = sw[j].unsqueeze(0)
                        Cj = model.model.extract_source(sj[:, :-1])
                        m = model.model.compress_session(Cj, m)
                    M_sh_list.append(m)
                M_sh = torch.cat(M_sh_list, 0).detach()
            # Score the matched chain's last session under M_sh.  This loss
            # SHOULD be larger than loss_match -- if it isn't, memory is not
            # chain-specific and we get a positive contrastive penalty.
            out_sh = self.wrapped(
                input_ids=last_input_ids,
                labels=last_labels,
                M_c=M_sh,
                attention_mask=causal_mask_dict,
            )
            loss_shuffle = out_sh.loss
            margin_loss = (loss_match - loss_shuffle + a.neg_chain_margin).clamp(min=0.0)
            total_loss = total_loss + cur_neg_weight * margin_loss

        # ------------------------------------------------------------
        # Intra-chain perturbation contrastive loss (the "did the fact
        # survive interference?" loss).
        #
        # Build a perturbed M_c by re-running TBPTT through the same
        # window with ONE earlier session swapped for a random
        # other-chain session.  Score the matched chain's last session
        # under both positive and perturbed M_c; require positive to be
        # at least `margin` lower in NLL.
        #
        # Gradient through the perturbed-build pressures the judge step
        # at every intermediate session to PRESERVE channels carrying
        # information that turns out to matter at the recall position
        # -- the QKV competition is forced to defend M_c_prev when
        # M_new (current session) does not contain the salient content.
        #
        # Cost: one extra TBPTT chain build (with grad) + one extra
        # forward+backward on the recall session.  ~30%% per-step
        # overhead at window_k=3.
        if a.in_chain_contrast_warmup_steps > 0:
            ic_init = a.in_chain_contrast_initial_weight
            if ic_init is None:
                ic_init = a.in_chain_contrast_weight
            ic_ramp = min(
                1.0,
                self.global_step / max(1, a.in_chain_contrast_warmup_steps),
            )
            cur_ic_weight = ic_init + (a.in_chain_contrast_weight - ic_init) * ic_ramp
        else:
            cur_ic_weight = a.in_chain_contrast_weight
        if cur_ic_weight > 0.0 and a.window_k >= 2 and M_c_window_start is not None:
            # 1. Pick the perturbation slot.  random_earlier rotates the
            #    pressure across all earlier slots so the model can't
            #    learn to "skip" a fixed position.
            #
            # NB: the slot index is named ``slot_idx`` (was ``F``); the
            # latter shadowed the module-level ``import torch.nn.functional
            # as F`` and silently broke any subsequent F.cross_entropy /
            # F.softmax call inside _train_step.  The InfoNCE block below
            # tripped this on the very first run.
            if a.in_chain_perturb_strategy == "session_zero":
                slot_idx = 0
            else:  # random_earlier
                slot_idx = rng.randint(0, max(0, a.window_k - 2))

            # 2. Sample a distractor session per batch element by
            #    drawing a random other-chain window and taking its
            #    session 0.  no_grad: just data plumbing, no params.
            with torch.no_grad():
                distractor_list = []
                for _ in range(B):
                    _, _, dw, _, _ = self.train_sampler.sample_window()
                    distractor_list.append(dw[0])
                distractor_slot = torch.stack(distractor_list).to(self.device)

            # 3. Re-build M_c through the window with session slot_idx swapped.
            #    WITH grad -- this is what carries the recall signal
            #    backwards into the intermediate judge / extract steps.
            M_c_pert = M_c_window_start.clone()
            for t in range(a.window_k - 1):
                if t == slot_idx:
                    sess_t = distractor_slot
                else:
                    sess_t = window[:, t]
                ids_t = sess_t[:, :-1]
                C_t_pert = model.model.extract_source(ids_t)
                M_c_pert = model.model.compress_session(C_t_pert, M_c_pert)

            # 4. Score the recall session under the perturbed M_c.
            out_pert = self.wrapped(
                input_ids=last_input_ids,
                labels=last_labels,
                M_c=M_c_pert,
                attention_mask=causal_mask_dict,
            )
            loss_pert = out_pert.loss

            # 5. Margin loss on the recall position only.  We use
            #    losses[-1] (the last-session loss with the same
            #    dropout state as the recall forward) so positive and
            #    perturbed are compared like-for-like.
            loss_recall = losses[-1]
            ic_margin = (
                loss_recall - loss_pert + a.in_chain_contrast_margin
            ).clamp(min=0.0)
            total_loss = total_loss + cur_ic_weight * ic_margin

        # ------------------------------------------------------------
        # Multi-negative InfoNCE contrastive loss.
        #
        # For each batch element i, score the last session i under
        # every batch element j's M_c, then cross-entropy with
        # diagonal=positive.  Three things happen via gradient:
        #
        # 1) L[i,i] decreases: readout uses chain i's own M_c well.
        # 2) L[i,j] increases for j != i: readout cannot fit chain
        #    i's last session well using chain j's M_c.
        # 3) Gradient through M_c[j] (j != i) pushes M_c[j] AWAY
        #    from chain i's last-session content.
        #
        # Direct attack on Δ_sh-m ≈ 0: we _force_ the readout to be
        # chain-discriminative on callback tokens by making
        # cross-chain confusion an explicit loss.
        if a.contrastive_infonce_warmup_steps > 0:
            nce_init = a.contrastive_infonce_initial_weight
            if nce_init is None:
                nce_init = a.contrastive_infonce_weight
            nce_ramp = min(
                1.0,
                self.global_step / max(1, a.contrastive_infonce_warmup_steps),
            )
            cur_nce_weight = nce_init + (a.contrastive_infonce_weight - nce_init) * nce_ramp
        else:
            cur_nce_weight = a.contrastive_infonce_weight

        if (
            cur_nce_weight > 0.0
            and B >= 2
            and last_input_ids is not None
            and last_labels is not None
            and last_cb_mask is not None
            and M_c_for_contrast is not None
        ):
            # Cross product: row b*B + j is (chain b's input, chain j's M_c).
            # repeat_interleave on inputs (each input row repeated B times)
            # paired with repeat (cycles M_c B times) yields the (i, j)
            # cross product in row-major order.
            inputs_xy = last_input_ids.repeat_interleave(B, dim=0)        # (B*B, S-1)
            labels_xy = last_labels.repeat_interleave(B, dim=0)            # (B*B, S-1)
            cb_xy = last_cb_mask.repeat_interleave(B, dim=0)               # (B*B, S-1)
            M_c_xy = M_c_for_contrast.repeat(B, 1, 1)                      # (B*B, K, d)

            out_nce = self.wrapped(
                input_ids=inputs_xy,
                labels=None,
                M_c=M_c_xy,
                attention_mask=causal_mask_dict,
            )
            logits_nce = out_nce.logits

            # Per-token NLL with the standard HF causal-LM shift.
            shift_logits = logits_nce[..., :-1, :].contiguous()      # (B*B, S-2, V)
            shift_labels = labels_xy[..., 1:].contiguous()            # (B*B, S-2)
            shift_cb = cb_xy[..., 1:].contiguous().to(shift_logits.dtype)
            nll_per_pos = F.cross_entropy(
                shift_logits.flatten(0, 1),
                shift_labels.flatten(),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)                                  # (B*B, S-2)
            valid = (shift_labels != -100).to(shift_logits.dtype)

            # Per-pair score: mean NLL on callback tokens (when present
            # and callback_only is set), else mean NLL on all valid
            # tokens.  Callback-only concentrates the contrastive
            # gradient on tokens that actually require memory.
            cb_mass = (shift_cb * valid).sum(dim=1)                  # (B*B,)
            full_denom = valid.sum(dim=1).clamp_min(1.0)
            full_mean = (nll_per_pos * valid).sum(dim=1) / full_denom
            if a.contrastive_infonce_callback_only:
                cb_denom = cb_mass.clamp_min(1.0)
                cb_mean = (nll_per_pos * shift_cb * valid).sum(dim=1) / cb_denom
                use_cb = cb_mass > 0
                score_nll = torch.where(use_cb, cb_mean, full_mean)
            else:
                score_nll = full_mean

            # Reshape into (B, B): row i = input chain i, col j = M_c chain j.
            L = score_nll.view(B, B)                                 # (B, B)
            T_nce = max(1e-3, float(a.contrastive_infonce_temperature))
            nce_logits = -L / T_nce                                  # higher = lower NLL = better fit
            targets = torch.arange(B, device=L.device)
            loss_nce = F.cross_entropy(nce_logits, targets, reduction="mean")
            total_loss = total_loss + cur_nce_weight * loss_nce

            # Telemetry: diag (positive NLL), off-diag (negative NLL),
            # gap (off - diag, want positive and growing over training).
            with torch.no_grad():
                diag = L.diagonal()
                off = L.sum() - diag.sum()
                n_off = max(1, B * B - B)
                self._last_contrastive_loss = float(loss_nce.detach().item())
                self._last_contrastive_diag = float(diag.mean().item())
                self._last_contrastive_offdiag = float(off.item() / n_off)
                self._last_contrastive_gap = (
                    self._last_contrastive_offdiag - self._last_contrastive_diag
                )
        else:
            self._last_contrastive_loss = 0.0
            self._last_contrastive_diag = 0.0
            self._last_contrastive_offdiag = 0.0
            self._last_contrastive_gap = 0.0

        total_loss = total_loss / a.grad_accum
        total_loss.backward()
        return float(loss_match.item()), M_c

    # ------------------------------------------------------------------
    # Recurrent evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _per_position_nll(
        self,
        model,
        input_ids: torch.Tensor,
        M_c: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return per-position next-token NLL of shape (S-1,).

        Uses fp32 log-softmax so that small CE differences (the regime
        we care about for callback-token Δ_sh-m on a few-thousand-vocab
        slice) are numerically meaningful.
        """
        out = model(input_ids=input_ids, M_c=M_c)
        logits = out.logits  # (1, S, V)
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return nll.squeeze(0)  # (S-1,)

    @torch.no_grad()
    def _phase_aligned_eval(self) -> dict:
        """Eval that matches the curriculum training distribution.

        For every chain that has an annotated callback position
        ``cb_pos >= 1`` (LongMemEval-style), pick one random evidence
        position ``e`` in ``[0, cb_pos)``, build M_c from session ``e``
        only (fresh start, single judge step -- exactly what the P0
        curriculum trains on), then score session ``cb_pos`` under
        three M_c regimes:

          - mem:     M_c built from this chain's evidence session.
          - nomem:   M_c = None.
          - shuffle: M_c built from a *different* chain's evidence
                     session (chosen uniformly from the eligible pool).

        Reduce per-position NLL two ways:

          - whole-session ("ws_*"): mean NLL over non-padding tokens of
            the callback session. The legacy reduction; dominated by
            filler tokens and therefore noisy.
          - callback-only ("cb_*"): mean NLL only over tokens flagged
            in ``session_callback_mask`` (the answer span). This is
            the strongest single signal that the readout is content-
            discriminative on the tokens that actually require memory
            to predict.

        Why this is the right "is the model learning?" lens:

          - Same M_c distribution as training (1 fresh evidence).
          - Same per-token weighting in spirit as
            ``--callback_loss_weight``.
          - Decoupled from the standard-eval distribution mismatch
            that makes Δ_nm-m look catastrophic during P0 training.

        Returns a dict with keys prefixed ``pa_`` so they can sit
        alongside the legacy standard-eval keys without collision.
        """
        a = self.args
        model = self._model()
        ev = self.eval_corpus
        rng = random.Random(a.seed + self.global_step + 17)

        eligible = [
            i for i in range(len(ev))
            if int(ev.chain_callback_position[i]) >= 1
        ]
        if len(eligible) < 2:
            return {"n_pa_scored": 0}
        rng.shuffle(eligible)
        eligible_for_score = eligible[: a.phase_aligned_eval_n_chains]

        ws_mem, ws_no, ws_sh = [], [], []
        cb_mem, cb_no, cb_sh = [], [], []
        # v11: parallel accumulators for the "no-evidence" floor.
        # When the eval chain has annotated evidence positions, we run a
        # SECOND scoring pass with M_c built from a non-evidence haystack
        # session; we should expect this M_c to NOT help on callback
        # tokens (it doesn't contain the answer). The gap between the
        # two passes ("evidence-aware" minus "evidence-absent") is the
        # honest signal that memory is doing something content-specific
        # rather than learning a generic style prior. Pre-v11 reports
        # mixed both passes together because evidence was sampled
        # uniformly, so a chain with 3.6% evidence prior produced 96%
        # noise + 4% signal averaged into one number. See README P3.
        cb_mem_floor, cb_no_floor = [], []
        n_with_ev_label = 0
        for ci in eligible_for_score:
            cb_pos = int(ev.chain_callback_position[ci])
            ev_positions = ev.chain_evidence_positions[ci]
            ev_in_range = [p for p in ev_positions if 0 <= p < cb_pos]
            if ev_in_range:
                e = rng.choice(ev_in_range)
                n_with_ev_label += 1
            else:
                # No annotated evidence -> uniform fallback (legacy
                # behaviour for non-LME chains).
                e = rng.randint(0, cb_pos - 1)
            evidence = ev.chain_session_at(ci, e).to(self.device).unsqueeze(0)
            callback = ev.chain_session_at(ci, cb_pos).to(self.device).unsqueeze(0)
            cb_mask = ev.session_callback_mask[
                int(ev.chain_starts[ci]) + cb_pos
            ].to(self.device)  # (S,)

            # mem: M_c from this chain's evidence (fresh, single judge step).
            C_e = model.model.extract_source(evidence[:, :-1])
            M_c = model.model.compress_session(C_e, None)

            # shuffle: M_c from a different chain's evidence.
            others = [j for j in eligible if j != ci]
            if not others:
                M_sh = None
            else:
                other_idx = rng.choice(others)
                o_cb_pos = int(ev.chain_callback_position[other_idx])
                o_e = rng.randint(0, o_cb_pos - 1)
                o_evidence = (
                    ev.chain_session_at(other_idx, o_e)
                    .to(self.device)
                    .unsqueeze(0)
                )
                C_o = model.model.extract_source(o_evidence[:, :-1])
                M_sh = model.model.compress_session(C_o, None)

            input_ids = callback[:, :-1]
            valid = (input_ids[0, :] != self.eos_id).float()
            cb_mask_in = cb_mask[: input_ids.shape[1]].float() * valid
            valid_sh = valid[1:]
            cb_mask_sh = cb_mask_in[1:]

            nll_mem = self._per_position_nll(model, input_ids, M_c)
            nll_no = self._per_position_nll(model, input_ids, None)
            if M_sh is not None:
                nll_sh = self._per_position_nll(model, input_ids, M_sh)
            else:
                nll_sh = None

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

            # v11: Evidence-absent floor. For chains that have annotated
            # evidence positions, ALSO score the callback under M_c built
            # from a haystack session that is *not* in evidence_positions.
            # If the readout is content-specific the gap (cb_no_floor -
            # cb_mem_floor) should be ~0 (memory built from irrelevant
            # context shouldn't help), while (cb_no - cb_mem) > 0 means
            # memory built from real evidence does help. The DIFFERENCE
            # of differences -- pa_cb_evidence_lift = (cb_no - cb_mem) -
            # (cb_no_floor - cb_mem_floor) -- is the strongest single
            # diagnostic that the channel is episodic and not generic.
            if ev_in_range and cb_sum > 0:
                non_ev_pre = [
                    p for p in range(0, cb_pos)
                    if p not in ev_in_range
                ]
                if non_ev_pre:
                    e_floor = rng.choice(non_ev_pre)
                    floor_evidence = (
                        ev.chain_session_at(ci, e_floor)
                        .to(self.device).unsqueeze(0)
                    )
                    C_f = model.model.extract_source(floor_evidence[:, :-1])
                    M_floor = model.model.compress_session(C_f, None)
                    nll_floor = self._per_position_nll(model, input_ids, M_floor)
                    cb_mem_floor.append(
                        float((nll_floor * cb_mask_sh).sum().item()) / cb_sum
                    )
                    cb_no_floor.append(
                        float((nll_no * cb_mask_sh).sum().item()) / cb_sum
                    )

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
        # v11 evidence-aware diagnostics (only meaningful when the eval
        # corpus carries chain_evidence_positions, e.g. the v11 LME
        # corpus). On pre-v11 corpora these are NaN.
        out["n_pa_cb_evidence_labelled"] = n_with_ev_label
        out["pa_cb_ce_mem_floor"] = m(cb_mem_floor)
        out["pa_cb_ce_nomem_floor"] = m(cb_no_floor)
        if cb_mem_floor and cb_no_floor:
            floor_dnm = out["pa_cb_ce_nomem_floor"] - out["pa_cb_ce_mem_floor"]
            out["pa_cb_dnm_floor"] = floor_dnm
            # Lift = (memory benefit when evidence is present) MINUS
            # (memory benefit when evidence is absent). > 0 means the
            # readout is content-specific to evidence-bearing M_c.
            out["pa_cb_evidence_lift"] = out["pa_cb_dnm"] - floor_dnm
        else:
            out["pa_cb_dnm_floor"] = float("nan")
            out["pa_cb_evidence_lift"] = float("nan")
        return out

    @torch.no_grad()
    def _routing_recruitment_summary(self) -> dict:
        """Routing-mode-aware per-sublayer recruitment signal.

        For ``simple_gate`` mode: snapshot the per-sublayer scalar gate
        directly (no forward needed). The gate values are *the* signal
        for which sublayers are recruiting memory.

        For ``attention_parity`` / ``attention_base``: run a small
        eval batch (a real callback session with a real evidence M_c)
        with ``collect_alpha_trace=True`` and report per-sublayer
        ``alpha_mem`` (the mass the depth router puts on the memory
        source). The bias param alone (mem_bias) does NOT report
        recruitment because the actual α depends on the pseudo-query
        × normalised value alignment.
        """
        a = self.args
        model = self._model()
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

        # attention_parity / attention_base path.
        ev = self.eval_corpus
        eligible = [
            i for i in range(len(ev))
            if int(ev.chain_callback_position[i]) >= 1
        ][: a.diag_routing_n_chains]
        if not eligible:
            return {"rec_mode": mode, "rec_alpha_mem_max": 0.0}

        per_sublayer_acc: list[list[float]] = []
        for ci in eligible:
            cb_pos = int(ev.chain_callback_position[ci])
            e = max(0, cb_pos - 1)
            evidence = (
                ev.chain_session_at(ci, e).to(self.device).unsqueeze(0)
            )
            callback = (
                ev.chain_session_at(ci, cb_pos).to(self.device).unsqueeze(0)
            )
            C_e = model.model.extract_source(evidence[:, :-1])
            M_c = model.model.compress_session(C_e, None)
            out = model(
                input_ids=callback[:, :-1],
                M_c=M_c,
                collect_alpha_trace=True,
            )
            trace = getattr(out, "alpha_trace", None)
            if not trace:
                continue
            per_sublayer_acc.append([float(a_t.float().mean().item()) for a_t in trace])

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
            # "Open" sublayer = α_mem above 5% (well above the
            # uniform-prior floor of 1/(N+2) ~ 10%).
            "rec_frac_open": float((arr > 0.05).float().mean()),
        }

    @torch.no_grad()
    def _readout_magnitude_diag(self) -> dict:
        """Pulse check: is the readout m^t even non-trivially scaled?

        Computes ``mean(||m^t||) / mean(||embed||)`` on the phase-
        aligned eval setup (fresh M_c from one evidence session,
        readout queried by the callback session embeddings).

        - A ratio of ~0 means the V projection has collapsed (or the
          M_c slots have collapsed onto a near-zero point).  Any
          gate / α opening downstream is moot.
        - A ratio of ~1 means m^t is comparable in magnitude to the
          token embeddings, which is the regime where it can move
          downstream logits non-trivially.
        - Initial-parity zero-init of W_V^read in MemoryReadout was
          REMOVED in the live code (default normal init), so a non-
          trivial ratio is expected from step 1.
        """
        a = self.args
        model = self._model()
        ev = self.eval_corpus
        eligible = [
            i for i in range(len(ev))
            if int(ev.chain_callback_position[i]) >= 1
        ][: a.diag_routing_n_chains]
        if not eligible:
            return {"mt_norm_ratio_mean": float("nan")}
        ratios = []
        for ci in eligible:
            cb_pos = int(ev.chain_callback_position[ci])
            e = max(0, cb_pos - 1)
            evidence = (
                ev.chain_session_at(ci, e).to(self.device).unsqueeze(0)
            )
            callback = (
                ev.chain_session_at(ci, cb_pos).to(self.device).unsqueeze(0)
            )
            C_e = model.model.extract_source(evidence[:, :-1])
            M_c = model.model.compress_session(C_e, None)
            X = model.model.embed_tokens(callback[:, :-1])
            m_t = model.model.memory_readout(X, M_c)
            mt_norm = m_t.float().norm(dim=-1).mean().item()
            h_norm = X.float().norm(dim=-1).mean().item()
            ratios.append(mt_norm / max(h_norm, 1e-6))
        return {
            "mt_norm_ratio_mean": float(sum(ratios) / len(ratios)),
            "mt_norm_ratio_max": float(max(ratios)),
        }

    # ------------------------------------------------------------------
    # D2 / D3: memory-dynamics diagnostic
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _memory_dynamics_eval(self) -> dict:
        """Single-pass audit of the writer over a few held-out chains.

        Returns a flat dict that goes through the same ``eval/<key>``
        wandb namespace as everything else.  Cost: one extract_source +
        one judge call per (chain, session); no LM forward, no readout.

        Metrics produced:

          D2 (judge decisiveness):
            judge_row_entropy_mean  -- mean over (chains, sessions, K
                rows) of -sum(p log p) of the judge softmax.  Floor:
                0 (one-hot judge); ceiling: log(2K) (uniform; the
                judge made no decision).
            judge_keep_mass_mean    -- fraction of attention weight on
                the [M_c^{t-1}] half across (chains, sessions, rows).
                0.5 == prior and candidate are interchangeable; far
                from 0.5 means the judge differentiates (good, but
                doesn't say which direction is desired -- read in
                conjunction with judge_keep_mass_var to detect
                row-level switching behaviour).
            judge_keep_mass_var     -- variance over rows.  Low var +
                ~0.5 mean => uniform-collapsed judge (BAD).  High var
                + middling mean => some rows keep, others write (GOOD).
            judge_effective_rank    -- exp(H(s_norm)) where s are the
                singular values of the average judge attention matrix
                ((K, 2K) over the corpus).  Low rank => the judge has
                learned a low-dimensional decision pattern.

          D3 (M_c stability + chain-distinguishability):
            Mc_step_delta_mean / max  -- mean / max
                ||M_c^t - M_c^{t-1}||_F / ||M_c^{t-1}||_F over
                sessions t>=1.  0 means the writer is a no-op; >>1
                means it's overwriting wholesale.
            Mc_pair_dist_mean / min   -- mean / min pairwise
                ||M_c^T_chainA - M_c^T_chainB||_F / sqrt(K * d)
                across all pairs of audited chains (T = final session
                of each chain).  Near-zero ==> writer is producing
                content-blind M_c (the SAME memory regardless of the
                chain content); large + uniform ==> distinguishable.
            Mc_self_dist_mean         -- ||M_c^T||_F / sqrt(K * d) for
                each chain, averaged.  Sanity-check against the pair
                distance: pair << self ==> chains DO collapse to
                roughly the same M_c.
        """
        a = self.args
        model = self._model()
        model.eval()
        ev = self.eval_corpus
        n_chains = min(a.diagnose_memory_dynamics_n_chains, len(ev))
        if n_chains < 1:
            return {}

        # extract_source / memory_block / write_gate live on the inner
        # Qwen3MemResModel, not on the ForCausalLM wrapper.
        inner = model.model
        K = int(model.config.memres_num_vectors)
        d = int(model.config.hidden_size)
        log_2K = math.log(2.0 * K)
        device = self.device

        all_row_entropies: list[float] = []
        all_keep_means: list[float] = []
        all_keep_vars: list[float] = []
        attn_running: torch.Tensor | None = None  # (K, 2K) accumulator
        attn_n = 0

        step_deltas: list[float] = []
        Mc_finals: list[torch.Tensor] = []  # one per chain, on CPU.

        for ci in range(n_chains):
            length = int(ev.chain_lengths[ci])
            if length < 2:
                continue
            M_c: torch.Tensor | None = None
            for t in range(length):
                sess = (ev.chain_session_at(ci, t)
                        .to(device).unsqueeze(0))  # (1, S)
                C = inner.extract_source(sess)  # (1, N, d)
                M_new = inner.memory_block.extract(C)  # (1, K, d)
                M_c_prev = (M_c if M_c is not None
                            else torch.zeros_like(M_new))
                attn = inner.memory_block.judge_attention(M_c_prev, M_new)
                # attn: (1, K, 2K)
                a_clipped = attn.clamp_min(1e-12)
                row_ent = -(a_clipped * a_clipped.log()).sum(dim=-1)  # (1, K)
                all_row_entropies.append(float(row_ent.mean().item()))
                keep = attn[..., :K].sum(dim=-1)  # (1, K)
                all_keep_means.append(float(keep.mean().item()))
                all_keep_vars.append(float(keep.var(unbiased=False).item()))
                # Running attention pattern for effective-rank.
                if attn_running is None:
                    attn_running = attn.squeeze(0).double().clone()
                else:
                    attn_running = attn_running + attn.squeeze(0).double()
                attn_n += 1
                # Now produce M_c^t through the gated/competitive path.
                M_c_new = inner.memory_block.judge(M_c_prev, M_new)
                if inner.memory_block.update_mode == "gated":
                    gate_input = torch.cat([M_c_prev, M_new], dim=-1)
                    g = torch.sigmoid(
                        inner.memory_block.write_gate(gate_input)
                    )
                    M_c_new = (1 - g) * M_c_prev + g * M_c_new
                if M_c is not None:
                    delta = (M_c_new - M_c).norm(p="fro")
                    base = M_c.norm(p="fro").clamp_min(1e-8)
                    step_deltas.append(float((delta / base).item()))
                M_c = M_c_new
            if M_c is not None:
                Mc_finals.append(M_c.squeeze(0).detach().cpu().float())

        out: dict[str, float] = {}
        if all_row_entropies:
            out["judge_row_entropy_mean"] = float(
                sum(all_row_entropies) / len(all_row_entropies)
            )
            out["judge_row_entropy_max_log_2K"] = log_2K
            out["judge_row_entropy_norm"] = (
                out["judge_row_entropy_mean"] / log_2K
            )
        if all_keep_means:
            out["judge_keep_mass_mean"] = float(
                sum(all_keep_means) / len(all_keep_means)
            )
            out["judge_keep_mass_var"] = float(
                sum(all_keep_vars) / len(all_keep_vars)
            )
        if attn_running is not None and attn_n > 0:
            avg_attn = (attn_running / attn_n).float()  # (K, 2K)
            S = torch.linalg.svdvals(avg_attn).clamp_min(1e-12)
            s_norm = S / S.sum()
            entropy = -(s_norm * s_norm.log()).sum().item()
            out["judge_effective_rank"] = float(math.exp(entropy))
        if step_deltas:
            out["Mc_step_delta_mean"] = float(
                sum(step_deltas) / len(step_deltas)
            )
            out["Mc_step_delta_max"] = float(max(step_deltas))
        if Mc_finals:
            scale = math.sqrt(float(K * d))
            self_norms = [float(m.norm(p="fro").item()) / scale
                          for m in Mc_finals]
            out["Mc_self_dist_mean"] = float(
                sum(self_norms) / len(self_norms)
            )
            if len(Mc_finals) >= 2:
                pair_dists: list[float] = []
                for i in range(len(Mc_finals)):
                    for j in range(i + 1, len(Mc_finals)):
                        d_ij = float(
                            (Mc_finals[i] - Mc_finals[j]).norm(p="fro").item()
                        ) / scale
                        pair_dists.append(d_ij)
                out["Mc_pair_dist_mean"] = float(
                    sum(pair_dists) / len(pair_dists)
                )
                out["Mc_pair_dist_min"] = float(min(pair_dists))
                # Diagnostic ratio: if pair << self, chains collapse to
                # the same M_c (writer is content-blind).
                if out["Mc_self_dist_mean"] > 1e-8:
                    out["Mc_pair_to_self_ratio"] = (
                        out["Mc_pair_dist_mean"] / out["Mc_self_dist_mean"]
                    )
        out["n_dynamics_chains"] = float(len(Mc_finals))
        return out

    @torch.no_grad()
    def evaluate(self) -> dict:
        a = self.args
        model = self._model()
        model.eval()
        ev = self.eval_corpus

        ce_mem, ce_no, ce_shuffle, ce_oracle = [], [], [], []
        for chain_idx in range(min(a.eval_n_chains, len(ev))):
            length = int(ev.chain_lengths[chain_idx])
            if length < a.eval_window + 1:
                continue
            # Compress sessions [0..end-1] into M_c, evaluate session [end].
            # We unroll the entire chain prefix recurrently for the full
            # cumulative-memory benefit.
            M_c = None
            # Score the LAST eval_window sessions (each conditioned on the
            # M_c built from all *strictly earlier* sessions).
            score_starts = range(length - a.eval_window, length)
            for end in range(length):
                sess = ev.chain_session_at(chain_idx, end).to(self.device).unsqueeze(0)
                input_ids = sess[:, :-1]
                if end in score_starts:
                    labels = input_ids
                    # Memory on
                    out_mem = model(input_ids=input_ids, labels=labels, M_c=M_c)
                    ce_mem.append(out_mem.loss.item())
                    # Memory off
                    out_no = model(input_ids=input_ids, labels=labels, M_c=None)
                    ce_no.append(out_no.loss.item())
                    # Shuffled memory: build a memory from a different chain's
                    # prefix of the same length end.
                    if M_c is not None:
                        other_idx = (chain_idx + 1) % len(ev)
                        other_len = int(ev.chain_lengths[other_idx])
                        clamp = min(end, other_len - 1)
                        if clamp > 0:
                            M_sh = None
                            for j in range(clamp):
                                osess = ev.chain_session_at(other_idx, j).to(self.device).unsqueeze(0)
                                C_o = model.model.extract_source(osess[:, :-1])
                                M_sh = model.model.compress_session(C_o, M_sh)
                            out_sh = model(input_ids=input_ids, labels=labels, M_c=M_sh)
                            ce_shuffle.append(out_sh.loss.item())

                    # Oracle: concat prior k-1 sessions + current as raw ctx.
                    # This is the "uncompressed full history" upper bound.
                    prior = ev.chain_window(
                        chain_idx, max(0, end - (a.eval_window - 1)), a.eval_window - 1
                    ).to(self.device)
                    if prior.numel() > 0:
                        prior_flat = prior.flatten().unsqueeze(0)  # (1, kp*S)
                        full = torch.cat([prior_flat, input_ids], dim=1)
                        labels_o = full.clone()
                        labels_o[:, : prior_flat.shape[1]] = -100
                        out_or = model(
                            input_ids=full, labels=labels_o, M_c=None
                        )
                        ce_oracle.append(out_or.loss.item())

                # Recurrent memory update is mandatory regardless of scoring.
                C_t = model.model.extract_source(sess[:, :-1])
                M_c = model.model.compress_session(C_t, M_c)

        def mean(xs):
            return float(sum(xs) / len(xs)) if xs else float("nan")

        legacy = {
            "n_scored": len(ce_mem),
            "ce_mem": mean(ce_mem),
            "ce_nomem": mean(ce_no),
            "ce_shuffle": mean(ce_shuffle),
            "ce_oracle_concat": mean(ce_oracle),
            "delta_nomem_minus_mem": mean(ce_no) - mean(ce_mem),
            "delta_shuffle_minus_mem": mean(ce_shuffle) - mean(ce_mem),
            "delta_oracle_minus_mem": mean(ce_oracle) - mean(ce_mem),
        }
        # NEW (v8+): the legacy standard-eval above measures CE over
        # 511 tokens of mostly-filler content with M_c built sequentially
        # through 40+ sessions -- a distribution the curriculum-trained
        # model has never seen and shouldn't be expected to dominate
        # before architectural recruitment is established.  The three
        # diagnostics below measure RECRUITMENT (gate / alpha_mem /
        # readout magnitude) and MATCHED-DISTRIBUTION discrimination
        # (phase-aligned callback-token Δ_sh-m), which is what we
        # actually save_best against in v8+ runs.
        pa = self._phase_aligned_eval()
        rec = self._routing_recruitment_summary()
        mt = self._readout_magnitude_diag()
        # D2/D3: only computed when explicitly requested (it pulls a
        # mid-epoch checkpoint sample of judge attention + M_c
        # trajectories, which is cheap but adds eval-time cost on a
        # tight schedule).
        dyn = (
            self._memory_dynamics_eval()
            if getattr(a, "diagnose_memory_dynamics", False)
            else {}
        )
        out = {**legacy, **pa, **rec, **mt, **dyn}
        # Ensure model is back in train mode (the helpers above keep
        # it eval; the legacy block at the top of evaluate() also did).
        model.train()
        return out

    def _save(self, tag: str, eval_metrics: dict | None = None) -> None:
        if not self.is_main:
            return
        ckpt = Path(self.args.out_dir) / tag
        ckpt.mkdir(parents=True, exist_ok=True)
        self._model().save_pretrained(ckpt)
        self.tokenizer.save_pretrained(ckpt)
        if eval_metrics:
            (ckpt / "eval_metrics.json").write_text(
                json.dumps(eval_metrics, indent=2), encoding="utf-8"
            )
        print(f"Saved checkpoint -> {ckpt}", flush=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _readout_warmup_freeze(self) -> None:
        """Phase 1 of fix A: freeze everything except the readout."""
        readout_params = set(id(p) for p in
                             self._model().model.memory_readout.parameters())
        for n, p in self._model().named_parameters():
            p.requires_grad_(id(p) in readout_params)

    def _readout_warmup_unfreeze(self) -> None:
        """Phase 2 of fix A: unfreeze everything for joint training."""
        for p in self._model().parameters():
            p.requires_grad_(True)

    def _writer_warmup_freeze(self) -> None:
        """v13 phase 1: freeze backbone + LM head + embed.  Train the
        full memres subsystem.

        Motivation (problems.md §5, recommendation 2): M_in and M_judge
        under joint LM-only training receive 10^-8-10^-9 of the
        backbone's gradient, which is below the optimiser's effective
        update scale -- the writer is effectively frozen at its init
        for the entire run.  Freezing the backbone / LM head / embed
        removes the 99% of parameters that dominate the loss landscape
        and concentrates gradient onto the memres subsystem.  Combined
        with ``_set_mem_bias(writer_warmup_router_bias)`` to force the
        depth softmax open, the writer receives direct LM gradient
        that is 5+ orders of magnitude larger than in joint training,
        over a stable backbone target.
        """
        markers = (
            "memory_block",
            "memory_readout",
            "depth_router",
            "memory_gate",
        )
        for n, p in self._model().named_parameters():
            keep = any(m in n for m in markers)
            p.requires_grad_(keep)

    def _writer_warmup_unfreeze(self, *, keep_backbone_frozen: bool) -> None:
        """v13 phase 2: unfreeze everything (or keep backbone frozen).

        When ``keep_backbone_frozen`` is True, the memres subsystem
        continues to train but the backbone stays frozen -- this mirrors
        the v12d_frozen regime (which produced the campaign's first
        positive ``evidence_lift``).
        """
        if keep_backbone_frozen:
            self._writer_warmup_freeze()
        else:
            for p in self._model().parameters():
                p.requires_grad_(True)

    def _set_mem_bias(self, value: float) -> None:
        """Force-set the router's mem_bias parameter to ``value`` for
        every routing step.  Operates in-place on the buffer; the
        optimiser may attempt to update it in phase 2 -- that is
        intended (we *want* the bias to be learnable post-warmup).

        In ``attention_parity`` routing this controls the memory
        column of the depth softmax (Eq. 9).  In ``simple_gate``
        routing depth_router is NOT on the forward path; the memory
        injection is controlled by ``memory_gate.gate`` (a per-
        sublayer scalar added to h via ``h + gate * m^t``).  This
        function therefore also force-sets the memory_gate to a
        value that gives ~50% memory contribution at the effective
        readout norm init (norm=0.05 -> ||m^t||/||embed|| ~3.9, so
        gate=0.5 gives gate*||m^t|| ~ 2 which is comparable to
        ||h|| magnitudes after a layer).  Without this, writer
        warmup in simple_gate mode leaves m^t disconnected from h
        and the writer receives zero LM gradient through the
        forward path -- defeating the entire purpose of the warmup.
        """
        bias = self._model().model.depth_router.mem_bias
        with torch.no_grad():
            bias.data.fill_(float(value))
        # simple_gate forward-path force-open.  Map mem_bias in
        # [0, 4] to gate in [0, ~0.5] via a saturating function
        # (tanh scaled); mem_bias = 4 -> gate ~ 0.50.  This gives
        # a reasonable memory contribution without crushing
        # backbone signal under a saturating gate.
        mode = getattr(self._model().config, "memres_mode", None)
        if mode == "simple_gate":
            gate = self._model().model.memory_gate.gate
            # Smooth map: gate_target = 0.5 * tanh(bias/2) for bias
            # in [0, inf).  bias=4 -> 0.5 * tanh(2) = ~0.482.
            import math as _m
            target = 0.5 * _m.tanh(float(value) / 2.0)
            with torch.no_grad():
                gate.data.fill_(target)

    def fit(self) -> None:
        a = self.args
        self.wrapped.train()
        rng = random.Random(a.seed + self.rank * 7919 + 13)
        carry = None

        # v13 architectural fix (writer warmup) takes precedence over
        # the v11r readout warmup when both are set -- writer warmup
        # trains a strict superset of parameters.
        if a.writer_warmup_steps > 0:
            self._writer_warmup_freeze()
            self._set_mem_bias(a.writer_warmup_router_bias)
            if self.is_main:
                n_train = sum(
                    p.numel() for p in self._model().parameters()
                    if p.requires_grad
                )
                print(
                    f"  [v13] writer warmup ENGAGED: phase 1 = "
                    f"{a.writer_warmup_steps} steps, "
                    f"router mem_bias forced to "
                    f"{a.writer_warmup_router_bias}, "
                    f"{n_train:,} trainable params (memres subsystem "
                    f"only: M_in, extract, M_judge, judge, readout, "
                    f"router, memres_gate).",
                    flush=True,
                )
        # Architectural fix A (v11r): scaffolded readout warmup.  Freeze
        # the writer + router + backbone + LM head and force the
        # routing toward memory before phase 2's joint optimisation can
        # close the pathway.  See parser help for rationale.  Skipped
        # if writer_warmup_steps > 0 (the superset training).
        elif a.readout_warmup_steps > 0:
            self._readout_warmup_freeze()
            self._set_mem_bias(a.readout_warmup_router_bias)
            if self.is_main:
                n_train = sum(
                    p.numel() for p in self._model().parameters()
                    if p.requires_grad
                )
                print(
                    f"  [fix A] readout warmup ENGAGED: phase 1 = "
                    f"{a.readout_warmup_steps} steps, "
                    f"router mem_bias forced to "
                    f"{a.readout_warmup_router_bias}, "
                    f"{n_train:,} trainable params (readout only).",
                    flush=True,
                )

        accum_loss = 0.0
        accum_step = 0
        tokens_seen = 0
        t0 = time.time()

        if self.is_main:
            print("  entering training loop", flush=True)

        self.optimizer.zero_grad(set_to_none=True)
        while self.global_step < a.steps:
            # Apply the warmup schedule each step.  Within phase 1 we
            # also re-apply the bias each step because, even though
            # the router parameters are frozen in this phase, the
            # bias buffer can drift on rare numerical paths -- this
            # makes the schedule explicit.
            if a.writer_warmup_steps > 0:
                if self.global_step < a.writer_warmup_steps:
                    self._set_mem_bias(a.writer_warmup_router_bias)
                elif self.global_step == a.writer_warmup_steps:
                    self._writer_warmup_unfreeze(
                        keep_backbone_frozen=(
                            a.writer_warmup_keep_backbone_frozen
                            or getattr(a, "freeze_backbone", False)
                        )
                    )
                    if self.is_main:
                        bb_note = (
                            "backbone STAYS FROZEN"
                            if a.writer_warmup_keep_backbone_frozen
                               or getattr(a, "freeze_backbone", False)
                            else "backbone unfreezes"
                        )
                        print(
                            f"  [v13] writer warmup COMPLETE "
                            f"@ step {self.global_step}: "
                            f"{bb_note}, annealing "
                            f"mem_bias {a.writer_warmup_router_bias} "
                            f"-> {a.router_mem_bias_init} over "
                            f"{a.writer_warmup_anneal_steps} steps",
                            flush=True,
                        )
                elif (self.global_step
                      < a.writer_warmup_steps
                      + a.writer_warmup_anneal_steps):
                    progress = (
                        (self.global_step - a.writer_warmup_steps)
                        / max(1, a.writer_warmup_anneal_steps)
                    )
                    target = (
                        a.writer_warmup_router_bias * (1.0 - progress)
                        + a.router_mem_bias_init * progress
                    )
                    self._set_mem_bias(target)
            elif a.readout_warmup_steps > 0:
                if self.global_step < a.readout_warmup_steps:
                    self._set_mem_bias(a.readout_warmup_router_bias)
                elif self.global_step == a.readout_warmup_steps:
                    self._readout_warmup_unfreeze()
                    if self.is_main:
                        print(
                            f"  [fix A] readout warmup COMPLETE "
                            f"@ step {self.global_step}: "
                            f"unfreezing all params, annealing "
                            f"mem_bias {a.readout_warmup_router_bias} "
                            f"-> {a.router_mem_bias_init} over "
                            f"{a.readout_warmup_anneal_steps} steps",
                            flush=True,
                        )
                elif (self.global_step
                      < a.readout_warmup_steps
                      + a.readout_warmup_anneal_steps):
                    progress = (
                        (self.global_step - a.readout_warmup_steps)
                        / max(1, a.readout_warmup_anneal_steps)
                    )
                    target = (
                        a.readout_warmup_router_bias * (1.0 - progress)
                        + a.router_mem_bias_init * progress
                    )
                    self._set_mem_bias(target)
            loss_val, carry = self._train_step(rng, carry)
            accum_loss += loss_val
            tokens_seen += a.window_k * a.session_len * a.batch_size
            accum_step += 1
            if accum_step < a.grad_accum:
                continue

            # D1: capture per-module gradient norms BEFORE clip so the
            # values reflect the optimiser's raw signal, not whatever
            # max_norm cropped them to.  Cheap (one .pow.sum per param),
            # gated to log_every cadence so it doesn't burn budget.
            # Note: self.global_step is incremented just below; check
            # against (global_step + 1) so the log fires the same step
            # the print/wandb logging branch picks it up.
            if (a.diagnose_grad_groups
                    and self.is_main
                    and (self.global_step + 1) % a.log_every == 0):
                self._last_grad_norms = self._compute_grad_norms()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.wrapped.parameters(), a.max_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            accum_step = 0

            if self.global_step % a.log_every == 0:
                # All-reduce loss across ranks (collective op).
                loss_t = torch.tensor(accum_loss / a.grad_accum, device=self.device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                if self.is_main:
                    elapsed = time.time() - t0
                    tok_sec = tokens_seen * self.world_size / max(elapsed, 1e-6)
                    lr_now = [g["lr"] for g in self.optimizer.param_groups]
                    gate_mean = self._model().model.memory_gate.gate.float().mean().item()
                    gate_max = self._model().model.memory_gate.gate.float().abs().max().item()
                    nce_str = ""
                    if a.contrastive_infonce_weight > 0.0:
                        nce_str = (
                            f"nce {self._last_contrastive_loss:.3f} "
                            f"diag {self._last_contrastive_diag:.3f} "
                            f"off {self._last_contrastive_offdiag:.3f} "
                            f"gap {self._last_contrastive_gap:+.3f} | "
                        )
                    floor_str = ""
                    if a.alpha_mem_floor_aux_weight > 0.0:
                        floor_str = (
                            f"a_floor {getattr(self, '_last_alpha_floor_loss', 0.0):.4f} "
                            f"a_mean {getattr(self, '_last_alpha_mem_batch_mean', 0.0):.4f} | "
                        )
                    print(
                        f"step {self.global_step:6d} | loss {loss_t.item():.4f} | "
                        f"lrs {lr_now} | grad_norm {float(grad_norm):.3f} | "
                        f"gate_mean {gate_mean:+.4f} max {gate_max:.4f} | "
                        f"{nce_str}"
                        f"{floor_str}"
                        f"{tok_sec/1e3:.1f}k tok/s",
                        flush=True,
                    )
                    # D1: per-module gradient-norm summary.  Format
                    # focuses on the memres subsystem first, then the
                    # backbone reference, so a pathological starvation
                    # signal (memres << backbone) is visually obvious.
                    if a.diagnose_grad_groups and self._last_grad_norms:
                        g = self._last_grad_norms
                        bb = max(g.get("backbone", 0.0), 1e-12)
                        ratios = {
                            k: g.get(k, 0.0) / bb
                            for k in ("M_in", "extract", "M_judge",
                                      "judge", "readout", "router",
                                      "memres_gate", "write_gate")
                            if k in g
                        }
                        ratios_str = " ".join(
                            f"{k}={v:.2e}" for k, v in ratios.items()
                        )
                        print(
                            f"  GRAD @ step {self.global_step}: "
                            f"backbone {g.get('backbone', 0.0):.3e} | "
                            f"|g|/|g_bb|: {ratios_str}",
                            flush=True,
                        )
                    if self.use_wandb:
                        import wandb
                        wandb_payload = {
                            "train/loss": loss_t.item(),
                            "train/lr_memres": lr_now[0],
                            "train/lr_backbone": lr_now[-1],
                            "train/grad_norm": float(grad_norm),
                            "train/tok_per_s": tok_sec,
                            "train/gate_mean": gate_mean,
                            "train/gate_abs_max": gate_max,
                        }
                        if a.contrastive_infonce_weight > 0.0:
                            wandb_payload["train/contrast_loss"] = self._last_contrastive_loss
                            wandb_payload["train/contrast_diag"] = self._last_contrastive_diag
                            wandb_payload["train/contrast_offdiag"] = self._last_contrastive_offdiag
                            wandb_payload["train/contrast_gap"] = self._last_contrastive_gap
                        if a.alpha_mem_floor_aux_weight > 0.0:
                            wandb_payload["train/alpha_floor_loss"] = getattr(
                                self, "_last_alpha_floor_loss", 0.0
                            )
                            wandb_payload["train/alpha_mem_batch_mean"] = getattr(
                                self, "_last_alpha_mem_batch_mean", 0.0
                            )
                        if a.diagnose_grad_groups and self._last_grad_norms:
                            for k, v in self._last_grad_norms.items():
                                wandb_payload[f"grad/{k}"] = v
                            bb = max(self._last_grad_norms.get("backbone", 0.0),
                                     1e-12)
                            for k, v in self._last_grad_norms.items():
                                if k != "backbone":
                                    wandb_payload[f"grad_ratio/{k}"] = v / bb
                        wandb.log(wandb_payload, step=self.global_step)
                t0 = time.time()
                tokens_seen = 0
            accum_loss = 0.0

            if self.global_step % a.eval_every == 0:
                metrics = self.evaluate()
                if metrics and self.is_main:
                    # Legacy standard eval (eval_window=8, sequential M_c).
                    print(
                        f"  EVAL @ step {self.global_step}: n={metrics['n_scored']} "
                        f"mem={metrics['ce_mem']:.4f} nomem={metrics['ce_nomem']:.4f} "
                        f"shuffle={metrics['ce_shuffle']:.4f} oracle={metrics['ce_oracle_concat']:.4f} "
                        f"Δnm-m={metrics['delta_nomem_minus_mem']:+.4f} "
                        f"Δsh-m={metrics['delta_shuffle_minus_mem']:+.4f} "
                        f"Δor-m={metrics['delta_oracle_minus_mem']:+.4f}",
                        flush=True,
                    )
                    # NEW: phase-aligned eval (matches train distribution).
                    if metrics.get("n_pa_scored", 0) > 0:
                        pa_ws_dnm = metrics.get("pa_ws_dnm", float("nan"))
                        pa_ws_dsh = metrics.get("pa_ws_dsh", float("nan"))
                        pa_cb_dnm = metrics.get("pa_cb_dnm", float("nan"))
                        pa_cb_dsh = metrics.get("pa_cb_dsh", float("nan"))
                        print(
                            f"  PA-EVAL @ step {self.global_step}: "
                            f"n={metrics['n_pa_scored']} cb_n={metrics['n_pa_cb_scored']} "
                            f"WS Δnm-m={pa_ws_dnm:+.4f} Δsh-m={pa_ws_dsh:+.4f} | "
                            f"CB Δnm-m={pa_cb_dnm:+.4f} Δsh-m={pa_cb_dsh:+.4f}",
                            flush=True,
                        )
                        # NEW (v11): evidence-aware decomposition. Only fires
                        # when the corpus has chain_evidence_positions populated
                        # (i.e., LME with answer_session_ids, not v6 legacy).
                        # pa_cb_evidence_lift = pa_cb_dnm - pa_cb_dnm_floor;
                        # > 0 iff memory helps MORE on evidence-labelled
                        # callback than on uniform-pick callback (i.e., the
                        # readout learned something content-specific, not just
                        # an unconditional pull).
                        n_ev = metrics.get("n_pa_cb_evidence_labelled", 0)
                        if n_ev and n_ev > 0:
                            print(
                                f"  EVID-EVAL @ step {self.global_step}: "
                                f"n_ev={n_ev} "
                                f"pa_cb_ce_mem={metrics.get('pa_cb_ce_mem', float('nan')):.4f} "
                                f"pa_cb_ce_mem_floor={metrics.get('pa_cb_ce_mem_floor', float('nan')):.4f} "
                                f"Δnm-m_floor={metrics.get('pa_cb_dnm_floor', float('nan')):+.4f} "
                                f"evidence_lift={metrics.get('pa_cb_evidence_lift', float('nan')):+.4f}",
                                flush=True,
                            )
                    # NEW: routing recruitment + readout magnitude.
                    if "rec_mode" in metrics:
                        rec_mode = metrics["rec_mode"]
                        if rec_mode == "simple_gate":
                            top = metrics.get("rec_gate_top", [])
                            top_str = ", ".join(
                                f"l{i}={v:+.4f}" for i, v in top
                            )
                            print(
                                f"  ROUTE @ step {self.global_step}: "
                                f"mode=simple_gate "
                                f"|gate|_max={metrics.get('rec_gate_max_abs', 0.0):.4f} "
                                f"|gate|_mean={metrics.get('rec_gate_mean_abs', 0.0):.4f} "
                                f"frac_open={metrics.get('rec_frac_open', 0.0):.2f} "
                                f"top=[{top_str}]",
                                flush=True,
                            )
                        else:
                            top = metrics.get("rec_alpha_mem_top", [])
                            top_str = ", ".join(
                                f"l{i}={v:.4f}" for i, v in top
                            )
                            print(
                                f"  ROUTE @ step {self.global_step}: "
                                f"mode={rec_mode} "
                                f"α_mem_max={metrics.get('rec_alpha_mem_max', 0.0):.4f} "
                                f"α_mem_mean={metrics.get('rec_alpha_mem_mean', 0.0):.4f} "
                                f"frac_open={metrics.get('rec_frac_open', 0.0):.2f} "
                                f"top=[{top_str}]",
                                flush=True,
                            )
                    if not math.isnan(metrics.get("mt_norm_ratio_mean", float("nan"))):
                        print(
                            f"  READOUT @ step {self.global_step}: "
                            f"||m^t|| / ||embed|| mean="
                            f"{metrics['mt_norm_ratio_mean']:.3f} "
                            f"max={metrics.get('mt_norm_ratio_max', float('nan')):.3f}",
                            flush=True,
                        )
                    # D2/D3: writer / judge dynamics summary.  Only
                    # printed when --diagnose_memory_dynamics is set
                    # (otherwise the eval dict never contains these
                    # keys).  Three lines: judge decisiveness, M_c
                    # step-stability, M_c chain-distinguishability.
                    if "judge_row_entropy_mean" in metrics:
                        print(
                            f"  D2-JUDGE @ step {self.global_step}: "
                            f"row_entropy={metrics['judge_row_entropy_mean']:.3f} "
                            f"(uniform={metrics['judge_row_entropy_max_log_2K']:.3f}; "
                            f"norm={metrics['judge_row_entropy_norm']:.3f}) | "
                            f"keep_mean={metrics.get('judge_keep_mass_mean', 0.0):.3f} "
                            f"keep_var={metrics.get('judge_keep_mass_var', 0.0):.4f} | "
                            f"eff_rank={metrics.get('judge_effective_rank', float('nan')):.2f}",
                            flush=True,
                        )
                    if "Mc_step_delta_mean" in metrics:
                        print(
                            f"  D3-MC    @ step {self.global_step}: "
                            f"Δ_step mean={metrics['Mc_step_delta_mean']:.3f} "
                            f"max={metrics.get('Mc_step_delta_max', 0.0):.3f} | "
                            f"self||M||={metrics.get('Mc_self_dist_mean', 0.0):.3f} | "
                            f"pair={metrics.get('Mc_pair_dist_mean', 0.0):.3f} "
                            f"min={metrics.get('Mc_pair_dist_min', 0.0):.3f} "
                            f"(pair/self={metrics.get('Mc_pair_to_self_ratio', float('nan')):.3f})",
                            flush=True,
                        )
                    if self.use_wandb:
                        import wandb
                        wandb.log({f"eval/{k}": v for k, v in metrics.items()
                                   if isinstance(v, (int, float))},
                                  step=self.global_step)

                    # save_best score (lower-is-better convention).
                    #
                    # ce_mem (legacy): minimise raw memory CE. Gameable by
                    #   channel collapse (model nulls memory, ce_mem drops
                    #   from overfitting backbone). Use only on pair-trained
                    #   warm-ups.
                    # composite: legacy v6/v7 standard-eval composite,
                    #   penalises channel collapse (Δ_nm-m -> 0) AND
                    #   shortcut learning (Δ_sh-m -> 0). Useful when
                    #   train and eval distributions match.
                    # phase_aligned (default v8+): matches the curriculum
                    #   training distribution (1 fresh evidence + callback
                    #   session). The callback-token-only Δ_sh-m is the
                    #   ultimate signal that the readout is content-
                    #   discriminative on the tokens that actually require
                    #   memory; the whole-session pa_ws_dsh is a regulariser
                    #   that prevents winning the cb_dsh by random noise.
                    if a.save_best_metric == "ce_mem":
                        score = metrics["ce_mem"]
                    elif a.save_best_metric == "phase_aligned":
                        cb_dsh = metrics.get("pa_cb_dsh", float("nan"))
                        ws_dsh = metrics.get("pa_ws_dsh", float("nan"))
                        # If the phase-aligned eval failed to score
                        # anything (no eligible chains), fall back to
                        # the composite to avoid saving every checkpoint.
                        if math.isnan(cb_dsh) and math.isnan(ws_dsh):
                            d_nm = metrics.get("delta_nomem_minus_mem", 0.0) or 0.0
                            d_sh = metrics.get("delta_shuffle_minus_mem", 0.0) or 0.0
                            score = -(d_nm + 2.0 * d_sh)
                        else:
                            cb_term = 0.0 if math.isnan(cb_dsh) else cb_dsh
                            ws_term = 0.0 if math.isnan(ws_dsh) else ws_dsh
                            score = -(cb_term + 0.5 * ws_term)
                    else:  # composite
                        d_nm = metrics.get("delta_nomem_minus_mem", 0.0) or 0.0
                        d_sh = metrics.get("delta_shuffle_minus_mem", 0.0) or 0.0
                        score = -(d_nm + 2.0 * d_sh)
                    if score < self.best_eval_ce:
                        self.best_eval_ce = score
                        self._save("best", eval_metrics={
                            **metrics,
                            "_save_best_metric": a.save_best_metric,
                            "_save_best_score": score,
                        })

            if self.global_step % a.save_every == 0:
                self._save(f"step-{self.global_step}")

        final_metrics = self.evaluate()
        self._save("final", eval_metrics=final_metrics)
        if self.is_main and self.use_wandb:
            import wandb
            wandb.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _eval_only_run(trainer: "Trainer") -> dict:
    """Run _phase_aligned_eval N times (with distinct RNG seeds) plus
    the cheap routing + readout diagnostics once, then aggregate.

    Returns a dict with, for every PA key, {mean, std, ci95, seeds}.
    """
    a = trainer.args
    # Scalar-metric keys actually produced by _phase_aligned_eval
    # (see definition in train_chain.py: pa_cb_dnm / pa_cb_dsh /
    # pa_ws_dnm / pa_ws_dsh plus the raw CE means).
    pa_keys = [
        "pa_ws_ce_mem", "pa_ws_ce_nomem", "pa_ws_ce_shuffle",
        "pa_ws_dnm", "pa_ws_dsh",
        "pa_cb_ce_mem", "pa_cb_ce_nomem", "pa_cb_ce_shuffle",
        "pa_cb_dnm", "pa_cb_dsh",
        "n_pa_scored", "n_pa_cb_scored",
    ]
    per_seed: dict[str, list[float]] = {k: [] for k in pa_keys}
    for s in range(a.eval_seeds):
        # _phase_aligned_eval seeds its RNG off a.seed + global_step + 17;
        # drift global_step to get a fresh seed per repeat.
        saved = trainer.global_step
        trainer.global_step = saved + 1_000_000 + 7919 * s
        pa = trainer._phase_aligned_eval()
        trainer.global_step = saved
        for k in pa_keys:
            if k in pa:
                per_seed[k].append(float(pa[k]))
    agg: dict = {}
    import statistics as _st
    for k, vs in per_seed.items():
        if not vs:
            continue
        m = _st.fmean(vs)
        sd = _st.pstdev(vs) if len(vs) > 1 else 0.0
        # Simple approx 95% CI on the mean (t-approx for small N).
        se = sd / max(len(vs) ** 0.5, 1e-9)
        agg[k] = {
            "mean": m, "std": sd, "se": se,
            "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se,
            "n_seeds": len(vs), "values": vs,
        }
    # Routing + readout are deterministic given the model; sample once.
    rec = trainer._routing_recruitment_summary()
    mt = trainer._readout_magnitude_diag()
    # D2/D3: also deterministic per checkpoint; sampled once.  Empty
    # dict when --diagnose_memory_dynamics is not set, which is the
    # legacy behaviour.
    dyn = (
        trainer._memory_dynamics_eval()
        if getattr(a, "diagnose_memory_dynamics", False)
        else {}
    )
    return {
        "pa_n_chains": a.phase_aligned_eval_n_chains,
        "n_seeds": a.eval_seeds,
        "pa": agg,
        "routing": rec,
        "readout": mt,
        "memory_dynamics": dyn,
    }


def main() -> None:
    args = parse_args()
    trainer = Trainer(args)
    if args.eval_only:
        if trainer.is_main:
            print(
                f"=== eval_only: n_chains={args.phase_aligned_eval_n_chains}"
                f" seeds={args.eval_seeds} ===",
                flush=True,
            )
        out = _eval_only_run(trainer)
        if trainer.is_main:
            pa = out.get("pa", {})
            for k in ("pa_cb_dnm", "pa_cb_dsh", "pa_ws_dnm", "pa_ws_dsh"):
                if k in pa:
                    v = pa[k]
                    print(
                        f"  {k:38s}: mean={v['mean']:+.4f}"
                        f"  std={v['std']:+.4f}"
                        f"  95% CI=[{v['ci95_lo']:+.4f},"
                        f" {v['ci95_hi']:+.4f}]"
                        f"  (n={args.phase_aligned_eval_n_chains},"
                        f" seeds={v['n_seeds']})",
                        flush=True,
                    )
            rt = out.get("routing", {})
            mt = out.get("readout", {})
            rec_max = rt.get(
                "rec_alpha_mem_max", rt.get("rec_gate_max_abs", 0.0)
            )
            print(
                f"  ROUTE: mode={rt.get('rec_mode', '?')}"
                f"  alpha_or_gate_max={rec_max:.4f}"
                f"  frac_open={rt.get('rec_frac_open', 0.0):.2f}",
                flush=True,
            )
            print(
                f"  READOUT: ||m^t||/||embed|| mean="
                f"{mt.get('mt_norm_ratio_mean', 0.0):.2f}",
                flush=True,
            )
            # D2/D3 summary -- only fires when the diagnostic was
            # requested (otherwise out["memory_dynamics"] == {}).
            dyn = out.get("memory_dynamics", {})
            if dyn:
                if "judge_row_entropy_mean" in dyn:
                    print(
                        f"  D2-JUDGE: row_entropy={dyn['judge_row_entropy_mean']:.3f}"
                        f" (uniform={dyn.get('judge_row_entropy_max_log_2K', 0.0):.3f}; "
                        f"norm={dyn.get('judge_row_entropy_norm', 0.0):.3f}) "
                        f"keep_mean={dyn.get('judge_keep_mass_mean', 0.0):.3f} "
                        f"keep_var={dyn.get('judge_keep_mass_var', 0.0):.4f} "
                        f"eff_rank={dyn.get('judge_effective_rank', float('nan')):.2f}",
                        flush=True,
                    )
                if "Mc_step_delta_mean" in dyn:
                    print(
                        f"  D3-MC   : Δ_step mean={dyn['Mc_step_delta_mean']:.3f} "
                        f"max={dyn.get('Mc_step_delta_max', 0.0):.3f} "
                        f"self||M||={dyn.get('Mc_self_dist_mean', 0.0):.3f} "
                        f"pair={dyn.get('Mc_pair_dist_mean', 0.0):.3f} "
                        f"min={dyn.get('Mc_pair_dist_min', 0.0):.3f} "
                        f"(pair/self={dyn.get('Mc_pair_to_self_ratio', float('nan')):.3f})",
                        flush=True,
                    )
            if args.eval_out:
                import json as _json
                Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.eval_out).write_text(
                    _json.dumps(out, indent=2), encoding="utf-8"
                )
                print(f"  wrote {args.eval_out}", flush=True)
        return
    trainer.fit()


if __name__ == "__main__":
    main()
