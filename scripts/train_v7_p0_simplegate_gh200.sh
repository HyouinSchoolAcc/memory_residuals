#!/usr/bin/env bash
# v7 P0 SIMPLE_GATE — architectural ablation against attention_parity routing.
#
# This is the third v7 P0 cell. The two local cells (SOFTERBIAS, V3BIAS)
# both use --memres_mode attention_parity, the depth-wise softmax router
# that was the structural bottleneck across v3-v6 (gate_max=0.0000, α_mem
# content-blind on the v3 routing trace). This cell strips that machinery
# and uses --memres_mode simple_gate instead:
#
#   - simple_gate: ReZero-style ONE learnable scalar per sublayer.
#     g_init = 0 (bit-exact init parity with bare Qwen3). The scalar
#     gate has a DIRECT gradient path from LM CE -- no saturated
#     softmax against a recent-block source to relax, no per-position
#     pseudo-query w_{n,i} to learn. The READOUT m^t still feeds
#     through the gate; the rest of the memory machinery (writer,
#     extract, judge, readout) is unchanged from the attention_parity
#     cells, so we get a clean single-axis A/B on the routing.
#
#   - depth_router params are FROZEN automatically in simple_gate mode
#     (see Trainer._apply_freeze) so they don't drift; M_c -> readout
#     -> per-sublayer scalar -> add to residual.
#
# What this cell answers:
#
#   Q1 (highest-prior): does simple_gate's direct gradient path open
#       the memory channel where attention_parity could not? If YES
#       (gate_max > 0 by step 200, Δ_sh-m > +0.005 by step 500), the
#       paper's choice of attention_parity over simple_gate is
#       structurally wrong and the next move is to rewrite the routing
#       section around simple_gate.
#
#   Q2: if simple_gate ALSO stays at gate_max=0 with the same P0
#       curriculum, the failure is downstream of routing -- the writer
#       is producing useless M_c, or the readout is producing
#       useless m^t, or the LM CE just doesn't reward memory use on
#       this corpus regardless of how easy we make the gate. Different
#       set of next experiments (extract source, readout dim, M_c init).
#
# Single-knob diff vs train_v7_p0_softerbias.sh:
#   --memres_mode  attention_parity  -> simple_gate
#   (REMOVED) --router_mem_bias_init -2
#   (REMOVED) --router_recent_bias_init 4
#   (router biases are unused in simple_gate mode; no defaults needed)
#
# Everything else identical: P0 curriculum (window_k=2,
# curriculum_evidence_bias=1.0), gated writer, hidden_14 extract,
# callback_loss_weight=10, no burn-in, no carry_state.
#
# Decision triggers (vs local v7 cells at matched step counts):
#   step 200: gate_max > 1e-4 (any signal at all). simple_gate's
#             init is g=0 strictly, so any movement here is real.
#   step 500: Δ_sh-m > +0.005. Match-or-beat the local cells.
#   step 1000: if gate_max > 1e-3 AND Δ_sh-m > +0.01, this is
#              the paper's new architectural baseline. Promote to
#              P1 (--window_k 3) on next launch.
set -eu
cd "${REPO:-/home/ubuntu/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 2 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 100 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 10.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v7_p0_simplegate \
    --out_dir output/chain_v7_p0_simplegate
