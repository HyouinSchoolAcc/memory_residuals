#!/usr/bin/env bash
# v8a P0 SIMPLE_GATE + readout RMSNorm
#
# Single-knob diff vs v7 SIMPLE_GATE (the GH200 cell): MemoryReadout now
# applies RMSNorm to its output. This is the architectural fix motivated
# by the v7 step-500 diagnostic in
# paper_artifacts/eval/diag_v7_simplegate_step500.json which showed
# ||m^t|| / ||embed|| = 165 (totally out of scale -- the readout
# was exploding because W_V^read had no scale control). The
# corresponding diagnostic on v7 SOFTERBIAS step-2000
# (paper_artifacts/eval/diag_v7_softerbias_step2000.json) showed the
# *opposite* failure mode: ||m^t|| / ||embed|| = 1.66e-10 (collapsed,
# weight_decay killed W_V^read because alpha_mem ~ 4e-4 gave it no
# meaningful gradient). RMSNorm on the readout output bounds m^t to
# ~sqrt(d) and makes both modes well-behaved.
#
# What this cell answers:
#
#   Q1: with the readout magnitude bounded, does simple_gate's gate
#       grow into a content-discriminative routing pattern? Concrete:
#       phase-aligned callback-token Δ_sh-m > +0.005 by step 500
#       (vs v7 SIMPLE_GATE where pa_cb_dsh = -0.0059 because m^t
#       was exploded).
#
#   Q2: does the readout magnitude diagnostic stabilise at
#       ||m^t||/||embed|| in [0.5, 5] across training? If it stays
#       near 1.0 the architecture fix is doing its job.
#
# Save_best metric is now `phase_aligned`: composite of
# pa_cb_dsh + 0.5 * pa_ws_dsh on the new phase-aligned eval that
# matches the curriculum training distribution. The legacy standard
# eval (eval_window=8, sequential M_c through 40+ sessions) is still
# emitted as a sanity check but DOES NOT drive checkpoint selection
# during P0 training -- it measures a distribution the model has
# never seen.
#
# Decision triggers (sharp, falsifiable):
#   step 200: ||m^t||/||embed|| in [0.3, 10] AND |gate|_max > 1e-3
#   step 500: pa_cb_dsh > +0.005 (matched-distribution callback-token
#             specificity)
#   step 1000: pa_cb_dsh > +0.020 AND pa_ws_dsh > +0.005 -- promote
#              to v8b (mixed-bias curriculum) on next launch.
set -eu
cd "$(dirname "$0")/.."
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
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v8a_p0_simplegate_rmsnorm \
    --out_dir output/chain_v8a_p0_simplegate_rmsnorm \
    2>&1 | tee logs/chain_v8a_p0_simplegate_rmsnorm.log
