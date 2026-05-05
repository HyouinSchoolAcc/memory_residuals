#!/usr/bin/env bash
# v11l-fix (LOCAL H100) -- attention_parity +4/0  *  FROZEN BACKBONE
#
# Re-do of the killed-at-startup v11l_ap_frozen_backbone_gh200 cell on
# the local H100. The GH200 launch failed with exit_code 2 (path bug,
# see runs.md "v11 campaign -- first 7 cells finished" §, line ~152).
# Hypothesis (b) -- backbone co-evolution crowding m^t out of the
# attention_parity depth softmax -- remains the only single-knob v11
# cell that has not been tested. We run it locally (the GH200 watchdog
# is dead and reviving it is not blocking).
#
# Single-knob diff vs v11g_ap_baseline_gh200: --freeze_backbone.
# Everything else (corpus, routing biases, init knobs, curriculum,
# callback weight, lr schedule, step budget, eval cadence) is held
# identical so any difference in the trajectory is causally
# attributable to backbone gradient flow.
#
# Per the v11g grow-then-decay trajectory:
#   step 200  alpha_mem_max 0.0047, peak l54+l53 (deep block summaries)
#   step 400  alpha_mem_max 0.0108, peak shifts to l11+l12+l13
#   step 600  alpha_mem_max 0.0124  (peak; mass concentrated at shallow)
#   step 1000 alpha_mem_max 0.0113  (decay starts on PA CB)
#   step 1400 alpha_mem_max 0.0102  (peak sublayers stable; PA CB negative)
#
# (a) readout/writer overfit OR (b) backbone co-evolution. (b) is
# structurally specific to attention_parity (block summaries
# compete with memory inside the same softmax). Freezing the backbone
# *eliminates* that co-evolution mechanically -- block summaries stay
# at their pretrained-init magnitude/content forever -- so any decay we
# still see comes from inside the memory subsystem itself.
#
# Decision triggers (sharp; mirrors v11g for direct comparability)
# -----------------------------------------------------------------
#   step  200 : alpha_mem_max > 0  AND  ||m^t||/||embed|| in [0.3, 50]
#   step  500 : alpha_mem_max > 5e-3  AND  pa_cb_dnm > +0.005
#                                     AND  pa_cb_evidence_lift > +0.005
#   step 1000 : alpha_mem_max > 1e-2  AND  pa_cb_dnm > +0.020
#                                     AND  pa_cb_dsh > +0.005
#                                     AND  alpha_mem_max NOT decaying
#                                          (vs v11g step 1000 = 0.011 and
#                                          decaying)
#   step 2000 : standard delta_sh-m > +0.005
#   KILL: step 1000 with alpha_mem_max < 1e-3.  Frozen backbone *and*
#         attention_parity collapsed -- rules out (b) entirely; the
#         decay is internal to memres and the v12 slot-judge cell on
#         D4 is the next thing to inspect.
#
# Memory budget. Backbone fwd/bwd still runs (we need gradient to flow
# *through* it to reach memres), but optimizer state is dropped for the
# 596M frozen params (~1.2 GB at fp32 m/v) and bwd grads are not
# accumulated for them (~1.2 GB more). On the 94 GB H100 NVL this is
# well within budget.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=0
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --freeze_backbone \
    --train_chains paper_artifacts/chains/v11_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 200 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v11l_ap_frozen_backbone_local \
    --out_dir output/chain_v11l_ap_frozen_backbone_local \
    2>&1 | tee logs/chain_v11l_ap_frozen_backbone_local.log
