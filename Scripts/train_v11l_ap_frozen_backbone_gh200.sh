#!/usr/bin/env bash
# v11l (GH200, 0.6B) -- attention_parity +4/0  *  FROZEN BACKBONE ablation
#
# User-directed ablation 2026-04-30 ~22:30 UTC after the v11g grow-then-decay
# review:
#   "It seems to me that a frozen backbone with only tuning the summarizer
#    for memory is now a necessary ablation study; can you make one in the
#    image of v11g and add it to the gh200 queue?"
#
# What this cell tests
# --------------------
# Single-knob diff vs v11g: --freeze_backbone. Every other axis (corpus,
# routing, biases, init knobs, curriculum, callback weight, lr, schedule,
# step budget, eval cadence) is held identical so any difference in the
# trajectory is causally attributable to backbone gradient flow.
#
# The hypothesis the v11g trajectory points at: the readout / writer /
# judge are converging to a useful state by step 400 (PA CB Δ_nm-m peak
# +0.036) but the trajectory reverses by step 1000 (PA CB Δ_nm-m -0.019)
# even on the *matched-distribution* phase-aligned eval. Two possible
# mechanisms compete:
#   (a) overfitting of the readout to the training distribution, or
#   (b) backbone co-evolution: as the backbone fine-tunes its block-
#       summary heads (b_0..b_{N-1}) for the LME chain task, the depth
#       softmax in `attention_parity` re-allocates mass to those better-
#       conditioned signals and away from m^t.
# (b) is structurally specific to attention_parity (block summaries
# compete with memory inside the same softmax). Freezing the backbone
# *eliminates* that co-evolution mechanically -- block summaries stay
# at their pretrained-init magnitude/content forever -- so any decay we
# still see comes from inside the memory subsystem itself (overfitting,
# multiplicative-coupling drift, etc.).
#
# A clean prediction: if α_mem_max stays growing past step 600 in v11l
# (vs decaying in v11g), the headline issue is (b) and the v11 paper
# needs an architectural intervention to break the b_k vs m^t softmax
# competition before the 4B run is meaningful. If α_mem_max still
# decays after step 600, (a) dominates and the fix is on the readout /
# regularisation side (memory dropout, contrastive aux, more data).
#
# Memory budget. Backbone fwd/bwd still runs (we need gradient to flow
# *through* it to reach memres), but optimizer state is dropped for the
# 596M frozen params (~1.2 GB at fp32 m/v) and bwd grads are not
# accumulated for them (~1.2 GB more). On the 94 GB GH200 this leaves
# ~22 GB of headroom vs v11g's ~24 GB peak; well within budget.
#
# Decision triggers (sharp, mirroring v11g for direct comparability)
# -------------------------------------------------------------------
#   step  200 : α_mem_max > 0  AND  ||m^t||/||embed|| ∈ [0.3, 50]
#   step  500 : α_mem_max > 5e-3  AND  pa_cb_dnm > +0.005
#                                AND  pa_cb_evidence_lift > +0.005
#   step 1000 : α_mem_max > 1e-2  AND  pa_cb_dnm > +0.020
#                                AND  pa_cb_dsh > +0.005
#                                AND  α_mem_max NOT decaying
#                                     (vs v11g step 1000 = 0.011 and
#                                     decaying)
#   step 2000 : standard Δ_sh-m > +0.005
#   KILL: step 1000 with α_mem_max < 1e-3.  Frozen backbone *and*
#         attention_parity collapsed -- rules out (b) entirely; the
#         decay is internal to memres and we escalate to a stronger
#         readout-regularisation cell.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
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
    --run_name chain_v11l_ap_frozen_backbone_gh200 \
    --out_dir output/chain_v11l_ap_frozen_backbone_gh200
