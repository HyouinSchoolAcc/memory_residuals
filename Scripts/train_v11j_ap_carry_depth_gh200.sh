#!/usr/bin/env bash
# v11j (GH200, 0.6B) -- attention_parity +4/0 with deeper recurrence
#
# A/B against v11g_ap_baseline. Adds:
#   --window_k 4              (4 sessions per backprop window, was 3)
#   --carry_state             (M_c persists across windows in the chain)
#   --burn_in_max 12          (consume up to 12 sessions silently before
#                              the loss window, building real recurrent
#                              context)
#   --burn_in_resample        (resample burn-in length per chain so the
#                              model sees recurrence at depths {0,...,12})
#
# Tests P5: the v9-v10 audit found that even the cells where alpha_mem
# opened never moved the standard Δ_sh-m metric, because eval ran at
# depths the model had never seen during training (eval gathers up to
# eval_window=8 sessions but training only saw burn_in_max=0 + window_k=3).
# This cell explicitly closes that gap. If the standard Δ_sh-m starts
# trending positive while v11g stays flat, P5 is verified.
#
# Decision triggers (the same content-quality bar as v11g, plus depth):
#   step 1000 : alpha_mem_max > 1e-2 AND pa_cb_dnm > +0.020
#   step 2000 : standard Δ_sh-m > +0.005 (DEPLOYMENT-distribution
#               memory benefit, not just phase-aligned)
#   step 4000 : standard Δ_sh-m > +0.020 (graduates to a real result)
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --train_chains paper_artifacts/chains/v11_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 4 \
    --carry_state \
    --burn_in_max 12 \
    --burn_in_resample \
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
    --run_name chain_v11j_ap_carry_depth_gh200 \
    --out_dir output/chain_v11j_ap_carry_depth_gh200
