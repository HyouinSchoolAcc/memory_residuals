#!/usr/bin/env bash
# v11i (GH200, 0.6B) -- attention_parity +4/-4 (legacy v3/v10b bias)
#
# A/B against v11g_ap_baseline. The ONLY knob change is
#   router_mem_bias_init  0  ->  -4
#
# At init, alpha_mem ~ exp(-4) / (exp(-4) + sum(1) + exp(4))
#                    ~ 1.7e-4 (~50x lower than v11g's ~9e-3)
#
# This is the v3 / v6 / v10b setting. The user asserted "+4/0 is better
# than v3" -- this cell tests the converse: with the data fix (P0) and
# magnitude fix (P2) both in place, can the v3 default bias still
# recover, or does it stay collapsed for the duration?
#
# Decision triggers:
#   step 500  : if alpha_mem_max < 1e-3 (< its init value) -> the
#               curriculum is failing to push memory open even with
#               evidence labels; v3 bias is structurally too tight
#   step 1000 : compare pa_cb_dnm to v11g; if v11g >> v11i, user's
#               claim is verified
#   step 2000 : if alpha_mem trajectory still flat, mark v11i FAILED
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init -4 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
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
    --run_name chain_v11i_ap_pm4_gh200 \
    --out_dir output/chain_v11i_ap_pm4_gh200
