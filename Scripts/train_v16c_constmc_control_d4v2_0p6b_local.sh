#!/usr/bin/env bash
# v16c_constmc_control_d4v2_0p6b_local (LOCAL H100 GPU 0)
#
# CONSTANT-M_c CONTROL on the SAME D4v2 corpus v14k won on.
# Single shared learnable parameter replaces the per-chain compressed
# M_c. Writer / extract / judge are bypassed entirely; only the
# readout, router, and the const_M_c parameter receive gradient.
#
# Predicted result if v14k's pa_cb_dnm was a learnt content-blind
# output prior: this control should reproduce v14k's pa_cb_dnm
# (~+1.4 nats on D4v2) within noise, while pa_cb_evidence_lift stays
# at exactly zero by construction (no chain-specific input ever
# reaches M_c). That outcome is the smoking gun we need before
# rewriting Architectural Prior #9 and the v15 OPEN AUDIT entry.
#
# Predicted result if v14k actually retrieves: this control's
# pa_cb_dnm should be a small fraction of v14k's (whatever the LM
# head can wring out of a content-blind 128-vector prior tensor).
#
# Local box, single H100 NVL (96 GB). Step budget shortened to 2000
# (no real signal expected past saturation of the prior).
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
    --memres_writer_kind slot_attention \
    --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal \
    --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --writer_warmup_steps 0 \
    --writer_warmup_router_bias 0.0 \
    --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --alpha_mem_floor_aux_weight 0.0 \
    --contrastive_infonce_weight 0.0 \
    --train_chains paper_artifacts/chains/synthd4v2_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 2000 \
    --warmup 200 \
    --max_norm 1.0 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --curriculum_competition_bias 0.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --constant_mc_control \
    --eval_every 100 \
    --save_every 1000 \
    --eval_n_chains 24 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 32 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v16c_constmc_control_d4v2_0p6b_local \
    --out_dir output/chain_v16c_constmc_control_d4v2_0p6b_local \
    2>&1 | tee logs/chain_v16c_constmc_control_d4v2_0p6b_local.log
