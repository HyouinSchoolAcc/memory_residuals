#!/usr/bin/env bash
# v9d ABLATION — competition curriculum on attention_parity routing
#
# Question: now that the v8 RMSNorm-on-readout fix is in, does the
# original paper-spec attention_parity routing actually work under the
# competition curriculum? attention_parity collapsed catastrophically
# in v6/v7 (||m^t|| -> 0 due to W_V^read decay), but the diagnosis was
# made with the OLD architecture and the OLD curriculum. v9d retests it
# with both the architectural fix AND the competition-curriculum signal
# that v9 showed is enough to actually open the gate.
#
# If v9d works, it validates the paper-spec routing. If v9d still
# collapses, the simple_gate vs attention_parity choice is a permanent
# part of the recipe.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_mem_bias_init 4.0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
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
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v9d_abl_attnparity_gh200 \
    --out_dir output/chain_v9d_abl_attnparity_gh200 \
    2>&1 | tee logs/chain_v9d_abl_attnparity_gh200.log
