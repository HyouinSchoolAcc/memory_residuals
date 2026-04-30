#!/usr/bin/env bash
# v9 PURE-COMPETITION SIMPLE_GATE + readout RMSNorm (LME-only, GH200)
#
# GH200 variant of the v9 cell. Uses LME-only training data so the
# comparison vs v8b (also LME-only) isolates the CURRICULUM DESIGN
# axis (competition pairs vs evidence-callback chains) from the data
# diversity axis (which v8c separately tests on local GPU 0).
#
# See scripts/train_v9_compete_simplegate_rmsnorm.sh header for the
# full curriculum-decomposition rationale.
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
    --run_name chain_v9_compete_lme_gh200 \
    --out_dir output/chain_v9_compete_lme_gh200 \
    2>&1 | tee logs/chain_v9_compete_lme_gh200.log
