#!/usr/bin/env bash
# v9b ABLATION — half competition + half mixed-bias (was pure competition)
#
# Question: is PURE competition curriculum necessary, or is half competition
# enough to recruit the judge? Sets curriculum_competition_bias=0.5 and
# curriculum_evidence_bias=0.5, so windows split:
#   * 50% paired competition windows (Sample A keep-prev, Sample B write-new)
#   * 25% P0 evidence-callback windows (window_k=2)
#   * 25% full window_k contiguous chains
# Tests whether the competition curriculum has to dominate the training mix
# to break readout-discrimination, or whether even minority exposure is
# enough.
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
    --window_k 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 200 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.5 \
    --curriculum_competition_bias 0.5 \
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
    --run_name chain_v9b_abl_mixed_gh200 \
    --out_dir output/chain_v9b_abl_mixed_gh200 \
    2>&1 | tee logs/chain_v9b_abl_mixed_gh200.log
