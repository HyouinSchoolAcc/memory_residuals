#!/usr/bin/env bash
# v10 4B MEGA CORPUS + ATTENTION_PARITY (+4/-4) + L_E=10 (GH200, 3 DAYS)
#
# The headline run. Central hypothesis: data diversity is the
# load-bearing axis that determines whether the memory channel
# survives training (v9c diverse was the only v9 ablation with good
# survivability; the v8/v9 LME-only cells drifted/collapsed). Scale
# diversity up by ~100x (8+ dialogue + narrative sources, ~300k
# chains) and train a deep extraction stack (L_E=10, 11-layer
# Perceiver) on Qwen3-4B with the original paper-spec attention_parity
# routing at softer +4/-4 biases (v3 default; 700x softmax advantage
# to recent at init but with non-trivial gradient to relax it).
#
# Backbone choice: Qwen3-4B (not 8B) because qwen3-8b-xlarge (L_E=10)
# hits ~106 GB peak HBM under full AdamW on 8.8B params -- beyond the
# 96 GB GH200 budget. Qwen3-4B-xlarge (4.3B total) peaks at ~52 GB
# with room for activations + gradient checkpointing. Preserves the
# user's direction of "full backbone training, shrink the model if
# it doesn't fit." Stepping up to 8B requires bitsandbytes (--use_adam8bit).
#
# Compute budget: 3 days × 86400 s = 259200 s. At an expected ~2000-
# 3000 tok/s on 4B+memres with gradient checkpointing, bs=2 ga=4
# window_k=4 session_len=512 => ~16k tok/step => 5-8 s/step. Plan:
# 30000 steps (~40-70 h), remaining budget absorbed by eval cycles.
#
# PREFLIGHT: the mega corpus must already exist. Run
# scripts/build_mega_corpus_gh200.sh first (idempotent; ~2 h build
# time). The watchdog spec for this cell runs the build first if the
# .pt file is missing.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs

CORPUS=${CORPUS:-paper_artifacts/chains/mega_train_s512.pt}
if [ ! -f "$CORPUS" ]; then
    echo "[v10-4b] corpus $CORPUS not found; running build_mega_corpus_gh200.sh"
    bash scripts/build_mega_corpus_gh200.sh
fi

exec python -u train_chain.py \
    --preset qwen3-4b-xlarge \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init -4 \
    --memres_update_mode gated \
    --memres_extract_source hidden_18 \
    --train_chains "$CORPUS" \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 3.0, "realtalk": 2.0, "ultrachat": 2.0, "pippa": 2.5, "soda": 1.5, "synthdlg": 1.5, "lmsys": 1.5}' \
    --window_k 4 \
    --carry_state \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 3e-5 \
    --lr_backbone 5e-6 \
    --steps 30000 \
    --warmup 500 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.3 \
    --curriculum_evidence_bias 0.3 \
    --curriculum_competition_bias 0.5 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 500 \
    --save_every 1000 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v10_4b_mega_attnparity_gh200 \
    --out_dir output/chain_v10_4b_mega_attnparity_gh200 \
    2>&1 | tee logs/chain_v10_4b_mega_attnparity_gh200.log
