#!/usr/bin/env bash
# v28d: v25a recipe (1.7B, LME, depth=4, floor=0.5) but F3 readout-probe OFF.
# Tests whether the v27b-seed4 +0.798 finding (F3 off > F3 on at 0.6B)
# scales to 1.7B. Same backbone as v25a (frozen), same training corpus,
# same step budget.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v28d_no_probe_seed4_1p7b_frozen_gh200"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u src/train_chain.py \
    --preset qwen3-1.7b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 --router_mem_bias_init 0 \
    --memres_update_mode gated --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --memres_readout_depth 4 \
    --writer_warmup_steps 0 --writer_warmup_router_bias 0.0 --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --readout_probe_enabled \
    --readout_probe_loss_weight 0.0 \
    --readout_probe_warmup_steps 200 \
    --alpha_mem_floor_aux_weight 0.5 \
    --alpha_mem_floor_target 0.10 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 3 --batch_size 2 --grad_accum 4 \
    --lr 1e-4 --lr_backbone 0 --steps 1500 --warmup 300 --max_norm 1.0 \
    --memory_dropout 0.10 --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 5.0 --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 --curriculum_competition_bias 0.0 \
    --burn_in_max 0 --mask_padding_loss --score_tail_frac 1.0 \
    --mask_evidence_session_loss \
    --kill_on_memory_collapse --kill_on_memory_collapse_min_step 200 \
    --eval_every 100 --save_every 500 \
    --eval_n_chains 24 --eval_window 8 \
    --phase_aligned_eval_n_chains 32 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric evidence_lift \
    --seed 4 \
    --run_name chain_v28d_no_probe_seed4_1p7b_frozen_gh200 \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v28d_no_probe_seed4_1p7b_frozen_gh200.log" 2>&1
echo "v28d finished"
