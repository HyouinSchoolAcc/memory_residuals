#!/usr/bin/env bash
# v31a: v27b recipe + uniform InfoNCE contrastive (Tier-1 (1) of the friend's plan).
# Single flag added: --contrastive_infonce_weight 0.5 (with warmup).
# Diff from v27b/seed1: --contrastive_infonce_weight 0.5,
# --contrastive_infonce_warmup_steps 200 (let LM-NLL stabilise first),
# --contrastive_infonce_callback_only (default; concentrate gradient on
# the tokens that actually require chain-binding).
#
# Hypothesis: contrastive should lift Δ_sh on LME by directly punishing
# off-diagonal cross-chain CE, while leaving Δ_cb intact (LM-NLL is
# still primary). At B=4 the cost is +B*B=+16 last-session forwards
# (~+50% per-step), so 1500 steps will take ~2 h instead of 1.5 h.
#
# Conditional on Tier-2 (4) succeeding: this is the friend's safest
# follow-on. Conditional on Tier-2 (4) failing: this is still worth a
# single-cell sanity test, because the per-category diagnostic showed
# hidden chain-conditional signal already exists in the writer at
# 1.7B; contrastive should amplify it without needing the architecture
# to bind literal facts.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v31a_lme_infonce_seed1_0p6b_frozen_local"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
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
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --lr_backbone 0 --steps 1500 --warmup 200 --max_norm 1.0 \
    --memory_dropout 0.10 --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --contrastive_infonce_weight 0.5 \
    --contrastive_infonce_warmup_steps 200 \
    --contrastive_infonce_temperature 1.0 \
    --callback_loss_weight 5.0 --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 --curriculum_competition_bias 0.0 \
    --burn_in_max 0 --mask_padding_loss --score_tail_frac 1.0 \
    --mask_evidence_session_loss \
    --kill_on_memory_collapse --kill_on_memory_collapse_min_step 200 \
    --eval_every 100 --save_every 500 \
    --eval_n_chains 24 --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric evidence_lift \
    --seed 1 \
    --run_name chain_v31a_lme_infonce_seed1_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v31a_lme_infonce_seed1_0p6b_frozen_local.log" 2>&1 &
echo "v31a launched pid=$!"
