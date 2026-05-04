#!/usr/bin/env bash
# v21c_f3_d4_pw05_codes_0p6b_frozen_local (LOCAL GPU 1)
# v20a recipe + probe weight 0.5. SINGLE-VARIABLE probe-weight sweep
# between v18a's 1.0 and v20a's 0.3.
#
# v20a (probe weight 0.3) gave the best §5 reading (+0.120). v18a/
# v19a/v19b (probe weight 1.0) gave §5 in [-0.004, +0.009]. We
# don't know whether 0.3 is the optimum or whether 0.5 is better.
# The two cells together let us interpolate the probe-weight curve
# at v20a's other settings (depth=4, strong floor 0.5/0.10).
#
# This is the original v20a recipe with one number changed:
#   --readout_probe_loss_weight 0.5   (was 0.3 in v20a)
#
# Predicted outcomes:
#   * v21c §5 > v20a's +0.120: probe weight 0.5 is closer to the
#     optimum than 0.3. Probe-weight sweep extends to 0.7 in v22.
#   * v21c §5 in [+0.05, +0.12]: 0.5 is on the chain-fingerprinting
#     side of the operating point; v20a's 0.3 stays the recipe.
#   * v21c §5 negative: probe weight 0.5 + depth=4 falls back into
#     the chain-fingerprinting trap that doomed v19b at probe 1.0.
#     v22 should explore probe weight 0.1, 0.2 (lower than 0.3).
#
# Note: paired with v21a (v20a + ReZero) on GPU 0. Together they
# cover the two main directions for refining the v20a recipe:
# better-conditioned magnitude (v21a) vs better-tuned probe (v21c).
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v21c_f3_d4_pw05_codes_0p6b_frozen_local"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=1 nohup python -u src/train_chain.py \
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
    --readout_probe_loss_weight 0.5 \
    --readout_probe_warmup_steps 200 \
    --alpha_mem_floor_aux_weight 0.5 \
    --alpha_mem_floor_target 0.10 \
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --lr_backbone 0 --steps 1000 --warmup 200 --max_norm 1.0 \
    --memory_dropout 0.10 --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
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
    --run_name chain_v21c_f3_d4_pw05_codes_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v21c_f3_d4_pw05_codes_0p6b_frozen_local.log" 2>&1 &
echo "v21c launched pid=$!"
