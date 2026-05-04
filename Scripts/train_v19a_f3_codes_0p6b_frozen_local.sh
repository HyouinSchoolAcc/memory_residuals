#!/usr/bin/env bash
# v19a_f3_codes_0p6b_frozen_local (LOCAL GPU 0)
# F3 (ReadoutProbeHead) + multi-layer readout depth=2 + strong alpha-floor.
#
# v18 verdict recap (results/exp2_chain_recipe/runs.md, 2026-05-03 17:00 UTC-5):
#   * v18a (F3 only) shifted §5 ttt_lift_vs_floor from -0.897 (v15a, no
#     intervention) -> -0.337 (v17a, F2 only) -> +0.005 (v18a) — the
#     largest causal shift in the §5 series and the first non-NEG.
#   * F3 alone is the right intervention; F2 + F3 compose worse than
#     F3 alone (writer-probe and readout-probe pull M_c along
#     different value-space directions).
#   * v18a's MIXED §5 reading (+0.005) sits at the boundary of the
#     architecture's read-side capacity. The single-layer
#     MemoryReadout is at the edge of capacity for the chain-specific
#     callback lookup.
#   * alpha_mem stayed at ~0.022 vs target 0.05 with weight 0.01;
#     the depth router was below the floor target throughout.
#
# v19a fix (single-variable A vs v18a):
#   --memres_readout_depth 2  stacks 2 additional residual cross-attn
#   refinement layers on top of the base MemoryReadout. Each layer
#     m_t^(i) = m_t^(i-1) + RMSNorm_i(softmax((X+m_t^(i-1)) W_Q_i,
#                                             M_c W_K_i) @ M_c W_V_i)
#   gives the readout iterative Perceiver-style refinement against
#   M_c, conditioning later layers on the partial m_t accumulated so
#   far. Init parity preserved by MemoryGate zero-init downstream
#   (verified: max|d|=0.00 vs base for depth in {0,2,4}).
#
#   --alpha_mem_floor_aux_weight 0.5 --alpha_mem_floor_target 0.10
#   strengthens the floor 50x in weight and 2x in target so the
#   depth router actually sits at the prescribed open level rather
#   than sneaking under it. With v19's read-side probe loss
#   pressuring m_t to be chain-specific, the floor is supporting
#   (not gameable).
#
# Single-variable test rationale (multi-layer readout isolated):
#   v19a: F3 + readout_depth=2 + strong floor.
#   v19b: F3 + readout_depth=4 + strong floor (capacity ceiling test).
#   Compare v19a vs v18a to isolate readout-depth contribution; v19b
#   vs v19a to bound the capacity ceiling.
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v19a_f3_d2_codes_0p6b_frozen_local"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 nohup python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 --router_mem_bias_init 0 \
    --memres_update_mode gated --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --memres_readout_depth 2 \
    --writer_warmup_steps 0 --writer_warmup_router_bias 0.0 --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --readout_probe_enabled \
    --readout_probe_loss_weight 1.0 \
    --readout_probe_warmup_steps 200 \
    --alpha_mem_floor_aux_weight 0.5 \
    --alpha_mem_floor_target 0.10 \
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --lr_backbone 0 --steps 1500 --warmup 200 --max_norm 1.0 \
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
    --run_name chain_v19a_f3_d2_codes_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v19a_f3_d2_codes_0p6b_frozen_local.log" 2>&1 &
echo "v19a launched pid=$!"
