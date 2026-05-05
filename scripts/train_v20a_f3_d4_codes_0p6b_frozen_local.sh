#!/usr/bin/env bash
# v20a_f3_d4_codes_0p6b_frozen_local (LOCAL GPU 0)
# F3 + readout depth=4 + STRONG floor + REDUCED probe weight (0.3).
#
# v19 verdict recap:
#   * v19a (depth=2 + strong floor + standard probe) KILLED at step
#     500 by --kill_on_memory_collapse: pair/self collapsed to 0.007
#     (slot writer fell into v13-era uniform fixed point).
#   * v19b (depth=4 + strong floor + standard probe) ran to step
#     1500 cleanly but in the WRONG REGIME: pair/self = 0.644 (writer
#     extremely chain-distinguishable), router opening (alpha_mem_max
#     = 0.20, frac_open = 0.16), readout producing huge m_t
#     (||m^t||/||embed|| = 12.7), but evidence_lift = -0.016 and
#     mem CE = 3.10 vs nomem CE = 1.14 — memory DOMINATES the LM
#     head's predictions but in a chain-fingerprinting direction
#     that hurts next-token CE.
#
# Diagnosis: the F3 probe loss at weight 1.0 + depth=4's amplified
# m_t supervises the readout to encode chain-identity signal at the
# first answer token position, and that signal dominates what the LM
# head reads. The LM-NLL gradient's signal is too weak relative to
# the F3 supervision once the readout has 4 layers' worth of
# capacity to amplify the probe-driven content.
#
# v20a fix (single-variable A vs v19b):
#   --readout_probe_loss_weight 0.3   (was 1.0)
#   Same depth=4 + strong floor recipe as v19b, but the F3 probe
#   contribution to total_loss is reduced 70%, which means the LM-
#   NLL pathway has comparable weight in determining what m_t
#   encodes. Probe still installs the chain-specific gradient
#   channel into MemoryReadout, but doesn't dominate.
#
# Single-variable test rationale: v20a vs v19b isolates probe-weight
# as the cause of v19b's "memory dominates LM, in wrong direction"
# regime. v20b runs the floor-weight ablation in parallel.
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v20a_f3_d4_pw03_codes_0p6b_frozen_local"
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
    --memres_readout_depth 4 \
    --writer_warmup_steps 0 --writer_warmup_router_bias 0.0 --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --readout_probe_enabled \
    --readout_probe_loss_weight 0.3 \
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
    --run_name chain_v20a_f3_d4_pw03_codes_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v20a_f3_d4_pw03_codes_0p6b_frozen_local.log" 2>&1 &
echo "v20a launched pid=$!"
