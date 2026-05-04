#!/usr/bin/env bash
# v23c_v21c_seed7: multi-seed v21c reproducibility check.
# v21c (probe weight 0.5, depth=4, strong floor 0.5/0.10) ended at
# evidence_lift = +0.0241 at step 1000 (default seed=42). v22a/v22c
# bracketed at probe weight 0.4/0.6 and ended at -0.003/+0.005. The
# probe-weight curve is NOT smooth in [0.3, 0.6] — v21c's reading
# could be (a) a true local optimum at probe 0.5, (b) a favorable
# random seed at any probe weight in that range, or (c) both.
# v23{a,b,c} re-run v21c verbatim at seeds {1,2,7} on local
# H100/cloud GH200 in parallel.
#
# Decision rule (after all 4 seeds {1,2,7,42} land):
#   * 95% CI of evidence_lift_final excludes 0 -> v21c is the
#     reproducible recipe; canonical for v24+ (backport to D4v2 + 1.7B)
#   * CI includes 0 -> v21c was favorable seed; need v24 architectural
#     change before backporting
#   * Spread > 0.05 nats -> high noise floor; need writer-side aux
#     loss or larger corpus to reduce variance before claiming the
#     architecture works
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v23c_v21c_seed7_codes_0p6b_frozen_gh200"
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
    --seed 7 \
    --run_name chain_v23c_v21c_seed7_codes_0p6b_frozen_gh200 \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v23c_v21c_seed7_codes_0p6b_frozen_gh200.log" 2>&1 &
echo "v23c launched pid=$!"
