#!/usr/bin/env bash
# v21a_f3_d4_rezero_codes_0p6b_frozen_local (LOCAL GPU 0)
# v20a recipe + ReZero refine init. SINGLE-VARIABLE refinement of
# v20a, the best §5 cell in the project.
#
# v20a §5 cross-check landed at ttt_lift_vs_floor = +0.120 (writer
# init), which is by far the best §5 reading in the project history
# (vs v18a's +0.005, v19a's +0.009). The trained read-side at
# v20a's best/ checkpoint (step 700, peak evidence_lift = +0.0157)
# clearly HAS the architectural capacity to decode chain-specific
# recall. End-to-end evidence_lift then drifted negative for most of
# the rest of training, recovering to +0.0054 at step 1500 — the
# late-training instability that ReZero is designed to bound.
#
# v21a single-variable change vs v20a:
#   --memres_readout_refine_zero_init   (NEW; was off in v20a)
#
# Mechanism: each refine_W_V[i].weight is zero-initialised at
# construction so refinement layer i contributes RMSNorm_i(softmax(...)
# @ M_c @ 0) = 0 at step 0. The depth=4 stack starts at depth=0-
# equivalent ||m_t|| (smoke-tested: 0.038 vs depth=0's 0.037, vs
# depth=4 default's 0.082). refine_W_V[i] grows only as gradient
# demands; m_t magnitude tracks chain-specific signal rather than
# accumulating from random init.
#
# Predicted outcomes:
#   * §5 ttt_lift_vs_floor >= v20a's +0.120 AND end-to-end
#     evidence_lift > +0.02: ReZero stabilises the late-training
#     drift; v21a is the canonical recipe.
#   * §5 in [+0.05, +0.12] but evidence_lift > v20a's: tradeoff —
#     ReZero loses a bit of read-side capacity but gains end-to-end
#     stability. Worth keeping for production runs (1.7B, D4v2).
#   * §5 collapses to ~v18a (+0.005): ReZero blocked the gradient
#     flow through refine layers entirely (because the residual is
#     m_t = m_t + RMSNorm(softmax @ V); if V=0, the entire
#     RMSNorm input is 0 too, so the gradient on refine_W_V is
#     well-defined but goes through a chain of zeros and may take
#     hundreds of steps to escape). v22 fix: small non-zero init
#     scale=0.01 instead of 0.0.
#   * Slot collapse (pair/self < 0.01): ReZero starves the writer
#     of chain-specific gradient because m_t is trivially close to
#     m_t^(0) for the first ~hundred steps. Same v22 fix.
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v21a_f3_d4_rezero_codes_0p6b_frozen_local"
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
    --memres_readout_refine_zero_init \
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
    --run_name chain_v21a_f3_d4_rezero_codes_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v21a_f3_d4_rezero_codes_0p6b_frozen_local.log" 2>&1 &
echo "v21a launched pid=$!"
