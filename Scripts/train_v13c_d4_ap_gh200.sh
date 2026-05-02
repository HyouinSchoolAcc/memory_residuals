#!/usr/bin/env bash
# v13c-d4-ap-gh200 (GH200 GPU 0) -- v13 full stack BUT keeping
# attention_parity routing (m^t stays in the Eq. 9 depth softmax, per
# the PDF spec-strict directive).
#
# Single-knob difference vs v13a_d4_trained_local:
#   v13a:  --memres_mode simple_gate  (m^t out of depth softmax)
#   v13c:  --memres_mode attention_parity  +  --router_recent_bias_init 4
#          --router_mem_bias_init 0  (v11g / v12d softer-bias recipe)
#
# Intent:
# - Ablation for recommendation #4 (take m^t out of the depth softmax).
#   If v13c hits the same metrics as v13a, then the routing change
#   wasn't load-bearing and the spec's Eq. 9 survives.
# - If v13a hits metrics and v13c plateaus at v12d_frozen-level
#   (evidence_lift ~ 0.03), then simple_gate is architecturally
#   required and the paper's Section 2.2 needs a revision.
# - Answers the single most important spec question in the campaign:
#   "can writer_warmup + symmetry break rescue attention_parity, or
#   is simple_gate the only path?"
#
# Uses the same decision triggers as v13a.  Runs on GH200 because the
# two local H100s are occupied by v13a/v13b.
#
# Hyperparameters copy v13a EXCEPT memres_mode and the AP-specific
# router bias inits.
set -eu
cd "$(dirname "$0")"/..
mkdir -p output logs
[ -f "$HOME/venv/bin/activate" ] && . "$HOME/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention \
    --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal \
    --memres_slot_positional \
    --writer_warmup_steps 500 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 200 \
    --train_chains paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
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
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v13c_d4_ap_gh200 \
    --out_dir output/chain_v13c_d4_ap_gh200
