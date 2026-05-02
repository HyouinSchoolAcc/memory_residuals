#!/usr/bin/env bash
# v13d-d4-ablate-warmup-gh200 (GH200 GPU 0) -- v13 full stack MINUS
# writer warmup.  Tests whether orthogonal + slot_positional +
# simple_gate + slot_attention *alone* breaks the uniform attractor,
# isolating the symmetry-break (S) and routing (R) levers from the
# objective (O) lever.
#
# Single-knob difference vs v13a_d4_trained_local:
#   v13a:  --writer_warmup_steps 500
#   v13d:  --writer_warmup_steps 0  (no warmup, joint training from step 0)
#
# Intent:
# - Isolates O's contribution.  If v13d plateaus at v12a-level (uniform
#   attractor at step 800-1000, evidence_lift ~ 0), then writer_warmup
#   is the load-bearing lever of the v13 stack.
# - If v13d succeeds, writer_warmup might be an "acceleration" rather
#   than a structural fix, and future work might be able to drop the
#   warmup phase (saving ~10% wall-clock per run).
# - Either way, v13d settles a structural question about what the
#   minimal v13 recipe actually is.
#
# Hyperparameters copy v13a exactly except --writer_warmup_steps 0.
set -eu
cd "$(dirname "$0")"/..
mkdir -p output logs
[ -f "$HOME/venv/bin/activate" ] && . "$HOME/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --router_recent_bias_init 0 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention \
    --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal \
    --memres_slot_positional \
    --writer_warmup_steps 0 \
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
    --run_name chain_v13d_d4_ablate_warmup_gh200 \
    --out_dir output/chain_v13d_d4_ablate_warmup_gh200
