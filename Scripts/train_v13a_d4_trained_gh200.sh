#!/usr/bin/env bash
# v13a-d4-trained-gh200 (GH200 GPU 0) -- HEADLINE v13 recipe on GH200.
#
# Pushed from local (was train_v13a_d4_trained_local.sh) because the
# local H100s must keep run-time under 30 minutes and 4000 steps on D4
# takes ~2-3 hours.
#
# Stack (all three levers at once):
#   O (Objective):  writer_warmup for 500 steps, force mem_bias=4 and
#                   memory_gate=0.48 (gate force-open since simple_gate
#                   uses the per-sublayer gate, NOT the depth router,
#                   for the forward path -- this is the fix that was
#                   applied after the v13a_local attempt showed gate
#                   staying at zero under writer_warmup)
#   S (Symmetry):   memres_queries_init=orthogonal + slot_positional
#                   (break the permutation symmetry that drives the
#                   symmetric uniform fixed point)
#   R (Routing):    simple_gate (m^t bypasses the depth softmax, so
#                   it doesn't compete against the mature backbone
#                   residual stream)
#   W (Writer):     slot_attention (iter=3) so slots specialize via
#                   softmax-over-slots rather than softmax-over-inputs
#
# Decision triggers (same as v13a_local):
#   - PA-eval NLL below bare baseline (nomem on val) within 2000 steps
#   - evidence_lift > 0.1 by step 1500, > 0.3 by step 3000
#   - gate_mean visibly non-zero during phase 1 (>0.2), NOT drifting
#     back to 0 in phase 2 (should anneal to memres_gate_init=0
#     but recover under LM gradient)
#   - judge attention entropy reducing from ~log(8)=2.08 baseline
#     toward 1.0 by step 2000 on PA-eval batches
# STOP if any of these fails.
set -eu
cd "$(dirname "$0")"/..
mkdir -p output logs
[ -f "$HOME/venv/bin/activate" ] && . "$HOME/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -u train_chain.py \
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
    --run_name chain_v13a_d4_trained_gh200 \
    --out_dir output/chain_v13a_d4_trained_gh200
