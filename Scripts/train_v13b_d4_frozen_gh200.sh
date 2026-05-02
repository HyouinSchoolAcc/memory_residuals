#!/usr/bin/env bash
# v13b-d4-frozen-gh200 (GH200 GPU 0) -- full v13 stack with FROZEN
# backbone, stacked on top of v12d_d4_frozen's first-ever positive
# evidence_lift signal.
#
# Pushed from local (was train_v13b_d4_frozen_local.sh) because the
# local H100s must keep run-time under 30 minutes and 4000 steps takes
# ~2-3 hours on either H100 or GH200.
#
# Difference vs v13a_d4_trained_gh200:
#   v13a:  backbone trainable at lr_backbone=2e-5, writer_warmup
#          unfreezes backbone at step 500
#   v13b:  backbone PERMANENTLY FROZEN via --freeze_backbone and
#          --writer_warmup_keep_backbone_frozen (phase 2 keeps
#          backbone frozen)
#
# Rationale: v12d_d4_frozen was the only run in the v12 campaign that
# showed positive evidence_lift (+0.03) -- small but the sign is right.
# A fully-frozen backbone forces all learning into the memres
# subsystem, so orth+pos+writer_warmup should amplify that signal.
# If v13b DOESN'T beat v12d_d4_frozen by >3x evidence_lift, then the
# memres subsystem has an architectural ceiling that changing the
# backbone training regime can't address.
#
# Decision triggers (frozen-backbone version):
#   - evidence_lift > 0.1 by step 1500 (vs v12d's +0.03)
#   - PA-eval NLL strictly below bare baseline (nomem=5.5 on D4 val)
#     within 2500 steps
#   - gate_mean > 0.2 DURING phase 1 (force-opened), staying non-zero
#     in phase 2 under LM gradient alone (no force-open)
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
    --writer_warmup_keep_backbone_frozen \
    --freeze_backbone \
    --train_chains paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 0 \
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
    --run_name chain_v13b_d4_frozen_gh200 \
    --out_dir output/chain_v13b_d4_frozen_gh200
