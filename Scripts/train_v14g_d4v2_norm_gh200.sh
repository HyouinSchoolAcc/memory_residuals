#!/usr/bin/env bash
# v14g_d4v2_norm_gh200 (GH200 GPU 0) -- HEADLINE FIX TEST.
#
# What this cell tests
# --------------------
# Hypothesis: the current v14 collapse on D4 is caused by the
# unnormalised context tensor C that MemoryBlock.extract receives when
# memres_extract_source = "hidden_14".  C carries the raw residual
# stream (RMS ~ 50 in fp16) which inflates extract.W_K / extract.W_V
# backward gradients by the same factor and forces the global-norm
# clipper (max_norm=1.0) to scale every memory parameter update down to
# noise.  Adding a single Qwen3RMSNorm on C before the cross-attention
# (or slot-attention) extractor should reduce extract.W_K backward
# gradient magnitude by ~50x and let alpha_mem actually accumulate
# gradient.
#
# This cell vs the broken reference (v14abl_a):
#   diff vs v14abl_a_full_d4_local:
#     +  --memres_extract_input_norm
#     swapped corpus -> D4v2 (double-core: n_evidence_sessions=2)
#     eval_every 200 -> 100 (catch early collapse vs recovery)
#
# The headline pair on GH200 is v14g (norm ON) vs v14h (norm OFF).
# Both run on D4v2 with frozen backbone, slot_attention writer, judge
# QK-LN, alpha_floor 0.05@0.01, InfoNCE 0.5 callback-only, warmup
# router bias 4.0 over 500 steps -> anneal 200.
#
# Decision triggers (4000-step budget, D4v2, frozen backbone):
#   step 200:  warmup phase -- gate_mean and ||m^t|| should NOT be
#              collapsing toward 0.  alpha_mem held at warmup_router_bias
#              forced level; we want extract.W_K |grad|_2 < 5
#              (vs broken v14abl_a where it was ~1e11 unclipped).
#   step 700:  anneal complete -- callback NLL should not have spiked,
#              alpha_mem_mean > 0.04 (above floor target 0.05 threshold
#              with some slack), pair self < 0.05.
#   step 1500: evidence_lift > +0.05 (writer is using callback-relevant
#              evidence).
#   step 4000: pa_cb_dsh > +0.05 (D4v2 callback is harder than D4 due
#              to double evidence).  KILL @ step 1500 if alpha_mem_mean
#              < 0.01 for 3 consecutive evals (norm fix didn't unblock
#              the writer).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
[ -f "$HOME/venv/bin/activate" ] && . "$HOME/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention \
    --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal \
    --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --writer_warmup_steps 500 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 200 \
    --writer_warmup_keep_backbone_frozen \
    --freeze_backbone \
    --alpha_mem_floor_aux_weight 0.01 \
    --alpha_mem_floor_target 0.05 \
    --contrastive_infonce_weight 0.5 \
    --contrastive_infonce_temperature 1.0 \
    --contrastive_infonce_callback_only \
    --contrastive_infonce_initial_weight 0.0 \
    --contrastive_infonce_warmup_steps 500 \
    --train_chains paper_artifacts/chains/synthd4v2_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 4000 \
    --warmup 300 \
    --max_norm 1.0 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 100 \
    --save_every 1000 \
    --eval_n_chains 24 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 32 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v14g_d4v2_norm_gh200 \
    --out_dir output/chain_v14g_d4v2_norm_gh200
