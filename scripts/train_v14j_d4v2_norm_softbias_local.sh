#!/usr/bin/env bash
# v14j_d4v2_norm_softbias_local (LOCAL H100 GPU 0) -- norm + softer warmup bias.
#
# What this cell tests
# --------------------
# v14g/v14h hold writer_warmup_router_bias=4.0 for 500 steps then
# anneal to 0 over 200 steps.  At bias=4 the router softmax forces
# alpha_mem ~= sigmoid(4) ~= 0.98, meaning every layer is fed ~all
# memory.  This is great for grad-flow into the writer, but it
# completely ignores the recent_bias prior the rest of the network
# learned.  Suspicion: the post-anneal step (700-1000) is when the
# router has to suddenly trust a near-uniform attention sink, and
# that's where alpha_mem can re-collapse to 0.
#
# Soft warm-start with bias=2.0 (alpha_mem ~ sigmoid(2) ~= 0.88) gives
# the writer enough gradient while letting the router softmax keep some
# probability mass on recent context, easing the anneal transition.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=0
exec python -u src/train_chain.py \
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
    --writer_warmup_router_bias 2.0 \
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
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v14j_d4v2_norm_softbias_local \
    --out_dir output/chain_v14j_d4v2_norm_softbias_local \
    2>&1 | tee logs/chain_v14j_d4v2_norm_softbias_local.log
