#!/usr/bin/env bash
# v15a_d4v2_norm_replicate_local (LOCAL H100 GPU 0) -- v14k replicate.
#
# Sanity replicate of the v14k recipe (no warmup, norm ON, frozen
# backbone) with a different random seed and the same data.  v14k was
# the only cell of the v14g..v14l ablation that delivered a sustained
# positive evidence_lift on D4v2 (peak +0.13 at step 2600).  Before
# building any larger experiment around the recipe we want to confirm
# that signal is reproducible rather than seed-lucky.
#
# Step budget shortened to 2500 (v14k plateaued by then).
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
    --writer_warmup_steps 0 \
    --writer_warmup_router_bias 0.0 \
    --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --alpha_mem_floor_aux_weight 0.01 \
    --alpha_mem_floor_target 0.05 \
    --contrastive_infonce_weight 0.5 \
    --contrastive_infonce_temperature 1.0 \
    --contrastive_infonce_callback_only \
    --contrastive_infonce_initial_weight 0.5 \
    --contrastive_infonce_warmup_steps 0 \
    --train_chains paper_artifacts/chains/synthd4v2_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 2500 \
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
    --run_name chain_v15a_d4v2_norm_replicate_local \
    --out_dir output/chain_v15a_d4v2_norm_replicate_local \
    2>&1 | tee logs/chain_v15a_d4v2_norm_replicate_local.log
