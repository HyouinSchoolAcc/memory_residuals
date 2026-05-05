#!/usr/bin/env bash
# v14abl_d_no_anneal_d4_local (LOCAL H100 GPU 1, second pair) --
# v14 full minus AP router_mem_bias warmup anneal.  Isolates: does
# force-holding mem_bias=+4 during writer_warmup (then annealing
# back) actually rescue alpha_mem during phase 1, or is it
# cosmetic?  Disabled by setting --writer_warmup_router_bias 0.0;
# router_mem_bias then stays at 0 (== --router_mem_bias_init) for
# the entire warmup window, no hold, no anneal.  Other warmup
# behaviour (frozen-backbone-during-warmup + memres-only training)
# remains.  See v14abl_a header for locked-in core.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=1
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
    --memres_judge_qk_layernorm \
    --writer_warmup_steps 500 \
    --writer_warmup_router_bias 0.0 \
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
    --train_chains paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 4000 \
    --warmup 300 \
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
    --run_name chain_v14abl_d_no_anneal_d4_local \
    --out_dir output/chain_v14abl_d_no_anneal_d4_local \
    2>&1 | tee logs/chain_v14abl_d_no_anneal_d4_local.log
