#!/usr/bin/env bash
# v16b_codes_1p7b_frozen_gh200 (GH200 GPU 0) -- D5 + 1.7B scaling.
#
# Same recipe as v16a (D5 random_codes, evidence-mask LM loss,
# save_best=evidence_lift) but on the qwen3-1.7b-large preset.
# Companion to v16a: lets us read off whether the writer/readout
# benefit from a bigger backbone on the corpus that *forces* the
# memory pathway.
#
# Why we need both 0.6B and 1.7B on D5:
#   * v15e (1.7B frozen on D4v2) showed pa_cb_evidence_lift go
#     NEGATIVE while pa_cb_dnm climbed -- ie the larger writer
#     actively overfit the dataset's per-category prior so hard that
#     real evidence content became adversarial. If on D5 the 1.7B
#     fixes that (evidence_lift positive and growing with scale), the
#     v15e signature is confirmed as a v15-corpus pathology rather
#     than a writer-architecture pathology, and we can move forward
#     to 4B with confidence the memory pathway is the load-bearing
#     mechanism.
#   * If on D5 the 1.7B *also* shows evidence_lift << pa_cb_dnm, then
#     the writer/readout has a deeper degeneracy that the corpus fix
#     alone doesn't cure, and the architecture itself needs revision
#     before scaling further.
#
# Step budget bumped to 3500 to give the larger model a fair chance
# of recovery; warmup 400 to avoid the early-step oscillation we saw
# on v15e at 300 warmup with the bigger writer.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
[ -f "$HOME/venv/bin/activate" ] && . "$HOME/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -u train_chain.py \
    --preset qwen3-1.7b-large \
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
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 3500 \
    --warmup 400 \
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
    --mask_evidence_session_loss \
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
    --save_best_metric evidence_lift \
    --run_name chain_v16b_codes_1p7b_frozen_gh200 \
    --out_dir output/chain_v16b_codes_1p7b_frozen_gh200 \
    2>&1 | tee logs/chain_v16b_codes_1p7b_frozen_gh200.log
