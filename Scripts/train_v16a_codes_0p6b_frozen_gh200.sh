#!/usr/bin/env bash
# v16a_codes_0p6b_frozen_gh200 (GH200 GPU 0) -- Vision-aligned diagnostic.
#
# WHY THIS CELL EXISTS (not just another ablation):
#   v14k+v15a+v15e all post pa_cb_dnm in the +0.4..+1.4 nats range while
#   their pa_cb_evidence_lift sits at +0.05..+0.13 (and even goes
#   negative on v15e). The architectural prior 'memory carries the
#   binding' predicts pa_cb_evidence_lift ~~ pa_cb_dnm; the empirical
#   ratio on D4v2 is ~5%. That gap is a *learnt content-blind output
#   prior*, not retrieval. Decisive intervention: change the corpus so
#   the dataset's per-callback marginal CANNOT carry that 95%, and
#   change the loss so the LM head is never directly supervised on the
#   binding template inside an evidence session.
#
# Two changes vs v14k recipe (everything else identical):
#   1. --train/eval_chains: synthd5_random_codes (high-entropy random
#      alphanumeric IDs, per-chain unique; assistant evidence ACK does
#      NOT echo the code; ~10 nats per-callback marginal vs ~3 nats on
#      D4v2). Built by tools/build_synthetic_random_codes.py.
#   2. --mask_evidence_session_loss: zero LM loss on EVIDENCE sessions
#      so the only supervised pressure on the answer span is the
#      callback session's, where the answer is not in local context
#      and memory is the only pathway.
#   3. --save_best_metric evidence_lift: save by what we actually care
#      about. The phase_aligned default is gameable by content-blind
#      writers (see v14k/v15a/v15e ledger).
#
# Frozen backbone is preserved on purpose: under D4v2 frozen actually
# delivered the best evidence_lift of all v14g..v14l cells (+0.13 at
# step 2600). On D5 the frozen backbone is the cleanest 'memory does
# the work' configuration -- nothing in the LM head can be retraining
# itself away from the marginal.
#
# GH200 repo layout is FLAT (train_chain.py at root, not src/). Local
# H100 box uses src/train_chain.py.
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
    --run_name chain_v16a_codes_0p6b_frozen_gh200 \
    --out_dir output/chain_v16a_codes_0p6b_frozen_gh200 \
    2>&1 | tee logs/chain_v16a_codes_0p6b_frozen_gh200.log
