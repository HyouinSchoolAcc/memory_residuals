#!/usr/bin/env bash
# v13p-lme-mega-gh200 (GH200 GPU 0) -- HEADLINE scaled cell.  The
# full v13 stack + frozen backbone (v13b mirror) on the v11_mega
# corpus (67 745 chains, the biggest LME+MSC+PG19+TV+RealTalk mix)
# for 16 000 steps.  This is the paper's eval cell -- if v13a
# clears the step-1000 trigger on D4 and v13b clears evidence_lift
# > +1.0 at step 2000, this queues automatically.
#
# Do not launch until v13b-d4-frozen has CONFIRMED the regime works
# on clean ground truth.  This is expensive (~16 h on GH200) and
# launching it on a non-working writer is exactly the mistake v11p
# / v12d_chinchilla_mega made.
#
# Decision triggers (scaled from D4 triggers; LME is log(|V|)
# ≈ 9-11 nats for typical answer-token entropy):
#   step 1000:  pa_cb_evidence_lift > +0.2  AND  alpha_mem_max >= 5e-3
#   step 2500:  pa_cb_dsh > +0.010  AND  evidence_lift > +0.5
#   step 5000:  pa_cb_dsh > +0.030  AND  evidence_lift > +1.0
#   step 8000:  pa_cb_dsh > +0.050  AND  evidence_lift > +2.0
#   step 16000: standard Δ_sh-m > +0.020  (the headline paper number)
#
# Frozen backbone + simple_gate + writer_warmup + orthogonal +
# slot_positional + slot_attention: all six levers composed.  LR
# bumped to 1e-4 (2x v13b) since the memres subsystem is the only
# thing training and the mega corpus gives ~260 M tokens (2x
# Chinchilla-budget for 9.7M params).
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
    --writer_warmup_steps 1000 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 400 \
    --writer_warmup_keep_backbone_frozen \
    --freeze_backbone \
    --train_chains paper_artifacts/chains/v11_mega_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 16000 \
    --warmup 500 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 12 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 500 \
    --save_every 1000 \
    --eval_n_chains 48 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v13p_lme_mega_gh200 \
    --out_dir output/chain_v13p_lme_mega_gh200
