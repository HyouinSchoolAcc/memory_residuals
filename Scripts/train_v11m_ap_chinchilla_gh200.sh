#!/usr/bin/env bash
# v11m (GH200, 0.6B) -- attention_parity +4/0  *  CHINCHILLA-BUDGETED memres
#
# User directive 2026-04-30 ~22:30 UTC:
#   "Think of a way we can apply the chincilla budget to our system."
#
# Chinchilla math for the memory subsystem
# ----------------------------------------
# 0.6B preset memres params: ~9.7 M (writer + judge + readout + depth_router
# + extraction layers). The MemRes subsystem is *literally from-scratch*
# (the load report shows 18/18 memres tensors MISSING from the pretrained
# Qwen3-0.6B checkpoint), so it is a Chinchilla-eligible model in the
# strict sense. Chinchilla-optimal token budget = ~20× param count
# = ~194 M tokens.
#
# v11g recipe at 4 000 steps:  4 000 × bs 4 × ga 2 × window_k 3 × 512 tok
#                            = 49 M tokens (gradient-flowing through memres)
#                            = ~25 % of Chinchilla.
#
# v11m at 16 000 steps × window_k 4:
#   16 000 × 4 × 2 × 4 × 512 = 262 M tokens = 1.35× Chinchilla.
#
# What this cell tests
# --------------------
# Single hypothesis: the v11g grow-then-decay isn't a routing-collapse or
# overfitting-decay -- it's the predicted behaviour of a 9.7 M-parameter
# from-scratch subnetwork that has only seen ~25% of its Chinchilla token
# budget. The v9c trajectory (the only "alive" v9 cell that ran to step
# 4000) supports this: it took 1 000 steps just to bootstrap (PA CB Δ_nm-m
# stayed *negative* through step 800 then turned positive at step 1000)
# and peaked at step 2400. Memres needs *more total optimisation steps*,
# not necessarily more peak lr or more warmup.
#
# Knob diff vs v11g
#   --steps        4000  -> 16000   (4× more optimiser steps)
#   --warmup        200  ->   800   (held at 5% of training, same fraction)
#   --window_k        3  ->     4   (deeper TBPTT, also closes part of P5)
#   --carry_state          ON       (M_c persists across windows -- exposes
#                                    the readout to the eval-shaped M_c
#                                    distribution during training)
#
# Eval cadence widened to every 400 steps so the eval cost stays a fixed
# fraction of training (eval_every / steps = 1/40 in v11g, kept here).
#
# Memory budget. Window_k=4 with full backbone training was the v11j
# OOM-on-cold-launch culprit in the watchdog launch race. As a single
# tenant it fits cleanly (the v11j OOM was caused by 5 specs racing for
# the GPU, fixed by the gpu_is_busy precheck; v9c ran window_k=3 with
# the same bs/ga happily). Peak HBM ~30 GB on the GH200; 60 GB headroom.
#
# Wall-time estimate. v11g runs at ~6.3k tok/s on GH200 → window_k 4 at
# similar tok/s → 16 000 × 16384 tok/step ÷ 6300 tok/s ≈ 11.6 hours.
# Allow 13 h with eval overhead.
#
# Decision triggers
# -----------------
#   step 1000 : α_mem_max > 5e-3  AND  pa_cb_dnm not yet collapsed
#                                      (matches v11g step 600-800)
#   step 4000 : pa_cb_dnm > +0.05         (4× more steps should give
#                                          v9c-tier gain by here:
#                                          v9c@4000 was +0.148)
#   step 8000 : pa_cb_dsh > +0.010
#                                          (content-specific, not just
#                                           an unconditional pull)
#   step 16000: standard Δ_sh-m > +0.005
#   KILL: step 4000 with pa_cb_dnm < 0 AND α_mem_max < 5e-3.
#         Chinchilla budget didn't fix the collapse → falsifies the
#         token-starvation hypothesis; escalate to v11p (frozen backbone
#         + Chinchilla + mega corpus) which is structurally cleanest.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --train_chains paper_artifacts/chains/v11_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 16000 \
    --warmup 800 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 12 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 400 \
    --save_every 1000 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v11m_ap_chinchilla_gh200 \
    --out_dir output/chain_v11m_ap_chinchilla_gh200
