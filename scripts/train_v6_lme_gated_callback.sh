#!/usr/bin/env bash
# v6 long-horizon recipe — first run with all four pivots from
# the user's directive ("rebuild corpora, tweak memory to non-replacing,
# new training loss, restart"):
#
#   1) NEW CORPUS — paper_artifacts/chains/lme_train_s512.pt (LongMemEval-S,
#      450 chains × ~50 sessions × ~10 turns/session, callback supervision
#      on the Q+A answer span). Built by paper_tools/build_conversational_callback_chains.py.
#      Replaces the v3-v5 PG-19/TV/MSC mix that was the WRONG domain
#      (books) for this paper's claim ("conversational memory recall").
#      The previous PG-19+passkey synthetic experiment (longhorizon_*.pt)
#      is archived under paper_artifacts/chains/archive_pg19_passkey/.
#
#   2) NON-REPLACING MEMORY — --memres_update_mode gated. Adds a per-slot
#      sigmoid write gate so the model can KEEP an existing slot rather
#      than overwrite. Init bias = -1.0 → g ~ 0.27 (modest writes early).
#      Counters the "every session clobbers M_c" pathology of competitive
#      mode on long chains (50 sessions of mostly-irrelevant filler can
#      destroy a session-1 fact under v3-v5).
#
#   3) NEW LOSS — --callback_loss_weight 10.0. Tokens marked in the
#      corpus's session_callback_mask (the answer span of each LongMemEval
#      callback session) get loss weight 11x baseline. Concentrates
#      gradient where memory retrieval actually matters; everything else
#      remains standard NLL. --callback_window_bias 0.7 biases the
#      sampler so 70% of windows include the callback session
#      (otherwise we'd see it ~1/L of the time on 50-session chains).
#
#   4) RESTARTED TRAINING — fresh run, no warm start from v5 ckpts.
#      Soft ±4 router init (proven necessary in v5; v4 hard ±32 saturated
#      bf16). hidden_14 extraction (proven necessary in v3 vs v2's bag-of-
#      embed). carry_state ON (v5 default). NO dropouts and NO contrastive
#      (proven harmful in v5 cell B; cell B prime confirmed dropout-removal
#      doesn't break things). gradient_checkpointing for memory budget.
#
# Headline knob differences vs cell B prime (v5 baseline):
#   --train_chains  stage1_msc_train_s512.pt -> lme_train_s512.pt
#   --eval_chains   stage1_msc_val_s512.pt   -> lme_val_s512.pt
#   --window_k      3                        -> 8   (chains are 50+ sessions)
#   --memres_update_mode  competitive (default) -> gated  [NEW FLAG]
#   --callback_loss_weight 0 (default)       -> 10.0       [NEW FLAG]
#   --callback_window_bias 0 (default)       -> 0.7        [NEW FLAG]
#   --burn_in_max   8                        -> 24  (much longer chains)
#   --eval_window   4                        -> 8
#   --source_weights removed (LME is a single source)
#
# Decision triggers (when to pull the plug or proceed to v7):
#   step  500: gate_max should be > 0 (memory channel opening at all)
#   step 1000: Δ_sh-m on lme_val should be > +0.005 OR ce_callback should
#              decline meaningfully (the callback span is the only place
#              memory actually buys you anything; uniform NLL is a coarser
#              proxy)
#   step 2500: if Δ_sh-m on lme_val isn't > +0.02 AND alpha_mem on a
#              routing trace isn't > 5%, the gated/callback recipe is
#              insufficient and we need to ablate which piece is missing
set -eu
cd "${REPO:-/home/exx/Desktop/fine-tune/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --memres_update_mode gated \
    --router_mem_bias_init -4 \
    --router_recent_bias_init 4 \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --steps 8000 \
    --warmup 100 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --carry_state \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 10.0 \
    --callback_window_bias 0.7 \
    --burn_in_max 24 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v6_lme_gated_callback \
    --out_dir output/chain_v6_lme_gated_callback
