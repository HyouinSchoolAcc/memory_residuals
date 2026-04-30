#!/usr/bin/env bash
# v11 4B HEADLINE -- attention_parity +4/0 + P0+P2 + L_E=10 (GH200, 3 days)
#
# This is the v11 headline cell: the user-directed initializers for the
# 4B run, composed with the load-bearing v11 fixes from the 0.6B audit.
# It is queued AFTER the 5 v11{g,h,i,j,k} 0.6B ablations so the
# initializer choice can be cross-checked against fresh 0.6B evidence
# before the 3-day budget burn -- but it does NOT block on those results;
# it embodies the user's prior that v3-style routing at the softer
# (+4/0) bias is the right architecture and the v11 fixes are the right
# data/init recipe.
#
# Composition (one line per axis; "WHY" in parens):
#   architecture
#     --memres_mode attention_parity                    (user's "v3 is better")
#     --router_recent_bias_init 4                       (user directive)
#     --router_mem_bias_init    0                       (user directive: alpha_mem
#                                                        ~ exp(0)/(exp(0)+depth+exp(4))
#                                                        ~ 0.009 at init,
#                                                        ~50x v3's +4/-4)
#     --memres_extract_source hidden_18                 (deeper than 0.6B's
#                                                        hidden_14: Qwen3-4B has
#                                                        36 layers, hidden_18 is
#                                                        the equivalent halfway
#                                                        depth where mid-context
#                                                        signal lives)
#     --memres_update_mode gated                        (mathematically the
#                                                        cleanest update; v9c's
#                                                        survivor)
#   bootstrap (P1+P2)
#     --memres_gate_init 0.0                            (gate not used by AP
#                                                        forward path; logged for
#                                                        completeness)
#     --memres_readout_norm_init 0.05                   (P2: ||m^t||/||embed||
#                                                        ~ 4 at init, not ~70;
#                                                        keeps the readout from
#                                                        getting suppressed by
#                                                        backbone before alpha_mem
#                                                        can grow)
#   data (P0)
#     --train_chains v11_mega_train_s512.pt             (67k chains; LME 450 now
#                                                        evidence-aware; non-LME
#                                                        unchanged from v10's
#                                                        mega corpus)
#     --eval_chains  lme_val_s512_v11.pt                (50 LME chains, evidence
#                                                        labels populated, no
#                                                        train overlap)
#     --curriculum_competition_bias 1.0                 (FULL: every batch is
#                                                        the four-session judge
#                                                        problem; competition is
#                                                        the supervision signal
#                                                        for keep-vs-write)
#   recurrence (P5)
#     --window_k 4
#     --carry_state                                     (M_c persists across
#                                                        windows -> training and
#                                                        eval see the same
#                                                        distribution of recurrent
#                                                        depths)
#     --burn_in_max 12
#     --burn_in_resample                                (uniform burn-in length
#                                                        per chain so the model
#                                                        has gradient signal at
#                                                        depths 0..12)
#   loss
#     --callback_loss_weight 3.0                        (v9a-confirmed: the
#                                                        callback span is what
#                                                        the model needs the
#                                                        memory FOR; up-weight it
#                                                        without zeroing the
#                                                        rest)
#     --neg_chain_weight 0.0                            (negative chains added
#                                                        zero in v6/v9; off here)
#     --memory_dropout 0.10
#     --context_dropout 0.05
#   compute (3-day budget on GH200; ~52 GB peak HBM, ~3-4 s/step at bs=2 ga=4)
#     --steps 25000
#     --warmup 500
#     --eval_every 500
#     --save_every 1000
#     --gradient_checkpointing
#     --batch_size 2 --grad_accum 4
#     --lr 3e-5 --lr_backbone 5e-6                      (4B-conservative; v10's)
#
# Decision triggers (sharp; use these to make a kill/keep call):
#   step  500 : ||m^t||/||embed|| in [0.3, 50]
#               alpha_mem_max   > 1e-3
#               pa_cb_evidence_lift > 0  (memory beats no-memory on
#                                          evidence-labelled callback)
#   step 2000 : alpha_mem_max   > 1e-2
#               pa_cb_dnm       > +0.020
#   step 5000 : alpha_mem_mean  > 1e-3   (recruitment, not just the
#                                          single hot sublayer)
#               standard Δ_sh-m > +0.005 (deployment metric moves)
#   step 12000: standard Δ_sh-m > +0.020 (publishable; this is the bar
#                                          v3-v10 never cleared)
#
# KILL conditions:
#   - At step 2000: alpha_mem_max < 5e-4         (collapsed, abort budget)
#   - At step 5000: pa_cb_evidence_lift <= 0     (memory is not content-
#                                                  specific even with evidence
#                                                  curriculum; abort, escalate
#                                                  to bigger router-bias
#                                                  intervention)
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs

CORPUS=${CORPUS:-paper_artifacts/chains/v11_mega_train_s512.pt}
if [ ! -f "$CORPUS" ]; then
    echo "[v11-4b] corpus $CORPUS not found; build it via scripts/build_v11_corpora_remote.sh"
    exit 2
fi

exec python -u train_chain.py \
    --preset qwen3-4b-xlarge \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_18 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --train_chains "$CORPUS" \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 3.0, "realtalk": 2.0, "ultrachat": 2.0, "pippa": 2.5, "soda": 1.5, "synthdlg": 1.5, "lmsys": 1.5}' \
    --window_k 4 \
    --carry_state \
    --burn_in_max 12 \
    --burn_in_resample \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 3e-5 \
    --lr_backbone 5e-6 \
    --steps 25000 \
    --warmup 500 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
    --curriculum_competition_bias 1.0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 500 \
    --save_every 1000 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v11_4b_mega_gh200 \
    --out_dir output/chain_v11_4b_mega_gh200
