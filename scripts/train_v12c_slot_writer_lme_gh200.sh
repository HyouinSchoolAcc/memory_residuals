#!/usr/bin/env bash
# v12c-slot-writer-LME (GH200) -- scale the winning D4 architecture
# (slot_attention OR slot_attention_full, decided by v12a/v12b) onto
# the v11_lme_msc corpus with the v11g recipe.
#
# This is the architectural-on-real-data cell.  D4 proves the
# slot-attention writer mechanism works on a controlled corpus; v12c
# tests whether the architecture survives the noise + sparsity +
# distribution shift of the LongMemEval-S + MSC mixture that broke
# the original writer in v11.
#
# Configuration
# -------------
# 0.6B-large preset, attention_parity +4/0, gated update, hidden_14
# extraction, RMSNorm(0.05), evidence-aware competition curriculum
# bias 1.0 -- this is the v11g recipe identically.  Single architectural
# axis: --memres_writer_kind = (slot_attention | slot_attention_full),
# inherited from the v12a / v12b winner.
#
# Override the writer kind on the command line:
#   bash train_v12c_slot_writer_lme_gh200.sh slot_attention
#   bash train_v12c_slot_writer_lme_gh200.sh slot_attention_full
# Defaults to slot_attention (v12a winner is the more conservative pick).
#
# Decision triggers (matches v11g for direct comparability)
# ---------------------------------------------------------
#   step  500 : alpha_mem_max > 5e-3  AND  pa_cb_dnm > +0.005
#                                     AND  pa_cb_evidence_lift > +0.005
#   step 1000 : alpha_mem_max > 1e-2  AND  pa_cb_dnm > +0.020
#                                     AND  pa_cb_dsh > +0.005
#                                     AND  alpha_mem_max NOT decaying
#                                     (vs v11g step 1000 = 0.011 and
#                                      decaying)
#   step 2000 : standard delta_sh-m > +0.010 (vs v11g step 1300 best of
#                                              +0.0024)
#   KILL: step 1000 with pa_cb_dnm <= v11g@1000 (= +0.027):
#                                     architecture wash on real data;
#                                     v12 does not provide a robustness
#                                     margin over v11; document as
#                                     "slot attention works only on
#                                     synthetic" and pivot.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
WRITER_KIND="${1:-slot_attention}"
case "$WRITER_KIND" in
    slot_attention|slot_attention_full) ;;
    *) echo "writer_kind must be slot_attention or slot_attention_full" >&2; exit 2 ;;
esac
# GH200 layout is flat (train_chain.py at project root; no src/ dir).
# Local layout has src/. Detect.
if [ -f train_chain.py ]; then
    TRAIN=train_chain.py
elif [ -f src/train_chain.py ]; then
    TRAIN=src/train_chain.py
else
    echo "train_chain.py not found in . or src/" >&2; exit 2
fi
exec python -u "$TRAIN" \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind "$WRITER_KIND" \
    --memres_slot_attention_iters 3 \
    --train_chains paper_artifacts/chains/v11_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 200 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
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
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name "chain_v12c_${WRITER_KIND}_lme_gh200" \
    --out_dir "output/chain_v12c_${WRITER_KIND}_lme_gh200"
