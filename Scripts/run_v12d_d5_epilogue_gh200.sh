#!/usr/bin/env bash
# v12d-d5-epilogue (GH200) -- D5 readout-TTT audit on v12d/best.
#
# Cheap epilogue (< 10 min on H100, per tools/d5_ttt_readout.py
# docstring) that runs AFTER v12d completes.  The watchdog dispatches
# specs in queued-timestamp order, so enqueueing this with a timestamp
# strictly greater than v12d ensures it only fires once v12d's
# checkpoint exists.
#
# What D5 measures
# ----------------
# Freezes every parameter except the readout's W_Q / W_K / W_V on the
# v12d/best checkpoint, then does ~4000 gradient steps on D4 train
# split with callback supervision.  Decision rule (from
# tools/d5_ttt_readout.py):
#
#   callback CE drops >= 30%   ==>  writer encoded info, readout was
#                                   the bottleneck.  Path R.
#   callback CE does not drop  ==>  writer did NOT encode info; the
#                                   slot-attention architecture also
#                                   fails to write content.  Path W.
#
# v12 thesis predicts Path R: the slot-attention writer architecturally
# specialises slots on disjoint content, so M_c should encode the
# persona facts the D4 callback queries.  If D5 on v12d/best shows
# Path W, the architectural change moved row_entropy off uniform but
# did not actually teach M_c to be content-aware -- in which case the
# v13 plan needs to add content-aware probing loss in addition to slot
# competition.
#
# Cheap = 4000 steps, ~10 minutes, single GPU.  Falsifiable:
#   pa_cb_ce_mem drops >= 30% from baseline -> Path R confirmed
#   pa_cb_ce_mem drops <  10% from baseline -> Path W; pivot v13
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs results/eval
# Caller selects which v12d arm to evaluate.  Defaults to the frozen
# arm (queued first); pass "trained" or "full_frozen" to switch.
ARM="${1:-frozen}"
case "$ARM" in
    frozen)       RUN_NAME=chain_v12d_slot_attention_d4_frozen_gh200 ;;
    trained)      RUN_NAME=chain_v12d_slot_attention_d4_trained_gh200 ;;
    full_frozen)  RUN_NAME=chain_v12d_slot_attention_full_d4_frozen_gh200 ;;
    full_trained) RUN_NAME=chain_v12d_slot_attention_full_d4_trained_gh200 ;;
    *) echo "ARM must be frozen|trained|full_frozen|full_trained" >&2; exit 2 ;;
esac
CKPT="output/${RUN_NAME}/best"
if [ ! -d "$CKPT" ]; then
    echo "ERROR: no checkpoint at $CKPT (did the parent run finish?)" >&2
    exit 2
fi
OUT="results/eval/${RUN_NAME}_d5_ttt_epilogue.json"
echo "[d5_epilogue] arm=$ARM ckpt=$CKPT out=$OUT"
# Path-detect d5 tool location (flat GH200 vs structured local).
if [ -f d5_ttt_readout.py ]; then
    D5=d5_ttt_readout.py
elif [ -f tools/d5_ttt_readout.py ]; then
    D5=tools/d5_ttt_readout.py
else
    echo "ERROR: d5_ttt_readout.py not found" >&2
    exit 2
fi
exec python -u "$D5" \
    --ckpt   "$CKPT" \
    --train_corpus paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_corpus  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
    --steps 4000 \
    --batch_size 4 \
    --lr 1e-3 \
    --out  "$OUT"
