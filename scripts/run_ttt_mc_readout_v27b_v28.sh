#!/usr/bin/env bash
# TTT-readout audit on the F3-OFF (v27b at 0.6B; v28 at 1.7B) headline
# checkpoints. Promised by the locked Paper C abstract: "a test-time
# readout audit that asks whether the writer encoded the answer at all".
#
# Decision rule (per tools/eval_ttt_mc_readout.py docstring):
#   ttt_lift_vs_floor  > +0.30 -> writer encoded the answer; readout
#                                 was the bottleneck. Read-side probe
#                                 (or deeper readout) would help.
#   in (0, +0.30]              -> partial; both readout-supervision and
#                                 deeper-readout architectural changes
#                                 would help.
#   <= 0                       -> writer did not encode the literal
#                                 answer. Consistent with the paper's
#                                 framing: M_c encodes chain-conditional
#                                 *context*, not literal evidence.
#
# Per-checkpoint cost: ~30s/chain at K=80 SGD steps on a frozen 0.6B;
# 1.7B is ~3x slower per chain. With n_chains=16 and 3 modes:
#   0.6B: 4 ckpts x 3 modes x 16 chains x ~30s ~= 95 min
#   1.7B: 2 ckpts x 3 modes x 16 chains x ~90s ~= 145 min
#   Total single-GPU: ~4 hours on H100. Split GPU0/GPU1 to halve.
#
# Run:
#   bash Scripts/run_ttt_mc_readout_v27b_v28.sh         # both scales, GPU0
#   GPU=1 SCALES=1p7b bash Scripts/run_ttt_mc_readout_v27b_v28.sh
#   SCALES=0p6b bash Scripts/run_ttt_mc_readout_v27b_v28.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# Auto-detect repo root (works locally on Windows shells via Git Bash and
# on the Linux server). Override with REPO=... if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

# Where eval_callback.py runs report checkpoints live: output/<run_name>/{best,final}.
# The launchers in Scripts/train_v27b_*.sh and train_v28*.sh use these run_names.
# `final` is the 1000-step end-of-training checkpoint (matches the paper headline).
declare -a CELLS_0P6B=(
  "v27b_seed1|chain_v27b_v24a_no_probe_seed1_0p6b_frozen_local"
  "v27b_seed2|chain_v27b_v24a_no_probe_seed2_0p6b_frozen_local"
  "v27b_seed3|chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200"
  "v27b_seed4|chain_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200"
)
declare -a CELLS_1P7B=(
  "v28a_seed1|chain_v28a_no_probe_seed1_1p7b_frozen_gh200"
  "v28b_seed2|chain_v28b_no_probe_seed2_1p7b_frozen_gh200"
  "v28c_seed3|chain_v28c_no_probe_seed3_1p7b_frozen_gh200"
)

CORPUS=paper_artifacts/chains/lme_val_s512_evpos.pt

OUT=results/ttt_mc_v27b_v28
LOG_DIR=logs/ttt_mc_v27b_v28
mkdir -p "$OUT" "$LOG_DIR"
LOG="$LOG_DIR/gpu${GPU:-0}_$(date +%Y%m%d_%H%M%S).log"

SCALES_TO_RUN="${SCALES:-0p6b,1p7b}"

run_cell () {
  local tag="$1" run_name="$2" preset="$3"
  local ckpt="output/${run_name}/final"
  if [[ ! -d "$ckpt" ]]; then
    # fall back to runs/<name>/best if output/<name>/final is absent
    if [[ -d "runs/${run_name}/best" ]]; then
      ckpt="runs/${run_name}/best"
    elif [[ -d "output/${run_name}/best" ]]; then
      ckpt="output/${run_name}/best"
    else
      echo "[skip] ${tag}: no checkpoint found at output/${run_name}/{final,best} or runs/${run_name}/best" | tee -a "$LOG"
      return 0
    fi
  fi
  for mode in v_only qkv qkv_reset; do
    local out_json="${OUT}/${tag}__${preset}__${mode}.json"
    if [[ -f "$out_json" ]]; then
      echo "[skip] ${out_json} already exists" | tee -a "$LOG"
      continue
    fi
    echo "============================================" | tee -a "$LOG"
    echo "RUN: ${tag} preset=${preset} mode=${mode} ckpt=${ckpt}" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"
    python tools/eval_ttt_mc_readout.py \
      --ckpt "$ckpt" \
      --eval_corpus "$CORPUS" \
      --n_chains 16 \
      --ttt_steps 80 \
      --ttt_lr_mc 1e-2 \
      --ttt_lr_readout 1e-3 \
      --readout_unfreeze "$mode" \
      --init_mode writer \
      --seed 0 \
      --out "$out_json" 2>&1 | tee -a "$LOG"
  done
}

if [[ ",${SCALES_TO_RUN}," == *",0p6b,"* ]]; then
  for cell in "${CELLS_0P6B[@]}"; do
    IFS='|' read -r tag run_name <<<"$cell"
    run_cell "$tag" "$run_name" "0p6b"
  done
fi

if [[ ",${SCALES_TO_RUN}," == *",1p7b,"* ]]; then
  for cell in "${CELLS_1P7B[@]}"; do
    IFS='|' read -r tag run_name <<<"$cell"
    run_cell "$tag" "$run_name" "1p7b"
  done
fi

echo "TTT-readout audit on v27b/v28 DONE  (log: ${LOG})" | tee -a "$LOG"