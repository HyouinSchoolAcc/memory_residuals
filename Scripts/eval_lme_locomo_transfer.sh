#!/usr/bin/env bash
# Standalone CE eval on LongMemEval val (in-domain) + LoCoMo + MSC test
# (OOD long-dialogue transfer). Same metric contract as eval_v14_v15_benchmarks.sh
# but swaps D4v2 for LME val to match v24+ training distribution.
#
# Usage:
#   bash Scripts/eval_lme_locomo_transfer.sh <ckpt_dir> [<output.json>]
#
# Example (v24a best 0.6B):
#   bash Scripts/eval_lme_locomo_transfer.sh output/chain_v24a_v21c_lme_seed1_0p6b_frozen_local/best
set -eu
ckpt="${1:-}"
out="${2:-results/eval_lme_locomo/$(basename "$ckpt")_$(date +%Y%m%d_%H%M%S).json}"

if [ -z "$ckpt" ] || [ ! -d "$ckpt" ]; then
    echo "usage: $0 <ckpt_dir> [<output.json>]" >&2
    echo "example: $0 output/chain_v24a_v21c_lme_seed1_0p6b_frozen_local/best" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$out")"

free_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | awk '{print $1}' | tr -d ',')
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$free_gpu}"
echo "[eval] ckpt=$ckpt out=$out gpu=$CUDA_VISIBLE_DEVICES"

python tools/eval_chain.py \
    --model_path "$ckpt" \
    --corpora \
        paper_artifacts/chains/lme_val_s512.pt \
        paper_artifacts/chains/locomo_s512.pt \
        paper_artifacts/chains/msc_test_s512.pt \
    --names lme_val locomo msc_test \
    --score_window 4 \
    --oracle_window 4 \
    --output "$out"

echo
echo "[eval] done. Summary (delta_nomem_minus_mem / delta_shuffle_minus_mem):"
python -c "
import json
out = json.load(open('$out'))
for name, m in out.items():
    print(f'  {name}: dnm={m[\"delta_nomem_minus_mem\"]:+.4f}  dsh={m[\"delta_shuffle_minus_mem\"]:+.4f}  capture={m[\"memory_capture_ratio\"]}')
"
