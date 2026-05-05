#!/usr/bin/env bash
# Run rigorous eval_chain.py on a checkpoint against:
#   D4v2 val (in-distribution)
#   LoCoMo (out-of-distribution generalisation)
#   MSC test (multi-session-chat, persona-aware OOD)
#
# Usage:
#   bash Scripts/eval_v14_v15_benchmarks.sh <ckpt_path> [<output_json>]
set -eu
ckpt="${1:-}"
out="${2:-results/eval_v14v15/$(basename "$ckpt")_$(date +%Y%m%d_%H%M%S).json}"

if [ -z "$ckpt" ] || [ ! -d "$ckpt" ]; then
    echo "usage: $0 <ckpt_dir> [<output.json>]" >&2
    echo "example: $0 output/chain_v14k_d4v2_norm_no_warmup_local/best" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$out")"

# Pick the GPU with the most free memory (we don't want to clash with
# whatever training is currently active).
free_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | awk '{print $1}' | tr -d ',')
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$free_gpu}"
echo "[eval] using ckpt=$ckpt out=$out gpu=$CUDA_VISIBLE_DEVICES"

python tools/eval_chain.py \
    --model_path "$ckpt" \
    --corpora \
        paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
        paper_artifacts/chains/locomo_s512.pt \
        paper_artifacts/chains/msc_test_s512.pt \
    --names d4v2_val locomo msc_test \
    --score_window 4 \
    --oracle_window 4 \
    --output "$out"

echo
echo "[eval] done.  Summary:"
python -c "
import json, sys
out = json.load(open('$out'))
for name, m in out.items():
    print(f'  {name}: dnm={m[\"delta_nomem_minus_mem\"]:+.4f}  dsh={m[\"delta_shuffle_minus_mem\"]:+.4f}  dor={m[\"delta_oracle_minus_mem\"]:+.4f}  capture={m[\"memory_capture_ratio\"]}')
"
