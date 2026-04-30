#!/usr/bin/env bash
# After both `chain_v3_softparity_full` and `chain_v3_attentionbase_full`
# finish, run the rigorous standalone eval + horizon analysis + figure
# rebuild + paper recompile.  Idempotent; checks for *_eval.json before
# re-running anything.

set -e
cd "$(dirname "$0")/.."

EVAL_DIR=paper_artifacts/eval

echo "[post-train] $(date) -- starting pipeline"

run_eval() {
  local model_dir="$1"
  local out_basename="$2"
  local cuda_id="$3"
  local out="${EVAL_DIR}/${out_basename}_eval.json"
  if [[ -f "$out" ]]; then
    echo "[post-train] eval already exists: $out (skipping)"
    return
  fi
  echo "[post-train] running eval -> $out"
  CUDA_VISIBLE_DEVICES="$cuda_id" python paper_tools/eval_chain.py \
    --model_path "$model_dir" \
    --corpora paper_artifacts/chains/stage1_validation_s512.pt \
              paper_artifacts/chains/stage1_test_s512.pt \
              paper_artifacts/chains/locomo_s512.pt \
    --names pg19_val pg19_test locomo \
    --score_window 4 --oracle_window 4 \
    --output "$out" 2>&1 | tail -5
}

# 1. Standalone eval on the new step-6000 best checkpoints.
run_eval output/chain_v3_softparity_full/best chain_v3_softparity_full 0
run_eval output/chain_v3_attentionbase_full/best chain_v3_attentionbase_full 1

# 1b. Callback probe on all four best checkpoints, on PG-19 test pairs.
run_callback() {
  local model_dir="$1"
  local out_basename="$2"
  local cuda_id="$3"
  local out="${EVAL_DIR}/${out_basename}_callback.json"
  if [[ -f "$out" ]]; then
    echo "[post-train] callback already exists: $out (skipping)"
    return
  fi
  echo "[post-train] running callback probe -> $out"
  CUDA_VISIBLE_DEVICES="$cuda_id" python paper_tools/callback_probe.py \
    --model_path "$model_dir" \
    --data_path paper_artifacts/pairs_eval/pg19_test.jsonl \
    --num_samples 64 --history_len 1024 --current_len 512 \
    --output "$out" 2>&1 | tail -5 || true
}
run_callback output/chain_v3_softparity_full/best chain_v3_softparity_full 0
run_callback output/chain_v3_attentionbase_full/best chain_v3_attentionbase_full 1
run_callback output/chain_v2_phaseA_softparity_b4/best chain_v2_phaseA_softparity_b4 0
run_callback output/chain_v2_abl_residual_mode/best chain_v2_abl_residual_mode 1

# 2. Horizon analysis.
echo "[post-train] horizon analysis"
python paper_tools/horizon_analysis.py \
  --inputs ${EVAL_DIR}/chain_v3_softparity_full_eval.json \
           ${EVAL_DIR}/chain_v3_attentionbase_full_eval.json \
  --out_dir ${EVAL_DIR}/

# 3. Rebuild the paper figures.
echo "[post-train] rebuild figures"
python paper_tools/build_figures.py \
  --softparity_log logs/chain_v3_softparity_full.log \
  --attentionbase_log logs/chain_v3_attentionbase_full.log \
  --simplegate_log logs/chain_v2_abl_residual_mode.log \
  --softparity_horizon ${EVAL_DIR}/chain_v3_softparity_full_eval_horizon.json \
  --attentionbase_horizon ${EVAL_DIR}/chain_v3_attentionbase_full_eval_horizon.json \
  --simplegate_horizon ${EVAL_DIR}/chain_v2_abl_residual_mode_step5200_eval_horizon.json \
  --softparity_ckpt output/chain_v3_softparity_full/best \
  --attentionbase_ckpt output/chain_v3_attentionbase_full/best \
  --simplegate_ckpt output/chain_v2_abl_residual_mode/best \
  --out_dir paper/figures

# 4. Update paper TeX with final numbers via build_tables.py, then compile.
echo "[post-train] update paper tables"
python paper_tools/build_tables.py

echo "[post-train] compile paper"
cd paper
pdflatex -interaction=nonstopmode memory_residuals_empirical.tex >/dev/null 2>&1 || true
bibtex memory_residuals_empirical >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode memory_residuals_empirical.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode memory_residuals_empirical.tex >/dev/null 2>&1 || true
echo "[post-train] paper -> paper/memory_residuals_empirical.pdf"
ls -la memory_residuals_empirical.pdf

cd ..

# 5. Hand the new ckpts off to the cloud watchdog so eval continues even
#    if the laptop is powered off overnight.  Best-effort; failure here
#    must not fail the local pipeline.
echo "[post-train] cloud handoff"
bash paper_tools/cloud_handoff.sh \
    chain_v3_softparity_full chain_v3_attentionbase_full || true

echo "[post-train] $(date) -- done"
