#!/usr/bin/env bash
# Wait for both GPU train queues to finish, then run LoRA sweep +
# synthd5 ladder + figure rebuild + patent docx assembly.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/patent_experiments

echo "[finalize] waiting for GPU0_TRAIN_QUEUE_DONE / GPU1_TRAIN_QUEUE_DONE"
while ! grep -q GPU0_TRAIN_QUEUE_DONE logs/patent_train_gpu0.log 2>/dev/null; do
  sleep 60
done
echo "[finalize] GPU0 queue done"
while ! grep -q GPU1_TRAIN_QUEUE_DONE logs/patent_train_gpu1.log 2>/dev/null; do
  sleep 60
done
echo "[finalize] GPU1 queue done"

# ensure src→runs symlink resolves
[ -L output ] || ln -sfn runs output

echo "[finalize] LoRA sweep (GPU0) + synthd5 stage ladder (GPU1)"
CUDA_VISIBLE_DEVICES=0 bash -c '
  set -e
  python tools/patent_lora_forgetting.py \
    --n_chains 8 --steps_per_session 8 --mode joint --lr 1e-4 \
    --general_source fineweb \
    --output results/patent_experiments/lora_joint_lr1e-4_0p6b.json
  python tools/patent_lora_forgetting.py \
    --n_chains 8 --steps_per_session 8 --mode joint --lr 2e-5 \
    --general_source fineweb \
    --output results/patent_experiments/lora_joint_lr2e-5_0p6b.json
  python tools/patent_lora_forgetting.py \
    --n_chains 8 --steps_per_session 8 --mode sequential --lr 2e-5 \
    --general_source fineweb \
    --output results/patent_experiments/lora_seq_lr2e-5_0p6b.json
' &
PID_LORA=$!
CUDA_VISIBLE_DEVICES=1 python tools/patent_stage_ablation.py \
  --model_path runs/chain_v32a_sparse_synthd5_seed1_0p6b_frozen_local/final \
  --corpus paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
  --n_chains_max 100 \
  --output results/patent_experiments/stage_ablation_synthd5_v32a.json &
PID_SYNTH=$!
wait $PID_LORA
wait $PID_SYNTH

# refresh existing-ablation aggregate (idempotent)
python - <<'PY'
import json, statistics
from pathlib import Path
base = Path("results/eval_v25_seed_pack_evpos")
def dnm(f):
    j = json.load(open(base/f)); k = "lme_val" if "lme_val" in j else list(j)[0]
    return j[k]["pa_cb_dnm"]
cells = {
  "full_0p6b": ["v27b_no_probe_final_lme_val_evpos.json","v27b_no_probe_seed2_final_lme_val_evpos.json","v27b_no_probe_seed3_final_lme_val_evpos.json","v27b_no_probe_seed4_final_lme_val_evpos.json"],
  "no_depth_R0_0p6b": ["v27a_no_depth_final_redo_lme_val_evpos.json","v27a_v24a_no_depth_seed2_0p6b_frozen_local_lme_val_evpos.json","v27a_v24a_no_depth_seed3_0p6b_frozen_local_lme_val_evpos.json","v27a_v24a_no_depth_seed4_0p6b_frozen_local_lme_val_evpos.json"],
  "no_floor_0p6b": ["v27c_no_floor_final_lme_val_evpos.json","v27c_v24a_no_floor_seed2_0p6b_frozen_local_lme_val_evpos.json","v27c_v24a_no_floor_seed3_0p6b_frozen_local_lme_val_evpos.json","v27c_v24a_no_floor_seed4_0p6b_frozen_local_lme_val_evpos.json"],
  "full_1p7b": ["v28a_no_probe_seed1_final_lme_val_evpos.json","v28b_no_probe_seed2_final_lme_val_evpos.json","v28c_no_probe_seed3_1p7b_frozen_gh200_lme_val_evpos.json"],
}
out = {}
for name, files in cells.items():
    vals = [dnm(f) for f in files]
    out[name] = {"n": len(vals), "dnm_mean": statistics.mean(vals),
                 "dnm_sd": statistics.stdev(vals), "seeds": vals}
Path("results/patent_experiments/trained_ablations_existing.json").write_text(json.dumps(out, indent=2))
print(out)
PY

# ensure converted source docx exists
if [ ! -f "/tmp/patent_doc/一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，初稿）1 (1).docx" ]; then
  mkdir -p /tmp/patent_doc
  soffice --headless --convert-to docx --outdir /tmp/patent_doc \
    "一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，初稿）1 (1).doc"
fi

echo "[finalize] rebuild figures + patent docx"
python tools/patent_make_figures.py
python tools/patent_write_docx.py \
  --src "/tmp/patent_doc/一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，初稿）1 (1).docx" \
  --out "一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，含实验验证）.docx"

echo PATENT_FINALIZE_DONE
ls -la "一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，含实验验证）.docx"
ls results/patent_experiments/figures/
