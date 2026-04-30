#!/usr/bin/env bash
# pull_overnight.sh
# -----------------------------------------------------------------------------
# Run this when you wake up.  Rsyncs the overnight eval JSONs (and the run
# summary) from the GH200 back to the local repo.  Stages them but does NOT
# auto-commit -- you eyeball the headline numbers, then commit manually.
# -----------------------------------------------------------------------------
set -euo pipefail

CLOUD_USER="${CLOUD_USER:-ubuntu}"
CLOUD_HOST="${CLOUD_HOST:-192.222.50.225}"
CLOUD_REPO="${CLOUD_REPO:-/home/ubuntu/memory_residuals}"
LOCAL_REPO="${LOCAL_REPO:-$HOME/Desktop/fine-tune/memory_residuals}"

echo "=== pulling overnight traces from $CLOUD_HOST ==="
rsync -avz --partial \
  "$CLOUD_USER@$CLOUD_HOST:$CLOUD_REPO/paper_artifacts/eval/" \
  "$LOCAL_REPO/paper_artifacts/eval/"

echo ""
echo "=== overnight summary ==="
cat "$LOCAL_REPO/paper_artifacts/eval/overnight_traces_summary.txt" 2>/dev/null \
  || echo "(no summary yet)"

echo ""
echo "=== headline numbers per (ckpt, corpus) ==="
for j in "$LOCAL_REPO"/paper_artifacts/eval/routing_*.json; do
  [ -f "$j" ] || continue
  name="$(basename "$j" .json)"
  python3 - "$j" "$name" <<'PY'
import json, sys
p, name = sys.argv[1], sys.argv[2]
d = json.load(open(p))
mem  = sum(d["alpha_mem_by_sublayer"]["mem"]) / max(1, len(d["alpha_mem_by_sublayer"]["mem"]))
shuf = sum(d["alpha_mem_by_sublayer"]["shuffle"]) / max(1, len(d["alpha_mem_by_sublayer"]["shuffle"]))
print(f"{name}: alpha_mem(mem)={mem:.4f}  alpha_mem(shuffle)={shuf:.4f}  "
      f"n_pos={d.get('n_score_positions', 0)}")
PY
done

for j in "$LOCAL_REPO"/paper_artifacts/eval/cf_*.json; do
  [ -f "$j" ] || continue
  name="$(basename "$j" .json)"
  python3 - "$j" "$name" <<'PY'
import json, sys
p, name = sys.argv[1], sys.argv[2]
d = json.load(open(p))
parts = []
for k in sorted(d["results"], key=lambda s: (s != "depth_ALL", s)):
    r = d["results"][k]
    parts.append(f"{k}: dN={r['delta_mean']:+.4f}+/-{r['delta_std']:.4f} (n={r['n']})")
print(f"{name}:\n  " + "\n  ".join(parts))
PY
done

echo ""
echo "Done.  To commit:"
echo "  cd $LOCAL_REPO && git add paper_artifacts/eval && git commit -m 'overnight traces' && git push"
