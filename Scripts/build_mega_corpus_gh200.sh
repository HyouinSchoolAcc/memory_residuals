#!/usr/bin/env bash
# v10 MEGA CORPUS build: ~300k-500k chains across 10+ sources for the
# 3-day GH200 8B run. Runs on the GH200 where ~3.8 TB / dev/vda1 is
# free and network throughput is high enough to stream HF datasets.
#
# Pipeline:
#   (1) tools/build_synthetic_dialogue_chains.py downloads each
#       HF source and produces per-chain JSONL files in
#       memory_residuals_data/mega_stage/<source>/<source>_<id>.jsonl
#   (2) tools/pretokenize_chains.py tokenises each source into a
#       per-source .pt (parallel over 16 workers for speed)
#   (3) tools/merge_chain_corpora.py concatenates every per-source
#       .pt together with the existing v6_lme_msc corpus (the
#       survivable base from v9c) into
#       paper_artifacts/chains/mega_train_s512.pt
#
# Idempotency: if mega_train_s512.pt already exists, this script is a
# no-op. Delete it to force a rebuild. The intermediate per-source .pt
# files are also cached and only re-tokenised if missing.
set -eu

cd "$(dirname "$0")/.."

OUT_PT=paper_artifacts/chains/mega_train_s512.pt
if [ -f "$OUT_PT" ]; then
    echo "[mega] $OUT_PT already exists - nothing to do"
    exit 0
fi

STAGE_DIR=../memory_residuals_data/mega_stage
CHAINS_DIR=paper_artifacts/chains
TOKENIZER=${TOKENIZER:-Qwen/Qwen3-0.6B}
# Per-source cap: 25000 is enough to fill a 3-day training budget
# while keeping the build under ~2 hours (most are streaming downloads).
MAX_PER_SOURCE=${MAX_PER_SOURCE:-25000}
SESSION_LEN=${SESSION_LEN:-512}

mkdir -p "$STAGE_DIR" "$CHAINS_DIR"

# --------------------------------------------------------------------
# (1) Download + convert each source into per-chain JSONL.
# --------------------------------------------------------------------
SOURCES=(ultrachat pippa soda oasst1 no_robots narrativeqa writingprompts)
# hh_rlhf and lmsys are optional (lmsys is gated; hh_rlhf sometimes has
# integrity issues). Add them via EXTRA_SOURCES="hh_rlhf lmsys" env var.
if [ -n "${EXTRA_SOURCES:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=($EXTRA_SOURCES)
    SOURCES=("${SOURCES[@]}" "${EXTRA[@]}")
fi

for SRC in "${SOURCES[@]}"; do
    src_dir="$STAGE_DIR/$SRC"
    if [ -d "$src_dir" ] && [ "$(ls -1 "$src_dir" 2>/dev/null | head -1)" ]; then
        echo "[mega] $SRC: already staged - skipping"
        continue
    fi
    echo "[mega] Staging $SRC (cap=$MAX_PER_SOURCE)..."
    # Tolerate per-source failures: if one HF repo is deprecated or
    # transiently unreachable, continue with the remaining sources.
    if ! python -u tools/build_synthetic_dialogue_chains.py \
        --sources "$SRC" \
        --out_dir "$STAGE_DIR" \
        --turns_per_session 4 \
        --min_turns_per_session 2 \
        --max_chains_per_source "$MAX_PER_SOURCE"; then
        echo "[mega] $SRC: build failed (non-fatal); continuing with other sources"
    fi
done

# --------------------------------------------------------------------
# (2) Pretokenise each source into its own .pt (parallel).
# --------------------------------------------------------------------
for SRC in "${SOURCES[@]}"; do
    src_dir="$STAGE_DIR/$SRC"
    src_pt="$CHAINS_DIR/${SRC}_train_s${SESSION_LEN}.pt"
    if [ -f "$src_pt" ]; then
        echo "[mega] $SRC: $src_pt exists - skipping tokenisation"
        continue
    fi
    if [ ! -d "$src_dir" ]; then
        echo "[mega] $SRC: no staged dir ($src_dir) - skipping"
        continue
    fi
    # Skip if the stage dir is empty (source failed to download any chains).
    if [ -z "$(ls -1 "$src_dir" 2>/dev/null | head -1)" ]; then
        echo "[mega] $SRC: staged dir is empty - skipping"
        continue
    fi
    echo "[mega] Pretokenising $SRC -> $src_pt..."
    if ! python -u tools/pretokenize_chains.py \
        --in_dir "$src_dir" \
        --out_path "$src_pt" \
        --tokenizer "$TOKENIZER" \
        --session_len "$SESSION_LEN" \
        --min_tokens 32 \
        --min_sessions_per_chain 2 \
        --workers 16; then
        echo "[mega] $SRC: pretokenize failed (non-fatal); continuing"
    fi
done

# --------------------------------------------------------------------
# (3) Merge: existing v6_lme_msc + every per-source .pt just built.
# --------------------------------------------------------------------
MERGE_ARGS=(--in paper_artifacts/chains/v6_lme_msc_train_s512.pt)
for SRC in "${SOURCES[@]}"; do
    src_pt="$CHAINS_DIR/${SRC}_train_s${SESSION_LEN}.pt"
    if [ -f "$src_pt" ]; then
        MERGE_ARGS+=(--in "$src_pt")
    fi
done

echo "[mega] Merging into $OUT_PT..."
python -u tools/merge_chain_corpora.py \
    "${MERGE_ARGS[@]}" \
    --out "$OUT_PT"

echo "[mega] DONE -> $OUT_PT"
python -u -c "
import torch
d = torch.load('$OUT_PT', map_location='cpu', weights_only=False)
print('chains :', len(d['chain_names']))
print('sessions :', d['session_ids'].shape[0])
print('cb chains :', int((d.get('chain_callback_position', torch.zeros(0)) >= 0).sum()))
from collections import Counter
c = Counter()
for n in d['chain_names']:
    if n.startswith('msc_'): c['msc'] += 1
    elif n.startswith('longmemeval_'): c['longmemeval'] += 1
    elif n.startswith('realtalk_'): c['realtalk'] += 1
    elif n.startswith('ultrachat_'): c['ultrachat'] += 1
    elif n.startswith('pippa_'): c['pippa'] += 1
    elif n.startswith('soda_'): c['soda'] += 1
    elif n.startswith('lmsys_'): c['lmsys'] += 1
    elif n.startswith('synthdlg_'): c['synthdlg'] += 1
    elif n[:1].isdigit(): c['pg19'] += 1
    else: c['tv'] += 1
for k, v in sorted(c.items()): print(f'  {k:<12} {v}')
"
