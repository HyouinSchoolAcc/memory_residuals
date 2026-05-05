#!/usr/bin/env bash
# Build the v11 evidence-aware corpora on a remote machine that already
# has the legacy v6_lme_msc_train_s512.pt and mega_train_s512.pt plus
# the freshly-rsynced lme_{train,val}_s512_v11.pt.
#
# Strategy: keep all non-LME chains from the legacy corpus (PG-19 / TV /
# MSC / and -- in the mega case -- ultrachat / pippa / soda / oasst1 /
# no_robots / narrativeqa / writingprompts / synthdlg), drop the legacy
# 450 LongMemEval chains (no evidence labels), and append the v11 LME
# 450 chains (with chain_evidence_positions populated). Result: same
# diverse training distribution, but the LME competition curriculum
# now picks evidence from real evidence sessions instead of uniformly
# at random.
#
# Outputs:
#   v11_lme_msc_train_s512.pt   (small mix; 0.6B GH200 ablations)
#   v11_mega_train_s512.pt      (mega mix; 4B GH200 headline run)
set -eu
cd "$(dirname "$0")/.."
python3 - <<'PYEOF'
import torch
from pathlib import Path

def merge_replace_lme(base_path: str, v11_lme_path: str, out_path: str) -> None:
    print(f"\n--- {out_path} ---")
    base = torch.load(base_path, map_location='cpu', weights_only=False)
    v11 = torch.load(v11_lme_path, map_location='cpu', weights_only=False)

    base_names = base['chain_names']
    keep_mask = [not n.startswith('longmemeval_') for n in base_names]
    n_lme_in_base = sum(1 for m in keep_mask if not m)
    print(f"  base: {len(base_names)} chains "
          f"({n_lme_in_base} legacy LME, {sum(keep_mask)} non-LME)")
    print(f"  v11_lme: {len(v11['chain_starts'])} chains "
          f"(with evidence labels)")

    base_starts = base['chain_starts'].tolist()
    base_lengths = base['chain_lengths'].tolist()
    new_sessions, new_cb_masks = [], []
    new_chain_starts, new_chain_lengths = [], []
    new_chain_names, new_chain_cb_pos, new_chain_ev_pos = [], [], []
    new_session_chain_id, new_session_position = [], []
    cursor, new_ci = 0, 0

    has_cb_in_base = 'session_callback_mask' in base
    has_cb_pos_in_base = 'chain_callback_position' in base

    for ci in range(len(base_starts)):
        if not keep_mask[ci]:
            continue
        L, s = base_lengths[ci], base_starts[ci]
        sess = base['session_ids'][s:s+L]
        if has_cb_in_base:
            cb_mask = base['session_callback_mask'][s:s+L]
        else:
            cb_mask = torch.zeros_like(sess, dtype=torch.int8)
        if has_cb_pos_in_base:
            cbp = int(base['chain_callback_position'][ci])
        else:
            cbp = -1
        new_sessions.append(sess)
        new_cb_masks.append(cb_mask)
        new_chain_starts.append(cursor)
        new_chain_lengths.append(L)
        new_chain_names.append(base_names[ci])
        new_chain_cb_pos.append(cbp)
        new_chain_ev_pos.append([])
        for sp in range(L):
            new_session_chain_id.append(new_ci)
            new_session_position.append(sp)
        cursor += L
        new_ci += 1

    v11_starts = v11['chain_starts'].tolist()
    v11_lengths = v11['chain_lengths'].tolist()
    v11_ev = v11['chain_evidence_positions']
    for ci in range(len(v11_starts)):
        L, s = v11_lengths[ci], v11_starts[ci]
        sess = v11['session_ids'][s:s+L]
        cb_mask = v11['session_callback_mask'][s:s+L]
        new_sessions.append(sess)
        new_cb_masks.append(cb_mask)
        new_chain_starts.append(cursor)
        new_chain_lengths.append(L)
        new_chain_names.append(v11['chain_names'][ci])
        new_chain_cb_pos.append(int(v11['chain_callback_position'][ci]))
        new_chain_ev_pos.append(list(v11_ev[ci]))
        for sp in range(L):
            new_session_chain_id.append(new_ci)
            new_session_position.append(sp)
        cursor += L
        new_ci += 1

    merged = {
        'session_ids': torch.cat(new_sessions, dim=0),
        'session_callback_mask': torch.cat(new_cb_masks, dim=0),
        'session_chain_id': torch.tensor(new_session_chain_id, dtype=torch.int32),
        'session_position': torch.tensor(new_session_position, dtype=torch.int32),
        'chain_starts': torch.tensor(new_chain_starts, dtype=torch.int64),
        'chain_lengths': torch.tensor(new_chain_lengths, dtype=torch.int64),
        'chain_callback_position': torch.tensor(new_chain_cb_pos, dtype=torch.int32),
        'chain_evidence_positions': new_chain_ev_pos,
        'chain_names': new_chain_names,
        'session_len': base['session_len'],
        'tokenizer': base['tokenizer'],
        'source': 'merged_v11',
    }
    torch.save(merged, out_path)
    n_with_ev = sum(1 for ep in new_chain_ev_pos if ep)
    n_total_ev = sum(len(ep) for ep in new_chain_ev_pos)
    print(f"  saved {out_path}: {merged['chain_starts'].shape[0]} chains, "
          f"{merged['session_ids'].shape[0]} sessions")
    print(f"    chains_with_evidence_labels: {n_with_ev}, "
          f"total_evidence_positions: {n_total_ev}")
    print(f"    callback_mask sum: {int(merged['session_callback_mask'].sum())}")

merge_replace_lme(
    'paper_artifacts/chains/v6_lme_msc_train_s512.pt',
    'paper_artifacts/chains/lme_train_s512_v11.pt',
    'paper_artifacts/chains/v11_lme_msc_train_s512.pt',
)

# Mega corpus may not exist on every host; only build if present.
mega = Path('paper_artifacts/chains/mega_train_s512.pt')
if mega.exists():
    merge_replace_lme(
        str(mega),
        'paper_artifacts/chains/lme_train_s512_v11.pt',
        'paper_artifacts/chains/v11_mega_train_s512.pt',
    )
else:
    print('\n  (skipping v11_mega: mega_train_s512.pt not present)')
PYEOF
echo "v11 corpora built."
