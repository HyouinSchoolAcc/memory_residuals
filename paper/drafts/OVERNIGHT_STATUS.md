# Overnight status — May 4 — POST-MORTEM (14:00 EDT, after all GH200 + local runs landed)

## TL;DR — what we now know

**F3-OFF reproduces and scales. The new project headline is locked.**

| recipe | size | n seeds | mean Δ_dnm | range | mean Δ_dsh |
|---|---|---|---|---|---|
| **v27b (NO F3 probe)** | **0.6B** | **4** | **+1.323 ± 0.530** | [+0.797, +1.833] | +0.000 ± 0.010 |
| **v28a/b (NO F3 probe)** | **1.7B** | **2** | **+0.926** | [+0.909, +0.944] | −0.005 |
| v24a (with F3 probe, ref) | 0.6B | 3 | +0.162 ± 0.083 | [+0.068, +0.227] | +0.007 ± 0.008 |
| v25a (with F3 probe, ref) | 1.7B | 2 | +0.118 | [+0.042, +0.193] | +0.003 |

**F3-off is ~8× larger than F3-on at 0.6B and ~8× larger at 1.7B.**
All 4 0.6B seeds are positive, both 1.7B seeds are positive,
shuffle-confound stays in noise throughout. No single-chain outlier
(per-chain check: 49/50 positive on the +1.83 seed). The decision
gates from the 03:55 plan (#1) clear: this is the new headline.

The NeurIPS abstract goes with **Branch A**: F3-off as the canonical
recipe. Locked numbers ledger: [`NEURIPS_NUMBERS.md`](NEURIPS_NUMBERS.md).
Updated ledger entry: [`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md)
(top section).

---

## Original overnight plan — what actually happened

Original plan (written 03:55 EDT, kept below for the diff):

> **F3 (readout-probe) HURTS, not helps.** Two seeds now agree:
> - v27b-seed1 (F3 off) `final`: Δ_dnm = **+0.797**
> - v27b-seed2 (F3 off) `best`:  Δ_dnm = **+0.930**  (final still in flight)
>
> Compare full recipe: Δ_dnm = **+0.16 ± 0.08** (n=3 seeds).
> F3-off looks ~5× better. Δ_dsh stays at 0 → not a shuffle confound.
>
> If seeds 3+4 (in GH200 queue) corroborate (3/4 positive), this is the new headline.
> Also queued: 1.7B F3-off (v28a/b) — answers "does F3-off scale?" by lunchtime/evening.

**What actually happened on the GH200 (from `~/memory_residuals/logs/gh200_overnight.log`):**

| # | cell | size | start (UTC) | end (UTC) | rc | local pull / eval |
|---|---|---|---|---|---|---|
| 1 | v27b-seed3 | 0.6B | 08:48 | 09:44 | 0 | pulled 13:43 EDT, eval'd 13:50 → +1.833 |
| 2 | v27b-seed4 | 0.6B | 09:44 | 10:37 | 0 | pulled 13:44 EDT, eval'd 13:53 → +1.721 |
| 3 | v28a (1.7B seed1) | 1.7B | 10:38 | 13:37 | 0 | pulled 13:48 EDT, eval'd 13:49 → +0.909 |
| 4 | v28b (1.7B seed2) | 1.7B | 13:37 | 16:45 | 0 | pulled 13:48 EDT, eval'd 13:53 → +0.944 |

The GH200 queue ran ~2 h ahead of the original ETA estimate — v28b
finished 12:45 EDT instead of the predicted 18:45 EDT. The local
`pull_gh200_overnight.sh` and `watcher_eval_overnight.sh` daemons
unfortunately died sometime after 03:56 EDT (probable SSH /
desktop-session disconnect), so the trained checkpoints sat on the
GH200 with no local copy until the morning manual pull. **For the
next overnight, run the daemons under `tmux` or `systemd-run --user`
so they survive the desktop session.**

**What actually happened on local:**

The local extra-seeds queue (`scripts/local_overnight_extra_seeds.sh`,
`scripts/local_overnight_followup.sh`) launched **v24a-seed5** but
not v24a-seed4 (queue stalled — v24a-seed4 directory empty). v24a-seed5
produced a `best/` checkpoint with Δ_dnm = **−0.162** (writer
collapse, expected from v23/synthd5 history but rare on LME);
`final/` not produced (training was killed before final save). v25a
1.7B seed3/4 follow-ups never launched (the followup script's
"after seed7 finishes" gate didn't release in time).

These local stalls do not affect the headline — the F3-off result is
already n=4 at 0.6B and n=2 at 1.7B, and the v24a 3-seed reference
mean is unchanged.

---

## Numbers landed since the original plan

### v27b (F3-off, 0.6B) — full 4-seed pack

| seed | host | best | final |
|---|---|---|---|
| 1 | local | −0.101 | +0.797 |
| 2 | local | +0.930 | +0.939 |
| 3 | GH200 | +1.824 | +1.833 |
| 4 | GH200 | +0.096 | +1.721 |

`final` mean over 4 seeds: **+1.323 ± 0.530 nats** (sample std).
The seed-1 and seed-4 `best` checkpoints are pre-final saves taken
on synthd5_val (the in-train eval surface) so are not directly
comparable to the LME-val numbers; we report `final/` for the paper.

### v28a/b (F3-off, 1.7B) — 2-seed pack

| seed | best | final |
|---|---|---|
| 1 (v28a) | +0.921 | +0.909 |
| 2 (v28b) | +0.620 | +0.944 |

`final` mean: **+0.926 nats** (n=2). Both seeds clearly positive,
direction-of-effect preserved at scale.

### v25a-seed7 (1.7B WITH F3, 2nd seed for the with-F3 reference)

`best`: +0.119; `final`: +0.042. Confirms v25a/seed1 was on the
high end of the with-F3 distribution. Two-seed mean for the
with-F3 1.7B reference is +0.118 nats, vs +0.926 for F3-off.

---

## What's running RIGHT NOW

Nothing. Both local GPUs are idle. GH200 queue completed at 16:45 UTC
and the launcher exited (`ALL DONE` in `logs/gh200_overnight.log`).
No daemons should be re-started until the abstract submission is in;
all eval JSONs are saved to `results/eval_v25_seed_pack_evpos/`.

---

## Original overnight TL;DR (preserved for the diff)


## What's running

### GH200 (single H100 80GB) — sequential queue, ~15h
| # | Cell                          | size | recipe                | seed | ETA (EDT) |
|---|-------------------------------|------|-----------------------|------|-----------|
| 1 | `v27b_no_probe_seed3_gh200`   | 0.6B | v24a but F3-off       | 3    | ~05:15    |
| 2 | `v27b_no_probe_seed4_gh200`   | 0.6B | v24a but F3-off       | 4    | ~06:45    |
| 3 | `v28a_no_probe_seed1_1p7b`    | 1.7B | v25a but F3-off (LME) | 1    | ~12:45    |
| 4 | `v28b_no_probe_seed2_1p7b`    | 1.7B | v25a but F3-off (LME) | 2    | ~18:45    |

Daemon: `tmux attach -t overnight` (on GH200) shows the launcher.
Queue script: `scripts/gh200_overnight_queue.sh`.

### Local (2× H100 80GB)
- **GPU 0 NOW**: `v24a_seed5_local` (0.6B, full recipe) — finishes ~05:15
- **GPU 1 NOW**: `v27b_no_probe_seed2_local` (0.6B, F3-off) — finishes ~04:00
- **GPU 1 NEXT** (auto): `v24a_seed4_local` (0.6B, full recipe) — finishes ~05:30
- **GPU 0 NEXT** (auto): `v25a_seed3_1p7b_local` (1.7B, full recipe) — finishes ~11:15
- **GPU 1 NEXT** (auto): `v25a_seed4_1p7b_local` (1.7B, full recipe) — finishes ~11:30

Daemons (running on local):
- `scripts/local_overnight_extra_seeds.sh` (pid checked) — launches v24a_seed4/5 as GPUs free
- `scripts/local_overnight_followup.sh` — launches v25a_seed3/4 1.7B after the 0.6B seeds finish
- `scripts/pull_gh200_overnight.sh` — rsyncs each GH200 cell's `best/` + `final/` once it has `final/`
- `scripts/pull_gh200_v25a_seed7.sh` — pulls the already-finished v25a-seed7 (started earlier)
- `scripts/watcher_eval_overnight.sh` — auto-runs `eval_callback.py` on each new ckpt against `lme_val_s512_evpos.pt`

Eval outputs land in `results/eval_v25_seed_pack_evpos/<tag>_lme_val_evpos.json`.

## What you'll have by morning (≈09:00 EDT)

### 0.6B v24a (full recipe) — headline mean
- seeds 1, 2, 3 already done (Δ_dnm = +0.16 ± 0.08 nats from earlier)
- **+ seed 4 and seed 5** (ready ~05:30)
- → headline n=5 seeds for the 0.6B mean

### 0.6B v27b (F3-off) — verification of the +0.798 single-seed finding
- seed 1 done (final: +0.798 nats, best: +0.??? — re-check JSON)
- **+ seed 2** (local, finishes ~04:00) — first verification
- **+ seed 3, seed 4** (GH200, finishes ~06:45) — second + third verification
- → if 3+/4 seeds reproduce, F3-off is real; rewrite ablation row in PAPER_PLAN

### 1.7B v25a (full recipe) — scaling row
- seed 1 done (Δ_dnm = +0.19 nats, single seed)
- seed 7 just finished — needs eval (`v25a_seed7_*` watcher will pick up)
- **+ seed 3 (local) and seed 4 (local)** finish ~11:15 — gives n=4 1.7B seeds

### 1.7B v28 (F3-off) — does F3-off scale?
- seeds 1, 2 finish on GH200 between 12:45 and 18:45 — won't be ready by morning
- by lunchtime: 1.7B F3-off n=1 result; by evening: n=2

## What to look at first (morning checklist)

```bash
cd ~/Desktop/fine-tune/memory_residuals

# What finished?
ls -la results/eval_v25_seed_pack_evpos/

# Compact summary of the F3-off seed pack
python - <<'PY'
import json, glob
for p in sorted(glob.glob("results/eval_v25_seed_pack_evpos/v27b_no_probe_*lme_val_evpos.json")
              + glob.glob("results/eval_v25_seed_pack_evpos/v24a_seed*lme_val_evpos.json")
              + glob.glob("results/eval_v25_seed_pack_evpos/v25a_seed*lme_val_evpos.json")
              + glob.glob("results/eval_v25_seed_pack_evpos/v28*lme_val_evpos.json")):
    d = json.load(open(p))
    name = p.rsplit("/", 1)[1].replace("_lme_val_evpos.json","")
    rows = d.get("corpora", {}).get("lme_val", {}).get("rows", [])
    if not rows: continue
    pa_dnm = sum(r["pa_cb_dnm"] for r in rows) / len(rows)
    pa_dsh = sum(r["pa_cb_dsh"] for r in rows) / len(rows)
    el     = sum(r["evidence_lift"] for r in rows) / len(rows)
    print(f"{name:50s}  Δ_dnm={pa_dnm:+.3f}  Δ_dsh={pa_dsh:+.3f}  ev_lift={el:+.3f}")
PY

# Tail GH200 queue progress
ssh ubuntu@192.222.50.225 'tail -3 ~/memory_residuals/logs/gh200_overnight.log'

# Check still-running trainers
pgrep -af "python.*train_chain"
```

## Decision gates (after looking at numbers)

1. **If F3-off reproduces (≥3/4 seeds positive at 0.6B)** → rewrite ablation as headline:
   - "Memory-channel floor + readout depth, no chain-content probe" recipe.
   - Wait for v28a/b at lunchtime to confirm at 1.7B.
2. **If F3-off does NOT reproduce** → seed 1 was an outlier. Keep current recipe, kill v28b
   to free GH200 for an additional v25a seed or for the 4B exploration cell the user asked for.
3. **If 1.7B headline mean is now n=4 with tight CI** → publish that as the strongest 1.7B claim.
4. **For the abstract submission today**: numbers are already enough; tighten language using
   morning's seed4/5 means and (if time) the v27b reproduction status.

## Files written tonight

- `Scripts/train_v27b_v24a_no_probe_seed{3,4}_0p6b_frozen_gh200.sh` — GH200 0.6B F3-off
- `Scripts/train_v28{a,b}_v25a_no_probe_seed{1,2}_1p7b_frozen_gh200.sh` — GH200 1.7B F3-off
- `Scripts/train_v24a_v21c_lme_seed{4,5}_0p6b_frozen_local.sh` — local 0.6B headline
- `Scripts/train_v25a_v21c_lme_seed{3,4}_1p7b_frozen_local.sh` — local 1.7B headline
- `scripts/gh200_overnight_queue.sh` — GH200 sequential cell runner
- `scripts/gh200_overnight_launcher.sh` — waits for v25a-seed7, then runs queue
- `scripts/local_overnight_extra_seeds.sh` — local 0.6B extra seeds daemon
- `scripts/local_overnight_followup.sh` — local 1.7B follow-up daemon
- `scripts/pull_gh200_overnight.sh` — local rsync daemon for GH200 outputs
- `scripts/watcher_eval_overnight.sh` — local auto-eval daemon

## "Slap a 4B model" parking lot

User asked to defer to "the day after". When ready:
- Qwen3-4B preset doesn't exist yet — see `src/train_chain.py` near `qwen3-1.7b-large`
  to add a preset (probably ~360M memres params for d_model 2560).
- Single 1.7B run takes ~6h on GH200; 4B will be ~12–14h.
- Recommend: only after F3-off ablation question is settled; otherwise 4B may pick wrong recipe.
