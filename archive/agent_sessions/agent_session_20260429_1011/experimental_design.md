# Experimental design — v5 soft-init 2x2 (active 2026-04-29 ~15:35 UTC)

## What's running

Three trainers running concurrently, all at soft `±4` parity init,
all sharing the running-headline knob set (`window_k=3`, dropouts,
`carry_state`, etc.):

| cell | extract source | corpus | machine | run_name | tmux session | log |
|------|----------------|--------|---------|----------|--------------|-----|
| **A** (HEADLINE) | `hidden_14` | PG-19 + TV + MSC | GH200 (cloud) | `chain_v5_softhidden14_msc` | `cwd-chain_v5_softhidden14_msc` | remote: `paper_tools/cloud_watchdog/logs/chain_v5_softhidden14_msc.log` |
| **B** | `embed` (legacy) | PG-19 + TV + MSC | local GPU 0 (H100 NVL) | `chain_v5_softembed_msc` | `local-chain_v5_softembed_msc` | local: `logs/chain_v5_softembed_msc.log` |
| **C** | `hidden_14` | PG-19 + TV (no MSC) | local GPU 1 (H100 NVL) | `chain_v5_softhidden14_pgtv` | `local-chain_v5_softhidden14_pgtv` | local: `logs/chain_v5_softhidden14_pgtv.log` |

**Cell D** is still TBD — `embed` + PG-19+TV with soft init. Will queue
behind whichever local run finishes first (B or C). For now the
existing v3sp run (`chain_v3_softparity_full`, also at soft `±4`,
`embed`, `window_k=8`, no dropouts, no contrastive ramp) serves as a
proximate but knob-mismatched cell-D baseline.

## What this answers

Each contrast isolates one knob, holding everything else constant:

| contrast | isolates | hypothesis |
|----------|----------|------------|
| **A vs B** (column at fixed corpus) | extract source: `hidden_14` vs `embed` | hidden_14 should beat embed on MSC `Δ_sh-m` because dialogue requires anaphora/entity binding that bag-of-tokens can't resolve |
| **A vs C** (row at fixed extract) | corpus: with-MSC vs without-MSC | adding MSC should improve dialogue eval (MSC val + LoCoMo); books eval should be roughly unchanged |
| **B vs C** (diagonal) | joint corpus + extract | combined effect; less informative than A-vs-B and A-vs-C separately |
| **B vs v4 (hard ±32)** | init alone | Δ_sh-m at v4 was bit-zero; if v5 cell B opens (>0.005), the init was the dominant blocker |

## What we need to see for each cell

Stop-or-go decision triggers from `recommended_next_run.md`:

1. By step ~1500: in-trainer `Δ_sh-m > +0.005`. If still flat zero,
   cell has the same failure as v4; kill that cell and reconsider.
2. By step ~4000: in-trainer `Δ_sh-m > +0.020`.
3. End-of-run post-train pipeline:
   - `α_mem^true > 1e-3` on at least one held-out corpus
   - mem-vs-shuffle gap > +5%
   - counterfactual `Δ(d=ALL) > 0.010`

## Why this is the right design

Recipe paper claims a four-piece recipe (hidden extract, MSC corpus,
contrastive ramp, parity-preserving init). The contrastive ramp + init
piece is now the soft `±4` for everything in v5. The remaining 2x2 (A
vs B vs C vs D) attributes any residual gain to the extract / corpus
pieces.

## Estimated wallclock

- Cell A (GH200): 6000 steps × ~24k tok/step / ~3k tok/s ≈ **14 h**, finishes ~2026-04-30 05:30 UTC.
- Cells B / C (H100 NVL): 6000 steps × ~24k tok / ~7.5k tok/s ≈ **5–6 h**, finish ~2026-04-29 21:00 UTC.

So B and C finish first by ~8 h, freeing local GPUs to launch cell D
(or a deeper window_k=8 ablation matching the README recipe).

## v4 hard-init "before" baselines (preserved as negative result data)

| cell | run | last step | reason stopped | log |
|------|-----|----------|----------------|-----|
| A | `chain_v4_hidden14_msc` | 5500 / 6000 | killed 2026-04-29 15:28 UTC after diagnosis | `logs/chain_v4_hidden14_msc_final.log` (local copy), remote `output/chain_v4_hidden14_msc/best/` (step ~500) |
| B | `chain_v4_embed_msc` | 1500 | local GPU needed by user before completion (per Apr 28 log) | `logs/chain_v4_embed_msc.log`, `output/chain_v4_embed_msc/best/` |
| C | `chain_v4_hidden14_pgtv` | 500 | local GPU needed by user before completion | `logs/chain_v4_hidden14_pgtv.log`, `output/chain_v4_hidden14_pgtv/best/` |

All three v4 cells ran with `mem_bias=-32, recent_bias=+32`. All show
`Δ_sh-m = +0.0000` to 4 decimals across every logged eval — the
softmax-saturation failure mode reproduces consistently.
