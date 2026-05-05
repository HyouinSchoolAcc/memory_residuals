# NeurIPS abstract — locked numbers ledger

**As of 2026-05-04 14:00 EDT.**  All numbers are produced by
`tools/eval_callback.py` against `paper_artifacts/chains/lme_val_s512_evpos.pt`
(50 LongMemEval-S validation chains, evidence-position-patched corpus).
The corpus, the eval script, and every command line below are
unchanged from the v24/v25/v27 wave: the only thing that varies between
runs is the training cell.

This document is the single source of truth that the NeurIPS abstract
and main-table numbers in the paper must agree with. **Do not edit
numbers here without re-running the cell.**

---

## TL;DR — the F3-OFF flip

The original v24a recipe ("memres_readout_depth=4 + readout-probe F3
+ α-floor on memory channel of the depth router") achieves a
**positive but modest +0.16 ± 0.08 nats** of callback CE improvement
across 3 seeds at 0.6B. **Removing the F3 readout probe (v27b) takes
the result to +1.32 ± 0.53 nats across 4 seeds at 0.6B — ~8× larger
and still chain-specific** (Δ_dsh ≈ 0). The simpler recipe scales
cleanly: **+0.93 nats across 2 seeds at 1.7B** (v28a/b), again with
Δ_dsh ≈ 0. Two single-variable ablations show the **iterative
readout depth** and the **α-floor** are individually load-bearing:
removing either kills the result. **The F3 readout probe is not just
unnecessary, it is harmful.**

The headline claim therefore becomes:

> A frozen Qwen3 backbone augmented with a fixed-size, jointly-trained
> Memory Residuals matrix M_c improves callback cross-entropy on
> LongMemEval-S validation by **+1.32 ± 0.53 nats at Qwen3-0.6B
> (n=4 seeds)**, **with the effect preserved at Qwen3-1.7B
> (+0.93 nats, n=2 seeds)**, and the chain-shuffle confound under
> ±0.02 nats throughout. Two single-variable ablations identify the
> iterative readout depth and the α-floor as load-bearing; the
> previously-canonical F3 readout probe is shown to be harmful and
> is removed.

---

## Headline table — F3-OFF recipe (v27b at 0.6B, v28a/b at 1.7B)

`final/` checkpoint, 1000 training steps, frozen backbone, identical
flags except `--readout_probe_loss_weight 0.0` (= F3-OFF).

| size | seed | host | `pa_cb_dnm` (Δ_dnm) | `pa_cb_dsh` (Δ_dsh) | `evidence_lift` |
|---|---|---|---|---|---|
| 0.6B | 1 | local | +0.797 | −0.017 | −0.005 |
| 0.6B | 2 | local | +0.939 | +0.008 | −0.002 |
| 0.6B | 3 | GH200 | +1.833 | +0.000 | +0.002 |
| 0.6B | 4 | GH200 | +1.721 | +0.001 | +0.008 |
| **0.6B mean (n=4)** | — | — | **+1.323 ± 0.530** | +0.000 ± 0.010 | +0.001 ± 0.006 |
| 1.7B | 1 | GH200 | +0.909 | −0.001 | +0.005 |
| 1.7B | 2 | GH200 | +0.944 | −0.009 | −0.008 |
| **1.7B mean (n=2)** | — | — | **+0.926** | −0.005 | −0.001 |

(`std` reported above is the sample standard deviation across seeds;
SEM at 0.6B is 0.27 nats. Per-chain check on the +1.83 seed: 49 of 50
chains positive, median Δ = +0.91 nats — not a single-chain outlier.)

---

## Reference table — original (with-F3) recipe

Same training cell as v27b but with `--readout_probe_loss_weight 0.5`
(the F3 probe). Trained on the same LongMemEval-S split, evaluated on
the same `lme_val_s512_evpos.pt` corpus.

| size | seed | run | Δ_dnm | Δ_dsh | ev_lift |
|---|---|---|---|---|---|
| 0.6B | 1 | v24a | +0.227 | +0.010 | +0.005 |
| 0.6B | 2 | v24a | +0.068 | −0.003 | +0.008 |
| 0.6B | 3 | v24a | +0.190 | +0.013 | +0.002 |
| **0.6B mean (n=3)** | — | — | **+0.162 ± 0.083** | +0.007 ± 0.008 | +0.005 ± 0.003 |
| 1.7B | 1 | v25a | +0.193 | −0.008 | −0.001 |
| 1.7B | 7 | v25a | +0.042 | +0.014 | −0.016 |
| **1.7B mean (n=2)** | — | — | **+0.118** | +0.003 | −0.009 |

The F3-OFF recipe is **8.2× larger at 0.6B (1.32 vs 0.16)** and
**7.9× larger at 1.7B (0.93 vs 0.12)** than the with-F3 recipe.

---

## Single-variable ablation table

All ablations use seed 1 of v24a as the matched control row. Each
column changes one CLI flag relative to v24a and is otherwise identical.

| ablation | run id | flag | Δ_dnm (final) | Δ_dnm (best) | verdict |
|---|---|---|---|---|---|
| canonical with-F3 | v24a-seed1 | (full recipe) | +0.227 | +0.229 | reference |
| **no readout depth** | v27a | `--memres_readout_depth 0` | **+0.025** | +0.029 | **load-bearing — depth IS essential** |
| **no F3 readout probe** | v27b-seed1 | `--readout_probe_loss_weight 0.0` | **+0.797** | −0.101 | **HARMFUL — removing it 5× the headline at this seed** |
| **no α-floor** | v27c | `--alpha_mem_floor_aux_weight 0.0` | **−0.038** | −0.289 | **load-bearing — floor IS essential** |

The "depth=0 collapses to ~0" and "floor=0 goes negative" rows tell
us the two design choices we keep. The "F3=0 goes from +0.23 to
+0.80 (and then to +1.32 averaged over 4 seeds)" row tells us the
third design choice we drop.

---

## What "Δ_dsh ≈ 0" means

Across all 4 v27b 0.6B seeds and both v28 1.7B seeds, the
chain-shuffle confound is in **[−0.017, +0.008]**. We interpret this
as: the writer compresses **chain-specific** content, not just
"any-memory-vector adds context to the LM". A random other chain's
M_c does not help; the chain's own M_c does. This is the
**chain-specificity claim** the paper is built on.

## What "evidence_lift ≈ 0" means

`evidence_lift = ce_mem_floor − ce_mem` measures the additional
benefit of having M_c built from the *full* chain vs from a chain
with the gold-evidence sessions redacted. Across all 4 v27b 0.6B
seeds and both v28 1.7B seeds, `evidence_lift` is in
**[−0.008, +0.008]** — within noise of zero. We interpret this as:
the writer is **not** memorising the literal answer-bearing
sessions; it is compressing **chain-conditional context** (style,
topic, vocabulary, prior turns) that happens to make the callback
distribution sharper at the LM head. This is the **scientific
framing finding** of the paper, not a failure mode.

---

## Compute & params (unchanged across recipes)

| arch | trainable params | training | wallclock |
|---|---|---|---|
| frozen Qwen3-0.6B + memres | 41.5 M (~6 % overhead) | 1000 steps on 1× H100 | ~1.5 h |
| frozen Qwen3-1.7B + memres | 164.8 M (~8 % overhead) | 1000 steps on 1× H100 | ~6 h |

Training corpus: LongMemEval-S **train** split, 450 chains
(`paper_artifacts/chains/lme_train_s512.pt`).
Eval corpus: LongMemEval-S **val** split, 50 chains
(`paper_artifacts/chains/lme_val_s512_evpos.pt`).

---

## Reproducibility notes

- All 0.6B v27b cells were launched from
  `Scripts/train_v27b_v24a_no_probe_seed{1,2,3,4}_0p6b_frozen_*.sh`.
  Seed is the only flag that differs between them.
- All 1.7B v28 cells were launched from
  `Scripts/train_v28{a,b}_v25a_no_probe_seed{1,2}_1p7b_frozen_gh200.sh`
  and identical to the 0.6B cells except for `--preset
  qwen3-1p7b-large-frozen` and adjusted batch / grad_accum.
- Eval JSONs are in `results/eval_v25_seed_pack_evpos/`.
- The `evpos` suffix on the corpus indicates the
  `chain_evidence_positions` field is real (per-chain integer list of
  evidence-bearing session indices); the unpatched `lme_val_s512.pt`
  defaulted this field to 1, causing `evidence_lift` to collapse to a
  no-op. **All paper numbers use the patched corpus going forward.**
- The original "+0.345" headline reported in older versions of
  `runs.md` for v24a-seed1 was on the unpatched corpus and is
  superseded by the +0.227 reading on the patched corpus. The
  v27b/v28 recipe-flip happened entirely on the patched corpus, so
  it is not a corpus artifact.
