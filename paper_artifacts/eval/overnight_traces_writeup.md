# Overnight routing-trace + counterfactual sweep

Run: 2026-04-29 03:07-03:11 UTC, GH200 (alongside `chain_v4_hidden14_msc` training).

## What ran

`paper_tools/run_overnight_traces.sh` swept three ckpts x two corpora x two
probes:

|              | stage1_validation (PG-19, 30-156 score positions) | locomo (synthetic dialogue, 40 score positions) |
| ------------ | ------------------------------------------------- | ----------------------------------------------- |
| v3 softparity (step 4400, attention_parity)        | routing + cf | routing + cf |
| v3 attentionbase (step 4400, attention_base)       | routing + cf | routing + cf |
| v2 phaseA softparity_b4 (early checkpoint)         | routing + cf | routing + cf |

12 JSONs total in `paper_artifacts/eval/{routing,cf}_v{2sp,3sp,3ab}_{val,locomo}.json`.

## Headline 1: alpha_mem is essentially zero on every checkpoint

`alpha_mem` = softmax mass that the depth-router puts on the memory source
b_{-1}, averaged over (sublayer, token, position).

|                   | val mem    | val shuffle | locomo mem | locomo shuffle |
| ----------------- | ---------- | ----------- | ---------- | -------------- |
| v3 softparity     | 0.00047    | 0.00045     | 0.00029    | 0.00030        |
| v3 attentionbase  | 0.00015    | 0.00015     | 0.00018    | 0.00018        |
| v2 softparity_b4  | 0.00037    | 0.00031     | 0.00029    | 0.00030        |

Two readings of this number:

- The architecture is sitting right at its parity init.  `mem_bias` was
  initialised to a strong negative (-8) so that alpha_mem ~ exp(-8)/N at
  step 0; `exp(-8)/56 ~ 6e-5` if the pseudo-query were exactly zero.  Our
  measured 1e-4 to 5e-4 means the router HAS moved off zero, by 2-8x of
  the init mass, but it has not opened up the memory channel by anything
  like an order of magnitude.

- The mem-vs-shuffle gap is < 10% of the absolute value at every ckpt.
  When we feed the wrong chain's M_c, the router does not change its
  routing decision in any meaningful way.

This is consistent with the v3 NLL plateau we documented in
`paper_artifacts/eval/chain_v3_training_summary.md`: the model is paying
for the routing parameters but is not using them to read the memory
matrix.

## Headline 2: counterfactual perturbations are within noise

`delta = NLL(perturbed) - NLL(normal)`.  Positive = the model used that
slot.

| depth | v3sp val           | v3sp locomo        | v3ab val           | v3ab locomo        | v2sp val           |
| ----- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
|     1 | +0.0094 +/- 0.0137 | +0.0016 +/- 0.0088 | +0.0031 +/- 0.0057 | -0.0007 +/- 0.0033 | +0.0085 +/- 0.0114 |
|     2 | +0.0008 +/- 0.0050 | +0.0006 +/- 0.0024 | +0.0002 +/- 0.0026 | +0.0016 +/- 0.0016 | +0.0006 +/- 0.0052 |
|     4 | +0.0003 +/- 0.0025 | +0.0013 +/- 0.0019 | +0.0001 +/- 0.0022 | +0.0004 +/- 0.0018 | -0.0003 +/- 0.0022 |
|     8 | +0.0000 +/- 0.0017 | +0.0011 +/- 0.0018 | -0.0001 +/- 0.0020 | +0.0009 +/- 0.0021 | +0.0000 +/- 0.0015 |
|   ALL | +0.0127 +/- 0.0100 | -0.0069 +/- 0.0103 | +0.0050 +/- 0.0061 | -0.0049 +/- 0.0075 | +0.0086 +/- 0.0083 |

All depth-1..8 deltas are within one standard deviation of zero.  Even at
depth_ALL (memory wiped entirely) the effect is in the third decimal of
NLL on val and indistinguishable from zero on locomo.  Compare against
`Delta_sh-m` we used to celebrate at training time (~0.05): that delta is
measured in-trainer with much heavier batch averaging and reflects how
the model fits the *training* chain orderings, not how it generalises to
held-out chains.

## What this means for Paper 1 vs Paper 2

This is not a bad outcome -- it is the cleanest separation of the two
papers' claims we have so far:

- **Paper 1 (drop-in primitive)** keeps its init-equivalence claim.  At
  step 0, alpha_mem ~ 1e-5; by step 4400 it has only moved to 1e-4 to
  5e-4.  The augmented model does not collapse the base distribution
  even after thousands of steps; this is the architectural safety
  property we wanted to demonstrate.

- **Paper 1 alone is not enough to prove memory utilisation.**  The
  routing pool is open, but the model is not using it.  We need the
  recipe (deeper extract source, contrastive negative chain, hidden_14
  injection, longer windows) before the alpha_mem numbers move.

- **Paper 2 now has its baseline.**  These five rows ARE Paper 2's
  before-numbers.  The v4_hidden14_msc run currently training on the
  GH200 will be the after-numbers.  When it finishes we re-run this
  exact script and put the two tables side by side.

## Files

- `routing_v{2sp,3sp,3ab}_{val,locomo}.json` -- per-sublayer alpha_mem
  under mem and shuffle, plus position-decile and chain-depth slices.
- `cf_v{2sp,3sp,3ab}_{val,locomo}.json` -- per-(chain, depth) NLL
  deltas, depths {1, 2, 4, 8, ALL}.
- `overnight_traces_summary.txt` -- the runner's full stdout log.
- `overnight_traces_writeup.md` -- this file.
