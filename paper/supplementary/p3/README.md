# Pair-Recipe Drop-in Primitive — Supplementary Material

This bundle contains the architecture, the pair-trainer, the three
routing-variant launcher scripts, and the figure source data needed to
reproduce the pair-recipe headline:

> A softly-initialised attention-residual router learns history-specific
> memory ~2× more sample-efficiently than a per-sublayer ReZero gate;
> a delta-source router with no parity-preserving init never recovers
> from its 34-nat init perturbation.

## Layout

```
code/
  src/modeling_memres.py    Memory Residuals architecture
  src/train_phase1.py       pair-based warmup trainer (used by this paper)
  src/presets.py            named (backbone, K, L_E, N) tuples
scripts/                    pair-corpus launchers:
  run_pair_h100_gpu0.sh
  run_pair_h100_gpu1.sh
  train_v11g_ap_baseline_gh200.sh   (AP soft-init baseline)
  train_v11h_ap_norm1_gh200.sh      (norm-1 variant)
  train_v11i_ap_pm4_gh200.sh        (±4 hard-bias variant)
  train_v11j_ap_carry_depth_gh200.sh
  train_v11k_ap_no_evidence_gh200.sh
figures/                    the three headline figures (PDF) + their
                            source pickles where applicable
README.md                   this file
```

## Reproducing the headline

Pre-tokenise PG-19 + TV dialogue chains per the paper's §3 description
(corpora are publicly available; the pair-corpus builder lives in
`tools/` of the parent repo). Launch any `train_v11*.sh` to reproduce
that variant's training trajectory, and the figure-3 trajectory.pdf
reproduces directly from the eval JSON dumps in `figures/`.
