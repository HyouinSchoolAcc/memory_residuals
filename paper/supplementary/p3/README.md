# Attention Parity Beats ReZero — Supplementary Material

This bundle contains the architecture, the pair-recipe trainer, the
routing-variant launcher scripts, and the three headline figures
referenced in the paper.

## Headline (paper §1, §6)

> Three concrete instantiations of a depth-wise attention-residual
> memory router on a frozen Qwen3-0.6B backbone, trained recurrently
> on ~113M PG-19 + TV-dialogue tokens:
>   * `attention_parity` (soft, ±4 bias) — learns history-specific
>     memory in ~2/5 the steps of a per-sublayer ReZero gate.
>   * `simple_gate` (ReZero) — opens the channel but plateaus.
>   * `attention_base` (no parity init) — never recovers from a
>     ~34-nat init perturbation under the same compute budget.

## Layout

```
code/src/
  modeling_memres.py    Memory Residuals architecture (the three routing
                        primitives are dispatched on `--memres_mode`).
  train_phase1.py       Pair-recipe TBPTT trainer used to produce the
                        matched-compute trajectory in Figure 1 / Table 2.
  presets.py            Named (backbone, K, L_E, N) tuples.
code/tools/
  routing_trace.py            Produces routing_v3*_val.json (Table 5).
  routing_trace_bootstrap.py  Cluster-bootstrap CI from those JSONs.
  counterfactual_eval.py      Produces cf_v3*_val.json (Table 6 + App. C).
  eval_chain.py               Produces chain_v2_phaseA_*_eval.json (Table 3).
  callback_probe.py           Per-callback-token NLL probe (auxiliary).
  horizon_analysis.py         Bucketed Δ_sh figure (Fig. A2 / Table A3).
  bootstrap_ci.py             Helper used by the *_bootstrap.py scripts.
scripts/
  run_pair_h100_gpu*.sh             Two-GPU pair-recipe launchers.
  train_v11g_ap_baseline_gh200.sh   attention_parity (soft, +4/0).
  train_v11h_ap_norm1_gh200.sh      readout_norm_init=1.0 ablation.
  train_v11i_ap_pm4_gh200.sh        ±4 vs ±32 bias ablation.
  train_v11j_ap_carry_depth_gh200.sh  deeper recurrence ablation.
  train_v11k_ap_no_evidence_gh200.sh  evidence-label ablation.
  train_v11l_ap_frozen_backbone_*.sh  frozen-backbone ablation
                                      (the paper's headline cell).
figures/
  trajectory.pdf            Figure 1 (in-trainer Δ_nm and Δ_sh).
  gate_profile.pdf          Figure A1 (per-sublayer routing weight).
  horizon_pg19_test.pdf     Figure A2 (horizon-bucketed Δ_sh).
eval_dumps/                 Source JSONs that produce the headline tables;
                            see "Reproducing tables from JSONs" below.
README.md                   this file.
```

## Reproducing tables from JSONs

The JSONs in `eval_dumps/` are the raw source-of-truth for every
numerical claim in the paper:

| paper claim | JSON file |
|---|---|
| Table 1 init parity                    | `init_parity_test.json` |
| Table 3 standalone (PG-19 test row)    | `chain_v2_phaseA_softparity_b4_step2000_eval.json` (key `pg19_test`) |
| Table A1 standalone (PG-19 val, LoCoMo)| same JSON, keys `pg19_val` and `locomo` |
| Table A3 horizon buckets               | `chain_v2_phaseA_softparity_b4_step2000_eval_horizon.json` |
| Table 5 routing-mass + 95% CIs         | `routing_trace_bootstrap_v3.json` |
| Table 5 raw alpha-trace traces         | `routing_v3sp_val.json` (parity), `routing_v3ab_val.json` (base) |
| Table 6 + App. C counterfactual        | `cf_v3sp_val.json` (parity), `cf_v3ab_val.json` (base) |

## Reproducing the headline

1. Pre-tokenise the PG-19 train/val/test splits and the 30 high-
   continuity TV transcripts at S=512 sessions/chain (the
   pair-corpus builder lives at `tools/build_pair_corpus.py` in the
   parent repo; corpora are publicly available — PG-19 under
   Apache-2.0 from the Compressive-Transformers release, LoCoMo
   under MIT, MSC under CC-BY-NC-4.0).
2. Launch any `scripts/train_v11*.sh`. The cell prints in-trainer
   eval points every 200 steps; collecting steps {200, 400, 600, ...,
   2000} reproduces Table 2 of the paper.
3. The three figure PDFs in `figures/` are pre-rendered. Figure 1
   (`trajectory.pdf`) plots the in-trainer eval trajectory; Figure A1
   (`gate_profile.pdf`) shows per-sublayer routing weight; Figure A2
   (`horizon_pg19_test.pdf`) shows the horizon-bucketed Δ_sh on PG-19
   test. Source eval dumps that produce the figures are not included
   in this bundle but the launcher scripts above generate them as a
   side effect at the configured `--eval_every` cadence.

## Compute

Each `train_v11*.sh` cell takes ~10 wall-clock hours on a single
H100 80GB at bf16 with gradient checkpointing (~10.8k tokens/s,
6,000 TBPTT steps).
