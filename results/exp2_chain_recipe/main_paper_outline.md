# Main Paper Outline

Title: **When Does Recurrent Memory Help? Six Failure Modes and a Methodology for Honest Evaluation**

## Abstract

Core claim: the v11--v16 campaign became an honest-evaluation paper rather than a positive architecture paper. The abstract foregrounds six failure modes and the F2 prescription.

Evidence:
- `runs.md` v16 pre-summary: v14k redaction survival, D5 motivation, v16a early result.
- `audit_a_corpus_prior.md`: D4v2 category prior CE and v15b collapse.
- `audit_a2_redaction.md`: redaction floor is clean.
- `v17_wildcards.md`: F2/NIAH writer warmup motivation.

## 1. Introduction

Frames the question: when is recurrent memory necessary, and how do we know it is the causal path? Introduces D4v2, D5, and the shift from "memory works" to "we found hidden pathways."

Evidence:
- `intro_v3.tex`: prior framing around abstract callback and callback-aware eval.
- `runs.md` v15/v16 summaries: sequence of v14k, v15, v16 findings.
- `audit_b_literature.md`: shortcut-learning framing.

## 2. Memory Residuals Architecture

Short self-contained description of the architecture: extraction from `C_t`, slot-attention writer into `M_c`, readout `m_t`, Block-Attention Residual routing.

Evidence:
- `results/exp1_pair_recipe/manuscript.tex`: architecture macros/style and baseline description.
- `draft.tex`: primitive section with extraction, judge, and readout equations.
- `section8_recipe_revised.tex`: input RMSNorm and revised recipe context.

## 3. Failure Modes At A Glance

Table 1 summarizes the six failure modes:
- token-averaged dilution,
- joint-training invisibility,
- category-cue leak,
- window leakage,
- redaction uncertainty,
- writer-objective failure.

Figure 1 sketches the five pathways from context/`M_c` to the LM head.

Evidence:
- `runs.md` v15 OPEN AUDIT and v16 pre-summary.
- `audit_a1_window_leakage.md`.
- `audit_a2_redaction.md`.
- `audit_a3_base_prior.md`.
- `v17_wildcards.md`.

## 4. Failure Modes 1--2: Dilution And Joint-Training Invisibility

Explains why standard token-averaged eval hides localized callback gains and why frozen vs joint training are measuring different causal pathways. Includes Table 2 with v14k/v15 ablation rows and audited values.

Evidence:
- `section_eval_callback.tex`: standard eval vs callback-aware contradiction.
- `results/eval_v14v15/v14k_best_callback_aware.json`: `pa_cb_dnm=1.43999`, `pa_cb_dsh=0.05379`, `pa_cb_evidence_lift=0.07087`.
- `results/eval_v14v15/v15a_best_callback_aware.json`: v15a post-hoc callback metrics.
- `audit_a3_base_prior.md`: v15a/v15b/v15e/v15f CE/prior gaps.
- `audit_a3_base_prior_1p7b_cloud.md`: 1.7B audit cross-checks.

## 5. Failure Mode 3: The Category-Cue Corpus Leak

Shows D4v2's callback template gives a 32-way prior and that joint-trained v15b matches it. Includes Table 3 comparing D4v2 and D5.

Evidence:
- `audit_a_corpus_prior.md`: `CE=2.91`, v15b `pa_cb_ce_mem ~= 2.97`.
- `audit_a3_base_prior.md`: `CE_template_prior=2.9087`, v15b best/final gaps.
- `audit_a1_window_leakage.md`: frozen-base CE values.

## 6. Failure Mode 4: Random IDs Are Not A Cure

Explains D5's design and the negative v16a result: removing the prior makes the evaluation honest but exposes the writer-objective failure.

Evidence:
- `runs.md` v16 corpus/loss/best-metric bullets.
- `v17_wildcards.md`: v16a `evidence_lift=+0.011`, `pa_cb_dnm=-0.076`, `nce_loss=1.375`, D3-MC pair/self `0.012`, `alpha_mem_mean=0.01`.

## 7. Failure Mode 5: Leak Audits As Methodology

Describes redaction-clean audit and window-leakage counts. Includes Figure 2 as a trajectory summary using audited checkpoint values rather than interpolated curves.

Evidence:
- `audit_a2_redaction.md`: default/cross/skip/zero agreement and direction.
- `audit_a1_window_leakage.md`: k=4 and k=3 leakage fractions.
- `runs.md` v15/v16 ledger values.

## 8. Prescription: F2 NIAH Writer Warmup

Presents F2 as the next intervention: direct writer-side extractive supervision through a probe head, discarded at eval. Includes Table 4 placeholder for D5 TTT-readout and Figure 3 placeholder for v17a/v17e.

Evidence:
- `v17_wildcards.md` section 1: NIAH writer warmup recipe and predictions.
- `v17_wildcards.md` section 5: TTT-on-`M_c`/readout capacity diagnostic.
- User assignment: parent will fill TTT and v17 numbers.

## 9. Related Work

Connects findings to shortcut learning and long-context benchmark pitfalls.

Evidence:
- `audit_b_literature.md`: NoLiMa, RULER, Lost in the Middle, LongBench, BABILong, LongMemEval, RMT, ARMT, AutoCompressors, Memorizing Transformers, etc.

## 10. Conclusion And Open Questions

States the conservative lesson: recurrent memory helps only when non-memory paths are uninformative and the writer has a direct enough gradient to store the binding. The next decisive experiment is F2.

Evidence:
- `runs.md` v16 summary.
- `v17_wildcards.md` top-1 recommendation.

## Appendix

Appendix sections:
- window leakage counts,
- redaction variants,
- template prior construction,
- callback-aware metric definition,
- base CE and redaction summary tables,
- limitations.

Evidence:
- `audit_a1_window_leakage.md`.
- `audit_a2_redaction.md`.
- `audit_a3_base_prior.md`.
- `section_eval_callback.tex`.

## Known Placeholders To Fill

- Table 2: v15c full CE/prior values, v15f callback `dnm`, any final v15e/v15f preferred lift convention.
- Table 3: D5 base CE if measured.
- Table 4: D5 TTT-readout results on v15a/best and v15e/best.
- Figure 3: v17a/v17e warmup/probe CE, callback `dnm`, and evidence lift.
- Any LoCoMo/MSC transfer numbers if they land before submission.
