# Findings — manuscript progress

## Starting point

`experiments/exp2_long_horizon_recipe/draft.tex` was a 22-line
placeholder (article preamble, title, "TBD" intro, end). No abstract,
no related work, no method, no recipe formalism, no experimental
table, no negative-result documentation.

## What I wrote (single editing pass)

Replaced the placeholder with a complete first-pass manuscript draft,
~600 lines, NeurIPS 2024 preprint style. Sections:

1. **Abstract** (single paragraph, ~200 words). States the
   architectural primitive context, the four-piece training recipe,
   and the headline negative-result finding (strict `±32` init is
   bf16-saturating).
2. **§1 Introduction.** Motivates conversational long-horizon recall,
   states what the recipe paper claims (separately from the primitive
   paper), enumerates the four pieces with one-paragraph rationale
   each, gives the 2×2 ablation framing, declares scope/non-scope,
   and ends with a roadmap.
3. **§2 Background: the architectural primitive.** Folded-in summary
   of `experiments/exp1_drop_in_primitive/manuscript.tex` so this
   manuscript stands on its own. Memory state, two-stage QKV write
   (Eq. extract / judge), off-sequence read, the
   `attention_parity` routing primitive, soft vs strict init
   distinction, recurrent training loop. Includes
   `Table tab:v3_baseline` reproducing the v3 in-trainer trajectory
   (200, 1000, 2000, 3000, 4000, 4400) so the reader has the
   book-only baseline numbers in front of them.
4. **§3 The four-piece training recipe.** Each piece (contextual
   extract, mixed corpus, contrastive loss, soft init) gets its own
   subsection with the failure mode it addresses, the proposed fix,
   and any init-parity preservation argument. Ends with the full
   `train_chain.py` launch command for the soft-init recipe.
5. **§4 Setup.** Backbone + parameter groups, corpus, training
   hyperparameters, evaluation metrics, mechanistic probes.
6. **§5 Experiments.** Headline 2×2 grid table; per-cell results
   table (placeholder, with v3 baseline numbers in cell D filled in);
   mention of RAG baseline. Several `\todo{}` markers where numbers
   from the soft-init rerun would land.
7. **§6 Negative result: strict ±32 init is operationally
   untrainable.** This is the most concrete contribution of the
   manuscript right now since the supporting evidence is in hand.
   Includes:
   - The empirical observation table (`tab:strict_eval`) with mem ==
     nomem == shuffle = +0.0000 across 7 representative steps.
   - The bf16 saturation argument with explicit numerics
     (`α_mem ≈ 1.6e-28`, gradient `~10^-56` underflows below bf16's
     `~10^-38` denormal floor).
   - Comparison to the soft-init book-only baseline ('the same
     mechanism works at ±4').
   - A reading recommendation: "any parity-preserving init for
     residual-stream routers should be calibrated such that the
     backward gradient through the routing softmax stays inside the
     float-format's representable range; for bf16 this places an
     upper bound of |bias| ≲ 8."
8. **§7 Discussion and limitations.** What the recipe is/isn't, why
   the four pieces are non-orthogonal, three concrete limitations
   (0.6B only, MSC/LoCoMo held out, soft-init rerun pending).
9. **Appendix A.** Per-step diagnostics from the strict-init run
   (placeholder for the full 27-row EVAL trajectory).
10. **Appendix B.** Mixed-corpus build documentation.

## Numerical anchors used (all sourced)

- `tab:v3_baseline` rows from
  `paper_artifacts/eval/chain_v3_training_summary.md`.
- v3 standalone bootstrap CIs from §"Standalone references" of same.
- v3 routing-trace `α_mem = 4.7e-4` and `+4.79%` mem-vs-shuffle gap
  from `paper_artifacts/eval/overnight_traces_writeup.md`.
- v3 counterfactual `Δ(d=ALL) = +0.013` nat from the same.
- `tab:strict_eval` (the negative-result table) rows from the actual
  GH200 log
  `~/memory_residuals/paper_tools/cloud_watchdog/logs/chain_v4_hidden14_msc.log`,
  selected step 200, 1000, 2000, 3000, 4000, 5000, 5400.
- bf16 floor (`~6e-8` representable, `~10^-38` denormal) is from the
  IEEE 754 binary16 / bfloat16 spec, no source needed.

## TODOs still in the draft (search `\todo{`)

- Throughput claim for soft-init rerun (currently parenthetical
  `\todo{check whether final run achieves higher throughput at soft
  init}`).
- Headline numbers for cells A, B, C in `tab:headline_results`
  (soft-init rerun pending).
- RAG baseline numbers (pipeline pending).
- Headline number to insert into §discussion conclusion.
- Per-step diagnostics appendix needs the full 27-row EVAL table
  appended verbatim.

These are flagged as `\todo{...}` markers that show up in red in the
compiled PDF.

## Word count delta (rough)

Was: ~30 words of placeholder text.
Now: ~3 700 words across all sections (counting math + table
captions; not counting verbatim bash blocks).

## What I did NOT modify

- `experiments/exp1_drop_in_primitive/manuscript.tex` (untouched —
  out of scope by user instruction).
- `paper_artifacts/eval/*` (read-only).
- Anything outside `experiments/exp2_long_horizon_recipe/`.
- Any code (no `paper_tools/` edits this session — I read
  `train_chain.py` to confirm the contrastive loss path but did not
  modify it).

## Compile attempt

Will attempt `pdflatex draft.tex` next to verify the draft is
syntactically clean. Bibliography (`references.bib` from exp1) is not
copied in yet, so the bibliographic citations will produce missing-
ref warnings but should not break the build.
