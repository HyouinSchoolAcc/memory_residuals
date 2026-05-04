# NeurIPS 2026 — Submission Forms (3 papers)

**Drafted 2026-05-04. Deadlines: abstract reg May 5 11:59 UTC; full PDF
May 7 11:59 UTC.** Each paper has its own OpenReview submission record;
fill out the form once per paper. Authors common to all three:

- **Yueze Liu**, S;G studio (lead, ship-driver)
- **Ajay Kumdam**, S;G studio

**OpenReview IDs (placeholder format — REGISTER BEFORE MAY 5):**
- `~Yueze_Liu1`
- `~Ajay_Kumdam1`

> If either of you has *already* registered with a different counter
> (e.g. `~Yueze_Liu2` because someone else claimed `~Yueze_Liu1`),
> swap that in everywhere below. The "1" suffix is OpenReview's
> default for the first profile claimed under that exact name.

---

## Cross-paper standing answers (same for P1, P2, P3)

These fields don't change between submissions — fill them identically:

| field | answer | rationale |
|---|---|---|
| **Reviewer Nomination** | `~Yueze_Liu1` | self-nominated as best-qualified per "if no qualified reviewer exists" clause; sole alternative is `~Ajay_Kumdam1` |
| **Responsible Reviewing** | ✅ acknowledge | required tickbox |
| **Academic Integrity** | ✅ acknowledge | required tickbox |
| **Declaration** | ✅ confirm | required tickbox |
| **License** | (accept NeurIPS pre-selected default) | typically CC BY 4.0 |
| **Financial Support** | (leave blank) | neither author flagged as student needing travel support |
| **LLM Usage** | tick: *Writing assistance*; *Code assistance*; *Research assistance / ideation* | the project was developed with heavy agent assistance — declare honestly per the spec |
| **Other LLM Usage** | "Cursor agent (Claude Opus 4.7) used throughout the project for code generation, experiment scripting, run-log analysis, and manuscript drafting; all numbers in the paper come from agent-launched but human-verified runs against the locked corpus and eval script." | one-liner that's both honest and bounded |
| **LLM Experiment** | (leave unchecked / opt out) | conservative default; opting in commits *all your future NeurIPS 2026 submissions in this track* to the experiment per the form text |
| **Ready For LLM Feedback** | (leave unchecked) | the program ends *today* AOE per the form blurb; PDFs will not be ready in time. Skip. |

---

## Submission timeline (T-minus, all times UTC)

| time | what must be done |
|---|---|
| **NOW (May 4 ~21:00 UTC)** | (a) register `~Yueze_Liu1` and `~Ajay_Kumdam1` OpenReview profiles; (b) start P1 PDF assembly — see §P1 PDF below |
| **May 5 11:59 UTC (~14 h)** | **abstract reg deadline**. Title + Authors + TL;DR + Abstract + Areas + Contribution Type must be filled per paper. PDF can be a placeholder/stub for the abstract registration; full PDF goes in by May 7 |
| **May 7 11:59 UTC (~63 h)** | **full submission deadline**. PDF must be 9 pp main + checklist; supplementary zip optional |

Practical: the abstract reg is what you can't miss. Three abstract
registrations × ~5 minutes each = 15 minutes of form filling tomorrow
morning. Use the per-paper sections below.

---

# P1 — Memory Residuals (MAIN, central, hard)

## Title*

```
Memory Residuals: A Frozen-Backbone Recurrent Memory for Long-Horizon Callback Recall in Pretrained LLMs
```

## Authors*

- `~Yueze_Liu1` (Yueze Liu, S;G studio)
- `~Ajay_Kumdam1` (Ajay Kumdam, S;G studio)

## TL;DR

```
A 41.5 M-parameter recurrent memory matrix bolted onto a frozen Qwen3 backbone improves long-horizon callback cross-entropy on LongMemEval-S by +1.32 ± 0.53 nats at 0.6B (n=4 seeds) and +0.93 nats at 1.7B (n=2 seeds), with chain-shuffle confound at zero — at ~6% parameter overhead and ~1.5 h on a single H100.
```

## Abstract*

(250-word version from `ABSTRACT_NEURIPS_v3.md`; trim the
evidence-redaction sentence to drop to ~200 words if the portal
hard-limits at 1750 chars.)

```
Adding long-horizon memory to a pretrained language model is typically done either by extending the context window — which scales quadratically in compute and is fragile beyond the pretraining length — or by attaching a retrieval index, which is decoupled from the LLM's training signal. We propose Memory Residuals (memres): a fixed-size, jointly-trained recurrent memory matrix M_c that is read and written by an otherwise frozen pretrained LLM through a depth-routed cross-attention pathway, trained end-to-end with a single auxiliary loss that prevents the depth router's memory channel from collapsing. With 41.5 M trainable parameters added on top of a frozen Qwen3-0.6B (~6% overhead) and ~1.5 h of single-H100 training on the LongMemEval-S 450-chain training split, the resulting model improves callback cross-entropy on LongMemEval-S validation by Δ_dnm = +1.32 ± 0.53 nats over no-memory across four seeds (chain-shuffle confound = 0.000 ± 0.010). The same recipe scales to Qwen3-1.7B: a single-H100 6-h run gives Δ_dnm = +0.93 nats (n=2 seeds, 1.7B), with a tight per-seed spread (0.91, 0.94) and Δ_dsh ≈ 0. Two single-variable ablations identify the load-bearing components: removing the iterative readout-depth refinement collapses Δ_dnm to +0.025; removing the depth-router's α-floor drives Δ_dnm to −0.04. A third ablation we initially expected to be load-bearing — a chain-specific readout probe supplying a direct gradient to the readout — strengthens the result by 8× when removed, indicating that joint LM-NLL training alone produces a richer compressed chain-conditional context than supervising the readout for chain identification. An evidence-redaction analysis (evidence_lift = +0.001 ± 0.006) shows the memory encodes chain-conditional context rather than literal evidence recall — a distinction prior memory work rarely measures and that we argue is the correct framing for fixed-size compressive memories. Because the backbone is frozen, the result is leak-controlled by construction.
```

## PDF

**Status: NOT YET ASSEMBLED.** This is the largest write-side task.

- Existing assets:
  - `ABSTRACT_NEURIPS_v3.md` — abstract + numbers ledger
  - `NEURIPS_NUMBERS.md` — locked tables for main + ablation
  - `memory_residuals.tex` — architectural spec (Eqs. 1–10) — reusable for §2 *Method*
  - `results/exp2_chain_recipe/main_paper.tex` — failure-modes draft;
    cannibalize §3 (architecture), §7 (eval methodology), §9 (related
    work), and the appendix for the new paper
- What's missing: results section with the v27b/v28 tables, scaling
  figure (0.6B vs 1.7B vs F3-on baseline), ablation bars (depth=0,
  α-floor=0, F3=0), and the **NeurIPS 2026 paper checklist**
  (mandatory or desk-reject — pull the LaTeX template from
  https://neurips.cc/public/guides/PaperChecklist)
- Target: 9 pp main + references + appendix + checklist
- For abstract reg (May 5): a stub PDF with title/abstract/skeleton
  is sufficient if you can't get a full draft in 14 h. Real PDF
  uploads by May 7

## Checklist Confirmation*

✅ I confirm that I have included a paper checklist in the paper PDF.

(Make sure you actually paste the NeurIPS 2026 checklist
LaTeX block into the PDF before final upload.)

## Supplementary Material

**BUILT** — `memory_residuals/memres_supplementary.zip` (51 files,
0.21 MB; anonymised; sha256[:12] = `677fc82c2617`). Upload as-is.

Contents:

```
p1/code/                  src/, tools/{eval_callback,eval_chain,eval_ttt_mc}.py,
                          tests/   (no __pycache__, no .pyc)
p1/recipes/               14 launcher scripts: v27b{1..4}, v28{a,b},
                          v27{a,c} ablations, v24a {1..3} reference
p1/numbers/               NEURIPS_NUMBERS.md
p1/evals/                 28 per-checkpoint eval JSONs (v24a / v25a /
                          v27{a,b,c} / v28{a,b}) on lme_val_s512_evpos
p1/README.md              reproduction instructions for v27b-seed1
```

Excluded by design (do **NOT** add to the zip): the
`paper_artifacts/chains/lme_*.pt` corpora (LongMemEval-S is
publicly downloadable; cite the upstream release) and the
`Runs/` checkpoints (~11 GB).

Rebuild any time with:
```
python tools/build_supplementary_zips.py
```

## Primary Area*

`Foundation or Frontier Models` → `Long-Context / Memory`
*(if the dropdown splits foundation models into sub-areas)*
or `Deep Learning` → `Architectures`
*(if foundation-models is a flat top-level area)*

## Secondary Area*

`Empirical analysis (of LLMs / foundation models)`

## Contribution Type*

`Empirical research` (with a methodological component)

> Per https://blog.neurips.cc/2026/04/16/a-choice-of-contribution-types-at-neurips-2026/
> — the recipe is the contribution; the empirical evidence is the
> evidence. If the dropdown offers a combined "Empirical /
> Methodological" pick that one.

---

# P2 — Six Failure Modes (METHODOLOGY side quest)

## Title*

```
When Does Recurrent Memory Help? Six Failure Modes and a Methodology for Honest Evaluation of Augmented LLMs
```

## Authors*

- `~Yueze_Liu1` (Yueze Liu, S;G studio)
- `~Ajay_Kumdam1` (Ajay Kumdam, S;G studio)

## TL;DR

```
A 16-version post-mortem of a slot-attention memory module on Qwen3 surfaces six independent failure modes — token-averaged dilution, joint-training invisibility, category-cue leak, window leakage, redaction uncertainty, writer-objective failure — and turns each into a falsifiable audit, recovering a clean recipe that holds up under all six.
```

## Abstract*

(Adapted from `results/exp2_chain_recipe/main_paper.tex` lines 47–69;
adds a one-sentence "and here is the recipe that survived" closing
that was added on May 4 after the v27b/v28 flip.)

```
We began with a simple question: why did a recurrent memory module sometimes look useful, sometimes invisible, and sometimes actively harmful on the same long-conversation callback task? In a 16-version campaign on Memory Residuals — a slot-attention memory matrix attached to Qwen3-0.6B and Qwen3-1.7B — the answer was not a single bug. It was an evaluation problem. Token-averaged chain evaluation diluted a localized callback gain by roughly two orders of magnitude; frozen and joint training measured different causal pathways; the synthetic D4v2 callback template leaked a 32-way category prior that joint-trained models matched directly; more than half of the D4v2 callback windows exposed evidence to the LM path; evidence redaction had to be audited before a zero lift could be trusted; and a random-ID corpus removed the prior but exposed a writer objective that still did not force chain-specific bindings into M_c. We turn these failures into a six-item methodology: callback-aware evaluation, evidence-redacted memory floors, empirical and base-model prior audits, window-leakage counts, a test-time readout audit that asks whether the writer encoded the answer at all, and a held-out-corpus pivot that breaks the templated-data confound. Applying the methodology end-to-end yields a single positive recipe — a frozen pretrained backbone with a fixed-size jointly-trained M_c, supervised by LM-NLL alone — that delivers +1.32 ± 0.53 nats of chain-specific callback CE improvement on LongMemEval-S validation across four seeds at Qwen3-0.6B, with a chain-shuffle confound statistically zero. The methodology rejects all of our earlier published-looking results.
```

## PDF

**Status: 9-page draft already exists**, dated May 3 (pre-flip).

- File: `results/exp2_chain_recipe/main_paper.pdf` (9 pp)
- Source: `results/exp2_chain_recipe/main_paper.tex`
- What to update before May 7:
  1. Add a final §11 / coda referencing v27b/v28 as "the recipe that
     survived all six audits" — 1–2 paragraphs + the headline table
     from `NEURIPS_NUMBERS.md`. This positions the paper as
     constructive ("we found a clean one") not just destructive.
  2. De-anonymize author block (currently `Anonymous Authors`) —
     **wait**: NeurIPS double-blinds, so leave anonymous in the
     submission PDF; add author info only at camera-ready
  3. Add the **NeurIPS 2026 paper checklist** (mandatory)

## Checklist Confirmation*

✅ I confirm that I have included a paper checklist in the paper PDF.

## Supplementary Material

**BUILT** — `memory_residuals/failure_modes_supplementary.zip`
(53 files, 0.43 MB; anonymised; sha256[:12] = `bd68fa8e604b`).
Upload as-is.

Contents:

```
p2/code/                  src/, tools/{eval_callback,eval_chain}.py
p2/audits/                12 audit files: audit_a1_window_leakage,
                          audit_a2_redaction, audit_a3_base_prior (+1.7B
                          cloud variant), audit_a3_template_prior,
                          audit_a_corpus_prior, audit_b_literature
p2/ledgers/               NEURIPS_NUMBERS.md, runs.md
p2/eval_jsons/            32 per-ckpt JSONs (v14k/v15{a,b,e,f} from the
                          audit subjects + v27{a,b,c}/v28{a,b} as the
                          surviving recipe)
p2/README.md              audit reproduction instructions
```

## Primary Area*

`Empirical analysis (of LLMs / foundation models)`
or `Datasets and Benchmarks Track` *(if the methodology + corpus
audits look more like a benchmarks-track contribution; the D4v2 →
LME corpus pivot is itself a benchmark-design lesson)*

## Secondary Area*

`Foundation or Frontier Models` → `Long-Context / Memory`

## Contribution Type*

`Methodological research` *(or `Empirical research` if the dropdown
treats the audit framework as empirical)*

> The thing being contributed is the audit framework + diagnostic
> battery; the recipe and the negative results are evidence.

---

# P3 — Memory Residuals as a Drop-in Primitive (PAIR-RECIPE side quest)

## Title*

```
Memory Residuals: Recurrent Latent Memory with Depth-Wise Attention Routing for Chain-Persistent Conversational Modelling
```

*(Subtitle option — if you want to differentiate from P1: "A Drop-in
Architectural Primitive with Bit-Exact Init Parity".)*

## Authors*

- `~Yueze_Liu1` (Yueze Liu, S;G studio)
- `~Ajay_Kumdam1` (Ajay Kumdam, S;G studio)

## TL;DR

```
Three concrete instantiations of a depth-wise attention-residual memory router on Qwen3-0.6B trained on PG-19 + TV dialogue: a softly-initialised attention-residual router learns history-specific memory ~2× more sample-efficiently than a per-sublayer ReZero gate, while a delta-source router without parity-preserving init never recovers from its 34-nat init perturbation.
```

## Abstract*

(From `results/exp1_pair_recipe/manuscript.tex` lines 46–74.)

```
We study Memory Residuals, an architecture for chain-persistent conversational modelling that augments a pretrained Transformer backbone with a fixed-size recurrent memory matrix M_c ∈ R^{K×d} updated once per session via two-stage QKV competition, and read at inference via a depth-wise attention-residual router. In contrast to the prevailing recurrent memory designs (Transformer-XL, Memorising Transformers, Recurrent Memory Transformer, Block-Recurrent Transformer), Memory Residuals admits a non-disruptive initialisation that produces bit-identical logits to the bare backbone at step 0, isolating memory learning to a strictly additive trajectory. We empirically compare three concrete instantiations of the routing primitive on the Qwen3-0.6B backbone, trained recurrently with truncated back-propagation through time on ≈113M Qwen tokens of PG-19 books and continuity-rich TV dialogue. At matched compute we find that a softly-initialised attention-residual router learns history-specific memory roughly 2× more sample-efficiently than a per-sublayer ReZero gate, while a delta-source attention router with no parity-preserving init recovers slowly from the initial ~34-nat logit perturbation and, at the same step budget, fails to surpass the bare backbone on the shuffle test. We release the code, pre-tokenised chain corpora, and full evaluation suite (memory-on/off/shuffle/oracle, callback probe, and horizon-bucketed analysis) so that the design choices we identify as load-bearing — separate extraction and judge parameters, softmax across the 2K-pool, off-sequence read via attention residual, and zero-sum forgetting — can be falsified directly.
```

## PDF

**Status: 12-page draft already exists**, but exceeds the 9-page main
limit.

- File: `results/exp1_pair_recipe/manuscript.pdf` (12 pp)
- Source: `results/exp1_pair_recipe/manuscript.tex`
- What to update before May 7:
  1. **Trim main text to 9 pp** — push the eval-suite reference,
     horizon buckets, and full ablation bars into the appendix
     (NeurIPS allows unlimited appendix). The architecture section,
     the three routing variants, and the headline two-figure result
     are the must-keep main-text content.
  2. Add the **NeurIPS 2026 paper checklist**
  3. Stays anonymous (currently `Anonymous Authors` — already correct
     for double-blind submission)
  4. Verify the paper does not significantly overlap P1's Method
     section — if it does, cite cross-reference as "concurrent
     submission" or merge to one paper

## Checklist Confirmation*

✅ I confirm that I have included a paper checklist in the paper PDF.

## Supplementary Material

**BUILT** — `memory_residuals/pair_recipe_supplementary.zip`
(16 files, 0.09 MB; nothing to anonymise; sha256[:12] =
`aaef5d671c0c`). Upload as-is.

Contents:

```
p3/code/src/              modeling_memres.py, train_phase1.py, presets.py
p3/scripts/               9 pair-recipe launchers: run_pair_h100_gpu{0,1},
                          train_v11{g,h,i,j,k,l}_*.sh covering the AP /
                          SG / delta-source variants
p3/figures/               gate_profile.pdf, horizon_pg19_test.pdf,
                          trajectory.pdf (the three headline figures)
p3/README.md              pair-recipe reproduction instructions
```

## Primary Area*

`Deep Learning` → `Architectures`

## Secondary Area*

`Foundation or Frontier Models` → `Long-Context / Memory`
*(or `Empirical analysis` if Architectures has no Long-Context
sub-bucket)*

## Contribution Type*

`Methodological research` (architectural primitive contribution)

---

# Decisions & risks (read this before submitting)

## Three-paper concurrent-submission risk

NeurIPS does not forbid an author from submitting multiple papers, but
reviewers occasionally cross-reference and mark "substantially
overlapping" submissions as dual-submission violations. Mitigations:

- **P1 vs P3 architecture overlap.** Both papers use the same
  Memory Residuals primitive. P1's contribution is the
  *frozen-backbone training recipe + chain corpus result*. P3's
  contribution is the *primitive itself + the soft-init parity property
  + pair-corpus comparison*. They are *plausibly* separate, but:
  - In each paper's Method section, write a single paragraph
    explicitly noting the concurrent submission and which sub-claim
    that paper carries.
  - If a reviewer flags overlap on either, the safer collapse is to
    **withdraw P3 in favor of P1**: P1's headline numbers are
    stronger (+1.32 nats vs ~+0.05 nats in P3's pair recipe) and
    LongMemEval is a more recognized benchmark than the PG-19+TV
    pair corpus.
  - **If you only ship two papers, drop P3, not P2.** P2 (failure
    modes) is a different reviewer pool entirely (methodology /
    benchmarks track); P3 risks reading as a slimmer version of P1.

- **P1 vs P2 result overlap.** P1 uses v27b/v28 as its headline; P2
  uses the same numbers as the validating "recipe that survived all
  audits" coda. This is fine — P2's contribution is the audit
  methodology itself, and the v27b/v28 numbers are evidence of its
  utility. Cross-cite freely.

- **Auto-reviewer trigger.** Per the form: "any author listed on 4 or
  more papers will be automatically signed up as a reviewer". You're
  at 3, both authors. **Do not add a fourth paper without budgeting
  4-paper review obligations for both authors.**

## Author-order question

I've put Yueze first on all three. If Ajay is the architectural-prior
lead on P3, swap order on P3. Tell me before the abstract registration
goes in.

## Affiliation string

Going with `S;G studio` exactly as you typed (preserves the semicolon —
read as Steins;Gate-style stylization). If you want to render it
plainer (`S&G Studio`, `SG Studio`, `S/G studio`), say so before the
PDF goes to camera-ready; for OpenReview the affiliation string is
free-text and editable later.

## Fields that REMAIN your call (no defensible default I can pick)

| field | needs you because |
|---|---|
| Primary / Secondary Area exact dropdown labels | won't know exact NeurIPS 2026 dropdown text until you hit the form; my picks above name the *category*, you pick the closest dropdown entry |
| Contribution Type exact dropdown labels | same as above; pick `Empirical research` for P1, `Methodological research` for P2/P3 if those exact strings exist |
| Whether to opt P1 into the LLM Experiment | I defaulted opt-out (safer, doesn't bind your future submissions) — flip to opt-in only if you actively want experimental-reviewer participation |
| Whether `Ready For LLM Feedback` is worth chasing | I defaulted to skip because the May 4 AOE deadline almost certainly won't fit a full P1 PDF; if you have a partial P1 PDF you're willing to send, you can tick it for free |

---

# Pre-submission checklist (per paper, do all three)

For each of P1, P2, P3:

1. [ ] Register OpenReview ID for both authors (do this **today**)
2. [ ] Open the NeurIPS 2026 OpenReview submission portal
3. [ ] Paste **Title** from this doc
4. [ ] Add both **Authors** by OpenReview ID search
5. [ ] Paste **TL;DR** from this doc
6. [ ] Paste **Abstract** from this doc (use 200-word version of P1's
       abstract if portal hard-limits at 1750 chars)
7. [ ] Pick **Primary** + **Secondary** area from dropdown (use this
       doc's recommendations as guide)
8. [ ] Pick **Contribution Type** (this doc's recommendation)
9. [ ] Tick **Checklist Confirmation** (and actually paste the
       checklist into your PDF before final upload)
10. [ ] Tick **Responsible Reviewing**, **Academic Integrity**,
        **Declaration**
11. [ ] Enter **Reviewer Nomination**: `~Yueze_Liu1`
12. [ ] Pick **LLM Usage** boxes per the cross-paper standing answer
13. [ ] Paste **Other LLM Usage** text from cross-paper standing
        answer
14. [ ] Leave **LLM Experiment** unchecked
15. [ ] Leave **Ready For LLM Feedback** unchecked
16. [ ] Accept default **License**
17. [ ] Upload PDF (placeholder fine for May 5 abstract reg; real
        PDF by May 7)
18. [ ] (After May 7) Upload **Supplementary Material** zip per the
        manifest in this doc

---

# Per-paper write-side TODOs (the actual hard work between now and May 7)

**P1 (highest effort, ~30–40 h of writing):**
- [ ] Spin up a fresh `paper_p1.tex` based on the NeurIPS 2026 style
- [ ] §1 Intro — adapt from `ABSTRACT_NEURIPS_v3.md` and
      `README.md` "Why we think this is groundbreaking" block
- [ ] §2 Method — adapt from `memory_residuals.tex` Eqs. 1–10 +
      v13/v14 symmetry-break stack
- [ ] §3 Experimental setup — corpus, eval script, frozen-backbone
      protocol, seed protocol
- [ ] §4 Results — paste headline table from `NEURIPS_NUMBERS.md`,
      scaling figure (0.6B vs 1.7B), per-chain sanity figure
- [ ] §5 Ablations — depth=0, α-floor=0, F3=0 single-variable
      results from `NEURIPS_NUMBERS.md`
- [ ] §6 Surprise + framing — F3-OFF flip, evidence-lift = 0
      reframed as chain-conditional context
- [ ] §7 Limitations — n=2 at 1.7B, OOD LoCoMo failure, future scale
- [ ] §8 Related work — cite RMT, BRT, AutoCompressors, NoLiMa,
      RULER, LongMemEval, LoCoMo
- [ ] Appendix — bootstrap-over-chains 95% CI, full per-chain
      tables, training curves, watcher logs
- [ ] **Paste the NeurIPS 2026 paper checklist before final upload**

**P2 (medium effort, ~10 h of polish):**
- [ ] Open `results/exp2_chain_recipe/main_paper.tex`
- [ ] Add §11 (or extend §10 Conclusion) with the v27b/v28 coda:
      "applying the audit battery prospectively, the recipe that
      survives is..." + headline table
- [ ] Verify §1 still reads as a methodology contribution (not a
      negative-results paper) — adjust framing in 2–3 sentences if
      it tilts too pessimistic
- [ ] **Paste the NeurIPS 2026 paper checklist**

**P3 (medium effort, ~8 h of trim):**
- [ ] Open `results/exp1_pair_recipe/manuscript.tex`
- [ ] Trim main text from 12 pp to 9 pp (push horizon buckets, full
      eval suite, and ablation bars into appendix)
- [ ] Add 1 paragraph in §1 Method noting concurrent submission
      relationship to P1 (claim split)
- [ ] **Paste the NeurIPS 2026 paper checklist**

---

# Anything else?

If you also want me to:

- (a) draft the NeurIPS 2026 paper checklist responses for each paper
  (the checklist is question-answer format with required justifications),
- (b) start writing P1's `paper_p1.tex` skeleton from the
  `memory_residuals.tex` + `main_paper.tex` reusable bits,
- (c) draft the §6 / surprise section text for P1 (the F3-OFF flip
  framing) so it doesn't need to be written from scratch,
- (d) compute the bootstrap-over-chains 95% CI for the P1 headline
  (~2 min compute, would tighten the abstract claim),

…just say which letters and I'll go.
