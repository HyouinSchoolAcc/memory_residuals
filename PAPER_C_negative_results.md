# Paper C — Three Ideologies of Recurrent Memory in Pretrained LLMs: A Negative-Results Audit

**Status:** plan (2026-05-04). Targets NeurIPS 2026 **Main Track** as
**`Negative Results`** contribution type — one of the five official
contribution types listed in the 2026 Main Track Handbook ("the main
contribution is in understanding a negative result; the bar for these
submissions is expected to be high"). Workshop fallback: NeurIPS 2026
**Workshop on I Can't Believe It's Not Better!** (negative-results
workshop, runs most years; 2025 was the eighth instance) or any
"Foundation Models / Long-Context / Memory" workshop with a
methodology focus. Final 2026 workshop list announced 2026-07-11.

---

## TL;DR

> A 16-version campaign on Memory Residuals — a slot-attention
> recurrent memory module attached to Qwen3-0.6B and Qwen3-1.7B —
> surfaces three competing **ideologies** under which the field is
> trying to give pretrained LLMs persistent memory: the *joint
> training* school, the *frozen backbone* school, and the *retrieval*
> school. Each ideology promises a different mechanism, and each
> *fails by default* unless the evaluation also rules out the other
> two. We catalogue six failure modes, show how each ideology is
> susceptible to a different subset of them, and argue that the
> current state of the literature systematically over-reports memory
> success because the controls are ideology-specific and rarely
> overlap. We close with a positive **methodology**: a six-item audit
> battery (callback-aware evaluation, evidence-redacted floor, base-
> model prior, template prior, window-leakage count, test-time
> readout audit) that any recurrent-memory paper should report
> regardless of ideology.

## One-line headline claim

`Recurrent memory in a pretrained LLM does not have one failure mode;
it has six. They live in three different "ideological" research
programs, are individually load-bearing, and most published results
are not robust to all six simultaneously.`

---

## The "three ideologies" framing

### Ideology I — Joint training (the optimistic school)

> "Just train the LM and the memory together; the gradient will sort
> out which path carries the binding."

- *Examples in literature*: AutoCompressors, TRIME, parts of
  Memorising Transformers, RIMs.
- *Why it's seductive*: Best-case end-to-end CE; no auxiliary losses
  to tune; no separate write/read circuits to design.
- *How it fails*: When multiple paths can solve the training
  objective, gradient descent allocates credit to the easiest path.
  If the LM head can solve the callback from a template prior or
  category cue, it does — and the memory channel receives noise
  gradient and collapses to content-blind. **(Failure modes: 2, 3)**

### Ideology II — Frozen backbone (the conservative school)

> "Freeze the LM. Train only the memory module. The only path from
> evidence to next-token loss is through M_c — by construction."

- *Examples*: Recurrent Memory Transformer (in its frozen-LM ablation),
  Block-Recurrent (with a cold-start protocol), our P1 / Paper A.
- *Why it's seductive*: Leak-controlled by construction; if memory
  helps, the help has to come through M_c.
- *How it fails*:
  - **Token-averaging dilutes localised callback gain by orders of
    magnitude** — standard chain-CE eval reports Δ ≈ 0 even when
    callback-aware eval reports Δ_cb = +1.44 nats.
  - **Δ_shuffle ≈ 0** is consistent with two interpretations: (a) the
    writer encodes chain-conditional context (style, topic, prior
    turns) that any other chain's M_c also carries, or (b) the
    writer is not using chain-specific information at all. Without
    a callback-aware metric and an evidence-redacted floor, the two
    cannot be separated. **(Failure modes: 1, 5, 6)**

### Ideology III — Retrieval (the practical school)

> "Don't bother with recurrent state. Index past sessions, retrieve
> at query time, append to context. Done."

- *Examples*: Lewis et al. 2020 (RAG), Memorising Transformers (kNN
  index), DOS-RAG (Laitenberger et al. 2025).
- *Why it's seductive*: Differentiable-free, decoupled from training,
  the practical default in production systems.
- *How it fails*:
  - **Surface-form / literal-match leakage** when query and evidence
    share lexical overlap; brittle when callbacks depend on implicit
    state (NoLiMa, RULER show this directly).
  - **The reading head must be trained to use retrieved context**;
    a bare pretrained LM at 0.6B cannot exploit appended sessions,
    and even at 1.7B oracle-RAG with gold evidence under-performs
    a frozen-backbone recurrent memory. **(Failure modes: 3, 4)**

The key argument of Paper C: **none of the three ideologies is
self-validating**. Joint training will report a clean "we beat the
no-memory baseline" without ruling out (3) the template prior. Frozen
backbone will report a clean "Δ_dnm > 0" without ruling out (1) the
token-averaging dilution. Retrieval will report a clean "we beat
no-context" without ruling out (4) literal-match leakage. Honest
evaluation requires a single audit battery that crosses all three.

---

## The six failure modes (catalogue)

(Adapted from `results/exp2_chain_recipe/main_paper.tex` Table 1; we
add a third column mapping each failure to one or more ideologies.)

| # | Failure mode | Symptom | Ideology susceptible | Diagnostic |
|---|---|---|---|---|
| 1 | **Token-averaged dilution** | Standard eval reports Δ_dnm ≈ 0 although callback tokens improve. | Frozen backbone (II) | `tools/eval_callback.py` callback-token-only CE |
| 2 | **Joint-training invisibility** | Unfrozen backbone collapses evidence-lift to ≈ 0. | Joint training (I) | Freeze-vs-joint ablation; compare to empirical prior. |
| 3 | **Category-cue leak / template prior** | "Favorite COLOR" reduces answer to a 32-way prior. | Joint training (I), Retrieval (III, partly) | `tools/audit_a3_template_prior.py` — `P(item∣category)` CE = 2.91 nats; v15b CE = 2.97 nats. |
| 4 | **Window leakage** | Evidence directly enters the LM attention window. | All three | `tools/audit_a1_window_leakage.py` — count evidence positions in callback windows; D4v2 leaks 58.5 % of train chains at `window_k = 4`. |
| 5 | **Redaction uncertainty** | Zero lift could be dismissed as a leaky floor. | Frozen backbone (II) | `tools/audit_a2_redaction.py` — default/cross/skip/zero redaction agreement within 0.05 nats. |
| 6 | **Writer-objective failure** | D5 random-IDs removes priors but evidence-lift stays near zero. | Frozen backbone (II), and any architecture without writer-side extractive supervision | D5 corpus (random codes, masked evidence-session loss) + TTT-readout audit (`tools/d5_ttt_readout.py`). |

The key claim of the paper: **the six failure modes are
*independent*** — fixing one does not fix the others. We document each
with a self-contained diagnostic and a worked numerical example from
the campaign.

---

## Story / outline (9 main pages, Negative Results contribution type)

### §1 Intro (1.25 pages)

A pretrained LLM has at least three plausible architectural routes
to long-horizon memory: jointly trained recurrent state, frozen-
backbone recurrent state, and retrieval. The literature has matured
each, and each has reported successes — but the *evaluation
controls* differ across the three programs. A joint-training paper
typically does not report an evidence-redacted floor; a frozen-
backbone paper typically does not report a template prior audit;
a RAG paper rarely reports a chain-shuffle confound. **The result
is that each ideology evaluates only the failure modes its own
prior literature is sensitive to**, and produces published numbers
that are not robust to the failure modes from the other two.

We make three claims:

1. **Three-ideology decomposition.** Joint training, frozen backbone,
   and retrieval are not interchangeable engineering choices; they
   are different **research programs**, each with a distinct set of
   failure modes that need separate diagnostics.
2. **Six-failure-mode catalogue.** The 16-version campaign on Memory
   Residuals (v11–v28, two model scales, two training corpora)
   surfaces six independent failure modes, each load-bearing on at
   least one ideology and ablatable.
3. **Six-item audit battery.** A single methodology — callback-aware
   evaluation, evidence-redacted floor, empirical and base-model
   prior audits, window-leakage counts, redaction validity audit, and
   test-time readout audit — applies to all three ideologies and is
   cheap (<1 H100-hour total per checkpoint).

The paper is presented as **negative results** in the NeurIPS 2026
Negative Results contribution type. The bar for that contribution
type is high; we meet it by (a) making the failures falsifiable, (b)
providing diagnostics that detect each, and (c) giving a constructive
methodology that survives all six.

### §2 Related work and the three ideologies (1.5 pages)

(Adapted from `audit_b_literature.md`; structure is one
sub-paragraph per ideology, three sub-sub-paragraphs per failure
mode citing the literature that sets up each.)

- **Joint training**: Memorising Transformers (Wu et al. 2022),
  AutoCompressors (Chevalier et al. 2023), TRIME (Zhong et al. 2022),
  RIMs (Goyal et al. 2021), NTM/DNC family. Cite each on:
  failure mode 2 (joint-training invisibility) and failure mode 3
  (template prior leak).
- **Frozen backbone**: RMT-frozen (Bulatov et al. 2022 ablation),
  Block-Recurrent Transformer (Hutchins et al. 2022),
  Compressive Transformer (Rae et al. 2019), AutoCompressors-frozen
  ablation. Cite on: failure mode 1 (token-averaging) and failure
  mode 5 (redaction uncertainty).
- **Retrieval**: RAG (Lewis et al. 2020), Memorising Transformers
  (kNN), DOS-RAG (Laitenberger et al. 2025), RAPTOR, Self-RAG. Cite
  on: failure mode 3 (template/category prior) and failure mode 4
  (window leakage / surface-form leak).
- **Anti-shortcut design lessons**: NoLiMa (Modarressi et al. 2025),
  RULER (Hsieh et al. 2024), Lost-in-the-Middle (Liu et al. 2024),
  LongBench (Bai et al. 2024), BABILong (Kuratov et al. 2024),
  ZeroSCROLLS, LongMemEval (Yu et al. 2024), Geirhos et al. 2020 on
  shortcut learning.

### §3 Memory Residuals as the running example (0.5 page)

Self-contained one-page summary of the architecture:

- M_c ∈ R^{K×d}, K = 128 slots
- Stage 1 extract (Eq. 1)
- Stage 2 judge with 2K-pool softmax (Eq. 2)
- Off-sequence read (Eq. 3) into a depth-routed attention residual

Memres is *deliberately* used as the running example because it
admits all three ideological framings: it can be trained joint
(v15a/b), frozen-backbone (v14k, v15a, v27b/v28), or paired with a
RAG baseline (`tools/eval_rag_baseline.py`). The campaign covers all
three; the failure-mode catalogue is observed across the same
architecture under different training regimes.

### §4 Failure mode 1 — Token-averaged dilution (0.5 page)

**Symptom.** v14k callback-aware eval reports Δ_dnm^cb = +1.44 nats;
standard chain-CE eval reports Δ_dnm = −0.085 on the *same*
checkpoint.

**Cause.** Callback answer tokens are ~1.4 % of the full-window
score; small noise on the other 98.6 % dominates the average.

**Diagnostic.** `tools/eval_callback.py` scores only annotated
callback answer tokens.

**Ideology susceptibility.** Frozen-backbone (II) papers that report
chain-level CE are most exposed. Joint-training (I) and retrieval
(III) papers usually report task-specific accuracy, which avoids
this failure but exposes them to others.

### §5 Failure mode 2 — Joint-training invisibility (0.5 page)

**Symptom.** v15b joint 0.6B reaches CE_mem ≈ 2.97 nats (within 0.06
of the empirical 32-way category prior) and evidence_lift ≈ 0. The
LM head solves the callback from the template, M_c is unused.

**Cause.** Joint training opens a non-memory path through the LM
head. Gradient descent allocates credit to the path with the lowest
optimisation cost — the LM-head path is shorter and richer than the
write→read→LM-head path.

**Diagnostic.** Compare frozen-backbone vs joint-training versions
of the *same* architecture at *matched data*. Report the empirical
prior CE explicitly. If joint matches the prior within 0.1 nats and
frozen does not, the joint result is "the LM learned the prior,"
not "the memory learned the binding."

**Ideology susceptibility.** Joint training (I) by definition.

### §6 Failure mode 3 — Category-cue leak / template prior (0.5 page)

**Symptom.** D4v2 callbacks repeat the category in the user question:
"What was my favorite COLOR again?" The 32-way category prior CE is
2.91 nats; v15b reaches 2.97. Smoking gun.

**Cause.** Synthetic benchmarks reuse a small closed-set vocabulary
and template their callbacks; the resulting `P(answer | category)`
is sharp and the LM head can fit it directly.

**Diagnostic.** `tools/audit_a3_template_prior.py` builds an empirical
Laplace-smoothed prior over categories; report it as a baseline for
every memory paper that uses a synthetic callback corpus.

**Ideology susceptibility.** Joint training (I) most exposed.
Retrieval (III) partially exposed (the LM head still fits the prior
even with retrieved context). Frozen backbone (II) immune by
construction *if and only if* the LM head is truly frozen — but
*not* immune at the supervised-readout-probe level (see §10 on F3).

### §7 Failure mode 4 — Window leakage (0.5 page)

**Symptom.** At `window_k = 4` (D4v2 default), 58.5 % of train chains
and 59.6 % of validation chains have at least one evidence session
inside the LM attention window at callback scoring time. Memory may
appear to help because evidence is in-window.

**Cause.** Default scoring windows in long-conversation benchmarks
often do not enforce a strict "evidence is outside the window" gap.

**Diagnostic.** `tools/audit_a1_window_leakage.py` — count the
fraction of chains where the gold evidence session falls inside the
callback scoring window.

**Ideology susceptibility.** All three. RAG (III) explicitly relies
on this path (retrieved evidence enters the window deliberately);
frozen and joint backbones are exposed when the corpus does not
enforce the gap.

### §8 Failure mode 5 — Redaction uncertainty (0.5 page)

**Symptom.** evidence_lift ≈ 0 could mean (a) the writer is not
using chain-specific information OR (b) the redaction is leaky and
M_c built from "redacted" data still has the answer.

**Cause.** Default redaction strategies (zero-token, cross-chain,
skip) can each carry partial signal. Without an ablation, a zero
lift cannot be distinguished from a leaky floor.

**Diagnostic.** `tools/audit_a2_redaction.py` — compare default,
cross-chain, skip-evidence, and zeroed-token redactions. On v15a/best:
default 5.1235, cross 5.0911, skip 5.0857, zero 5.0896. Spread =
0.038 nats, within the pre-declared 0.05-nat band. evidence_lift ≈ 0
is **not** an artefact of leaky redaction.

**Ideology susceptibility.** Frozen backbone (II) most exposed.

### §9 Failure mode 6 — Writer-objective failure (0.75 page, headline)

**Symptom.** D5 random-codes corpus (per-chain random alphanumeric
ID, evidence-session LM loss masked, no category cue) removes the
prior — and evidence_lift *still* stays near zero. v16a reports
evidence_lift = −0.0024 at step 100; later state evidence_lift =
+0.011 with Δ_dnm^cb = −0.076 and α_mem ≈ 0.01.

**Cause.** Without a category prior, the LM cannot "win" via shortcut
— but the writer (the M_c→M_c update) receives only a long, weak
gradient through `callback NLL → LM head → depth-router → readout →
M_c → writer`. Near random init, "wrong code" and "uninformative
memory" look similarly bad to the LM loss. The writer has no direct
extractive objective.

**Diagnostic.** Test-time training (TTT) readout audit: freeze the
model and optimise *only* readout-side parameters on callback
tokens. If TTT lifts evidence_lift even after the writer is fixed,
the writer encoded the answer (the readout was the bottleneck). If
TTT does not lift it, the writer never encoded the answer (the
writer is the bottleneck).

**Ideology susceptibility.** Frozen backbone (II) primarily, but
the lesson generalises: any architecture whose memory module is
trained only by long-path gradient through the LM head is at risk.

### §10 Surprising finding — F3 readout probe is harmful (0.5 page)

(This is the *positive negative result* — the empirical surprise of
the campaign.)

**Setup.** The intuitive fix to failure mode 6 is a **readout
probe**: an auxiliary loss that asks the readout to predict a chain-
specific signal (e.g., chain ID) directly from M_c. Call this the
F3 probe.

**Surprise.** Removing the F3 probe **8× the result**. The v24a
recipe (with F3 probe) at 0.6B scores Δ_dnm = +0.16 ± 0.08 nats
(n = 3 seeds); the v27b recipe (same recipe minus F3) scores Δ_dnm =
+1.32 ± 0.53 nats (n = 4 seeds). Same effect at 1.7B: +0.118 nats
with F3 → +0.926 nats without (v25a vs v28). Two single-variable
ablations confirm:

- removing the iterative readout depth (v27a) collapses Δ_dnm to
  +0.025 (load-bearing — keep)
- removing the α-floor on the depth router (v27c) drives Δ_dnm to
  −0.04 (load-bearing — keep)
- removing the F3 readout probe (v27b/v28) **strengthens** Δ_dnm by
  8× (NOT load-bearing — drop)

**Interpretation.** Joint LM-NLL training alone produces a richer
chain-conditional context than supervising the readout for chain
identification. The F3 probe was solving the wrong task — chain ID
identification — while the actual useful objective is "compress
this chain into a context vector that lowers next-token NLL on
*any* future callback in the chain". The probe's gradient was
pulling the readout in a direction orthogonal to (or worse,
contrary to) the useful one.

**Why this counts as a negative result for an ideology.** The F3
probe is the standard fix proposed in the joint-training (I)
literature and partially in the frozen-backbone (II) literature for
failure mode 6. **Empirically, this fix backfires.** That is a
falsifiable, reproducible, ideology-level finding.

### §11 Methodology — six-item audit battery (1 page)

Constructive synthesis. Any recurrent-memory paper, regardless of
ideology, should report:

1. **Callback-aware CE** (failure mode 1).
2. **Evidence-redacted floor** (failure mode 5).
3. **Empirical / base-model prior** (failure modes 2, 3).
4. **Window-leakage count** (failure mode 4).
5. **Redaction validity** (default vs cross vs skip vs zero, ≤ 0.05 nat
   spread; failure mode 5).
6. **Test-time readout audit** (failure mode 6).

We provide all six as `tools/eval_callback.py`,
`tools/audit_a1_window_leakage.py`,
`tools/audit_a2_redaction.py`,
`tools/audit_a3_template_prior.py`,
`tools/audit_a_corpus_prior.py`,
`tools/d5_ttt_readout.py` in the supplementary zip.

Total compute cost per checkpoint: <1 H100-hour. Total compute cost
to run the battery on a published model: ≈ 1 H100-hour for the
diagnostics + the cost of running the original eval. Cheap.

### §12 Conclusion (0.5 page)

The three-ideology framing should be taken seriously: joint
training, frozen backbone, and retrieval are different research
programs and the literature should not treat them as interchangeable
"baselines". Each has its own failure modes, and a paper that
adopts one ideology should report the controls that rule out the
failures the *other* two ideologies are sensitive to. The six-item
audit battery does this for free at all three ideologies.

---

## Numbers ledger

All numbers come from existing run logs in
`results/exp2_chain_recipe/runs.md`, the audit files in
`results/exp2_chain_recipe/audit_*.md`, and the locked numbers in
`NEURIPS_NUMBERS.md`. The campaign covers v11–v28 across two
backbone sizes; specific numbers cited above are:

| failure mode | example number | source |
|---|---|---|
| 1 dilution | v14k Δ_dnm^cb = +1.44 vs Δ_dnm = −0.085 | results/eval_v14v15/, callback-aware vs standard chain-CE |
| 2 joint-invisibility | v15b joint 0.6B CE_mem = 2.97 vs prior 2.91 | results/exp2_chain_recipe/audit_a3_template_prior.md |
| 3 template prior | empirical prior CE 2.9087 nats | tools/audit_a3_template_prior.py output |
| 4 window leakage | 58.5 % chains leak at `window_k=4` | tools/audit_a1_window_leakage.py output |
| 5 redaction uncertainty | spread 0.038 nats on v15a/best | tools/audit_a2_redaction.py output |
| 6 writer-objective | v16a evidence_lift = −0.002 → +0.011, α_mem ≈ 0.01 | results/exp2_chain_recipe/runs.md, v16a entries |
| F3 surprise | v24a +0.16 ± 0.08 → v27b +1.32 ± 0.53 (n=4) | NEURIPS_NUMBERS.md headline + reference tables |
| F3 surprise (1.7B) | v25a +0.118 → v28 +0.926 | NEURIPS_NUMBERS.md |

---

## What still has to land before May 7 PDF deadline

The base manuscript already exists at
`results/exp2_chain_recipe/main_paper.tex` (9 pp, dated May 3 pre-
flip). Three changes:

1. **Add §1.5 / §2 "three ideologies" framing.** The current draft is
   one ideology (frozen backbone) and tells the failure-modes story
   as a Memory-Residuals post-mortem. Re-title and re-frame as a
   field-level observation. ~2 pages of new writing.
2. **Add §10 F3-OFF surprise as the headline negative result.** The
   draft is dated May 3, before the v27b/v28 flip. Add a
   self-contained 0.5-page section + the headline Δ_dnm comparison
   table from `NEURIPS_NUMBERS.md`.
3. **Drop or shrink the F2 NIAH writer warmup section** (current
   §10 of the draft) — the v17 results never landed, the placeholders
   are still `\todo{NUMBER}` markers. Replace with the F3-OFF
   finding: it is the actual empirical resolution of the writer-
   objective failure mode (the LM-NLL signal alone is sufficient
   when the readout supervision is removed). Keep the F2 prescription
   as a one-paragraph "future work" note.
4. **NeurIPS 2026 paper checklist**.
5. **Re-anonymise** — the draft is already `Anonymous Authors`;
   double-check there are no S;G-studio references in audit files
   shipped with the supplementary zip.
6. **Rebuild supplementary zip** — `failure_modes_supplementary.zip`
   already exists at the repo root (sha256 `bd68fa8e604b`); add the
   v27b/v28 eval JSONs and the F3-OFF launcher scripts.

## Estimated write-time

- §1.5 / §2 three-ideology framing (~2 pp new writing): 4 h
- §10 F3-OFF headline negative result (~0.5 pp): 1 h
- F2 section trim/replace: 1 h
- §11 methodology consolidation (existing draft, light polish): 1 h
- Anonymisation + cross-references + checklist: 2 h
- Final pass: 1 h

Total: ~10 h. Achievable May 5–7.

## Submission metadata (OpenReview form fields)

- **Title:** `Three Ideologies of Recurrent Memory in Pretrained LLMs: A Negative-Results Audit`
- **Track:** Main Track
- **Contribution Type:** **`Negative Results`** — exact dropdown text
  per the 2026 Main Track Handbook
- **Primary Area:** `Empirical analysis (of LLMs / foundation models)`
  *or* `Datasets and Benchmarks Track` if the audit framework is
  read more as a benchmarking contribution (the six-item audit
  battery is a methodology + diagnostic tool release)
- **Secondary Area:** `Foundation or Frontier Models` → `Long-Context / Memory`
- **TL;DR:** see top of this file
- **Abstract:** ~250 words; cannibalise from `main_paper.tex` lines 47–69,
  add the three-ideology framing in the lead sentence and the F3
  surprise as the closing sentence
- **Keywords:** recurrent memory, frozen backbone, joint training, RAG,
  shortcut learning, negative results, evaluation methodology,
  template prior, evidence redaction, LongMemEval, callback-aware
  evaluation

## Risk register

| risk | mitigation |
|---|---|
| Reviewer reads "three ideologies" as too rhetorical / not quantitative | The decomposition is rhetorical at the framing level but mapped to *six concrete failure modes with falsifiable diagnostics*. Frame in §1 as "we use 'ideology' to mean a research program with its own characteristic controls"; cite Kuhn 1962 and Geirhos et al. 2020 on shortcut learning to ground the term. |
| Negative-Results contribution type bar is high | The bar is met by (a) the six-mode catalogue is genuinely independent (each ablation only fixes one), (b) the F3-OFF surprise is a 8× quantitative flip from a documented prior recipe, and (c) the methodology is a *constructive* output (the six-item audit battery is software, not just complaints). |
| Reviewer prefers a positive headline | Re-pitch the F3-OFF surprise as the positive headline if it dominates the discussion. The campaign's negative result IS the positive one: the recipe that survives all six audits achieves +1.32 ± 0.53 nats. Flip §10 to be the lead and re-shape §1 around it. |
| Concurrent submission with P1 / Paper A | The contribution is methodological (audit framework + diagnostic battery), not architectural. P1 / Paper A are about *whether memres works*; Paper C is about *how to evaluate any recurrent memory honestly*. Numbers overlap is fine; cite explicitly as concurrent submission in §3. |
| Reviewer demands a single positive headline number from each ideology | We can add a 1-row-per-ideology summary table in §11 — frozen-backbone v27b/v28 +1.32/+0.93 nats (this paper's Memory Residuals running example), joint v15b CE 2.97 ≈ prior (failure example), RAG v28-comparison +0.25 to +0.39 nats (Paper A's RAG cells). All three reported under the same six-item audit battery — the table itself is the methodological contribution. |
| Workshop fallback feels weaker | "I Can't Believe It's Not Better!" workshop is the canonical NeurIPS negative-results venue and runs every year. Even if the main-track flip rejects, this workshop is a strong fallback, with an existing community that values exactly this contribution shape. |
