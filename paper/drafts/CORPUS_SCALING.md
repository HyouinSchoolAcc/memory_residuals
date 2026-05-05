# Corpus scaling for 1.7B → 8B memory-residuals training

Design document. Derived from a careful audit of `synthd5_random_codes`
(the corpus that produced v18-v23's results on Qwen3-0.6B-large) plus
the architectural constraints established in `runs.md` v18-v22 and
`V20_ARCHITECTURE.md`. **Not yet shipped.** This is a plan — pick the
phases you want and we build them.

---

## 1. What we have now (`synthd5_random_codes`, 2026-04 → 2026-05)

```
n_chains          : 5 000
sessions / chain  : 10 (fixed)
evidence sessions : 2 / chain (random body positions)
callback session  : 1 / chain (fixed at last position)
session_len       : 512 tokens (Qwen3 BPE)
total sessions    : 50 000
total tokens      : 25.6 M
file size         : 245 MB train + 25 MB val
backbone          : Qwen3-0.6B-large
```

Real-content density per session, measured by inspection of the first
chain:

```
session  type     real-tokens    padding-tokens   density
─────────────────────────────────────────────────────────
0        filler   ~22            ~490             4.3 %
1        filler   ~22            ~490             4.3 %
2        filler   ~16            ~496             3.1 %
3        EV       ~25            ~487             4.9 %
4        filler   ~21            ~491             4.1 %
5        filler   ~22            ~490             4.3 %
6        filler   ~22            ~490             4.3 %
7        filler   ~16            ~496             3.1 %
8        EV       ~25            ~487             4.9 %
9        CB       ~30            ~482             5.9 %
─────────────────────────────────────────────────────────
chain    avg      ~22            ~490             4.3 %
```

**~95 % of the trained tokens are padding.** Of the remaining ~5 %, the
distractor sessions cycle through **50 unique filler templates**
(`tools/build_synthetic_random_codes.py` lines 155-206), so the *real*
distractor content the model sees is ~50 × 22 = 1.1 K tokens, repeated
many times per chain. The evidence sessions follow **8 templates**
(`tools/build_synthetic_random_codes.py` lines 100-148, one template
per category) with a 5-char random code substituted in. The callback
sessions follow the same 8 templates with the answer template attached.

This was deliberate — the v15-era audit showed that more diverse
content invited template-prior leakage (`audit_a3_template_prior.md`),
so synthd5 went the other way and made the *answer* random while
keeping templates fixed. It worked: v23a/b/c at 0.6B will give us the
read-side capacity verdict against this corpus.

But for 1.7B / 8B, the corpus is starving the larger backbones in
several ways:

### 1.1 The information-density problem

Backbone forward and backward passes process all 512 tokens of each
session, regardless of content. The `MemoryBlock.extract` cross-attn
extracts compressed slot vectors from the entire session. With 95 %
padding:

* `extract` is doing 19× more work per session than the content
  warrants. Backbone activation memory and gradient cost scale with
  the padded length.
* The writer learns to compress padding-heavy sessions; this generalises
  poorly to dense long-context use cases.
* The depth router's per-position `α_mem` is computed at every padded
  position, diluting the per-real-position routing signal.

### 1.2 The template-monoculture problem

All 8 evidence categories share the structural pattern
`User: <fixed prefix> <CODE>. \nAssistant: <fixed acknowledgement>`.
Fillers cycle through 50 fixed scripts. The 1.7B and 8B backbones have
plenty of representational headroom to *memorise the corpus templates
themselves* — at which point the model isn't learning "compress
chain-specific evidence into M_c" but "produce M_c that lets the
readout retrieve from this 50-template distribution." The architecture
delivers the right *value* at training time but the *task* it solves
is much narrower than the position paper claims.

The 0.6B backbone's saturation hides this: 0.6B can't easily memorise
templates *and* learn the memres pathway, so the architecture has no
shortcut and is forced to actually compress evidence. At 1.7B the
shortcut becomes available; at 8B it dominates.

### 1.3 The compute-headroom problem (Chinchilla-adjacent)

Chinchilla optimum is ~20 trained tokens per active parameter. We
freeze the backbone, so the relevant parameter count is the memres
stack:

```
component                      0.6B run     1.7B run     8B run
                              (~30 M memres,  ~30 M memres,  ~30 M memres,
                               + 600 M frozen)  + 1.7 B frozen)  + 8 B frozen)
```

The memres stack is ~30 M params at every backbone size (it scales with
`hidden_size`, which is similar across these checkpoints). So the
Chinchilla-optimal token budget for the *trainable* component is fixed
at ~600 M tokens regardless of backbone size — and we're at 25 M, ~25×
under-trained.

**Why the backbone size matters anyway**: the memres pathway interacts
with backbone hidden states (`extract_source = hidden_14`,
`m_t` flows through every depth-routed sublayer). The *quality* of the
representations the writer extracts from depends on what the backbone
can offer. A 1.7B backbone has finer-grained representations, so the
writer extracts richer M_c — but only if the *training distribution*
has the diversity to surface that richness. With our current 50-filler
× 8-template corpus, the 1.7B writer extracts richer-but-narrower M_c
than the 0.6B writer would; we don't actually exercise the 1.7B's
capacity.

### 1.4 The variance-floor problem (v22 finding)

v22 ran 4 cells at probe weight ∈ {0.3, 0.4, 0.5, 0.6} with otherwise
identical recipes and found end-of-training `evidence_lift` variance of
~0.025 nats — exactly the magnitude of v21c's "winner" reading
(+0.0241). v23 (currently running) is 3 seeds at v21c's recipe to
bound this noise floor. **One reason for the high variance is the
small corpus**: 50 K sessions × 2 epochs = 100 K examples; that's a
small training set for a 30 M-param model, especially when the per-
example signal is concentrated in 5 % of tokens. A larger and more
diverse corpus should reduce this variance and let the architectural
effect actually be measurable.

### 1.5 What's right about the current corpus (must preserve)

Before enumerating fixes, the corpus design choices that should NOT
change:

1. **Random codes for evidence content.** The `audit_a*` work
   (`audit_a1_window_leakage`, `audit_a2_redaction`, `audit_a3_template_prior`)
   established that any non-random evidence (favorite color = red,
   etc.) leaks through the backbone's output prior even with frozen-
   backbone training. Random alphanumeric codes (60 M unique 5-char
   IDs) keep memory the only path.
2. **No assistant echo of the code in the evidence session.** "User:
   my code is RPOIG. Assistant: Got it." not "Assistant: Your code
   RPOIG is saved." Otherwise the assistant turn directly supervises
   the binding inside the evidence session.
3. **Multiple categories per chain.** Forces the readout to
   discriminate which slot to retrieve from at callback time. Not just
   "1 evidence + 1 callback".
4. **Callback at fixed position, not interleaved.** Keeps the eval
   metric (callback CE on a session that contains *only* the callback)
   clean.
5. **Padded to fixed `session_len = 512`.** Trainer's TBPTT and
   `M_c.update_per_session` semantics depend on a fixed session length.

---

## 2. Three corpus phases for 1.7B → 8B

### Phase A — `synthd6_dense` (~7-10 days, 100-300 M tokens)

**Goal.** Get to a Chinchilla-meaningful token budget for the 30 M
memres params on a 1.7B frozen backbone. Eliminate the obvious template
shortcut that 1.7B+ can exploit.

**Single-cell deliverable.** A new corpus
`synthd6_dense_random_codes_train_s512.pt` and corresponding val,
schema-compatible with the existing `chain_*_s512.pt` format so the
trainer needs zero changes. ~200 M tokens of high-density chains.

**Concrete changes vs `synthd5_random_codes`:**

| dimension | synthd5 (current) | synthd6_dense (target) | mechanism |
|---|---|---|---|
| chains | 5 000 | 50 000 (10×) | scale n_chains in builder |
| sessions / chain | fixed 10 | uniform [6, 16] | random per-chain |
| evidence sessions / chain | fixed 2 | uniform [1, 4] | random per-chain |
| callback position | fixed last | uniform in last third | random per-chain |
| session density (real tokens / 512) | ~5 % | ~70 % | multi-turn dialogue per session |
| filler templates | 50 hand-written | 5 000+ LLM-generated | offline batch via Claude/GPT |
| evidence categories | 8 (PIN, locker, etc.) | 24 (existing 8 + 16 new) | extend `CATEGORIES` dict |
| code length | 5 chars (60 M space) | mix of 4-7 (variable) | random per-evidence |
| answer formats | 1 fixed per category | 3-5 per category | template list per category |
| unique chain names | 5 K | 50 K | unique per chain (scales with n_chains) |
| total tokens | 25.6 M | ~250 M (10×) | n_chains × session_len × density |

#### A.1 Multi-turn dialogue density

Replace the current single-exchange filler:

```
User: How was your day?
Assistant: It was pretty quiet, mostly catching up on emails.
```

with multi-turn rich dialogue, target 350-450 real tokens per session:

```
User: How was your day?
Assistant: Oh, mixed. Had three back-to-back meetings in the
morning that ran over, then I tried to power through some
documentation in the afternoon.
User: That sounds rough. What were the meetings about?
Assistant: Mostly project alignment. We've got a deadline on
Friday for the migration plan and there's still some
disagreement about whether to roll it out region-by-region or
all at once. Spent a lot of time on whiteboards.
User: Region-by-region sounds safer, right?
Assistant: That's my preference, but the argument against is
that it doubles the support burden during the rollout window
because you have two configurations live simultaneously...
[continue to ~400 tokens]
```

Generation strategy: prompt Claude/GPT with **chain-context-aware
multi-turn dialogue generation** at scale. ~5 K USD on Anthropic /
OpenAI APIs would generate ~50 K dense filler dialogues at 400 tokens
each = 20 M filler tokens; cycle 10×= 200 M chain tokens. (Cost
budget detail in §4.)

Multi-turn fillers have a useful side effect: they give the depth
router *something to learn about routing* on non-evidence sessions.
With templated 22-token fillers, the router has trivially nothing to
attend to in 95 % of positions. With dense fillers, the router has a
real distribution over which sublayer benefits from `m_t` injection at
each real-content position — which is what the production case
(serving the 1.7B with memres on a real conversation) will look like.

#### A.2 Evidence-category expansion

Current 8 categories: `locker_code`, `pin`, `employee_id`, `apartment`,
`flight`, `confirmation`, `tracking`, `voucher`. All are
"alphanumeric-code-shaped" and follow the same surface form. Extend to
24 categories spanning:

* **8 alphanumeric-code categories (existing).**
* **8 number-with-units categories**: dosage ("I take 12.5 mg of
  metoprolol"), price ("paid $1,743 for the laptop"), date ("my
  appointment is on November 7"), duration ("the lease is 18
  months"), distance ("warehouse is 4.2 km away"), capacity ("battery
  is rated for 3 200 mAh"), weight ("package weighed 2.1 kg"),
  count ("ordered 17 packs").
* **8 named-entity categories**: person ("my dentist's name is
  $RANDOM_NAME"), place ("conference is in $RANDOM_CITY"), product
  ("I bought the $RANDOM_PRODUCT_NAME model"), org ("hired by
  $RANDOM_ORG_NAME"), street ("apartment is on $RANDOM_STREET"),
  car ("driving a 2019 $RANDOM_CAR"), book ("just finished
  $RANDOM_BOOK_TITLE"), team ("draft pick is from
  $RANDOM_TEAM_NAME").

For named-entity categories, **synthesize the entities** rather than
sample from real-world dictionaries (which would give the backbone
priors). Use Markov-chain or LLM-generated novel-but-plausible
identifiers; verify with the audit_a3 template-prior tool that 1.7B's
output prior on the synthesised entity is uniform over the candidate
set.

#### A.3 Callback-format diversity

Current: `Quick question, what was my new PIN? \nAssistant: Your PIN
is RPOIG.`

Extend to 3-5 surface forms per category:

```
"User: Hey, can you remind me what my PIN was again?
 Assistant: Sure, your PIN is RPOIG."

"User: I forget — what's the PIN I set?
 Assistant: It's RPOIG."

"User: What's that PIN I set up the other day?
 Assistant: That was RPOIG."

"User: Quick check, my PIN is...
 Assistant: RPOIG."
```

This forces the readout to handle multiple callback templates, which
is closer to "real" inference and prevents the readout from learning
just one prefix-completion pattern.

#### A.4 Variable chain structure

```python
chain_length     = rng.randint(6, 16)        # was: fixed 10
n_evidence       = rng.randint(1, min(4, chain_length - 2))  # was: fixed 2
callback_pos     = rng.randint(int(chain_length * 0.66), chain_length - 1)
                                              # was: fixed chain_length - 1
evidence_positions = rng.sample(range(callback_pos), n_evidence)  # body, before callback
```

Why this matters: the v22 evidence_lift variance is partly because the
*structure* is fixed and any seed-related fluctuation in writer
training produces a wide spread on the *single* structural
configuration. Variable structure averages over many configurations,
gives a more stable signal, and lets the architecture demonstrate
graceful degradation as chains get longer / evidence gets more
distant.

#### A.5 Estimated effort and cost

| stage | wall-clock | effort | cost |
|---|---|---|---|
| extend `tools/build_synthetic_random_codes.py` to v6 schema | 1-2 days dev | code | 0 |
| LLM-generate 50 K dense filler dialogues (Claude Opus or GPT-5) | 8-12 h API time | 1 day dev | ~$200-400 |
| LLM-generate 5 templates × 24 categories of callback Q&A surface forms | 2 h | half day | ~$5 |
| run synthd6 v1: 50 K chains, ~250 M tokens, on local box | 6-12 h CPU | tokenisation | 0 |
| audit_a* re-run on synthd6 (template prior, redaction, window leakage) | 2 h GH200 | half day | ~$50 |
| **total** | **~5-7 calendar days** | | **~$300-500** |

Output: `paper_artifacts/chains/synthd6_dense_random_codes_train_s512.pt`
and `..._val_s512.pt`. Schema unchanged so the trainer takes
`--train_chains synthd6_*` without modification.

#### A.6 Phase A verdict criteria

Train v24 (v21c recipe) on synthd6 at:
* 0.6B for 1500 steps (sanity that the architecture still converges
  on the bigger corpus).
* 1.7B-large frozen for 3000 steps.

Decisive cells:
* `evidence_lift_final >= +0.05` end-to-end on 1.7B (vs synthd5's
  +0.024 at 0.6B): synthd6 unlocked the 1.7B's expressive headroom.
* `evidence_lift` variance across 3 seeds < 0.01 nats: the corpus is
  large enough to reduce noise floor.
* §5 `ttt_lift_vs_floor > +0.10` on writer init: read-side capacity
  preserved when the 1.7B has more representational room.

If any of these miss, Phase B mitigations apply.

---

### Phase B — `synthd7_mixed` (real-corpus augmentation, 2-3 weeks, 500 M-1 B tokens)

**Goal.** Cover the gap from "synthetic chains the architecture solves"
to "real long-context use cases the architecture is supposed to
handle." Phase A keeps the synthetic shape that controls leakage but
generalises poorly; Phase B adds real-corpus chains so the architecture
sees what production looks like.

**Sources to mix:**

1. **LongMemEval (LME).** Already pre-tokenised at
   `paper_artifacts/chains/lme_train_s512.pt`. ~56 MB train, ~6 MB val.
   Real human-LLM conversation logs, multi-session, with ground-truth
   "what did the user say earlier" callbacks. Use as-is, weight ~3-4×
   in the chain trainer's source-weighted dataloader (`pg19=1.0,
   tv=4.0, ...`).

2. **Multi-Session Chat (MSC).** `msc_test_s512.pt` 5 MB val present.
   Get the train split from FAIR (it's open) and tokenise. Real
   persona-driven multi-session conversation. The "persona" sentences
   in MSC are exactly the "evidence" we need for the memres claim
   ("I'm vegetarian and I have a golden retriever named Biscuit").

3. **PG-19 long fiction.** 28 K novels, average length ~70 K tokens.
   Construct chains by splitting a novel into 10-30 sessions of 512
   tokens, then synthesise callback Q&A from earlier-chapter
   facts. This is the "compress book chapters → generate next
   chapter" use case that v3 / v9c demonstrated worked at modest
   scale. Construction tool exists at
   `tools/build_synthetic_dialogue_chains.py` and produced positive
   results in v9c (Δ_nm-m grew −0.03 → +0.16 nats over 4 K steps on
   PG-19+TV+LME+MSC).

4. **LongMemEval-style augmentation for v6 categories.** For each of
   the 24 evidence categories from Phase A, prompt a strong LLM to
   produce 2-5 K real-style multi-session dialogues *containing the
   evidence-statement-then-callback pattern* but with naturally-flowing
   conversational context. Replaces some fraction of `synthd6_dense`
   distractor density with realistic semi-task-oriented dialogues.

**Mixing recipe:**

```
weight    source                                     ~tokens
─────────────────────────────────────────────────────────
1.0       synthd6_dense (Phase A synthetic)          250 M
0.5       LongMemEval                                 70 M
0.5       MSC train                                  ~150 M (after pull)
0.3       PG-19 chains (synthetic callbacks on        ~200 M
             real long-form text)
0.4       LME-style augmented chains                 ~200 M
─────────────────────────────────────────────────────────
total                                                 ~870 M
```

Schema-compatible mixing already exists at
`tools/merge_chain_corpora.py`; extend to accept the source-weight
flags the trainer's `dataloader.source_weights` consumes.

**Phase B verdict criteria:**

Train v25 (v21c recipe) at:
* 1.7B-large frozen, 5 000 steps, 4 seeds.
* Held-out evals on `lme_val`, `msc_test`, `realtalk_eval`, and
  synthd6_val to test cross-source generalisation.

Decisive:
* `evidence_lift > +0.05` on at least 2 of {LME, MSC, realtalk}.
  (The synthetic eval is sanity; the real-eval transfer is the claim.)
* §5 readings remain MIXED+ on the real-eval splits.

---

### Phase C — `synthd8_atscale` (8B path, ~6-8 weeks, 3-5 B tokens)

**Goal.** Supply enough data to train 30 M memres params on top of an
8B backbone such that the architectural claim is not artificially
limited by token budget. 30 M params × 100 tokens/param (5× Chinchilla
optimum, justified by the strongly-supervised probe loss + LM-NLL dual
objective and the relatively narrow task) = 3 B tokens.

**Sources** (this is largely an LLM-augmentation play; the synthetic
ceiling is roughly Phase B's ~1 B):

1. **Phase A + B base** (~1 B tokens).

2. **LLM-augmented LongMemEval at scale.** Use the existing 56 MB LME
   train as seed; for each chain, generate 5-10 *paraphrased and
   topically-shifted* variants with a strong LLM. Targets ~500 M
   tokens of LME-style conversation.

3. **Wikipedia-derived chains.** Take Wikipedia article sequences
   (linked-article chains: an article and 5-10 articles it links to)
   and construct a synthetic conversation around them, with the
   callback being a fact from an earlier article. Anti-leakage harder
   here (the backbone has Wikipedia priors), so use **only post-cutoff
   Wikipedia** (articles edited after the backbone's training cutoff)
   or **modify entity names** to break the prior. Targets ~1 B tokens.

4. **Books3 / arxiv long-form chains.** Same construction as PG-19 but
   on technical / academic long-form content. Stresses the architecture
   on a different distribution than novels (denser propositional
   content, more numerical / named-entity facts). Targets ~500 M
   tokens.

5. **Programmatic structured-data chains.** Generate synthetic
   "session N records measurement X, session N+5 asks about earlier
   measurement" chains from random data tables (lab logs,
   transactions, time-series). High signal/noise ratio for the
   memres claim because the answer is structurally clean. Targets
   ~500 M tokens.

**Estimated cost.** LLM augmentation at 3 B tokens at $0.50 / M output
tokens (Claude / GPT current rates with batch APIs) = **$1 500-2 500 in
API spend**. Plus tokenisation / disk: ~30 GB on disk after compression.

**Phase C verdict criteria:** memres on 8B-frozen-backbone delivers
positive end-to-end `evidence_lift` (any size) on real-corpus held-out
evals at multiple seeds. If yes, the architectural claim transfers to
production scale; if no, the architecture is fundamentally rate-limited
at the writer / readout / corpus interaction we haven't yet diagnosed.

---

## 3. Cross-cutting code work (do once, benefits all phases)

### 3.1 Streaming corpus loader

Current loader at `tools/pretokenize_chains.py` and `eval_callback.py`
loads the entire corpus into memory at trainer start. For Phase B+
(1 B+ tokens, ~1.5-3 GB tokenised), this is OK on H100s but tight on
GH200 alone with a 1.7B backbone in bf16. Phase C definitely needs
streaming.

* Add a chunked-on-disk format (mmap or HF datasets cache).
* Keep `--train_chains` semantics; just read sessions on demand.
* ~1 day of dev.

### 3.2 Source-weighted dataloader (already exists, needs extension)

`train_chain.py`'s `source_weights` dict (lines ~1525) supports
weighting by `corpus_kind`. Extend to:
* Read source weights from a sidecar JSON (one mixing recipe per
  experiment).
* Log per-source loss / per-source `evidence_lift` separately so we
  can see which mixture component is hurting / helping.

### 3.3 `audit_a*` extensions to mixed-source corpora

`tools/audit_a1_window_leakage.py`, `audit_a2_redaction.py`,
`audit_a3_template_prior.py` already exist for synthd4v2/synthd5. Need
extensions:
* `audit_a3` on real-text corpora (LME, MSC) requires a different
  baseline (CE under chain-redacted vs full-context). Roughly half a
  day of work per audit.
* New audit `audit_a4`: **per-source contamination check.** For each
  chain in the mixed corpus, verify the answer-bearing tokens are not
  predictable from chain-prefix-only by the bare backbone. This is
  audit_a3 generalised to non-synthetic data.

### 3.4 Config-as-data: `synthd*` recipe schema

Currently each corpus is "whatever the build script bakes in". Move
to a recipe-file approach so synthd6 / synthd7 / synthd8 are
parameterised by a recipe JSON:

```jsonc
{
  "corpus_kind": "synthd6_dense",
  "corpus_version": "v1",
  "n_chains": 50000,
  "session_len": 512,
  "chain_length_dist": "uniform[6,16]",
  "n_evidence_dist": "uniform[1,4]",
  "callback_position_dist": "uniform[0.66, 1.0)",
  "categories": ["alphanum_code:8", "number_unit:8", "named_entity:8"],
  "filler_source": "llm_generated_v1",
  "filler_density_target": 0.7,
  "callback_format_count_per_category": 4,
  "augmentation_sources": []
}
```

Build script reads recipe, produces tensorized output. ~1-2 days dev,
saves much pain across phases.

---

## 4. Cost / wall-clock summary

| phase | calendar | dev work | API cost | training compute | training cost (lab + cloud) |
|---|---|---|---|---|---|
| A: `synthd6_dense` | 1 week | ~3-4 days | ~$300-500 | 0.6B + 1.7B × 4 seeds | already paid (own H100s + GH200 hours) |
| B: `synthd7_mixed` | 2-3 weeks | ~4-5 days | ~$1 K-2 K | 1.7B × 8 seeds | ~$500 GH200 hours |
| C: `synthd8_atscale` | 6-8 weeks | ~10-15 days | ~$2-3 K | 8B × 4 seeds | ~$3-5 K cloud (4-8 H100s × ~50 hours) |
| **end-to-end** | **~10 weeks** | **~20 days** | **~$3-5 K** | | **~$3-5 K** |

For an 8B end-to-end with the canonical recipe and a meaningful 4-seed
reproducibility check on 1 B+ tokens, total project cost is in the
**$6-10 K** range. Each phase produces a publishable result; if Phase
A alone delivers a clean `evidence_lift > +0.05` on 1.7B, that's a
defensible "memory residuals work at intermediate scale" paper.

---

## 5. Recommended next concrete step (after v23 lands)

If v23 confirms v21c is reproducible at 0.6B (3 seeds with
`evidence_lift` 95 % CI excluding 0):

1. **Ship Phase A.1** — extend the random-codes builder to support
   variable chain structure + 24 categories (~2 days of code; no LLM
   spend yet). Run v24 on the variable-structure version of synthd5
   first (still 50-filler templates) to see if the variance drops
   purely from the structural change.
2. **Ship Phase A.2-A.4** — the LLM-generated dense filler corpus +
   callback-format diversity. ~$300 of API.
3. **v25 wave**: 1.7B + synthd6_dense + v21c recipe + 3 seeds. ~6 h
   total wall-clock. Verdict: does the architecture deliver
   `evidence_lift > +0.05` on a model that has the representational
   headroom to use it?

If v23 *doesn't* confirm v21c (say variance > effect size at 0.6B):
the architecture itself needs more work (writer-side intervention?
deeper extractor? different read-side primitive?), and ramping the
corpus to 1.7B is premature. The v22 signal already suggests this is
the more likely outcome — be ready to pivot to a writer-side v24
architectural change instead of a corpus phase A.

---

## Appendix: schema invariants the builder must preserve

```python
{
    "session_ids":              torch.LongTensor,    # (n_sessions, S)
    "session_callback_mask":    torch.Int8Tensor,    # (n_sessions, S)
    "session_evidence_mask":    torch.Int8Tensor,    # (n_sessions, S)
    "session_role":             torch.Int8Tensor,    # (n_sessions,)
                                                     # 0=filler, 1=evidence, 2=callback
    "chain_starts":             torch.LongTensor,    # (n_chains,)
    "chain_lengths":            torch.LongTensor,    # (n_chains,)
    "chain_callback_position":  torch.LongTensor,    # (n_chains,)
                                                     # session index (0-based) within chain
    "chain_evidence_positions": list[list[int]],     # per-chain ev positions
    "chain_names":              list[str],
    "session_len":              int,
    "tokenizer":                str,
    "corpus_kind":              str,  # e.g. "synthd6_dense"
    "corpus_version":           str,  # e.g. "v1"
}
```

Any new corpus must produce exactly these fields with these dtypes, or
the trainer's TBPTT / phase-aligned-eval / `eval_callback` paths break.
Verified by `python -c "import torch; d = torch.load('...'); ..."`
(see §1 above for the working invocation).
