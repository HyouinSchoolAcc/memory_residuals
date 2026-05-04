# v17 wildcards — proposal packet (WILDCARD-V17, 2026-05-03 ~05:00 UTC-5)

> **Read-only deliverable.** No training launched, no `.sh` files saved.
> Launch scripts are sketched inline; the parent agent will harden them.
>
> **Scope.** The v14k smoking gun (95% of `pa_cb_dnm` survives evidence
> redaction → content-blind output prior, not retrieval) and v16a's
> non-cure (`evidence_lift = +0.011`, `pa_cb_dnm = -0.076`,
> `nce_loss = 1.375 ≈ log 4`, D3-MC pair/self = 0.012,
> `α_mem_mean = 0.01`) jointly say the same thing in two different ways:
> **the writer has no effective gradient signal to encode chain-specific
> content into `M_c`, and the readout/router does not provide one
> either.** D5+evidence-mask+evidence_lift-best surfaced the failure
> honestly but did not cure it. v17 should *cure* it.

---

## Diagnosis-frame (what every proposal below targets)

The chain trainer currently has three pathways that could shape `M_c`:

1. **LM-head NLL on the callback span**, propagated backward through
   the depth-router → readout → `M_c` → judge → extract → `C_t`.
   v14k showed this is content-blind (writer learns a per-callback
   marginal it can inject regardless of `M_c`). v16a's
   `mask_evidence_session_loss` was supposed to remove the marginal
   shortcut, and arithmetically it does (D5's marginal CE ≈ 10 nats
   makes the prior costly to learn) — but *the new gradient that
   should appear in its place is too weak to break symmetry on K=128
   anonymous slots*. The writer/readout sit at their lottery init.
2. **InfoNCE callback-only (B² cross product, B=4)**.
   `nce_loss = 1.375 ≈ log 4 = 1.386` — random. The B² grid evaluates
   per-pair callback NLL through the *same content-blind LM head*, so
   the off-diagonal "negative" pairs have nearly identical NLL to the
   diagonal "positive" ones. InfoNCE's chain-discriminability gradient
   is currently routed through the same bottleneck that v14k showed
   is the content-blind shortcut.
3. **`alpha_mem_floor` aux loss (target 0.05, weight 0.01)**. Forces
   `α_mem ≥ 0.05` on average. Achieves `frac_open = 0.0` ⇒ the floor
   is being satisfied by spreading `α_mem` thinly rather than opening
   real channel mass. Diagnostic of the symmetric fixed point, not
   pressure against it.

Every (1)-(2)-(3) pathway is *funneled through the same content-blind
LM-head shortcut* (writer → judge → readout → router → LM head). What
is missing is a gradient channel that **directly couples `M_c` content
to a chain-specific target** without going through that shortcut. The
five proposals below each propose a different way to install that
channel; they are not mutually exclusive (A+D in particular compose
cleanly).

---

## Empirical-support map (transparency)

| proposal | category | empirical anchor | speculative |
|---|---|---|---|
| **§1 NIAH writer pretraining** (A) | warmup objective | v14k 95% redaction-survival; v16a `nce ≈ log 4` | low — direct attack on a measured failure |
| **§2 DeltaNet writer** (B) | architecture | uniform-softmax fixed point (Architectural Prior #5); D3-MC pair/self = 0.012 | medium — principled but unproven on this task |
| **§3 Procedural-rule chains** (C) | corpus | v16a writer-storage failure; manuscript §6 LRMT-style horizon overfit | medium — new corpus, untested |
| **§4 Probe-head bypass loss** (D) | loss | v14k content-blind LM head; v16a NCE through same head | low — direct attack on a measured failure |
| **§5 TTT-on-`M_c`** (E) | wildcard | LM-only chain training collapses `α_mem`; writer-vs-readout bottleneck unidentified | medium — pure capacity diagnostic |

Plain reading: **§1 and §4 are direct interventions on the v14k+v16a
empirical signal.** §2, §3, §5 are principled but speculative. §5 is
the cheapest of the speculative three (pure eval-time, no training).

---

## §1 — NIAH-as-warmup: pre-train the writer with direct M_c→answer supervision

**Hypothesis it tests.** The writer never receives a chain-specific
gradient signal under joint LM training because every gradient channel
flows through the content-blind LM-head shortcut. Pre-training the
writer/judge with a *direct* supervision signal — "given a chain whose
evidence session contains needle `X`, the readout from `M_c` must
attend to `X`" — populates `M_c` with chain-specific content **before**
the LM head ever sees it, and the joint phase then has a working
writer to fine-tune rather than an i.i.d.-init writer to coax out of
the symmetric fixed point.

**Why it's not just another ablation.** v11–v16 are all *joint*
optimisations. The writer's only gradient signal across all 16
versions has been LM-NLL backpropagated through a readout that v14k
proved is content-blind. A direct M_c → answer probe supervision
rewires the gradient graph: `loss_probe → W_V_probe → M_c → judge →
extract → C_t`, with **zero LM-head involvement during warmup**. None
of v11–v16 has this. It is the missing column in the design matrix.

**Concrete recipe (`v17a_niah_writer_warmup`).**

1. **Corpus.** Use the existing
   `synthd5_random_codes_train_s512.pt` (no rebuild). All metadata
   already there: `session_role` (0=filler, 1=evidence, 2=callback),
   `session_evidence_mask` (per-token mask on the ID span),
   `session_callback_mask` (per-token mask on the answer span).
2. **Probe head (new module, ~120 LOC in `modeling_memres.py`).**
   ```
   class WriterProbeHead(nn.Module):
       def __init__(self, d, n_categories=8):
           self.W_Q = nn.Linear(d, d, bias=False)
           self.W_K = nn.Linear(d, d, bias=False)
           self.W_V = nn.Linear(d, d, bias=False)
           self.cat_query = nn.Parameter(torch.randn(n_categories, d) * d**-0.5)
           self.proj = nn.Linear(d, vocab_size, bias=False)  # tied to embed_tokens
       def forward(self, M_c, category_id):
           q = self.W_Q(self.cat_query[category_id])    # (B, d)
           K = self.W_K(M_c); V = self.W_V(M_c)
           a = F.softmax(q.unsqueeze(1) @ K.transpose(-2,-1) * d**-0.5, dim=-1)
           v = (a @ V).squeeze(1)                        # (B, d)
           return self.proj(v)                           # (B, V)
   ```
   The probe is **structurally distinct** from the readout: a single
   per-category query (not per-position), a separate Q/K/V (separate
   parameters from `MemoryReadout`), and projects through `proj` (we
   recommend tying to `embed_tokens.weight` to avoid blowing up
   parameter count).
3. **Warmup loss.** For each batch, after the writer compresses the
   evidence session (single judge step from `M_c=None`),
   ```
   logits = probe(M_c, category_id_of_evidence)          # (B, V)
   loss_probe = F.cross_entropy(logits, answer_first_token_id)
   ```
   The writer/judge/extract get gradient through the probe; the LM
   head and the depth-router are **frozen** for the entirety of
   warmup. (Backbone already frozen.)
4. **Auxiliary anti-collapse.** Add the existing
   `--alpha_mem_floor_aux_weight 0.0` (i.e. *off* during warmup; we
   are not yet asking the readout to do anything).
5. **Schedule.** 800 steps of probe-only warmup at lr=1e-4, then
   800 steps of probe + LM (probe weight 0.1, LM weight 1.0,
   evidence-mask still on), then 1500 steps of LM-only with
   `evidence_lift` best-saving (this last phase is the existing v16a
   recipe). Total 3100 steps.
6. **Anti-circularity.** The probe head is *thrown away at eval time*
   — `tools/eval_callback.py` only knows about the LM-head readout.
   So a positive eval result at the end of warmup → joint phase
   measures the LM-head/readout's ability to *use* a writer-encoded
   `M_c`, not the probe head's ability to *circumvent* it. This is
   the entire point.

**Launch sketch (single-page; do not save).**
```bash
# v17a_niah_writer_warmup_0p6b_frozen_gh200.sh
# Warmup-phase 1 launches first; phases 2/3 chained via best/.
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity --router_recent_bias_init 4 \
    --memres_writer_kind slot_attention --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal --memres_slot_positional \
    --memres_extract_input_norm --memres_extract_source hidden_14 \
    --freeze_backbone \
    --writer_probe_warmup_steps 800 \           # NEW
    --writer_probe_loss_weight 1.0 \            # NEW
    --writer_probe_freeze_lm_head \             # NEW (route gradient to writer only)
    --writer_probe_freeze_router \              # NEW
    --writer_probe_freeze_readout \             # NEW
    --writer_probe_n_categories 8 \             # NEW
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --steps 3100 --warmup 200 \
    --mask_evidence_session_loss \
    --save_best_metric evidence_lift \
    --eval_every 100 --eval_n_chains 24 --phase_aligned_eval_n_chains 32 \
    --diagnose_memory_dynamics \
    --kill_on_memory_collapse --kill_on_memory_collapse_min_step 200 \
    --run_name chain_v17a_niah_warmup_0p6b_frozen_gh200 \
    --out_dir output/chain_v17a_niah_warmup_0p6b_frozen_gh200
```

**Predicted result if right.** End of warmup phase 1 (step 800):
probe-CE on the answer span ≤ 4 nats (vs ~10-nat marginal). D3-MC
pair/self ≥ 0.5 (writer storing chain-specific content). End of phase
3 (step 3100): `evidence_lift ≥ +0.20`, `pa_cb_dnm ≥ +0.5` with a
`pa_cb_dnm_floor / pa_cb_dnm ≤ 0.4` (≥ 60% of the benefit is
chain-specific, not the v14k content-blind 95%).

**Predicted result if wrong.** Probe-CE drops to ~4 nats (the writer
*can* learn the binding under direct supervision), but joint phase
3's `evidence_lift` stays ≤ +0.05 anyway. **Then the bottleneck is
the readout/router, not the writer**, and v17b–e ranking shifts to
prefer §4 (probe-head bypass *at inference*, not just warmup) and
§2 (architecture). Alternative failure: probe-CE plateaus at log(V)
within phase 1 ⇒ `M_c` cannot store this binding *even with direct
supervision* under the slot_attention writer ⇒ the architecture is
broken at the writer level (motivates §2).

**Cost estimate.** 3100 steps × ~7s/step (0.6B frozen, GH200,
probe-head adds ~5% per step) ≈ **6 GH200·h**. Phase 1 alone is
~1.5h; the falsification test is cheap.

---

## §2 — Architecture pivot: replace slot-attention writer with a Linear-Attention (DeltaNet/Mamba-style) writer

**Hypothesis it tests.** The slot-attention writer's symmetric-softmax
+ shared-GRU update has the structural property that its fixed points
are permutation-invariant (Architectural Prior #5: this is *the* fixed
point that v11–v16 keep falling into). v13's `slot_positional` +
`memres_queries_init=orthogonal` *break* the symmetry at init but the
training dynamics (joint LM-NLL gradient, B=4 batch) cannot maintain
it — every gradient step nudges back toward the symmetric attractor,
and v14k's content-blind prior is exactly the symmetric-fixed-point
behaviour at the readout level. A **cumulative-outer-product writer
has no symmetric fixed point**: each new (k, v) pair is *added* to a
state matrix `S ∈ R^{d×d}` rather than *allocated* to one of K
anonymous slots. There is nothing for slots to "agree on".

**Why it's not just another ablation.** v11–v16 explore three writer
kinds (`original`, `slot_attention`, `slot_attention_full`); all three
share the same softmax-over-pool structure with K queries. The
qualitative claim is that *no instance of slot-allocation writer can
escape the symmetric attractor under joint LM training* — Prior #5
argues this is structural. A linear-attention writer is a different
structural class (no slots, no softmax, recursive outer-product
update). It is the smallest *architectural* category-jump that
preserves the rest of the pipeline.

**Choice rationale (single pick).** Three candidates:

- **8 named-category slots** (one slot per fact category, hard
  routing). Strong inductive bias but corpus-coupled (only works on
  D4/D5; tied to the fact-lookup taxonomy). Rejected: violates the
  project's vision of a *general* chain memory.
- **Single fat slot (K=1, larger d)**. Eliminates symmetry trivially
  but reduces capacity; the readout is then a single-token attention
  (not very meaningful). Rejected: doesn't generalise to long
  many-fact chains.
- **DeltaNet-style cumulative outer product writer**. Update is
  `S_t = β · S_{t-1} + (1-β) · k_t v_t^T` with `β ∈ (0,1)` learnable
  per dimension; readout is `m_t = q_x · S_t` for query `q_x = W_Q x`.
  No slot symmetry, content-conditioned write (k, v are functions of
  C_t), differentiable forgetting. Read mechanism reuses
  `MemoryReadout` shape-wise (`m_t ∈ R^{S×d}`), so the depth-router
  doesn't change. **This is the pick.**

**Concrete recipe (`v17b_deltanet_writer_0p6b`).**

1. **Code change (~250 LOC in `modeling_memres.py`).** New
   `DeltaNetWriter(nn.Module)` that replaces the slot writer in
   `MemoryBlock.judge`. Internally:
   ```
   class DeltaNetWriter(nn.Module):
       def __init__(self, d, K_compat=128):
           # K_compat: shape-compat with the rest of the codebase
           # which expects M_c of shape (B, K, d). We pack the
           # state matrix S ∈ R^{d×d} into K rows by storing
           # S[:K, :] (rank-K projection); see compat note.
           self.beta_gate = nn.Linear(d, 1, bias=True)
           self.W_K = nn.Linear(d, d, bias=False)
           self.W_V = nn.Linear(d, d, bias=False)
           self.K_compat = K_compat
       def forward(self, M_c_prev, M_new):
           # M_new is (B, K, d) — treat as a *sequence* of (k,v)
           # writes. Recover S from M_c_prev via fixed projection.
           S = unpack_state(M_c_prev)             # (B, d, d)
           for i in range(M_new.shape[1]):
               h = M_new[:, i]                    # (B, d)
               k = self.W_K(h); v = self.W_V(h); β = sigmoid(self.beta_gate(h))
               S = β.unsqueeze(-1) * S + (1-β).unsqueeze(-1) * (k.unsqueeze(-1) @ v.unsqueeze(-2))
           return pack_state(S)                    # back to (B, K_compat, d)
   ```
2. **Stage 1 (extract) unchanged.** The Perceiver-style cross-attn
   stack still produces `M_new ∈ R^{K×d}`. The DeltaNet writer
   replaces only the Stage-2 *judge*: instead of a slot-axis softmax
   over `[M_c_prev || M_new]`, we treat `M_new`'s K rows as K
   sequential writes to the state matrix.
3. **State packing.** `S ∈ R^{d×d}` is too large to checkpoint per
   chain (d=1024 → 4MB/chain × 5000 chains = 20GB). Instead we keep
   `M_c` in its existing `(B, K, d)` shape and recover S on-the-fly:
   `S = M_c.transpose(-1,-2) @ M_c / K` (a low-rank approx). The
   readout becomes `m_t = X · M_c^T · M_c · W_V_read` which is
   exactly the existing `MemoryReadout` math up to a scalar.
4. **Loss.** Identical to v16a: `mask_evidence_session_loss`,
   `--save_best_metric evidence_lift`,
   `--alpha_mem_floor_aux_weight 0.01`,
   `--contrastive_infonce_weight 0.5` (NCE still in, see §4 for why
   it might still not bite — but no reason to remove it here).
5. **Step budget.** 2500 steps, frozen 0.6B backbone. Identical to
   v16a otherwise (so the DeltaNet vs slot writer is the only change
   in the design matrix).

**Predicted result if right.** D3-MC pair/self ≥ 0.3 by step 500
(writer no longer collapses to identical state across chains).
`evidence_lift > +0.05` by step 1500. Crucially: the
`pa_cb_dnm_floor / pa_cb_dnm` ratio is ≤ 0.5 (the readout is now
*genuinely* using chain content; v14k's 95% redaction-survival was a
*slot-allocation-symmetry* artefact, not a *readout* artefact).

**Predicted result if wrong.** Writer trains (D3-MC pair/self lifts
off zero, ‖m^t‖/‖embed‖ stays in [1, 5]), but `evidence_lift` still
stalls at ~0. **Then v14k's content-blind output prior is a
*readout-side* pathology** (the readout learns to inject a fixed bias
regardless of what it reads), not a writer-side one. This pivots the
project to §4 (probe-head bypass loss) as the next intervention,
because §1 and §2 both miss the readout layer.

**Cost estimate.** 2500 steps × ~6.5s/step (0.6B frozen, GH200) ≈
**4.5 GH200·h**. The DeltaNet step is *cheaper* than slot-attention
(no GRU, no per-iter softmax) but the implementation/debug cost
dominates: budget +1 dev day.

---

## §3 — Corpus pivot: procedural-rule chains (behavioural memory, not lookup)

**Hypothesis it tests.** D5 random codes is a *fact-lookup* task: the
writer must store a 5-token alphanumeric string and the readout must
emit those exact 5 tokens. The CE landscape on the answer span is
*peaked* (only one correct continuation per chain) so the gradient
the LM head provides on a slightly-wrong `M_c` is nearly identical
to the gradient on a totally-wrong `M_c` — the loss is locally flat
around random init, which is exactly why the writer never gets a
useful signal even with `mask_evidence_session_loss`. A *behavioural*
chain (the chain establishes a transformation rule that the model
must continue applying) gives the LM head a **smooth** CE surface:
even a partial encoding of the rule shifts probability mass on every
callback token, not just on a single 3-BPE-token answer. The writer
gets gradient before it has memorised the binding exactly, escaping
the cold-start trap.

**Why it's not just another ablation.** D1, D2, D3, D4, D4v2, D5 are
all *fact-lookup* corpora (different ID alphabets, different masking,
but the same "evidence session contains a binding, callback session
asks for it" structure). A procedural-rule corpus is a different
*task class* — semantic memory of a behaviour, not episodic memory of
a binding. The project's vision (memory_residuals.pdf §1) explicitly
admits both: "compressed semantic memory" and "episodic recall". v3's
PG-19 result was procedural in this sense (style continuity). v9c
saw mono-tonic growth on dialogue precisely because dialogue has more
procedural structure than fact lookup. We have *positive evidence*
the architecture can learn procedural patterns; we have *no positive
evidence* it can learn fact-lookup. Build the corpus that plays to
the architecture's known strength.

**Concrete recipe (`v17c_procedural_rules`).**

1. **Corpus (`synthd6_procedural_rules`, ~200 LOC builder).** Each
   chain has 10 sessions:
   - 1 `rule-establishment` session: a user message describes a
     transformation in natural language ("From now on, when I ask
     for a number greater than 100, give me the number plus 7." or
     "When I say 'Q:', respond in pig latin.").
   - 4 `filler` sessions: unrelated chitchat.
   - 4 `rule-application` (training-callback) sessions: user asks a
     question that triggers the rule; assistant applies the rule.
     The first 3 are *visible* to the LM during compression (added
     to `M_c`); the last is the *evaluation* callback.
   - 1 `null` session: a question that does *not* trigger the rule;
     assistant ignores the rule. Tests precision (rule isn't a
     style cue applied indiscriminately).
2. **Rule families (10 categories).**
   - Arithmetic: "+k", "×k", "mod k", "Fibonacci continuation".
   - String: "shift letters by k", "reverse word order", "every
     other character", "always end with codeword X".
   - Format: "respond in monosyllables", "respond in haiku",
     "translate to pig latin".
   - Persona: "respond as if you are a 19th-century explorer".
   Rule parameters (k, codeword X) are random per-chain; **no two
   chains share the same (rule, parameter) tuple**.
3. **Loss.** `mask_evidence_session_loss` zeroes the LM loss on the
   rule-establishment session (we don't want the LM head learning
   the rule template). LM loss is taken on rule-application + null
   sessions. Crucially: the LM-NLL on a rule-application session is
   *low everywhere the rule is being applied* (every token of the
   transformed assistant response is supervised), giving a much
   denser gradient than D5's 5-token answer span.
4. **Best metric.** Same `--save_best_metric evidence_lift`, but
   evidence-redacted floor is now built from a *different chain's
   rule-establishment session* (so the floor model has *some* rule
   in `M_c`, just not this one's; tests rule-specificity, parallel
   to the cross-chain shuffle on D4v2).
5. **Anti-template-leakage.** The user phrasing of each
   rule-establishment is heavily randomised (5 paraphrases per
   rule); the rule-application session uses a *different* user
   phrasing template that does not contain the rule description.
6. **Step budget.** 4000 steps, frozen 0.6B (rules are harder to
   compress than 5-char codes, deserve more steps). Same v16a recipe
   otherwise.

**Predicted result if right.** `evidence_lift ≥ +0.10` by step 1500
(the rule signal is everywhere in the application sessions, so the
gradient bites early). D3-MC pair/self ≥ 0.4 by step 800 (each
chain's `M_c` is meaningfully different because the rule it stores
is different). `pa_cb_dnm_floor / pa_cb_dnm ≤ 0.3` (high specificity).
**Bonus signal.** A rule-correct accuracy metric on the held-out
rule-application callback (parseable rules like "+7" admit exact
grading): if accuracy > 30% (vs ~5% chance for the arithmetic
families), the architecture has learned to *use* the rule, not just
to weakly bias the LM head.

**Predicted result if wrong.** `evidence_lift` and rule-accuracy
both stay ≤ baseline. **Then the writer/readout pathway has a
deeper degeneracy that no choice of corpus can paper over** — the
architecture itself cannot maintain a chain-specific persistent
state under joint training. This is a much stronger negative result
than v16a's, and motivates an architectural pivot (§2) or a recipe
pivot (§4 + §1 in combination).

**Cost estimate.** 4000 steps × ~6s/step (0.6B frozen, GH200) ≈
**6.5 GH200·h** + ~1 dev-day to build the corpus generator.

---

## §4 — Loss pivot: probe-head bypass on the callback span (M_c → answer without LM head)

**Hypothesis it tests.** v14k's smoking gun says the LM head learns a
content-blind output prior. v16a's nce ≈ log 4 says the LM head doesn't
even discriminate B² cross-product chains. Both say: *the LM head
between M_c and the loss is a one-way valve.* A separate probe head
that takes `M_c` → answer-token logits *without* going through the LM
head installs a gradient channel that the content-blind shortcut
can't satisfy: the probe head's parameters are fresh, so a
content-blind `M_c` cannot fool them with a learned prior.

**Why it's not just another ablation.** The InfoNCE loss in v14k/v15/
v16 *is* a probe-style loss (it scores B² pair NLLs), but it routes
through the LM head, so it inherits the same content-blind bottleneck
as the LM-NLL loss (this is exactly why nce ≈ log 4 on v16a — the LM
head can't discriminate any pair, so neither can NCE). A *separate*
small probe head — initialised from scratch, no pretrained content-
blind structure — is the architectural difference. None of v11–v16
has a head distinct from the LM head reading from `M_c`. It is the
missing column.

**Concrete recipe (`v17d_probe_head_loss`).**

1. **Probe head (~80 LOC, same as §1's `WriterProbeHead`).** A
   single per-chain query, attends `M_c`, projects to `vocab_size`
   via a `nn.Linear(d, V)` (recommend `bias=False`, *not* tied to
   `embed_tokens`).
2. **Loss.** During training, at every batch, in addition to the
   existing LM-NLL loss:
   ```
   logits_probe = probe(M_c, callback_query_vector)   # (B, V)
   loss_probe = F.cross_entropy(logits_probe,
                                callback_first_token_id)
   total_loss = total_loss + λ_probe · loss_probe
   ```
   `callback_query_vector` is built by mean-pooling the callback
   session's question tokens (everything before the first
   `session_callback_mask` position) — this gives the probe a
   query-conditioned hook so the probe is asking *the right
   question*. `callback_first_token_id` is the first answer token
   (via the existing `session_callback_mask`).
3. **`λ_probe` schedule.** Linear warmup 0 → 1.0 over 300 steps,
   constant 1.0 thereafter. The LM-NLL loss is unchanged.
4. **Eval-time.** Probe head **disabled at eval** (consistent with
   the project's vision: at inference, only the LM head reads from
   `M_c`). `tools/eval_callback.py` does not change.
5. **Anti-circularity.** The probe head is initialised from scratch
   so it cannot already encode a content-blind prior at step 0.
   *During training*, if the writer ever encodes a content-blind
   M_c, the probe's loss is at log(V) (~10 nats on D5) and provides
   strong gradient. Only when the writer encodes chain-specific
   content will the probe loss drop. **The probe loss is therefore
   a non-gameable training-time signal** in a way the LM-NLL loss
   is not.
6. **Step budget.** 2500 steps, frozen 0.6B, identical to v16a
   except for the probe head. Compose-able with §3 corpus pivot.

**Predicted result if right (composing on D5).** Probe loss drops
from log(V) ≈ 10 → ≤ 4 nats by step 500 (writer storing first
answer token). `evidence_lift ≥ +0.05` by step 1500 (LM head learns
to use the chain-specific content the probe forced into M_c).
Crucially: `pa_cb_dnm_floor / pa_cb_dnm ≤ 0.5` — *the LM-head
readout, having seen a non-degenerate writer, escapes the
content-blind attractor*.

**Predicted result if wrong.** Probe loss drops to ~4 nats (writer
*can* learn under direct supervision) but `evidence_lift` and
`pa_cb_dnm` are unchanged at LM-eval. **Then the readout/router
genuinely cannot use a non-degenerate `M_c`** — every previous v11-
v16 cell's failure is a readout-side failure, not a writer-side
failure. Pivots to §2 (architecture). Alternative: probe loss
plateaus at log(V) — the writer cannot store the binding even with
direct supervision. Pivots to §2 with stronger urgency. Alternative:
both lift, but `pa_cb_dnm_floor / pa_cb_dnm` stays > 0.9 — readout
*accepts* chain-specific writer output but still *emits* the same
content-blind prior at the LM head. This is the project's *worst*
outcome and motivates a hypernetwork-style read pathway (§5
adjacent).

**Cost estimate.** 2500 steps × ~7s/step (probe is the cheapest
addition: 1 extra forward through ~`d × V ≈ 150M` params) ≈ **5
GH200·h**.

---

## §5 — Wildcard: test-time training (TTT) on M_c during the chain

**Hypothesis it tests.** The "writer is content-blind" story has two
sub-hypotheses we have *not* yet decoupled:
(H_w) the writer architecture can't compute a chain-specific
representation;
(H_o) the writer-training *objective* is the wrong gradient signal.
Both predict the same v14k/v16a observations. A literature-borne
test that decouples them: **at eval/inference, freeze every parameter
except `M_c` itself, then run K steps of SGD on `M_c` against the
chain's evidence-session NLL.** This is mesa-optimisation /
fast-weights-style TTT. It is a *capacity probe* on the readout +
LM-head pathway: if no `M_c` exists that makes the LM head emit the
chain's answer with positive likelihood lift, **the architecture
itself is incapable of representing chain-specific recall**, and
nothing the writer does at training time will help. If TTT *does*
work, the writer is the bottleneck (motivates §1/§2/§4) — the
readout pathway is fine.

**Why it's not just another ablation.** v11–v16 are all training-time
interventions on the writer. TTT-on-`M_c` is a *strict capacity test*
on the readout/router/LM-head subsystem in isolation, with the writer
**bypassed entirely**. It produces a number — `evidence_lift_TTT` —
that no v11–v16 configuration can produce, because all 16 use a
trained writer. It is also the *single cheapest experiment in this
packet*: ~30 minutes on existing v14k/v16a/v15a checkpoints.

**Concrete recipe (`v17e_ttt_on_mc`).**

1. **No new training run.** Run as a post-train *eval* on existing
   `chain_v14k_d4v2_nowarmup_slot_floor_local/best`,
   `chain_v15a_d4v2_norm_replicate_local/best`,
   `chain_v16a_codes_0p6b_frozen_gh200/best`,
   `chain_v16a_codes_0p6b_frozen_gh200/step-500` (whichever lands
   first). All four checkpoints share the same architecture; TTT
   tests the readout+LM-head pair across recipes.
2. **TTT procedure (per chain).** For each chain c with annotated
   evidence + callback positions:
   - Build `M_c^0` from the writer (the existing trained writer's
     output on this chain's evidence session). Or alternatively
     **start from i.i.d. init** — this is the more aggressive test.
   - Run K = 50 steps of Adam (lr=1e-2) on `M_c` *only*, with
     loss = NLL of the *evidence* session under the LM head, with
     `M_c` as the only learnable. Backbone, readout, router all
     frozen.
   - Then score the *callback* session under the TTT'd `M_c`,
     evidence-redacted floor, and full v16a `pa_cb_*` metrics.
3. **Implementation (~150 LOC).** New file
   `tools/eval_ttt_mc.py`. Imports
   `tools/eval_callback.py`'s scoring path, replaces the
   `compress_session(...)` call with a `_optimise_M_c(...)` inner
   loop. Single GPU; runs in 20–40 minutes per checkpoint.
4. **Decision rules.**
   - **TTT `evidence_lift > +0.3`** ⇒ readout+LM-head can
     represent chain-specific recall; the writer is the bottleneck.
     Strongly back **§1 (NIAH warmup)** and/or **§4 (probe head)**.
   - **TTT `evidence_lift ∈ [0, +0.3]`** ⇒ partial capacity;
     readout has signal but is also part of the problem. Compose
     §4 + §2.
   - **TTT `evidence_lift ≤ 0`** ⇒ readout/LM-head subsystem is
     incapable of chain-specific recall regardless of `M_c`
     content. **Architecture must change before any further training
     run is worth launching.** Back §2 unconditionally.

**Predicted result if right (i.e. our preferred outcome — readout is
fine, writer is broken).** TTT `evidence_lift ≥ +0.5` on D5,
≥ +1.0 on D4v2 (where the LM head has more vocabulary leverage).
Resolves project ambiguity in 30 minutes.

**Predicted result if wrong (the alarming outcome).** TTT
`evidence_lift ≤ 0` even with 50 SGD steps and i.i.d. M_c init.
Means `MemoryReadout`+ `BlockAttnResRouter` + LM head jointly cannot
emit a chain-specific answer for any value of M_c. We do not know
which of the three is the bottleneck (sub-decompose by also TTT'ing
the readout's W_V parameter), but the headline result is decisive:
**v14k's `+1.44 pa_cb_dnm` is *fundamentally* a content-blind output
prior, not a marginal-shortcut artefact**, and the project needs
read-side architecture work before any further training campaign.

**Cost estimate.** 0.5 GH200·h for the full sweep across 4
checkpoints. **Cheapest experiment in this packet by 10×.** Run
this *first*; all other proposals' priorities depend on its
outcome.

---

## Composition recommendation

If TTT-on-M_c (§5) reports `evidence_lift > +0.3`, the readout is not
the bottleneck and the priority order is:

1. **§1 + §4 composed** (`v17af`): NIAH warmup phase 1 + probe-head
   loss in phases 2–3. The probe head trains during *both* warmup
   and joint phases; LM-NLL is added in phase 2. Shared probe head
   means one new module, ~200 LOC total. **If §5 is positive this
   is the top-1 move.**
2. **§2** (DeltaNet writer) as a parallel independent run — same
   recipe as v17af but architecturally different writer; gives a
   clean A/B on writer architecture under the new training signal.
3. **§3** (procedural-rules corpus) as the long-tail bet — broader
   scope, but the highest-vision payoff if it works.

If §5 reports `evidence_lift ≤ 0`, **stop launching writer-side
campaigns** and route all GH200 hours to architectural read-side
work (§2 or a new wildcard not in this packet).

---

## Summary table

| § | proposal | category | step budget | GH200·h | falsification time | empirical anchor strength |
|---|---|---|---:|---:|---:|---|
| 1 | NIAH writer warmup (probe-supervised) | A: warmup objective | 3100 | ~6 | step 800 (~1.5h) | strong |
| 2 | DeltaNet writer | B: architecture | 2500 | ~4.5 | step 1500 (~3h) | medium |
| 3 | Procedural-rule corpus | C: corpus | 4000 | ~6.5 | step 1500 (~2.5h) | medium |
| 4 | Probe-head bypass loss | D: loss | 2500 | ~5 | step 500 (~1h) | strong |
| 5 | TTT-on-M_c (eval-only) | E: wildcard | — | ~0.5 | 30 min | medium (but pre-conditional on §1–4) |

**Total compute if all five run: ~22 GH200·h (~1 day).**
**Compute to run §5 alone: ~0.5 GH200·h. Run §5 first.**

---

## Caveats / honest limitations

- **The probe-head ideas (§1 and §4) share a vulnerability.** A probe
  head trained with cross-entropy on the *first* answer token only is
  a one-token classifier. The full answer (5-char ID on D5) is 3 BPE
  tokens. The probe forcing M_c to encode the first token may not
  cascade to encoding all three. Mitigation: in v17a/v17d, extend the
  probe to predict *all* answer-span tokens via a small autoregressive
  head reading from M_c. This adds ~50 LOC.
- **§2 (DeltaNet writer)'s state-packing trick is a kludge.** If
  `S = M_c^T M_c / K` is used, the readout is mathematically
  unchanged, but the *gradient flow* through that pack/unpack is not
  the same as a true DeltaNet. A proper implementation would store S
  as a separate `(B, d, d)` buffer; at d=1024 this is 4MB/chain so
  feasible but not free. Budget +0.5 dev-day if the kludge fails.
- **§3 (procedural rules)'s graders are corpus-specific.** Building
  the 10 rule families is straightforward; building a *grader* that
  actually scores rule-correct continuation is harder for some
  families ("respond in haiku" — no automatic grader). Limit to the
  6 parseable families for the rule-accuracy metric; use NLL only
  for the unparseable 4.
- **§5 (TTT-on-M_c) is a capacity test, not a method.** Even a
  positive result does not give us a writer that learns chain-
  specific content via gradient descent at training time. It only
  tells us *whether such an M_c exists*. The follow-up move is
  always one of §1–§4.
- **Numerical safety.** v16a reports ~1.5% bf16 forward-NaN events;
  any new module added in §1–§4 should be wrapped in the same
  per-step `clip_grad_norm_(max_norm=1.0)` guard, and the probe head
  should use fp32 logits for its CE (cf. `_per_position_nll`'s
  `log_softmax(... .float())`). NaN events in the probe head would
  be silent (loss skipped) but corrupting; add a `--probe_loss_nan_clip`
  guard mirroring the existing main-loss handling.

---

## Top-1 pick (with one-paragraph rationale)

**§5 first (cheap diagnostic), then §1 (NIAH writer warmup) as the
top-1 backed-with-compute move, optionally composed with §4 (probe-
head bypass loss).**

§1 is the top-1 because it is the *only* proposal that gives the
writer a chain-specific gradient signal that does not flow through
the v14k-confirmed content-blind LM-head shortcut. The user explicitly
proposed it in a prior chat ("we need to somehow do a pre-train on the
memory block for it to make out semantic memories... I feel like this
would be easy just by doing needle-in-haystack construction"); it has
direct empirical anchoring in the v14k 95%-redaction-survival result
(which is exactly the symptom of "writer never learned to store
content"); it is implementable in <500 LOC against the existing v16a
recipe and corpus (no new corpus needed); the falsification check at
step 800 is cheap (~1.5 GH200·h); and unlike §2/§3 it does not require
a new architectural class or a new corpus, so it is the lowest-risk
high-payoff move in the packet. §1 + §4 composed (the probe head is
shared between warmup and joint phases, total ~200 LOC) is the
strongest compute-allocation if §5 reports the readout is fine.

## Top fail-fast pick (cheapest to disprove)

**§5 (TTT-on-M_c).**

§5 is the cheapest experiment by an order of magnitude (~0.5 GH200·h
vs ~5 GH200·h for any of §1–§4) and is the only one that runs against
*existing checkpoints* with no new training. Its outcome is
*decision-changing* for the rest of the packet: a positive TTT result
(readout has capacity, writer is the bottleneck) prioritises §1/§4; a
negative TTT result (readout is fundamentally content-blind for any
M_c) deprioritises §1/§3/§4 entirely and forces §2 (architectural
change) as the only honest path. **§5 should run before any v17
training cell launches**; the answer it returns will tell the parent
agent which of the four training-cost proposals is worth the GH200
hours.
