# Inner/Outer Discrepancy Rule — Writer Rubric Discovery

*2026-05-09, follow-up to `runs_audit_2026_05_08.md`*

> **Scope.** Records a writer-rubric proposal that emerged while
> reviewing the `presets_lin_lu_CN/` pending bucket. Written down
> so it survives past the chat that produced it. No code in any
> repo was modified.

---

## TL;DR

The structured `[客观] / [主观]` reasoning-chain split that the
data-labeler rubric currently asks writers to fill in *on every
character turn* is the wrong unit. The right rule:

- **`[客观]`** — exclusively for **fact recall, memory management,
  and perspectivization**. "What stored thing am I retrieving from
  earlier in this chain (or about this user / about myself), and
  why is it relevant *here*?" Empty if the turn isn't pulling on
  any prior fact.
- **`[主观]`** — exclusively for the **NOW-state residual** that the
  outer dialogue underdetermines: current feelings, current
  intent, real-time mood shifts, active steering / filtering. Only
  written when there is a **mismatch the model should pick up**.
  Empty otherwise.

This is the explicit form of what the legacy passed corpus
(`user_0`, `user_1`, `user_7` chains in `presets_lin_lu_CN/`) was
already doing implicitly with its free-form `reasoning_chain`
field. The structured-format writers got told to do it *on every
turn*, and the resulting paraphrase-collapse is the dominant
quality failure in the current pending bucket.

---

## Why this maps onto the architecture

memory_residuals' two load-bearing thesis sentences:

> *"…getting the model to lean on **specific facts** stored in M_c…
> rather than the smoothed 'vibe of this conversation' posterior
> it currently uses."* — `runs.md:139–144`

> *"The healthier long-term recipe almost certainly looks like
> jointly-trained **fast memory + slow memory** + unfrozen
> backbone."* — `README.md:108–110`

The split is precisely those two channels:

| field | what it teaches | architectural target | timescale |
|---|---|---|---|
| **客观** | "what do I remember about this person/situation that's relevant right now" | **slow memory (M_c)** — chain-persistent state | spans sessions; k≥1-day callbacks |
| **主观** | "what's the current emotional / intent residual that the slow state cannot predict" | **fast memory** — per-turn residual perturbation | within session; this-turn-only |

This is not a metaphor. The +1.32-nat headline lives entirely in
the slow channel because the backbone is frozen. The follow-up
work the README flags (jointly-trained fast + slow) is exactly
what `[主观]`-as-mismatch-only would supervise.

---

## The "only when there's a mismatch" rule is the load-bearing piece

Three properties that follow from making `[主观]` sparse:

1. **It becomes a labeled supervision signal, not vibe.** Currently
   `[主观]` fires on ~100% of character turns in pending files
   (e.g. `presets_lin_lu_CN/user_16_Day8_dup_1_simplified.json`)
   and collapses to outer-line paraphrase, so the trainer learns
   nothing — the gradient on those tokens is just LM-NLL on a
   restated surface line. If `[主观]` is *only present when inner ≠
   predictable*, it becomes structurally identical to
   `session_callback_mask` (cf. `runs_audit_2026_05_08.md:85`):
   a per-turn `mismatch_mask` that says *"here, the surface
   dialogue underdetermines the next-turn distribution; force the
   read head to pull from fast-memory."* That is directly trainable.

2. **It solves the paraphrase collapse without writer discipline.**
   The reason writers collapse `[主观] ≈ [客观] ≈ outer line` today
   is that the rubric demands `[主观]` on every turn, and most
   turns genuinely have no inner-outer gap, so writers fabricate
   one. Make `[主观]` optional and exceptional, and the failure
   mode disappears by construction.

3. **It gives `[客观]` a clean job too.** Current `[客观]` does
   context-grounded reasoning *plus* fact recall *plus*
   self-cognition all in one cell — which is why writers conflate
   it with `[主观]`. Restricting `[客观]` to fact-recall-and-
   perspectivization makes every `[客观]` entry a candidate
   `chain_evidence_position`. The converter the audit dreams about
   (`runs_audit_2026_05_08.md:215–223`, "~150 LOC of Python")
   becomes essentially trivial under this rule:
   - every `[客观]` entry annotates an evidence pointer
   - every `[主观]` entry annotates a fast-write event
   - callbacks fall out from `[客观]` content as a substring match
     against earlier-day text.

---

## Operationalization for writers

A few sharpening points that need to ship with this rule, or
writers will revert to vacuous filling.

### (a) `[主观]` is two shapes — call them out

| label | shape | example |
|---|---|---|
| `[心情]` (reactive) | mood-state delta from prior turn | "outer reply is polite, but I'm actually irritated by what they just said" |
| `[意图]` (prospective) | intent for the next move | "I'm trying to steer this conversation away from her ex" |

Both are NOW-state, both are fast-memory, but they're orthogonal.
If we only list "feelings" as the trigger, writers will only fill
the reactive half and miss the agentic half.

### (b) Concrete trigger taxonomy for `[主观]`

Otherwise "only when there's a mismatch" collapses back to vibe
check. Fill `[主观]` if and only if **any** of:

1. **Outer-tone masks inner-state.** Polite-but-annoyed,
   calm-but-excited, joke-but-concerned.
2. **Outer-content omits the actual reason.** The surface answer
   isn't the *real* reason for the reply.
3. **Active steering / filtering.** Deciding to skip a topic,
   deciding to dial down expertise, self-monitoring ("I'd come
   off preachy if I kept going"), choosing a register.
4. **Recent emotional carryover.** Yesterday's fight is still
   residual; current turn is polite but charged.
5. **Anticipatory state.** Knows something is coming; bracing or
   prepping.

If none apply, leave `[主观]` empty. The rubric must accept empty
as valid.

### (c) "Highly NOW-focused" ≠ "began on this turn"

`[主观]` is NOW-state even when it integrates recent history.
Test: *"is this entry describing the residual that's active
during this specific turn?"* — not *"did this state begin only
on this turn?"* "Starting to like them, trying not to show it"
is a valid `[主观]` even though the liking has been building.

### (d) The audit rubric itself has to change

Currently `writing_main.html`'s rubric treats missing
`reasoning_chain` as a quality blocker. Under this rule, **missing
`[主观]` with present `[客观]`** is a *signal*, not a defect. The
rubric needs an explicit `inner_outer_discrepancy_present`
boolean (defaulting to absent) so reviewers can pass turns that
legitimately have no fast-state event. Otherwise the new rule
fights the old rubric and writers get conflicting feedback.

### (e) What this does NOT solve

`session_callback_mask` (the answer-span tokens of the callback
session — the supervision the +1.32-nat recipe leans on) still
has to be derived. `[客观]`'s content tells you *which earlier-
session fact is being recalled*, but not the answer-span tokens
in the *target* session. That requires either:

- (i) a small annotation pass mapping `[客观]` entries to
  substrings in earlier-day dialogue, or
- (ii) accepting an unsupervised LM-NLL baseline on the bulk of
  the chain.

The audit's §(a) field-shape gap still applies. This rule solves
the *data-collection* problem cleanly, not the trainer-ingest
plumbing.

---

## What this changes about the pending verdict

Re-reading the `presets_lin_lu_CN/` pending bucket under the new
rule (vs. my prior note, which assumed every `[主观]` entry
needed to be *rewritten*):

- **`user_16` Days 1–27** (the 28-day chain): most paraphrase-
  collapsed `[主观]` entries should become **empty**, not rewritten.
  That's a deletion pass plus highlighting the 5–10 turns per day
  that *do* mark a real mismatch — much cheaper than the rewrite
  I had originally proposed. The cross-day callback structure
  (Days 5/7/8/10/11/13/15/16/19/20/21/24/25/27) survives intact;
  it's already doing fact-recall-shaped work and matches the new
  `[客观]` definition.
- **`user_15`, `user_13`, `user_28`**: same deletion pass on
  `[主观]`, plus light cleanup on `[客观]` where it's currently
  doing in-turn justification rather than fact-pull.
- **Legacy passed corpus (`user_0`, `user_1`, `user_7`)**: was
  already following this rule by intuition. Free-form RC fires
  only when there's a real mismatch, leaves most turns empty.
  This is why the passed corpus reads as having genuine
  interiority while the structured-format pending bucket reads
  as vacuous — the writers were told to do it on every turn,
  not when it mattered.

That last point is the satisfying unification: **the new rule
isn't a deviation from the passed corpus, it's the explicit form
of what the passed corpus was doing implicitly.** The rubric
hasn't been fighting writer skill — it's been fighting itself.

---

## Cross-references

- `runs_audit_2026_05_08.md` §(d) — reasoning-chain density audit
  (~0% across the Kurisu-CN corpus; ~13–16% on Lin-Lu-CN, but
  quality-collapsed in the structured-format bucket)
- `runs_audit_2026_05_08.md` §(a) — field-shape compatibility
  table; `[客观]` annotations would feed `chain_evidence_positions`
- `runs.md:139–144` — "specific facts in M_c rather than vibe
  posterior" thesis
- `README.md:108–110` — fast-memory + slow-memory follow-up
- `data_labeler_testing_env/templates/writing_main.html:4222–4247`
  — current rubric definition of the structured RC format
  (needs the `[心情]/[意图]` split + `inner_outer_discrepancy_present`
  boolean per §(a) and §(d) above)

---

## Status

Proposal, not implementation. Items needed before this is in
production:

- [ ] Update writer-side rubric copy in `writing_main.html` with
  the trigger taxonomy from §(b) and the `[心情]/[意图]` split
  from §(a).
- [ ] Update AI auditor to stop treating empty `[主观]` as a
  blocker; treat it as the default state.
- [ ] Add `inner_outer_discrepancy_present` to the per-turn
  rubric output so reviewers have an explicit yes/no field.
- [ ] Re-run pending bucket under new rule (deletion pass, not
  rewrite) — see §"What this changes" above.
- [ ] Decide §(e)(i) vs §(e)(ii) for the trainer-ingest pass.
