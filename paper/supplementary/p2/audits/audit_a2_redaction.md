# Audit A2 — Eval-redaction trace + adversarial leak test

Audit target: candidate leak #3 in `runs.md` v15 OPEN AUDIT
("Same-chain evidence visibility via the rolling memory state").
Produced under the v15 OPEN AUDIT track on 2026-05-03, GPU 1, n=128
chains from `synthd4v2_persona_callback_val_s512.pt`.

Companion JSONs:
* `results/exp2_chain_recipe/audit_a2_v15a.json`
* `results/exp2_chain_recipe/audit_a2_v15b.json`

## TL;DR

* **The standalone-eval redaction in `tools/eval_callback.py` is not
  leaking evidence content** in the dangerous direction. On v15a/best
  (the canonical positive baseline with `pa_cb_dnm = +1.34`), the
  default redaction's `pa_cb_ce_mem_floor = 5.123` is **0.034 nats above**
  a true cross-chain redaction (`5.091`) and **0.038 nats above** a
  zeroed-input redaction (`5.090`) — i.e. the *default* floor M_c
  produces *worse* callback predictions than the truly-uninformative
  floor, which is the **opposite** sign of an evidence leak.
* All four evidence redactions on v15a (default cb-1 distractors,
  cross-chain non-evidence sessions, skip-evidence, zeroed-token
  sessions) collapse to within a 0.04-nat envelope, well inside the
  user-defined "valid redaction" band (≤ 0.05 nats).
* The standalone-eval `evidence_lift` is **negative under every
  redaction style** on v15a (default −0.031, cross −0.063, skip
  −0.069, zero −0.065). On v15b/best it is essentially zero
  (≤ 0.008 nats in magnitude across all four floors). The redaction
  is sound; the lift signal itself is what's broken.
* In-trainer redaction (`src/train_chain.py:2342–2548`) and standalone
  redaction (`tools/eval_callback.py:64–101`) compute *different*
  floors: the trainer uses a single fresh-start non-evidence session
  in the SAME chain, while `eval_callback.py` uses a rolling-state
  full prefix with evidence positions overwritten by cb-1 sessions
  drawn from the first 32 corpus chains. Both bottom out near zero
  lift on v15a/v15b.
* **Verdict: not leaky.** The minimal observed gap between the
  default redaction and the strongest adversarial redaction is
  0.034 nats on v15a and 0.006 nats on v15b — both well under the
  0.05-nat "valid" threshold, and the gap goes the *wrong way* for a
  leak. The collapse of evidence_lift to ≤ 0 nats is therefore a
  real property of the readout (template-prior dominance,
  candidate #2), not an artefact of the floor measurement.

## §1 In-trainer redaction trace

`_phase_aligned_eval` is the in-trainer method that produces
`pa_cb_ce_mem`, `pa_cb_ce_mem_floor`, and `evidence_lift` on every
eval step. Two facts up front:

1. The trainer's "mem" path builds M_c **fresh from a single
   evidence session**, NOT a rolling prefix:

```2425:2427:memory_residuals/src/train_chain.py
            # mem: M_c from this chain's evidence (fresh, single judge step).
            C_e = model.model.extract_source(evidence[:, :-1])
            M_c = model.model.compress_session(C_e, None)
```

2. The trainer's "floor" path builds M_c from a **single
   non-evidence haystack session of the same chain**, also fresh:

```2472:2501:memory_residuals/src/train_chain.py
            # v11: Evidence-absent floor. For chains that have annotated
            # evidence positions, ALSO score the callback under M_c built
            # from a haystack session that is *not* in evidence_positions.
            # If the readout is content-specific the gap (cb_no_floor -
            # cb_mem_floor) should be ~0 (memory built from irrelevant
            # context shouldn't help), while (cb_no - cb_mem) > 0 means
            # memory built from real evidence does help. The DIFFERENCE
            # of differences -- pa_cb_evidence_lift = (cb_no - cb_mem) -
            # (cb_no_floor - cb_mem_floor) -- is the strongest single
            # diagnostic that the channel is episodic and not generic.
            if ev_in_range and cb_sum > 0:
                non_ev_pre = [
                    p for p in range(0, cb_pos)
                    if p not in ev_in_range
                ]
                if non_ev_pre:
                    e_floor = rng.choice(non_ev_pre)
                    floor_evidence = (
                        ev.chain_session_at(ci, e_floor)
                        .to(self.device).unsqueeze(0)
                    )
                    C_f = model.model.extract_source(floor_evidence[:, :-1])
                    M_floor = model.model.compress_session(C_f, None)
                    nll_floor = self._per_position_nll(model, input_ids, M_floor)
                    cb_mem_floor.append(
                        float((nll_floor * cb_mask_sh).sum().item()) / cb_sum
                    )
                    cb_no_floor.append(
                        float((nll_no * cb_mask_sh).sum().item()) / cb_sum
                    )
```

The floor reduction and `evidence_lift` definition:

```2532:2547:memory_residuals/src/train_chain.py
        # v11 evidence-aware diagnostics (only meaningful when the eval
        # corpus carries chain_evidence_positions, e.g. the v11 LME
        # corpus). On pre-v11 corpora these are NaN.
        out["n_pa_cb_evidence_labelled"] = n_with_ev_label
        out["pa_cb_ce_mem_floor"] = m(cb_mem_floor)
        out["pa_cb_ce_nomem_floor"] = m(cb_no_floor)
        if cb_mem_floor and cb_no_floor:
            floor_dnm = out["pa_cb_ce_nomem_floor"] - out["pa_cb_ce_mem_floor"]
            out["pa_cb_dnm_floor"] = floor_dnm
            # Lift = (memory benefit when evidence is present) MINUS
            # (memory benefit when evidence is absent). > 0 means the
            # readout is content-specific to evidence-bearing M_c.
            out["pa_cb_evidence_lift"] = out["pa_cb_dnm"] - floor_dnm
        else:
            out["pa_cb_dnm_floor"] = float("nan")
            out["pa_cb_evidence_lift"] = float("nan")
```

So the in-trainer "floor" is a **same-chain non-evidence session**.
M_c starts at `None` (zeros) and integrates exactly one session.
There is no rolling prefix and therefore no opportunity for a
prior-pass leak. The only way information about the evidence text
could leak into this floor is if the *non-evidence haystack
sessions* of the same chain themselves contain evidence content
(template echoes, persona reuse, or filler-session paraphrases of
the answer). That's a corpus-construction concern, not a redaction
bug.

## §2 Standalone redaction trace

`tools/eval_callback.py` operates differently. It builds a **rolling
M_c over the full prefix `[0, cb_pos)`** and only replaces the
evidence-position sessions:

```64:101:memory_residuals/tools/eval_callback.py
@torch.no_grad()
def build_Mc(model, blob, ci, end, device, *, evidence_redact: bool = False,
             redact_with: torch.Tensor | None = None):
    """Build M_c from chain[ci][:end].

    If ``evidence_redact`` is True, replace evidence sessions
    (positions in ``blob['chain_evidence_positions'][ci]``) with
    ``redact_with`` rows (a (n_evidence, S) tensor of distractor
    session ids).
    """
    cfg = model.config
    K, d = cfg.memres_num_vectors, cfg.hidden_size
    M_c = torch.zeros(1, K, d, device=device, dtype=torch.bfloat16)
    if end <= 0:
        return M_c

    evidence_positions: set[int] = set()
    if evidence_redact and "chain_evidence_positions" in blob:
        try:
            evidence_positions = {
                int(p) for p in blob["chain_evidence_positions"][ci]
                if int(p) < end
            }
        except (TypeError, KeyError):
            evidence_positions = set()

    redact_iter = iter(redact_with) if redact_with is not None else None
    for j in range(end):
        if j in evidence_positions and redact_iter is not None:
            try:
                sess = next(redact_iter).to(device).unsqueeze(0).long()
            except StopIteration:
                sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
        else:
            sess = chain_session(blob, ci, j).to(device).unsqueeze(0)
        C = model.model.extract_source(sess[:, :-1])
        M_c = model.model.compress_session(C, M_c)
    return M_c
```

Distractor pool construction (cb-1 of the first 32 chains):

```136:143:memory_residuals/tools/eval_callback.py
    # Pool of distractor sessions for evidence redaction: take the
    # last session of every chain (callback session is rich in
    # template structure but devoid of evidence text).
    distractor_pool: list[torch.Tensor] = []
    for ci in range(min(redact_pool_size * 4, n_chains)):
        cb_pos = int(blob["chain_callback_position"][ci])
        last_filler_pos = max(0, cb_pos - 1)
        distractor_pool.append(chain_session(blob, ci, last_filler_pos))
```

Per-chain redact-row selection (deterministic offset):

```159:167:memory_residuals/tools/eval_callback.py
        n_evidence = len(blob.get("chain_evidence_positions", [[]])[ci])
        offset = (ci * 7) % max(1, len(distractor_pool) - n_evidence)
        redact_rows = torch.stack(
            distractor_pool[offset : offset + max(1, n_evidence)]
        ) if distractor_pool else None
        Mc_floor = build_Mc(
            model, blob, ci, cb_pos, device,
            evidence_redact=True, redact_with=redact_rows,
        )
```

Two things to note about how the redaction is wired:

* `evidence_redact=True` does **not** zero or skip evidence sessions.
  It substitutes them with cb-1 sessions drawn from the first
  `redact_pool_size*4 = 32` chains of the same corpus. These cb-1
  sessions are non-evidence (the corpus places callbacks at position 9
  and evidence at positions in [0,8]) but they are *callback-precursor
  filler turns* heavy in conversational template structure.
* `redact_with` is consumed by an iterator with a `StopIteration`
  fallback to the **original session**. As long as
  `len(redact_rows) >= n_evidence_in_prefix`, the fallback never
  triggers; in practice n_evidence ≤ 2 for this corpus and the pool
  has 32 entries, so the substitution is complete.

## §3 Static analysis of leak surfaces

The candidate leak path is "rolling memory state retains
evidence-derived information across the redaction boundary". Going
through each surface:

1. **M_c reset.** `build_Mc` constructs `M_c = torch.zeros(1, K, d,
   ...)` at the entry of every call (`eval_callback.py:76`). The
   match-floor / shuffle / floor passes therefore each start from a
   fresh zero state. No state carries across passes.
2. **Detach boundaries.** Eval is wrapped in `torch.no_grad()`
   (`@torch.no_grad()` on `build_Mc` at `eval_callback.py:64`,
   `callback_loss` at `:104`, and `_phase_aligned_eval` and
   `_per_position_nll` in the trainer). There's no autograd state to
   spill across passes.
3. **Writer GRU side channels.**
   `MemoryBlock.forward(C, M_c_prev)` (`modeling_memres.py:1096–1145`)
   takes only the current-session source `C` and prior memory
   `M_c_prev`. `extract_source` (lines 1591–) takes only
   `input_ids` and a (frozen) layer index. There is no path by
   which evidence text from a parallel pass could enter the writer.
4. **Slot persistence.** `M_c` is a function argument threaded
   explicitly through `compress_session` calls in a loop local to
   `build_Mc`; no module attribute caches it. Verified by reading
   the loop at `eval_callback.py:90–101` and the
   `Qwen3MemResForCausalLM` forward path (no `M_c` cache attribute
   set anywhere).
5. **Same-chain prefix bleed-through.** This is the one real
   leakage surface. In the rolling path, the *non-evidence*
   sessions at positions ≠ evidence are still fed into M_c. If
   those filler sessions paraphrase / reference the evidence
   (template repetition, persona priming, etc.) the floor M_c
   carries an indirect signal. The only way to bound this
   surface is empirical: substitute cross-chain or zeroed
   sessions for evidence and see how much the floor moves. That's
   §4.
6. **Distractor pool overlap.** The cb-1 pool covers the FIRST 32
   chains, so for `ci ∈ [0, 32)` the pool *includes this chain's
   own cb-1 session*. The cb-1 session is non-evidence (evidence
   positions live in [0, cb_pos-1)) so this is not a direct leak,
   but it does mean the "default" redaction occasionally injects
   the chain's own template/persona context into M_c — which
   biases the default floor toward looking *more* helpful, not
   less. Empirically (§4) the default floor is *less* helpful than
   cross-chain distractors, so this isn't manifesting as a leak.

## §4 Adversarial test results

Same corpus and `n=128` for both checkpoints. Five floor variants
were measured in a single pass per chain to keep cost bounded:

| variant            | how evidence positions are filled                               |
|--------------------|-----------------------------------------------------------------|
| `default`          | cb-1 sessions of first 32 chains, deterministic offset (current)|
| `cross`            | random non-evidence positions of OTHER chains (this script)     |
| `skip`             | evidence sessions are not compressed at all                     |
| `zero`             | replace evidence input_ids with all `pad_token_id=0`            |
| `match` (control)  | original evidence sessions (no redaction)                       |

### v15a/best  (`Runs/chain_v15a_d4v2_norm_replicate_local/best`)

`pa_cb_dnm = +1.338`  (CE_nomem 6.492 − CE_mem 5.154);
`pa_cb_dsh = +0.0002`  (matched vs cross-chain shuffle).

| floor variant   | `ce_mem_floor` | `pa_cb_dnm_floor` | `evidence_lift` |
|-----------------|----------------|-------------------|-----------------|
| `match` (no redact) | 5.1545     | 1.3377            | 0.0000 (def.)   |
| `default`       | 5.1235         | 1.3687            | **−0.0310**     |
| `cross`         | 5.0911         | 1.4011            | −0.0634         |
| `skip`          | 5.0857         | 1.4065            | −0.0688         |
| `zero`          | 5.0896         | 1.4026            | −0.0648         |

* Spread between strongest adversarial floor (`skip`, 5.0857) and
  weakest (`default`, 5.1235) = **0.038 nats**. Inside the
  user-specified ≤ 0.05-nat "valid" band.
* Direction: **default is *higher* CE than cross/skip/zero**, i.e.
  the default floor is *less* helpful than truly content-free
  floors. A leak would show the opposite (the leaky floor would be
  *more* helpful because it would still encode evidence). So the
  default redaction is, if anything, slightly *anti*-helpful — most
  likely because cb-1 distractors are template-rich filler that
  jams the slots with off-topic conversational scaffolding.
* `evidence_lift` is **negative under all four redactions**. This is
  not a redaction-floor problem — every redaction agrees that
  redacted-M_c outperforms evidence-M_c on callback CE. The
  readout is using M_c as a "memory regime is on" template prior
  rather than as evidence content (consistent with the
  `pa_cb_dsh ≈ 0` finding: shuffled M_c gives the same callback CE
  as matched M_c).

### v15b/best  (`Runs/chain_v15b_d4v2_norm_jointtrain_local/best`)

`pa_cb_dnm = +0.0074`;  `pa_cb_dsh = −0.0102`.

| floor variant   | `ce_mem_floor` | `pa_cb_dnm_floor` | `evidence_lift` |
|-----------------|----------------|-------------------|-----------------|
| `match` (no redact) | 2.9742     | 0.0074            | 0.0000 (def.)   |
| `default`       | 2.9721         | 0.0095            | **−0.0021**     |
| `cross`         | 2.9665         | 0.0152            | −0.0078         |
| `skip`          | 2.9731         | 0.0085            | −0.0011         |
| `zero`          | 2.9690         | 0.0126            | −0.0052         |

* All five `ce_mem` / `ce_mem_floor` values lie within 0.008 nats of
  each other. The redaction-style spread is **0.0066 nats** (cross
  vs skip), an order of magnitude under the 0.05-nat threshold.
* The joint-trained backbone (v15b) is essentially memory-blind:
  every M_c gives the same callback CE up to 1e-2 nats. There is
  no evidence-vs-floor signal at all and no way for a leak to be
  hiding in the noise.

### Cross-checkpoint comparison

|                         | v15a/best | v15b/best |
|-------------------------|-----------|-----------|
| `pa_cb_dnm` (matched)   | +1.3377   | +0.0074   |
| `dnm_floor_default`     | +1.3687   | +0.0095   |
| `dnm_floor_cross`       | +1.4011   | +0.0152   |
| `dnm_floor_zero`        | +1.4026   | +0.0126   |
| `evidence_lift_default` | −0.0310   | −0.0021   |
| `evidence_lift_cross`   | −0.0634   | −0.0078   |
| `default ↔ cross gap`   | 0.0324    | 0.0057    |
| `default ↔ skip gap`    | 0.0378    | −0.0010   |
| `default ↔ zero gap`    | 0.0339    | 0.0031    |

Both checkpoints land inside the 0.05-nat "valid redaction" band.
The leak (if any) is *not* bigger on the joint-trained backbone —
v15b's redaction-style spread (0.006 nats) is roughly 6× tighter
than v15a's (0.038 nats), the opposite of the original concern.

## §5 Verdict

**The standard `eval_callback.py` redaction is trustworthy as a
no-evidence floor.** On the canonical positive baseline (v15a/best,
where `pa_cb_dnm = +1.34` confirms evidence is genuinely changing
the readout's output), the default cb-1 distractor redaction sits
0.034 nats *higher* (worse) than a true cross-chain redaction and
0.038 nats *higher* than a truly zeroed redaction. That is:

1. Inside the user-specified validity band (≤ 0.05 nats).
2. The wrong direction for an evidence leak. A leaky redaction
   would have left evidence-derived structure in M_c, lowering
   `ce_mem_floor` *below* the cross/zero floor, not above it.

The most plausible reading of the small default-vs-cross gap is the
opposite of a leak: the cb-1 sessions used as distractors are
**template-rich callback-precursors**, and folding 1–2 of those into
M_c slightly pollutes the slots with off-topic scaffolding noise.
The cross-chain and zero-input redactions don't have this
interference, so they end up looking ~0.04 nats *more* favourable.

Therefore:

* `evidence_lift ≤ 0` on v15a is **not a redaction artefact**. It
  reflects the real readout behaviour: the model treats M_c as a
  uniform template prior, not as content. This is the
  candidate #2 ("template prior") symptom, and it stands
  independently of how the floor is built.
* `evidence_lift ≈ 0` on v15b is genuinely zero — the model
  doesn't use M_c at all and no redaction style can reveal a
  signal that isn't there.

**No fix needed for the redaction itself.** If anything, switching
the standalone default to **cross-chain non-evidence sessions** (§4
`cross` mode) would be a marginal improvement: it removes the
cb-1-template interference and gives a cleaner floor by ~0.03 nats
on v15a, while keeping v15b unchanged. That's a one-line code
change in `eval_callback.py:139–143` (replace the cb-1 pool with a
random sample of non-evidence positions across all chains) and is
listed below as a recommended polish, not a correctness fix.

The in-trainer redaction is also clean: it builds the floor M_c
fresh from a single non-evidence session of the same chain, so the
only residual leak surface is corpus construction (filler sessions
that paraphrase the answer), which would have to be addressed at
chain-build time.

The audit therefore *clears* candidate #3 (rolling-state evidence
visibility) and **redirects the load-bearing question back to
candidate #2 (template prior)**. The fact that
`pa_cb_ce_mem_floor < pa_cb_ce_mem` on v15a (5.123 < 5.155) under
every redaction style we tried is the strongest available evidence
that what the readout has learned to do is "shift output
distribution when M_c ≠ None", not "retrieve content from M_c".

## §6 Reproduction

GPU 1 only (so we don't collide with Audit A1 on GPU 0).

```bash
# v15a — frozen positive baseline (pa_cb_dnm = +1.34)
cd /home/anon/Desktop/fine-tune/memory_residuals && \
CUDA_VISIBLE_DEVICES=1 python tools/audit_a2_redaction.py \
    --model_path Runs/chain_v15a_d4v2_norm_replicate_local/best \
    --corpus    paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
    --n_chains_max 128 \
    --output    results/exp2_chain_recipe/audit_a2_v15a.json
```

```bash
# v15b — joint-trained backbone (where the leak concern was raised)
cd /home/anon/Desktop/fine-tune/memory_residuals && \
CUDA_VISIBLE_DEVICES=1 python tools/audit_a2_redaction.py \
    --model_path Runs/chain_v15b_d4v2_norm_jointtrain_local/best \
    --corpus    paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
    --n_chains_max 128 \
    --output    results/exp2_chain_recipe/audit_a2_v15b.json
```

Each run produces a JSON with the five floor variants and per-chain
breakdowns. Expected wall time: ~95 s per checkpoint on a single
H100 NVL at bf16. Total audit cost: ~3 minutes of GPU time.

The helper script `tools/audit_a2_redaction.py` is self-contained
and does not modify `tools/eval_callback.py`. It re-implements
`build_Mc` with a `mode` parameter (`match` / `default_pool` /
`cross_chain` / `skip` / `zero`) and runs all five floors per chain
in one pass.
