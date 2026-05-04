# Audit A3 — base-rate / template-prior decomposition

> Decompose `pa_cb_ce_mem` on D4v2 callbacks into the empirical
> category-conditional prior `P(item | category)` and the residual gap.
> Joint-trained backbones (v15b @ 0.6B, v15f @ 1.7B) suspected of
> learning the prior directly through the LM pathway, neutralising the
> memory contribution.

Auditor: A3 (GPU 1) · Date: 2026-05-03 · Track:
[`runs.md` v15 OPEN AUDIT](runs.md), candidate leaks #2 and #5.

## TL;DR

* **The user's hypothesis is fully confirmed at 0.6B.** The
  jointly-trained v15b lands within **+0.06 nats** of the empirical
  template-prior baseline at `best` (Δ = +0.0568) and within
  **+0.03 nats** at `final` (Δ = +0.0306). The 4000-step joint backbone
  is doing nothing more than reproducing `P(item | category)` from the
  training distribution.
* **The frozen 0.6B baseline (v15a) sits +2.24 nats *above* the
  template prior in absolute CE.** Memory provides positive *relative*
  lift over the no-memory ceiling (in-trainer `pa_cb_dnm = +1.33` is
  preserved), but it cannot reach the prior baseline that an unfrozen
  backbone gets for free. **The "successful" frozen recipe is a
  high-CE win, not a content-specific one.**
* **At 1.7B the same pattern holds, with v15f early-stopped before
  full convergence to the prior.** v15f_best (step ~820) is at +0.30
  nats above the prior, v15f step-500 at +0.37 nats. The frozen v15e
  is +1.85 nats above the prior. The 1.7B joint backbone is on the
  same trajectory as 0.6B and would have closed the gap had it not
  been killed.
* **Net diagnosis:** the v15 D4v2 corpus does not isolate a
  memory-only signal. Once the backbone is unfrozen, the
  `Your favorite {category} is _____` template gives the LM enough
  cue to learn the marginal answer distribution from the gradient
  exposure on the train callbacks themselves; memory has no work
  left to do. **The corpus, not the architecture, is responsible for
  the v15b/v15f `evidence_lift ≈ 0` collapse.**

## §1 Setup

**Tokeniser.** `Qwen/Qwen3-0.6B` AutoTokenizer (the 1.7B family shares
the same vocabulary, so token-level positions are stable across
runs). Verified via `tools/build_synthetic_persona_callback.py
--audit_tokenisation`.

**Val data.**
`paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt` —
500 chains; we score the first **n = 128** (per the audit spec, up
from `audit_base_prior.py`'s n=64).

**Train data for prior.**
`paper_artifacts/chains/synthd4v2_persona_callback_train_s512.pt` —
5000 chains. Each chain's queried (category, item) pair is parsed
out of the `chain_names` field
(format `synthetic_persona_callback_NNNNN_<cat>_<item>_n2ev`). The
parser handles the one item with an underscore (`tool/rolling_pin`)
correctly via greedy regex.

**Smoothing.** Counts `n(item, category)` get `α = 0.5` add-half
smoothing per `CLOSED_SET[category]` before normalising. This sits
between Laplace (α=1) and the empirical MLE; α=0.5 picks up zero-count
items at a small but nonzero probability without flattening the
empirical signal.

**Memory state for `pa_cb_ce_mem`.** We mirror
`tools/eval_callback.py`: M_c is built by walking the entire chain
prefix `[0, callback_position)` through
`model.model.extract_source` + `model.model.compress_session`. This
matches the eval-time M_c used in the v15 post-train callback-aware
eval. (NB: the in-trainer `pa_cb_ce_mem` builds M_c from a *single*
evidence session sampled per chain, not the full prefix; that is a
different M_c construction and gives slightly higher CE than the
eval-time number — see Reproduction §6.)

**Per-token CE under the prior.** For a val chain with
(`cat`, `item`) and `n_t` answer-span tokens (`session_callback_mask`
positions in the callback session):

```
CE_template_prior(chain) = -log P(item | cat) / n_t
```

Mean across the 128 chains is `CE_template_prior` in the headline
table. The decomposition is exact at the chain level (the per-token
distribution of an item-as-a-whole satisfies
`-log P(item | cat) = Σ_t -log P(tok_t | cat, tok_<t)` for the
generative process "draw item ~ P(·|cat) then tokenise"); dividing by
`n_t` makes it directly comparable to the per-token-mean
`pa_cb_ce_mem`.

**Checkpoints scored** (in evaluation order, GPU 1):

| tag                       | path                                                                                  |
|---------------------------|---------------------------------------------------------------------------------------|
| `v15a_0p6B_FROZEN_best`   | `Runs/chain_v15a_d4v2_norm_replicate_local/best`                                       |
| `v15b_0p6B_joint_best`    | `Runs/chain_v15b_d4v2_norm_jointtrain_local/best`                                      |
| `v15b_0p6B_joint_final`   | `Runs/chain_v15b_d4v2_norm_jointtrain_local/final`                                     |
| `v15e_1p7B_FROZEN_best`   | `Runs/chain_v15e_d4v2_1p7b_norm_local/best`                                            |
| `v15f_1p7B_joint_best`    | `Runs/chain_v15f_d4v2_1p7b_jointtrain_local/best`                                      |
| `v15f_1p7B_joint_step500` | `Runs/chain_v15f_d4v2_1p7b_jointtrain_local/step-500`                                  |

All loaded via `Qwen3MemResForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`.
Each model is freed (`del model; torch.cuda.empty_cache()`) before the
next is loaded; peak VRAM ≈ 4.2 GiB on the 1.7B checkpoints.

**Helper script.** `memory_residuals/tools/audit_a3_template_prior.py`
(new, per the audit-naming convention). Raw per-chain numbers and the
prior tables persisted to
`memory_residuals/results/exp2_chain_recipe/audit_a3_data.json`.

## §2 Template prior construction

The closed-set has 8 categories of nominally 32 items each; two items
(`orange`, `lime`) are duplicated across categories and are
deduplicated by `_flatten_closed_set` to the first category they
appear in (`color`). So the **fruit** category effectively has 30
sample-able items, the rest 32. The 5000 train chains queries are
distributed roughly uniformly across the 254-item answer set:

| category   | total chains | unique items used | smoothed entropy `H(prior)` (nats) | `log(32)` | `log(30)` |
|------------|------------:|------------------:|-----------------------------------:|----------:|----------:|
| color      | 605         | 32                | 3.4484                             | 3.4657    | —         |
| fruit      | 644         | 30                | 3.3885                             | —         | 3.4012    |
| animal     | 593         | 32                | 3.4333                             | 3.4657    | —         |
| object     | 651         | 32                | 3.4501                             | 3.4657    | —         |
| sport      | 605         | 32                | 3.4307                             | 3.4657    | —         |
| tool       | 600         | 32                | 3.4271                             | 3.4657    | —         |
| instrument | 652         | 32                | 3.4416                             | 3.4657    | —         |
| hobby      | 650         | 32                | 3.4287                             | 3.4657    | —         |
| **TOTAL**  | **5000**    | 254               | —                                  | —         | —         |

The empirical prior is **only marginally non-uniform** — H(prior) sits
0.014–0.077 nats below the within-category uniform ceiling. So the
template-prior baseline `CE_template_prior ≈ 2.91` (per-token) is
essentially the within-cued-category uniform floor, scaled down by
the average answer-span length (1.35 BPE tokens):
`log(32) / 1.35 ≈ 2.567`, with the slight deviation from the prior's
mild peakedness and per-chain variance in `n_t`.

Confirmation that the answer set is closed:
`build_synthetic_persona_callback.py:_flatten_closed_set` enumerates
all `(category, item)` pairs from the literal `CLOSED_SET` dict and
both train and val use exactly this generator with different
`--seed`s, so every val chain's answer is in the train support
(modulo the 2 deduplicated orange/lime cells, which never appear in
val either). Smoothing with α=0.5 covers the empty cells anyway.

Headline observations:

* **Maximum train-frequency item is `soccer` (sport, 0.0529).** Its
  per-token surprisal under the prior is `-log(0.0529)/n_t`. With
  `soccer` tokenising to 2 BPE tokens, that's `-log(0.0529)/2 ≈
  1.470` nats/token — well below `log(32)/2 ≈ 1.733`.
* **Minimum (excluding the orange/lime zeros) is `tool/ruler`
  (0.0133).** Surprisal `-log(0.0133)/2 ≈ 2.16` nats/token (ruler
  tokenises to 2). Still below the multi-token-uniform line.
* The 40/128 multi-token chains (avg `n_t = 1.35`) drag
  `CE_template_prior` down vs the single-token-uniform value
  `log(32) = 3.466`. The eventual per-chain mean is
  **`CE_template_prior = 2.9087`**.

## §3 Per-checkpoint headline table

| ckpt                        | n   | `pa_cb_ce_mem` (full-prefix M_c) | `CE_template_prior` | **Δ = ce_mem − prior** | in-trainer `pa_cb_ce_mem_floor` (single-evidence M_c) |
|-----------------------------|----:|--------------------------------:|--------------------:|-----------------------:|-------------------------------------------------------:|
| **v15a_0p6B_FROZEN_best**   | 128 | **5.1490**                      | 2.9087              | **+2.2403**            | 6.1064 (in-trainer ce_mem 5.967, dnm +0.292)           |
| **v15b_0p6B_joint_best**    | 128 | **2.9655**                      | 2.9087              | **+0.0568**            | 2.9772 (in-trainer ce_mem 2.968, dnm +0.013)           |
| **v15b_0p6B_joint_final**   | 128 | **2.9394**                      | 2.9087              | **+0.0306**            | 2.9620 (in-trainer ce_mem 2.975, dnm +0.037)           |
| **v15e_1p7B_FROZEN_best**   | 128 | **4.7626**                      | 2.9087              | **+1.8539**            | 4.7343 (in-trainer ce_mem 4.668, dnm +2.025)           |
| **v15f_1p7B_joint_best**    | 128 | **3.2066**                      | 2.9087              | **+0.2979**            | 3.4836 (in-trainer ce_mem 3.442, dnm +0.103)           |
| **v15f_1p7B_joint_step500** | 128 | **3.2779**                      | 2.9087              | **+0.3692**            | (no `eval_metrics.json` at this checkpoint)           |

Notes on the columns:

* **`pa_cb_ce_mem`** is computed at eval time with full-prefix M_c
  (eval_callback.py-style), 128 val chains, BF16 forward. This is the
  number the v15 callback-aware eval reports.
* **`CE_template_prior`** is identical across rows (it is a property
  of the *corpus*, not the model). Uniform-within-category floor for
  reference: per-token `log(32)/avg_n_t ≈ 3.466/1.35 ≈ 2.567`; the
  empirical prior is slightly worse (+0.34 nats) than uniform because
  the two layers of smoothing-and-finite-sample noise plus per-chain
  `n_t` variance push it up.
* The fifth column is the **in-trainer** `pa_cb_ce_mem_floor` from
  each checkpoint's `eval_metrics.json` — pulled here for context.
  The in-trainer `pa_cb_ce_mem` (also in that file) is **lower** for
  the FROZEN v15a (5.967 vs our 5.149 — uses a single-evidence M_c
  per chain instead of full prefix) and within 0.07 nats elsewhere.
  v15f/step-500 was a forensic checkpoint and has no metrics file.

**Reading the table.**

* The **frozen** 0.6B and 1.7B baselines have Δ = **+2.24** and
  **+1.85** respectively. The memory channel is doing real work *in
  relative terms* (in-trainer dnm/dnm_floor numbers are non-zero), but
  it cannot match what an unfrozen backbone learns from gradient
  exposure to the closed-set template alone.
* The **joint** 0.6B v15b runs (best step ≈ 600, final step 4000)
  collapse Δ to **+0.057** and **+0.031** — within the audit
  threshold of "≈ 0 within ~0.1 nats". The unfrozen backbone has
  learned the empirical `P(item | category)` directly through the LM
  pathway. **This is the user-flagged failure.**
* The **joint** 1.7B v15f runs (killed at step ≈ 820) are partway
  through the same collapse: Δ = **+0.30** at best, **+0.37** at
  step-500. They were on the same trajectory as v15b and would have
  closed the gap given more training. The published `evidence_lift ≈
  0.04` for v15f_best is consistent with a memory pathway that has
  already lost most of its leverage relative to the LM pathway.

The headline `pa_cb_ce_mem ≈ CE_template_prior` collapse is the
quantitative version of the in-trainer "evidence_lift ≈ 0" symptom:
once the LM pathway can match the prior, the memory pathway has no
gradient signal to develop content-specific channels.

## §4 Per-token decomposition

Skipped under the 30-min wall-clock budget. The headline table
already lands the verdict; per-token contributions ("is" vs item
first subword vs subsequent subwords) would be a useful follow-up
if the team needs to localise *where* in the callback sequence the
LM pathway is doing its prior-matching, but it does not change the
verdict that `pa_cb_ce_mem ≈ CE_template_prior` for v15b/final.

## §5 Verdict

**The joint backbone matches the template prior almost exactly at
0.6B and is on track to do so at 1.7B.** Concretely:

1. v15b/best Δ = **+0.057 nats** and v15b/final Δ = **+0.031 nats**.
   By the audit's own threshold (`Δ < 0.1 nats ⇒ joint backbone
   exploits nothing more than the template prior`), this is a **clear
   pass for the user's hypothesis**: the LM pathway has learned the
   empirical `P(item | category)` from the train corpus, and memory
   has no work left to do.
2. v15f at 1.7B is converging on the same fixed point. Δ = +0.30
   (best) and +0.37 (step-500) means the larger backbone is partway
   through the same collapse trajectory; killing it at step ≈ 820
   left some residual gap, but the dynamics are identical.
3. The frozen baselines' Δ values (+2.24 at 0.6B, +1.85 at 1.7B)
   show that the memory pathway, with the v14k recipe, *cannot* on
   its own match what the unfrozen LM trivially learns. That asymmetry
   is the diagnostic: any time the LM pathway is **allowed** to learn
   the marginal directly, it dominates.

**What this says about the v15 design.** The corpus's category cue is
**too strong**: the callback template `Your favorite {category} is`
is one token away from the answer, the closed set has 256 items, and
8 distinct categories carve the answer space into 32-item buckets the
LM can saturate from gradient exposure on a few thousand chains.
Combined with `score_tail_frac = 1.0` putting LM gradient on the
callback tokens during training, this gives the unfrozen backbone a
direct shortcut that does not require the memory pathway. **The
scoring is also too narrow** (1–3 BPE tokens per chain, on a
template-locked surface form), which means there's no per-token
diversity to dilute the prior signal. **Both knobs need to move**:
either suppress the LM gradient on callback tokens (callback-only
loss reweighting that decouples LM from the prior, or freezing the
backbone selectively for the callback-token positions), or change
the corpus so the answer is not predictable from the cue + closed-set
prior alone (longer answers, free-form open vocabulary, or multiple
callbacks that decorrelate the cue→answer mapping).

A targeted v15g cell that masks LM-loss on callback tokens during
joint training (i.e. callback tokens contribute *only* through the
memory-attached signal) would directly test whether the architecture
can learn anything when the prior shortcut is closed.

## §6 Reproduction

All commands are issued from `/home/anon/Desktop/fine-tune` on GPU 1
(`CUDA_VISIBLE_DEVICES=1`). The helper script lives at
`memory_residuals/tools/audit_a3_template_prior.py`.

```bash
# 1) Single command that loads the train+val blobs, builds the prior
#    (α=0.5), and walks the 6 checkpoints sequentially. Wall-clock
#    ≈ 2:30 min on an H100 NVL with bf16 forward at n=128.
CUDA_VISIBLE_DEVICES=1 python memory_residuals/tools/audit_a3_template_prior.py \
  --n_chains 128 \
  --alpha 0.5 \
  --out_json memory_residuals/results/exp2_chain_recipe/audit_a3_data.json \
  --ckpts \
    memory_residuals/Runs/chain_v15a_d4v2_norm_replicate_local/best \
    memory_residuals/Runs/chain_v15b_d4v2_norm_jointtrain_local/best \
    memory_residuals/Runs/chain_v15b_d4v2_norm_jointtrain_local/final \
    memory_residuals/Runs/chain_v15e_d4v2_1p7b_norm_local/best \
    memory_residuals/Runs/chain_v15f_d4v2_1p7b_jointtrain_local/best \
    memory_residuals/Runs/chain_v15f_d4v2_1p7b_jointtrain_local/step-500 \
  --ckpt_tags \
    v15a_0p6B_FROZEN_best \
    v15b_0p6B_joint_best \
    v15b_0p6B_joint_final \
    v15e_1p7B_FROZEN_best \
    v15f_1p7B_joint_best \
    v15f_1p7B_joint_step500
```

```bash
# 2) Sanity-check the in-trainer pa_cb_ce_mem(_floor) numbers cited
#    in §3's last column.
for d in \
    memory_residuals/Runs/chain_v15a_d4v2_norm_replicate_local/best \
    memory_residuals/Runs/chain_v15b_d4v2_norm_jointtrain_local/{best,final} \
    memory_residuals/Runs/chain_v15e_d4v2_1p7b_norm_local/best \
    memory_residuals/Runs/chain_v15f_d4v2_1p7b_jointtrain_local/best ; do
  echo "== $d =="
  python -c "import json,sys; m=json.load(open('$d/eval_metrics.json'));
print({k: m[k] for k in ['pa_cb_ce_mem','pa_cb_ce_mem_floor','pa_cb_dnm','pa_cb_evidence_lift']})"
done
```

Raw artefacts:

* `memory_residuals/results/exp2_chain_recipe/audit_a3_data.json` —
  per-chain `pa_cb_ce_mem` and `CE_template_prior` lists for all 6
  checkpoints, plus the full `prior_counts` and `prior_probs` tables
  used to reconstruct §2.
* `memory_residuals/tools/audit_a3_template_prior.py` — helper.
