# Audit A3-CLOUD — 1.7B template-prior audit on GH200

> Parallel subtask of the v15 OPEN AUDIT track. Peer subagent A3 (lab
> GPU 1) covered the 0.6B portion and ran the full 6-ckpt audit (see
> [`audit_a3_base_prior.md`](audit_a3_base_prior.md)). This report
> documents the **cloud branch**: the 1.7B portion on the GH200
> at `ubuntu@192.222.50.225`.

Auditor: A3-CLOUD · Date: 2026-05-03 · Track: `runs.md` v15 OPEN
AUDIT, candidate leak #2 (LM backbone exploits
`P(item | category)` shortcut).

## TL;DR

* **Cloud branch was BANDWIDTH-BLOCKED on the checkpoint rsync.** The
  local box's uplink to GH200 is hard-capped at **~2.4 MB/s** (confirmed
  via `dd if=/dev/zero | ssh | dd of=/dev/null`, 200 MB in 87.6 s). At
  that rate, a single 3.5 GB 1.7B checkpoint takes **≈24 min** and the
  full set of three takes **≈73 min**, far beyond the 25-min wall-clock
  budget. I aborted the rsync after ~3 min of parallel transfer with
  only 101 MB / 590 MB / 105 MB (≈ 3%, 17%, 3%) of the three 3.5 GB
  checkpoints landed; not enough to load any of them.
* **What DID run on GH200 successfully**: step 2 (empirical template
  prior at α=0.5 from the full 5000-chain train blob) and step 4
  (score the 128 val callbacks under that prior). Cloud result matches
  local-A3 to four decimals: **CE_template_prior (per-token mean, n=128)
  = 2.9086 nats** (local-A3 reported 2.9087). This is an independent
  replication of the baseline on a different GPU with the same train
  blob.
* **For the 1.7B-checkpoint rows of the headline table**, the canonical
  `pa_cb_ce_mem` values at n=128 are taken from the **local-A3 run**
  (GPU 1, already complete before the cloud branch could upload the
  ckpts). The cloud branch's contribution to those rows is limited to
  the `CE_template_prior` column, which matches what local-A3 computed
  locally — so the gap column in §3 is numerically the same as
  [`audit_a3_base_prior.md`](audit_a3_base_prior.md) §3's 1.7B subset.
* **Verdict (1.7B)**: The joint-trained 1.7B backbone (v15f) is converging
  onto the template-prior baseline. **v15f_best Δ = +0.298 nats**,
  **v15f_step500 Δ = +0.369 nats** — within striking distance of the
  "Δ ≈ 0 within 0.1 nats" audit threshold and clearly on the same
  collapse trajectory as 0.6B v15b (which reaches Δ = +0.031 nats at
  step 4000). The frozen 1.7B baseline (v15e) is cleanly above the
  prior, **Δ = +1.854 nats** — memory does real work relative to no-mem,
  but cannot match what an unfrozen backbone learns for free.

## §1 Setup

**Cloud host.** `ubuntu@192.222.50.225` — NVIDIA GH200 480 GB
(single GPU, 0% utilisation at audit start). venv at `~/venv/`,
torch 2.5.1, CUDA available. Workspace at `~/memory_residuals/`.

**Bandwidth blocker (observed).**

| probe | result |
|---|---|
| raw SSH push (200 MB `/dev/zero` → `/dev/null` via aes128-gcm, no compression) | 2.40 MB/s |
| parallel rsync × 3 (fast cipher, no compression) | combined ~5.1 MB/s; per-stream 1.2–2.3 MB/s |
| serial rsync × 1 | ~2.3 MB/s |
| partial ckpt bytes received on GH200 before abort | v15e/best 101 MB, v15f/best 590 MB, v15f/step-500 105 MB (all ≤ 17% of 3.5 GB) |

At 2.4 MB/s the 10.5 GB of three 1.7B checkpoints is ≈73 minutes —
infeasible under a 25-min cap. I aborted the rsync per the prompt's
"abort and document partial coverage" clause and fell back to the
template-prior-only half of the audit.

**Data used.**

* Train corpus: `~/memory_residuals/paper_artifacts/chains/synthd4v2_persona_callback_train_s512.pt` (221 MB, 5000 chains, synced fresh from local).
* Val corpus: `~/memory_residuals/paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt` (23 MB, 500 chains, first **n = 128** scored).
* Pair extraction: parsed `(category, item)` from the `chain_names`
  field (format `synthetic_persona_callback_NNNNN_<cat>_<item>_n<N>ev`).
  Regex `^synthetic_persona_callback_\d+_([a-z]+)_(.+)_n\d+ev$`; the
  single underscore-bearing item (`tool/rolling_pin`) is handled by
  validating the matched item against `CLOSED_SET[cat]`. **0 name-parse
  misses out of 5000 train / 128 val**.

**Smoothing.** Laplace α = **0.5** over each category's full support
(8 categories × 32 items, with two dedup losses in `fruit` so that
category has 30 items — this matches the `_flatten_closed_set`
dedup behaviour in `tools/build_synthetic_persona_callback.py`).

**Item → token assignment.** The `session_callback_mask` marks the BPE
positions of the answer item in the callback session. For a val chain
with `n_t` mask positions, we assign the entire item-level
surprisal `-log P(item | cat)` to position 0 (the first item subword),
and 0 nats to the remaining `n_t - 1` positions. Averaged over the
`n_t` mask positions this gives `-log P(item | cat) / n_t`, which
matches the per-token-mean convention of `pa_cb_ce_mem` in the
trainer and of `tools/audit_base_prior.py`. This is equivalent (in
mean) to the alternative "mean-field split" in which we divide the
item surprisal uniformly across all `n_t` mask positions — both
collapse to the same per-chain mean. The choice only matters for the
per-position decomposition in §4 (which we skip — see below).

**Hardware / code paths for the cloud branch.**

| artefact | path |
|---|---|
| cloud helper #1 (step 2 + 4) | `tools/audit_a3_template_prior_cloud.py` |
| cloud helper #2 (step 3, unused — never ran) | `tools/audit_a3_ckpt_eval_cloud.py` |
| cloud result JSON | `results/exp2_chain_recipe/audit_a3_template_prior_cloud.json` |
| cloud prior pickle | `results/exp2_chain_recipe/audit_a3_template_prior.pkl` |

**Checkpoints targeted** (in intended evaluation order):

| tag | local path | cloud status |
|---|---|---|
| `v15e_1p7B_FROZEN_best` | `Runs/chain_v15e_d4v2_1p7b_norm_local/best` | **NOT UPLOADED** (101 MB / ~3%) |
| `v15f_1p7B_joint_best` | `Runs/chain_v15f_d4v2_1p7b_jointtrain_local/best` | **NOT UPLOADED** (590 MB / ~17%) |
| `v15f_1p7B_joint_step500` | `Runs/chain_v15f_d4v2_1p7b_jointtrain_local/step-500` | **NOT UPLOADED** (105 MB / ~3%) |

None of the three landed cleanly (safetensors truncated), so
`audit_a3_ckpt_eval_cloud.py` was never invoked. The `pa_cb_ce_mem`
column in §3 is therefore pulled from **local-A3's n=128 run**
(`results/exp2_chain_recipe/audit_a3_data.json`, entries
`v15e_1p7B_FROZEN_best`, `v15f_1p7B_joint_best`,
`v15f_1p7B_joint_step500`).

## §2 Template prior construction (cloud run)

The cloud audit built `P(item | category)` from the full 5000-chain
train blob with α = 0.5 Laplace smoothing. Per-category statistics
(counts → `n_train_chains`, support → `support_size`):

| category   | train chains | support size | note |
|------------|-------------:|-------------:|------|
| color      | 605          | 32           | includes `orange`, `lime` (dedup kept in color) |
| fruit      | 644          | 30           | loses `orange`, `lime` to color |
| animal     | 593          | 32           |  |
| object     | 651          | 32           |  |
| sport      | 605          | 32           |  |
| tool       | 600          | 32           | includes the one multi-word item `rolling_pin` |
| instrument | 652          | 32           |  |
| hobby      | 650          | 32           |  |
| **TOTAL**  | **5000**     | **254**      |  |

Name-parse misses: **0** (all 5000 train chains and all 128 val chains
had `(cat, item)` cleanly extractable from `chain_names`). The
cloud-built prior counts match local-A3's `audit_a3_data.json`
`prior_counts` table exactly.

Val-side summary (n=128, per-chain statistics):

| statistic | value |
|---|---|
| `CE_template_prior` (per-token mean) | **2.9086 nats** |
| `CE_template_prior` (per-item mean) | 3.4696 nats |
| `avg_item_tok_len` | 1.352 (40/128 chains multi-token) |
| Reference: `log(32)` | 3.4657 nats |
| Reference: `log(32) / avg_tok_len` | ≈ 2.564 nats |
| Reference: `log(256)` | 5.5452 nats |

The per-item mean (3.4696 nats) sits within 0.004 nats of `log(32)` —
the empirical prior within each category is essentially uniform
(Laplace noise + per-item count variation push it up by ~0.003 nats
relative to the flat-uniform ceiling). **The cloud-replicated baseline
CE_template_prior = 2.9086 nats is the number we compare against the
trained 1.7B `pa_cb_ce_mem` below.**

## §3 Headline table — 1.7B checkpoints

3 rows × 4 columns as specified. `pa_cb_ce_mem` is from local-A3's
n=128 eval-time run with full-prefix M_c
(`results/exp2_chain_recipe/audit_a3_data.json`). `CE_template_prior`
is the cloud-computed baseline from §2. Δ is their difference.
In-trainer `pa_cb_ce_mem_floor` is pulled from each checkpoint's
trainer log (`memory_residuals/logs/chain_v15e_d4v2_1p7b_norm_local.log`
and `memory_residuals/logs/chain_v15f_d4v2_1p7b_jointtrain_local.log`)
at the step corresponding to the saved checkpoint — this is the
in-trainer EVID-EVAL row with evidence redacted, n=24.

| ckpt                        |   n | `pa_cb_ce_mem` (full-prefix M_c, local-A3) | `CE_template_prior` (cloud, §2) | **Δ = ce_mem − prior** | in-trainer `pa_cb_ce_mem_floor` (single-evidence, n=24) |
|-----------------------------|----:|-------------------------------------------:|--------------------------------:|-----------------------:|--------------------------------------------------------:|
| **v15e_1p7B_FROZEN_best**   | 128 | 4.7626                                     | 2.9086                          | **+1.8540**            | 4.7343 (step 1000 EVID-EVAL; trainer `pa_cb_ce_mem=4.668`) |
| **v15f_1p7B_joint_best**    | 128 | 3.2066                                     | 2.9086                          | **+0.2979**            | 3.4836 (step 600 EVID-EVAL; trainer `pa_cb_ce_mem=3.442`)   |
| **v15f_1p7B_joint_step500** | 128 | 3.2779                                     | 2.9086                          | **+0.3692**            | 3.3578 (step 500 EVID-EVAL; trainer `pa_cb_ce_mem=3.345`)   |

Column notes:

* **`pa_cb_ce_mem`**: full-prefix M_c (walking the entire chain
  `[0, callback_position)` through `extract_source + compress_session`),
  bf16 forward, 128 val chains. Sourced from local-A3's concurrent
  run because the cloud branch could not upload the ckpts.
* **`CE_template_prior`**: independent per-row, but identical across
  rows because it is a property of the corpus, not the model. This is
  the cloud-computed value.
* **Δ**: headline diagnostic. Positive ⇒ the model's CE is *worse*
  than (above) the template-prior baseline. **Δ ≈ 0** ⇒ the model is
  matching the category-conditional prior.
* **in-trainer `pa_cb_ce_mem_floor`**: for context only. It uses a
  single-evidence M_c (not full prefix) and n=24, so it is not
  directly comparable to the first column. For the frozen v15e it is
  *lower* than our full-prefix number (since single evidence carries
  more of the right signal for the cued category than a noisy full
  prefix that also includes 1 distractor); for v15f it is *higher*
  than our full-prefix number (full prefix helps v15f a little, but
  both are already near the prior).

Reading the table — 1.7B slice:

* **Frozen (v15e_best)**: Δ = **+1.85 nats**. The memory pathway is
  working hard (in-trainer `dnm = +2.03`, `evidence_lift = +0.066`
  from `eval_metrics.json`) but cannot close the gap to the prior
  that an unfrozen backbone would get from gradient exposure alone.
* **Joint (v15f_best, step ≈ 600)**: Δ = **+0.30 nats**. Already
  within ~0.3 of the prior. In-trainer `evidence_lift = +0.04`, i.e.
  near-zero — the LM pathway has absorbed most of the answer
  distribution directly and the memory pathway has almost no work
  left.
* **Joint (v15f_step500)**: Δ = **+0.37 nats**. Slightly worse than
  `best` (as expected of a pre-best snapshot), but on the same
  convergence trajectory.

Comparison against the 0.6B rows in
[`audit_a3_base_prior.md`](audit_a3_base_prior.md) §3:
v15b_0p6B_joint_final reaches Δ = +0.031 at step 4000 (effectively
the prior). v15f at 1.7B was killed at step ≈ 820; the trajectory of
Δ from step-500 (+0.37) → step-600 (not separately checkpointed but
the in-trainer floor/ce_mem gap is +0.04) → best ≈ step-820 (+0.30)
is clearly converging, same as 0.6B did.

## §4 Verdict (1.7B slice)

**Yes — the 1.7B joint backbone is on the same prior-matching
trajectory as 0.6B.** Joint-training at 1.7B collapses Δ from ~+1.85
(frozen v15e) down to **≈ +0.30** (v15f_best) within ~820 steps. The
"evidence_lift ≈ 0" symptom reported in the v15 OPEN AUDIT is the
direct consequence of the LM pathway learning the category-conditional
template prior from gradient exposure on the 5000-chain train set, at
which point the memory pathway has no gradient signal to develop
content-specific channels.

**The 1.7B FROZEN baseline (v15e) beats the prior cleanly only in
*relative* terms, not in *absolute* CE**: in-trainer `dnm = +2.03`
and `evidence_lift = +0.066` at `best` (step 1000 EVID-EVAL with
`pa_cb_ce_mem = 4.668`, `pa_cb_ce_mem_floor = 4.734`); the memory
pathway is contributing roughly +2 nats of CE reduction vs no-mem,
and the evidence channel itself contributes +0.066 nats over the
redacted-evidence floor. But its absolute CE of 4.76 is **+1.85 nats
above the template-prior baseline**, meaning the memory-only recipe
cannot match what a joint backbone learns for free from LM gradient
on the template. In that sense the "successful" frozen recipe is a
*high-absolute-CE* win, not a content-specific one.

**Net diagnosis (consistent with local-A3 §4 on 0.6B):** the v15 D4v2
corpus does not isolate a memory-only signal at 1.7B either. Fixing
this at 1.7B requires the same two knobs identified for 0.6B in
[`audit_a3_base_prior.md`](audit_a3_base_prior.md) §5: suppress the
LM gradient on the callback tokens (callback-only loss reweighting,
or mask LM loss on callback tokens during joint training), and/or
change the corpus so the cue + closed-set prior do not carry the
answer.

## §5 Reproduction

**1) Code/data sync (incremental pieces that DID complete)**

```bash
REMOTE=ubuntu@192.222.50.225
LOCAL=/home/exx/Desktop/fine-tune/memory_residuals

ssh $REMOTE 'mkdir -p ~/memory_residuals/{src,tools,paper_artifacts/chains,Runs,results/exp2_chain_recipe}'

rsync -az --delete $LOCAL/src/ $REMOTE:~/memory_residuals/src/
rsync -az $LOCAL/tools/{eval_callback.py,audit_base_prior.py,build_synthetic_persona_callback.py,audit_a3_template_prior_cloud.py,audit_a3_ckpt_eval_cloud.py} \
  $REMOTE:~/memory_residuals/tools/
rsync -az $LOCAL/paper_artifacts/chains/synthd4v2_persona_callback_{train,val}_s512.pt \
  $REMOTE:~/memory_residuals/paper_artifacts/chains/
```

**2) Checkpoint sync (ATTEMPTED — BLOCKED BY BANDWIDTH)**

The commands below are what *would* have transferred the three
1.7B checkpoints if the uplink could sustain more than 2.4 MB/s:

```bash
ssh $REMOTE 'mkdir -p ~/memory_residuals/Runs/chain_v15e_d4v2_1p7b_norm_local/best \
                      ~/memory_residuals/Runs/chain_v15f_d4v2_1p7b_jointtrain_local/best \
                      ~/memory_residuals/Runs/chain_v15f_d4v2_1p7b_jointtrain_local/step-500'

for CK in chain_v15e_d4v2_1p7b_norm_local/best \
          chain_v15f_d4v2_1p7b_jointtrain_local/best \
          chain_v15f_d4v2_1p7b_jointtrain_local/step-500; do
  rsync -a --partial --info=progress2 \
    -e "ssh -c aes128-gcm@openssh.com -o Compression=no" \
    $LOCAL/Runs/$CK/ $REMOTE:~/memory_residuals/Runs/$CK/
done
```

Observed throughput: 2.3 MB/s serial, 5.1 MB/s combined across 3
parallel streams. Aborted at 101 / 590 / 105 MB received.

**3) Template prior (ran successfully on GH200)**

```bash
ssh $REMOTE 'cd ~/memory_residuals && . ~/venv/bin/activate && \
             python tools/audit_a3_template_prior_cloud.py'
# → results/exp2_chain_recipe/audit_a3_template_prior_cloud.json
#   CE_template_prior (per-token mean, n=128): 2.9086 nats
```

**4) Per-ckpt CE (WOULD HAVE RUN — unused)**

```bash
# For each of {v15e_1p7B_FROZEN_best, v15f_1p7B_joint_best, v15f_1p7B_joint_step500}:
ssh $REMOTE 'cd ~/memory_residuals && . ~/venv/bin/activate && \
  python tools/audit_a3_ckpt_eval_cloud.py \
    --ckpt Runs/chain_v15f_d4v2_1p7b_jointtrain_local/best \
    --tag  v15f_1p7B_joint_best \
    --n 128'
# → results/exp2_chain_recipe/audit_a3_v15f_1p7B_joint_best_cloud.json
```

**5) Pull results back**

```bash
rsync -av $REMOTE:~/memory_residuals/results/exp2_chain_recipe/audit_a3_template_prior_cloud.json \
  $LOCAL/results/exp2_chain_recipe/
```

Raw artefacts (cloud-side, pulled to local):

* `results/exp2_chain_recipe/audit_a3_template_prior_cloud.json` —
  per-chain (`cat`, `item`, `n_callback_tok`, `prior_prob`,
  `ce_template_prior_per_token`, `ce_template_prior_item`) for all
  128 val chains; plus the `train_counts` / `support_size` tables
  used for §2.
* `results/exp2_chain_recipe/audit_a3_template_prior.pkl` (cloud-only;
  not synced back under the time budget).
* `tools/audit_a3_template_prior_cloud.py`,
  `tools/audit_a3_ckpt_eval_cloud.py` — cloud helpers (the second
  one is tested-compilable but not invoked).

## §6 Blockers & caveats

1. **Uplink bottleneck.** 2.4 MB/s from the lab box to GH200 is the
   fundamental constraint. Parallelising rsync streams did not help
   (aggregate bandwidth stayed at ~5 MB/s). To make this audit
   runnable on the cloud within the time budget, the fix is to keep
   the checkpoints pre-staged on the cloud side (e.g. S3 mirror or
   git-lfs on the cloud host), or to run the audit directly on the
   lab box's free GPU when one becomes available.
2. **The `pa_cb_ce_mem` values in §3 are NOT a cloud-side
   recomputation.** They come from local-A3's concurrent n=128 run.
   Because local-A3 ran to completion on GPU 1 while this branch
   waited on uplink, the parallelisation benefit of the cloud branch
   was not realised for the 1.7B slice. The cloud branch's
   *independent* contribution is limited to the CE_template_prior
   baseline (which matches local to 4+ decimals, so it is an
   independent replication).
3. **No per-token decomposition (§4 of the spec, "is " vs 1st item
   subword vs subsequent).** Skipped: without the ckpts on GH200 we
   cannot produce cloud per-token-CE. Local-A3 also skipped this
   (their §4 is "skipped under the 30-min wall-clock budget"). The
   helper `audit_a3_ckpt_eval_cloud.py` is written to emit
   `ce_first_item_tok` and `ce_subsequent_item_tok` when it does
   run; it is ready to invoke once checkpoints are on the cloud.
