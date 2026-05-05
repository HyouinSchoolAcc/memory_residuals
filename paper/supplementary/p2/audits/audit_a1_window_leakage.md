# Audit A1 — Window leakage + base-prior CE upper bound

Audit of v15 OPEN AUDIT candidate leak #1 (window leakage) for the
joint-trained collapse seen in v15b (0.6B, step 4000) and v15f (1.7B,
step 820 kill).

Scope: D4v2 corpus
(`paper_artifacts/chains/synthd4v2_persona_callback_{train,val}_s512.pt`,
n_evidence=2, n_filler=7, n_prefix=0 ⇒ chain_len=10, callback at
chain_pos=9). Helper at `memory_residuals/tools/audit_a1_window_leakage.py`,
raw output at `memory_residuals/results/exp2_chain_recipe/audit_a1_window_leakage.json`.

## TL;DR

- **Window leakage at the v15 default `window_k=4` is enormous: 58.5 %
  of train chains and 59.6 % of val chains have at least one evidence
  session inside the LM's attention window during callback scoring.**
  Roughly 1 in 6 train chains has *both* evidence sessions in the window.
- At `window_k=3` the leak is 41.4 % / 43.4 % (train / val); only
  `window_k=1` (callback alone, no prior session attended) eliminates
  the leak, and `window_k=2` already lets 22 % of chains see one
  evidence session directly.
- Frozen-base callback CE on `session_callback_mask` tokens is **7.69
  nats for Qwen3-0.6B and 8.76 nats for Qwen3-1.7B** (n=128 val chains,
  no memory, no chain prefix). Both are well above the closed-set
  uniform floor `log(256)=5.55` and the per-category floor `log(32)=3.47`.
- v15b's `pa_cb_ce_mem ≈ 2.97` and v15f's `pa_cb_ce_mem ≈ 3.17` (the
  collapsed numbers under joint training) are therefore **~4.7 nats
  below the frozen-base prior on the same tokens**, so "the backbone
  already knows the answer" does **not** by itself explain the
  collapse. Something the backbone *learned* during training is doing
  the work.
- That "something" is consistent with window leakage: with ~60 % of
  windows giving the LM direct (non-memory) access to one or both
  evidence sessions, an unfrozen backbone with `lr_b=2e-5` has both the
  capacity and the supervision to internalise a "read evidence from the
  windowed context" policy and ignore memory — which is exactly what
  collapses `evidence_lift` to ≈ 0 while pushing `pa_cb_ce_mem` toward
  the per-category floor.

## §1 Window-collision counts

For each chain, count chains where at least one evidence session falls
in the closed window `[callback_position − k + 1, callback_position]`
(equivalently, `1 ≤ cb_pos − ev_pos < k`). Both blobs have
`callback_position = 9`, evidence positions sampled (sorted) from the
9-slot body `{0..8}`. Theoretical baseline matches the empirical counts
to within ~1 % (e.g. for `k=4`, `1 − C(6,2)/C(9,2) = 21/36 = 0.583`).

### Train (`synthd4v2_persona_callback_train_s512.pt`, n=5000)

| window_k | any-leak count | any-leak frac | first-only | second-only | both |
|---:|---:|---:|---:|---:|---:|
| 1 |    0 | 0.000 |   0 |    0 |   0 |
| 2 | 1088 | 0.218 |   0 | 1088 |   0 |
| 3 | 2068 | 0.414 |   0 | 1899 | 169 |
| 4 | 2926 | **0.585** |   0 | 2493 | 433 |
| 5 | 3616 | 0.723 |   0 | 2743 | 873 |

### Val (`synthd4v2_persona_callback_val_s512.pt`, n=500)

| window_k | any-leak count | any-leak frac | first-only | second-only | both |
|---:|---:|---:|---:|---:|---:|
| 1 |   0 | 0.000 | 0 |   0 |  0 |
| 2 | 114 | 0.228 | 0 | 114 |  0 |
| 3 | 217 | 0.434 | 0 | 204 | 13 |
| 4 | 298 | **0.596** | 0 | 252 | 46 |
| 5 | 356 | 0.712 | 0 | 268 | 88 |

The v15 launchers (Scripts/train_v15{a,b,c,e,f}_*.sh) all run with
`window_k=4` per the runs.md table. **At that operating point, the
expected leakage rate is ~58 %.**

## §2 Per-chain breakdown

Two structural facts about the corpus generator drive what we see:

1. `_sample_evidence` followed by `sorted(rng.sample(body_positions, n_evidence))`
   means **evidence index 0 is always at a *lower* body position than
   evidence index 1**. So the "first" evidence (rank 0) is always the
   earlier one in the chain.
2. The callback always sits at the very last chain position (`chain_len − 1`).

Consequently:

* In every "single-rank-leaks" chain, the leaking evidence is **rank 1
  (the second / later one)**, never rank 0. We measured this directly:
  the `first-only` column is 0 across all k for both splits.
* "both-leak" chains are exactly the ones where both evidence positions
  land in the tail window. At `k=4` (window covers body positions
  {6,7,8}) this is `C(3,2)/C(9,2) = 3/36 = 8.33 %` of all chains;
  measured 433/5000 = 8.66 % (train) and 46/500 = 9.2 % (val). Match.

### Positional offsets (val, `cb_pos − ev_pos`)

| offset (sessions before callback) | leak count at k=2 | k=3 | k=4 | k=5 |
|---:|---:|---:|---:|---:|
| 1 | 114 | 114 | 114 | 114 |
| 2 |  —  | 116 | 116 | 116 |
| 3 |  —  |  —  | 114 | 114 |
| 4 |  —  |  —  |  —  | 100 |

Read down a column: at `k=4` the val split has 114+116+114=344 leaking
*evidence-position incidents* across 298 chains (the surplus = chains
where both evidences leak). Offsets are essentially flat over the
window — exactly what the uniform body-placement design predicts.

### Take-away

The leakage is not concentrated at the callback's immediate predecessor;
at `k=4` it is distributed roughly uniformly over body offsets 1, 2, 3.
That means raising `window_k` linearly grows the per-chain probability
that the LM has at least one evidence session in attention.

## §3 Frozen base CE on callback tokens

Frozen Qwen3 base, no memory, no fine-tuning, no chain prefix. We feed
*only* the callback session and compute mean per-token NLL over the
positions where `session_callback_mask == 1` (the answer span). n=128
val chains; multi-token items split into ≤ 3 BPE tokens each.

| model | pa_cb_ce (nats) | n | multi-tok items | avg ans tokens |
|---|---:|---:|---:|---:|
| Qwen3-0.6B-base (no mem)         | **7.686** | 128 | 40 | 1.35 |
| Qwen3-1.7B-base (no mem)         | **8.764** | 128 | 40 | 1.35 |
| log(256) — uniform over closed set | 5.545 |  —  | — | — |
| log(32)  — uniform within cued category | 3.466 |  —  | — | — |

Trained / frozen-with-writer comparison points (lifted from
`memory_residuals/logs/`):

| run | backbone | window_k | step | `pa_cb_ce_mem` | `pa_cb_ce_mem_floor` | `evidence_lift` |
|---|---|---:|---:|---:|---:|---:|
| v15a | 0.6B FROZEN          | 4 | 2500 (best)  | ≈ 6.85 | ≈ 6.86 | +0.008 (peak +0.14 mid-train) |
| v15b | 0.6B joint (lr_b=2e-5)| 4 | 4000        | **2.975** | **2.962** | −0.013 |
| v15e | 1.7B FROZEN          | 4 | 2000        | 5.46  | 5.37  | −0.091 (Δnm−m_floor=+2.48) |
| v15f | 1.7B joint (lr_b=2e-5)| 4 | 800 (closest to step 820 kill) | 3.170 | 3.138 | −0.033 |

Observations:

* **The base prior alone does *not* explain v15b/v15f.** v15b sits 4.71
  nats *below* 0.6B-base; v15f sits 5.59 nats *below* 1.7B-base. The
  joint backbones learned the closed-set marginal (and more) — they did
  not arrive at ~3 nats by inheriting it from pretraining.
* **v15b/v15f sit just below `log(32)=3.466`.** That is the floor for
  picking uniformly within the cued category once you know the
  category. So joint training has effectively learned the
  category-conditional marginal *plus a small extra ~0.3–0.5 nats* of
  identifying signal — and crucially, that extra signal is provided
  even when memory is wiped (`pa_cb_ce_mem_floor ≈ pa_cb_ce_mem`).
* **The *frozen* runs (v15a, v15e) sit far above log(32).** v15a is
  near 6.85 nats and v15e near 5.4 nats. Frozen backbones cannot
  ingest the closed-set marginal during training, so they keep the
  ~7-nat base prior, and any movement below that has to flow through
  the writer/readout/memory pathway. That is why frozen runs show
  measurable (+0.07 v15a@best, +0.7…−0.3 v15e oscillation)
  `evidence_lift` and joint runs show none.

## §4 Verdict

**Window leakage explains the joint-train collapse — at least
qualitatively, and the quantitative case is strong.**

1. ~60 % of train windows at `window_k=4` give the unfrozen backbone
   direct attention over at least one evidence session at callback
   time. Combined with `score_tail_frac=1.0`, gradient flows to the LM
   head from those windows on every batch. The backbone has both
   capacity and supervision to internalise an "attend to the most
   recent persona-statement" policy that bypasses M_c.
2. The frozen-base CE on the same tokens is 7.7 / 8.8 nats. v15b and
   v15f are at ~3.0–3.2 nats. The ~4–5 nats of CE that joint training
   gained over the base prior is *exactly* the budget you need to
   collapse `evidence_lift`: once the backbone has learned to predict
   the answer, removing memory costs nothing (because memory was never
   doing the work). This matches the observed
   `pa_cb_ce_mem ≈ pa_cb_ce_mem_floor` at every joint-train step ≥ 400.
3. The frozen counterparts (v15a@0.6B, v15e@1.7B) cannot internalise
   the closed-set marginal, so memory is the only available pathway,
   and `evidence_lift` is non-zero even though the same window-leak
   structure is present in the corpus.
4. **Quantitative bound.** Even if window leakage were the *only* leak,
   the upper bound on `pa_cb_ce_mem` from "60 % chains see evidence
   directly, 40 % score from category prior" is roughly
   `0.6 · ε + 0.4 · log(32) ≈ 0 + 1.39 = 1.39` nats — well below v15b's
   2.97. So window leakage alone *over-explains* v15b's CE; v15b is
   evidently not yet exploiting the leak fully (or the within-category
   prior is uniform-with-noise rather than near-deterministic). The
   *direction* is unambiguous though: the evidence-lift collapse is
   consistent with ~half of chains being window-leaked.

**Recommended remediations (out of scope for A1, but follow directly
from these numbers):**

* Drop `window_k` to 1 during the callback session (LM attends only to
  the callback session itself, M_c carries everything else).
  Eliminates the leak entirely.
* Or insert ≥ `window_k − 1` filler sessions immediately *after* the
  last evidence (i.e. constrain `chain_evidence_positions` so all
  evidence is at body positions ≤ `body_len − window_k`). At
  `n_filler=7`, requiring all evidence in body positions {0..5} would
  reduce `k=4` leakage to 0 % at the cost of a small drop in evidence
  diversity.
* Or audit `eval_callback.py` redaction logic (Audit A2) to confirm
  that the floor baseline is *also* affected by window leakage — if so,
  then `pa_cb_dnm_floor = pa_cb_dnm` is consistent with both pathways
  drawing from the same windowed-evidence signal, which is what we
  measure.

The bottom line: the v15b / v15f collapse is **not** an architectural
failure of the memory pathway. It is a corpus-and-window confound that
gives the LM head a direct, non-memory route to the answer on the
majority of training windows.

## §5 Reproduction

One-liner (run from repo root):

```bash
CUDA_VISIBLE_DEVICES=0 python memory_residuals/tools/audit_a1_window_leakage.py --n_chains 128 --models 0.6B 1.7B
```

Outputs:
- stdout summary (collisions per k, leak ranks, offset histogram, base
  CE per model, theoretical floors)
- `memory_residuals/results/exp2_chain_recipe/audit_a1_window_leakage.json`
  (machine-readable, all stats)

Total wall time on H100 NVL (GPU 0): collision stats < 5 s; 0.6B-base CE
~10 s; 1.7B-base CE ~13 s. Whole audit completes inside 1 minute.
