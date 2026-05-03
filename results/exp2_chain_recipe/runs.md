# Runs ledger — experiment 2 (long-horizon recipe)

Single source of truth for which checkpoint produced which number in
the recipe paper. Active state only. Newest at the top. Everything
KILLED, SUPERSEDED, or whose mechanism has been promoted to the
README's "Architectural priors" block has been folded into
[`archive/COMPREHENSIVE.md`](../../archive/COMPREHENSIVE.md) Part VI
(v11 → v14) per the folding convention in Part VII of that file.

---

## v15 — extract_input_norm + double-evidence D4v2 (active; 2026-05-02 → present)

### Pre-summary (TL;DR for the next reader)

Rebuilt the synthetic corpus to **two evidence sessions per chain**
(`synthd4v2_persona_callback_*_s512.pt` via
`tools/build_synthetic_persona_callback.py --n_evidence_sessions 2`),
landed `--memres_extract_input_norm` (RMSNorm on the context tensor
`C` before `MemoryBlock.extract`), removed `writer_warmup` entirely,
and ran the v14g..l + v15a..f matrix. The numerical headline has
shifted **twice** since the v14abl post-mortem:

* **Flip 1 — eval-tooling artefact (v14k @ 0.6B, frozen).**
  `eval_chain.py` averages CE over the entire score window
  (`score_tail_frac=1.0` ⇒ all 4 sessions), so a localised
  callback-token effect is diluted ~38× into noise. Built
  `tools/eval_callback.py` (mirrors the in-trainer `pa_cb_*` metric:
  callback-token positions only, evidence-redacted memory floor for
  the baseline). On v14k_best: `pa_cb_dnm = +1.44`,
  `evidence_lift = +0.071`. ✅ confirms in-trainer number is real.

* **Flip 2 — concerning (v15e @ 1.7B, frozen, norm ON).**
  `Δnm-m_floor = +2.5 nat` (memory provides 2.5 nat help vs no
  memory) but `evidence_lift` swings *negative*: `−0.10 to −0.27` for
  the bulk of training, briefly positive at step 200–300
  (+0.07 to +0.31) before the writer overfits. **Evidence-redacted
  memory beats full memory.** The 1.7B writer is using non-evidence
  content (filler embeddings, persona-prior, callback-template) to
  "pre-route" the callback answer; adding the actual evidence into
  M_c during the evidence sessions disrupts that prior.

* **Flip 3 — fundamental setup bug, audit in progress (v15b @ 0.6B,
  joint train).** Both `Δnm-m_floor` and `evidence_lift` collapse
  to ≈ 0 for the full 4000-step run with `lr_backbone=2e-5`. The
  unfrozen 0.6B backbone learns the callback distribution **directly**
  — which should not be possible if memory is the only pathway
  carrying evidence. **The user has flagged this as a fundamental
  setup bug, not an ablation question** (2026-05-03 ~02:30 UTC-5).
  See "v15 OPEN AUDIT" below; v15g is postponed pending audit.

### v15 cells launched (2026-05-02 → 2026-05-03)

| cell | machine | preset | backbone | window_k | step | result | location |
|---|---|---|---|---:|---:|---|---|
| **v14g** | local H100 GPU 0 | qwen3-0.6b | FROZEN | 4 | 2500 | norm ON, warmup 200 → mid Δsh-m, ~0 evidence_lift | `output/chain_v14g_d4v2_warmup_norm_local/best` |
| **v14h** | local H100 GPU 1 | qwen3-0.6b | FROZEN | 4 | 2500 | norm OFF, warmup 200 → ditto, slightly worse | `output/chain_v14h_d4v2_warmup_nonorm_local/best` |
| **v14i** | GH200 | qwen3-0.6b | FROZEN | 4 | 2500 | warmup_router_bias 8.0 → recent-bias lock-in, neg lift | `(GH200) output/chain_v14i_d4v2_strongwarmup_gh200/best` |
| **v14j** | local H100 GPU 0 | qwen3-0.6b | FROZEN | 4 | 2500 | warmup 0, norm ON, slot writer → mid lift (+0.04) | `output/chain_v14j_d4v2_nowarmup_slot_local/best` |
| **v14k** | local H100 GPU 1 | qwen3-0.6b | FROZEN | 4 | 2500 | warmup 0, norm ON, slot writer, alpha-floor + InfoNCE → **+0.071 evidence_lift** ✅ | `output/chain_v14k_d4v2_nowarmup_slot_floor_local/best` |
| **v14l** | GH200 | qwen3-0.6b | FROZEN | 4 | 2500 | as v14k but writer=cross_attention (no slot) → similar lift, slightly noisier | `(GH200) output/chain_v14l_d4v2_nowarmup_xattn_floor_gh200/best` |
| **v15a** | local H100 GPU 0 | qwen3-0.6b | FROZEN | 4 | 2500 | replicate v14k recipe → `pa_cb_dnm +1.33, dsh +0.02` ✅ reproducible | `output/chain_v15a_d4v2_norm_replicate_local/best` |
| **v15b** | local H100 GPU 1 | qwen3-0.6b | **trained** (lr_b=2e-5) | 4 | 4000 | joint training → `evidence_lift ≈ 0` for the full run; **THIS is the bug** | `output/chain_v15b_d4v2_norm_jointtrain_local/best` |
| **v15c** | GH200 | qwen3-0.6b | FROZEN | 4 | 2500 | extract source = `embed` instead of `hidden_14` → `evidence_lift +0.005-+0.02`; hidden_14 is better | `(GH200) output/chain_v15c_d4v2_norm_extract_embed_gh200/best` |
| **v15e** | local H100 GPU 0 | qwen3-1.7b-large | FROZEN | 4 | 2000 | 1.7B frozen, norm ON → `Δnm-m_floor +2.5` but `evidence_lift -0.18` (writer overfits) | `output/chain_v15e_d4v2_1p7b_norm_local/best` |
| **v15f** | local H100 GPU 0 | qwen3-1.7b-large | trained (lr_b=2e-5) | 4 | 3000 | 1.7B joint train (started 2026-05-03 02:17 UTC-5; RUNNING ~step 100) | `output/chain_v15f_d4v2_1p7b_jointtrain_local/{step-N,best,final}` |

Launchers live at `Scripts/train_v14g..l_*.sh` and
`Scripts/train_v15{a,b,c,e,f}_*.sh`. The local sequencers are
`Scripts/orchestrate_gpu0_v15_wave.sh` and
`Scripts/orchestrate_gpu1_v15_wave.sh`.

### v15 code shipped

* `src/modeling_memres.py`:
  * `Qwen3MemResConfig.memres_extract_input_norm: bool` — when true,
    `MemoryBlock.__init__` constructs
    `self.extract_input_norm = Qwen3RMSNorm(d, eps=...)` and
    `MemoryBlock.extract` wraps `C` through it before the
    cross-attn / slot-attn path. Diagnosed as the dominant root
    cause of `M_new` norm explosions (~50× backward grads on `W_Q`).
  * `MemoryBlock.forward` write_gate sigmoid is **bypassed entirely**
    for `writer_kind ∈ {slot_attention, slot_attention_full}`
    (Locatello GRUCell already gates per-slot; stacking the external
    sigmoid saturated within 50 steps and locked M_c at zero).
  * For `writer_kind=original`, the `gate_input` is RMSNormed on
    each side before the sigmoid (`write_gate_norm_prev`,
    `write_gate_norm_new`). Verified: feeding `‖C‖₂ = 100·N(0,1)`
    drops post-RMSNorm `gate_input` magnitude to ~0.78 and the
    sigmoid sits at the intended 0.27 init.
* `src/train_chain.py`:
  * `--memres_extract_input_norm` flag forwarded through
    `memres_kwargs` into the config override.
  * `--kill_on_memory_collapse` — loud-halt guardrail in the eval
    loop. Two consecutive evals (after
    `--kill_on_memory_collapse_min_step=200`) with
    `Mc_pair_to_self_ratio < 0.01` *or* `mt_norm_ratio_mean < 0.01`
    aborts the run with exit code 42 and persists a forensic
    `killed-step-N` checkpoint. Converts the silent-failure burn
    (v13/v14/v15a-style) into a loud halt that
    cloud_watchdog / CI pick up.
* `src/presets.py`: added `qwen3-1.7b-small` (L_E=0) and
  `qwen3-1.7b-large` (L_E=4); both `memres_num_vectors=128`,
  `memres_num_blocks=8`.
* `tools/build_synthetic_persona_callback.py`: new
  `--n_evidence_sessions` (default **2**) and `--n_prefix_sessions`
  flags. Multi-evidence forces the writer to integrate two
  distinct-category facts before the readout discriminates which one
  the callback queries — removes the trivial single-evidence "keep
  everything" optimum that v13/v14 D4 had.
* `tools/eval_callback.py` (new): standalone callback-aware eval
  that mirrors the in-trainer `pa_cb_*` metric — scores only
  callback-mask token positions in the callback session, with the
  floor baseline using an evidence-redacted memory state. **Use this
  for D4v2 post-train eval; `eval_chain.py` averages over the score
  window and dilutes localised callback effects ~38×.**
* `tools/locomo_to_chains.py` + `tools/pretokenize_chains.py`: built
  `paper_artifacts/chains/locomo_s512.pt` (10 conversations,
  272 sessions) and `paper_artifacts/chains/msc_test_s512.pt`
  (500 chains, 2500 sessions) for cross-domain transfer eval.
* `Scripts/eval_v14_v15_benchmarks.sh`: wraps `eval_chain.py` for
  D4v2-val + LoCoMo + MSC-test sweeps.
* Stale duplicate `train_chain.py` and `modeling_memres.py` at the
  repo root were deleted; only `src/*.py` remain. The roots were
  never updated past v11 and would crash with "unrecognized
  arguments" if any v12+ launcher accidentally pointed at them.

### v15 OPEN AUDIT (2026-05-03 ~02:30 UTC-5) — the user-flagged fundamental issue

**Concern.** In v15b (and v15f), an unfrozen backbone with
`lr_backbone = 2e-5` collapses both `Δnm-m_floor` and `evidence_lift`
to ~0. This should be impossible in a well-designed setup: callback
tokens should be unpredictable from anything *except* the memory
pathway (since the evidence sessions live outside the LM-attended
window). If the backbone CAN learn the callback distribution
directly, that means the non-memory pathway is leaking
answer-bearing information.

Five candidate leaks under audit:

1. **Window leakage** — at `window_k=3` the LM attends to the last 3
   sessions of the chain. Evidence is placed at random body
   positions, sometimes inside the last 3 ⇒ direct (non-memory)
   access. The trainer's `score_tail_frac=1.0` makes this leak
   *training-active* on every window where evidence happens to land
   in the tail. Quantitative test: count chains where any evidence
   position is in `[chain_callback_position − window_k + 1,
   chain_callback_position]`.
2. **Template / prior leakage** — callback session text always reads
   `Assistant: Your favorite {category} is {item}.` The "is" is one
   token before the answer. A 1.7B model trained on enough chains
   can learn `Your favorite color is` → marginal over the 32 colors,
   pushing CE down even without evidence. Closed-set has 256 items
   in 8 categories ⇒ ceiling guess CE = log(32) = 3.5 nats; with
   persona priors over the dataset (e.g. "red" appears more often),
   the prior could push CE down meaningfully.
3. **Same-chain evidence visibility via the rolling memory state** —
   the evidence-redacted "floor" baseline in `_phase_aligned_eval`
   may not actually be removing information from M_c if the
   redaction operates on a rolling state that already integrated the
   evidence in earlier sessions of the same window.
4. **Cross-window state leakage** — BPTT / detach boundaries between
   chain windows; if the backbone's hidden state at window boundary
   carries any cross-session signal, that's another non-memory
   pathway.
5. **Tokeniser-level leakage** — closed-set items are 1-3 BPE
   tokens. If the *first* token of an item is unique enough (e.g.
   "stra" → only "strawberry"), and the prior token is "is " plus
   the question's category cue, then BPE structure makes the answer
   guessable.

Three independent auditors spawned in parallel (`Task` subagents,
results to be linked here once complete):

* **A1** — corpus-and-window leakage audit: enumerate D4v2-train
  chains, count evidence-in-window collisions for `k=3,4`, measure
  CE-on-callback for a frozen base model with no memory at all
  (i.e. the upper bound on what the backbone can learn directly).
* **A2** — eval-redaction audit: trace `_phase_aligned_eval` /
  `eval_callback.py` redaction logic; verify that the "floor"
  baseline truly excludes evidence from M_c at the moment the
  callback is scored. Adversarial test: shuffle evidence to a
  different chain at score time and check CE doesn't change.
* **A3** — base-rate / template prior audit: for the v15b/v15f joint
  runs, decompose CE into prior + memory contributions; show
  whether the unfrozen backbone's callback CE matches a pure
  category-prior baseline.

Pending audit results before any further training cells fire. v15g
(was queued as 1.7B-small on GPU 1) has been **postponed** until
A1/A2/A3 land.

---

## v11 → v14 — folded into [`archive/COMPREHENSIVE.md`](../../archive/COMPREHENSIVE.md) Part VI

The v11 (g/h/i/j/k/l-fix/m/p/q/r), v12 (slot-attention writer; D4
retarget), v13 (writer_warmup + orth + slot_positional + the
config-merge bugfix; v13c2/v13r/v13q), and v14 (judge_qk_layernorm +
alpha_mem_floor + InfoNCE + AP warmup anneal; v14abl_a/b, v14a,
v14g..l) ledger entries have all been folded into
`archive/COMPREHENSIVE.md` Part VI on 2026-05-03 per the folding
convention in Part VII.

Last-eval / verdict summary (full tables + result trajectories +
inline launch flags in Part VI):

| campaign | verdict | leading checkpoint at end of campaign |
|---|---|---|
| **v11** (P0+P2+P3 fix; cells g/h/i/j/k/l-fix/m/p/q/r) | writer is content-blind under LM-only objective; chain-identity hash under InfoNCE; D5 audit on v11g/best identifies the readout as bottleneck | `Runs/chain_v11g_ap_baseline_gh200/best` step 600 (PA CB Δ_nm-m=+0.030, decayed by 1400) |
| **v12** (slot-attention writer) | hits the same uniform fixed point at step 800; slot-attention alone is necessary but not sufficient — GRU shares weights across slots, symmetry re-emerges | `Runs/chain_v12a_slot_judge_d4_local/step-200` only (peak before collapse) |
| **v13** (writer_warmup + orth + slot_positional + config-merge bugfix) | symmetry permanently broken (D3-MC pair/self = 0.004 sustained); v13c2 hit `evidence_lift +1.4` mid-warmup; phase-2 backbone unfreeze destroys writer specialisation | `Runs/chain_v13c_d4_ap_gh200/best` step 600 (warmup peak only) |
| **v14** (judge_qk_layernorm + alpha_mem_floor + InfoNCE + AP anneal; D4v2) | judge_qk_ln × slot_attention is anti-causal (writer never lifts off zero); without QK-LN, writer specialises but Δ_nm-m goes negative — InfoNCE satisfies itself with non-LM-useful chain-distinguishable M_c. v14k @ FROZEN backbone is the first reproducible positive: `evidence_lift +0.071` | `Runs/chain_v14k_d4v2_nowarmup_slot_floor_local/best` |

---

## v10 → v6 — folded into [`archive/COMPREHENSIVE.md`](../../archive/COMPREHENSIVE.md) Parts IV–V

Same convention; older entries previously folded.

---

## How to add a new run

1. Create / pick a `Scripts/train_*.sh` for the run. The script's
   header MUST document what it varies vs the closest existing run.
2. Pre-allocate the `chain_<version>_<descriptor>` run name in this
   ledger as `Status: NOT STARTED` with the script path filled in.
3. Launch (locally in tmux or via the cloud watchdog
   `enqueue.sh`).
4. Update Status to TRAINING with start timestamp + machine.
5. On finish / kill / failure, update Status with end timestamp,
   reason, last step, last `Δ_sh-m`, location of the `best/` ckpt,
   and link to the log.
6. When the post-train pipeline runs, link the eval JSONs.

Do NOT delete entries when a run is superseded. Mark them KILLED /
SUPERSEDED-BY and keep the paper trail. When the active ledger gets
crowded, fold superseded entries into `archive/COMPREHENSIVE.md`
Part VI per the convention in Part VII of that file.

## Conventions

- Run names: `chain_v<N>_<descriptor>` where `<N>` increments on a
  major recipe / architectural change.
- Tmux naming: cloud watchdog uses `cwd-<run_name>`; local launches
  use `local-<run_name>`.
- Log paths: cloud → `~/memory_residuals/tools/cloud_watchdog/logs/<run_name>.log`;
  local → `logs/<run_name>.log`.
- Output dirs: always `Runs/<run_name>/{step-N, best/}`.
