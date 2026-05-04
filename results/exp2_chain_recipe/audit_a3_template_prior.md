## v15 OPEN AUDIT — A3 LOCAL: Template-prior baseline vs jointly-trained backbones

**Status (2026-05-03, 04:50 UTC-5):** complete. Local 0.6B + cloud-deferred 1.7B
checkpoints scored against the empirical Laplace-smoothed `P(item | category)`
prior built from `synthd4v2_persona_callback_train_s512.pt`.

### TL;DR

* The v15 OPEN AUDIT candidate leak #2 — *jointly trained backbones learn the
  closed-set per-category marginal directly inside the LM head and bypass the
  memory pathway* — **lands**. v15b (0.6B joint) and v15f (1.7B joint) both
  converge to a callback CE that sits within ≤0.4 nats of the template prior,
  while every FROZEN baseline sits 1.8–2.2 nats above the same prior.
* This is decisive evidence that the headline `pa_cb_dnm` numbers from the
  joint-trained cells **were dominated by the empirical D4v2 marginal**, not by
  the M_c → readout pathway. The architectural prior #9 ("memory is the only
  pathway that carries chain-specific bindings") is empirically false on D4v2
  for any cell with an unfrozen LM head.
* The 1.7B frozen baseline (v15e) is simultaneously the *worst* memoriser of
  the marginal (+1.85 nats above the prior) and the cell with the best
  frozen-pathway evidence_lift in the v15 ledger — i.e. its small `pa_cb_dnm`
  (≈ +0.01 nats) is delivered without prior fitting, but is also tiny.
* The v15a 0.6B frozen baseline sits 2.24 nats above the template prior —
  mean callback CE 5.15 nats — which is the empirical floor for "what a frozen
  Qwen3-0.6B's LM head plus α≈0.05 memory injection can do on D4v2 if it
  isn't allowed to fit the marginal." That floor matters because the v16
  evidence-lift target lives entirely within the gap between this floor and
  what `M_c`-based retrieval could in principle deliver.

### §1 Setup

| field | value |
|---|---|
| corpus | synthd4v2_persona_callback (8 categories × 32 items, closed set) |
| n val chains scored | 128 |
| smoothing | Laplace α = 0.5 |
| token assignment | full per-token CE on each item subword (matches `audit_base_prior.py`) |
| machine | Lab-box GPU 1 (H100 NVL); 0.6B + 1.7B ckpts ran locally without rsync once the cloud upload stalled. |

Code: `memory_residuals/tools/audit_a3_template_prior.py`. Raw per-chain CEs:
`memory_residuals/results/exp2_chain_recipe/audit_a3_data.json`.

### §2 Template prior construction

Train counts per category sum to 593–652 (≈ 4900 chains). Most-likely items per
category after Laplace smoothing:

```
color: purple/maroon/turquoise (4.1%)
fruit: blackberry/coconut/avocado (4.6%)
animal: rabbit (4.8%) > koala/cheetah (4.3-4.5%)
object: pillow (4.3%) > bucket (4.1%)
sport: soccer (5.2%) > boxing (4.9%) > hockey (4.4%)
tool: stapler (5.1%) > tape (4.3%) > scale/whisk (4.1%)
instrument: ocarina (4.4%) > violin (4.3%) > oboe (4.1%)
hobby: filmmaking (5.2%) > knitting/calligraphy (4.7%) > collecting (4.3%)
```

`CE_template_prior(val) = 2.9087 nats` (n=128 val chains, weighted by per-chain
answer-token count; raw per-chain values in `audit_a3_data.json`).

### §3 Headline table

| ckpt | scale | backbone | step | `pa_cb_ce_mem` | `CE_template_prior` | gap = mem − prior |
|---|---|---|---:|---:|---:|---:|
| v15a 0.6B FROZEN best | 0.6B | frozen | 1700 | 5.149 | 2.909 | **+2.240** |
| v15b 0.6B joint best  | 0.6B | joint  | best | 2.966 | 2.909 | **+0.057** |
| v15b 0.6B joint final | 0.6B | joint  | 1500 | 2.939 | 2.909 | **+0.031** |
| v15e 1.7B FROZEN best | 1.7B | frozen | 700  | 4.763 | 2.909 | **+1.854** |
| v15f 1.7B joint best  | 1.7B | joint  | 500  | 3.207 | 2.909 | **+0.298** |
| v15f 1.7B joint s500  | 1.7B | joint  | 500  | 3.278 | 2.909 | **+0.369** |

(`pa_cb_ce_mem` = the trainer's own scored callback CE on n=128 val chains
with each chain's true `M_c` injected; values rerun on GPU 1 through the same
`audit_base_prior.py` scoring loop, not pulled from logs, to keep the
comparison rigid.)

### §4 Verdict

**The 1.7B joint backbones (v15f) hit the template prior at 0.30 nats above
the floor.** Their `pa_cb_dnm` headline numbers (+1.4, +0.7 in the ledger) are
delivered by *the LM head having internalised the empirical
P(item|category)*, not by the memory pathway. The 0.6B joint cell (v15b) is
even tighter — 0.03–0.06 nats above the prior, indistinguishable from
"the LM head learned the prior outright." Neither cell can be cited as
evidence the architecture works; their wins are corpus-shaped, not
retrieval-shaped.

**The FROZEN baselines (v15a, v15e) sit 1.8–2.2 nats above the same prior.**
Frozen Qwen3 has not been fine-tuned to fit the closed-set marginal, so the
template prior is *unreachable* via direct LM-head supervision in those cells.
That makes them the only architecturally honest cells in v14g..v15e — and the
ones whose `evidence_lift` numbers (v15a +0.07, v15e ≈ 0) are the *only*
honest signal in the v15 ledger.

**Resulting reframe of the v15 OPEN AUDIT.** Of the five candidate leaks
A1..A5 listed in `runs.md`, A2 (template-prior absorption by the LM head) is
no longer hypothesis-grade — it is the dominant driver of the headline number
in every joint-trained cell. The other four (A1 Δ_step, A3 base-prior gap, A4
shuffle-vs-mem, A5 readout-bypass) become *post-A2* questions: "now that we
know joint cells are running on the marginal, what is the FROZEN cell's tiny
positive lift coming from?"

The v16 design (random-codes corpus + evidence-mask + evidence_lift
best-metric) is the right response to A2: it strips the marginal so this
audit's gap-to-prior plot collapses by construction. **But v16a's first 600
steps are showing the writer can't store binding even when the marginal is
unavailable** — i.e. the corpus fix exposed a deeper writer/readout
degeneracy that A2 was masking. That is now the load-bearing question for
the v17 family (parent agent is queueing the experiments below).

### §5 Reproduction

```bash
cd /home/exx/Desktop/fine-tune
CUDA_VISIBLE_DEVICES=1 python memory_residuals/tools/audit_a3_template_prior.py \
  --n_chains 128 --alpha 0.5 \
  --out_json memory_residuals/results/exp2_chain_recipe/audit_a3_data.json \
  --ckpts \
    memory_residuals/Runs/chain_v15a_d4v2_norm_replicate_local/best \
    memory_residuals/Runs/chain_v15b_d4v2_norm_jointtrain_local/best \
    memory_residuals/Runs/chain_v15b_d4v2_norm_jointtrain_local/final \
    memory_residuals/Runs/chain_v15e_d4v2_1p7b_norm_local/best \
    memory_residuals/Runs/chain_v15f_d4v2_1p7b_jointtrain_local/best \
    memory_residuals/Runs/chain_v15f_d4v2_1p7b_jointtrain_local/step-500 \
  --ckpt_tags \
    v15a_0p6B_FROZEN_best v15b_0p6B_joint_best v15b_0p6B_joint_final \
    v15e_1p7B_FROZEN_best v15f_1p7B_joint_best v15f_1p7B_joint_step500
```

Wall-clock 1m50s on H100 NVL. The 1.7B-on-cloud rsync (separate parallel A3
deliverable) was abandoned at ~10–15% upload due to network instability with
the GH200; the local box was fast enough that the cloud branch wasn't needed.
