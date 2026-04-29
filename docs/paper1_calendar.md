# Paper 1 � 6-Day Calendar

## Claim

> *On multi-session dialogue benchmarks (LoCoMo, MSC test, NIAH,
> RULER), at matched compute, Memory Residuals with the soft-parity
> Block AttnRes routing pool reach lower NLL and higher callback-EM
> than (a) the no-memory baseline, (b) BM25 retrieval, (c) dense
> MiniLM retrieval, and (d) a fine-tuned dense retriever.*

## Resources

| asset | location | hours / day | notes |
|---|---|---:|---|
| local GPU 0 | `cuda:0` (H100 NVL 96 GB) | 16 | residential power-down 8 h |
| local GPU 1 | `cuda:1` (H100 NVL 96 GB) | 16 | residential power-down 8 h |
| cloud GPU   | `192.222.50.225` (GH200 480GB) | 24 | $2.49/h � 96 h ? $240, $260 reserve |

## Day-by-day

### Day 0 (today)

**Local.** `chain_v3_softparity_full` and `chain_v3_attentionbase_full`
running in parallel. Both at step ~2000 / 6000. ETA ~7 h. They
extend the v2 ablation matrix to step 6000 with cleaner hyperparameters.

**Cloud.** Environment up; v2 phaseA softparity ckpt synced; watchdog
+ heartbeat running; first three jobs done (NIAH grid, two
bootstrap CIs).

**Decision triggers.**

- If v3 softparity finishes with `?_sh-m > +0.020` on PG-19 standalone
  eval ? adopt as Paper 1's "PG-19 reference" headline number.
- If v3 attentionbase reaches `?_sh-m > +0.010` ? keep the third row
  of the ablation table; the parity-init contribution is real but
  bounded.

### Day 1

**Local GPU 0.** **E1 � Mixed-corpus softparity** (PG-19 40% + TV 20% +
MSC 40%). Same hyperparameters as v3_softparity_full. 10 h.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train_chain.py \
  --preset qwen3-0.6b-large --memres_mode attention_parity \
  --router_mem_bias_init -4.0 --router_recent_bias_init 4.0 \
  --window_k 8 --session_len 512 \
  --steps 6000 --batch_size 4 --grad_accum 4 \
  --warmup 200 --lr 3e-4 --lr_backbone 3e-5 \
  --train_chains paper_artifacts/chains/mixed_train_s512.pt \
  --eval_chains  paper_artifacts/chains/locomo_s512.pt \
  --gradient_checkpointing --eval_every 200 --save_every 1000 --log_every 25 \
  --eval_n_chains 24 --eval_window 4 --save_best_metric composite \
  --out_dir output/chain_mixed_softparity --seed 42 \
  > logs/chain_mixed_softparity.log 2>&1 &
```

**Local GPU 1.** **E2 � Mixed-corpus simple_gate** baseline. Same
recipe, just `--memres_mode simple_gate`. 10 h.

**Cloud.** While locals churn:
- Run NIAH on v3 softparity best/ + v3 attentionbase best/ as soon as
  they sync (overnight).
- Build callback EM eval harness (LoCoMo, generation quality on the
  100-callback subset).
- Pre-fetch HuggingFace assets needed for RAG baselines:
  `sentence-transformers/all-MiniLM-L6-v2`, `facebook/contriever`.

**Decision triggers.**

- Pre-Day-1 morning, must build `paper_artifacts/chains/mixed_train_s512.pt`
  via `paper_tools/build_msc_chains.py` + concat. *Hard blocker for E1/E2.*
  If this isn't ready, fall back to PG-19 + TV only and degrade to "soft
  signal on dialogue is paper 2 territory."

### Day 2

**Local GPU 0.** Standalone eval on E1 / softparity ckpt
(`eval_chain.py` on PG-19 val + test + LoCoMo + MSC test + TV-held-out).

**Local GPU 1.** Standalone eval on E2 / simple_gate ckpt + run NIAH
locally on both.

**Cloud.**
- RAG baselines on every eval split: BM25, MiniLM, Contriever,
  MiniLM-FT, all at matched FLOPs (count chunks read, normalize).
- RULER setup: download S=4k subset (niah-mk, niah-mv, vt, qa-1, qa-2)
  + adapter that converts to our session-chain format.

**Decision triggers.**

- If E1 LoCoMo `?_sh-m` ? E2 LoCoMo `?_sh-m` + 1 SE ? headline holds,
  proceed to write �5.
- If E1 LoCoMo `?_sh-m` < E2 LoCoMo `?_sh-m` ? **stop and
  reframe**. Either: routing pool needs more steps (extend on Day 3),
  or the architectural advantage is bounded by data, not architecture
  (move toward paper 2 framing).

### Day 3

**Local GPU 0.** Counterfactual sensitivity probe (alter session-$t-k$,
measure ?NLL on session-$t$, for $k \in \{1, 5, 10\}$) + per-depth
routing trace ($\alpha_{\text{mem}}$ averaged across LoCoMo, by sublayer).

**Local GPU 1.** **E3 � Mixed-corpus `attention_base`** to fill the
ablation table.

**Cloud.** RULER eval on E1 / E2 / E3 ckpts as they become available.
Begin LongBench-en (NarrativeQA, MultiFieldQA, Qasper, GovReport).

### Day 4

**Local GPU 0.** **E4 � Hidden-state extraction ablation.** Modify
`compress_session` to consume `hidden_states[layer_idx=14]` instead
of raw `embed_tokens(input_ids)`. Same training recipe as E1.

```python
# in train_chain.py, replace:
#     C_t = model.model.embed_tokens(input_ids)
# with:
with torch.no_grad():
    h = model.model(input_ids, output_hidden_states=True).hidden_states
    C_t = h[14].detach()
```

10 h. This is the architectural answer to "we're compressing token
embeddings, not semantic representations" � the key ablation a
reviewer asked about.

**Local GPU 1.** Spare; either re-eval or run a longer baseline.

**Cloud.** Final RAG / RULER / LongBench cells. Callback EM on
generation outputs from each ckpt.

### Day 5

Local both GPUs: standalone eval on E4 ckpt; ablation table fill.
Cloud: release after final eval. **Save remaining $260.**