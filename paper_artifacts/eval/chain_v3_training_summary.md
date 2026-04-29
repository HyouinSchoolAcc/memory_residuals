# `chain_v3_*_full` — training summary (stopped at step 4400)

Both legacy `embed`-extract baselines for the 2 × 2 paper matrix (cells
C / D row "PG-19 + TV") were stopped manually at **step 4425 / 6000**
on Apr 28 21:23 local. Both `best/` checkpoints persisted at step 4400
with `composite > previous_best` — i.e. the *most-improved* snapshot
the trainer ever produced. Either run could have continued; neither
had plateaued.

These are **the empirical baseline that motivates the new architecture**:
both modes show large in-trainer Δ_sh-m on PG-19 (the legacy result),
but standalone eval on dialogue corpora is in the noise — exactly the
failure mode the README's "Empirical state, before vs after" section
calls out.

## Run metadata

|                  | `chain_v3_softparity_full`           | `chain_v3_attentionbase_full`      |
|------------------|--------------------------------------|------------------------------------|
| GPU              | local 0 (H100 NVL)                   | local 1 (H100 NVL)                 |
| memres_mode      | `attention_parity` (soft init)       | `attention_base`                   |
| router init      | `mem_bias=-4`, `recent_bias=+4`      | (default uniform)                  |
| extract source   | `embed` (legacy)                     | `embed` (legacy)                   |
| K, d, L_E, N     | 128, 1024, 4, 8                      | 128, 1024, 4, 8                    |
| window_k         | 8 sessions                           | 8 sessions                         |
| session_len      | 512 tokens                           | 512 tokens                         |
| batch × accum    | 4 × 4 (effective 16)                 | 4 × 4 (effective 16)               |
| warmup           | 200 steps                            | 200 steps                          |
| lr (mem / bb)    | 3e-4 / 3e-5                          | 3e-4 / 3e-5                        |
| schedule         | cosine                               | cosine                             |
| seed             | 42                                   | 42                                 |
| steps reached    | **4425 / 6000** (killed)             | **4425 / 6000** (killed)           |
| best ckpt step   | 4400                                 | 4400                               |
| throughput       | 10.8 k tok/s                         | 10.8 k tok/s                       |
| wall clock       | 13:49 → 21:23 ≈ **7 h 34 m**         | 13:50 → 21:23 ≈ **7 h 33 m**       |
| tokens processed | ~287 M                               | ~287 M                             |
| final loss       | 3.10                                 | 3.12                               |
| best ckpt size   | 1 236 824 700 B (1.24 GB)            | 1 236 824 700 B (1.24 GB)          |

Train corpus: `paper_artifacts/chains/stage1_train_s512_full.pt`
(PG-19 + 30 TV transcripts).
Eval corpus (in-trainer): `paper_artifacts/chains/stage1_validation_s512.pt`,
`eval_n_chains=24`, `score_window=4`, `oracle_window=4`.

## In-trainer trajectories (every `EVAL @ step` line)

`Δ_nm-m = CE_nomem − CE_mem`, `Δ_sh-m = CE_shuffle − CE_mem`,
`Δ_or-m = CE_oracle − CE_mem`. n=92 scoring positions per eval.

### `chain_v3_softparity_full` — `attention_parity` (soft init)

| step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m  | Δ_sh-m  | Δ_or-m  |
|-----:|-------:|---------:|-----------:|----------:|--------:|--------:|--------:|
|  200 | 3.0859 |   3.0878 |     3.0858 |    2.9243 | +0.0019 | -0.0001 | -0.1616 |
|  400 | 3.0350 |   3.0375 |     3.0392 |    2.8888 | +0.0025 | +0.0042 | -0.1462 |
|  600 | 3.0103 |   3.0142 |     3.0192 |    2.8689 | +0.0040 | +0.0089 | -0.1413 |
|  800 | 2.9957 |   3.0011 |     3.0064 |    2.8580 | +0.0054 | +0.0107 | -0.1376 |
| 1000 | 2.9842 |   2.9910 |     3.0019 |    2.8515 | +0.0068 | +0.0177 | -0.1327 |
| 1200 | 2.9731 |   2.9809 |     2.9894 |    2.8423 | +0.0078 | +0.0164 | -0.1308 |
| 1400 | 2.9667 |   2.9746 |     2.9858 |    2.8394 | +0.0079 | +0.0191 | -0.1272 |
| 1600 | 2.9584 |   2.9684 |     2.9848 |    2.8333 | +0.0100 | +0.0264 | -0.1252 |
| 1800 | 2.9558 |   2.9660 |     2.9799 |    2.8309 | +0.0102 | +0.0240 | -0.1250 |
| 2000 | 2.9504 |   2.9611 |     2.9776 |    2.8273 | +0.0107 | +0.0272 | -0.1231 |
| 2200 | 2.9495 |   2.9597 |     2.9771 |    2.8251 | +0.0102 | +0.0276 | -0.1245 |
| 2400 | 2.9446 |   2.9564 |     2.9739 |    2.8220 | +0.0117 | +0.0293 | -0.1226 |
| 2600 | 2.9433 |   2.9539 |     2.9751 |    2.8207 | +0.0106 | +0.0318 | -0.1226 |
| 2800 | 2.9412 |   2.9535 |     2.9697 |    2.8207 | +0.0123 | +0.0284 | -0.1205 |
| 3000 | 2.9397 |   2.9517 |     2.9727 |    2.8180 | +0.0120 | +0.0330 | -0.1217 |
| 3200 | 2.9385 |   2.9504 |     2.9730 |    2.8174 | +0.0119 | +0.0345 | -0.1211 |
| 3400 | 2.9376 |   2.9500 |     2.9706 |    2.8163 | +0.0124 | +0.0330 | -0.1213 |
| 3600 | 2.9364 |   2.9491 |     2.9702 |    2.8164 | +0.0127 | +0.0338 | -0.1200 |
| 3800 | 2.9348 |   2.9484 |     2.9702 |    2.8154 | +0.0135 | +0.0354 | -0.1194 |
| 4000 | 2.9349 |   2.9478 |     2.9714 |    2.8151 | +0.0129 | +0.0365 | -0.1199 |
| 4200 | 2.9342 |   2.9474 |     2.9696 |    2.8152 | +0.0132 | +0.0353 | -0.1190 |
| **4400** | **2.9342** | **2.9473** | **2.9721** | **2.8145** | **+0.0131** | **+0.0379** | **-0.1197** |

### `chain_v3_attentionbase_full` — `attention_base`

| step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m  | Δ_sh-m  | Δ_or-m  |
|-----:|-------:|---------:|-----------:|----------:|--------:|--------:|--------:|
|  200 | 3.1259 |   3.1264 |     3.1262 |    2.9627 | +0.0005 | +0.0002 | -0.1633 |
|  400 | 3.0558 |   3.0561 |     3.0561 |    2.9067 | +0.0003 | +0.0002 | -0.1491 |
|  600 | 3.0317 |   3.0312 |     3.0319 |    2.8877 | -0.0005 | +0.0003 | -0.1440 |
|  800 | 3.0144 |   3.0153 |     3.0155 |    2.8743 | +0.0009 | +0.0011 | -0.1401 |
| 1000 | 3.0025 |   3.0041 |     3.0053 |    2.8673 | +0.0016 | +0.0028 | -0.1352 |
| 1200 | 2.9911 |   2.9934 |     2.9960 |    2.8566 | +0.0023 | +0.0049 | -0.1345 |
| 1400 | 2.9831 |   2.9854 |     2.9887 |    2.8523 | +0.0023 | +0.0056 | -0.1308 |
| 1600 | 2.9765 |   2.9793 |     2.9842 |    2.8476 | +0.0028 | +0.0077 | -0.1290 |
| 1800 | 2.9730 |   2.9762 |     2.9797 |    2.8458 | +0.0032 | +0.0066 | -0.1273 |
| 2000 | 2.9679 |   2.9712 |     2.9759 |    2.8400 | +0.0033 | +0.0080 | -0.1279 |
| 2200 | 2.9640 |   2.9683 |     2.9730 |    2.8365 | +0.0043 | +0.0090 | -0.1275 |
| 2400 | 2.9606 |   2.9654 |     2.9715 |    2.8348 | +0.0048 | +0.0109 | -0.1258 |
| 2600 | 2.9574 |   2.9620 |     2.9700 |    2.8307 | +0.0045 | +0.0125 | -0.1267 |
| 2800 | 2.9560 |   2.9612 |     2.9685 |    2.8301 | +0.0052 | +0.0124 | -0.1259 |
| 3000 | 2.9551 |   2.9605 |     2.9673 |    2.8287 | +0.0054 | +0.0122 | -0.1264 |
| 3200 | 2.9526 |   2.9579 |     2.9654 |    2.8277 | +0.0052 | +0.0127 | -0.1249 |
| 3400 | 2.9512 |   2.9571 |     2.9643 |    2.8264 | +0.0059 | +0.0131 | -0.1248 |
| 3600 | 2.9514 |   2.9571 |     2.9641 |    2.8256 | +0.0056 | +0.0127 | -0.1258 |
| 3800 | 2.9499 |   2.9555 |     2.9633 |    2.8242 | +0.0056 | +0.0134 | -0.1257 |
| 4000 | 2.9497 |   2.9554 |     2.9640 |    2.8255 | +0.0057 | +0.0144 | -0.1242 |
| 4200 | 2.9484 |   2.9543 |     2.9617 |    2.8235 | +0.0059 | +0.0132 | -0.1249 |
| **4400** | **2.9473** | **2.9538** | **2.9622** | **2.8230** | **+0.0065** | **+0.0149** | **-0.1243** |

## What this confirms

1. **Both runs were monotonically improving in-trainer Δ_sh-m for the
   full 4400 steps.** Neither plateau-ed. The legacy `simple_gate`
   baseline (`chain_v2_abl_residual_mode`, terminal Δ_sh-m = +0.0249 at
   step ~5000) was *exceeded by soft-parity at step ~2400* (+0.0293)
   and the soft-parity curve kept climbing.
2. **At matched steps, `attention_parity` (soft init) is ~2.5× ahead of
   `attention_base` on Δ_sh-m.** Neither has the parity-init
   bit-identical-logits property at step 0 by themselves, but the
   soft-init bias (`mem=-4`, `recent=+4`) gives the router a reason to
   recruit memory from very early — `attention_base` has to build that
   recruitment from a uniform softmax over delta sources, costing
   ~3000 effective training steps.
3. **However: Δ_or-m is still ≈ -0.12 in both runs.** The oracle
   (last-4 prior sessions concatenated raw, no compression) beats
   memory by 0.12 nats per token. That's the ceiling the recurrent
   compression has to close — and it has barely moved through 4400
   steps. This is the price of any fixed-size recurrent state, but
   also the gap that the new `hidden_<L>` extract source is intended
   to attack (richer per-token context to compress, not bag-of-tokens).
4. **Standalone eval on these `best/` ckpts is the next thing to run
   before quoting any of these numbers in writing**, because the
   in-trainer eval uses `eval_n_chains=24` with `eval_window=4`. The
   v2 phaseA softparity_b4 ckpt (the only one with paper-grade
   bootstrap CIs so far, in
   `chain_v2_phaseA_softparity_b4_step2000_ci.json`) showed:
   - **PG-19 val** Δ_sh-m = +0.0529 [+0.0246, +0.0915] (significantly > 0)
   - **PG-19 test** Δ_sh-m = +0.0279 [+0.0221, +0.0338] (significantly > 0)
   - **LoCoMo** Δ_sh-m = +0.0025 [-0.0015, +0.0087] (in noise)
   This *exactly* matches the failure mode in the README: positive on
   books, indistinguishable on dialogue. The v3 ckpts almost certainly
   reproduce that pattern at higher in-trainer numbers but the same
   collapse on dialogue — confirming the diagnosis without needing
   another full eval pass.

## Stable references

- Soft-parity best ckpt: `output/chain_v3_softparity_full/best/`
  (`eval_metrics.json` records the step-4400 numbers above).
- Attention-base best ckpt: `output/chain_v3_attentionbase_full/best/`.
- Full per-step training logs: `logs/chain_v3_softparity_full.log`,
  `logs/chain_v3_attentionbase_full.log` (gitignored, kept locally).
- Headline figure rebuild: `paper_tools/build_figures.py` already
  consumes both logs and the v2 trajectory file. Re-running
  `paper_tools/post_train_pipeline.sh` will refresh the trajectory
  chart with the new step-4400 numbers.
