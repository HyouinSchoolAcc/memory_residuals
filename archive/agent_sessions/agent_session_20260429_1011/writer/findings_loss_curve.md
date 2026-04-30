# Findings — `chain_v4_hidden14_msc` loss curve (step 200 → 5400)

**Source:** `/home/ubuntu/memory_residuals/paper_tools/cloud_watchdog/logs/chain_v4_hidden14_msc.log`
on the GH200, captured 2026-04-29 15:13 UTC. Last train step: 5460.

## Train NLL trajectory

Train loss at print intervals (every 20 steps; values are per-step
unaveraged, so they're noisy):

| step | train loss (approx) |
|-----:|---------------------|
|   20 | 3.49                |
|  100 | 3.10                |
|  500 | 2.94                |
| 1000 | 2.83                |
| 2000 | ~2.80 (noisy 2.6–2.95) |
| 3000 | ~2.78                |
| 4000 | ~2.78                |
| 5000 | ~2.65                |
| 5460 | ~2.88 (instantaneous) |

Trend is the **expected pretraining-style decay** (3.5 → ~2.7 over
5000 steps with a cosine-decayed lr from 2e-4 → 1.4e-6 on memres
params and 2e-5 → 1.4e-7 on backbone). Grad norm stable at 2.5–3.3
across the whole run, no NaN / OOM / divergence.

## In-trainer EVAL trajectory (the headline panel)

n=124 score positions per eval, eval_window=4, score_tail_frac=1.0.

| step  | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m  | Δ_sh-m  | Δ_or-m  |
|------:|-------:|---------:|-----------:|----------:|--------:|--------:|--------:|
|   200 | 3.1231 | **3.1231** | **3.1231** | 2.9174   | **+0.0000** | **+0.0000** | -0.2057 |
|   400 | 3.1044 | 3.1044   | 3.1044     | 2.9102    | +0.0000 | +0.0000 | -0.1942 |
|   600 | 3.1169 | 3.1169   | 3.1169     | 2.9123    | +0.0000 | +0.0000 | -0.2046 |
|   800 | 3.0985 | 3.0985   | 3.0985     | 2.9083    | +0.0000 | +0.0000 | -0.1902 |
|  1000 | 3.0876 | 3.0876   | 3.0876     | 2.9021    | +0.0000 | +0.0000 | -0.1855 |
|  1200 | 3.1003 | 3.1003   | 3.1003     | 2.9077    | +0.0000 | +0.0000 | -0.1926 |
|  1400 | 3.1091 | 3.1091   | 3.1091     | 2.9111    | +0.0000 | +0.0000 | -0.1981 |
|  1600 | 3.0907 | 3.0907   | 3.0907     | 2.9032    | +0.0000 | +0.0000 | -0.1875 |
|  1800 | 3.1068 | 3.1068   | 3.1068     | 2.9169    | +0.0000 | +0.0000 | -0.1899 |
|  2000 | 3.0995 | 3.0995   | 3.0995     | 2.9164    | +0.0000 | +0.0000 | -0.1831 |
|  2200 | 3.1037 | 3.1037   | 3.1037     | 2.9206    | +0.0000 | +0.0000 | -0.1831 |
|  2400 | 3.0985 | 3.0985   | 3.0985     | 2.9169    | +0.0000 | +0.0000 | -0.1816 |
|  2600 | 3.0890 | 3.0890   | 3.0890     | 2.9135    | +0.0000 | +0.0000 | -0.1755 |
|  2800 | 3.0946 | 3.0946   | 3.0946     | 2.9171    | +0.0000 | +0.0000 | -0.1774 |
|  3000 | 3.0934 | 3.0934   | 3.0934     | 2.9139    | +0.0000 | +0.0000 | -0.1795 |
|  3200 | 3.0927 | 3.0927   | 3.0927     | 2.9157    | +0.0000 | +0.0000 | -0.1769 |
|  3400 | 3.0884 | 3.0884   | 3.0884     | 2.9106    | +0.0000 | +0.0000 | -0.1778 |
|  3600 | 3.0896 | 3.0896   | 3.0896     | 2.9111    | +0.0000 | +0.0000 | -0.1785 |
|  3800 | 3.0910 | 3.0910   | 3.0910     | 2.9113    | +0.0000 | +0.0000 | -0.1797 |
|  4000 | 3.0909 | 3.0909   | 3.0909     | 2.9113    | +0.0000 | +0.0000 | -0.1796 |
|  4200 | 3.0950 | 3.0950   | 3.0950     | 2.9145    | +0.0000 | +0.0000 | -0.1805 |
|  4400 | 3.0899 | 3.0899   | 3.0899     | 2.9115    | +0.0000 | +0.0000 | -0.1784 |
|  4600 | 3.0908 | 3.0908   | 3.0908     | 2.9112    | +0.0000 | +0.0000 | -0.1796 |
|  4800 | 3.0883 | 3.0883   | 3.0883     | 2.9109    | +0.0000 | +0.0000 | -0.1774 |
|  5000 | 3.0902 | 3.0902   | 3.0902     | 2.9108    | +0.0000 | +0.0000 | -0.1793 |
|  5200 | 3.0899 | 3.0899   | 3.0899     | 2.9108    | +0.0000 | +0.0000 | -0.1791 |
|  5400 | 3.0894 | 3.0894   | 3.0894     | 2.9103    | +0.0000 | +0.0000 | -0.1790 |

## What this says

1. **Mem == NoMem == Shuffle to ≥4 decimals at every EVAL.** This is
   functionally a bit-equivalent forward pass, exactly what the hard
   `mem_bias=-32, recent_bias=+32` init was *designed* to produce at
   step 0, but it has not budged in 5400 steps despite the contrastive
   loss ramp.
2. **mem CE is decreasing on its own (3.12 → 3.09).** That is the
   backbone refining its fit on the new corpus mix (PG-19+TV+MSC at
   {1,4,8} weights), not memory help. The decrease tracks the train-NLL
   decrease from 2.94 to 2.78 over the same window.
3. **Δ_or-m around -0.18 throughout.** The 4-session raw-concat oracle
   is ~0.18 nat better than the augmented model on the held-out mix —
   essentially identical to the v3 numbers (~0.12 on PG-19, here ~0.18
   on PG-19+MSC val because MSC sessions are short and concat is more
   informative). The architecture is leaving 0.18 nat on the table.
4. **No NaN, no OOM, no stall.** Throughput is 2.8 k tok/s with
   gradient checkpointing on the 1×GH200; loss is noisy but
   monotonically descending in moving average.
5. **Negative-chain contrastive loss column is not surfaced in the
   per-step log line.** The trainer prints `loss / lrs / grad_norm /
   gate_mean / max / tok/s` only — neg-chain loss isn't broken out.
   We can't tell from the log alone whether it was computed at all,
   but the ramp schedule (0.05 → 0.5 by step 1000) is in the launch
   command and the code path
   (`train_chain.py:870 ff., cur_neg_weight > 0.0`) is gated only on
   the ramp, so it was being computed.

6. **CAVEAT (per monitor `concern_memory_channel_closed.md`).** The
   `gate_mean` / `gate_max` log column reports `model.memory_gate.gate`
   (`MemoryGate`, `modeling_memres.py:262`), which is **only consumed
   by the `simple_gate` memres mode**. In `attention_parity` mode
   (which this run uses) the forward pass goes through
   `route_if_needed()` → `depth_router.route(...)` and never touches
   `memory_gate.gate`. So `gate_mean = +0.0000` per train line is
   *uninformative* — it is the unused ReZero scalar at its zero init,
   not "the gate has not moved" evidence. The honest in-trainer
   evidence that the channel is closed is the EVAL line
   `mem == nomem == shuffle to 4 decimals`, not the gate column.

## Implication for next iteration

The "contrastive ramp will drive memory off the parity init" thesis
is **falsified at the hard `±32` setting**. Either:

- Switch to soft init (`±4`) so softmax gradients are non-vanishing, OR
- Add an explicit auxiliary loss that *directly* penalizes
  α_mem-near-zero (e.g. KL of routing softmax against a target with
  ε mass on `b_{-1}`), OR
- Switch routing primitive entirely to `simple_gate` for the
  long-horizon recipe and accept the gate-collapse risk on dialogue.

See `next_iteration_proposal.md`.
