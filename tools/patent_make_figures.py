#!/usr/bin/env python3
"""Render patent supporting-experiment figures (Chinese labels).

Produces fig_exp1..5 + patent_numbers.json from whatever result JSONs
are present under results/patent_experiments/. Missing optional cells
(trained overwrite / K-sweep / LoRA joint) are skipped with a warning
so this script can be re-run as each cell lands.
"""

from __future__ import annotations

import json
import statistics
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/patent_experiments"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


def load(p: Path):
    return json.load(open(p))


def maybe(name: str):
    p = RES / name
    if p.exists():
        return load(p)
    warnings.warn(f"missing {p}")
    return None


def dnm_of(j):
    key = "lme_val" if "lme_val" in j else list(j.keys())[0]
    return j[key]["pa_cb_dnm"]


numbers: dict = {}

# ------------------------------------------------------------------ exp 1
j = load(RES / "o1_bench_0p6b.json")
H = [r["H"] for r in j["baseline"]]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

ax = axes[0]
ax.plot(H, [r["prefill_ms"] for r in j["baseline"]], "o-", color="#c0392b",
        label="扩展上下文窗口方案")
ax.plot(H, [r["prefill_ms"] for r in j["memres"]], "s-", color="#2471a3",
        label="本发明方法")
ax.set_xscale("log", base=2)
ax.set_xlabel("历史长度 H（token）")
ax.set_ylabel("单次推理预填充延时（ms）")
ax.set_title("(a) 推理延时")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(H, [r["peak_vram_gib"] for r in j["baseline"]], "o-", color="#c0392b",
        label="扩展上下文窗口方案")
ax.plot(H, [r["peak_vram_gib"] for r in j["memres"]], "s-", color="#2471a3",
        label="本发明方法")
ax.set_xscale("log", base=2)
ax.set_xlabel("历史长度 H（token）")
ax.set_ylabel("推理峰值显存（GiB）")
ax.set_title("(b) 显存占用")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(H, [r["kv_cache_bytes"] / 2**20 for r in j["baseline"]], "o-",
        color="#c0392b", label="KV缓存（扩展上下文）")
ax.plot(H, [(r["kv_cache_bytes"] + r["mc_bytes"]) / 2**20
            for r in j["memres"]], "s-", color="#2471a3",
        label="KV缓存+记忆矩阵（本发明）")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("历史长度 H（token）")
ax.set_ylabel("会话状态存储（MiB，对数轴）")
ax.set_title("(c) 状态存储")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("实验一：推理开销随历史长度的变化（Qwen3-0.6B，H100）", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "fig_exp1_cost.png", dpi=300, bbox_inches="tight")
plt.close(fig)

numbers["exp1"] = {
    "baseline": j["baseline"], "memres": j["memres"],
    "prefill_growth": j["baseline"][-1]["prefill_ms"] / j["baseline"][0]["prefill_ms"],
    "kv_at_32k_mib": j["baseline"][-1]["kv_cache_bytes"] / 2**20,
    "mc_kib": j["memres"][0]["mc_bytes"] / 1024,
}

# ------------------------------------------------------------------ exp 2
def rag_delta(name):
    r = load(ROOT / f"results/rag_baseline/{name}.json")
    return r["pa_cb_drag"]

mem06 = [load(RES / f"stage_ablation_0p6b_seed{s}.json")["delta_vs_nomem"]["full"]
         for s in [1, 2, 3, 4]]
mem17 = [load(RES / f"stage_ablation_1p7b_seed{s}.json")["delta_vs_nomem"]["full"]
         for s in [1, 2, 3]]

rag06 = {
    "BM25检索 top-1": rag_delta("qwen3_0p6b_bm25_top1"),
    "BM25检索 top-3": rag_delta("qwen3_0p6b_bm25_top3"),
    "稠密检索 top-1": rag_delta("qwen3_0p6b_dense_top1"),
    "稠密检索 top-3": rag_delta("qwen3_0p6b_dense_top3"),
    "理想检索(oracle) top-3": rag_delta("qwen3_0p6b_oracle_top3"),
}
rag17 = {
    "BM25检索 top-3": rag_delta("qwen3_1p7b_bm25_top3"),
    "稠密检索 top-3": rag_delta("qwen3_1p7b_dense_top3"),
    "理想检索(oracle) top-3": rag_delta("qwen3_1p7b_oracle_top3"),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, rag, mem, title in [
    (axes[0], rag06, mem06, "(a) Qwen3-0.6B"),
    (axes[1], rag17, mem17, "(b) Qwen3-1.7B"),
]:
    labels = list(rag.keys()) + ["本发明方法"]
    vals = list(rag.values()) + [statistics.mean(mem)]
    colors = ["#909497"] * len(rag) + ["#2471a3"]
    bars = ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("回调交叉熵改善 Δ（nats，越大越好）")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    for b, v in zip(bars, vals):
        ax.text(max(v, 0) + 0.02, b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center", fontsize=8)
    ax.set_xlim(min(min(vals) - 0.1, -0.15), max(vals) + 0.3)
fig.suptitle("实验二：与检索增强生成（RAG）方案的记忆召回质量对比（LongMemEval-S 验证集）", y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig_exp2_rag.png", dpi=300, bbox_inches="tight")
plt.close(fig)

numbers["exp2"] = {
    "rag_0p6b": rag06, "rag_1p7b": rag17,
    "memres_0p6b_mean": statistics.mean(mem06),
    "memres_0p6b_sd": statistics.stdev(mem06),
    "memres_1p7b_mean": statistics.mean(mem17),
    "memres_1p7b_sd": statistics.stdev(mem17),
}

# ------------------------------------------------------------------ exp 3
ret = load(RES / "retention_0p6b_seed3.json")
# Prefer the strongest LoRA variant available (joint lr=2e-5 > joint 1e-4 >
# sequential 2e-5 > original sequential 1e-4).
lora_candidates = [
    "lora_joint_lr2e-5_0p6b.json",
    "lora_joint_lr1e-4_0p6b.json",
    "lora_seq_lr2e-5_0p6b.json",
    "lora_forgetting_0p6b.json",
]
lora = None
lora_name = None
for cand in lora_candidates:
    lora = maybe(cand)
    if lora is not None:
        lora_name = cand
        break
assert lora is not None

# Also load all LoRA variants for the numbers table
lora_all = {}
for cand in lora_candidates:
    jlora = maybe(cand)
    if jlora is not None:
        lora_all[cand] = {k: v for k, v in jlora.items() if k != "per_chain"}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

grid = np.linspace(0, 1, 21)
mem_curves = []
for r in ret["per_chain"]:
    c = np.array(r["curve"])
    change = c - c[0]
    x = np.linspace(0, 1, len(c))
    mem_curves.append(np.interp(grid, x, change))
mem_curves = np.stack(mem_curves)

lora_curves = []
for r in lora["per_chain"]:
    tt = np.array([p["t"] for p in r["trace"]], dtype=float)
    cc = np.array([p["callback_ce"] for p in r["trace"]])
    lora_curves.append(np.interp(grid, tt / max(tt[-1], 1e-6), cc - cc[0]))
lora_curves = np.stack(lora_curves)

ax = axes[0]
for curves, color, label in [
    (mem_curves, "#2471a3", "本发明（竞争式更新，50链平均）"),
    (lora_curves, "#c0392b", f"用户级LoRA（{lora.get('mode','seq')}，8链平均）"),
]:
    m, s = curves.mean(0), curves.std(0)
    ax.plot(grid * 100, m, "-", color=color, label=label)
    ax.fill_between(grid * 100, m - s, m + s, color=color, alpha=0.12)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("已按会话持续更新的历史比例（%）")
ax.set_ylabel("回调交叉熵变化（nats，越低越好）")
ax.set_title("(a) 依赖早期信息的回调表现")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ts = sorted({p["t"] for r in lora["per_chain"] for p in r["trace"]})
gen = {t: [] for t in ts}
for r in lora["per_chain"]:
    tt = [p["t"] for p in r["trace"]]
    gg = [p["general_ce"] for p in r["trace"]]
    for t in ts:
        gen[t].append(float(np.interp(t, tt, gg)))
xs = ts
ys = [statistics.mean(gen[t]) for t in ts]
g0 = ys[0]

ax = axes[1]
ax.plot(xs, [y - g0 for y in ys], "o-", color="#c0392b",
        label="用户级LoRA微调（域外通用能力漂移）")
ax.axhline(0, color="#2471a3", lw=2,
           label="本发明（本体冻结，漂移恒为0）")
ax.set_xlabel("已按会话持续更新的会话数 t")
ax.set_ylabel("域外通用文本交叉熵漂移（nats，越高越差）")
ax.set_title("(b) 通用能力漂移（灾难性遗忘）")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("实验三：抗遗忘能力对比（与用户级微调方案，Qwen3-0.6B）", y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig_exp3_retention.png", dpi=300, bbox_inches="tight")
plt.close(fig)

numbers["exp3"] = {
    "memres_mean_total_drop": ret["mean_total_drop"],
    "memres_mean_ce_t0": ret["mean_ce_t0"],
    "memres_mean_ce_final": ret["mean_ce_final"],
    "lora_primary": lora_name,
    "lora_summary": {k: v for k, v in lora.items() if k != "per_chain"},
    "lora_all_variants": lora_all,
    "lora_general_drift_curve": {"t": xs, "drift": [y - g0 for y in ys]},
    "lora_callback_change_final": float(lora_curves.mean(0)[-1]),
    "memres_callback_change_final": float(mem_curves.mean(0)[-1]),
    "curve_grid_pct": (grid * 100).tolist(),
    "memres_callback_change_mean": mem_curves.mean(0).tolist(),
    "lora_callback_change_mean": lora_curves.mean(0).tolist(),
}

# ------------------------------------------------------------------ exp 4
# Prefer TRAINED overwrite ablation when available; always show existing
# trained R=0 / no-floor; keep eval-time S1+S2 ladder as secondary.
trained = maybe("trained_ablations_existing.json") or {}
ow1 = maybe("eval_trained_overwrite_seed1.json")
ow2 = maybe("eval_trained_overwrite_seed2.json")
overwrite_dnms = []
for jow in (ow1, ow2):
    if jow is not None:
        overwrite_dnms.append(dnm_of(jow))

# Eval-time ladder (secondary)
eval_ladder_06 = {
    c: statistics.mean([
        load(RES / f"stage_ablation_0p6b_seed{s}.json")["delta_vs_nomem"][c]
        for s in [1, 2, 3, 4]
    ])
    for c in ["s1_zero", "s1s2_last", "s1s2_mean", "full"]
}
eval_ladder_06_sd = {
    c: statistics.stdev([
        load(RES / f"stage_ablation_0p6b_seed{s}.json")["delta_vs_nomem"][c]
        for s in [1, 2, 3, 4]
    ])
    for c in ["s1_zero", "s1s2_last", "s1s2_mean", "full"]
}

# Panel (a): TRAINED ablations (primary patent evidence)
# Panel (b): eval-time stage ladder (secondary)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

ax = axes[0]
labels_t = ["完整方法\n(v27b)", "无读出精化\n(R=0)", "无深度路由\nα-floor",
            "无竞争更新\n(覆盖式,从零训练)"]
vals_t = [
    trained.get("full_0p6b", {}).get("dnm_mean", statistics.mean(mem06)),
    trained.get("no_depth_R0_0p6b", {}).get("dnm_mean", float("nan")),
    trained.get("no_floor_0p6b", {}).get("dnm_mean", float("nan")),
    statistics.mean(overwrite_dnms) if overwrite_dnms else float("nan"),
]
sds_t = [
    trained.get("full_0p6b", {}).get("dnm_sd", statistics.stdev(mem06)),
    trained.get("no_depth_R0_0p6b", {}).get("dnm_sd", 0.0),
    trained.get("no_floor_0p6b", {}).get("dnm_sd", 0.0),
    statistics.stdev(overwrite_dnms) if len(overwrite_dnms) > 1 else 0.0,
]
# drop NaN bars
keep = [(i, v, s) for i, (v, s) in enumerate(zip(vals_t, sds_t)) if v == v]
idx = [i for i, _, _ in keep]
vals_plot = [v for _, v, _ in keep]
sds_plot = [s for _, _, s in keep]
labs_plot = [labels_t[i] for i in idx]
colors = ["#2471a3"] + ["#909497"] * (len(vals_plot) - 1)
bars = ax.bar(range(len(vals_plot)), vals_plot, yerr=sds_plot, capsize=4, color=colors)
ax.set_xticks(range(len(vals_plot)))
ax.set_xticklabels(labs_plot, fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("回调交叉熵改善 Δ（nats）")
ax.set_title("(a) 从零训练的消融（主证据）")
ax.grid(alpha=0.3, axis="y")
for b, v in zip(bars, vals_plot):
    ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.04,
            f"{v:+.3f}", ha="center", fontsize=8)

ax = axes[1]
labs_e = ["无记忆", "仅阶段一\n(零记忆)", "阶段一+二\n(覆盖式)", "阶段一+二\n(均值式)", "三阶段全用"]
vals_e = [0.0, eval_ladder_06["s1_zero"], eval_ladder_06["s1s2_last"],
          eval_ladder_06["s1s2_mean"], eval_ladder_06["full"]]
sds_e = [0.0, eval_ladder_06_sd["s1_zero"], eval_ladder_06_sd["s1s2_last"],
         eval_ladder_06_sd["s1s2_mean"], eval_ladder_06_sd["full"]]
colors = ["#909497"] * 4 + ["#2471a3"]
bars = ax.bar(range(5), vals_e, yerr=sds_e, capsize=4, color=colors)
ax.set_xticks(range(5))
ax.set_xticklabels(labs_e, fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("回调交叉熵改善 Δ（nats）")
ax.set_title("(b) 评测时阶段消融（辅证，0.6B，4种子）")
ax.grid(alpha=0.3, axis="y")
for b, v in zip(bars, vals_e):
    ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.04,
            f"{v:+.3f}", ha="center", fontsize=8)

fig.suptitle("实验四：三阶段与关键组件消融（LongMemEval-S 验证集）", y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig_exp4_ablation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

numbers["exp4"] = {
    "trained": trained,
    "trained_overwrite_seeds": overwrite_dnms,
    "trained_overwrite_mean": (statistics.mean(overwrite_dnms)
                               if overwrite_dnms else None),
    "eval_ladder_0p6b": {
        c: {"mean": eval_ladder_06[c], "sd": eval_ladder_06_sd[c]}
        for c in eval_ladder_06
    },
    "eval_ladder_1p7b": {
        c: {
            "mean": statistics.mean([
                load(RES / f"stage_ablation_1p7b_seed{s}.json")
                ["delta_vs_nomem"][c] for s in [1, 2, 3]
            ]),
            "sd": statistics.stdev([
                load(RES / f"stage_ablation_1p7b_seed{s}.json")
                ["delta_vs_nomem"][c] for s in [1, 2, 3]
            ]),
        }
        for c in ["s1_zero", "s1s2_last", "s1s2_mean", "full"]
    },
}

# ------------------------------------------------------------------ exp 5: K sweep
k_cells = {
    32: maybe("eval_trained_k32_seed1.json"),
    128: trained.get("full_0p6b"),  # use multi-seed mean
    512: maybe("eval_trained_k512_seed1.json"),
}
k_xs, k_ys = [], []
for k, cell in k_cells.items():
    if cell is None:
        continue
    if k == 128:
        k_xs.append(k)
        k_ys.append(cell["dnm_mean"])
    else:
        k_xs.append(k)
        k_ys.append(dnm_of(cell))

if len(k_xs) >= 2:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(k_xs, k_ys, "s-", color="#2471a3", markersize=8)
    for x, y in zip(k_xs, k_ys):
        ax.text(x, y + 0.04, f"K={x}\n{y:+.3f}", ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_xs)
    ax.set_xticklabels([str(x) for x in k_xs])
    ax.set_xlabel("记忆槽位数 K")
    ax.set_ylabel("回调交叉熵改善 Δ（nats）")
    ax.set_title("实验五：记忆容量 K 扫描（Qwen3-0.6B，从零训练）")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp5_capacity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

numbers["exp5"] = {
    "K": k_xs,
    "dnm": k_ys,
    "mc_bytes": {str(k): k * 1024 * 2 for k in k_xs},  # d=1024, bf16
}

# ------------------------------------------------------------------ exp 6
# Stage-3 structural comparison: exact claim-1 embodiment (single-round
# 2K-softmax judge) vs the iterative slot-competitive variant vs the
# closest-prior-art update structures, everything else held identical.
def acc_of(j):
    key = "lme_val" if "lme_val" in j else list(j.keys())[0]
    return {k: j[key].get(k) for k in
            ("acc_mem", "acc_nomem", "acc_shuffle", "acc_gain")}

claim1_cells = [maybe(f"eval_trained_claim1judge_seed{s}.json")
                for s in (1, 2, 3, 4)]
claim1_dnms = [dnm_of(j) for j in claim1_cells if j is not None]
selfattn_cell = maybe("eval_trained_selfattn_seed1.json")
gnj_cell = maybe("eval_trained_gatednojudge_seed1.json")

exp6 = {
    "claim1_dnms": claim1_dnms,
    "claim1_mean": statistics.mean(claim1_dnms) if claim1_dnms else None,
    "claim1_sd": (statistics.stdev(claim1_dnms)
                  if len(claim1_dnms) > 1 else None),
    "slotattn_mean": trained.get("full_0p6b", {}).get("dnm_mean"),
    "slotattn_sd": trained.get("full_0p6b", {}).get("dnm_sd"),
    "slotattn_n": trained.get("full_0p6b", {}).get("n"),
    "selfattn_dnm": dnm_of(selfattn_cell) if selfattn_cell else None,
    "gated_nojudge_dnm": dnm_of(gnj_cell) if gnj_cell else None,
    "overwrite_mean": (statistics.mean(overwrite_dnms)
                       if overwrite_dnms else None),
    "overwrite_n": len(overwrite_dnms),
    "mean_pool_eval": {"mean": eval_ladder_06["s1s2_mean"],
                       "sd": eval_ladder_06_sd["s1s2_mean"]},
}
# per-variant accuracy (present once evals run with the updated
# eval_callback that reports token-level recall accuracy)
acc_table = {}
for tag, j in [
    ("claim1_seed1", claim1_cells[0]), ("claim1_seed2", claim1_cells[1]),
    ("claim1_seed3", claim1_cells[2]), ("claim1_seed4", claim1_cells[3]),
    ("selfattn", selfattn_cell), ("gated_nojudge", gnj_cell),
    ("headline_k128", maybe("eval_trained_k128_seed3_acc.json")),
]:
    if j is None:
        continue
    a = acc_of(j)
    if a.get("acc_mem") is not None:
        acc_table[tag] = a
exp6["acc"] = acc_table
numbers["exp6"] = exp6

# NOTE: the "claim1_*" keys hold the single-round 2K-softmax judge cells
# (named before the 2026-07 restructure made the iterative slot-competitive
# update the preferred embodiment (claim 6) and the single-round judge a
# claimed simplified variant (claim 10).
bars6 = [
    ("迭代式槽位竞争更新\n(权利要求6，本发明)", exp6["slotattn_mean"],
     exp6["slotattn_sd"] or 0.0, "#2471a3"),
    ("单轮2K-softmax判定\n(权利要求10)", exp6["claim1_mean"],
     exp6["claim1_sd"] or 0.0, "#2e86c1"),
    ("RMC式自注意力更新\n(无判定查询)", exp6["selfattn_dnm"], 0.0, "#909497"),
    ("门控替换更新\n(无竞争)", exp6["gated_nojudge_dnm"], 0.0, "#909497"),
    ("覆盖式更新\n(无竞争)", exp6["overwrite_mean"], 0.0, "#909497"),
    ("均值池化更新\n(评测时)", exp6["mean_pool_eval"]["mean"],
     exp6["mean_pool_eval"]["sd"], "#909497"),
]
bars6 = [(l, v, s, c) for l, v, s, c in bars6 if v is not None]
if len(bars6) >= 3:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    xs = range(len(bars6))
    bs = ax.bar(xs, [v for _, v, _, _ in bars6],
                yerr=[s for _, _, s, _ in bars6], capsize=4,
                color=[c for _, _, _, c in bars6])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l for l, _, _, _ in bars6], fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("回调交叉熵改善 Δ（nats）")
    ax.set_title("实验六：阶段三更新结构对比（其余全部相同，从零训练，Qwen3-0.6B）")
    ax.grid(alpha=0.3, axis="y")
    for b, (_, v, _, _) in zip(bs, bars6):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.04,
                f"{v:+.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp6_stage3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# synthd5 ladder if present
synth = maybe("stage_ablation_synthd5_v32a.json")
if synth is not None:
    numbers["synthd5_ladder"] = {
        "ce_mean": synth["ce_mean"],
        "delta_vs_nomem": synth["delta_vs_nomem"],
        "n_chains": synth["n_chains_scored"],
    }

(RES / "patent_numbers.json").write_text(json.dumps(numbers, indent=2))
print("figures ->", FIG)
print("numbers ->", RES / "patent_numbers.json")
print("overwrite seeds:", overwrite_dnms)
print("K sweep:", list(zip(k_xs, k_ys)))
print("lora primary:", lora_name)
