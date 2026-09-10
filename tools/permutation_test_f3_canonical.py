#!/usr/bin/env python3
"""Chain-level paired permutation test for the F3-OFF flip, against the
*canonical* eval_callback.py pipeline (results/eval_v25_seed_pack_evpos/).

This is the canonical-pipeline counterpart of permutation_test_f3.py
(which reads the per-category pipeline). The two pipelines agree on
sign and approximate magnitude; this script's numbers are the ones
that should be quoted in the paper because they come from the same
eval_callback.py invocation as NEURIPS_NUMBERS.md.

Same statistical design as permutation_test_f3.py: paired sign-flip
permutation across the 50 LongMemEval-S validation chains, with
F3-ON / F3-OFF labels exchangeable per chain under H0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "results" / "eval_v25_seed_pack_evpos"
OUT_DIR = REPO_ROOT / "results" / "exp2_chain_recipe" / "permutation_test_canonical"

# F3-ON, 0.6B (final checkpoint, 1000 steps): three v24a seeds
F3_ON_FINAL = [
    "v24a_seed1_final_lme_val_evpos.json",
    "v24a_seed2_final_lme_val_evpos.json",
    "v24a_seed3_final_lme_val_evpos.json",
]
# F3-OFF, 0.6B: four v27b seeds. Seed 1 lacks the seed-number suffix
# (just "no_probe_final"); seeds 2-4 are explicit.
F3_OFF_FINAL = [
    "v27b_no_probe_final_lme_val_evpos.json",
    "v27b_no_probe_seed2_final_lme_val_evpos.json",
    "v27b_no_probe_seed3_final_lme_val_evpos.json",
    "v27b_no_probe_seed4_final_lme_val_evpos.json",
]


def load_per_chain(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    blob = json.loads(path.read_text())
    pc = blob["lme_val"]["per_chain"]
    ids = [r["chain_id"] for r in pc]
    ce_mem = np.array([r["ce_mem"] for r in pc], dtype=np.float64)
    ce_nomem = np.array([r["ce_nomem"] for r in pc], dtype=np.float64)
    return ids, ce_mem, ce_nomem


def stack_dnm(seeds: list[str]) -> tuple[list[str], np.ndarray]:
    rows = [load_per_chain(EVAL_DIR / s) for s in seeds]
    ref_ids = rows[0][0]
    for ids, _, _ in rows[1:]:
        assert ids == ref_ids, "chain order mismatch across seeds"
    dnm = np.stack([nm - me for _, me, nm in rows], axis=0)
    return ref_ids, dnm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    ids_on, dnm_on = stack_dnm(F3_ON_FINAL)
    ids_off, dnm_off = stack_dnm(F3_OFF_FINAL)
    assert ids_on == ids_off, "F3-ON / F3-OFF chain order mismatch"

    n_chains = len(ids_on)
    delta_per_chain = dnm_off.mean(axis=0) - dnm_on.mean(axis=0)
    observed_T = float(delta_per_chain.mean())

    signs = rng.choice([-1.0, 1.0], size=(args.n_perm, n_chains))
    perm_T = (signs * delta_per_chain[None, :]).mean(axis=1)
    p_two = float(np.mean(np.abs(perm_T) >= abs(observed_T)))
    p_one = float(np.mean(perm_T >= observed_T))

    n_pos = int((delta_per_chain > 0).sum())
    sign = binomtest(n_pos, n_chains, 0.5, alternative="greater")
    p_sign_one = float(sign.pvalue)

    boot_idx = rng.integers(0, n_chains, size=(args.n_boot, n_chains))
    boot_T = delta_per_chain[boot_idx].mean(axis=1)
    ci_lo, ci_hi = (float(x) for x in np.quantile(boot_T, [0.025, 0.975]))

    pair_winrates = {}
    for off in F3_OFF_FINAL:
        _, off_me, off_nm = load_per_chain(EVAL_DIR / off)
        off_dnm = off_nm - off_me
        for on in F3_ON_FINAL:
            _, on_me, on_nm = load_per_chain(EVAL_DIR / on)
            on_dnm = on_nm - on_me
            wins = int((off_dnm > on_dnm).sum())
            key = f"{Path(off).stem}__vs__{Path(on).stem}"
            pair_winrates[key] = wins

    out = {
        "design": {
            "n_chains": n_chains,
            "n_seeds_F3_ON": len(F3_ON_FINAL),
            "n_seeds_F3_OFF": len(F3_OFF_FINAL),
            "n_perm": args.n_perm,
            "n_boot": args.n_boot,
            "rng_seed": args.seed,
            "F3_ON_seeds": F3_ON_FINAL,
            "F3_OFF_seeds": F3_OFF_FINAL,
            "eval_pipeline": "canonical (eval_callback.py via eval_v25_seed_pack_evpos)",
        },
        "results": {
            "observed_T_nats": observed_T,
            "ci95_bootstrap_nats": [ci_lo, ci_hi],
            "p_value_one_sided_perm": p_one,
            "p_value_two_sided_perm": p_two,
            "n_chains_F3OFF_beats_F3ON": n_pos,
            "p_value_one_sided_sign": p_sign_one,
            "median_delta_per_chain_nats": float(np.median(delta_per_chain)),
            "min_delta_per_chain_nats": float(delta_per_chain.min()),
            "max_delta_per_chain_nats": float(delta_per_chain.max()),
            "F3_ON_mean_dnm_per_seed": [float(x) for x in dnm_on.mean(axis=1)],
            "F3_OFF_mean_dnm_per_seed": [float(x) for x in dnm_off.mean(axis=1)],
        },
        "per_seed_pair_winrates_out_of_50": pair_winrates,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "permutation_test_results.json"
    json_path.write_text(json.dumps(out, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.hist(perm_T, bins=80, color="0.75", edgecolor="0.4")
        ax.axvline(observed_T, color="C3", lw=2,
                   label=f"observed = {observed_T:+.3f} nats")
        ax.set_xlabel("mean(delta_per_chain) under permuted signs")
        ax.set_ylabel("count")
        ax.set_title(f"Null distribution (n_perm={args.n_perm}); "
                     f"p_one-sided = {p_one:.4f}")
        ax.legend(loc="upper left", fontsize=9)

        ax = axes[1]
        order = np.argsort(delta_per_chain)
        sorted_d = delta_per_chain[order]
        colors = ["C2" if d > 0 else "C3" for d in sorted_d]
        ax.bar(np.arange(n_chains), sorted_d, color=colors, width=1.0)
        ax.axhline(0, color="0.2", lw=0.7)
        ax.set_xlabel("chain (sorted by delta_c)")
        ax.set_ylabel("delta_c (F3-OFF minus F3-ON, nats)")
        ax.set_title(f"Per-chain effect; "
                     f"{n_pos}/{n_chains} positive "
                     f"(sign-test p = {p_sign_one:.2e})")

        fig.tight_layout()
        png_path = OUT_DIR / "permutation_test.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        out["png_path"] = str(png_path.relative_to(REPO_ROOT))
    except Exception as exc:
        out["plot_error"] = repr(exc)

    json_path.write_text(json.dumps(out, indent=2))

    print(f"observed_T (mean delta_per_chain) = {observed_T:+.4f} nats")
    print(f"95% bootstrap CI = [{ci_lo:+.4f}, {ci_hi:+.4f}] nats")
    print(f"one-sided permutation p = {p_one:.5f} "
          f"(out of {args.n_perm:d})")
    print(f"two-sided permutation p = {p_two:.5f}")
    print(f"chains where F3-OFF beats F3-ON: {n_pos} / {n_chains}")
    print(f"one-sided sign-test p = {p_sign_one:.3e}")
    print(f"median delta_c = {np.median(delta_per_chain):+.4f} nats; "
          f"range [{delta_per_chain.min():+.3f}, "
          f"{delta_per_chain.max():+.3f}]")
    print(f"F3-ON  mean D_dnm per seed: "
          f"{[round(float(x), 3) for x in dnm_on.mean(axis=1)]}")
    print(f"F3-OFF mean D_dnm per seed: "
          f"{[round(float(x), 3) for x in dnm_off.mean(axis=1)]}")
    print(f"per-seed-pair F3-OFF beats F3-ON wins (out of 50):")
    for k, v in pair_winrates.items():
        print(f"  {v:2d}/50  {k}")

    print(f"\nresults written to {json_path}")


if __name__ == "__main__":
    main()