#!/usr/bin/env python3
"""Aggregate ``results/eval_per_category/*.json`` into a markdown table.

Produces one row per (recipe, scale) bucket with seed-mean / seed-std
of Δ_cb, Δ_sh-random, and Δ_sh-samecat at the corpus level, plus a
per-category breakdown averaged across seeds.

Usage:
    python tools/aggregate_per_category_eval.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


RECIPE_RX = re.compile(r"^(?P<recipe>v[0-9]+[a-z]*)_seed(?P<seed>\d+)_(?P<scale>[0-9]p[0-9]+b)_")


def main() -> None:
    root = Path("results/eval_per_category")
    files = sorted(root.glob("*_lme_val.json"))
    if not files:
        raise SystemExit(f"no JSONs found in {root}/")

    rows = []
    for fp in files:
        m = RECIPE_RX.match(fp.stem)
        if not m:
            print(f"  skip (parse): {fp.name}")
            continue
        d = json.loads(fp.read_text())["lme_val"]
        rows.append({
            "recipe": m.group("recipe"),
            "scale": m.group("scale"),
            "seed": int(m.group("seed")),
            "dnm": d["pa_cb_dnm"],
            "dsh_rand": d["pa_cb_dsh_random"],
            "dsh_samecat": d["pa_cb_dsh_samecat"],
            "per_category": d.get("per_category", {}),
        })

    # -- Corpus-level aggregate per (recipe, scale).
    print("\n# Per-recipe corpus-level summary")
    print()
    print("| recipe | scale | n_seeds | Δ_cb (mean ± std) | Δ_sh random | Δ_sh same-category |")
    print("|---|---|---|---|---|---|")
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["recipe"], r["scale"])].append(r)
    for (recipe, scale), bucket in sorted(by_key.items()):
        n = len(bucket)
        dnm_m = mean([b["dnm"] for b in bucket])
        dnm_s = stdev([b["dnm"] for b in bucket]) if n > 1 else 0.0
        dsr_m = mean([b["dsh_rand"] for b in bucket])
        dsr_s = stdev([b["dsh_rand"] for b in bucket]) if n > 1 else 0.0
        dss_m = mean([b["dsh_samecat"] for b in bucket])
        dss_s = stdev([b["dsh_samecat"] for b in bucket]) if n > 1 else 0.0
        print(f"| {recipe} | {scale} | {n} | "
              f"{dnm_m:+.3f} ± {dnm_s:.3f} | "
              f"{dsr_m:+.4f} ± {dsr_s:.4f} | "
              f"{dss_m:+.4f} ± {dss_s:.4f} |")

    # -- Per-category breakdown for the headline (v27b) and reference (v24a).
    for recipe in ("v27b", "v28a", "v24a"):
        bucket = [r for r in rows if r["recipe"] == recipe]
        if not bucket:
            continue
        # Combine same-recipe seeds; keep scale separate.
        for scale in sorted(set(r["scale"] for r in bucket)):
            scale_bucket = [r for r in bucket if r["scale"] == scale]
            if not scale_bucket:
                continue
            cats: dict[str, dict[str, list[float]]] = {}
            for r in scale_bucket:
                for cat, c in r["per_category"].items():
                    cats.setdefault(cat, {"dnm": [], "dsr": [], "dss": [], "n": []})
                    cats[cat]["dnm"].append(c["pa_cb_dnm"])
                    cats[cat]["dsr"].append(c["pa_cb_dsh_random"])
                    cats[cat]["dss"].append(c["pa_cb_dsh_samecat"])
                    cats[cat]["n"].append(c["n_chains"])
            n_seeds = len(scale_bucket)
            print(f"\n# {recipe} ({scale}, n_seeds={n_seeds}) — per-category")
            print()
            print("| category | n_chains | Δ_cb mean | Δ_sh rand | Δ_sh same-cat | Δ_sh hidden? (Δsamecat − Δrand) |")
            print("|---|---|---|---|---|---|")
            for cat in sorted(cats.keys()):
                d = cats[cat]
                n_chains = d["n"][0]
                dnm = mean(d["dnm"])
                dsr = mean(d["dsr"])
                dss = mean(d["dss"])
                hidden = dss - dsr  # how much extra signal same-cat negs surface
                star = " **★**" if abs(hidden) > 0.02 else ""
                print(f"| {cat} | {n_chains} | "
                      f"{dnm:+.3f} | {dsr:+.4f} | {dss:+.4f} | "
                      f"{hidden:+.4f}{star} |")

    # -- Per-seed table for v27b.
    print("\n# v27b 0.6B per-seed")
    print()
    print("| seed | Δ_cb | Δ_sh rand | Δ_sh same-cat |")
    print("|---|---|---|---|")
    for r in sorted([r for r in rows if r["recipe"] == "v27b" and r["scale"] == "0p6b"], key=lambda r: r["seed"]):
        print(f"| {r['seed']} | {r['dnm']:+.3f} | {r['dsh_rand']:+.4f} | {r['dsh_samecat']:+.4f} |")

    # -- Per-seed table for v28 (1.7B).
    print("\n# v28 1.7B per-seed")
    print()
    print("| recipe | seed | Δ_cb | Δ_sh rand | Δ_sh same-cat |")
    print("|---|---|---|---|---|")
    for r in sorted([r for r in rows if r["scale"] == "1p7b"], key=lambda r: (r["recipe"], r["seed"])):
        print(f"| {r['recipe']} | {r['seed']} | {r['dnm']:+.3f} | {r['dsh_rand']:+.4f} | {r['dsh_samecat']:+.4f} |")


if __name__ == "__main__":
    main()
