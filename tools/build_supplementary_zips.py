"""Build the three NeurIPS 2026 supplementary zips for P1/P2/P3.

Stages each paper's payload under `supplementary_build/<paper>/`,
anonymizes author/host/path identifiers in-place on the staged copy,
then writes `<paper>_supplementary.zip` next to this script.

Usage:  python tools/build_supplementary_zips.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent  # memory_residuals/
BUILD = ROOT / "supplementary_build"
OUT_DIR = ROOT  # zips land at memory_residuals/<name>_supplementary.zip

# ---------------------------------------------------------------------------
# Anonymisation rules. Conservative: lossless on numerics, scrubs identifiers.
# ---------------------------------------------------------------------------
_ANON_PAIRS_RE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/home/exx\b"), "/home/anon"),
    (re.compile(r"\b192\.222\.50\.225\b"), "<remote-host>"),
    (re.compile(r"\bubuntu@"), "<remote-user>@"),
    (re.compile(r"\bYueze\s+Liu\b"), "Anonymous Author"),
    (re.compile(r"\bAjay\s+Kumdam\b"), "Anonymous Author"),
    (re.compile(r"\bYueze\b"), "Anon"),
    (re.compile(r"\bAjay\b"), "Anon"),
    (re.compile(r"\byuezel2@illinois\.edu\b"), "<email-anon>"),
    (re.compile(r"\bS;G\s+studio\b", re.IGNORECASE), "<affiliation-anon>"),
    (re.compile(r"\bexx@\b"), "<user>@"),
]

# Skip these paths entirely when staging (no cache, no compiled bytecode).
_SKIP_PATH_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd"}


def _is_skipped(path: Path) -> bool:
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return True
    return any(part in _SKIP_PATH_PARTS for part in path.parts)


def _ignore_for_copytree(_dir: str, names: list[str]) -> list[str]:
    out = []
    for n in names:
        if n in _SKIP_PATH_PARTS or Path(n).suffix.lower() in _SKIP_SUFFIXES:
            out.append(n)
    return out

ANON_TEXT_EXTS = {
    ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg",
    ".ini", ".toml", ".tex", ".bib",
}


def anonymise_file(path: Path) -> bool:
    """In-place anonymise a single text file. Returns True if changed."""
    if path.suffix.lower() not in ANON_TEXT_EXTS:
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    out = original
    for pat, repl in _ANON_PAIRS_RE:
        out = pat.sub(repl, out)
    if out != original:
        path.write_text(out, encoding="utf-8")
        return True
    return False


def anonymise_tree(root: Path) -> tuple[int, int]:
    """Walk a directory tree and anonymise every file in place."""
    seen = changed = 0
    for p in root.rglob("*"):
        if p.is_file():
            seen += 1
            if anonymise_file(p):
                changed += 1
    return seen, changed


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------
def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(
            src, dst, dirs_exist_ok=True, ignore=_ignore_for_copytree,
        )
    else:
        if _is_skipped(src):
            return
        shutil.copy2(src, dst)


def _strip_caches(root: Path) -> int:
    """Belt-and-braces: walk staged tree and rip out any cache that snuck in."""
    removed = 0
    for p in list(root.rglob("*")):
        if not p.exists():
            continue
        if p.is_dir() and p.name in _SKIP_PATH_PARTS:
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
        elif p.is_file() and p.suffix.lower() in _SKIP_SUFFIXES:
            p.unlink()
            removed += 1
    return removed


def stage_files(stage_root: Path, mapping: list[tuple[str, str]]) -> None:
    """mapping: list of (src_relpath_from_ROOT, dst_relpath_from_stage_root)."""
    for src_rel, dst_rel in mapping:
        src = ROOT / src_rel
        if not src.exists():
            print(f"  SKIP missing: {src_rel}")
            continue
        _safe_copy(src, stage_root / dst_rel)


def stage_glob(
    stage_root: Path, src_dir_rel: str, pattern: str, dst_dir_rel: str,
) -> int:
    src_dir = ROOT / src_dir_rel
    dst_dir = stage_root / dst_dir_rel
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.glob(pattern):
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Per-paper readme generators (anonymised already)
# ---------------------------------------------------------------------------
P1_README = """# Memory Residuals — Supplementary Material

This bundle contains the source code, training scripts, evaluation
scripts, locked numbers ledger, and per-checkpoint evaluation JSONs
needed to reproduce the headline result of the paper:

> A frozen Qwen3 backbone augmented with a fixed-size, jointly-trained
> Memory Residuals matrix M_c improves callback cross-entropy on
> LongMemEval-S validation by +1.32 ± 0.53 nats at Qwen3-0.6B
> (n=4 seeds) and +0.93 nats at Qwen3-1.7B (n=2 seeds), with a
> chain-shuffle confound of 0.000 ± 0.010 throughout.

## Layout

```
code/                     architecture, trainer, evaluator
  src/                    modeling_memres.py, train_chain.py, presets.py, ...
  tools/eval_callback.py  canonical post-train evaluator (the script the
                          paper's headline numbers come from)
  tests/                  pytest harness
recipes/                  launcher scripts for the headline + ablation cells
  v27b_*.sh               headline 0.6B (4 seeds, F3-OFF, depth=4, floor ON)
  v28*.sh                 scaling 1.7B (2 seeds, same recipe)
  v27a_*.sh               ablation: --memres_readout_depth 0 (depth-off)
  v27c_*.sh               ablation: --alpha_mem_floor_aux_weight 0.0 (floor-off)
  v24a_*.sh               reference: F3-ON recipe (the original v24a)
numbers/
  NEURIPS_NUMBERS.md      single source of truth for every paper number
evals/                    per-checkpoint evaluator output (v24a / v25a /
                          v27a / v27b{seed1..4} / v27c / v28a / v28b on the
                          patched lme_val_s512_evpos corpus)
```

## Reproducing the headline (v27b-seed1, ~1.5 h on 1× H100)

1. Acquire LongMemEval-S from the upstream release and pre-tokenise the
   training and validation splits to `paper_artifacts/chains/lme_train_s512.pt`
   and `paper_artifacts/chains/lme_val_s512_evpos.pt` (the validation
   corpus must include the `chain_evidence_positions` field).
2. Launch `recipes/v27b_v24a_no_probe_seed1_0p6b_frozen_local.sh`.
3. After ~1.5 h, evaluate with:
   ```
   python tools/eval_callback.py \\
     --ckpt Runs/<run_dir>/final \\
     --chain_pt paper_artifacts/chains/lme_val_s512_evpos.pt
   ```
   Expected `pa_cb_dnm = +0.797` for this seed; the four-seed mean is
   +1.323 ± 0.530 nats.

## Notes

- Backbone weights are frozen throughout training (`--freeze_backbone
  --lr_backbone 0`); the leak-control claim follows directly.
- The corpus filenames assume the layout described above; any
  reproducer is free to symlink or repath as long as the eval script
  finds `chain_evidence_positions`.
- The `v25a_seed5` checkpoint referenced by some ledger rows is the
  third 1.7B seed; including it would tighten the 1.7B mean if you
  budget the additional ~6 h of compute.
"""

P2_README = """# Six Failure Modes — Supplementary Material

This bundle contains the audit data, evaluator, and ledger needed to
reproduce the six failure-mode audits that motivate the methodology
contribution of the paper.

## Layout

```
code/
  src/                    architecture (referenced by the audits)
  tools/eval_callback.py  callback-aware evaluator (Failure Mode 1
                          dilution audit hinges on this vs eval_chain.py)
audits/                   the six audit reports + raw JSONs:
  audit_a1_window_leakage.{md,json}
  audit_a2_redaction.md, audit_a2_v15a.json, audit_a2_v15b.json
  audit_a3_base_prior.md, audit_a3_base_prior_1p7b_cloud.md
  audit_a3_data.json, audit_a3_template_prior.{md,cloud.json}
  audit_a_corpus_prior.md
  audit_b_literature.md
ledgers/
  NEURIPS_NUMBERS.md      v27b/v28 numbers cited as the surviving recipe
  runs.md                 active run ledger (post-flip)
eval_jsons/               per-checkpoint evaluator output for v14k / v15a /
                          v15b / v15e / v15f (the failure-mode audit
                          subjects) + v27b / v28 (the surviving recipe)
```

## Reproducing the audits

Each audit doc in `audits/` documents its own command line and inputs.
The `audit_a1_window_leakage.json` and `audit_a3_data.json` files are
self-contained (no external corpus reads).
"""

P3_README = """# Pair-Recipe Drop-in Primitive — Supplementary Material

This bundle contains the architecture, the pair-trainer, the three
routing-variant launcher scripts, and the figure source data needed to
reproduce the pair-recipe headline:

> A softly-initialised attention-residual router learns history-specific
> memory ~2× more sample-efficiently than a per-sublayer ReZero gate;
> a delta-source router with no parity-preserving init never recovers
> from its 34-nat init perturbation.

## Layout

```
code/
  src/modeling_memres.py    Memory Residuals architecture
  src/train_phase1.py       pair-based warmup trainer (used by this paper)
  src/presets.py            named (backbone, K, L_E, N) tuples
scripts/                    pair-corpus launchers:
  run_pair_h100_gpu0.sh
  run_pair_h100_gpu1.sh
  train_v11g_ap_baseline_gh200.sh   (AP soft-init baseline)
  train_v11h_ap_norm1_gh200.sh      (norm-1 variant)
  train_v11i_ap_pm4_gh200.sh        (±4 hard-bias variant)
  train_v11j_ap_carry_depth_gh200.sh
  train_v11k_ap_no_evidence_gh200.sh
figures/                    the three headline figures (PDF) + their
                            source pickles where applicable
README.md                   this file
```

## Reproducing the headline

Pre-tokenise PG-19 + TV dialogue chains per the paper's §3 description
(corpora are publicly available; the pair-corpus builder lives in
`tools/` of the parent repo). Launch any `train_v11*.sh` to reproduce
that variant's training trajectory, and the figure-3 trajectory.pdf
reproduces directly from the eval JSON dumps in `figures/`.
"""


# ---------------------------------------------------------------------------
# Paper specs
# ---------------------------------------------------------------------------
def build_p1(stage: Path) -> None:
    print(f"\n[P1] staging at {stage}")
    stage.mkdir(parents=True, exist_ok=True)

    stage_files(stage, [
        ("src", "code/src"),
        ("tools/eval_callback.py", "code/tools/eval_callback.py"),
        ("tools/eval_chain.py", "code/tools/eval_chain.py"),
        ("tools/eval_ttt_mc.py", "code/tools/eval_ttt_mc.py"),
        ("tests", "code/tests"),
        ("NEURIPS_NUMBERS.md", "numbers/NEURIPS_NUMBERS.md"),
    ])

    n_recipes = 0
    for pat in ("train_v27*.sh", "train_v28*.sh", "train_v24a*.sh"):
        n_recipes += stage_glob(stage, "Scripts", pat, "recipes")
    print(f"  staged {n_recipes} recipe scripts")

    n_evals = stage_glob(
        stage, "results/eval_v25_seed_pack_evpos", "*.json", "evals",
    )
    print(f"  staged {n_evals} eval JSONs")

    (stage / "README.md").write_text(P1_README, encoding="utf-8")


def build_p2(stage: Path) -> None:
    print(f"\n[P2] staging at {stage}")
    stage.mkdir(parents=True, exist_ok=True)

    stage_files(stage, [
        ("src", "code/src"),
        ("tools/eval_callback.py", "code/tools/eval_callback.py"),
        ("tools/eval_chain.py", "code/tools/eval_chain.py"),
        ("NEURIPS_NUMBERS.md", "ledgers/NEURIPS_NUMBERS.md"),
        ("results/exp2_chain_recipe/runs.md", "ledgers/runs.md"),
    ])

    n_audits = 0
    for pat in ("audit_*.md", "audit_*.json"):
        n_audits += stage_glob(
            stage, "results/exp2_chain_recipe", pat, "audits",
        )
    print(f"  staged {n_audits} audit files")

    n_evals = 0
    n_evals += stage_glob(stage, "results/eval_v14v15", "*.json", "eval_jsons")
    n_evals += stage_glob(
        stage, "results/eval_v25_seed_pack_evpos", "*.json", "eval_jsons",
    )
    print(f"  staged {n_evals} eval JSONs")

    (stage / "README.md").write_text(P2_README, encoding="utf-8")


def build_p3(stage: Path) -> None:
    print(f"\n[P3] staging at {stage}")
    stage.mkdir(parents=True, exist_ok=True)

    for f in ("modeling_memres.py", "train_phase1.py", "presets.py"):
        src = ROOT / "src" / f
        if src.exists():
            _safe_copy(src, stage / "code/src" / f)

    n_scripts = 0
    for pat in (
        "run_pair_*.sh",
        "train_v11g_*.sh",
        "train_v11h_*.sh",
        "train_v11i_*.sh",
        "train_v11j_*.sh",
        "train_v11k_*.sh",
        "train_v11l_*.sh",
    ):
        n_scripts += stage_glob(stage, "Scripts", pat, "scripts")
    print(f"  staged {n_scripts} pair-recipe scripts")

    figs = ROOT / "results/exp1_pair_recipe/figures"
    if figs.exists():
        _safe_copy(figs, stage / "figures")
        print(f"  staged figures dir")

    (stage / "README.md").write_text(P3_README, encoding="utf-8")


# ---------------------------------------------------------------------------
# Zip + verify
# ---------------------------------------------------------------------------
def zip_dir(src_dir: Path, out_zip: Path) -> tuple[int, int, str]:
    if out_zip.exists():
        out_zip.unlink()
    n_files = 0
    h = hashlib.sha256()
    with ZipFile(out_zip, "w", ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                arc = p.relative_to(src_dir.parent)
                zf.write(p, arc)
                h.update(p.read_bytes())
                n_files += 1
    size_mb = out_zip.stat().st_size / 1024 / 1024
    return n_files, int(size_mb * 100) / 100, h.hexdigest()[:12]


# ---------------------------------------------------------------------------
def main() -> int:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    specs = [
        ("memres",       "p1", build_p1),
        ("failure_modes", "p2", build_p2),
        ("pair_recipe",  "p3", build_p3),
    ]

    summary = []
    for name, stage_name, build_fn in specs:
        stage = BUILD / stage_name
        build_fn(stage)
        purged = _strip_caches(stage)
        if purged:
            print(f"  purged {purged} cache entries")
        seen, changed = anonymise_tree(stage)
        print(f"  anonymised {changed}/{seen} files")
        out_zip = OUT_DIR / f"{name}_supplementary.zip"
        n_files, size_mb, sha12 = zip_dir(stage, out_zip)
        cap_ok = "OK" if size_mb < 100 else "TOO LARGE"
        print(f"  -> {out_zip.name}: {n_files} files, {size_mb} MB ({cap_ok})")
        summary.append({
            "paper": name,
            "zip": out_zip.name,
            "files": n_files,
            "size_mb": size_mb,
            "sha256_12": sha12,
            "cap_ok": cap_ok,
        })

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
