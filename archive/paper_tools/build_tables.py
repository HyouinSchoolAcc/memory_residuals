#!/usr/bin/env python3
"""Update paper/memory_residuals_empirical.tex's results tables from
the latest *_eval.json files.

Edits two regions, demarcated by sentinel comments:

    % ---- BEGIN: standalone-eval table ----
    ...
    % ---- END: standalone-eval table ----

    % ---- BEGIN: headtohead trajectory table ----
    ...
    % ---- END: headtohead trajectory table ----

so the template is hand-editable everywhere else.

If a JSON we'd like to source from doesn't exist yet, the corresponding
row is filled with "---" so the paper still compiles.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_DIR = Path("paper_artifacts/eval")
PAPER_TEX = Path("paper/memory_residuals_empirical.tex")


def fmt(v: float | None, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "---"
    if sign:
        s = "+" if v >= 0 else ""
        return f"${s}{v:.4f}$"
    return f"${v:.4f}$"


def load_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def standalone_table_rows() -> list[str]:
    rows: list[str] = []
    sources = [
        ("output/chain_v3_softparity_full",
         EVAL_DIR / "chain_v3_softparity_full_eval.json",
         r"soft \textsc{attn\_parity}"),
        ("output/chain_v3_attentionbase_full",
         EVAL_DIR / "chain_v3_attentionbase_full_eval.json",
         r"\textsc{attn\_base}"),
        ("output/chain_v2_phaseA_softparity_b4",
         EVAL_DIR / "chain_v2_phaseA_softparity_b4_step2000_eval.json",
         r"soft \textsc{attn\_parity} (step 2000)"),
        ("output/chain_v2_abl_residual_mode",
         EVAL_DIR / "chain_v2_abl_residual_mode_step5200_eval.json",
         r"\textsc{simple\_gate} (step 5200)"),
    ]
    for ckpt_dir, eval_json, label in sources:
        blob = load_eval(eval_json)
        if not blob:
            continue
        for corpus_name, display in [
            ("pg19_val",  "PG-19 val"),
            ("pg19_test", "PG-19 test"),
            ("locomo",    "LoCoMo"),
        ]:
            m = blob.get(corpus_name) or {}
            if not m:
                continue
            d_nm = m.get("delta_nomem_minus_mem")
            d_sh = m.get("delta_shuffle_minus_mem")
            d_or = m.get("delta_oracle_minus_mem")
            cap = m.get("memory_capture_ratio")
            cap_s = f"${cap:.2f}$" if isinstance(cap, (int, float)) else "---"
            rows.append(
                f"  {label} & {display} & {fmt(d_nm, sign=True)} & "
                f"{fmt(d_sh, sign=True)} & {fmt(d_or, sign=True)} & {cap_s} \\\\"
            )
        rows.append(r"  \midrule")
    if rows and rows[-1].strip() == r"\midrule":
        rows.pop()
    return rows


def callback_table_rows() -> list[str]:
    rows: list[str] = []
    sources = [
        (EVAL_DIR / "chain_v3_softparity_full_callback.json",
         r"soft \textsc{attn\_parity} (step 6000)"),
        (EVAL_DIR / "chain_v3_attentionbase_full_callback.json",
         r"\textsc{attn\_base} (step 6000)"),
        (EVAL_DIR / "chain_v2_phaseA_softparity_b4_callback.json",
         r"soft \textsc{attn\_parity} (step 2000)"),
        (EVAL_DIR / "chain_v2_abl_residual_mode_callback.json",
         r"\textsc{simple\_gate} (step 5200)"),
    ]
    for path, label in sources:
        blob = load_eval(path)
        if not blob:
            rows.append(f"  {label} & --- & --- & --- \\\\")
            continue
        cb = blob.get("delta_callback_nomem_minus_mem")
        fl = blob.get("delta_filler_nomem_minus_mem")
        rt = blob.get("ratio_callback_over_filler_help")
        rt_s = f"${rt:.2f}\\times$" if isinstance(rt, (int, float)) else "---"
        rows.append(
            f"  {label} & {fmt(cb, sign=True)} & {fmt(fl, sign=True)} & {rt_s} \\\\"
        )
    return rows


SENTINEL_PATTERNS = [
    ("standalone", r"% ---- BEGIN: standalone-eval table ----",
                   r"% ---- END: standalone-eval table ----"),
]


def replace_block(tex: str, sentinel_begin: str, sentinel_end: str,
                  new_inside: str) -> str:
    pattern = re.compile(
        re.escape(sentinel_begin) + r".*?" + re.escape(sentinel_end),
        flags=re.DOTALL,
    )
    if not pattern.search(tex):
        return tex
    replacement = f"{sentinel_begin}\n{new_inside}\n{sentinel_end}"
    return pattern.sub(lambda _m: replacement, tex)


def main() -> None:
    if not PAPER_TEX.exists():
        print(f"[tables] paper tex not found at {PAPER_TEX}")
        return
    tex = PAPER_TEX.read_text(encoding="utf-8")

    standalone_rows = standalone_table_rows()
    standalone_inside = "\n".join(standalone_rows) if standalone_rows else \
        "  --- & --- & --- & --- & --- & --- \\\\"
    tex = replace_block(
        tex,
        "% ---- BEGIN: standalone-eval table ----",
        "% ---- END: standalone-eval table ----",
        standalone_inside,
    )

    callback_rows = callback_table_rows()
    callback_inside = "\n".join(callback_rows) if callback_rows else \
        "  --- & --- & --- & --- \\\\"
    tex = replace_block(
        tex,
        "% ---- BEGIN: callback-probe table ----",
        "% ---- END: callback-probe table ----",
        callback_inside,
    )

    PAPER_TEX.write_text(tex, encoding="utf-8")
    print(f"[tables] updated {PAPER_TEX}")


if __name__ == "__main__":
    main()
