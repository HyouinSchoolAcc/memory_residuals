"""Audit script for v12d shell scripts: validates that every CLI flag
in the v12d shell scripts is recognized by train_chain.py / d5_ttt_readout.py
on the GH200, and that all referenced corpus/checkpoint paths exist.

Run via: ssh ubuntu@... 'cd ~/memory_residuals && python audit_v12d_scripts.py'
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SCRIPT_TRAIN = [
    "scripts/train_v12d_d4_frozen_gh200.sh",
    "scripts/train_v12d_d4_trained_gh200.sh",
]
SCRIPT_D5 = "scripts/run_v12d_d5_epilogue_gh200.sh"

CORPORA = [
    "paper_artifacts/chains/synthd4_persona_callback_train_s512.pt",
    "paper_artifacts/chains/synthd4_persona_callback_val_s512.pt",
]


def extract_flags(path: str, ignore: set[str] = frozenset()) -> set[str]:
    text = open(path).read()
    found = set(re.findall(r"--[a-z_][a-z_0-9]*", text))
    return found - ignore


def known_train_flags() -> set[str]:
    src = open("train_chain.py").read()
    return set(re.findall(r"add_argument\(\s*['\"](--[a-z_][a-z_0-9]*)", src))


def known_d5_flags() -> set[str]:
    src = open("d5_ttt_readout.py").read()
    return set(re.findall(r"add_argument\(\s*['\"](--[a-z_][a-z_0-9]*)", src))


def writer_kind_choices() -> list[str]:
    src = open("modeling_memres.py").read()
    m = re.search(r"_VALID_WRITER_KINDS\s*=\s*\(([^)]*)\)", src)
    if not m:
        return []
    return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]


def main() -> int:
    fails = 0

    print("=" * 60)
    print("AUDIT 1 -- train_chain CLI flags")
    print("=" * 60)
    train_known = known_train_flags()
    print(f"  train_chain.py exposes {len(train_known)} flags")
    for path in SCRIPT_TRAIN:
        used = extract_flags(path)
        unknown = used - train_known
        status = "PASS" if not unknown else "FAIL"
        print(f"  {status}  {path}: {len(used)} flags, unknown={sorted(unknown)}")
        if unknown:
            fails += 1

    print()
    print("=" * 60)
    print("AUDIT 2 -- d5 CLI flags")
    print("=" * 60)
    d5_known = known_d5_flags()
    print(f"  d5_ttt_readout.py argparse exposes: {sorted(d5_known)}")
    used = extract_flags(SCRIPT_D5)
    unknown = used - d5_known
    status = "PASS" if not unknown else "FAIL"
    print(f"  {status}  {SCRIPT_D5}: {len(used)} flags, unknown={sorted(unknown)}")
    if unknown:
        fails += 1

    print()
    print("=" * 60)
    print("AUDIT 3 -- corpus paths exist")
    print("=" * 60)
    for c in CORPORA:
        exists = os.path.isfile(c)
        size = os.path.getsize(c) if exists else 0
        status = "PASS" if exists else "FAIL"
        print(f"  {status}  {c}  ({size / 1e6:.1f} MB)" if exists
              else f"  {status}  {c}  MISSING")
        if not exists:
            fails += 1

    print()
    print("=" * 60)
    print("AUDIT 4 -- writer_kind choices")
    print("=" * 60)
    choices = writer_kind_choices()
    expected = {"original", "slot_attention", "slot_attention_full"}
    if set(choices) == expected:
        print(f"  PASS  --memres_writer_kind choices = {choices}")
    else:
        print(f"  FAIL  --memres_writer_kind choices = {choices}, expected {expected}")
        fails += 1

    print()
    print("=" * 60)
    print("AUDIT 5 -- python files importable")
    print("=" * 60)
    for mod in ["train_chain", "modeling_memres", "presets"]:
        try:
            __import__(mod)
            print(f"  PASS  import {mod}")
        except Exception as e:
            print(f"  FAIL  import {mod}: {e}")
            fails += 1

    print()
    print("=" * 60)
    print("AUDIT 6 -- queue specs reference existing scripts")
    print("=" * 60)
    queue_dir = "paper_tools/cloud_watchdog/queue"
    if os.path.isdir(queue_dir):
        import json
        for fn in sorted(os.listdir(queue_dir)):
            try:
                spec = json.load(open(os.path.join(queue_dir, fn)))
                cmd = spec.get("cmd", "")
                m = re.search(r"bash (scripts/\S+)", cmd)
                if m:
                    sp = m.group(1)
                    exists = os.path.isfile(sp)
                    status = "PASS" if exists else "FAIL"
                    print(f"  {status}  {fn[:60]:60s}  -> {sp}")
                    if not exists:
                        fails += 1
                else:
                    print(f"  ?     {fn}: cmd={cmd!r}")
            except Exception as e:
                print(f"  FAIL  {fn}: {e}")
                fails += 1

    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if fails == 0 else f'{fails} FAILURES'}")
    print("=" * 60)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
