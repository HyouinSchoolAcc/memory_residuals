#!/usr/bin/env python3
"""Per-callback NLL probe (PITFALLS.md §3, the *killer experiment*).

For each (history, current) pair we identify "callback" tokens in the
current session whose underlying word also appears in the history (a simple
proper-noun heuristic captures named-entity callbacks).  We then report
mean cross-entropy at these callback positions with vs. without memory,
and the same numbers for filler positions.

The architecture passes the test if memory helps callback tokens more than
filler tokens -- i.e. (NLL_off - NLL_on) is larger on callback positions
than on filler positions.  Aggregate NLL improvements that are not driven
by callback positions indicate the memory has learned style/genre cues
rather than episodic content.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_memres import Qwen3MemResForCausalLM  # noqa: E402


WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9'\-]{3,}\b")


def history_words(history: str, min_capitalised_count: int = 1) -> set[str]:
    """Return the set of multi-character words to call back on.

    Default: only words that appear at least once in the history with an
    upper-case first letter (named entities) AND have >=4 characters.
    """
    counts: dict[str, int] = {}
    cap_counts: dict[str, int] = {}
    for word in WORD_RE.findall(history):
        key = word.lower()
        counts[key] = counts.get(key, 0) + 1
        if word[0].isupper():
            cap_counts[key] = cap_counts.get(key, 0) + 1
    return {w for w in counts if cap_counts.get(w, 0) >= min_capitalised_count}


def label_callback_tokens(
    tokenizer,
    current_text: str,
    callback_words: set[str],
    max_len: int,
) -> tuple[list[int], list[bool]]:
    """Tokenize current_text, return (token_ids, is_callback) of length up to max_len.

    is_callback[i] is True iff token i is part of a word found in callback_words.
    Word boundaries inside Qwen tokenization are detected by re-decoding each
    token and matching against the running word stream.
    """
    enc = tokenizer(
        current_text, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = enc["input_ids"][:max_len]
    offsets = enc["offset_mapping"][:max_len]
    flags: list[bool] = [False] * len(ids)
    if not callback_words:
        return ids, flags
    for word_match in WORD_RE.finditer(current_text):
        if word_match.group(0).lower() not in callback_words:
            continue
        ws, we = word_match.start(), word_match.end()
        for i, (ts, te) in enumerate(offsets):
            if ts >= we:
                break
            if te <= ws:
                continue
            flags[i] = True
    return ids, flags


@torch.no_grad()
def per_position_nll(
    model: Qwen3MemResForCausalLM,
    input_ids: torch.Tensor,
    M_c: torch.Tensor | None,
) -> torch.Tensor:
    out = model(input_ids=input_ids, M_c=M_c)
    logits = out.logits  # (B, S, V)
    targets = input_ids[:, 1:]  # next-token targets
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.squeeze(0)  # (S-1,)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", type=Path, required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--history_len", type=int, default=1024)
    p.add_argument("--current_len", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output", type=Path, default=Path("paper_artifacts/eval/callback_probe.json")
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model_path)
    model = (
        Qwen3MemResForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )

    samples: list[dict] = []
    with args.data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(samples) >= args.num_samples:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    cb_mem_nll, cb_no_nll = [], []
    fl_mem_nll, fl_no_nll = [], []
    sh_cb_nll, sh_fl_nll = [], []
    n_cb_total, n_fl_total = 0, 0
    encoded_histories: list[torch.Tensor] = []
    for s in samples:
        h_text = s.get("history") or ""
        c_text = s.get("current") or ""
        if not h_text or not c_text:
            continue
        h_ids = tok.encode(h_text, add_special_tokens=False)[-args.history_len :]
        if len(h_ids) < 16:
            continue
        h_ids = h_ids + [tok.eos_token_id] * (args.history_len - len(h_ids))
        encoded_histories.append(
            torch.tensor(h_ids[: args.history_len], dtype=torch.long)
        )

    for idx, s in enumerate(tqdm(samples, desc="probe")):
        h_text = s.get("history") or ""
        c_text = s.get("current") or ""
        if not h_text or not c_text:
            continue
        cb_words = history_words(h_text)
        c_ids, flags = label_callback_tokens(tok, c_text, cb_words, args.current_len + 1)
        if len(c_ids) < 4:
            continue
        c_ids_t = torch.tensor(c_ids, dtype=torch.long, device=args.device).unsqueeze(0)
        h_ids_t = encoded_histories[idx % len(encoded_histories)].unsqueeze(0).to(args.device)
        sh_idx = (idx + 1) % len(encoded_histories)
        sh_ids_t = encoded_histories[sh_idx].unsqueeze(0).to(args.device)

        M_c = model.model.compute_memory(h_ids_t)
        M_sh = model.model.compute_memory(sh_ids_t)
        nll_mem = per_position_nll(model, c_ids_t, M_c)
        nll_no = per_position_nll(model, c_ids_t, None)
        nll_sh = per_position_nll(model, c_ids_t, M_sh)

        flags_t = torch.tensor(flags[1:], device=args.device)
        for nll, mem_buf, fl_buf in [
            (nll_mem, cb_mem_nll, fl_mem_nll),
            (nll_no, cb_no_nll, fl_no_nll),
        ]:
            mem_buf.extend(nll[flags_t].float().cpu().tolist())
            fl_buf.extend(nll[~flags_t].float().cpu().tolist())
        sh_cb_nll.extend(nll_sh[flags_t].float().cpu().tolist())
        sh_fl_nll.extend(nll_sh[~flags_t].float().cpu().tolist())

        n_cb_total += int(flags_t.sum())
        n_fl_total += int((~flags_t).sum())

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    metrics = {
        "n_callback_tokens": n_cb_total,
        "n_filler_tokens": n_fl_total,
        "callback_nll_mem": mean(cb_mem_nll),
        "callback_nll_nomem": mean(cb_no_nll),
        "callback_nll_shuffle": mean(sh_cb_nll),
        "filler_nll_mem": mean(fl_mem_nll),
        "filler_nll_nomem": mean(fl_no_nll),
        "filler_nll_shuffle": mean(sh_fl_nll),
        "delta_callback_nomem_minus_mem": mean(cb_no_nll) - mean(cb_mem_nll),
        "delta_filler_nomem_minus_mem": mean(fl_no_nll) - mean(fl_mem_nll),
        "ratio_callback_over_filler_help": (
            (mean(cb_no_nll) - mean(cb_mem_nll))
            / max(1e-6, mean(fl_no_nll) - mean(fl_mem_nll))
        ),
        "delta_callback_shuffle_minus_mem": mean(sh_cb_nll) - mean(cb_mem_nll),
        "delta_filler_shuffle_minus_mem": mean(sh_fl_nll) - mean(fl_mem_nll),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
