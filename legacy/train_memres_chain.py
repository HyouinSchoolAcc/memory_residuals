#!/usr/bin/env python3
"""Train Qwen3-MemRes on ordered session chains with recurrent memory carry.

This complements ``train_memres.py``. The original trainer consumes independent
``history/current`` pairs; this trainer samples ordered windows from PG-19/TV
chains and backpropagates through a short recurrent window, so loss at session
t+1 can train the memory written from session t.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig, AutoTokenizer

from modeling_memres import Qwen3MemResConfig, Qwen3MemResForCausalLM
from presets import PRESETS, apply_preset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chain_dir", action="append", required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--session_len", type=int, default=512)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--memory_dropout", type=float, default=0.10)
    p.add_argument("--detach_history_embeddings", action="store_true")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default="output/chain_memres")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1536)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--memres_num_vectors", type=int, default=128)
    p.add_argument("--memres_extraction_depth", type=int, default=0)
    p.add_argument("--memres_num_blocks", type=int, default=8)
    args = p.parse_args()
    if args.preset is not None:
        apply_preset(args, args.preset)
    return args


def cosine_with_warmup(step: int, warmup: int, total: int, lr_min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def render_row(row: dict[str, Any]) -> str:
    if "text" in row:
        return row["text"]
    if "turns" in row:
        return "\n".join(
            f"{turn.get('speaker') or 'Narration'}: {turn.get('text') or ''}"
            for turn in row["turns"]
            if turn.get("text")
        )
    return row.get("current") or ""


def load_chains(chain_dirs: list[str]) -> list[list[str]]:
    chains: list[list[str]] = []
    for chain_dir in chain_dirs:
        for path in sorted(Path(chain_dir).glob("*.jsonl")):
            rows = list(iter_jsonl(path))
            rows.sort(key=lambda r: int(r.get("session_index", r.get("episode_index", 0))))
            sessions = [render_row(row) for row in rows]
            sessions = [s for s in sessions if len(s) > 128]
            if len(sessions) >= 2:
                chains.append(sessions)
    if not chains:
        raise ValueError(f"No usable chains found in {chain_dirs}")
    return chains


class ChainSampler:
    def __init__(self, chains: list[list[str]], tokenizer, session_len: int, window: int, seed: int):
        self.chains = chains
        self.tokenizer = tokenizer
        self.session_len = session_len
        self.window = window
        self.rng = random.Random(seed)

    def sample_one(self) -> torch.Tensor:
        chain = self.rng.choice(self.chains)
        if len(chain) > self.window:
            start = self.rng.randrange(0, len(chain) - self.window)
        else:
            start = 0
        sessions = chain[start : start + self.window]
        encoded = []
        for text in sessions:
            ids = self.tokenizer.encode(text, add_special_tokens=False)[: self.session_len + 1]
            if len(ids) < 2:
                ids = ids + [self.tokenizer.eos_token_id] * (2 - len(ids))
            ids = ids + [self.tokenizer.eos_token_id] * (self.session_len + 1 - len(ids))
            encoded.append(ids[: self.session_len + 1])
        return torch.tensor(encoded, dtype=torch.long)

    def batch(self, batch_size: int) -> torch.Tensor:
        return torch.stack([self.sample_one() for _ in range(batch_size)], dim=0)


def build_model(args: argparse.Namespace) -> Qwen3MemResForCausalLM:
    memres_kwargs = dict(
        memres_num_vectors=args.memres_num_vectors,
        memres_extraction_depth=args.memres_extraction_depth,
        memres_num_blocks=args.memres_num_blocks,
    )
    if args.pretrained:
        base_cfg = AutoConfig.from_pretrained(args.pretrained)
        config = Qwen3MemResConfig(**{**base_cfg.to_dict(), **memres_kwargs})
        return Qwen3MemResForCausalLM.from_pretrained(args.pretrained, config=config, dtype=torch.bfloat16)
    head_dim = args.head_dim or (args.hidden_size // args.num_heads)
    config = Qwen3MemResConfig(
        vocab_size=151936,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.session_len * 2,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=head_dim,
        **memres_kwargs,
    )
    return Qwen3MemResForCausalLM(config)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    chains = load_chains(args.chain_dir)
    sampler = ChainSampler(chains, tokenizer, args.session_len, args.window, args.seed)

    model = build_model(args).to(dtype=torch.bfloat16, device=device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, args.warmup, args.steps, args.lr_min / args.lr),
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            batch = sampler.batch(args.batch_size).to(device)
            M_c = None
            window_loss = 0.0
            for t in range(args.window):
                current = batch[:, t]
                input_ids = current[:, :-1]
                labels = input_ids
                read_memory = M_c
                if M_c is not None and random.random() < args.memory_dropout:
                    read_memory = None
                out = model(input_ids=input_ids, labels=labels, M_c=read_memory)
                window_loss = window_loss + out.loss / args.window

                C = model.model.embed_tokens(input_ids)
                if args.detach_history_embeddings:
                    C = C.detach()
                M_c = model.model.compress_session(C, M_c)
            loss = window_loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            elapsed = max(time.time() - t0, 1e-6)
            tok = args.batch_size * args.grad_accum * args.window * args.session_len
            print(
                f"step {step:6d} | loss {accum_loss:.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | grad_norm {float(grad_norm):.3f} | "
                f"{tok / elapsed / 1e3:.1f}k tok/s"
            )
            t0 = time.time()
        if step % args.save_every == 0:
            ckpt = Path(args.out_dir) / f"step-{step}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
    final = Path(args.out_dir) / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Saved final checkpoint -> {final}")


if __name__ == "__main__":
    main()
