"""
Train Qwen3-MemRes on paired (history, current) conversation data.

Memory Residuals require multi-session data: each training example
is a (history_ids, current_ids) pair where the model must use compressed
history to predict current_ids.

Data format (JSONL, one conversation-pair per line):
    {"history": "...", "current": "..."}

If no paired data is available, --simulate_sessions synthesizes pairs
by splitting FineWeb-Edu documents: the first half is treated as history,
the second half as the current session. This is an approximation but
trains the memory mechanism end-to-end.

Usage:
    # With simulated sessions (no paired data needed)
    torchrun --nproc_per_node=8 train_memres.py --simulate_sessions

    # With real multi-session data
    torchrun --nproc_per_node=8 train_memres.py --data_path conversations.jsonl

    # Memory + AttnRes (both depth-wise and cross-session)
    torchrun --nproc_per_node=8 train_memres.py --simulate_sessions --attnres_mode block
"""

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from modeling_memres import Qwen3MemResConfig, Qwen3MemResForCausalLM
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_path", default=None,
                   help="Path to JSONL file with {history, current} pairs. "
                        "If None, uses --dataset with --simulate_sessions.")
    p.add_argument("--simulate_sessions", action="store_true",
                   help="Simulate sessions by splitting FineWeb-Edu documents.")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="default")

    # Model
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1536)

    # AttnRes (depth-wise)
    p.add_argument("--attnres_mode", default="block", choices=["block", "full"],
                   help="Depth-wise AttnRes mode.")
    p.add_argument("--attnres_num_blocks", type=int, default=4)
    p.add_argument("--attnres_gate_type", default="bias",
                   choices=["bias", "sigmoid_scalar", "sigmoid_vector", "learnable_alpha"])

    # MemRes (cross-session)
    p.add_argument("--memres_num_memory_vectors", type=int, default=128,
                   help="K: number of latent memory slots.")
    p.add_argument("--memres_num_heads", type=int, default=8,
                   help="Attention heads in MemoryBlock and per-layer cross-attention.")
    p.add_argument("--memres_apply_at", default="attn", choices=["attn", "mlp", "both"],
                   help="Which sublayer to inject memory cross-attention.")
    p.add_argument("--memres_gate_init", type=float, default=-2.0,
                   help="Initial gate logit (σ(-2) ≈ 0.12 — near-zero at start).")

    # Sequence lengths
    p.add_argument("--history_len", type=int, default=1024,
                   help="Token length of history sequence fed to MemoryBlock.")
    p.add_argument("--current_len", type=int, default=1024,
                   help="Token length of current session (predicted).")

    # Training
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr_min", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--wandb_project", default="memory_residuals")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_with_warmup(step, warmup, total, lr_min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


def jsonl_session_stream(data_path, tokenizer, history_len, current_len, rank, world_size, seed):
    """Stream (history_ids, current_ids) pairs from a JSONL file."""
    import random
    rng = random.Random(seed + rank)

    with open(data_path) as f:
        lines = f.readlines()

    rng.shuffle(lines)
    lines = lines[rank::world_size]  # shard by rank

    while True:
        rng.shuffle(lines)
        for line in lines:
            obj = json.loads(line)
            history_text = obj.get("history", "")
            current_text = obj.get("current", "")
            if not history_text or not current_text:
                continue

            h_ids = tokenizer.encode(history_text, add_special_tokens=False)[:history_len]
            c_ids = tokenizer.encode(current_text, add_special_tokens=False)[:current_len + 1]

            if len(h_ids) < 16 or len(c_ids) < 2:
                continue

            # Pad history to history_len
            h_ids = h_ids + [tokenizer.eos_token_id] * (history_len - len(h_ids))
            yield (
                torch.tensor(h_ids[:history_len], dtype=torch.long),
                torch.tensor(c_ids[:current_len + 1], dtype=torch.long),
            )


def simulated_session_stream(dataset_name, config_name, tokenizer,
                             history_len, current_len, rank, world_size, seed):
    """
    Simulate multi-session pairs by splitting FineWeb-Edu documents.
    First half → history, second half → current session.
    This trains the memory mechanism without requiring paired dialogue data.
    """
    from datasets import load_dataset

    total_len = history_len + current_len + 1  # +1 for label shift
    ds = load_dataset(dataset_name, name=config_name, split="train",
                      streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed + rank, buffer_size=10_000)
    ds = ds.skip(rank)

    buf = []
    for sample in ds:
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)

        while len(buf) >= total_len:
            chunk = buf[:total_len]
            buf = buf[world_size * total_len:]

            h_ids = chunk[:history_len]
            c_ids = chunk[history_len:]  # length current_len + 1

            yield (
                torch.tensor(h_ids, dtype=torch.long),
                torch.tensor(c_ids, dtype=torch.long),
            )


def build_model(args, device):
    common = dict(
        vocab_size=151936,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=(args.history_len + args.current_len) * 2,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=args.hidden_size // args.num_heads,
        # AttnRes
        attnres_num_blocks=args.attnres_num_blocks,
        attnres_recency_bias_init=0.0,
        attnres_mode=args.attnres_mode,
        attnres_gate_type=args.attnres_gate_type,
        # MemRes
        memres_num_memory_vectors=args.memres_num_memory_vectors,
        memres_num_heads=args.memres_num_heads,
        memres_apply_at=args.memres_apply_at,
        memres_gate_init=args.memres_gate_init,
    )

    config = Qwen3MemResConfig(**common)
    model = Qwen3MemResForCausalLM(config)
    return model.to(dtype=torch.bfloat16, device=device)


def main():
    args = parse_args()

    if args.run_name is None:
        args.run_name = (
            f"memres-d{args.hidden_size}-L{args.num_layers}"
            f"-K{args.memres_num_memory_vectors}-{args.steps // 1000}k"
        )
    if args.out_dir is None:
        args.out_dir = f"./output/{args.run_name}"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)

    use_wandb = False
    if is_main:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config=vars(args),
            )
            use_wandb = True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")

    if is_main:
        print(f"Building MemRes model...")

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    if is_main:
        n_memres = sum(p.numel() for n, p in model.named_parameters()
                       if "memory_block" in n or "mem_cross" in n)
        print(f"Model: {n_params:.1f}M total params | MemRes params: {n_memres / 1e3:.1f}K")
        print(f"Memory vectors K={args.memres_num_memory_vectors} | "
              f"history_len={args.history_len} | current_len={args.current_len}")

    find_unused = args.attnres_gate_type != "bias"
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)

    optimizer = AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8,
    )
    lr_min_ratio = args.lr_min / args.lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio),
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    if args.data_path is not None:
        stream = jsonl_session_stream(
            args.data_path, tokenizer,
            args.history_len, args.current_len,
            rank, world_size, args.seed,
        )
    elif args.simulate_sessions:
        stream = simulated_session_stream(
            args.dataset, args.dataset_name, tokenizer,
            args.history_len, args.current_len,
            rank, world_size, args.seed,
        )
    else:
        raise ValueError("Provide --data_path or --simulate_sessions")

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()

    global_step = 0
    accum_step = 0
    accum_loss = 0.0
    t0 = time.time()
    tokens_seen = 0

    for history_ids, current_ids in stream:
        if global_step >= args.steps:
            break

        history_ids = history_ids.unsqueeze(0).to(device)   # (1, history_len)
        current_ids = current_ids.unsqueeze(0).to(device)   # (1, current_len + 1)
        input_ids = current_ids[:, :-1]                     # (1, current_len)
        labels = input_ids

        # Step 1: encode history → M_c
        # We run a forward pass over history_ids with no memory to get hidden states,
        # then compress them via MemoryBlock.
        with torch.no_grad():
            history_out = model.module.model(input_ids=history_ids)
            memory_state = model.module.model.compress_history(
                history_out.last_hidden_state
            )  # (1, K, d)

        # Step 2: train on current session conditioned on M_c
        out = model(input_ids=input_ids, labels=labels, memory_state=memory_state)
        loss = out.loss / args.grad_accum
        loss.backward()

        accum_loss += loss.item()
        accum_step += 1
        tokens_seen += args.current_len

        if accum_step < args.grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        accum_step = 0

        if global_step % args.log_every == 0:
            loss_t = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            if is_main:
                elapsed = time.time() - t0
                tok_sec = tokens_seen * world_size / elapsed
                avg_loss = loss_t.item()
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step {global_step:6d} | loss {avg_loss:.4f} | "
                    f"lr {lr_now:.2e} | grad_norm {grad_norm:.3f} | "
                    f"{tok_sec / 1e3:.1f}k tok/s"
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "train/grad_norm": grad_norm,
                        "train/tok_per_s": tok_sec,
                    }, step=global_step)

                tokens_seen = 0
                t0 = time.time()
        accum_loss = 0.0

        if is_main and global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"step-{global_step}")
            model.module.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint → {ckpt_dir}")

    if is_main:
        final_dir = os.path.join(args.out_dir, "final")
        model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Training done. Final model → {final_dir}")
        if use_wandb:
            import wandb
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
