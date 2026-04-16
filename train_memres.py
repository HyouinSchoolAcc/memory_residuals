"""
Train Qwen3-MemRes on paired (history, current) conversation data.

Data format (JSONL, one conversation-pair per line):
    {"history": "...", "current": "..."}

If no paired data is available, --simulate_sessions synthesizes pairs
by splitting documents: first half = history, second half = current.

Usage:
    torchrun --nproc_per_node=8 train_memres.py --simulate_sessions
    torchrun --nproc_per_node=8 train_memres.py --data_path conversations.jsonl
"""

import argparse
import json
import math
import os
import random
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

    p.add_argument("--data_path", default=None)
    p.add_argument("--simulate_sessions", action="store_true")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="default")
    p.add_argument(
        "--pretrained",
        default=None,
        help="HF model id or path to load pretrained Qwen3 weights onto; "
        "MemRes params stay randomly initialized. Ignores size flags.",
    )

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1536)
    p.add_argument("--head_dim", type=int, default=None)

    p.add_argument("--memres_num_memory_vectors", type=int, default=128)
    p.add_argument("--memres_num_memory_heads", type=int, default=8)
    p.add_argument("--memres_apply_at", default="both", choices=["attn", "mlp", "both"])
    p.add_argument("--memres_use_gate", action="store_true")
    p.add_argument("--memres_gate_init", type=float, default=-2.0)

    p.add_argument("--history_len", type=int, default=1024)
    p.add_argument("--current_len", type=int, default=1024)

    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=4)
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


class SessionStream:
    """Base for (history_ids, current_ids) producers, sharded across ranks."""

    def __init__(self, tokenizer, history_len, current_len, rank, world_size, seed):
        self.tokenizer = tokenizer
        self.history_len = history_len
        self.current_len = current_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def __iter__(self):
        raise NotImplementedError


class JsonlSessionStream(SessionStream):
    def __init__(self, data_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path

    def __iter__(self):
        rng = random.Random(self.seed + self.rank)
        with open(self.data_path) as f:
            lines = f.readlines()
        rng.shuffle(lines)
        lines = lines[self.rank :: self.world_size]

        tok = self.tokenizer
        H, C = self.history_len, self.current_len
        while True:
            rng.shuffle(lines)
            for line in lines:
                obj = json.loads(line)
                ht, ct = obj.get("history", ""), obj.get("current", "")
                if not ht or not ct:
                    continue
                h_ids = tok.encode(ht, add_special_tokens=False)[:H]
                c_ids = tok.encode(ct, add_special_tokens=False)[: C + 1]
                if len(h_ids) < 16 or len(c_ids) < 2:
                    continue
                h_ids = h_ids + [tok.eos_token_id] * (H - len(h_ids))
                yield (
                    torch.tensor(h_ids[:H], dtype=torch.long),
                    torch.tensor(c_ids[: C + 1], dtype=torch.long),
                )


class SimulatedSessionStream(SessionStream):
    """Split streamed documents into first-half/second-half pairs."""

    def __init__(self, dataset_name, config_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.config_name = config_name

    def __iter__(self):
        from datasets import load_dataset

        total_len = self.history_len + self.current_len + 1
        ds = load_dataset(
            self.dataset_name, name=self.config_name, split="train", streaming=True
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        ds = ds.shard(num_shards=self.world_size, index=self.rank)

        buf = []
        for sample in ds:
            text = sample.get("text") or sample.get("content") or ""
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.tokenizer.eos_token_id)
            buf.extend(ids)

            while len(buf) >= total_len:
                chunk = buf[:total_len]
                buf = buf[total_len:]
                yield (
                    torch.tensor(chunk[: self.history_len], dtype=torch.long),
                    torch.tensor(chunk[self.history_len :], dtype=torch.long),
                )


class Trainer:
    def __init__(self, args):
        self.args = args
        self._init_distributed()
        self._resolve_names()
        torch.manual_seed(args.seed + self.rank)

        self.use_wandb = self._init_wandb()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.model = self._build_model()
        self._log_param_counts()
        self.ddp = DDP(
            self.model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        self.optimizer, self.scheduler = self._build_optimizer()
        self.stream = self._build_stream()
        os.makedirs(args.out_dir, exist_ok=True)

    def _init_distributed(self):
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.is_main = self.rank == 0

    def _resolve_names(self):
        a = self.args
        if a.run_name is None:
            a.run_name = (
                f"memres-d{a.hidden_size}-L{a.num_layers}"
                f"-K{a.memres_num_memory_vectors}-{a.steps // 1000}k"
            )
        if a.out_dir is None:
            a.out_dir = f"./output/{a.run_name}"

    def _init_wandb(self):
        if not self.is_main:
            return False
        try:
            import wandb

            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.run_name,
                config=vars(self.args),
            )
            return True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")
            return False

    def _build_model(self):
        a = self.args
        memres_kwargs = dict(
            memres_num_memory_vectors=a.memres_num_memory_vectors,
            memres_num_memory_heads=a.memres_num_memory_heads,
            memres_apply_at=a.memres_apply_at,
            memres_use_gate=a.memres_use_gate,
            memres_gate_init=a.memres_gate_init,
        )

        if a.pretrained:
            from transformers import AutoConfig

            base_cfg = AutoConfig.from_pretrained(a.pretrained)
            base_dict = base_cfg.to_dict()
            base_dict["tie_word_embeddings"] = False
            config = Qwen3MemResConfig(**{**base_dict, **memres_kwargs})
            model = Qwen3MemResForCausalLM.from_pretrained(
                a.pretrained, config=config, dtype=torch.bfloat16
            )
            if self.is_main:
                print(f"Loaded pretrained base: {a.pretrained}")
        else:
            head_dim = a.head_dim or (a.hidden_size // a.num_heads)
            config = Qwen3MemResConfig(
                vocab_size=151936,
                hidden_size=a.hidden_size,
                num_hidden_layers=a.num_layers,
                num_attention_heads=a.num_heads,
                num_key_value_heads=a.num_kv_heads,
                intermediate_size=a.intermediate_size,
                max_position_embeddings=(a.history_len + a.current_len) * 2,
                rms_norm_eps=1e-6,
                tie_word_embeddings=True,
                head_dim=head_dim,
                **memres_kwargs,
            )
            model = Qwen3MemResForCausalLM(config)
        return model.to(dtype=torch.bfloat16, device=self.device)

    def _log_param_counts(self):
        if not self.is_main:
            return
        a = self.args
        n_total = sum(p.numel() for p in self.model.parameters()) / 1e6
        n_memres = sum(
            p.numel()
            for n, p in self.model.named_parameters()
            if "memory_block" in n or "memres_attn" in n or "memres_mlp" in n
        )
        print(f"Model: {n_total:.1f}M total | MemRes: {n_memres / 1e3:.1f}K")
        print(
            f"K={a.memres_num_memory_vectors} | history_len={a.history_len} | "
            f"current_len={a.current_len}"
        )

    def _build_optimizer(self):
        a = self.args
        optimizer = AdamW(
            self.ddp.parameters(),
            lr=a.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8,
        )
        lr_min_ratio = a.lr_min / a.lr
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda s: cosine_with_warmup(s, a.warmup, a.steps, lr_min_ratio),
        )
        return optimizer, scheduler

    def _build_stream(self):
        a = self.args
        common = dict(
            tokenizer=self.tokenizer,
            history_len=a.history_len,
            current_len=a.current_len,
            rank=self.rank,
            world_size=self.world_size,
            seed=a.seed,
        )
        if a.data_path is not None:
            return iter(JsonlSessionStream(a.data_path, **common))
        if a.simulate_sessions:
            return iter(SimulatedSessionStream(a.dataset, a.dataset_name, **common))
        raise ValueError("Provide --data_path or --simulate_sessions")

    def _compute_memory_state(self, history_ids):
        # History encoder runs without grad; compress_history keeps grad
        # on the MemoryBlock so it learns end-to-end.
        with torch.no_grad():
            h_out = self.ddp.module.model(input_ids=history_ids)
        return self.ddp.module.model.compress_history(h_out.last_hidden_state.detach())

    def _train_step(self, history_ids, current_ids):
        input_ids = current_ids[:, :-1]
        labels = input_ids
        memory_state = self._compute_memory_state(history_ids)
        out = self.ddp(input_ids=input_ids, labels=labels, memory_state=memory_state)
        loss = out.loss / self.args.grad_accum
        loss.backward()
        return loss.item()

    def _log(self, step, accum_loss, grad_norm, tokens_seen, t0):
        loss_t = torch.tensor(accum_loss, device=self.device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        if not self.is_main:
            return
        elapsed = time.time() - t0
        tok_sec = tokens_seen * self.world_size / elapsed
        lr_now = self.scheduler.get_last_lr()[0]
        avg_loss = loss_t.item()
        print(
            f"step {step:6d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | "
            f"grad_norm {grad_norm:.3f} | {tok_sec / 1e3:.1f}k tok/s"
        )
        if self.use_wandb:
            import wandb

            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/lr": lr_now,
                    "train/grad_norm": grad_norm,
                    "train/tok_per_s": tok_sec,
                },
                step=step,
            )

    def _save(self, tag: str):
        if not self.is_main:
            return
        ckpt_dir = os.path.join(self.args.out_dir, tag)
        self.ddp.module.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved checkpoint -> {ckpt_dir}")

    def fit(self):
        a = self.args
        self.ddp.train()
        self.optimizer.zero_grad()

        global_step = 0
        accum_step = 0
        accum_loss = 0.0
        tokens_seen = 0
        t0 = time.time()
        batch_h, batch_c = [], []

        for history_ids, current_ids in self.stream:
            if global_step >= a.steps:
                break

            batch_h.append(history_ids)
            batch_c.append(current_ids)
            if len(batch_h) < a.batch_size:
                continue

            hist = torch.stack(batch_h).to(self.device)
            curr = torch.stack(batch_c).to(self.device)
            batch_h, batch_c = [], []

            accum_loss += self._train_step(hist, curr)
            accum_step += 1
            tokens_seen += a.current_len * a.batch_size

            if accum_step < a.grad_accum:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ddp.parameters(), a.max_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            global_step += 1
            accum_step = 0

            if global_step % a.log_every == 0:
                self._log(global_step, accum_loss, grad_norm, tokens_seen, t0)
                tokens_seen = 0
                t0 = time.time()
            accum_loss = 0.0

            if global_step % a.save_every == 0:
                self._save(f"step-{global_step}")

        self._save("final")
        if self.is_main and self.use_wandb:
            import wandb

            wandb.finish()
        dist.destroy_process_group()


def main():
    Trainer(parse_args()).fit()


if __name__ == "__main__":
    main()
