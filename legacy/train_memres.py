"""
Train Qwen3-MemRes on three-segment recurrent conversation data.

Data format (JSONL, one example per line, three segments required):
    {"prior": "...", "history": "...", "current": "..."}

Each step compresses prior -> M_c_1 (judge sees zeros), then compresses
history with M_c_prev=M_c_1 -> M_c_2 (judge sees real prior memory --
this is the regime that actually trains the Forgetting Defense), then
forwards over current with M_c=M_c_2.  The legacy two-key schema
({"history","current"}) is no longer supported and will raise on load.

Rows that don't have enough tokens to fill all three segments at the
configured (--prior_len, --history_len, --current_len) are skipped
rather than padded.  This avoids polluting MemoryBlock extraction with
pad tokens; reach for shorter segment lengths if you need to keep more
of a corpus.

If no paired data is available, --simulate_sessions synthesizes
examples by splitting streamed documents into thirds.

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
from presets import PRESETS, apply_preset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", default=None)
    p.add_argument("--simulate_sessions", action="store_true")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="default")
    p.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS),
        help="Named training configuration. Pins --pretrained, "
        "--memres_num_vectors, --memres_extraction_depth, and "
        "--memres_num_blocks to a known-good combination. "
        "Overrides any manual values for those flags.",
    )
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

    p.add_argument("--memres_num_vectors", type=int, default=128)
    p.add_argument(
        "--memres_extraction_depth",
        type=int,
        default=0,
        help="L_E: Perceiver-style refinement layers on top of Eq. 1 "
        "(default 0 -> single-layer extraction).",
    )
    p.add_argument(
        "--memres_num_blocks",
        type=int,
        default=4,
        help="N: number of Block AttnRes blocks partitioning the L "
        "attention/MLP sublayers (Section 2.2, Eqs. 7-10). "
        "N == 2 * num_hidden_layers recovers Full AttnRes; N == 1 collapses "
        "to standard residuals over (b_{-1}, b_0).",
    )

    p.add_argument(
        "--prior_len",
        type=int,
        default=None,
        help="Length of the prior segment (compressed first; the judge sees "
        "zeros on this call). Defaults to --history_len.",
    )
    p.add_argument("--history_len", type=int, default=1024)
    p.add_argument(
        "--current_len",
        type=int,
        default=1024,
        help="Length of the current segment, used as both input and labels. "
        "HF's auto-shift means the number of scored positions is "
        "current_len - 1.",
    )

    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr_min", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument(
        "--train_memres_only",
        action="store_true",
        help="Freeze the pretrained backbone and optimize only Memory Residuals "
        "modules. Useful for 8B pilots on single H100s where full AdamW "
        "optimizer state does not fit.",
    )
    p.add_argument(
        "--detach_history_embeddings",
        action="store_true",
        help="Detach history token embeddings before memory compression to save memory. "
        "By default gradients flow end-to-end through the history embedding lookup.",
    )
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--wandb_project", default="memory_residuals")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.preset is not None:
        apply_preset(args, args.preset)
    if args.prior_len is None:
        args.prior_len = args.history_len
    return args


def cosine_with_warmup(step, warmup, total, lr_min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


class SessionStream:
    """Base for (prior_ids, history_ids, current_ids) producers, sharded across ranks."""

    def __init__(
        self, tokenizer, prior_len, history_len, current_len, rank, world_size, seed
    ):
        self.tokenizer = tokenizer
        self.prior_len = prior_len
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
        if not lines:
            raise ValueError(f"No rows in {self.data_path}")

        # Schema check on the first non-empty row. The legacy two-key schema
        # ({"history","current"}) is intentionally not back-compat: it had no
        # prior segment, which is the regime change this trainer exists to
        # introduce.
        first = next((json.loads(l) for l in lines if l.strip()), None)
        if first is None or "current" not in first or "history" not in first:
            raise ValueError(
                f"{self.data_path}: rows must include 'prior', 'history', "
                "'current' keys."
            )
        if "prior" not in first:
            raise ValueError(
                f"{self.data_path}: legacy two-key schema "
                "({'history','current'}) is not supported. Each row must "
                "include a 'prior' segment so the judge sees a real prior "
                "memory on the second compress call."
            )

        rng.shuffle(lines)
        lines = lines[self.rank :: self.world_size]

        tok = self.tokenizer
        P, H, C = self.prior_len, self.history_len, self.current_len
        while True:
            rng.shuffle(lines)
            for line in lines:
                if not line.strip():
                    continue
                obj = json.loads(line)
                pt = obj.get("prior", "")
                ht = obj.get("history", "")
                ct = obj.get("current", "")
                if not pt or not ht or not ct:
                    continue
                p_ids = tok.encode(pt, add_special_tokens=False)
                h_ids = tok.encode(ht, add_special_tokens=False)
                c_ids = tok.encode(ct, add_special_tokens=False)
                # Strict: only emit rows that fully fill all three segments.
                # The MemoryBlock has no key-padding mask so any pad tokens
                # in a partial segment would contaminate Stage 1 extraction.
                if len(p_ids) < P or len(h_ids) < H or len(c_ids) < C:
                    continue
                yield (
                    torch.tensor(p_ids[:P], dtype=torch.long),
                    torch.tensor(h_ids[:H], dtype=torch.long),
                    torch.tensor(c_ids[:C], dtype=torch.long),
                )


class SimulatedSessionStream(SessionStream):
    """Split streamed documents into prior/history/current thirds."""

    def __init__(self, dataset_name, config_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.config_name = config_name

    def __iter__(self):
        from datasets import load_dataset

        P, H, C = self.prior_len, self.history_len, self.current_len
        total_len = P + H + C
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
                    torch.tensor(chunk[:P], dtype=torch.long),
                    torch.tensor(chunk[P : P + H], dtype=torch.long),
                    torch.tensor(chunk[P + H : P + H + C], dtype=torch.long),
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
        self._apply_trainable_filter()
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
                f"-K{a.memres_num_vectors}-{a.steps // 1000}k"
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
            memres_num_vectors=a.memres_num_vectors,
            memres_extraction_depth=a.memres_extraction_depth,
            memres_num_blocks=a.memres_num_blocks,
        )

        if a.pretrained:
            from transformers import AutoConfig

            base_cfg = AutoConfig.from_pretrained(a.pretrained)
            base_dict = base_cfg.to_dict()
            # Preserve the upstream tie_word_embeddings so the trainer's
            # parameter count matches the advertised model size (e.g. 0.606B
            # for Qwen3-0.6B, which natively ties embed/lm_head).  Untying
            # silently adds ~vocab*d params (~155M for Qwen3-0.6B).
            config = Qwen3MemResConfig(**{**base_dict, **memres_kwargs})
            model = Qwen3MemResForCausalLM.from_pretrained(
                a.pretrained, config=config, dtype=torch.bfloat16
            )
            if self.is_main:
                tied = config.tie_word_embeddings
                print(
                    f"Loaded pretrained base: {a.pretrained}  "
                    f"(tie_word_embeddings={tied})"
                )
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

    def _apply_trainable_filter(self):
        if not self.args.train_memres_only:
            return
        trainable_markers = ("memory_block", "memory_readout", "depth_router")
        for name, param in self.model.named_parameters():
            param.requires_grad = any(marker in name for marker in trainable_markers)
        if self.is_main:
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            print(
                f"Training MemRes modules only: {n_trainable / 1e6:.2f}M / "
                f"{n_total / 1e9:.3f}B parameters require grad"
            )

    def _log_param_counts(self):
        if not self.is_main:
            return
        a = self.args
        named = list(self.model.named_parameters())
        n_total = sum(p.numel() for _, p in named)
        n_mb = sum(p.numel() for n, p in named if "memory_block" in n)
        n_mr = sum(p.numel() for n, p in named if "memory_readout" in n)
        n_dr = sum(p.numel() for n, p in named if "depth_router" in n)
        n_attn = sum(p.numel() for n, p in named if ".self_attn." in n)
        n_mlp = sum(p.numel() for n, p in named if ".mlp." in n)
        memres = n_mb + n_mr + n_dr
        backbone = n_total - memres
        L_layers = self.model.config.num_hidden_layers
        per_layer = (n_attn + n_mlp) / max(L_layers, 1)
        memres_vs_layer = memres / per_layer if per_layer else float("nan")

        preset_tag = f" | preset={a.preset}" if a.preset else ""
        print(
            f"Model: {n_total / 1e9:.3f}B total "
            f"({backbone / 1e9:.3f}B backbone + "
            f"{memres / 1e6:.2f}M MemRes) "
            f"| per-layer={per_layer / 1e6:.2f}M | "
            f"MemRes vs 1 layer = {memres_vs_layer:.2f}x{preset_tag}"
        )
        print(
            f"  MemoryBlock={n_mb / 1e6:.2f}M | "
            f"MemoryReadout={n_mr / 1e6:.2f}M | "
            f"BlockAttnResRouter={n_dr / 1e3:.1f}K"
        )
        print(
            f"  K={a.memres_num_vectors} | L_E={a.memres_extraction_depth} | "
            f"N_blocks={a.memres_num_blocks} | "
            f"prior_len={a.prior_len} | history_len={a.history_len} | "
            f"current_len={a.current_len}"
        )

    def _build_optimizer(self):
        a = self.args
        if a.lr_min > a.lr:
            raise ValueError(
                f"--lr_min ({a.lr_min}) must be <= --lr ({a.lr}); "
                "the cosine schedule decays from lr down to lr_min"
            )
        lr_min_ratio = a.lr_min / a.lr

        memres_markers = ("memory_block", "memory_readout", "depth_router")
        is_memres = lambda name: any(m in name for m in memres_markers)

        if a.pretrained:
            # Two groups: pretrained backbone at --lr, fresh MemRes modules
            # at 10 * --lr so they can keep up with a converged backbone.
            # Same cosine schedule applies to both (LambdaLR multiplies each
            # group's `initial_lr` by lr_lambda(step), so the 10x ratio is
            # preserved at every step).
            base_params, memres_params = [], []
            for n, p in self.ddp.named_parameters():
                if not p.requires_grad:
                    continue
                (memres_params if is_memres(n) else base_params).append(p)
            param_groups = [
                {"params": base_params, "lr": a.lr},
                {"params": memres_params, "lr": 10.0 * a.lr},
            ]
            optimizer = AdamW(
                param_groups,
                betas=(0.9, 0.95),
                weight_decay=0.1,
                eps=1e-8,
            )
            if self.is_main:
                n_base = sum(p.numel() for p in base_params)
                n_mem = sum(p.numel() for p in memres_params)
                print(
                    f"Optimizer: 2 groups | base={n_base / 1e6:.2f}M @ "
                    f"lr={a.lr:.2e} | memres={n_mem / 1e6:.2f}M @ "
                    f"lr={10.0 * a.lr:.2e}"
                )
        else:
            params = [p for p in self.ddp.parameters() if p.requires_grad]
            optimizer = AdamW(
                params,
                lr=a.lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
                eps=1e-8,
            )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda s: cosine_with_warmup(s, a.warmup, a.steps, lr_min_ratio),
        )
        return optimizer, scheduler

    def _build_stream(self):
        a = self.args
        common = dict(
            tokenizer=self.tokenizer,
            prior_len=a.prior_len,
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

    def _compute_memory(self, prior_ids, history_ids):
        """Two-step recurrent compression: prior -> M_c_1 -> history -> M_c.

        The first compress runs with M_c_prev=None so the judge sees zeros
        (this is the original training regime, retained as one of the two
        regimes the model now sees per step).  The second compress passes
        M_c_1 as M_c_prev so the judge competes a real prior memory against
        the new candidate -- this is the fix for issue #1, the regime the
        Forgetting Defense actually trains on.

        The detach_history_embeddings flag (off by default) keeps the cheaper
        behavior for memory-constrained runs by detaching the embedding
        lookup before compression; gradients still flow through the
        MemoryBlock parameters either way.
        """
        backbone = self.ddp.module.model
        C_prior = backbone.embed_tokens(prior_ids)
        C_hist = backbone.embed_tokens(history_ids)
        if self.args.detach_history_embeddings:
            C_prior = C_prior.detach()
            C_hist = C_hist.detach()
        M_c_1 = backbone.compress_session(C_prior)
        M_c = backbone.compress_session(C_hist, M_c_1)
        return M_c

    def _train_step(self, prior_ids, history_ids, current_ids):
        # Drop the off-by-one: input=current, labels=current. HF's auto-shift
        # gives current_len - 1 scored positions.
        input_ids = current_ids
        labels = current_ids
        M_c = self._compute_memory(prior_ids, history_ids)
        out = self.ddp(input_ids=input_ids, labels=labels, M_c=M_c)
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
        batch_p, batch_h, batch_c = [], [], []

        for prior_ids, history_ids, current_ids in self.stream:
            if global_step >= a.steps:
                break

            batch_p.append(prior_ids)
            batch_h.append(history_ids)
            batch_c.append(current_ids)
            if len(batch_h) < a.batch_size:
                continue

            prio = torch.stack(batch_p).to(self.device)
            hist = torch.stack(batch_h).to(self.device)
            curr = torch.stack(batch_c).to(self.device)
            batch_p, batch_h, batch_c = [], [], []

            accum_loss += self._train_step(prio, hist, curr)
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
