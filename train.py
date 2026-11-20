"""Editable single-GPU Transformer baseline for autoresearch-evolution.

Only this file is modified by the evolutionary agent. Fixed data and metric
come from prepare.py; wall-clock training runs for exactly TIME_BUDGET seconds.
Run with: uv run train.py
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, evaluate_bpb, get_batch, load_tokens

# Fixed evaluation seed. Change architecture/training logic, not this value.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# All of these hyperparameters are valid mutation targets.
DEPTH = 6
WIDTH = 384
NUM_HEADS = 6
MLP_MULT = 4
DROPOUT = 0.0
LEARNING_RATE = 0.0006
MIN_LR_MULT = 0.12
WEIGHT_DECAY = 0.1
BATCH_SIZE = 8
GRAD_ACCUM = 2
WARMUP_FRACTION = 0.05
GRAD_CLIP = 1.0


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    depth: int = DEPTH
    width: int = WIDTH
    num_heads: int = NUM_HEADS
    mlp_mult: int = MLP_MULT
    dropout: float = DROPOUT


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight)


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.width % cfg.num_heads == 0
        self.heads = cfg.num_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)

    def forward(self, x):
        batch, context, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, context, self.heads, -1).transpose(1, 2)
        k = k.reshape(batch, context, self.heads, -1).transpose(1, 2)
        v = v.reshape(batch, context, self.heads, -1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(attended.transpose(1, 2).contiguous().view(batch, context, channels))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.width)
        self.attn = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.width)
        self.up = nn.Linear(cfg.width, cfg.width * cfg.mlp_mult, bias=False)
        self.down = nn.Linear(cfg.width * cfg.mlp_mult, cfg.width, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.down(F.silu(self.up(self.norm2(x)))))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.max_seq_len, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = RMSNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        _, length = tokens.shape
        if length > self.cfg.max_seq_len:
            raise ValueError("context exceeds fixed MAX_SEQ_LEN")
        positions = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU/CUDA required for five-minute autoresearch training")
    device = torch.device("cuda")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    total_start = time.monotonic()
    train_tokens = load_tokens("train")
    cfg = ModelConfig()
    model = Transformer(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    use_bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    generator = np.random.default_rng(SEED)
    # Explicit startup synchronization: starts the timed section only after
    # model and data initialization. Training itself is a fixed 300 seconds.
    torch.cuda.synchronize()
    training_start = time.monotonic()
    step = 0
    total_tokens = 0
    while True:
        elapsed = time.monotonic() - training_start
        if elapsed >= TIME_BUDGET:
            break
        progress = min(1.0, elapsed / TIME_BUDGET)
        if progress < WARMUP_FRACTION:
            multiplier = max(0.01, progress / max(1e-9, WARMUP_FRACTION))
        else:
            phase = (progress - WARMUP_FRACTION) / max(1e-9, 1.0 - WARMUP_FRACTION)
            multiplier = MIN_LR_MULT + (1.0 - MIN_LR_MULT) * 0.5 * (1.0 + math.cos(math.pi * phase))
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * multiplier
        optimizer.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(train_tokens, BATCH_SIZE, MAX_SEQ_LEN, device, generator)
            with torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            total_tokens += int(y.numel())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % 25 == 0:
            torch.cuda.synchronize()
            print(f"step={step} elapsed={time.monotonic()-training_start:.1f}s loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)
    torch.cuda.synchronize()
    training_seconds = time.monotonic() - training_start
    val_bpb = evaluate_bpb(model, batch_size=min(4, BATCH_SIZE))
    torch.cuda.synchronize()
    total_seconds = time.monotonic() - total_start
    params = sum(p.numel() for p in model.parameters())
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated()/1e6:.1f}")
    print("mfu_percent:      n/a")  # Not estimated for this simple baseline
    print(f"total_tokens_M:   {total_tokens/1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {params/1e6:.1f}")
    print(f"depth:            {cfg.depth}")


if __name__ == "__main__":
    main()
