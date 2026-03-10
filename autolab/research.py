"""
AutoLab Research Training Pipeline
=========================

Single-GPU research training script used by the AutoLab
Autonomous AI Research Laboratory.

AutoLab agents use this script to run controlled experiments
under a fixed time budget.

The goal is not large-scale training, but **rapid research
iteration**.

Agents can autonomously modify:

• model architecture
• optimizer parameters
• training schedules
• hyperparameters

and evaluate results automatically.

Usage
-----

    python train.py

Design Principles
-----------------

AutoLab is designed for **AI-driven research loops**:

    Idea → Experiment → Evaluation → Iteration

This script implements the **Experiment stage**.

"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel

from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    make_dataloader,
    evaluate_bpb,
)

# =============================================================================
# Kernel Setup
# =============================================================================

cap = torch.cuda.get_device_capability()

repo = (
    "varunneal/flash-attention-3"
    if cap == (9, 0)
    else "kernels-community/flash-attn3"
)

fa3 = get_kernel(repo).flash_attn_interface

# =============================================================================
# Logging
# =============================================================================


def log(msg):
    print(f"[AutoLab] {msg}")


# =============================================================================
# GPT Model Configuration
# =============================================================================


@dataclass
class GPTConfig:

    sequence_len: int = 2048
    vocab_size: int = 32768

    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

    window_pattern: str = "SSSL"


# =============================================================================
# Model Components
# =============================================================================


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):

    d = x.shape[3] // 2

    x1 = x[..., :d]
    x2 = x[..., d:]

    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    return torch.cat([y1, y2], dim=3)


# =============================================================================
# Attention
# =============================================================================


class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx):

        super().__init__()

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd

        self.head_dim = self.n_embd // self.n_head

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = 32

        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin, window_size):

        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:

            ve = ve.view(B, T, self.n_kv_head, self.head_dim)

            gate = 2 * torch.sigmoid(
                self.ve_gate(x[..., :self.ve_gate_channels])
            )

            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = norm(q)
        k = norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        y = y.contiguous().view(B, T, -1)

        return self.c_proj(y)


# =============================================================================
# Feed Forward
# =============================================================================


class MLP(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):

        x = self.c_fc(x)

        x = F.relu(x).square()

        return self.c_proj(x)


# =============================================================================
# Transformer Block
# =============================================================================


class Block(nn.Module):

    def __init__(self, config, layer_idx):

        super().__init__()

        self.attn = CausalSelfAttention(config, layer_idx)

        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):

        x = x + self.attn(norm(x), ve, cos_sin, window_size)

        x = x + self.mlp(norm(x))

        return x


# =============================================================================
# GPT Model
# =============================================================================


class GPT(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, i) for i in range(config.n_layer)]
                ),
            }
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):

        x = self.transformer.wte(idx)

        for block in self.transformer.h:

            x = block(x, None, None, None)

        logits = self.lm_head(x)

        if targets is None:
            return logits

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )


# =============================================================================
# Hyperparameters
# =============================================================================

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**19

DEPTH = 8

DEVICE_BATCH_SIZE = 128

# =============================================================================
# Setup
# =============================================================================

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda")

autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = Tokenizer.from_directory()

vocab_size = tokenizer.get_vocab_size()

log(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):

    base_dim = depth * ASPECT_RATIO

    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM

    num_heads = model_dim // HEAD_DIM

    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)

log(f"Model config: {asdict(config)}")

model = GPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_loader = make_dataloader(
    tokenizer,
    DEVICE_BATCH_SIZE,
    MAX_SEQ_LEN,
    "train",
)

x, y, epoch = next(train_loader)

log(f"Time budget: {TIME_BUDGET}s")


# =============================================================================
# Training Loop
# =============================================================================

t_start = time.time()

step = 0

total_training_time = 0

while True:

    torch.cuda.synchronize()

    t0 = time.time()

    with autocast_ctx:

        loss = model(x, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    x, y, epoch = next(train_loader)

    torch.cuda.synchronize()

    dt = time.time() - t0

    total_training_time += dt

    log(
        f"step {step:05d} | "
        f"loss {loss.item():.4f} | "
        f"time {dt:.2f}s | "
        f"epoch {epoch}"
    )

    step += 1

    if total_training_time >= TIME_BUDGET:

        break


# =============================================================================
# Evaluation
# =============================================================================

log("Running final evaluation")

model.eval()

with autocast_ctx:

    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

log(f"Validation BPB: {val_bpb:.6f}")

log("Experiment finished")

