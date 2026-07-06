from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 32
    num_heads: int = 2
    num_layers: int = 1
    ffn_hidden: int = 128
    context_length: int = 16


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype) / head_dim)
    )
    angles = torch.outer(positions, inv_freq)
    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]

    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = (even * cos) - (odd * sin)
    rotated[..., 1::2] = (even * sin) + (odd * cos)
    return rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        mask = torch.triu(torch.ones(config.context_length, config.context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        q = apply_rope(self.split_heads(self.q_proj(x)))
        k = apply_rope(self.split_heads(self.k_proj(x)))
        v = self.split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = weights @ v
        return self.out_proj(self.combine_heads(output))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_hidden),
            nn.GELU(),
            nn.Linear(config.ffn_hidden, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AdditionTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"seq_len={seq_len} exceeds context_length={self.config.context_length}")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
