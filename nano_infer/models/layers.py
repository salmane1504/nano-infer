# Attention, MLP, RoPE primitives
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from nano_infer.config import ModelConfig

if TYPE_CHECKING:
    from nano_infer.cache.kv_cache import KVCache


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Computes cos/sin tables for given positions (half-split layout, as in HF)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        inv_freq = self._apply_scaling(inv_freq, config.rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _apply_scaling(inv_freq: torch.Tensor, scaling: dict | None) -> torch.Tensor:
        if scaling is None:
            return inv_freq
        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type in (None, "default"):
            return inv_freq
        if rope_type != "llama3":
            raise NotImplementedError(f"rope scaling '{rope_type}' is not supported")

        factor = scaling["factor"]
        low = scaling["low_freq_factor"]
        high = scaling["high_freq_factor"]
        orig = scaling["original_max_position_embeddings"]

        wavelen = 2 * math.pi / inv_freq
        scaled = torch.where(wavelen > orig / low, inv_freq / factor, inv_freq)
        smooth = (orig / wavelen - low) / (high - low)
        smoothed = (1 - smooth) * inv_freq / factor + smooth * inv_freq
        medium = (wavelen <= orig / low) & (wavelen >= orig / high)
        return torch.where(medium, smoothed, scaled)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, heads, T, D); cos, sin: (T, D)."""
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, kv_heads, S, D) -> (B, kv_heads * n_rep, S, D)."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = config.num_kv_groups

        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v, start_pos)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        S = k.shape[2]
        if T == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        elif S == T:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Chunked prefill: queries sit at the end of the S cached positions.
            q_pos = torch.arange(S - T, S, device=x.device)[:, None]
            mask = torch.arange(S, device=x.device)[None, :] <= q_pos
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    """SwiGLU feed-forward."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
