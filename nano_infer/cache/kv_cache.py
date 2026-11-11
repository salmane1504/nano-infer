# Static/dynamic Key-Value cache implementation
from __future__ import annotations

import torch


class KVCache:
    """Static cache: buffers are preallocated for max_seq_len and written in place."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k, v (B, kv_heads, T, D) at start_pos; return all cached keys/values up to start_pos + T."""
        end = start_pos + k.shape[2]
        if end > self.max_seq_len:
            raise ValueError(f"sequence length {end} exceeds cache size {self.max_seq_len}")
        self.k[layer_idx, :, :, start_pos:end] = k
        self.v[layer_idx, :, :, start_pos:end] = v
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]
