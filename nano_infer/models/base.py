# Base Transformer protocol
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

from nano_infer.config import ModelConfig

if TYPE_CHECKING:
    from nano_infer.cache.kv_cache import KVCache


class CausalLM(nn.Module, ABC):
    """Interface the engine depends on; concrete models implement it."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Map (batch, seq) token ids to (batch, seq, vocab) logits.

        start_pos is the absolute position of input_ids[:, 0], used for RoPE and cache writes.
        """

    @abstractmethod
    def make_cache(self, batch_size: int, max_seq_len: int) -> KVCache:
        """Create an empty KV cache sized for this model."""
