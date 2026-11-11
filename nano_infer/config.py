# Dataclass for model hyperparameters
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


# Field names match Hugging Face config.json so HF configs load directly.
@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    head_dim: int | None = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False  # Qwen2 uses QKV bias

    def __post_init__(self) -> None:
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))
