# Llama/Qwen implementation (RMSNorm, RoPE, SwiGLU)
from __future__ import annotations

import torch
from torch import nn

from nano_infer.cache.kv_cache import KVCache
from nano_infer.config import ModelConfig
from nano_infer.models.base import CausalLM
from nano_infer.models.layers import MLP, Attention, RMSNorm, RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None,
        start_pos: int,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, start_pos)
        return x + self.mlp(self.post_attention_layernorm(x))


# Attribute names mirror Hugging Face so checkpoint keys load without remapping.
class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None, start_pos: int) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        positions = torch.arange(start_pos, start_pos + input_ids.shape[1], device=input_ids.device)
        cos, sin = self.rotary_emb(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, cache, start_pos)
        return self.norm(x)


class LlamaForCausalLM(CausalLM):
    """Serves Llama and Qwen2; differences are driven by ModelConfig."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, cache, start_pos))

    def make_cache(self, batch_size: int, max_seq_len: int) -> KVCache:
        weight = self.model.embed_tokens.weight
        return KVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            num_kv_heads=self.config.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.config.head_dim,
            dtype=weight.dtype,
            device=weight.device,
        )
