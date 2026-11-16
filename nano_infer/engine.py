"""Single-prompt, cache-backed text generation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from nano_infer.cache.kv_cache import KVCache
from nano_infer.models.base import CausalLM
from nano_infer.sampler import Sampler
from nano_infer.tokenizer import Tokenizer
from nano_infer.weights import load_pretrained


class Engine:
    """Generate continuations with an independent KV cache for each request."""

    def __init__(self, model: CausalLM, tokenizer: Tokenizer, *, prefill_chunk_size: int = 512) -> None:
        if (
            not isinstance(prefill_chunk_size, int)
            or isinstance(prefill_chunk_size, bool)
            or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.prefill_chunk_size = prefill_chunk_size

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        *,
        prefill_chunk_size: int = 512,
    ) -> Engine:
        """Load a model and tokenizer from a local directory or Hub repository."""
        model, model_dir = load_pretrained(model_id_or_path, dtype=dtype, device=device)
        return cls(model, Tokenizer.from_dir(model_dir), prefill_chunk_size=prefill_chunk_size)

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        logits = None
        for start in range(0, input_ids.shape[1], self.prefill_chunk_size):
            chunk = input_ids[:, start : start + self.prefill_chunk_size]
            # Do not retain the full chunk's vocabulary logits across prefill steps.
            logits = self.model(chunk, cache=cache, start_pos=start)[:, -1, :].clone()
        if logits is None:
            raise ValueError("input_ids must not be empty")
        return logits

    @torch.inference_mode()
    def _decode(self, token_id: int, cache: KVCache, start_pos: int, device: torch.device) -> torch.Tensor:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
        return self.model(input_ids, cache=cache, start_pos=start_pos)[:, -1, :]

    def generate_tokens(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int = 128,
        *,
        sampler: Sampler | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[int]:
        """Yield generated token IDs, excluding the prompt and any EOS token."""
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        tokens = list(input_ids)
        if not tokens:
            raise ValueError("input_ids must not be empty")
        if any(
            not isinstance(token, int)
            or isinstance(token, bool)
            or token < 0
            or token >= self.model.config.vocab_size
            for token in tokens
        ):
            raise ValueError("input_ids must contain token IDs within the model vocabulary")
        if len(tokens) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("prompt and requested tokens exceed the model context length")
        if max_new_tokens == 0:
            return

        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise ValueError("model must have a parameter to determine its device")
        device = parameter.device
        cache = self.model.make_cache(batch_size=1, max_seq_len=len(tokens) + max_new_tokens - 1)
        prompt = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self._prefill(prompt, cache)
        sampler = sampler if sampler is not None else Sampler(temperature=0)
        eos_token_ids = set(self.tokenizer.eos_token_ids)

        for step in range(max_new_tokens):
            token_id = int(sampler.sample(logits, generator=generator).item())
            if token_id in eos_token_ids:
                return
            yield token_id
            if step + 1 < max_new_tokens:
                logits = self._decode(token_id, cache, len(tokens) + step, device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        *,
        sampler: Sampler | None = None,
        generator: torch.Generator | None = None,
        add_special_tokens: bool = True,
    ) -> str:
        """Return only the generated text, excluding special tokens."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        output = self.generate_tokens(input_ids, max_new_tokens, sampler=sampler, generator=generator)
        return self.tokenizer.decode(list(output))

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        *,
        sampler: Sampler | None = None,
        generator: torch.Generator | None = None,
        add_special_tokens: bool = True,
    ) -> Iterator[str]:
        """Yield generated text as soon as complete characters are available."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        token_stream = self.tokenizer.stream()
        for token_id in self.generate_tokens(input_ids, max_new_tokens, sampler=sampler, generator=generator):
            text = token_stream.push(token_id)
            if text:
                yield text
        remaining = token_stream.finish()
        if remaining:
            yield remaining
