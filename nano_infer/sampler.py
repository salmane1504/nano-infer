"""Temperature, top-k, and nucleus sampling for next-token logits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sampler:
    """Sample from the last dimension of a tensor of logits.

    A temperature of zero selects the most likely token deterministically.
    Set top_k to zero or top_p to one to disable the respective filter.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if isinstance(self.top_p, bool) or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return token IDs with the same leading dimensions as ``logits``."""
        if logits.ndim == 0 or logits.numel() == 0:
            raise ValueError("logits must have a non-empty vocabulary and batch")
        if not logits.is_floating_point():
            raise TypeError("logits must be floating point")

        scores = logits if logits.dtype == torch.float64 else logits.float()
        invalid = (
            torch.isnan(scores).any()
            | torch.isposinf(scores).any()
            | (~torch.isfinite(scores).any(dim=-1)).any()
        )
        if invalid.item():
            raise ValueError("each logits row must contain a finite value and no NaN or positive infinity")
        if self.temperature == 0:
            return scores.argmax(dim=-1)

        vocab_size = scores.shape[-1]
        scores = scores.reshape(-1, vocab_size)
        indices = None
        if 0 < self.top_k < vocab_size:
            scores, indices = torch.topk(scores, self.top_k, dim=-1)

        temperature = self.temperature
        limits = torch.finfo(scores.dtype)
        if temperature >= 1:
            # Scale first to avoid overflowing the difference between large logits.
            if temperature > limits.max:
                scores = scores / limits.max
                temperature /= limits.max
            if temperature > limits.max:
                scores = torch.where(torch.isfinite(scores), 0.0, -torch.inf)
            else:
                scores = scores / temperature
            scores = scores - scores.max(dim=-1, keepdim=True).values
        else:
            scores = scores - scores.max(dim=-1, keepdim=True).values
            if temperature < limits.tiny:
                # Scale both operands so very small temperatures remain representable.
                scores = scores / limits.tiny
                temperature /= limits.tiny
                if temperature < limits.tiny * limits.eps:
                    scores = torch.where(scores == 0, 0.0, -torch.inf)
                else:
                    scores = scores / temperature
            else:
                scores = scores / temperature
        probabilities = torch.softmax(scores, dim=-1)

        if self.top_p < 1:
            probabilities, order = probabilities.sort(dim=-1, descending=True)
            cumulative = probabilities.cumsum(dim=-1)
            remove = torch.zeros_like(probabilities, dtype=torch.bool)
            remove[:, 1:] = cumulative[:, :-1] >= self.top_p
            probabilities = probabilities.masked_fill(remove, 0)
            indices = order if indices is None else indices.gather(-1, order)

        sampled = torch.multinomial(probabilities, 1, generator=generator)
        if indices is not None:
            sampled = indices.gather(-1, sampled)
        return sampled.reshape(logits.shape[:-1])
