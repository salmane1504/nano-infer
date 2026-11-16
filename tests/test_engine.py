from __future__ import annotations

import weakref

import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers
from torch import nn

from nano_infer.cache.kv_cache import KVCache
from nano_infer.config import ModelConfig
from nano_infer.engine import Engine
from nano_infer.models import build_model
from nano_infer.models.base import CausalLM
from nano_infer.sampler import Sampler
from nano_infer.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    vocab = {"<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "<eos>": 5, "<eos2>": 6, "\ufffd": 7}
    backend = HFTokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(["<eos>", "<eos2>"])
    return Tokenizer(backend, None, None, "<eos>", [5, 6])


class RecordingModel(CausalLM):
    def __init__(self, predictions: dict[int, int], max_positions: int = 16) -> None:
        super().__init__(
            ModelConfig(
                model_type="llama",
                vocab_size=8,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                max_position_embeddings=max_positions,
            )
        )
        self.anchor = nn.Parameter(torch.zeros(1))
        self.predictions = predictions
        self.calls: list[tuple[list[int], int, KVCache | None, bool, bool]] = []
        self.caches: list[KVCache] = []

    def make_cache(self, batch_size: int, max_seq_len: int) -> KVCache:
        cache = KVCache(1, batch_size, 1, max_seq_len, 4, self.anchor.dtype, self.anchor.device)
        self.caches.append(cache)
        return cache

    def forward(
        self, input_ids: torch.Tensor, cache: KVCache | None = None, start_pos: int = 0
    ) -> torch.Tensor:
        self.calls.append((input_ids[0].tolist(), start_pos, cache, torch.is_grad_enabled(), self.training))
        logits = torch.full((*input_ids.shape, self.config.vocab_size), -100.0, device=input_ids.device)
        for index in range(input_ids.shape[1]):
            token = self.predictions.get(start_pos + index + 1, 5)
            logits[:, index, token] = 0
        return logits


class UniformModel(RecordingModel):
    def forward(
        self, input_ids: torch.Tensor, cache: KVCache | None = None, start_pos: int = 0
    ) -> torch.Tensor:
        return super().forward(input_ids, cache, start_pos).zero_()


def test_prefill_releases_logits_from_previous_chunks(tokenizer: Tokenizer) -> None:
    class MemoryCheckingModel(RecordingModel):
        def __init__(self) -> None:
            super().__init__({3: 4})
            self.previous_logits: weakref.ReferenceType[torch.Tensor] | None = None
            self.previous_released: list[bool] = []

        def forward(
            self, input_ids: torch.Tensor, cache: KVCache | None = None, start_pos: int = 0
        ) -> torch.Tensor:
            if self.previous_logits is not None:
                self.previous_released.append(self.previous_logits() is None)
            logits = super().forward(input_ids, cache, start_pos)
            self.previous_logits = weakref.ref(logits)
            return logits

    model = MemoryCheckingModel()
    engine = Engine(model, tokenizer, prefill_chunk_size=1)
    assert list(engine.generate_tokens([1, 2, 3], max_new_tokens=1)) == [4]
    assert model.previous_released == [True, True]


def test_chunked_prefill_decode_positions_and_multiple_eos_tokens(tokenizer: Tokenizer) -> None:
    model = RecordingModel({3: 4, 4: 7, 5: 6})
    engine = Engine(model, tokenizer, prefill_chunk_size=2)

    assert list(engine.generate_tokens([1, 2, 3], max_new_tokens=5)) == [4, 7]
    assert [(ids, start) for ids, start, _, _, _ in model.calls] == [
        ([1, 2], 0),
        ([3], 2),
        ([4], 3),
        ([7], 4),
    ]
    assert len(model.caches) == 1
    assert model.caches[0].max_seq_len == 7
    assert all(cache is model.caches[0] for _, _, cache, _, _ in model.calls)
    assert all(not grad and not training for _, _, _, grad, training in model.calls)


def test_yielding_does_not_change_the_callers_gradient_mode(tokenizer: Tokenizer) -> None:
    model = RecordingModel({2: 3, 3: 4})
    engine = Engine(model, tokenizer)
    with torch.enable_grad():
        output = engine.generate_tokens([1, 2], max_new_tokens=2)
        assert next(output) == 3
        assert torch.is_grad_enabled()
        assert len(model.calls) == 1
        assert next(output) == 4
        assert torch.is_grad_enabled()
        with pytest.raises(StopIteration):
            next(output)
    assert len(model.calls) == 2


def test_zero_tokens_and_immediate_eos_do_not_decode(tokenizer: Tokenizer) -> None:
    model = RecordingModel({2: 5})
    engine = Engine(model, tokenizer)
    assert list(engine.generate_tokens([1, 2], max_new_tokens=0)) == []
    assert engine.generate("a b", max_new_tokens=0) == ""
    assert list(engine.stream("a b", max_new_tokens=0)) == []
    assert model.calls == []
    assert model.caches == []

    assert list(engine.generate_tokens([1, 2], max_new_tokens=5)) == []
    assert len(model.calls) == 1


def test_generate_and_stream_return_continuation_with_fresh_caches(tokenizer: Tokenizer) -> None:
    model = RecordingModel({2: 3, 3: 4, 4: 6})
    engine = Engine(model, tokenizer)

    assert engine.generate("a b", max_new_tokens=4) == "c d"
    assert "".join(engine.stream("a b", max_new_tokens=4)) == "c d"
    assert len(model.caches) == 2
    assert model.caches[0] is not model.caches[1]


def test_stream_flushes_a_final_incomplete_character(tokenizer: Tokenizer) -> None:
    model = RecordingModel({1: 7, 2: 5})
    engine = Engine(model, tokenizer)

    assert engine.generate("a", max_new_tokens=2) == "\ufffd"
    assert list(engine.stream("a", max_new_tokens=2)) == ["\ufffd"]
    assert engine.generate("a", max_new_tokens=1) == "\ufffd"
    assert list(engine.stream("a", max_new_tokens=1)) == ["\ufffd"]


def test_supplied_sampler_and_generator_control_token_selection(tokenizer: Tokenizer) -> None:
    tokenizer.eos_token_ids.clear()
    model = UniformModel({})
    engine = Engine(model, tokenizer)
    sampler = Sampler(temperature=0.7, top_k=4, top_p=0.8)
    reference_generator = torch.Generator().manual_seed(23)
    expected = [
        int(sampler.sample(torch.zeros(1, 8), generator=reference_generator).item())
        for _ in range(5)
    ]

    actual = list(
        engine.generate_tokens(
            [1, 2], max_new_tokens=5, sampler=sampler, generator=torch.Generator().manual_seed(23)
        )
    )
    assert actual == expected


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_prefill_chunk_size_is_rejected(tokenizer: Tokenizer, chunk_size: int) -> None:
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        Engine(RecordingModel({}), tokenizer, prefill_chunk_size=chunk_size)


@pytest.mark.parametrize(
    ("input_ids", "max_new_tokens", "message"),
    [
        ([], 1, "must not be empty"),
        ([-1], 1, "vocabulary"),
        ([8], 1, "vocabulary"),
        ([True], 1, "vocabulary"),
        ([1.5], 1, "vocabulary"),
        ([1], -1, "max_new_tokens"),
        ([1], True, "max_new_tokens"),
        ([1], 1.5, "max_new_tokens"),
        ([1, 2, 3], 2, "context length"),
        ([1, 2, 3, 4, 1], 0, "context length"),
    ],
)
def test_invalid_generation_requests_are_rejected(
    tokenizer: Tokenizer, input_ids: list[int], max_new_tokens: int, message: str
) -> None:
    model = RecordingModel({}, max_positions=4)
    engine = Engine(model, tokenizer)
    with pytest.raises(ValueError, match=message):
        list(engine.generate_tokens(input_ids, max_new_tokens))
    assert model.calls == []
    assert model.caches == []


def test_context_limit_can_be_filled_exactly(tokenizer: Tokenizer) -> None:
    model = RecordingModel({2: 3, 3: 4}, max_positions=4)
    engine = Engine(model, tokenizer)
    assert list(engine.generate_tokens([1, 2], max_new_tokens=2)) == [3, 4]


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"))],
)
def test_cached_generation_matches_uncached_tiny_transformer(tokenizer: Tokenizer, device: str) -> None:
    tokenizer.eos_token_ids.clear()
    config = ModelConfig(
        model_type="llama",
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    torch.manual_seed(4)
    model = build_model(config).to(device).eval()
    engine = Engine(model, tokenizer, prefill_chunk_size=2)
    prompt = [1, 2, 3]
    reference = prompt.copy()
    with torch.inference_mode():
        for _ in range(5):
            logits = model(torch.tensor([reference], device=device))
            reference.append(int(logits[0, -1].argmax()))

    expected = reference[len(prompt) :]
    assert list(engine.generate_tokens(prompt, max_new_tokens=5)) == expected
    assert list(engine.generate_tokens(prompt, max_new_tokens=5)) == expected
