from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nano_infer.engine import Engine
from nano_infer.sampler import Sampler


MODEL_DIR = Path(__file__).resolve().parents[1] / "model_files" / "qwen2.5_0.5B-instruct"
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
pytestmark = pytest.mark.skipif(
    not all((MODEL_DIR / name).is_file() for name in REQUIRED_FILES),
    reason="the local Qwen2.5 checkpoint is not available",
)


@pytest.fixture(scope="module")
def qwen_engine() -> Engine:
    with patch("nano_infer.weights.download_model", side_effect=AssertionError("unexpected model download")):
        return Engine.from_pretrained(MODEL_DIR, prefill_chunk_size=2)


def test_local_qwen_greedy_generation_and_streaming(qwen_engine: Engine) -> None:
    prompt = qwen_engine.tokenizer.apply_chat_template([{"role": "user", "content": "Say hello."}])
    prompt_ids = qwen_engine.tokenizer.encode(prompt, add_special_tokens=False)
    expected_ids = [9707, 0, 2585, 646, 358, 7789, 498, 3351, 30]
    expected_text = "Hello! How can I assist you today?"

    assert list(qwen_engine.generate_tokens(prompt_ids, max_new_tokens=24)) == expected_ids
    assert qwen_engine.generate(prompt, max_new_tokens=24, add_special_tokens=False) == expected_text
    assert "".join(qwen_engine.stream(prompt, max_new_tokens=24, add_special_tokens=False)) == expected_text


def test_local_qwen_sampling_is_reproducible(qwen_engine: Engine) -> None:
    prompt_ids = qwen_engine.tokenizer.encode("Hello, my name is")
    sampler = Sampler(temperature=0.7, top_k=20, top_p=0.8)
    first = list(
        qwen_engine.generate_tokens(
            prompt_ids, max_new_tokens=8, sampler=sampler, generator=torch.Generator().manual_seed(12)
        )
    )
    second = list(
        qwen_engine.generate_tokens(
            prompt_ids, max_new_tokens=8, sampler=sampler, generator=torch.Generator().manual_seed(12)
        )
    )
    assert first
    assert first == second


def test_local_qwen_utf8_stream_flush(qwen_engine: Engine) -> None:
    tokenizer = qwen_engine.tokenizer
    ids = tokenizer.encode("\u200d", add_special_tokens=False)
    assert len(ids) == 2

    complete = tokenizer.stream()
    assert complete.push(ids[0]) == ""
    assert complete.push(ids[1]) == "\u200d"
    assert complete.finish() == ""

    incomplete = tokenizer.stream()
    assert incomplete.push(ids[0]) == ""
    assert incomplete.finish() == tokenizer.decode(ids[:1]) == "\ufffd"
    assert incomplete.finish() == ""


def test_local_qwen_chat_matches_transformers(qwen_engine: Engine) -> None:
    transformers = pytest.importorskip("transformers")
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16
    ).eval()
    prompt = qwen_engine.tokenizer.apply_chat_template([{"role": "user", "content": "Say hello."}])
    prompt_ids = qwen_engine.tokenizer.encode(prompt, add_special_tokens=False)
    inputs = torch.tensor([prompt_ids])
    cache = qwen_engine.model.make_cache(1, len(prompt_ids))
    with torch.inference_mode():
        actual_logits = qwen_engine._prefill(inputs, cache).float()
        reference_logits = reference(inputs).logits[:, -1, :].float()
        reference_ids = reference.generate(
            inputs,
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=24,
            eos_token_id=qwen_engine.tokenizer.eos_token_ids,
            pad_token_id=qwen_engine.tokenizer.eos_token_ids[0],
        )[0, len(prompt_ids) :].tolist()
    differences = (actual_logits - reference_logits).abs()
    assert differences.max().item() < 0.5
    assert differences.mean().item() < 0.1
    assert reference_ids[-1] in qwen_engine.tokenizer.eos_token_ids
    reference_ids.pop()

    actual_ids = list(qwen_engine.generate_tokens(prompt_ids, max_new_tokens=24))
    assert actual_ids == reference_ids
    assert qwen_engine.generate(prompt, max_new_tokens=24, add_special_tokens=False) == (
        qwen_engine.tokenizer.decode(reference_ids)
    )
