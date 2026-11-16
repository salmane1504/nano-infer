import math

import pytest
import torch

from nano_infer.sampler import Sampler


def test_greedy_sampling_preserves_batch_shape_and_input() -> None:
    logits = torch.tensor(
        [[[1.0, 3.0, -torch.inf], [4.0, 2.0, 1.0]], [[0.0, -1.0, 2.0], [1.0, 5.0, 3.0]]],
        requires_grad=True,
    )
    original = logits.detach().clone()
    result = Sampler(temperature=0, top_k=1, top_p=0.5).sample(logits)

    assert torch.equal(result, torch.tensor([[1, 0], [2, 1]]))
    assert torch.equal(logits.detach(), original)
    assert not result.requires_grad
    assert Sampler(temperature=0).sample(torch.tensor([1.0, 3.0])).shape == torch.Size([])


def test_top_k_samples_only_from_the_largest_logits() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0]).expand(8192, -1)
    generator = torch.Generator().manual_seed(42)
    result = Sampler(top_k=2).sample(logits, generator=generator)

    assert set(result.tolist()) == {2, 3}
    expected = math.exp(3) / (math.exp(2) + math.exp(3))
    assert abs((result == 3).float().mean().item() - expected) < 0.03


def test_top_p_includes_the_token_crossing_the_threshold() -> None:
    logits = torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125])).expand(4096, -1)

    for top_p, expected in [(0.5, {0}), (0.7, {0, 1}), (0.01, {0})]:
        result = Sampler(top_p=top_p).sample(logits, generator=torch.Generator().manual_seed(5))
        assert set(result.tolist()) == expected


def test_top_p_is_applied_after_top_k() -> None:
    logits = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1])).expand(512, -1)
    result = Sampler(top_k=2, top_p=0.55).sample(logits)

    assert set(result.tolist()) == {0}


def test_temperature_changes_the_distribution() -> None:
    logits = torch.tensor([0.0, 2.0], dtype=torch.bfloat16).expand(8192, -1)
    result = Sampler(temperature=2).sample(logits, generator=torch.Generator().manual_seed(7))
    expected = math.exp(1) / (1 + math.exp(1))

    assert abs((result == 1).float().mean().item() - expected) < 0.03
    low_temperature = Sampler(temperature=0.1).sample(logits, generator=torch.Generator().manual_seed(7))
    assert (low_temperature == 1).float().mean().item() > 0.99


def test_tiny_temperature_remains_numerically_stable() -> None:
    logits = torch.tensor([0.0, 1e-40]).expand(8192, -1)
    result = Sampler(temperature=1e-40).sample(logits, generator=torch.Generator().manual_seed(11))
    expected = math.exp(1) / (1 + math.exp(1))
    assert abs((result == 1).float().mean().item() - expected) < 0.03

    tied = torch.tensor([0.0, 0.0, -1.0]).expand(256, -1)
    result = Sampler(temperature=5e-324).sample(tied, generator=torch.Generator().manual_seed(1))
    assert set(result.tolist()) == {0, 1}


@pytest.mark.parametrize("temperature", [2e38, 1e39])
def test_large_temperature_remains_numerically_stable(temperature: float) -> None:
    logits = torch.tensor([3e38, -3e38])
    expected = 1 / (1 + math.exp(-(logits[0].item() - logits[1].item()) / temperature))
    result = Sampler(temperature=temperature).sample(
        logits.expand(16384, -1), generator=torch.Generator().manual_seed(19)
    )
    assert abs((result == 0).float().mean().item() - expected) < 0.015

    masked = torch.tensor([1.0, -torch.inf, 0.0]).expand(256, -1)
    result = Sampler(temperature=1e300).sample(masked, generator=torch.Generator().manual_seed(2))
    assert set(result.tolist()) == {0, 2}


def test_large_float64_logits_remain_numerically_stable() -> None:
    logits = torch.tensor([1e308, -1e308], dtype=torch.float64).expand(16384, -1)
    result = Sampler(temperature=1e308).sample(logits, generator=torch.Generator().manual_seed(3))
    expected = 1 / (1 + math.exp(-2))

    assert abs((result == 0).float().mean().item() - expected) < 0.015


def test_sampling_is_reproducible_and_allows_masked_logits() -> None:
    logits = torch.tensor([[1.0, -torch.inf, 2.0], [0.0, 1.0, -torch.inf]]).expand(16, 2, 3)
    sampler = Sampler(temperature=0.7, top_k=100, top_p=0.9)
    first = sampler.sample(logits, generator=torch.Generator().manual_seed(123))
    second = sampler.sample(logits, generator=torch.Generator().manual_seed(123))

    assert first.shape == (16, 2)
    assert torch.equal(first, second)
    assert (first[:, 0] != 1).all()
    assert (first[:, 1] != 2).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_sampling_with_a_device_generator() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0], device="mps").expand(128, -1)
    sampler = Sampler(temperature=0.7, top_k=3, top_p=0.9)
    first = sampler.sample(logits, generator=torch.Generator(device="mps").manual_seed(5))
    second = sampler.sample(logits, generator=torch.Generator(device="mps").manual_seed(5))

    assert first.device.type == "mps"
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -1},
        {"temperature": True},
        {"temperature": math.nan},
        {"temperature": math.inf},
        {"top_k": -1},
        {"top_k": 1.5},
        {"top_k": True},
        {"top_p": 0},
        {"top_p": True},
        {"top_p": -0.5},
        {"top_p": 1.1},
        {"top_p": math.nan},
        {"top_p": math.inf},
    ],
)
def test_invalid_sampling_settings_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Sampler(**kwargs)


@pytest.mark.parametrize("temperature", [0, 1])
@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([-float("inf"), -float("inf")]),
        torch.tensor([[1.0, 2.0], [-float("inf"), -float("inf")]]),
        torch.empty(0),
        torch.empty(0, 2),
        torch.tensor(1.0),
    ],
)
def test_invalid_logits_are_rejected(logits: torch.Tensor, temperature: float) -> None:
    with pytest.raises(ValueError):
        Sampler(temperature=temperature).sample(logits)


def test_non_floating_logits_are_rejected() -> None:
    with pytest.raises(TypeError, match="floating point"):
        Sampler().sample(torch.tensor([1, 2]))
