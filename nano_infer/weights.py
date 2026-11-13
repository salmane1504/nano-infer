# Safetensors weight loader & tensor remapping
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from nano_infer.config import ModelConfig
from nano_infer.models import build_model
from nano_infer.models.base import CausalLM

_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def download_model(repo_id: str, revision: str | None = None) -> Path:
    """Fetch config, weights and tokenizer files from the Hub (cached locally)."""
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=_ALLOW_PATTERNS))


def iter_tensors(model_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def remap_key(name: str) -> str:
    """Translate a checkpoint tensor name to this project's parameter name."""
    return name  # Module names already mirror Hugging Face.


@torch.no_grad()
def load_weights(model: CausalLM, model_dir: Path) -> None:
    params = model.state_dict()  # shares storage with the live parameters
    tied = model.config.tie_word_embeddings
    seen: set[str] = set()
    unexpected: list[str] = []

    for raw_name, tensor in iter_tensors(model_dir):
        name = remap_key(raw_name)
        if name.endswith("rotary_emb.inv_freq") or (tied and name == "lm_head.weight"):
            continue
        target = params.get(name)
        if target is None:
            unexpected.append(name)
            continue
        if tensor.shape != target.shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != model shape {tuple(target.shape)}")
        target.copy_(tensor)  # casts to the model's dtype and device
        seen.add(name)

    missing = set(params) - seen - ({"lm_head.weight"} if tied else set())
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch; missing={sorted(missing)[:5]}, unexpected={unexpected[:5]}")


def load_pretrained(
    model_id_or_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> tuple[CausalLM, Path]:
    """Build the model from config.json and fill it with the checkpoint weights."""
    path = Path(model_id_or_path)
    model_dir = path if path.is_dir() else download_model(str(model_id_or_path))
    config = ModelConfig.from_json(model_dir / "config.json")

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = build_model(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    load_weights(model, model_dir)
    return model.eval(), model_dir
