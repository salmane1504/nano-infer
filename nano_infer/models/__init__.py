# Model architectures package
from nano_infer.config import ModelConfig
from nano_infer.models.base import CausalLM
from nano_infer.models.llama import LlamaForCausalLM

MODEL_REGISTRY: dict[str, type[CausalLM]] = {
    "llama": LlamaForCausalLM,
    "qwen2": LlamaForCausalLM,
}


def build_model(config: ModelConfig) -> CausalLM:
    try:
        return MODEL_REGISTRY[config.model_type](config)
    except KeyError:
        raise ValueError(f"unsupported model_type '{config.model_type}'") from None
