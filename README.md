# nano-infer

A small PyTorch inference engine for Llama and Qwen2 models.

## Generate text

```python
from nano_infer.engine import Engine
from nano_infer.sampler import Sampler

engine = Engine.from_pretrained("model_files/qwen2.5_0.5B-instruct")
prompt = engine.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain what a KV cache does."}]
)

# Chat templates already contain their special tokens.
print(engine.generate(prompt, max_new_tokens=64, add_special_tokens=False))

sampler = Sampler(temperature=0.7, top_k=20, top_p=0.8)
for chunk in engine.stream(prompt, max_new_tokens=64, sampler=sampler, add_special_tokens=False):
    print(chunk, end="", flush=True)
```

`Engine.from_pretrained` accepts a local model directory or a Hugging Face model
ID. It defaults to CPU and `torch.bfloat16`; pass `device` and `dtype` to change
them. Generation returns only the continuation, stops at any configured EOS
token, and defaults to greedy decoding. Supply a `Sampler` for temperature,
top-k, and top-p sampling; pass a device-matched `torch.Generator` to make
sampling reproducible. `top_k=0` and `top_p=1.0` disable their respective filters, and
`temperature=0` selects the most likely token. Other settings from a model's
`generation_config.json`, such as repetition penalties, are not applied.

For already-tokenized prompts, `Engine.generate_tokens` yields generated token
IDs. Each request gets a fresh KV cache, and long prompts are prefetched in
chunks (adjustable with `prefill_chunk_size`). Prompts must be nonempty and the
requested prompt plus continuation must fit the model's context length.

## Tests

Run `uv run --locked --extra dev python -m pytest -q` to install the development
dependencies and run the tests. Tests that need the Qwen checkpoint use the local
`model_files/qwen2.5_0.5B-instruct` directory and skip if it is absent; they do
not download model files.