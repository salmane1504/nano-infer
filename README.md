# nano-infer

A small PyTorch inference engine for Llama and Qwen2 models: safetensors weight
loading, KV-cached generation, top-k/top-p/temperature sampling, streaming
output, and a command-line interface.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # runtime dependencies
uv sync --extra dev    # plus pytest and transformers (for the tests)
```

Get a model into a local directory (needs `config.json`, `tokenizer.json`,
`tokenizer_config.json` and `*.safetensors`; `generation_config.json` is
optional). For example:

```bash
uv run hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir model_files/qwen2.5_0.5B-instruct
```

Supported `model_type` values are `llama` and `qwen2`. The `model` argument
below can also be a Hugging Face repo ID, which is downloaded to the Hugging Face
cache.

## Command line

Run `uv run python -m nano_infer <command> <model> [options]` (`python main.py`
works too).

**Interactive chat** (keeps history; `/reset` clears it, `/exit` or Ctrl-D quits):

```bash
uv run python -m nano_infer chat model_files/qwen2.5_0.5B-instruct
uv run python -m nano_infer chat model_files/qwen2.5_0.5B-instruct --system "Answer briefly."
```

**One chat message, then exit:**

```bash
uv run python -m nano_infer chat model_files/qwen2.5_0.5B-instruct -m "Say hello."
```

**Raw text completion** (prompt as an argument, or piped through stdin):

```bash
uv run python -m nano_infer complete model_files/qwen2.5_0.5B-instruct "Hello, my name is"
echo "Once upon a time" | uv run python -m nano_infer complete model_files/qwen2.5_0.5B-instruct
```

Options (both commands):

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `auto` | `cpu`, `mps`, `cuda`, or `auto` (CUDA, then Apple GPU, then CPU) |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, or `float32` |
| `--max-new-tokens` | `256` | Maximum tokens to generate |
| `--temperature` | from `generation_config.json` | `0` is greedy |
| `--top-k` | from `generation_config.json` | `0` disables |
| `--top-p` | from `generation_config.json` | `1` disables |
| `--seed` | none | Makes sampling reproducible |
| `--prefill-chunk-size` | `512` | Prompt tokens processed per forward pass |

If the model directory's `generation_config.json` enables sampling, its
temperature, top-k and top-p are the defaults; otherwise decoding is greedy.
Flags always override. In chat, the oldest exchanges are dropped when the
conversation would exceed the model's context window. Repetition penalties from
`generation_config.json` are not applied. Progress messages go to stderr, so
stdout contains only the model output.

## Python API

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
IDs. Each request gets a fresh KV cache, and long prompts are prefilled in
chunks (adjustable with `prefill_chunk_size`). Prompts must be nonempty and the
requested prompt plus continuation must fit the model's context length.

## Tests

Run `uv run --locked --extra dev python -m pytest -q` to install the development
dependencies and run the tests. Tests that need the Qwen checkpoint use the local
`model_files/qwen2.5_0.5B-instruct` directory and skip if it is absent; they do
not download model files.