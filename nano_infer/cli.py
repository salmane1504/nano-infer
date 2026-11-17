"""Command-line interface: one-shot completion and interactive chat."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from nano_infer.engine import Engine
from nano_infer.sampler import Sampler
from nano_infer.tokenizer import Tokenizer
from nano_infer.weights import load_pretrained

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
EXIT_COMMANDS = {"/exit", "/quit"}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("model", help="local model directory or Hugging Face repo ID")
    common.add_argument("--device", default="auto", help="cpu, mps, cuda, or auto (default)")
    common.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    common.add_argument("--max-new-tokens", type=int, default=256)
    common.add_argument("--temperature", type=float, help="0 means greedy (default: from generation_config.json)")
    common.add_argument("--top-k", type=int, help="0 disables top-k filtering")
    common.add_argument("--top-p", type=float, help="1 disables nucleus filtering")
    common.add_argument("--seed", type=int, help="make sampling reproducible")
    common.add_argument("--prefill-chunk-size", type=int, default=512)

    parser = argparse.ArgumentParser(prog="nano-infer", description="Run Llama/Qwen2 models locally.")
    commands = parser.add_subparsers(dest="command", required=True)

    complete = commands.add_parser("complete", parents=[common], help="continue a raw text prompt")
    complete.add_argument("prompt", nargs="?", help="prompt text (read from stdin if omitted)")

    chat = commands.add_parser("chat", parents=[common], help="chat with an instruct model")
    chat.add_argument("-m", "--message", help="send one message and exit instead of starting a session")
    chat.add_argument("--system", help="system prompt (default: the model's chat template default)")
    return parser


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_sampler(args: argparse.Namespace, model_dir: Path) -> Sampler:
    """Explicit flags win; otherwise use generation_config.json, or greedy if it disables sampling."""
    path = model_dir / "generation_config.json"
    defaults = json.loads(path.read_text()) if path.exists() else {}
    sampling = defaults.get("do_sample", False)
    temperature = args.temperature
    if temperature is None:
        temperature = defaults.get("temperature", 1.0) if sampling else 0.0
    top_k = args.top_k if args.top_k is not None else (defaults.get("top_k", 0) if sampling else 0)
    top_p = args.top_p if args.top_p is not None else (defaults.get("top_p", 1.0) if sampling else 1.0)
    return Sampler(temperature=temperature, top_k=top_k, top_p=top_p)


def load_engine(args: argparse.Namespace) -> tuple[Engine, Path, str]:
    device = resolve_device(args.device)
    print(f"Loading {args.model} on {device}...", file=sys.stderr)
    model, model_dir = load_pretrained(args.model, dtype=DTYPES[args.dtype], device=device)
    engine = Engine(model, Tokenizer.from_dir(model_dir), prefill_chunk_size=args.prefill_chunk_size)
    return engine, model_dir, device


def make_generator(args: argparse.Namespace, device: str) -> torch.Generator | None:
    if args.seed is None:
        return None
    return torch.Generator(device=device).manual_seed(args.seed)


def run_complete(args: argparse.Namespace, engine: Engine, sampler: Sampler, generator: torch.Generator | None) -> None:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if not prompt:
        raise ValueError("prompt is empty")
    print(prompt, end="", flush=True)
    for chunk in engine.stream(prompt, args.max_new_tokens, sampler=sampler, generator=generator):
        print(chunk, end="", flush=True)
    print()


def fit_history(engine: Engine, messages: list[dict[str, str]], max_new_tokens: int) -> list[int]:
    """Tokenize the conversation, dropping the oldest exchanges until it fits the context window."""
    limit = engine.model.config.max_position_embeddings - max_new_tokens
    first_turn = 1 if messages[0]["role"] == "system" else 0
    while True:
        prompt = engine.tokenizer.apply_chat_template(messages)
        ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= limit:
            return ids
        if len(messages) - first_turn <= 1:
            raise ValueError("message is too long for the model's context window")
        del messages[first_turn : first_turn + 2]


def reply(
    engine: Engine,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    sampler: Sampler,
    generator: torch.Generator | None,
) -> str:
    ids = fit_history(engine, messages, args.max_new_tokens)
    stream = engine.tokenizer.stream()
    parts: list[str] = []
    for token_id in engine.generate_tokens(ids, args.max_new_tokens, sampler=sampler, generator=generator):
        text = stream.push(token_id)
        if text:
            parts.append(text)
            print(text, end="", flush=True)
    tail = stream.finish()
    if tail:
        parts.append(tail)
        print(tail, end="", flush=True)
    print()
    return "".join(parts)


def run_chat(args: argparse.Namespace, engine: Engine, sampler: Sampler, generator: torch.Generator | None) -> None:
    system = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(system)

    if args.message is not None:
        messages.append({"role": "user", "content": args.message})
        reply(engine, messages, args, sampler, generator)
        return

    print("Chat started. Type /reset to clear history, /exit (or Ctrl-D) to quit.", file=sys.stderr)
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in EXIT_COMMANDS:
            return
        if user == "/reset":
            messages = list(system)
            print("History cleared.", file=sys.stderr)
            continue
        messages.append({"role": "user", "content": user})
        print("bot> ", end="", flush=True)
        try:
            answer = reply(engine, messages, args, sampler, generator)
        except KeyboardInterrupt:
            print()
            messages.pop()
            continue
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer})


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        engine, model_dir, device = load_engine(args)
        sampler = make_sampler(args, model_dir)
        generator = make_generator(args, device)
        if args.command == "complete":
            run_complete(args, engine, sampler, generator)
        else:
            run_chat(args, engine, sampler, generator)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
