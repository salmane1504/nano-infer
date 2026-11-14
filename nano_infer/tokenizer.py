# Hugging Face tokenizers wrapper
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as HFTokenizer


def _token_text(value: str | dict[str, Any] | None) -> str | None:
    return value["content"] if isinstance(value, dict) else value


def _raise_exception(message: str) -> None:
    raise TemplateError(message)


class Tokenizer:
    def __init__(
        self,
        tokenizer: HFTokenizer,
        chat_template: str | None,
        bos_token: str | None,
        eos_token: str | None,
        eos_token_ids: list[int],
    ) -> None:
        self._tok = tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.eos_token_ids = eos_token_ids
        self._chat_template = None
        if chat_template is not None:
            env = ImmutableSandboxedEnvironment(
                trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
            )
            env.globals["raise_exception"] = _raise_exception
            env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
            self._chat_template = env.from_string(chat_template)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> Tokenizer:
        model_dir = Path(model_dir)

        def read_json(name: str) -> dict[str, Any]:
            path = model_dir / name
            return json.loads(path.read_text()) if path.exists() else {}

        tok_cfg = read_json("tokenizer_config.json")
        tokenizer = HFTokenizer.from_file(str(model_dir / "tokenizer.json"))

        template = tok_cfg.get("chat_template")
        template_file = model_dir / "chat_template.jinja"
        if template_file.exists():
            template = template_file.read_text()
        elif isinstance(template, list):  # [{"name": ..., "template": ...}]
            named = {t["name"]: t["template"] for t in template}
            template = named.get("default", next(iter(named.values())))

        eos_token = _token_text(tok_cfg.get("eos_token"))
        eos_ids = read_json("generation_config.json").get("eos_token_id")
        if eos_ids is None:
            eos_ids = read_json("config.json").get("eos_token_id")
        if eos_ids is None and eos_token is not None:
            eos_ids = tokenizer.token_to_id(eos_token)
        eos_ids = [] if eos_ids is None else [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)

        return cls(tokenizer, template, _token_text(tok_cfg.get("bos_token")), eos_token, eos_ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Render messages to a prompt string; encode it with add_special_tokens=False."""
        if self._chat_template is None:
            raise ValueError("this tokenizer has no chat template")
        return self._chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token or "",
            eos_token=self.eos_token or "",
        )

    def stream(self) -> TokenStream:
        return TokenStream(self)


class TokenStream:
    """Incremental detokenizer that holds back incomplete UTF-8 sequences."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0

    def push(self, token_id: int) -> str:
        """Add a token; return the newly completed text, or '' if it is still partial."""
        self._ids.append(token_id)
        prefix_text = self._tokenizer.decode(self._ids[self._prefix : self._read])
        text = self._tokenizer.decode(self._ids[self._prefix :])
        if len(text) > len(prefix_text) and not text.endswith("\ufffd"):
            self._prefix, self._read = self._read, len(self._ids)
            return text[len(prefix_text) :]
        return ""
