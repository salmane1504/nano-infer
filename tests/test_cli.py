import json
from pathlib import Path

import pytest

from nano_infer import cli
from nano_infer.sampler import Sampler

MODEL_DIR = Path(__file__).resolve().parents[1] / "model_files" / "qwen2.5_0.5B-instruct"
needs_model = pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").is_file(), reason="local Qwen2.5 missing")


def parse(*argv: str):
    return cli.build_parser().parse_args(["complete", "model", *argv])


def test_sampler_defaults_come_from_generation_config(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 0.7, "top_k": 20, "top_p": 0.8})
    )
    assert cli.make_sampler(parse(), tmp_path) == Sampler(0.7, 20, 0.8)
    assert cli.make_sampler(parse("--temperature", "0", "--top-k", "5"), tmp_path) == Sampler(0.0, 5, 0.8)


def test_sampler_is_greedy_without_sampling_defaults(tmp_path: Path) -> None:
    assert cli.make_sampler(parse(), tmp_path) == Sampler(0.0, 0, 1.0)
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False, "temperature": 0.7}))
    assert cli.make_sampler(parse(), tmp_path) == Sampler(0.0, 0, 1.0)


def test_missing_arguments_exit_with_usage_error() -> None:
    with pytest.raises(SystemExit):
        cli.main(["chat"])


@needs_model
def test_invalid_sampling_value_returns_error_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["complete", str(MODEL_DIR), "hi", "--device", "cpu", "--temperature", "-1"]) == 1
    assert "temperature" in capsys.readouterr().err


@needs_model
def test_chat_one_shot_and_complete(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["chat", str(MODEL_DIR), "-m", "Say hello.", "--device", "cpu", "--max-new-tokens", "24"]) == 0
    assert capsys.readouterr().out == "Hello! How can I assist you today?\n"

    argv = ["complete", str(MODEL_DIR), "Hello, my name is", "--device", "cpu", "--max-new-tokens", "8"]
    assert cli.main([*argv, "--temperature", "0"]) == 0
    assert capsys.readouterr().out == "Hello, my name is Alex and I am a 17\n"


@needs_model
def test_chat_session_history_reset_and_trimming(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    lines = iter(["", "Say hello.", "/reset", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(lines))
    argv = ["chat", str(MODEL_DIR), "--device", "cpu", "--temperature", "0", "--max-new-tokens", "24"]
    assert cli.main(argv) == 0
    assert capsys.readouterr().out.count("Hello! How can I assist you today?") == 1

    args = cli.build_parser().parse_args(argv)
    engine, _, _ = cli.load_engine(args)
    messages = [{"role": "system", "content": "s"}] + [
        {"role": role, "content": "word " * 50} for role in ("user", "assistant") * 3
    ] + [{"role": "user", "content": "hi"}]
    engine.model.config = type(engine.model.config)(
        **{**engine.model.config.__dict__, "max_position_embeddings": 400}
    )
    ids = cli.fit_history(engine, messages, 24)
    assert len(ids) <= 376
    assert messages[0]["role"] == "system" and messages[-1]["content"] == "hi"
    assert messages[1]["role"] == "user"
    with pytest.raises(ValueError, match="too long"):
        cli.fit_history(engine, [{"role": "user", "content": "word " * 500}], 24)
