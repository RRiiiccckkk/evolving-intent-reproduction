from __future__ import annotations

import json
import sqlite3

import pytest

from reproduction import run_with_cc_switch as launcher
from intent_construction.intent_extraction.core import llm_utils


def _database(tmp_path, settings):
    path = tmp_path / "cc-switch.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE providers "
        "(id TEXT, app_type TEXT, settings_config TEXT)"
    )
    connection.execute(
        "INSERT INTO providers VALUES (?, 'codex', ?)",
        (launcher.PROVIDER_ID, json.dumps(settings)),
    )
    connection.commit()
    connection.close()
    return path


def test_load_key_reads_locked_provider_without_persisting_it(tmp_path):
    database = _database(tmp_path, {"auth": {"OPENAI_API_KEY": "secret-value"}})
    assert launcher.load_key(database) == "secret-value"
    assert "secret-value" not in repr(launcher.build_environment("x", tmp_path / "l"))


@pytest.mark.parametrize(
    "argument",
    [
        "--max_tokens=5",
        "--max_completion_tokens",
        "--max_output_tokens=100",
    ],
)
def test_validate_command_rejects_output_limits(argument):
    with pytest.raises(launcher.CredentialError, match="output token limit"):
        launcher.validate_command(["python", "runner.py", argument])


def test_build_environment_locks_model_and_removes_cost_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_COST_HARD_CAP_USD", "30")
    monkeypatch.setenv("LLM_DEFAULT_MAX_OUTPUT_TOKENS", "8192")
    environment = launcher.build_environment("key", tmp_path / "usage.jsonl")
    assert environment["LLM_LOCKED_MODEL"] == "kimi-k2.6"
    assert environment["LLM_REASONING_EFFORT"] == "medium"
    assert json.loads(environment["LLM_MODEL_MAP"]) == {
        "kimi-k2.6": "kimi-k2.6"
    }
    assert environment["LLM_DISABLE_OUTPUT_LIMITS"] == "1"
    assert "LLM_COST_HARD_CAP_USD" not in environment
    assert "LLM_DEFAULT_MAX_OUTPUT_TOKENS" not in environment


def test_parse_args_preserves_child_command(tmp_path):
    args = launcher.parse_args(
        ["--ledger", str(tmp_path / "usage.jsonl"), "--", "python", "x.py"]
    )
    assert args.command == ["python", "x.py"]


def test_shared_provider_refuses_any_other_requested_model(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "compatible")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_LOCKED_MODEL", "kimi-k2.6")
    monkeypatch.setenv("LLM_MODEL_MAP", '{"kimi-k2.6":"kimi-k2.6"}')
    assert llm_utils.resolve_model_name("kimi-k2.6") == "kimi-k2.6"
    with pytest.raises(llm_utils.LLMAccountingError, match="model lock"):
        llm_utils.resolve_model_name("gpt-5.1")


def test_locked_request_forces_medium_reasoning_and_no_output_limit(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "compatible")
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")
    monkeypatch.setenv("LLM_DISABLE_OUTPUT_LIMITS", "1")
    monkeypatch.delenv("LLM_USAGE_LEDGER_PATH", raising=False)
    captured = {}

    def create(**payload):
        captured.update(payload)
        return object()

    llm_utils._accounted_api_call(
        create,
        {
            "model": "kimi-k2.6",
            "messages": [],
            "temperature": 0.7,
            "max_completion_tokens": 2048,
        },
        requested_model="kimi-k2.6",
        resolved_model="kimi-k2.6",
        api="chat.completions",
        max_output_tokens=2048,
    )
    assert captured["reasoning_effort"] == "medium"
    assert "temperature" not in captured
    for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        assert field not in captured
