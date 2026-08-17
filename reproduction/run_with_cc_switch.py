#!/usr/bin/env python3
"""Run a command with the locked Kimi credential loaded from CC Switch.

The secret is read from the SQLite database into process memory, exported only
to the child environment, and never written to a file or command argument.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path


LOCKED_MODEL = "kimi-k2.6"
PROVIDER_ID = "597d31e6-33dd-4554-9f6a-de211b7d4c80"
DEFAULT_CC_SWITCH_DB = Path("/Users/rick/.cc-switch/cc-switch.db")
BASE_URL = "https://api.moonshot.cn/v1"
FORBIDDEN_ARGUMENTS = {
    "--max_tokens",
    "--max_completion_tokens",
    "--max_output_tokens",
}


class CredentialError(RuntimeError):
    """Raised when the locked CC Switch provider cannot be used safely."""


def load_key(database: Path) -> str:
    if not database.is_file():
        raise CredentialError(f"CC Switch database does not exist: {database}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT settings_config FROM providers "
            "WHERE id = ? AND app_type = 'codex'",
            (PROVIDER_ID,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise CredentialError("locked Kimi Codex provider is missing from CC Switch")
    try:
        value = json.loads(row[0])["auth"]["OPENAI_API_KEY"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CredentialError("locked Kimi provider has no readable API key") from exc
    if not isinstance(value, str) or not value.strip():
        raise CredentialError("locked Kimi provider API key is empty")
    return value.strip()


def assert_model_available(api_key: str) -> None:
    request = urllib.request.Request(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:
        raise CredentialError("unable to verify the locked model with Moonshot") from exc
    model_ids = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    if LOCKED_MODEL not in model_ids:
        raise CredentialError(f"Moonshot does not advertise {LOCKED_MODEL}")


def validate_command(command: list[str]) -> None:
    if not command:
        raise CredentialError("a child command is required after --")
    for argument in command:
        field = argument.split("=", 1)[0]
        if field in FORBIDDEN_ARGUMENTS:
            raise CredentialError(f"output token limit is forbidden: {field}")


def build_environment(api_key: str, ledger: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_MAP",
        "LLM_COST_HARD_CAP_USD",
        "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
        "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
    ):
        environment.pop(name, None)

    environment.update(
        {
            "LLM_BACKEND": "compatible",
            "LLM_API_KEY": api_key,
            "LLM_BASE_URL": BASE_URL,
            "LLM_MODEL_MAP": json.dumps({LOCKED_MODEL: LOCKED_MODEL}),
            "LLM_LOCKED_MODEL": LOCKED_MODEL,
            "LLM_REASONING_EFFORT": "medium",
            "LLM_DISABLE_OUTPUT_LIMITS": "1",
            "LLM_USAGE_LEDGER_PATH": str(ledger.resolve()),
            "LLM_PRICE_MAP": json.dumps(
                {
                    LOCKED_MODEL: {
                        "input": 0.95,
                        "output": 4.0,
                        "cached_input": 0.16,
                        "reasoning": 4.0,
                    }
                }
            ),
            # Some domain runners construct an OpenAI client directly.
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": BASE_URL,
        }
    )
    return environment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cc-switch-db",
        type=Path,
        default=DEFAULT_CC_SWITCH_DB,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--skip-model-preflight", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_command(args.command)
    api_key = load_key(args.cc_switch_db)
    if not args.skip_model_preflight:
        assert_model_available(api_key)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    environment = build_environment(api_key, args.ledger)
    print(
        f"Launching with locked model {LOCKED_MODEL} at medium reasoning; "
        f"usage ledger: {args.ledger.resolve()}",
        flush=True,
    )
    os.execvpe(args.command[0], args.command, environment)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CredentialError as exc:
        print(f"credential launcher error: {exc}", file=sys.stderr)
        raise SystemExit(2)
