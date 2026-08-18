#!/usr/bin/env python3
"""Launch the 5090 pipeline with a CC Switch key sent only over SSH stdin."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproduction.run_with_cc_switch import (
    BASE_URL,
    DEFAULT_CC_SWITCH_DB,
    LOCKED_MODEL,
    CredentialError,
    load_key,
)


ALLOWED_HOSTS = {
    "fanruibo-5090-1",
    "fanruibo-10-123-4-13",
}
REMOTE_BASE = PurePosixPath("/nvme/home/fanruibo")
HF_ENDPOINT = "https://hf-mirror.com"
SCRUBBED_ENVIRONMENT = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_MAP",
    "LLM_COST_HARD_CAP_USD",
    "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
    "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
)
PRICE_MAP = {
    LOCKED_MODEL: {
        "input": 0.95,
        "output": 4.0,
        "cached_input": 0.16,
        "reasoning": 4.0,
    }
}

REMOTE_BOOTSTRAP = r"""
import json
import os
import subprocess
import sys

payload = json.load(sys.stdin)
os.umask(0o077)
environment = os.environ.copy()
for name in payload["scrub_environment"]:
    environment.pop(name, None)
environment.update(payload["environment"])
os.makedirs(os.path.dirname(payload["log_path"]), exist_ok=True)
with open(payload["log_path"], "ab", buffering=0) as log_handle:
    process = subprocess.Popen(
        payload["command"],
        cwd=payload["cwd"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(json.dumps({"pid": process.pid, "log_path": payload["log_path"]}))
""".strip()


def validate_remote_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or path == REMOTE_BASE:
        raise CredentialError(f"{label} must be a specific absolute path")
    try:
        path.relative_to(REMOTE_BASE)
    except ValueError as exc:
        raise CredentialError(f"{label} must be under {REMOTE_BASE}") from exc
    if ".." in path.parts:
        raise CredentialError(f"{label} must not contain parent traversal")
    return str(path)


def build_remote_environment(api_key: str, ledger_path: str) -> dict[str, str]:
    if not api_key.strip():
        raise CredentialError("locked Kimi API key is empty")
    return {
        "LLM_BACKEND": "compatible",
        "LLM_API_KEY": api_key,
        "LLM_BASE_URL": BASE_URL,
        "LLM_MODEL_MAP": json.dumps({LOCKED_MODEL: LOCKED_MODEL}),
        "LLM_LOCKED_MODEL": LOCKED_MODEL,
        "LLM_REASONING_EFFORT": "medium",
        "LLM_DISABLE_OUTPUT_LIMITS": "1",
        "LLM_REQUIRE_USAGE_ACCOUNTING": "1",
        "LLM_USAGE_LEDGER_PATH": ledger_path,
        "LLM_PRICE_MAP": json.dumps(PRICE_MAP),
        "HF_ENDPOINT": HF_ENDPOINT,
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": BASE_URL,
    }


def build_launch_payload(
    *,
    api_key: str,
    remote_cwd: str,
    remote_root: str,
    remote_cache: str,
    remote_log: str,
    remote_python: str,
) -> dict[str, Any]:
    cwd = validate_remote_path(remote_cwd, "remote cwd")
    root = validate_remote_path(remote_root, "remote root")
    cache = validate_remote_path(remote_cache, "remote cache")
    log_path = validate_remote_path(remote_log, "remote log")
    ledger = str(PurePosixPath(root) / "launcher_usage.jsonl")
    command = [
        remote_python,
        "-m",
        "reproduction.browsecomp_pipeline_5090",
        "--root",
        root,
        "--cache-root",
        cache,
    ]
    return {
        "cwd": cwd,
        "log_path": log_path,
        "command": command,
        "environment": build_remote_environment(api_key, ledger),
        "scrub_environment": list(SCRUBBED_ENVIRONMENT),
    }


def launch(host: str, payload: dict[str, Any]) -> dict[str, Any]:
    if host not in ALLOWED_HOSTS:
        raise CredentialError(f"unapproved 5090 SSH host: {host}")
    encoded = base64.b64encode(REMOTE_BOOTSTRAP.encode("utf-8")).decode("ascii")
    remote_command = (
        "python3 -c \"import base64;"
        f"exec(base64.b64decode('{encoded}'))\""
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            remote_command,
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-1000:]
        raise CredentialError(f"5090 launcher failed with code {result.returncode}: {detail}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CredentialError("5090 launcher returned invalid metadata") from exc
    if not isinstance(response, dict) or not isinstance(response.get("pid"), int):
        raise CredentialError("5090 launcher returned no process ID")
    return {"host": host, **response, "model": LOCKED_MODEL, "reasoning": "medium"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=sorted(ALLOWED_HOSTS), required=True)
    parser.add_argument("--remote-cwd", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-cache", required=True)
    parser.add_argument("--remote-log", required=True)
    parser.add_argument("--remote-python", default=".venv/bin/python")
    parser.add_argument(
        "--cc-switch-db",
        type=Path,
        default=DEFAULT_CC_SWITCH_DB,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = load_key(args.cc_switch_db)
    payload = build_launch_payload(
        api_key=api_key,
        remote_cwd=args.remote_cwd,
        remote_root=args.remote_root,
        remote_cache=args.remote_cache,
        remote_log=args.remote_log,
        remote_python=args.remote_python,
    )
    report = launch(args.host, payload)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except CredentialError as exc:
        print(f"5090 credential launcher error: {exc}")
        raise SystemExit(2)
