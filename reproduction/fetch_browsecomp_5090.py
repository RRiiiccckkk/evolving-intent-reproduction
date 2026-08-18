#!/usr/bin/env python3
"""Fetch a completed private BrowseComp+ run from the selected RTX 5090."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Sequence

from reproduction.browsecomp_construction_5090 import validate_migrated_root
from reproduction.browsecomp_plan_a import WorkflowError
from reproduction.finalize_remaining_experiments import (
    FinalizationError,
    validate_browsecomp_evaluation_artifacts,
)


ALLOWED_HOSTS = {"fanruibo-5090-1", "fanruibo-10-123-4-13"}
DEFAULT_REMOTE_ROOT = "/nvme/home/fanruibo/evolving-intent-private/browsecomp-n100"
DEFAULT_REMOTE_REPO = "/nvme/home/fanruibo/evolving-intent-reproduction"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_BASE = PurePosixPath("/nvme/home/fanruibo")


class FetchError(RuntimeError):
    """Raised when a completed private run cannot be fetched safely."""


def validate_remote_path(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", value):
        raise FetchError(f"{label} contains unsupported characters")
    path = PurePosixPath(value)
    if not path.is_absolute() or path == REMOTE_BASE or ".." in path.parts:
        raise FetchError(f"{label} must be a specific absolute path")
    try:
        path.relative_to(REMOTE_BASE)
    except ValueError as exc:
        raise FetchError(f"{label} must be under {REMOTE_BASE}") from exc
    return str(path)


def _run(arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip()[-1000:]
        raise FetchError(
            f"command failed with code {result.returncode}: {arguments[0]}: {detail}"
        )
    return result


def remote_state(host: str, remote_root: str) -> dict[str, Any]:
    if host not in ALLOWED_HOSTS:
        raise FetchError(f"unapproved 5090 SSH host: {host}")
    remote_root = validate_remote_path(remote_root, "remote root")
    result = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            "cat",
            f"{remote_root}/pipeline_state.json",
        ]
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FetchError("remote pipeline state is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FetchError("remote pipeline state is not an object")
    return payload


def _assert_ignored(repo_root: Path, path: Path) -> None:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise FetchError(f"private destination is outside repository: {path}") from exc
    probe = relative / ".private-fetch-probe"
    result = _run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", str(probe)],
        check=False,
    )
    if result.returncode != 0:
        raise FetchError(f"private destination is not ignored by Git: {relative}")


def construction_rsync_command(
    host: str,
    remote_root: str,
    local_root: Path,
) -> list[str]:
    includes = (
        "/manifest.json",
        "/pipeline_state.json",
        "/remote_audit.json",
        "/evaluation_usage.jsonl",
        "/usage/",
        "/usage/***",
        "/run/",
        "/run/***",
        "/final_dataset/",
        "/final_dataset/***",
    )
    command = ["rsync", "-a", "--checksum", "--prune-empty-dirs"]
    for value in includes:
        command.extend(["--include", value])
    command.extend(
        ["--exclude", "*", f"{host}:{remote_root.rstrip('/')}/", f"{local_root}/"]
    )
    return command


def evaluation_rsync_command(
    host: str,
    remote_repo: str,
    scenario: str,
    local_destination: Path,
) -> list[str]:
    if scenario not in {"fully_specified", "combined_independent"}:
        raise FetchError(f"unsupported BrowseComp+ scenario directory: {scenario}")
    remote = (
        f"{remote_repo.rstrip('/')}/evaluation/experiments/{scenario}/"
        "browsecomp_plus_n100/"
    )
    return ["rsync", "-a", "--checksum", f"{host}:{remote}", f"{local_destination}/"]


def fetch(
    *,
    host: str,
    remote_root: str,
    remote_repo: str,
    repo_root: Path,
) -> dict[str, Any]:
    remote_root = validate_remote_path(remote_root, "remote root")
    remote_repo = validate_remote_path(remote_repo, "remote repository")
    state = remote_state(host, remote_root)
    if state.get("status") != "complete":
        raise FetchError(
            f"refusing to fetch an incomplete remote pipeline: {state.get('status')!r}"
        )

    repo_root = repo_root.resolve()
    local_root = repo_root / "reproduction/runs/browsecomp-plan-a-n100"
    local_single = (
        repo_root
        / "evaluation/experiments/fully_specified/browsecomp_plus_n100"
    )
    local_evolve = (
        repo_root
        / "evaluation/experiments/combined_independent/browsecomp_plus_n100"
    )
    for destination in (local_root, local_single, local_evolve):
        _assert_ignored(repo_root, destination)
        destination.mkdir(parents=True, exist_ok=True)

    _run(construction_rsync_command(host, remote_root, local_root))
    _run(
        evaluation_rsync_command(
            host, remote_repo, "fully_specified", local_single
        )
    )
    _run(
        evaluation_rsync_command(
            host, remote_repo, "combined_independent", local_evolve
        )
    )

    construction_usage = local_root / "usage/llm_usage.jsonl"
    audit_candidates = (
        local_root / "run/remote_audit.json",
        local_root / "remote_audit.json",
    )
    audit = next((path for path in audit_candidates if path.is_file()), audit_candidates[0])
    if not construction_usage.is_file() or not audit.is_file():
        raise FetchError("completed construction export is missing audit or usage")
    shutil.copy2(construction_usage, local_root / "llm_usage.jsonl")
    remote_audit = local_root / "remote_audit.json"
    if audit.resolve() != remote_audit.resolve():
        shutil.copy2(audit, remote_audit)
    shutil.copy2(audit, local_root / "local_audit.json")

    try:
        preflight = validate_migrated_root(local_root)
    except WorkflowError as exc:
        raise FetchError(f"fetched construction checkpoint failed validation: {exc}") from exc
    if preflight["stage_counts"].get("stage3_predecessor") != 100:
        raise FetchError("fetched Stage 3 coverage is not 100/100")

    expected_results = (
        local_single
        / "kimi-k2.6_naturalized_reasoning-medium_force-final.json",
        local_evolve
        / "kimi-k2.6_t7_g2_p2_naturalized_reasoning-medium_force-final.json",
    )
    missing = [path.name for path in expected_results if not path.is_file()]
    if missing:
        raise FetchError(f"fetched evaluation results are missing: {missing}")
    task_ids_path = (
        repo_root
        / "intent_construction/eval_indices/browsecomp_plus_task_ids.json"
    )
    try:
        task_id_payload = json.loads(task_ids_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FetchError("fixed BrowseComp+ task IDs cannot be read") from exc
    task_ids = (
        task_id_payload.get("task_ids")
        if isinstance(task_id_payload, dict)
        else task_id_payload
    )
    if not isinstance(task_ids, list):
        raise FetchError("fixed BrowseComp+ task IDs are not a list")
    try:
        evaluation_audit = validate_browsecomp_evaluation_artifacts(
            repo_root=repo_root,
            task_ids=task_ids,
            single_path=expected_results[0],
            evolve_path=expected_results[1],
        )
    except FinalizationError as exc:
        raise FetchError(str(exc)) from exc
    return {
        "status": "complete",
        "runtime": state.get("runtime"),
        "stage_counts": preflight["stage_counts"],
        "evaluation": evaluation_audit,
        "private_destinations_git_ignored": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=sorted(ALLOWED_HOSTS), required=True)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = fetch(
            host=args.host,
            remote_root=args.remote_root,
            remote_repo=args.remote_repo,
            repo_root=args.repo_root,
        )
    except (FetchError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
