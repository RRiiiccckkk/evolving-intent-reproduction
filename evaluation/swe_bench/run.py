#!/usr/bin/env python3
"""Run the fixed Kimi SWE-bench Verified reproduction with fail-closed resume."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dataset import prepare_filtered_official_dataset
from .state import (
    EXPECTED_MODEL,
    EXPECTED_REASONING_EFFORT,
    MODEL_STEP_LIMIT_PER_TURN,
    PUBLISHED_TASK_COUNT,
    SCENARIOS,
    TOOL_CALL_LIMIT_PER_TURN,
    HardeningError,
    TaskCheckpointStore,
    atomic_write_json,
    file_sha256,
    ledger_offset,
    load_published_id_map,
    load_published_task_ids,
    read_json,
    read_usage_events,
    validate_aggregate_results,
    validate_data_coverage,
    validate_runtime_environment,
    validate_usage_events,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_MANIFEST = (
    REPO_ROOT / "intent_construction" / "eval_indices" / "swe_bench_verified_eval_ids.json"
)
PUBLISHED_TASK_IDS = (
    REPO_ROOT / "intent_construction" / "eval_indices" / "swe_bench_verified_task_ids.json"
)
RUNNER = REPO_ROOT / "evaluation" / "runners" / "run_swe_mini_agent.py"
DEFAULT_DATA_PATH = REPO_ROOT / "final_dataset" / "swe_bench_verified_final.json"
DEFAULT_OFFICIAL_DATASET_PATH = (
    REPO_ROOT / "tmp" / "source-data" / "swe_bench_verified_princeton.parquet"
)
DEFAULT_RUN_DIR = REPO_ROOT / "evaluation" / "swe_runs" / "kimi-k2.6"


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _scenario_manifest() -> list[dict[str, Any]]:
    return [
        {
            "name": scenario.name,
            "turns": scenario.turns,
            "revisions": scenario.revisions,
            "switches": scenario.switches,
            "tool_call_limit_per_turn": scenario.tool_call_limit_per_turn,
        }
        for scenario in SCENARIOS
    ]


def _new_manifest(
    *,
    run_dir: Path,
    data_path: Path,
    resolved_model: str,
    task_ids: Sequence[str],
    ledger_path: Path,
    official_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "initialized",
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
        "repository": {
            "path": str(REPO_ROOT),
            "commit": _git_commit(),
        },
        "models": {
            "requested": [EXPECTED_MODEL],
            "resolved": [resolved_model],
        },
        "backend": "compatible",
        "dataset": {
            "name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "data_path": _relative_or_absolute(data_path),
            "published_manifest": _relative_or_absolute(PUBLISHED_MANIFEST),
            "published_manifest_sha256": file_sha256(PUBLISHED_MANIFEST),
            "published_task_ids": _relative_or_absolute(PUBLISHED_TASK_IDS),
            "published_task_ids_sha256": file_sha256(PUBLISHED_TASK_IDS),
            "expected_count": PUBLISHED_TASK_COUNT,
            "task_ids": list(task_ids),
            "official_verifier_dataset": dict(official_dataset),
        },
        "scenarios": _scenario_manifest(),
        "runtime": {
            "agent_environment": "swerex_modal",
            "harness_modal": True,
            "parallel_tool_calls": False,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "model_step_limit_per_turn": MODEL_STEP_LIMIT_PER_TURN,
            "output_token_limit": None,
            "cost_hard_cap_usd": None,
            "usage_ledger": _relative_or_absolute(ledger_path),
            "run_dir": str(run_dir),
        },
        "scenario_runs": {},
        "usage": {},
    }


def _validate_existing_manifest(
    manifest: Mapping[str, Any],
    *,
    data_path: Path,
    resolved_model: str,
    task_ids: Sequence[str],
    official_dataset: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": 2,
        "models": {
            "requested": [EXPECTED_MODEL],
            "resolved": [resolved_model],
        },
        "scenarios": _scenario_manifest(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise HardeningError(
                f"existing manifest {key!r} does not match this hardened run"
            )
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise HardeningError("existing manifest has no dataset section")
    if dataset.get("task_ids") != list(task_ids):
        raise HardeningError("existing manifest task IDs do not match the published IDs")
    if dataset.get("data_path") != _relative_or_absolute(data_path):
        raise HardeningError("existing manifest data path changed; use a new run directory")
    if dataset.get("published_manifest_sha256") != file_sha256(PUBLISHED_MANIFEST):
        raise HardeningError("published SWE eval manifest changed during resume")
    if dataset.get("published_task_ids_sha256") != file_sha256(PUBLISHED_TASK_IDS):
        raise HardeningError("published SWE task-ID file changed during resume")
    if dataset.get("official_verifier_dataset") != dict(official_dataset):
        raise HardeningError(
            "official local SWE verifier dataset changed; use a new run directory"
        )
    if not isinstance(manifest.get("usage"), dict) or not isinstance(
        manifest.get("scenario_runs"), dict
    ):
        raise HardeningError("existing manifest has incomplete run-state sections")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get(
        "reasoning_effort"
    ) != EXPECTED_REASONING_EFFORT:
        raise HardeningError(
            "existing manifest reasoning effort does not match the hardened run"
        )


def build_runner_command(
    *,
    data_path: Path,
    task_ids_path: Path,
    run_dir: Path,
    official_dataset_path: Path,
    scenario_name: str,
    workers: int,
) -> list[str]:
    scenario = next(
        (candidate for candidate in SCENARIOS if candidate.name == scenario_name),
        None,
    )
    if scenario is None:
        raise HardeningError(f"unknown scenario: {scenario_name!r}")
    return [
        sys.executable,
        "-u",
        str(RUNNER),
        "--data_path",
        str(data_path),
        "--models",
        EXPECTED_MODEL,
        "--source",
        "paraphrase",
        "--task_ids_file",
        str(task_ids_path),
        "--num_workers",
        str(workers),
        "--cost_limit",
        "0",
        "--step_limit",
        "0",
        "--step_limit_per_turn",
        str(MODEL_STEP_LIMIT_PER_TURN),
        "--tool_call_limit_per_turn",
        str(TOOL_CALL_LIMIT_PER_TURN),
        "--environment_class",
        "swerex_modal",
        "--harness_dataset_path",
        str(official_dataset_path),
        "--use_tool_calling",
        "--reasoning_effort",
        EXPECTED_REASONING_EFFORT,
        "--strict_task_ids",
        "--checkpoint_dir",
        str(run_dir / "checkpoints" / scenario.name),
        "--checkpoint_scenario",
        scenario.name,
        "--output_path",
        str(run_dir / "results" / f"{scenario.name}.json"),
        "--num_turns",
        str(scenario.turns),
        "--num_revisions",
        str(scenario.revisions),
        "--num_switches",
        str(scenario.switches),
    ]


def _run_subprocess(
    command: Sequence[str],
    *,
    log_path: Path,
    environment: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_iso_now()}] command: {json.dumps(list(command))}\n")
        log.flush()
        completed = runner(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise HardeningError(
            f"SWE subprocess failed with exit code {completed.returncode}; see {log_path}"
        )


def _checkpoint_pending_ids(
    run_dir: Path,
    scenario_name: str,
    task_ids: Sequence[str],
    resolved_model: str,
) -> list[str]:
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    store = TaskCheckpointStore(
        run_dir / "checkpoints" / scenario.name,
        scenario=scenario,
        requested_model=EXPECTED_MODEL,
        resolved_model=resolved_model,
    )
    pending: list[str] = []
    for task_id in task_ids:
        if store.exists(task_id):
            store.load(task_id)
        else:
            pending.append(task_id)
    return pending


def _validate_scenario_output(
    run_dir: Path,
    scenario_name: str,
    task_ids: Sequence[str],
    resolved_model: str,
) -> dict[str, Any]:
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    store = TaskCheckpointStore(
        run_dir / "checkpoints" / scenario.name,
        scenario=scenario,
        requested_model=EXPECTED_MODEL,
        resolved_model=resolved_model,
    )
    output_path = run_dir / "results" / f"{scenario.name}.json"
    results = read_json(output_path, label=f"{scenario.name} aggregate result")
    return validate_aggregate_results(results, task_ids, store=store)


def run_hardened(
    *,
    data_path: str | Path,
    official_dataset_path: str | Path,
    run_dir: str | Path,
    workers: int,
    dry_run: bool = False,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    if workers < 1:
        raise HardeningError("workers must be at least 1")
    data = Path(data_path).resolve()
    output_dir = Path(run_dir).resolve()
    task_ids = load_published_task_ids(PUBLISHED_MANIFEST, PUBLISHED_TASK_IDS)
    validate_data_coverage(
        data,
        task_ids,
        expected_original_ids=load_published_id_map(PUBLISHED_MANIFEST),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    published_id_map = load_published_id_map(PUBLISHED_MANIFEST)
    official_dataset = prepare_filtered_official_dataset(
        official_dataset_path,
        instance_ids=[published_id_map[task_id] for task_id in task_ids],
        output_path=output_dir / "data" / "swe_bench_verified_official_n50.json",
    )
    filtered_official_path = Path(official_dataset["filtered_path"])
    ledger_path = output_dir / "usage.jsonl"
    runtime_environment = os.environ.copy()
    runtime_environment["LLM_USAGE_LEDGER_PATH"] = str(ledger_path)
    for setting, expected in (
        ("LLM_LOCKED_MODEL", EXPECTED_MODEL),
        ("LLM_REASONING_EFFORT", EXPECTED_REASONING_EFFORT),
    ):
        configured = runtime_environment.get(setting, "").strip()
        if configured and configured != expected:
            raise HardeningError(
                f"{setting} is {configured!r}, expected {expected!r}"
            )
        runtime_environment[setting] = expected
    resolved_model = validate_runtime_environment(runtime_environment)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path, label="SWE run manifest")
        if not isinstance(manifest, dict):
            raise HardeningError("SWE run manifest must be a JSON object")
        _validate_existing_manifest(
            manifest,
            data_path=data,
            resolved_model=resolved_model,
            task_ids=task_ids,
            official_dataset=official_dataset,
        )
    else:
        manifest = _new_manifest(
            run_dir=output_dir,
            data_path=data,
            resolved_model=resolved_model,
            task_ids=task_ids,
            ledger_path=ledger_path,
            official_dataset=official_dataset,
        )
    manifest["status"] = "validated" if dry_run else "running"
    manifest.pop("failure", None)
    manifest["updated_at"] = _iso_now()
    invocation_start = ledger_offset(ledger_path)
    manifest["usage"]["invocation_start_offset"] = invocation_start
    atomic_write_json(manifest_path, manifest)

    if dry_run:
        return manifest

    try:
        for scenario in SCENARIOS:
            pending_ids = _checkpoint_pending_ids(
                output_dir,
                scenario.name,
                task_ids,
                resolved_model,
            )
            pending = len(pending_ids)
            start_offset = ledger_offset(ledger_path)
            command = build_runner_command(
                data_path=data,
                task_ids_path=PUBLISHED_TASK_IDS,
                run_dir=output_dir,
                official_dataset_path=filtered_official_path,
                scenario_name=scenario.name,
                workers=workers,
            )
            _run_subprocess(
                command,
                log_path=output_dir / "logs" / f"{scenario.name}.log",
                environment=runtime_environment,
                runner=subprocess_runner,
            )
            results = _validate_scenario_output(
                output_dir,
                scenario.name,
                task_ids,
                resolved_model,
            )
            end_offset = ledger_offset(ledger_path)
            new_events = read_usage_events(ledger_path, start_offset=start_offset)
            new_usage = validate_usage_events(
                new_events,
                requested_model=EXPECTED_MODEL,
                resolved_model=resolved_model,
            )
            if pending and not new_events:
                verifier_only_retry = all(
                    (results[task_id].get("metadata") or {}).get(
                        "agent_reused_from_trajectory"
                    )
                    is True
                    for task_id in pending_ids
                )
                if not verifier_only_retry:
                    raise HardeningError(
                        f"scenario {scenario.name} completed pending tasks without recorded usage"
                    )
            manifest["scenario_runs"][scenario.name] = {
                "status": "complete",
                "expected": PUBLISHED_TASK_COUNT,
                "completed": len(results),
                "pending_before_run": pending,
                "ledger_start_offset": start_offset,
                "ledger_end_offset": end_offset,
                "usage": new_usage,
                "result_path": _relative_or_absolute(
                    output_dir / "results" / f"{scenario.name}.json"
                ),
            }
            manifest["updated_at"] = _iso_now()
            atomic_write_json(manifest_path, manifest)

        all_events = read_usage_events(ledger_path)
        total_usage = validate_usage_events(
            all_events,
            requested_model=EXPECTED_MODEL,
            resolved_model=resolved_model,
        )
        invocation_events = read_usage_events(
            ledger_path,
            start_offset=invocation_start,
        )
        invocation_usage = validate_usage_events(
            invocation_events,
            requested_model=EXPECTED_MODEL,
            resolved_model=resolved_model,
        )
        manifest["usage"].update(
            {
                "ledger_end_offset": ledger_offset(ledger_path),
                "invocation": invocation_usage,
                "total": total_usage,
            }
        )
        manifest["status"] = "complete"
        manifest["updated_at"] = _iso_now()
        atomic_write_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_at"] = _iso_now()
        manifest["usage"]["ledger_end_offset"] = ledger_offset(ledger_path)
        atomic_write_json(manifest_path, manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--official-dataset-path",
        default=str(DEFAULT_OFFICIAL_DATASET_PATH),
        help="Local Princeton SWE-bench Verified parquet (filtered to the published 50 before use).",
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate IDs, data, provider policy, and manifest without starting agents.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run_hardened(
            data_path=args.data_path,
            official_dataset_path=args.official_dataset_path,
            run_dir=args.run_dir,
            workers=args.workers,
            dry_run=args.dry_run,
        )
    except (HardeningError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"SWE reproduction status={manifest['status']} "
        f"model={EXPECTED_MODEL} tasks={PUBLISHED_TASK_COUNT} "
        f"manifest={Path(args.run_dir).resolve() / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
