#!/usr/bin/env python3
"""Run one isolated evolve task through the Modal agent and official verifier."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evaluation.common.swe_harness import SWEHarness

from .dataset import prepare_filtered_official_dataset
from .run import (
    DEFAULT_DATA_PATH,
    DEFAULT_OFFICIAL_DATASET_PATH,
    PUBLISHED_MANIFEST,
    PUBLISHED_TASK_IDS,
    REPO_ROOT,
    RUNNER,
)
from .state import (
    EXPECTED_MODEL,
    MODEL_STEP_LIMIT_PER_TURN,
    SCENARIO_BY_NAME,
    TOOL_CALL_LIMIT_PER_TURN,
    HardeningError,
    TaskCheckpointStore,
    atomic_write_json,
    load_published_id_map,
    load_published_task_ids,
    read_json,
    read_usage_events,
    validate_aggregate_results,
    validate_runtime_environment,
    validate_usage_events,
)


def _validate_reusable_agent_result(
    candidate: Any,
    *,
    task_id: str,
    original_id: str,
) -> str:
    """Validate a completed canary agent result before verifier-only resume."""
    if not isinstance(candidate, dict):
        raise HardeningError("existing SWE canary result is not an object")
    if candidate.get("task_id") != task_id:
        raise HardeningError("existing SWE canary task ID changed")
    if candidate.get("success") is not True or candidate.get("error") not in (None, ""):
        raise HardeningError("existing SWE canary agent result is failed")
    patch = candidate.get("prediction")
    if not isinstance(patch, str) or not patch.strip():
        raise HardeningError("existing SWE canary agent result has no patch")
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        raise HardeningError("existing SWE canary agent metadata is missing")
    expected = {
        "original_id": original_id,
        "requested_model": EXPECTED_MODEL,
        "resolved_model": EXPECTED_MODEL,
        "reasoning_effort": "medium",
        "checkpoint_scenario": "evolve",
        "n_user_turns_delivered": SCENARIO_BY_NAME["evolve"].turns,
        "tool_call_limit_per_turn": TOOL_CALL_LIMIT_PER_TURN,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise HardeningError(
            "existing SWE canary agent metadata changed: " + ", ".join(mismatches)
        )
    per_turn = metadata.get("per_turn_tool_calls")
    if (
        not isinstance(per_turn, list)
        or len(per_turn) != SCENARIO_BY_NAME["evolve"].turns
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count > TOOL_CALL_LIMIT_PER_TURN
            for count in per_turn
        )
    ):
        raise HardeningError("existing SWE canary per-turn tool accounting is invalid")
    return patch


def _verify_existing_agent_result(
    candidate: dict[str, Any],
    *,
    original_id: str,
    official_path: Path,
    workspace: Path,
) -> dict[str, Any]:
    """Run only the official verifier and return the updated result row."""
    patch = str(candidate["prediction"])
    result = SWEHarness(
        workspace=workspace,
        dataset_name=str(official_path),
        timeout_s=1800,
        use_cache=False,
        modal=True,
    ).verify_patch(
        instance_id=original_id,
        patch=patch,
        model_name=f"{EXPECTED_MODEL}-mini-agent",
        run_id="canary_verifier_resume",
    )
    updated = dict(candidate)
    updated["correct"] = result.resolved
    updated["swe_eval"] = {
        "resolved": result.resolved,
        "patch_extracted": result.patch_extracted,
        "patch_apply_ok": result.patch_apply_ok,
        "ftp_pass": result.ftp_pass,
        "ftp_fail": result.ftp_fail,
        "ptp_pass": result.ptp_pass,
        "ptp_fail": result.ptp_fail,
        "harness_error": result.harness_error,
        "duration_s": result.duration_s,
        "from_cache": result.from_cache,
        "instance_id": original_id,
    }
    return updated


def build_canary_runner_command(
    *,
    data_path: Path,
    task_ids_path: Path,
    official_dataset_path: Path,
    run_dir: Path,
) -> list[str]:
    scenario = SCENARIO_BY_NAME["evolve"]
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
        "1",
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
        "medium",
        "--rerun_failed",
        "--checkpoint_scenario",
        "evolve",
        "--output_path",
        str(run_dir / "results/evolve.json"),
        "--num_turns",
        str(scenario.turns),
        "--num_revisions",
        str(scenario.revisions),
        "--num_switches",
        str(scenario.switches),
    ]


def _select_one_data_row(data_path: Path, task_id: str) -> dict[str, Any]:
    payload = read_json(data_path, label="SWE canary construction data")
    if not isinstance(payload, list):
        raise HardeningError("SWE canary construction data must be a list")
    matches = [row for row in payload if isinstance(row, dict) and row.get("task_id") == task_id]
    if len(matches) != 1:
        raise HardeningError(
            f"SWE canary data must contain exactly one row for {task_id}; found {len(matches)}"
        )
    row = matches[0]
    if not isinstance(row.get("original_id"), str) or not row["original_id"]:
        raise HardeningError("SWE canary row has no original_id")
    return row


def run_canary(
    *,
    task_id: str,
    data_path: str | Path,
    official_source_path: str | Path,
    run_dir: str | Path,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    formal_run_path = (REPO_ROOT / "evaluation/swe_runs/kimi-k2.6").resolve()
    if run_path == formal_run_path or formal_run_path in run_path.parents:
        raise HardeningError("canary must not use the formal 50-task run directory")
    run_path.mkdir(parents=True, exist_ok=True)
    published_ids = load_published_task_ids(PUBLISHED_MANIFEST, PUBLISHED_TASK_IDS)
    if task_id not in published_ids:
        raise HardeningError("canary task must be one of the published 50 IDs")
    row = _select_one_data_row(Path(data_path).resolve(), task_id)
    expected_original = load_published_id_map(PUBLISHED_MANIFEST)[task_id]
    if row["original_id"] != expected_original:
        raise HardeningError("canary task/original ID mapping differs from the published set")

    effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
    if effort and effort != "medium":
        raise HardeningError("SWE canary requires LLM_REASONING_EFFORT=medium")
    os.environ["LLM_REASONING_EFFORT"] = "medium"
    os.environ["LLM_REQUIRE_USAGE_ACCOUNTING"] = "1"
    os.environ["LLM_USAGE_LEDGER_PATH"] = str((run_path / "usage.jsonl").resolve())
    resolved_model = validate_runtime_environment(os.environ)

    selected_data_path = run_path / "data/canary.json"
    task_ids_path = run_path / "data/task_ids.json"
    official_path = run_path / "data/official.json"
    atomic_write_json(selected_data_path, [row])
    atomic_write_json(task_ids_path, {"task_ids": [task_id], "n_total": 1})
    official = prepare_filtered_official_dataset(
        official_source_path,
        instance_ids=[expected_original],
        output_path=official_path,
    )
    results_path = run_path / "results/evolve.json"
    if results_path.exists():
        results = read_json(results_path, label="SWE canary aggregate")
        candidate = results.get(task_id) if isinstance(results, dict) else None
        _validate_reusable_agent_result(
            candidate,
            task_id=task_id,
            original_id=expected_original,
        )
        swe_eval = candidate.get("swe_eval") if isinstance(candidate, dict) else None
        if not isinstance(swe_eval, dict) or swe_eval.get("harness_error") not in (None, ""):
            candidate = _verify_existing_agent_result(
                candidate,
                original_id=expected_original,
                official_path=official_path,
                workspace=run_path / "official_workspace",
            )
            results = {task_id: candidate}
            atomic_write_json(results_path, results)
    else:
        command = build_canary_runner_command(
            data_path=selected_data_path,
            task_ids_path=task_ids_path,
            official_dataset_path=official_path,
            run_dir=run_path,
        )
        log_path = run_path / "canary.log"
        with log_path.open("a", encoding="utf-8") as log:
            completed = runner(
                command,
                cwd=REPO_ROOT,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise HardeningError(
                f"SWE canary runner failed with exit code {completed.returncode}; see {log_path}"
            )
        results = read_json(results_path, label="SWE canary aggregate")
        candidate = results.get(task_id) if isinstance(results, dict) else None

    if not isinstance(candidate, dict):
        raise HardeningError("SWE canary aggregate has no task result")
    if candidate.get("success") is not True or candidate.get("error") not in (None, ""):
        raise HardeningError(
            "SWE canary agent failed before verification: "
            f"{candidate.get('error') or 'unknown error'}"
        )
    swe_eval = candidate.get("swe_eval")
    if not isinstance(swe_eval, dict) or swe_eval.get("harness_error") not in (None, ""):
        error = swe_eval.get("harness_error") if isinstance(swe_eval, dict) else None
        raise HardeningError(
            "SWE canary official verifier failed: " + str(error or "missing result")
        )
    store = TaskCheckpointStore(
        run_path / "checkpoints/evolve",
        scenario=SCENARIO_BY_NAME["evolve"],
        requested_model=EXPECTED_MODEL,
        resolved_model=resolved_model,
    )
    store.write(task_id, candidate)
    validate_aggregate_results(results, [task_id], store=store)
    usage = validate_usage_events(
        read_usage_events(run_path / "usage.jsonl"),
        requested_model=EXPECTED_MODEL,
        resolved_model=resolved_model,
    )
    if usage["calls"] < 1:
        raise HardeningError("SWE canary completed without provider usage records")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "mode": "one_task_agent_and_official_verifier_canary",
        "task_id": task_id,
        "original_id": expected_original,
        "model": EXPECTED_MODEL,
        "resolved_model": resolved_model,
        "reasoning_effort": "medium",
        "model_step_limit_per_turn": MODEL_STEP_LIMIT_PER_TURN,
        "agent_environment": "swerex_modal",
        "official_harness_modal": True,
        "official_dataset": official,
        "coverage": 1,
        "usage": usage,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(run_path / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--official-source-path", default=str(DEFAULT_OFFICIAL_DATASET_PATH))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = run_canary(
        task_id=args.task_id,
        data_path=args.data_path,
        official_source_path=args.official_source_path,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
