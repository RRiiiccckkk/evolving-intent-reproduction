#!/usr/bin/env python3
"""Validate and summarize the three remaining paper experiments.

The finalizer is deliberately read-only with respect to experiment artifacts.
It only writes the requested JSON and HTML reports, and it projects raw result
files onto aggregate counts so BrowseComp+ questions and documents can never be
copied into a report.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL = "kimi-k2.6"
EXPECTED_REASONING_EFFORT = "medium"
BROWSE_RETRIEVER_REVISION = (
    "browsecomp-retriever-"
    "e10361ce3ec95089dd79d3058cb33b73cb3b1da043b239c3525a6c7a7abd5ece"
)
EXPECTED_OUTPUT_LIMIT_FIELDS = {
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
}
EXPECTED_COUNTS = {
    "bird_sql": 100,
    "browsecomp_plus": 100,
    "swe_bench_verified": 50,
}
SCENARIOS = {
    "single": {"turns": 1, "revisions": 0, "switches": 0},
    "evolve": {"turns": 7, "revisions": 2, "switches": 2},
}
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "reasoning_tokens",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FinalizationError(RuntimeError):
    """Raised when a formal report cannot be proven complete and compliant."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True)
class ReportLayout:
    repo_root: Path
    config_path: Path
    experiments_dir: Path
    bird_run_dir: Path
    browse_run_dir: Path
    swe_construction_run_dir: Path
    swe_run_dir: Path
    output_json: Path
    output_html: Path
    extra_ledgers: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)

    @classmethod
    def defaults(
        cls,
        repo_root: Path = REPOSITORY_ROOT,
        *,
        swe_run_dir: Path | None = None,
        output_json: Path | None = None,
        output_html: Path | None = None,
        extra_ledgers: Mapping[str, Sequence[Path]] | None = None,
    ) -> "ReportLayout":
        root = repo_root.resolve()
        json_path = output_json or root / "reproduction" / "remaining_experiments_report.json"
        html_path = output_html or root / "reproduction" / "remaining_experiments_report.html"
        return cls(
            repo_root=root,
            config_path=root / "reproduction" / "config" / "paper_remaining_kimi_k2_6.json",
            experiments_dir=root / "evaluation" / "experiments",
            bird_run_dir=root / "reproduction" / "runs" / "bird-sql-kimi-k2.6",
            browse_run_dir=root / "reproduction" / "runs" / "browsecomp-plan-a-n100",
            swe_construction_run_dir=(
                root / "reproduction" / "runs" / "swe-bench-verified-kimi-k2.6"
            ),
            swe_run_dir=(swe_run_dir or root / "evaluation" / "swe_runs" / "kimi-k2.6").resolve(),
            output_json=Path(json_path).resolve(),
            output_html=Path(html_path).resolve(),
            extra_ledgers={
                name: tuple(Path(path).resolve() for path in paths)
                for name, paths in (extra_ledgers or {}).items()
            },
        )


@dataclass
class _Issues:
    values: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        if message not in self.values:
            self.values.append(message)


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, issues: _Issues, label: str) -> Any | None:
    if not path.is_file():
        issues.add(f"{label}: artifact is missing")
        return None
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("empty file")
        return json.loads(text, object_pairs_hook=_json_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        issues.add(f"{label}: artifact is not valid JSON ({type(exc).__name__})")
        return None


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return f"<external>/{path.name}"


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_task_ids(
    path: Path,
    expected_count: int,
    issues: _Issues,
    label: str,
) -> list[str]:
    payload = _read_json(path, issues, label)
    values = payload.get("task_ids") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        issues.add(f"{label}: task_ids is not a string list")
        return []
    if len(values) != len(set(values)):
        issues.add(f"{label}: task IDs contain duplicates")
    if len(values) != expected_count:
        issues.add(
            f"{label}: expected {expected_count} task IDs, found {len(values)}"
        )
    declared = payload.get("num_samples") if isinstance(payload, Mapping) else None
    if declared is not None and declared != expected_count:
        issues.add(f"{label}: declared sample count is not {expected_count}")
    return list(values)


def _validate_global_policy(config: Any, issues: _Issues) -> bool:
    before = len(issues.values)
    if not isinstance(config, Mapping):
        issues.add("run policy: configuration is missing or malformed")
        return False
    lock = config.get("model_lock")
    if not isinstance(lock, Mapping):
        issues.add("run policy: model lock is missing")
    else:
        for role in ("construction", "evaluation", "judge", "naturalizer"):
            if lock.get(role) != EXPECTED_MODEL:
                issues.add(f"run policy: {role} model is not {EXPECTED_MODEL}")
        if lock.get("reasoning_effort") != EXPECTED_REASONING_EFFORT:
            issues.add(
                f"run policy: reasoning effort is not {EXPECTED_REASONING_EFFORT}"
            )
        if lock.get("allow_other_models") is not False:
            issues.add("run policy: other models are not explicitly forbidden")

    accounting = config.get("accounting")
    if not isinstance(accounting, Mapping) or accounting.get("hard_cap_usd", "missing") is not None:
        issues.add("run policy: hard cost cap is not explicitly null")
    if not isinstance(accounting, Mapping) or accounting.get(
        "reservation_is_confirmed_spend"
    ) is not False:
        issues.add("run policy: reservations are not excluded from confirmed spend")
    api_policy = config.get("api_policy")
    forbidden = api_policy.get("forbidden_request_fields") if isinstance(api_policy, Mapping) else None
    if (
        not isinstance(forbidden, list)
        or not all(isinstance(value, str) for value in forbidden)
        or not EXPECTED_OUTPUT_LIMIT_FIELDS.issubset(set(forbidden))
    ):
        issues.add("run policy: output-token request fields are not all forbidden")

    scenarios = config.get("scenarios")
    for name, expected in SCENARIOS.items():
        row = scenarios.get(name) if isinstance(scenarios, Mapping) else None
        if not isinstance(row, Mapping):
            issues.add(f"run policy: {name} scenario is missing")
            continue
        observed = {
            "turns": row.get("turns"),
            "revisions": row.get("argument_revisions"),
            "switches": row.get("function_switches"),
        }
        if observed != expected:
            issues.add(f"run policy: {name} scenario does not match the paper setting")

    benchmarks = config.get("benchmarks")
    for name, count in EXPECTED_COUNTS.items():
        row = benchmarks.get(name) if isinstance(benchmarks, Mapping) else None
        if not isinstance(row, Mapping) or row.get("sample_count") != count:
            issues.add(f"run policy: {name} sample count is not {count}")
    return len(issues.values) == before


def _record_validation_counts(
    issues: _Issues,
    label: str,
    counts: Counter[str],
) -> None:
    messages = {
        "not_object": "rows are not objects",
        "task_id": "rows have a mismatched task ID",
        "success": "rows are failed or carry an error",
        "responses": "rows have incomplete responses",
        "prediction": "rows have no final prediction",
        "model": f"rows do not use only {EXPECTED_MODEL}",
        "reasoning": f"rows do not use reasoning={EXPECTED_REASONING_EFFORT}",
        "verifier": "rows do not carry a successful native verifier result",
        "correct": "rows have no boolean correctness result",
        "scenario": "rows do not match the scenario policy",
    }
    for key in sorted(counts):
        issues.add(f"{label}: {counts[key]} {messages.get(key, key)}")


def _bird_row(row: Any, task_id: str, scenario: str) -> tuple[bool, bool, str | None]:
    if not isinstance(row, Mapping):
        return False, False, "not_object"
    if row.get("task_id") != task_id:
        return False, False, "task_id"
    if row.get("success") is not True or row.get("error") not in (None, ""):
        return False, False, "success"
    responses = row.get("decoding")
    if not isinstance(responses, list) or len(responses) != SCENARIOS[scenario]["turns"]:
        return False, False, "responses"
    if any(not isinstance(value, str) or not value.strip() for value in responses):
        return False, False, "responses"
    if row.get("model_name") != EXPECTED_MODEL:
        return False, False, "model"
    verifier = row.get("bird_verifier")
    if not isinstance(verifier, Mapping) or verifier.get("llm_judge") is not False:
        return False, False, "verifier"
    if not isinstance(row.get("correct"), bool):
        return False, False, "correct"
    return True, row["correct"], None


def _browse_row(row: Any, task_id: str, scenario: str) -> tuple[bool, bool, str | None]:
    if not isinstance(row, Mapping):
        return False, False, "not_object"
    if row.get("task_id") != task_id:
        return False, False, "task_id"
    if row.get("success") is not True or row.get("error") not in (None, ""):
        return False, False, "success"
    prediction = row.get("prediction")
    if not isinstance(prediction, str) or not prediction.strip():
        return False, False, "prediction"
    responses = row.get("responses")
    if not isinstance(responses, list) or len(responses) != SCENARIOS[scenario]["turns"]:
        return False, False, "responses"
    if any(not isinstance(value, str) or not value.strip() for value in responses):
        return False, False, "responses"
    if not isinstance(row.get("correct"), bool):
        return False, False, "correct"
    return True, row["correct"], None


def _swe_row(row: Any, task_id: str, scenario: str) -> tuple[bool, bool, str | None]:
    if not isinstance(row, Mapping):
        return False, False, "not_object"
    if row.get("task_id") != task_id:
        return False, False, "task_id"
    if row.get("success") is not True or row.get("error") not in (None, ""):
        return False, False, "success"
    prediction = row.get("prediction")
    if not isinstance(prediction, str) or not prediction.strip():
        return False, False, "prediction"
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return False, False, "scenario"
    if metadata.get("requested_model") != EXPECTED_MODEL or metadata.get(
        "resolved_model"
    ) != EXPECTED_MODEL:
        return False, False, "model"
    if metadata.get("reasoning_effort") != EXPECTED_REASONING_EFFORT:
        return False, False, "reasoning"
    if metadata.get("checkpoint_scenario") != scenario or metadata.get(
        "n_user_turns_delivered"
    ) != SCENARIOS[scenario]["turns"]:
        return False, False, "scenario"
    verifier = row.get("swe_eval")
    if (
        not isinstance(verifier, Mapping)
        or not isinstance(verifier.get("resolved"), bool)
        or verifier.get("harness_error") not in (None, "")
        or verifier.get("patch_extracted") is not True
    ):
        return False, False, "verifier"
    return True, bool(verifier["resolved"]), None


def _scenario_result(
    *,
    path: Path,
    expected_ids: Sequence[str],
    scenario: str,
    validator: Callable[[Any, str, str], tuple[bool, bool, str | None]],
    issues: _Issues,
    label: str,
    root: Path,
) -> tuple[dict[str, Any], set[str], Mapping[str, Any] | None]:
    payload = _read_json(path, issues, label)
    expected_set = set(expected_ids)
    if not isinstance(payload, Mapping):
        if payload is not None:
            issues.add(f"{label}: result artifact is not a task-ID object")
        return (
            {
                "expected": len(expected_ids),
                "present": 0,
                "completed": 0,
                "correct": 0,
                "accuracy": None,
                "artifact": _display_path(path, root),
            },
            set(),
            None,
        )

    extra_count = len(set(payload) - expected_set)
    if extra_count:
        issues.add(f"{label}: result artifact has {extra_count} unexpected task IDs")
    counts: Counter[str] = Counter()
    valid_ids: set[str] = set()
    correct = 0
    present = 0
    for task_id in expected_ids:
        if task_id not in payload:
            continue
        present += 1
        valid, is_correct, reason = validator(payload[task_id], task_id, scenario)
        if valid:
            valid_ids.add(task_id)
            correct += int(is_correct)
        elif reason is not None:
            counts[reason] += 1
    _record_validation_counts(issues, label, counts)
    if len(valid_ids) != len(expected_ids):
        issues.add(
            f"{label}: exact coverage failed ({len(valid_ids)}/{len(expected_ids)})"
        )
    accuracy = correct / len(valid_ids) if valid_ids else None
    return (
        {
            "expected": len(expected_ids),
            "present": present,
            "completed": len(valid_ids),
            "correct": correct,
            "accuracy": accuracy,
            "artifact": _display_path(path, root),
        },
        valid_ids,
        payload,
    )


def _relative_delta(single: Mapping[str, Any], evolve: Mapping[str, Any]) -> dict[str, Any]:
    single_accuracy = single.get("accuracy")
    evolve_accuracy = evolve.get("accuracy")
    if not isinstance(single_accuracy, (int, float)) or not isinstance(
        evolve_accuracy, (int, float)
    ):
        return {"absolute": None, "relative": None}
    absolute = float(evolve_accuracy) - float(single_accuracy)
    relative = absolute / float(single_accuracy) if single_accuracy else None
    return {"absolute": absolute, "relative": relative}


def _browse_checkpoint_policy(
    *,
    result_path: Path,
    aggregate: Mapping[str, Any] | None,
    expected_ids: Sequence[str],
    scenario: str,
    issues: _Issues,
    label: str,
) -> int:
    directory = Path(f"{result_path}.checkpoints")
    if not directory.is_dir():
        issues.add(f"{label}: bound checkpoint directory is missing")
        return 0
    envelopes: dict[str, Mapping[str, Any]] = {}
    malformed = 0
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path, issues, f"{label} checkpoint")
        task_id = payload.get("task_id") if isinstance(payload, Mapping) else None
        if not isinstance(task_id, str) or not task_id or task_id in envelopes:
            malformed += 1
            continue
        envelopes[task_id] = payload
    if malformed:
        issues.add(f"{label}: {malformed} checkpoints are malformed or duplicated")
    extras = set(envelopes) - set(expected_ids)
    if extras:
        issues.add(f"{label}: checkpoint directory has {len(extras)} unexpected task IDs")

    valid = 0
    failure_counts: Counter[str] = Counter()
    expected_scenario_name = "fully_specified" if scenario == "single" else "combined"
    for task_id in expected_ids:
        envelope = envelopes.get(task_id)
        if envelope is None:
            failure_counts["missing"] += 1
            continue
        policy = envelope.get("policy")
        expected_policy = {
            "workflow": "browsecomp-plan-a-evaluation",
            "schema_version": 2,
            "requested_model": EXPECTED_MODEL,
            "resolved_model": EXPECTED_MODEL,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "naturalizer_model": EXPECTED_MODEL,
            "judge_model": EXPECTED_MODEL,
            "max_tokens": None,
            "max_tool_calls": 50,
            "max_search_iterations": 51,
            "force_final_answer": True,
            "no_retrieval_history": False,
            "recap_method": None,
        }
        if envelope.get("schema_version") != 2 or envelope.get("task_id") != task_id:
            failure_counts["envelope"] += 1
            continue
        if not isinstance(envelope.get("input_sha256"), str) or not _SHA256.fullmatch(
            envelope["input_sha256"]
        ):
            failure_counts["binding"] += 1
            continue
        if not isinstance(policy, Mapping) or any(
            policy.get(key, "missing") != value for key, value in expected_policy.items()
        ):
            failure_counts["policy"] += 1
            continue
        scenario_policy = policy.get("scenario")
        expected_scenario = {
            "name": expected_scenario_name,
            "num_turns": SCENARIOS[scenario]["turns"],
            "num_revisions": SCENARIOS[scenario]["revisions"],
            "num_switches": SCENARIOS[scenario]["switches"],
            "ordering": "interleaved",
        }
        if not isinstance(scenario_policy, Mapping) or dict(scenario_policy) != expected_scenario:
            failure_counts["scenario"] += 1
            continue
        if policy.get("selected_task_ids") != list(expected_ids):
            failure_counts["selection"] += 1
            continue
        retriever = policy.get("retriever")
        if (
            not isinstance(retriever, Mapping)
            or retriever.get("k") != 5
            or retriever.get("revision") != BROWSE_RETRIEVER_REVISION
        ):
            failure_counts["retriever"] += 1
            continue
        if aggregate is None or envelope.get("result") != aggregate.get(task_id):
            failure_counts["aggregate"] += 1
            continue
        valid += 1

    checkpoint_messages = {
        "missing": "bound checkpoints are missing",
        "envelope": "checkpoint envelopes are invalid",
        "binding": "checkpoint input bindings are invalid",
        "policy": "checkpoint model/reasoning/limit policies are invalid",
        "scenario": "checkpoint scenario policies are invalid",
        "selection": "checkpoint task selections are invalid",
        "retriever": "checkpoint retriever revision is invalid",
        "aggregate": "checkpoints differ from the aggregate result",
    }
    for key in sorted(failure_counts):
        issues.add(f"{label}: {failure_counts[key]} {checkpoint_messages[key]}")
    if valid != len(expected_ids):
        issues.add(f"{label}: policy-bound checkpoint coverage failed ({valid}/{len(expected_ids)})")
    return valid


def validate_browsecomp_evaluation_artifacts(
    *,
    repo_root: Path,
    task_ids: Sequence[str],
    single_path: Path,
    evolve_path: Path,
) -> dict[str, Any]:
    """Validate a fetched BrowseComp+ pair without returning private rows."""
    expected_count = EXPECTED_COUNTS["browsecomp_plus"]
    if len(task_ids) != expected_count or not all(
        isinstance(task_id, str) and task_id for task_id in task_ids
    ):
        raise FinalizationError("BrowseComp+ task IDs are not the fixed unique 100")
    if len(set(task_ids)) != expected_count:
        raise FinalizationError("BrowseComp+ task IDs are not the fixed unique 100")

    issues = _Issues()
    scenarios: dict[str, dict[str, Any]] = {}
    for scenario, path, label in (
        ("single", single_path, "BrowseComp+ single"),
        ("evolve", evolve_path, "BrowseComp+ evolve"),
    ):
        summary, valid_ids, aggregate = _scenario_result(
            path=path,
            expected_ids=task_ids,
            scenario=scenario,
            validator=_browse_row,
            issues=issues,
            label=label,
            root=repo_root,
        )
        checkpoint_count = _browse_checkpoint_policy(
            result_path=path,
            aggregate=aggregate,
            expected_ids=task_ids,
            scenario=scenario,
            issues=issues,
            label=label,
        )
        scenarios[scenario] = {
            "completed": len(valid_ids),
            "checkpoints": checkpoint_count,
            "correct": summary["correct"],
        }

    if issues.values:
        raise FinalizationError(
            "BrowseComp+ fetched evaluation validation failed: "
            + "; ".join(issues.values)
        )
    return {
        "status": "complete",
        "expected": expected_count,
        "scenarios": scenarios,
    }


def _validate_browse_construction(run_dir: Path, issues: _Issues, root: Path) -> dict[str, Any]:
    candidates = [run_dir / "local_audit.json", run_dir / "remote_audit.json"]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    payload = _read_json(path, issues, "BrowseComp+ construction audit")
    verified = True
    if not isinstance(payload, Mapping):
        verified = False
    else:
        if payload.get("model") != EXPECTED_MODEL:
            issues.add("BrowseComp+ construction audit: model is not kimi-k2.6")
            verified = False
        if payload.get("sample_count") != EXPECTED_COUNTS["browsecomp_plus"]:
            issues.add("BrowseComp+ construction audit: coverage is not 100/100")
            verified = False
        stage_counts = payload.get("stage_counts")
        if not isinstance(stage_counts, Mapping) or any(
            stage_counts.get(stage) != EXPECTED_COUNTS["browsecomp_plus"]
            for stage in ("stage1", "stage2", "stage3")
        ):
            issues.add("BrowseComp+ construction audit: stage coverage is incomplete")
            verified = False
        if payload.get("independence_verification") is not True:
            issues.add("BrowseComp+ construction audit: independence verification is missing")
            verified = False
    return {"verified": verified, "artifact": _display_path(path, root)}


def _validate_bird_construction(
    root: Path,
    expected_ids: Sequence[str],
    issues: _Issues,
) -> dict[str, Any]:
    checkpoints = (
        (
            "extraction",
            root
            / "intent_construction/intent_extraction/output/bird_sql/"
            "extracted_checkpoint.json",
            "bird_extraction",
            False,
        ),
        (
            "counterfactual",
            root
            / "intent_construction/retrospective_expansion/counterfactual/output/"
            "bird_sql/argument_counterfactual_checkpoint.json",
            "bird_counterfactual",
            False,
        ),
        (
            "predecessor",
            root
            / "intent_construction/retrospective_expansion/predecessor/output/"
            "bird_sql/predecessor_checkpoint.json",
            "bird_predecessor",
            True,
        ),
    )
    verified = True
    artifacts: list[str] = []
    predecessor_results: list[Any] | None = None
    for stage, path, expected_stage, require_predecessors in checkpoints:
        artifacts.append(_display_path(path, root))
        payload = _read_json(path, issues, f"BIRD {stage} checkpoint")
        if not isinstance(payload, Mapping):
            verified = False
            continue
        results = payload.get("results")
        expected_envelope = (
            payload.get("schema_version") == 1
            and payload.get("stage") == expected_stage
            and payload.get("model") == EXPECTED_MODEL
            and payload.get("required_task_ids") == list(expected_ids)
            and payload.get("processed_ids") == list(expected_ids)
            and payload.get("complete") is True
            and payload.get("failures") == {}
        )
        if not expected_envelope or not isinstance(results, list) or len(results) != len(
            expected_ids
        ):
            issues.add(f"BIRD {stage} checkpoint: coverage or model policy is incomplete")
            verified = False
            continue
        if require_predecessors:
            predecessor_results = results
        row_ids = [row.get("task_id") if isinstance(row, Mapping) else None for row in results]
        row_models = [
            row.get("model_name") if isinstance(row, Mapping) else None for row in results
        ]
        if row_ids != list(expected_ids) or set(row_models) != {EXPECTED_MODEL}:
            issues.add(f"BIRD {stage} checkpoint: result IDs or models are invalid")
            verified = False
        if require_predecessors:
            for row in results:
                predecessor_info = row.get("predecessor_info") if isinstance(row, Mapping) else None
                predecessors = row.get("predecessor_functions") if isinstance(row, Mapping) else None
                if (
                    not isinstance(predecessor_info, Mapping)
                    or predecessor_info.get("model") != EXPECTED_MODEL
                    or predecessor_info.get("naturalizer_model") != EXPECTED_MODEL
                    or not isinstance(predecessors, list)
                    or len(predecessors) < 2
                ):
                    issues.add(
                        "BIRD predecessor checkpoint: model lock or predecessor coverage is invalid"
                    )
                    verified = False
                    break

    final_path = root / "final_dataset/bird_sql_final.json"
    artifacts.append(_display_path(final_path, root))
    final_rows = _read_json(final_path, issues, "BIRD final construction dataset")
    if not isinstance(final_rows, list) or len(final_rows) != len(expected_ids):
        if final_rows is not None:
            issues.add("BIRD final construction dataset: coverage is incomplete")
        verified = False
    else:
        final_ids = [
            row.get("task_id") if isinstance(row, Mapping) else None for row in final_rows
        ]
        if final_ids != list(expected_ids):
            issues.add("BIRD final construction dataset: task IDs are invalid")
            verified = False
        if predecessor_results is None or final_rows != predecessor_results:
            issues.add(
                "BIRD final construction dataset: content differs from the predecessor checkpoint"
            )
            verified = False
    return {"verified": verified, "artifacts": artifacts}


def _validate_swe_construction(run_dir: Path, issues: _Issues, root: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    payload = _read_json(path, issues, "SWE construction manifest")
    verified = True
    if not isinstance(payload, Mapping):
        verified = False
    else:
        expected = {
            "status": "complete",
            "model": EXPECTED_MODEL,
            "resolved_model": EXPECTED_MODEL,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "output_token_limit": None,
            "cost_hard_cap_usd": None,
            "final_coverage": EXPECTED_COUNTS["swe_bench_verified"],
        }
        for key, value in expected.items():
            if payload.get(key, "missing") != value:
                issues.add(f"SWE construction manifest: {key} does not match the formal run")
                verified = False
    return {"verified": verified, "artifact": _display_path(path, root)}


def _validate_swe_manifest(
    path: Path,
    expected_ids: Sequence[str],
    issues: _Issues,
    root: Path,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    payload = _read_json(path, issues, "SWE evaluation manifest")
    verified = True
    if not isinstance(payload, Mapping):
        verified = False
        payload = None
    else:
        if payload.get("schema_version") != 2 or payload.get("status") != "complete":
            issues.add("SWE evaluation manifest: run is not complete schema v2")
            verified = False
        models = payload.get("models")
        if not isinstance(models, Mapping) or models.get("requested") != [
            EXPECTED_MODEL
        ] or models.get("resolved") != [EXPECTED_MODEL]:
            issues.add("SWE evaluation manifest: model lock is invalid")
            verified = False
        dataset = payload.get("dataset")
        if not isinstance(dataset, Mapping) or dataset.get("expected_count") != len(
            expected_ids
        ) or dataset.get("task_ids") != list(expected_ids):
            issues.add("SWE evaluation manifest: published task selection is invalid")
            verified = False
        runtime = payload.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("reasoning_effort") != EXPECTED_REASONING_EFFORT
            or runtime.get("output_token_limit", "missing") is not None
            or runtime.get("cost_hard_cap_usd", "missing") is not None
        ):
            issues.add("SWE evaluation manifest: reasoning or no-limit policy is invalid")
            verified = False
        expected_scenarios = [
            {
                "name": name,
                "turns": values["turns"],
                "revisions": values["revisions"],
                "switches": values["switches"],
                "tool_call_limit_per_turn": 200,
            }
            for name, values in SCENARIOS.items()
        ]
        if payload.get("scenarios") != expected_scenarios:
            issues.add("SWE evaluation manifest: scenarios do not match the paper settings")
            verified = False
        runs = payload.get("scenario_runs")
        for name in SCENARIOS:
            run = runs.get(name) if isinstance(runs, Mapping) else None
            if (
                not isinstance(run, Mapping)
                or run.get("status") != "complete"
                or run.get("expected") != len(expected_ids)
                or run.get("completed") != len(expected_ids)
            ):
                issues.add(f"SWE evaluation manifest: {name} run is incomplete")
                verified = False
    return (
        {"verified": verified, "artifact": _display_path(path, root)},
        payload,
    )


def _usage_candidates(run_dirs: Iterable[Path], extras: Iterable[Path]) -> list[Path]:
    candidates: list[Path] = []
    for directory in run_dirs:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.jsonl"):
            lowered = path.name.lower()
            if "usage" in lowered or path.parent.name.lower() == "usage":
                candidates.append(path)
    candidates.extend(extras)
    unique: list[Path] = []
    identities: set[tuple[int, str] | str] = set()
    for path in sorted(candidates, key=lambda item: str(item)):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            identity: tuple[int, str] | str = (
                path.stat().st_size,
                digest.hexdigest(),
            )
        except OSError:
            identity = str(path.resolve())
        if identity not in identities:
            identities.add(identity)
            unique.append(path)
    return unique


def _ledger_has_usage(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line, object_pairs_hook=_json_pairs)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(event, Mapping) and event.get("event", "usage") == "usage":
            return True
    return False


def _require_ledger_role(paths: Sequence[Path], issues: _Issues, label: str) -> None:
    if not any(_ledger_has_usage(path) for path in paths):
        issues.add(f"{label}: no confirmed usage ledger was found")


def _usage_summary(
    paths: Sequence[Path],
    issues: _Issues,
    label: str,
    root: Path,
) -> dict[str, Any]:
    totals: dict[str, int] = {field: 0 for field in TOKEN_FIELDS}
    calls = 0
    cost = 0.0
    first: datetime | None = None
    last: datetime | None = None
    requested: set[Any] = set()
    resolved: set[Any] = set()
    invalid_lines = 0
    unpriced = 0
    limit_policy_violations = 0
    for path in paths:
        if not path.is_file():
            issues.add(f"{label}: configured usage ledger is missing")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            issues.add(f"{label}: usage ledger cannot be read")
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line, object_pairs_hook=_json_pairs)
            except (json.JSONDecodeError, ValueError):
                invalid_lines += 1
                continue
            if not isinstance(event, Mapping) or event.get("event", "usage") != "usage":
                continue
            calls += 1
            if event.get("cost_hard_cap_usd") is not None or any(
                event.get(field) is not None for field in EXPECTED_OUTPUT_LIMIT_FIELDS
            ):
                limit_policy_violations += 1
            requested_value = event.get("requested_model")
            resolved_value = event.get("resolved_model")
            if not isinstance(requested_value, str) or not isinstance(
                resolved_value, str
            ):
                invalid_lines += 1
            else:
                requested.add(requested_value)
                resolved.add(resolved_value)
            for field in TOKEN_FIELDS:
                value = event.get(field, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    invalid_lines += 1
                    value = 0
                totals[field] += value
            value = event.get("cost_usd")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(
                float(value)
            ) or float(value) < 0:
                unpriced += 1
            else:
                cost += float(value)
            timestamp = _as_utc(event.get("timestamp"))
            if timestamp is not None:
                first = timestamp if first is None or timestamp < first else first
                last = timestamp if last is None or timestamp > last else last
    if not paths:
        issues.add(f"{label}: no usage ledger was found")
    if calls == 0:
        issues.add(f"{label}: no confirmed usage events were found")
    if invalid_lines:
        issues.add(f"{label}: {invalid_lines} usage records are malformed")
    if unpriced:
        issues.add(f"{label}: {unpriced} usage calls have no confirmed cost")
    if limit_policy_violations:
        issues.add(
            f"{label}: {limit_policy_violations} usage calls record a hard cap or output limit"
        )
    if requested != {EXPECTED_MODEL} or resolved != {EXPECTED_MODEL}:
        issues.add(f"{label}: usage model audit is not uniquely {EXPECTED_MODEL}")
    elapsed = (last - first).total_seconds() if first is not None and last is not None else None
    billable_tokens = totals["input_tokens"] + totals["output_tokens"]
    return {
        "calls": calls,
        **totals,
        "billable_tokens": billable_tokens,
        "confirmed_cost_usd": round(cost, 12),
        "unpriced_calls": unpriced,
        "requested_models": sorted(str(value) for value in requested),
        "resolved_models": sorted(str(value) for value in resolved),
        "ledgers": [_display_path(path, root) for path in paths],
        "first_event_at": _iso(first),
        "last_event_at": _iso(last),
        "observed_elapsed_seconds": elapsed,
    }


def _runtime_from_usage_and_manifest(
    usage: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates_start = [_as_utc(usage.get("first_event_at"))]
    candidates_end = [_as_utc(usage.get("last_event_at"))]
    if isinstance(manifest, Mapping):
        candidates_start.extend(
            _as_utc(manifest.get(key)) for key in ("created_at", "started_at")
        )
        candidates_end.extend(
            _as_utc(manifest.get(key))
            for key in ("updated_at", "completed_at")
        )
    starts = [value for value in candidates_start if value is not None]
    ends = [value for value in candidates_end if value is not None]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    elapsed = (end - start).total_seconds() if start is not None and end is not None else None
    return {
        "started_at": _iso(start),
        "ended_at": _iso(end),
        "elapsed_seconds": elapsed,
        "source": "artifact_and_usage_timestamps",
    }


def _benchmark_result(
    name: str,
    expected_ids: Sequence[str],
    single: dict[str, Any],
    evolve: dict[str, Any],
    single_ids: set[str],
    evolve_ids: set[str],
    usage: dict[str, Any],
    runtime: dict[str, Any],
    issue_start: int,
    issues: _Issues,
    **extra: Any,
) -> dict[str, Any]:
    paired = len(single_ids & evolve_ids)
    if paired != len(expected_ids):
        issues.add(f"{name}: paired coverage failed ({paired}/{len(expected_ids)})")
    result = {
        "status": "complete" if len(issues.values) == issue_start else "incomplete",
        "coverage": {
            "expected": len(expected_ids),
            "single_completed": len(single_ids),
            "evolve_completed": len(evolve_ids),
            "paired_completed": paired,
        },
        "single": single,
        "evolve": evolve,
        "delta": _relative_delta(single, evolve),
        "usage": usage,
        "runtime": runtime,
        **extra,
    }
    if len(issues.values) != issue_start:
        result["status"] = "incomplete"
    return result


def _bird_report(layout: ReportLayout, task_ids: Sequence[str], issues: _Issues) -> dict[str, Any]:
    issue_start = len(issues.values)
    dataset = "bird_sql_n100"
    single_path = layout.experiments_dir / "fully_specified" / dataset / f"{EXPECTED_MODEL}.json"
    evolve_path = (
        layout.experiments_dir
        / "combined_independent"
        / dataset
        / f"{EXPECTED_MODEL}_t7_g2_p2.json"
    )
    single, single_ids, _ = _scenario_result(
        path=single_path,
        expected_ids=task_ids,
        scenario="single",
        validator=_bird_row,
        issues=issues,
        label="BIRD single",
        root=layout.repo_root,
    )
    evolve, evolve_ids, _ = _scenario_result(
        path=evolve_path,
        expected_ids=task_ids,
        scenario="evolve",
        validator=_bird_row,
        issues=issues,
        label="BIRD evolve",
        root=layout.repo_root,
    )
    construction = _validate_bird_construction(layout.repo_root, task_ids, issues)
    extra_ledgers = list(layout.extra_ledgers.get("bird_sql", ()))
    _require_ledger_role(
        [layout.bird_run_dir / "usage.jsonl"], issues, "BIRD construction usage"
    )
    _require_ledger_role(
        [layout.bird_run_dir / "evaluation_usage.jsonl", *extra_ledgers],
        issues,
        "BIRD evaluation usage",
    )
    paths = _usage_candidates([layout.bird_run_dir], extra_ledgers)
    usage = _usage_summary(paths, issues, "BIRD usage", layout.repo_root)
    return _benchmark_result(
        "BIRD-SQL",
        task_ids,
        single,
        evolve,
        single_ids,
        evolve_ids,
        usage,
        _runtime_from_usage_and_manifest(usage),
        issue_start,
        issues,
        construction=construction,
    )


def _browse_report(layout: ReportLayout, task_ids: Sequence[str], issues: _Issues) -> dict[str, Any]:
    issue_start = len(issues.values)
    dataset = "browsecomp_plus_n100"
    single_path = (
        layout.experiments_dir
        / "fully_specified"
        / dataset
        / f"{EXPECTED_MODEL}_naturalized_reasoning-medium_force-final.json"
    )
    evolve_path = (
        layout.experiments_dir
        / "combined_independent"
        / dataset
        / f"{EXPECTED_MODEL}_t7_g2_p2_naturalized_reasoning-medium_force-final.json"
    )
    single, single_ids, single_payload = _scenario_result(
        path=single_path,
        expected_ids=task_ids,
        scenario="single",
        validator=_browse_row,
        issues=issues,
        label="BrowseComp+ single",
        root=layout.repo_root,
    )
    evolve, evolve_ids, evolve_payload = _scenario_result(
        path=evolve_path,
        expected_ids=task_ids,
        scenario="evolve",
        validator=_browse_row,
        issues=issues,
        label="BrowseComp+ evolve",
        root=layout.repo_root,
    )
    checkpoint_coverage = {
        "single": _browse_checkpoint_policy(
            result_path=single_path,
            aggregate=single_payload,
            expected_ids=task_ids,
            scenario="single",
            issues=issues,
            label="BrowseComp+ single",
        ),
        "evolve": _browse_checkpoint_policy(
            result_path=evolve_path,
            aggregate=evolve_payload,
            expected_ids=task_ids,
            scenario="evolve",
            issues=issues,
            label="BrowseComp+ evolve",
        ),
    }
    construction = _validate_browse_construction(
        layout.browse_run_dir, issues, layout.repo_root
    )
    extra_ledgers = list(layout.extra_ledgers.get("browsecomp_plus", ()))
    _require_ledger_role(
        [layout.browse_run_dir / "llm_usage.jsonl"],
        issues,
        "BrowseComp+ construction usage",
    )
    _require_ledger_role(
        [
            layout.browse_run_dir / "evaluation_usage.jsonl",
            layout.browse_run_dir / "usage.jsonl",
            *extra_ledgers,
        ],
        issues,
        "BrowseComp+ evaluation usage",
    )
    paths = _usage_candidates([layout.browse_run_dir], extra_ledgers)
    usage = _usage_summary(paths, issues, "BrowseComp+ usage", layout.repo_root)
    return _benchmark_result(
        "BrowseComp+",
        task_ids,
        single,
        evolve,
        single_ids,
        evolve_ids,
        usage,
        _runtime_from_usage_and_manifest(usage),
        issue_start,
        issues,
        checkpoint_policy_coverage=checkpoint_coverage,
        construction=construction,
    )


def _swe_report(layout: ReportLayout, task_ids: Sequence[str], issues: _Issues) -> dict[str, Any]:
    issue_start = len(issues.values)
    manifest_summary, manifest = _validate_swe_manifest(
        layout.swe_run_dir / "manifest.json", task_ids, issues, layout.repo_root
    )
    construction = _validate_swe_construction(
        layout.swe_construction_run_dir, issues, layout.repo_root
    )
    single, single_ids, _ = _scenario_result(
        path=layout.swe_run_dir / "results" / "single.json",
        expected_ids=task_ids,
        scenario="single",
        validator=_swe_row,
        issues=issues,
        label="SWE single",
        root=layout.repo_root,
    )
    evolve, evolve_ids, _ = _scenario_result(
        path=layout.swe_run_dir / "results" / "evolve.json",
        expected_ids=task_ids,
        scenario="evolve",
        validator=_swe_row,
        issues=issues,
        label="SWE evolve",
        root=layout.repo_root,
    )
    extra_ledgers = list(layout.extra_ledgers.get("swe_bench_verified", ()))
    _require_ledger_role(
        [layout.swe_construction_run_dir / "usage.jsonl"],
        issues,
        "SWE construction usage",
    )
    _require_ledger_role(
        [layout.swe_run_dir / "usage.jsonl", *extra_ledgers],
        issues,
        "SWE evaluation usage",
    )
    paths = _usage_candidates(
        [layout.swe_construction_run_dir, layout.swe_run_dir], extra_ledgers
    )
    usage = _usage_summary(paths, issues, "SWE usage", layout.repo_root)
    return _benchmark_result(
        "SWE-bench Verified",
        task_ids,
        single,
        evolve,
        single_ids,
        evolve_ids,
        usage,
        _runtime_from_usage_and_manifest(usage, manifest),
        issue_start,
        issues,
        evaluation_manifest=manifest_summary,
        construction=construction,
    )


def _sum_usage(benchmarks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("calls", *TOKEN_FIELDS, "billable_tokens", "unpriced_calls")
    totals = {
        field: sum(int(benchmark["usage"].get(field, 0)) for benchmark in benchmarks.values())
        for field in fields
    }
    totals["confirmed_cost_usd"] = round(
        sum(float(benchmark["usage"].get("confirmed_cost_usd", 0.0)) for benchmark in benchmarks.values()),
        12,
    )
    return totals


def _overall_runtime(benchmarks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    starts = [
        _as_utc(benchmark["runtime"].get("started_at"))
        for benchmark in benchmarks.values()
    ]
    ends = [
        _as_utc(benchmark["runtime"].get("ended_at"))
        for benchmark in benchmarks.values()
    ]
    valid_starts = [value for value in starts if value is not None]
    valid_ends = [value for value in ends if value is not None]
    start = min(valid_starts) if valid_starts else None
    end = max(valid_ends) if valid_ends else None
    return {
        "started_at": _iso(start),
        "ended_at": _iso(end),
        "elapsed_seconds": (
            (end - start).total_seconds() if start is not None and end is not None else None
        ),
    }


def build_report(
    layout: ReportLayout,
    *,
    allow_incomplete: bool = False,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Read, validate, and aggregate formal artifacts without modifying them."""
    issues = _Issues()
    config = _read_json(layout.config_path, issues, "run policy")
    policy_verified = _validate_global_policy(config, issues)
    indices = layout.repo_root / "intent_construction" / "eval_indices"
    bird_ids = _load_task_ids(
        indices / "bird_sql_task_ids.json",
        EXPECTED_COUNTS["bird_sql"],
        issues,
        "BIRD published IDs",
    )
    browse_ids = _load_task_ids(
        indices / "browsecomp_plus_task_ids.json",
        EXPECTED_COUNTS["browsecomp_plus"],
        issues,
        "BrowseComp+ published IDs",
    )
    swe_ids = _load_task_ids(
        indices / "swe_bench_verified_task_ids.json",
        EXPECTED_COUNTS["swe_bench_verified"],
        issues,
        "SWE published IDs",
    )

    benchmarks = {
        "bird_sql": _bird_report(layout, bird_ids, issues),
        "browsecomp_plus": _browse_report(layout, browse_ids, issues),
        "swe_bench_verified": _swe_report(layout, swe_ids, issues),
    }
    completed = not issues.values and all(
        benchmark["status"] == "complete" for benchmark in benchmarks.values()
    )
    report = {
        "schema_version": 1,
        "status": "complete" if completed else "incomplete",
        "generated_at": _iso(
            (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        ),
        "policy": {
            "verified": policy_verified,
            "model": EXPECTED_MODEL,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "hard_cap_usd": None,
            "output_token_limit": None,
        },
        "coverage": {
            name: {
                "completed": benchmark["coverage"]["paired_completed"],
                "expected": benchmark["coverage"]["expected"],
            }
            for name, benchmark in benchmarks.items()
        },
        "benchmarks": benchmarks,
        "totals": {
            "usage": _sum_usage(benchmarks),
            "runtime": _overall_runtime(benchmarks),
        },
        "issues": list(issues.values),
        "redaction": {
            "browsecomp_plaintext_query_exported": False,
            "browsecomp_gold_docs_exported": False,
            "credential_material_exported": False,
        },
    }
    if not allow_incomplete and not completed:
        preview = "; ".join(issues.values[:12])
        if len(issues.values) > 12:
            preview += f"; plus {len(issues.values) - 12} more issue(s)"
        raise FinalizationError(f"formal finalization failed closed: {preview}")
    return report


def _percent(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.1%}"


def _number(value: Any) -> str:
    return "-" if value is None else f"{int(value):,}"


def _duration(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def render_html(report: Mapping[str, Any]) -> str:
    """Render only aggregate, escaped values; raw examples are never accepted."""
    labels = {
        "bird_sql": "BIRD-SQL",
        "browsecomp_plus": "BrowseComp+",
        "swe_bench_verified": "SWE-bench Verified",
    }
    rows: list[str] = []
    for name, benchmark in report["benchmarks"].items():
        coverage = benchmark["coverage"]
        usage = benchmark["usage"]
        delta = benchmark["delta"]
        status = "完成" if benchmark["status"] == "complete" else "未完成"
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels[name])}</td>"
            f"<td>{coverage['paired_completed']}/{coverage['expected']}</td>"
            f"<td>{_percent(benchmark['single']['accuracy'])} "
            f"({benchmark['single']['correct']}/{benchmark['single']['completed']})</td>"
            f"<td>{_percent(benchmark['evolve']['accuracy'])} "
            f"({benchmark['evolve']['correct']}/{benchmark['evolve']['completed']})</td>"
            f"<td>{_percent(delta['relative'])}</td>"
            f"<td>US${usage['confirmed_cost_usd']:.4f}</td>"
            f"<td>{_number(usage['input_tokens'])} / {_number(usage['output_tokens'])}</td>"
            f"<td>{_duration(benchmark['runtime']['elapsed_seconds'])}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    issue_items = "".join(
        f"<li>{html.escape(str(issue))}</li>" for issue in report.get("issues", [])
    ) or "<li>无</li>"
    totals = report["totals"]["usage"]
    complete = report["status"] == "complete"
    status_text = "复现完成" if complete else "进度预览：尚未完成"
    status_class = "ok" if complete else "pending"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>剩余实验复现报告</title>
  <style>
    :root {{ color-scheme: light; --ink:#182026; --muted:#5d6972; --line:#d9dee2;
      --paper:#ffffff; --wash:#f4f6f7; --ok:#176b45; --warn:#9a5b00; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--wash); color:var(--ink); font-family:-apple-system,
      BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.5; }}
    main {{ width:min(1180px, calc(100% - 32px)); margin:28px auto 48px; background:var(--paper);
      border:1px solid var(--line); padding:28px; }}
    h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:28px 0 10px; font-size:18px; letter-spacing:0; }}
    p {{ margin:6px 0; }} .muted {{ color:var(--muted); }}
    .status {{ display:inline-block; margin-top:14px; padding:5px 10px; border:1px solid currentColor;
      font-weight:650; }} .status.ok {{ color:var(--ok); }} .status.pending {{ color:var(--warn); }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); }}
    table {{ width:100%; border-collapse:collapse; min-width:940px; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ background:#eef1f2; font-size:13px; }} tr:last-child td {{ border-bottom:0; }}
    dl {{ display:grid; grid-template-columns:220px 1fr; margin:0; border-top:1px solid var(--line); }}
    dt,dd {{ margin:0; padding:9px 0; border-bottom:1px solid var(--line); }} dt {{ color:var(--muted); }}
    ul {{ margin:8px 0 0; padding-left:22px; }} code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    @media (max-width:700px) {{ main {{ width:100%; margin:0; border:0; padding:20px 16px; }}
      dl {{ grid-template-columns:1fr; }} dt {{ border-bottom:0; padding-bottom:0; }} dd {{ padding-top:2px; }} }}
  </style>
</head>
<body><main>
  <h1>剩余实验复现报告</h1>
  <p class="muted">生成时间：{html.escape(str(report['generated_at']))}</p>
  <div class="status {status_class}">{status_text}</div>

  <h2>实验结果</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>数据集</th><th>配对覆盖</th><th>Single 准确率</th><th>Evolve 准确率</th>
      <th>相对变化</th><th>确认费用</th><th>输入 / 输出 tokens</th><th>观测运行时间</th><th>状态</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>

  <h2>执行约束</h2>
  <dl>
    <dt>唯一模型</dt><dd><code>{html.escape(str(report['policy']['model']))}</code></dd>
    <dt>Reasoning</dt><dd><code>{html.escape(str(report['policy']['reasoning_effort']))}</code></dd>
    <dt>费用硬上限</dt><dd>无</dd>
    <dt>输出 token 上限</dt><dd>无</dd>
    <dt>确认总费用</dt><dd>US${totals['confirmed_cost_usd']:.6f}</dd>
    <dt>调用数</dt><dd>{_number(totals['calls'])}</dd>
    <dt>总输入 tokens</dt><dd>{_number(totals['input_tokens'])}</dd>
    <dt>总输出 tokens</dt><dd>{_number(totals['output_tokens'])}</dd>
    <dt>总 reasoning tokens</dt><dd>{_number(totals['reasoning_tokens'])}</dd>
    <dt>整体时间窗</dt><dd>{_duration(report['totals']['runtime']['elapsed_seconds'])}</dd>
  </dl>

  <h2>待处理项</h2>
  <ul>{issue_items}</ul>

  <h2>脱敏状态</h2>
  <p>报告只含聚合指标。BrowseComp+ 明文问题、gold docs 和凭据均未导出。</p>
</main></body></html>
"""


def write_report(layout: ReportLayout, report: Mapping[str, Any]) -> None:
    outputs = (layout.output_json.resolve(), layout.output_html.resolve())
    if outputs[0] == outputs[1]:
        raise FinalizationError("JSON and HTML report paths must be different")
    protected_directories = (
        layout.experiments_dir,
        layout.bird_run_dir,
        layout.browse_run_dir,
        layout.swe_construction_run_dir,
        layout.swe_run_dir,
    )
    for output in outputs:
        if any(output.is_relative_to(directory.resolve()) for directory in protected_directories):
            raise FinalizationError(
                "report outputs must be outside experiment and run directories"
            )
        if output == layout.config_path.resolve() or any(
            output == ledger.resolve()
            for ledgers in layout.extra_ledgers.values()
            for ledger in ledgers
        ):
            raise FinalizationError("report output overlaps a protected input artifact")
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(layout.output_json, serialized)
    _atomic_write_text(layout.output_html, render_html(report))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--swe-run-dir",
        type=Path,
        default=None,
        help="Downloaded SWE run directory containing manifest.json and results/.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a redacted status preview instead of failing on partial coverage.",
    )
    parser.add_argument("--bird-ledger", action="append", type=Path, default=[])
    parser.add_argument("--browse-ledger", action="append", type=Path, default=[])
    parser.add_argument("--swe-ledger", action="append", type=Path, default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = ReportLayout.defaults(
        args.repo_root,
        swe_run_dir=args.swe_run_dir,
        output_json=args.output_json,
        output_html=args.output_html,
        extra_ledgers={
            "bird_sql": args.bird_ledger,
            "browsecomp_plus": args.browse_ledger,
            "swe_bench_verified": args.swe_ledger,
        },
    )
    try:
        report = build_report(layout, allow_incomplete=args.allow_incomplete)
        write_report(layout, report)
    except (FinalizationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"finalization status={report['status']} "
        f"json={layout.output_json} html={layout.output_html}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
