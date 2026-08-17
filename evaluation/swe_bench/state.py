"""Fail-closed state and configuration checks for the SWE reproduction.

This module is standard-library only so all guards can be tested offline,
without importing mini-swe-agent, Modal, SWE-bench, or an LLM client.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_MODEL = "kimi-k2.6"
EXPECTED_REASONING_EFFORT = "medium"
PUBLISHED_TASK_COUNT = 50
TOOL_CALL_LIMIT_PER_TURN = 200
# Zero keeps per-turn model-call accounting enabled without imposing a model-step cap.
MODEL_STEP_LIMIT_PER_TURN = 0
CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: int
    revisions: int
    switches: int
    tool_call_limit_per_turn: int = TOOL_CALL_LIMIT_PER_TURN


SCENARIOS = (
    Scenario("single", turns=1, revisions=0, switches=0),
    Scenario("evolve", turns=7, revisions=2, switches=2),
)
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}

_OUTPUT_LIMIT_FIELDS = frozenset(
    {"max_tokens", "max_completion_tokens", "max_output_tokens"}
)
_FORBIDDEN_LIMIT_ENV = (
    "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
    "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
    "LLM_COST_HARD_CAP_USD",
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class HardeningError(RuntimeError):
    """Raised before incomplete or policy-violating work can be accepted."""


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file and atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: str | Path, *, label: str = "JSON file") -> Any:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise HardeningError(f"cannot read {label} {target}: {exc}") from exc
    if not text.strip():
        raise HardeningError(f"{label} is empty: {target}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HardeningError(f"{label} is invalid JSON: {target}: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_requested_models(models: Sequence[str]) -> None:
    if list(models) != [EXPECTED_MODEL]:
        raise HardeningError(
            f"the hardened SWE run requires exactly one requested model: {EXPECTED_MODEL}"
        )


def _load_string_list(payload: Any, *, key: str, label: str) -> list[str]:
    values = payload.get(key) if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise HardeningError(f"{label} must contain a non-empty string list at {key!r}")
    return list(values)


def validate_exact_ids(
    actual: Iterable[str],
    expected: Sequence[str],
    *,
    label: str,
    require_order: bool = False,
) -> list[str]:
    actual_list = list(actual)
    duplicates = sorted(
        {item for item in actual_list if actual_list.count(item) > 1}
    )
    if duplicates:
        raise HardeningError(f"{label} contains duplicate IDs: {duplicates[:5]}")
    expected_list = list(expected)
    missing = sorted(set(expected_list) - set(actual_list))
    extra = sorted(set(actual_list) - set(expected_list))
    if missing or extra or len(actual_list) != len(expected_list):
        raise HardeningError(
            f"{label} does not exactly cover the published IDs "
            f"(expected={len(expected_list)}, actual={len(actual_list)}, "
            f"missing={missing[:5]}, extra={extra[:5]})"
        )
    if require_order and actual_list != expected_list:
        raise HardeningError(f"{label} is not in published order")
    return actual_list


def load_published_task_ids(
    eval_manifest_path: str | Path,
    task_ids_path: str | Path,
) -> list[str]:
    manifest = read_json(eval_manifest_path, label="published SWE eval manifest")
    if not isinstance(manifest, Mapping):
        raise HardeningError("published SWE eval manifest must be a JSON object")
    if manifest.get("num_samples") != PUBLISHED_TASK_COUNT:
        raise HardeningError(
            f"published SWE eval manifest must declare {PUBLISHED_TASK_COUNT} samples"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != PUBLISHED_TASK_COUNT:
        raise HardeningError(
            f"published SWE eval manifest must contain exactly {PUBLISHED_TASK_COUNT} samples"
        )
    manifest_ids: list[str] = []
    original_ids: list[str] = []
    for position, sample in enumerate(samples, start=1):
        if not isinstance(sample, Mapping):
            raise HardeningError(f"published sample {position} must be an object")
        task_id = sample.get("task_id")
        original_id = sample.get("original_id")
        if not isinstance(task_id, str) or not task_id:
            raise HardeningError(f"published sample {position} has no task_id")
        if not isinstance(original_id, str) or not original_id:
            raise HardeningError(f"published sample {position} has no original_id")
        manifest_ids.append(task_id)
        original_ids.append(original_id)
    validate_exact_ids(
        manifest_ids,
        manifest_ids,
        label="published eval manifest task IDs",
    )
    if len(set(original_ids)) != PUBLISHED_TASK_COUNT:
        raise HardeningError("published eval manifest contains duplicate original IDs")

    task_payload = read_json(task_ids_path, label="published SWE task-ID file")
    task_ids = _load_string_list(
        task_payload,
        key="task_ids",
        label="published SWE task-ID file",
    )
    if len(task_ids) != PUBLISHED_TASK_COUNT:
        raise HardeningError(
            f"published SWE task-ID file must contain exactly {PUBLISHED_TASK_COUNT} IDs"
        )
    validate_exact_ids(
        task_ids,
        manifest_ids,
        label="published SWE task-ID file",
        require_order=True,
    )
    return task_ids


def load_published_id_map(eval_manifest_path: str | Path) -> dict[str, str]:
    manifest = read_json(eval_manifest_path, label="published SWE eval manifest")
    samples = manifest.get("samples") if isinstance(manifest, Mapping) else None
    if not isinstance(samples, list):
        raise HardeningError("published SWE eval manifest has no samples list")
    mapping: dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise HardeningError("published SWE samples must be objects")
        task_id = sample.get("task_id")
        original_id = sample.get("original_id")
        if not isinstance(task_id, str) or not isinstance(original_id, str):
            raise HardeningError("published SWE sample is missing an ID")
        if task_id in mapping:
            raise HardeningError("published SWE eval manifest contains duplicate task IDs")
        mapping[task_id] = original_id
    return mapping


def validate_data_coverage(
    data_path: str | Path,
    expected_ids: Sequence[str],
    *,
    expected_original_ids: Mapping[str, str] | None = None,
) -> None:
    payload = read_json(data_path, label="SWE evaluation data")
    if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
        raise HardeningError("SWE evaluation data must be a list of objects")
    seen: dict[str, int] = {}
    for item in payload:
        task_id = item.get("task_id")
        if isinstance(task_id, str):
            seen[task_id] = seen.get(task_id, 0) + 1
    duplicate_expected = sorted(
        task_id for task_id in expected_ids if seen.get(task_id, 0) > 1
    )
    missing = sorted(task_id for task_id in expected_ids if seen.get(task_id, 0) == 0)
    if missing or duplicate_expected:
        raise HardeningError(
            "SWE evaluation data cannot provide exact published coverage "
            f"(missing={missing[:5]}, duplicate={duplicate_expected[:5]})"
        )
    if expected_original_ids is not None:
        rows = {item.get("task_id"): item for item in payload if item.get("task_id") in expected_ids}
        mismatched = [
            task_id
            for task_id in expected_ids
            if rows[task_id].get("original_id") != expected_original_ids.get(task_id)
        ]
        if mismatched:
            raise HardeningError(
                "SWE evaluation data task/original ID mapping differs from the published set: "
                f"{mismatched[:5]}"
            )


def _parse_json_object(raw: str, *, setting: str, allow_empty: bool = False) -> dict[str, Any]:
    if not raw.strip():
        if allow_empty:
            return {}
        raise HardeningError(f"{setting} is required")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HardeningError(f"{setting} must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise HardeningError(f"{setting} must be a JSON object")
    return payload


def _validate_prices(price_map: Mapping[str, Any], requested: str, resolved: str) -> None:
    raw_prices = None
    for key in (requested, resolved, "*", "default"):
        if key in price_map:
            raw_prices = price_map[key]
            break
    if not isinstance(raw_prices, Mapping):
        raise HardeningError(
            "LLM_PRICE_MAP must include prices for the requested or resolved Kimi model"
        )
    for key in ("input", "output"):
        try:
            value = float(raw_prices[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise HardeningError(f"LLM_PRICE_MAP requires numeric {key!r} price") from exc
        if not math.isfinite(value) or value < 0:
            raise HardeningError(f"LLM_PRICE_MAP {key!r} price must be non-negative")


def validate_runtime_environment(environment: Mapping[str, str]) -> str:
    backend = environment.get("LLM_BACKEND", "").strip().lower()
    if backend not in {"compatible", "openai-compatible", "openai_compatible", "generic"}:
        raise HardeningError("the hardened SWE run requires LLM_BACKEND=compatible")
    if not environment.get("LLM_API_KEY", "").strip() or not environment.get(
        "LLM_BASE_URL", ""
    ).strip():
        raise HardeningError(
            "compatible Kimi access requires LLM_API_KEY and LLM_BASE_URL in the environment"
        )
    if environment.get("LLM_DISABLE_OUTPUT_LIMITS", "").strip().lower() not in _TRUE_VALUES:
        raise HardeningError("LLM_DISABLE_OUTPUT_LIMITS=1 is required")
    if environment.get("LLM_LOCKED_MODEL", "").strip() != EXPECTED_MODEL:
        raise HardeningError(f"LLM_LOCKED_MODEL={EXPECTED_MODEL} is required")
    if (
        environment.get("LLM_REASONING_EFFORT", "").strip()
        != EXPECTED_REASONING_EFFORT
    ):
        raise HardeningError(
            f"LLM_REASONING_EFFORT={EXPECTED_REASONING_EFFORT} is required"
        )
    forbidden = [
        name for name in _FORBIDDEN_LIMIT_ENV if environment.get(name, "").strip()
    ]
    if forbidden:
        raise HardeningError(
            "output limits and cost gates are forbidden for this run: " + ", ".join(forbidden)
        )

    model_map = _parse_json_object(
        environment.get("LLM_MODEL_MAP", ""),
        setting="LLM_MODEL_MAP",
        allow_empty=True,
    )
    resolved = model_map.get(EXPECTED_MODEL, EXPECTED_MODEL)
    if not isinstance(resolved, str) or not resolved.strip():
        raise HardeningError(f"LLM_MODEL_MAP[{EXPECTED_MODEL!r}] must be a non-empty string")
    if resolved != EXPECTED_MODEL:
        raise HardeningError(
            f"LLM_MODEL_MAP must resolve {EXPECTED_MODEL!r} to itself, got {resolved!r}"
        )
    price_map = _parse_json_object(
        environment.get("LLM_PRICE_MAP", ""),
        setting="LLM_PRICE_MAP",
    )
    _validate_prices(price_map, EXPECTED_MODEL, resolved)
    return resolved


def assert_no_output_limit_fields(payload: Any, *, location: str = "configuration") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in _OUTPUT_LIMIT_FIELDS:
                raise HardeningError(f"{location} contains forbidden output limit field {key!r}")
            assert_no_output_limit_fields(value, location=f"{location}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_no_output_limit_fields(value, location=f"{location}[{index}]")


def assert_kimi_tool_policy(model_config: Mapping[str, Any]) -> None:
    """Require provider-auto tool choice and disable parallel tool calls."""
    model_kwargs = model_config.get("model_kwargs", {})
    if not isinstance(model_kwargs, Mapping):
        raise HardeningError("Kimi model_kwargs must be an object")
    if model_kwargs.get("tool_choice") is not None:
        raise HardeningError(
            "Kimi thinking mode forbids sending an explicit tool_choice"
        )
    if model_kwargs.get("parallel_tool_calls") is not False:
        raise HardeningError("Kimi strict mode requires parallel_tool_calls=false")


class ToolCallCounter:
    """Count actual environment calls and reset the count at user-turn boundaries."""

    def __init__(self, limit: int = TOOL_CALL_LIMIT_PER_TURN) -> None:
        if limit != TOOL_CALL_LIMIT_PER_TURN:
            raise HardeningError(
                f"tool-call limit must be exactly {TOOL_CALL_LIMIT_PER_TURN}"
            )
        self.limit = limit
        self.current = 0
        self.completed_turns: list[int] = []

    def consume(self) -> int:
        if self.current >= self.limit:
            raise HardeningError(
                f"per-turn tool-call limit exceeded ({self.limit})"
            )
        self.current += 1
        return self.current

    def finish_turn(self) -> int:
        count = self.current
        self.completed_turns.append(count)
        self.current = 0
        return count


def _validate_result(
    result: Any,
    *,
    task_id: str,
    scenario: Scenario,
    requested_model: str,
    resolved_model: str,
) -> dict[str, Any]:
    if not isinstance(result, dict) or not result:
        raise HardeningError(f"checkpoint result is empty for {task_id}")
    if result.get("task_id") != task_id:
        raise HardeningError(f"checkpoint result task_id mismatch for {task_id}")
    if result.get("success") is not True or result.get("error") not in (None, ""):
        raise HardeningError(f"checkpoint contains a failed result for {task_id}")
    prediction = result.get("prediction")
    if not isinstance(prediction, str) or not prediction.strip():
        raise HardeningError(f"checkpoint has no patch for {task_id}")
    swe_eval = result.get("swe_eval")
    if not isinstance(swe_eval, dict) or not swe_eval:
        raise HardeningError(f"checkpoint has no SWE evaluation for {task_id}")
    if not isinstance(swe_eval.get("resolved"), bool):
        raise HardeningError(f"checkpoint has no resolved boolean for {task_id}")
    if swe_eval.get("harness_error") not in (None, ""):
        raise HardeningError(f"checkpoint has a harness error for {task_id}")
    if swe_eval.get("patch_extracted") is not True:
        raise HardeningError(f"checkpoint patch was not extracted for {task_id}")

    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise HardeningError(f"checkpoint has no metadata for {task_id}")
    expected_metadata = {
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "reasoning_effort": EXPECTED_REASONING_EFFORT,
        "checkpoint_scenario": scenario.name,
        "tool_call_limit_per_turn": scenario.tool_call_limit_per_turn,
        "n_user_turns_delivered": scenario.turns,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise HardeningError(
                f"checkpoint metadata {key!r} mismatch for {task_id}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    per_turn = metadata.get("per_turn_tool_calls")
    if not isinstance(per_turn, list) or len(per_turn) != scenario.turns:
        raise HardeningError(f"checkpoint has incomplete per-turn tool counts for {task_id}")
    if not all(
        isinstance(count, int) and 0 <= count <= scenario.tool_call_limit_per_turn
        for count in per_turn
    ):
        raise HardeningError(f"checkpoint has invalid per-turn tool counts for {task_id}")
    return result


class TaskCheckpointStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        scenario: Scenario,
        requested_model: str,
        resolved_model: str,
    ) -> None:
        self.directory = Path(directory)
        self.scenario = scenario
        self.requested_model = requested_model
        self.resolved_model = resolved_model

    def path_for(self, task_id: str) -> Path:
        if not _SAFE_TASK_ID.fullmatch(task_id):
            raise HardeningError(f"unsafe task ID for checkpoint path: {task_id!r}")
        return self.directory / f"{task_id}.json"

    def exists(self, task_id: str) -> bool:
        return self.path_for(task_id).exists()

    def load(self, task_id: str) -> dict[str, Any]:
        path = self.path_for(task_id)
        envelope = read_json(path, label="task checkpoint")
        if not isinstance(envelope, dict) or not envelope:
            raise HardeningError(f"task checkpoint is empty: {path}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "task_id": task_id,
            "scenario": self.scenario.name,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "tool_call_limit_per_turn": self.scenario.tool_call_limit_per_turn,
        }
        for key, value in expected.items():
            if envelope.get(key) != value:
                raise HardeningError(
                    f"task checkpoint {key!r} mismatch at {path}: "
                    f"expected {value!r}, got {envelope.get(key)!r}"
                )
        return _validate_result(
            envelope.get("result"),
            task_id=task_id,
            scenario=self.scenario,
            requested_model=self.requested_model,
            resolved_model=self.resolved_model,
        )

    def write(self, task_id: str, result: dict[str, Any]) -> Path:
        validated = _validate_result(
            result,
            task_id=task_id,
            scenario=self.scenario,
            requested_model=self.requested_model,
            resolved_model=self.resolved_model,
        )
        path = self.path_for(task_id)
        atomic_write_json(
            path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "task_id": task_id,
                "scenario": self.scenario.name,
                "requested_model": self.requested_model,
                "resolved_model": self.resolved_model,
                "reasoning_effort": EXPECTED_REASONING_EFFORT,
                "tool_call_limit_per_turn": self.scenario.tool_call_limit_per_turn,
                "result": validated,
            },
        )
        return path


def validate_aggregate_results(
    results: Any,
    expected_ids: Sequence[str],
    *,
    store: TaskCheckpointStore,
) -> dict[str, Any]:
    if not isinstance(results, dict) or not results:
        raise HardeningError("aggregate result is empty")
    validate_exact_ids(results.keys(), expected_ids, label="aggregate results")
    for task_id in expected_ids:
        _validate_result(
            results[task_id],
            task_id=task_id,
            scenario=store.scenario,
            requested_model=store.requested_model,
            resolved_model=store.resolved_model,
        )
    return results


def ledger_offset(path: str | Path) -> int:
    target = Path(path)
    return target.stat().st_size if target.exists() else 0


def read_usage_events(path: str | Path, *, start_offset: int = 0) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    try:
        with target.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            if start_offset < 0 or start_offset > size:
                raise HardeningError(
                    f"usage ledger start offset {start_offset} is outside {target}"
                )
            handle.seek(start_offset)
            raw = handle.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HardeningError(f"cannot read usage ledger {target}: {exc}") from exc
    events: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HardeningError(
                f"usage ledger contains invalid JSON after offset {start_offset}, line {index}"
            ) from exc
        if not isinstance(event, dict):
            raise HardeningError("usage ledger events must be JSON objects")
        if event.get("event", "usage") == "usage":
            events.append(event)
    return events


def validate_usage_events(
    events: Sequence[Mapping[str, Any]],
    *,
    requested_model: str,
    resolved_model: str,
) -> dict[str, Any]:
    requested = {event.get("requested_model") for event in events}
    resolved = {event.get("resolved_model") for event in events}
    if requested - {requested_model}:
        raise HardeningError(
            f"usage ledger contains unexpected requested models: {sorted(requested, key=str)}"
        )
    if resolved - {resolved_model}:
        raise HardeningError(
            f"usage ledger contains unexpected resolved models: {sorted(resolved, key=str)}"
        )
    token_fields = ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")
    totals: dict[str, Any] = {"calls": len(events)}
    for field in token_fields:
        total = 0
        for event in events:
            value = event.get(field, 0)
            if not isinstance(value, int) or value < 0:
                raise HardeningError(f"usage ledger contains invalid {field}")
            total += value
        totals[field] = total
    costs = [event.get("cost_usd") for event in events]
    totals["cost_usd"] = round(
        sum(float(cost) for cost in costs if cost is not None), 12
    )
    totals["requested_models"] = sorted(requested, key=str)
    totals["resolved_models"] = sorted(resolved, key=str)
    return totals
