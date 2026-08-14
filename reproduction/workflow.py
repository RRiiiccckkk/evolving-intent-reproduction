"""Pure helpers for the Plan A reproduction workflow.

This module deliberately uses only the Python standard library. Live dataset
and model dependencies are imported by :mod:`reproduction.plan_a` only when an
online stage is executed, so validation and CI remain lightweight and offline.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "plan_a.json"
PLAN_A_SETTING_NAMES = (
    "single_t1",
    "evolve_t4_g1_p1",
    "repeat_control_t7",
    "evolve_t7_g2_p2",
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorkflowError(RuntimeError):
    """Raised when a workflow invariant is not satisfied."""


class DeadlineReached(WorkflowError):
    """Raised before starting more paid work after the cutoff buffer."""


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, target)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = read_json(config_path)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise WorkflowError("plan config schema_version must be 1")
    if config.get("dataset") != "gsm8k" or config.get("split") != "test":
        raise WorkflowError("Plan A is fixed to the GSM8K test split")

    primary = config.get("sample_count")
    fallback = config.get("fallback_sample_count")
    if not isinstance(primary, int) or not isinstance(fallback, int):
        raise WorkflowError("sample counts must be integers")
    if not (0 < fallback <= primary):
        raise WorkflowError("fallback_sample_count must be in [1, sample_count]")

    settings = config.get("evaluation", {}).get("settings", [])
    names = tuple(setting.get("name") for setting in settings)
    if names != PLAN_A_SETTING_NAMES:
        raise WorkflowError(
            "Plan A settings must be ordered as " + ", ".join(PLAN_A_SETTING_NAMES)
        )

    expected = {
        "single_t1": (1, 0, 0, 0),
        "evolve_t4_g1_p1": (4, 1, 1, 1),
        "evolve_t7_g2_p2": (7, 2, 2, 2),
    }
    for setting in settings:
        name = setting["name"]
        if setting.get("kind") == "standard":
            turns = int(setting["turns"])
            switches = int(setting["switches"])
            revisions = int(setting["revisions"])
            reveals = turns - 1 - switches - revisions
            if (turns, switches, revisions, reveals) != expected[name]:
                raise WorkflowError(f"incorrect transition composition for {name}")
        elif name == "repeat_control_t7":
            base = setting.get("base", {})
            if (
                int(setting.get("turns", 0)) != 7
                or int(setting.get("repeat_turns", 0)) != 3
                or (base.get("turns"), base.get("switches"), base.get("revisions"))
                != (4, 1, 1)
            ):
                raise WorkflowError("repeat control must append three repeats to t4/g1/p1")
        else:
            raise WorkflowError(f"unknown setting kind for {name}")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def select_official_samples(
    eval_manifest_path: str | Path,
    count: int,
) -> list[dict[str, Any]]:
    """Select the first ``count`` entries from the published eval manifest."""
    manifest = read_json(eval_manifest_path)
    samples = manifest.get("samples") if isinstance(manifest, dict) else None
    if not isinstance(samples, list):
        raise WorkflowError("official eval manifest has no samples list")
    if count < 1 or count > len(samples):
        raise WorkflowError(
            f"requested {count} samples, official manifest contains {len(samples)}"
        )

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_original: set[int] = set()
    for position, item in enumerate(samples[:count], start=1):
        task_id = item.get("task_id")
        original_id = item.get("original_id")
        if not isinstance(task_id, str) or not isinstance(original_id, int):
            raise WorkflowError("official eval entry is missing task_id/original_id")
        if task_id in seen_ids or original_id in seen_original:
            raise WorkflowError("official eval manifest contains duplicate IDs")
        seen_ids.add(task_id)
        seen_original.add(original_id)
        selected.append(
            {
                "position": position,
                "task_id": task_id,
                "original_id": original_id,
            }
        )
    return selected


def choose_sample_count(
    config: Mapping[str, Any],
    requested: int | None,
    *,
    auto_fallback: bool,
    now: datetime | None = None,
    deadline: datetime | None = None,
) -> tuple[int, str]:
    if requested is not None:
        allowed = {config["sample_count"], config["fallback_sample_count"]}
        if requested not in allowed:
            raise WorkflowError(f"sample_count must be one of {sorted(allowed)}")
        return requested, "explicit"

    primary = int(config["sample_count"])
    if not auto_fallback or deadline is None:
        return primary, "primary"

    current = now or datetime.now(timezone.utc).astimezone()
    if current.tzinfo is None or deadline.tzinfo is None:
        raise WorkflowError("deadline and current time must include a timezone")
    minutes_left = (deadline - current).total_seconds() / 60
    threshold = int(config["deadline"]["fallback_threshold_minutes"])
    if minutes_left < threshold:
        return int(config["fallback_sample_count"]), "automatic_deadline_fallback"
    return primary, "primary"


def parse_deadline(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise WorkflowError("deadline must include an explicit UTC offset")
    return parsed


def enforce_deadline(
    deadline: datetime | None,
    *,
    buffer_minutes: int = 0,
    ignore: bool = False,
    now: datetime | None = None,
) -> None:
    if ignore or deadline is None:
        return
    current = now or datetime.now(timezone.utc).astimezone()
    seconds_left = (deadline - current).total_seconds()
    if seconds_left <= buffer_minutes * 60:
        raise DeadlineReached(
            f"deadline guard reached ({seconds_left / 60:.1f} minutes left; "
            f"buffer={buffer_minutes} minutes)"
        )


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    config: Mapping[str, Any],
    *,
    run_id: str,
    sample_count: int,
    selection_reason: str,
    construction_model: str,
    evaluation_model: str,
    deadline: datetime | None,
    dry_run: bool,
) -> dict[str, Any]:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise WorkflowError("run_id may contain only letters, digits, '.', '_' and '-'")
    eval_manifest = resolve_repo_path(config["selection"]["eval_manifest"])
    selected = select_official_samples(eval_manifest, sample_count)
    settings = []
    for raw in config["evaluation"]["settings"]:
        setting = dict(raw)
        if setting["kind"] == "standard":
            setting["reveals"] = (
                int(setting["turns"])
                - 1
                - int(setting["switches"])
                - int(setting["revisions"])
            )
        else:
            base = setting["base"]
            setting["base_reveals"] = (
                int(base["turns"])
                - 1
                - int(base["switches"])
                - int(base["revisions"])
            )
        settings.append(setting)

    return {
        "schema_version": 1,
        "plan": "A",
        "run_id": run_id,
        "status": "dry_run" if dry_run else "initialized",
        "created_at": iso_now(),
        "repository": {
            "path": str(REPO_ROOT),
            "commit": _git_output("rev-parse", "HEAD"),
            "upstream_commit": config.get("upstream_commit"),
        },
        "dataset": {
            "name": "gsm8k",
            "split": "test",
            "selection_source": str(eval_manifest.relative_to(REPO_ROOT)),
            "selection_source_sha256": file_sha256(eval_manifest),
            "selection_rule": "first N entries in the published eval manifest",
            "requested_count": sample_count,
            "selection_reason": selection_reason,
            "selected_samples": selected,
            "eligible_task_ids": [],
        },
        "seed": int(config["seed"]),
        "models": {
            "construction": construction_model,
            "evaluation": evaluation_model,
        },
        "settings": settings,
        "deadline": deadline.isoformat() if deadline else None,
        "budget": dict(config["budget"]),
        "artifacts": {},
        "coverage": {},
    }


def selected_task_ids(manifest: Mapping[str, Any]) -> list[str]:
    return [item["task_id"] for item in manifest["dataset"]["selected_samples"]]


def selected_original_ids(manifest: Mapping[str, Any]) -> list[int]:
    return [int(item["original_id"]) for item in manifest["dataset"]["selected_samples"]]


def load_list(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if not isinstance(payload, list) or not all(isinstance(x, dict) for x in payload):
        raise WorkflowError(f"expected a list of objects in {path}")
    return payload


def canonicalize_stage_file(path: str | Path, ordered_ids: Sequence[str]) -> None:
    target = Path(path)
    if not target.exists():
        return
    rows = load_list(target)
    by_id = {row.get("task_id"): row for row in rows if row.get("task_id")}
    atomic_write_json(target, [by_id[task_id] for task_id in ordered_ids if task_id in by_id])


def inspect_stage(
    path: str | Path,
    selected_ids: Sequence[str],
    stage: str,
    *,
    required_revisions: int = 2,
    required_switches: int = 2,
) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {
            "stage": stage,
            "path": str(target),
            "exists": False,
            "count": 0,
            "missing_task_ids": list(selected_ids),
            "invalid_task_ids": [],
            "complete": False,
        }

    rows = load_list(target)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        if task_id in by_id:
            duplicates.append(task_id)
        by_id[task_id] = row

    missing = [task_id for task_id in selected_ids if task_id not in by_id]
    invalid: list[str] = []
    for task_id in selected_ids:
        row = by_id.get(task_id)
        if row is None:
            continue
        arguments = row.get("arguments")
        if not row.get("function") or not isinstance(arguments, list) or not arguments:
            invalid.append(task_id)
            continue
        if stage in {"stage2", "stage3"}:
            counterfactual_count = sum(
                len(argument.get("counterfactual_arguments", []))
                for argument in arguments
                if isinstance(argument, dict)
            )
            if counterfactual_count < required_revisions:
                invalid.append(task_id)
                continue
        if stage == "stage3":
            predecessors = row.get("predecessor_functions")
            if not isinstance(predecessors, list) or len(predecessors) < required_switches:
                invalid.append(task_id)

    complete = not missing and not invalid and not duplicates
    return {
        "stage": stage,
        "path": str(target),
        "exists": True,
        "sha256": file_sha256(target),
        "count": len(by_id),
        "missing_task_ids": missing,
        "invalid_task_ids": invalid,
        "duplicate_task_ids": sorted(set(duplicates)),
        "complete": complete,
    }


def repeat_last_user_turn(sample: Any, repeats: int) -> Any:
    """Append exact copies of the last user turn without changing intent state."""
    if repeats < 0:
        raise WorkflowError("repeats cannot be negative")
    user_turns = [turn for turn in sample.turns if turn.get("role") == "user"]
    if not user_turns:
        raise WorkflowError("cannot repeat a sample with no user turn")
    last = dict(user_turns[-1])
    sample.turns.extend(dict(last) for _ in range(repeats))
    if hasattr(sample, "structured_contents"):
        sample.structured_contents = None
    if hasattr(sample, "recap_texts"):
        sample.recap_texts = None
    metadata = dict(getattr(sample, "metadata", {}) or {})
    base_turns = len(user_turns)
    change_plan = copy.deepcopy(metadata.get("change_plan"))
    if isinstance(change_plan, dict):
        trajectory = change_plan.get("intent_trajectory")
        transitions = change_plan.get("transitions")
        if isinstance(trajectory, list) and trajectory:
            final_state = copy.deepcopy(trajectory[-1])
            trajectory.extend(copy.deepcopy(final_state) for _ in range(repeats))
        if isinstance(transitions, list):
            transitions.extend(
                {
                    "type": "repeat",
                    "revealed_ids": None,
                    "changed_arguments": None,
                    "old_function": None,
                    "new_function": None,
                }
                for _ in range(repeats)
            )
        change_plan["num_turns"] = base_turns + repeats
        metadata["change_plan"] = change_plan
    metadata["num_turns"] = base_turns + repeats
    metadata["requested_num_turns"] = base_turns + repeats
    metadata["repeat_control"] = {
        "base_user_turns": base_turns,
        "repeated_user_turns": repeats,
        "policy": "exact_copy_of_previous_final_user_turn",
        "state_policy": "copy_final_intent_state_without_change",
    }
    sample.metadata = metadata
    return sample


def validate_setting_sample(
    sample: Any,
    setting: Mapping[str, Any],
    *,
    base_sample: Any | None = None,
) -> None:
    """Reject silently degraded or semantically invalid scheduler output."""
    name = str(setting["name"])
    expected_turns = int(setting["turns"])
    user_turns = [turn for turn in sample.turns if turn.get("role") == "user"]
    metadata = getattr(sample, "metadata", {}) or {}
    plan = metadata.get("change_plan") or {}
    trajectory = plan.get("intent_trajectory") or []
    transitions = plan.get("transitions") or []

    if len(user_turns) != expected_turns:
        raise WorkflowError(
            f"{name} produced {len(user_turns)} user turns for {sample.task_id}; "
            f"expected {expected_turns}"
        )
    expected_switches = int(
        setting.get("switches", setting.get("base", {}).get("switches", 0))
    )
    expected_revisions = int(
        setting.get("revisions", setting.get("base", {}).get("revisions", 0))
    )
    if int(metadata.get("num_predecessor_functions", -1)) != expected_switches:
        raise WorkflowError(f"{name} scheduler reduced the requested function switches")
    if int(metadata.get("num_counterfactual_arguments", -1)) != expected_revisions:
        raise WorkflowError(f"{name} scheduler reduced the requested argument revisions")
    if len(trajectory) != expected_turns or len(transitions) != expected_turns - 1:
        raise WorkflowError(f"{name} change plan does not match its user-turn count")

    final = trajectory[-1] if trajectory else {}
    if (
        final.get("function") != metadata.get("source_function")
        or final.get("is_fully_specified") is not True
        or final.get("has_counterfactual_values") is not False
    ):
        raise WorkflowError(f"{name} does not end at the fully specified source intent")

    if setting["kind"] == "standard":
        expected_counts = {
            "argument_reveal": expected_turns - 1 - expected_switches - expected_revisions,
            "function_change": expected_switches,
            "argument_change": expected_revisions,
        }
        observed = {
            transition_type: sum(
                transition.get("type") == transition_type for transition in transitions
            )
            for transition_type in expected_counts
        }
        if observed != expected_counts:
            raise WorkflowError(
                f"{name} transition counts are {observed}; expected {expected_counts}"
            )
        if any(
            not transition.get("revealed_ids")
            for transition in transitions
            if transition.get("type") == "argument_reveal"
        ):
            raise WorkflowError(f"{name} contains an empty argument reveal")
        return

    if base_sample is None:
        raise WorkflowError("repeat control validation requires its frozen t4 base sample")
    base_users = [turn for turn in base_sample.turns if turn.get("role") == "user"]
    base_plan = (getattr(base_sample, "metadata", {}) or {}).get("change_plan") or {}
    base_trajectory = base_plan.get("intent_trajectory") or []
    base_transitions = base_plan.get("transitions") or []
    repeats = int(setting["repeat_turns"])
    if user_turns[:-repeats] != base_users or any(
        turn != base_users[-1] for turn in user_turns[-repeats:]
    ):
        raise WorkflowError("repeat control is not the exact frozen t4 user trajectory")
    if trajectory[:-repeats] != base_trajectory or transitions[:-repeats] != base_transitions:
        raise WorkflowError("repeat control changed its frozen t4 intent trajectory")
    if any(state != base_trajectory[-1] for state in trajectory[-repeats:]):
        raise WorkflowError("repeat control did not copy the final t4 intent state")
    for transition in transitions[-repeats:]:
        if transition.get("type") != "repeat" or any(
            transition.get(field) not in (None, [])
            for field in ("revealed_ids", "changed_arguments", "old_function", "new_function")
        ):
            raise WorkflowError("repeat control contains a state-changing repeat")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total are inconsistent")
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z2 / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def unwrap_results(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        payload = payload["results"]
    if not isinstance(payload, dict):
        raise WorkflowError("result artifact must contain an object keyed by task_id")
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def append_ledger(path: str | Path, event: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": iso_now(), **dict(event)}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkflowError(f"invalid ledger line {line_number}: {exc}") from exc
            if isinstance(value, dict):
                events.append(value)
    return events


def summarize_ledger(path: str | Path, budget: Mapping[str, Any]) -> dict[str, Any]:
    events = read_ledger(path)
    usage = [event for event in events if event.get("event") == "usage"]
    if any(event.get("cost_usd") is None for event in usage):
        raise WorkflowError("usage ledger contains an unpriced response")
    return {
        "ledger_path": str(path),
        "events": len(events),
        "requests": len(usage),
        "actual_usd_recorded": round(
            sum(float(event.get("cost_usd") or 0.0) for event in usage), 6
        ),
        "input_tokens": sum(int(event.get("input_tokens") or 0) for event in usage),
        "output_tokens": sum(int(event.get("output_tokens") or 0) for event in usage),
        "cached_tokens": sum(int(event.get("cached_tokens") or 0) for event in usage),
        "reasoning_tokens": sum(int(event.get("reasoning_tokens") or 0) for event in usage),
        "hard_cap_usd": budget.get("hard_cap_usd"),
        "planned_range_usd": budget.get("planned_range_usd"),
        "measurement_note": (
            "Provider-reported usage is priced with LLM_PRICE_MAP and recorded "
            "without a local spend cutoff."
        ),
    }


def aggregate_results(
    manifest: Mapping[str, Any],
    result_payloads: Mapping[str, Any],
    *,
    ledger_path: str | Path,
) -> dict[str, Any]:
    selected_ids = selected_task_ids(manifest)
    eligible = manifest.get("dataset", {}).get("eligible_task_ids", [])
    expected_ids = list(eligible) if eligible else selected_ids
    settings: dict[str, Any] = {}
    unwrapped: dict[str, dict[str, dict[str, Any]]] = {}

    for name in PLAN_A_SETTING_NAMES:
        results = unwrap_results(result_payloads.get(name, {}))
        unwrapped[name] = results

    successful_by_setting = {
        name: {
            task_id
            for task_id in expected_ids
            if task_id in results
            and bool(results[task_id].get("success", True))
            and results[task_id].get("error") is None
        }
        for name, results in unwrapped.items()
    }
    common_success = [
        task_id
        for task_id in expected_ids
        if all(task_id in successful_by_setting[name] for name in PLAN_A_SETTING_NAMES)
    ]

    for name in PLAN_A_SETTING_NAMES:
        results = unwrapped[name]
        raw_completed = [task_id for task_id in expected_ids if task_id in results]
        correct = sum(bool(results[task_id].get("correct", False)) for task_id in common_success)
        total = len(common_success)
        low, high = wilson_interval(correct, total)
        settings[name] = {
            "completed": total,
            "requested": len(expected_ids),
            "coverage": total / len(expected_ids) if expected_ids else 0.0,
            "raw_completed": len(raw_completed),
            "raw_successful": len(successful_by_setting[name]),
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "wilson_95": {"low": low, "high": high},
            "failed_calls": len(raw_completed) - len(successful_by_setting[name]),
        }

    baseline = unwrapped["single_t1"]
    paired: dict[str, Any] = {}
    for name in PLAN_A_SETTING_NAMES[1:]:
        comparison = unwrapped[name]
        common = common_success
        improved = sum(
            not bool(baseline[task_id].get("correct", False))
            and bool(comparison[task_id].get("correct", False))
            for task_id in common
        )
        regressed = sum(
            bool(baseline[task_id].get("correct", False))
            and not bool(comparison[task_id].get("correct", False))
            for task_id in common
        )
        paired[name] = {
            "baseline": "single_t1",
            "paired_samples": len(common),
            "accuracy_difference": (
                (improved - regressed) / len(common) if common else 0.0
            ),
            "percentage_point_difference": (
                100 * (improved - regressed) / len(common) if common else 0.0
            ),
            "improved": improved,
            "regressed": regressed,
            "unchanged": len(common) - improved - regressed,
        }

    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "generated_at": iso_now(),
        "sample_selection": {
            "requested": len(expected_ids),
            "originally_selected": len(selected_ids),
            "task_ids": expected_ids,
            "common_successful_task_ids": common_success,
        },
        "settings": settings,
        "paired_differences": paired,
        "cost": summarize_ledger(ledger_path, manifest.get("budget", {})),
        "interpretation": (
            "This small run tests directional behavior only. Wilson intervals are "
            "wide at N=20 (or N=10 fallback), so it cannot establish paper-scale effect sizes."
        ),
    }


def write_summary_csv(summary: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "setting",
                "completed",
                "requested",
                "correct",
                "accuracy",
                "wilson_95_low",
                "wilson_95_high",
                "paired_pp_vs_single",
            ]
        )
        for name in PLAN_A_SETTING_NAMES:
            metrics = summary["settings"][name]
            paired = summary["paired_differences"].get(name, {})
            writer.writerow(
                [
                    name,
                    metrics["completed"],
                    metrics["requested"],
                    metrics["correct"],
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['wilson_95']['low']:.6f}",
                    f"{metrics['wilson_95']['high']:.6f}",
                    f"{paired.get('percentage_point_difference', 0.0):.3f}",
                ]
            )


def write_summary_html(summary: Mapping[str, Any], path: str | Path) -> None:
    rows = []
    for name in PLAN_A_SETTING_NAMES:
        metrics = summary["settings"][name]
        paired = summary["paired_differences"].get(name)
        difference = "baseline" if paired is None else f"{paired['percentage_point_difference']:+.1f} pp"
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics['correct']}/{metrics['completed']}</td>"
            f"<td>{100 * metrics['accuracy']:.1f}%</td>"
            f"<td>{100 * metrics['wilson_95']['low']:.1f}% to "
            f"{100 * metrics['wilson_95']['high']:.1f}%</td>"
            f"<td>{difference}</td>"
            "</tr>"
        )
    cost = summary["cost"]
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Plan A reproduction summary</title>
  <style>
    body {{ font: 16px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 960px; padding: 0 1rem; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #cbd2d9; padding: .65rem; text-align: left; }}
    th {{ background: #f2f4f5; }}
    code {{ background: #f2f4f5; padding: .1rem .25rem; }}
  </style>
</head>
<body>
  <h1>Plan A reproduction summary</h1>
  <p>Run <code>{html.escape(str(summary['run_id']))}</code>. Small-sample directional check.</p>
  <table>
    <thead><tr><th>Setting</th><th>Correct</th><th>Accuracy</th><th>Wilson 95% CI</th><th>Paired vs single</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Cost ledger</h2>
  <p>Recorded provider usage: ${cost['actual_usd_recorded']:.4f}; requests: {cost['requests']}; input/output tokens: {cost['input_tokens']}/{cost['output_tokens']}.</p>
  <p>{html.escape(summary['interpretation'])}</p>
</body>
</html>
"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")


def contains_secret_material(value: Any) -> bool:
    """Conservative check used by offline validation and tests."""
    serialized = json.dumps(value, ensure_ascii=False).lower()
    patterns = ("sk-", "api_key=", "api-key=", "bearer ")
    return any(pattern in serialized for pattern in patterns)


def iter_result_files(run_dir: str | Path) -> Iterable[tuple[str, Path]]:
    base = Path(run_dir) / "results"
    for name in PLAN_A_SETTING_NAMES:
        yield name, base / f"{name}.json"
