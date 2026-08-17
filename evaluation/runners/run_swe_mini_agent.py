#!/usr/bin/env python3
"""
Mini-SWE-agent v2 baseline runner for SWE-bench Verified.

Runs each sample as a single user-turn conversation through mini-swe-agent v2
(text-based action parsing, shared-client-routed model). The agent's final patch is
verified against the official SWE-bench harness via our existing
``swe_harness.py``. With ``--num_turns 1`` this is the single-shot SWE-bench
baseline, but with a real coding agent under the hood instead of one LLM
completion; higher ``--num_turns`` enables the multi-turn simulation scenarios.

Usage
-----

    # Single-turn baseline on 50 stratified samples
    python evaluation/runners/run_swe_mini_agent.py \
        --data_path intent_construction/retrospective_expansion/predecessor/output/swe_bench_verified/paired_g1_n251.json \
        --models gpt-5.1 \
        --task_ids_file intent_construction/eval_indices/swe_bench_verified_task_ids.json \
        --num_workers 4 \
        --step_limit 60

Multi-turn SWE scenarios (under-specified / argument-revision / function-switch /
combined) are driven by *this same runner* via ``--num_turns``, ``--num_switches``
and ``--num_revisions``. With those at their defaults the runner produces the
*single-turn* paraphrase / raw baselines that demand the higher-fidelity scaffold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent

from intent_construction.intent_extraction.core.llm_utils import (  # noqa: E402
    clean_model_name,
    resolve_model_name,
)
from evaluation.common.swe_harness import SWEHarness  # noqa: E402
from evaluation.common.swe_minisweagent_scaffold import (  # noqa: E402
    MiniAgentResult,
    run_evolvingintent_with_mini_agent,
)
from situated_simulation.user_simulation import EvolvingIntent  # noqa: E402
from evaluation.swe_bench.state import (  # noqa: E402
    EXPECTED_REASONING_EFFORT,
    MODEL_STEP_LIMIT_PER_TURN,
    SCENARIO_BY_NAME,
    TOOL_CALL_LIMIT_PER_TURN,
    HardeningError,
    TaskCheckpointStore,
    assert_no_output_limit_fields,
    atomic_write_json,
    validate_aggregate_results,
    validate_exact_ids,
    validate_requested_models,
    read_json,
    ledger_offset,
    load_published_id_map,
    load_published_task_ids,
    read_usage_events,
    validate_runtime_environment,
    validate_usage_events,
)

EXPERIMENTS_DIR = _REPO / "evaluation" / "experiments"
SWEREX_COMMAND_TIMEOUT_SECONDS = 30 * 60
SWEREX_RUNTIME_TIMEOUT_SECONDS = 24 * 60 * 60
PUBLISHED_EVAL_MANIFEST = (
    _REPO / "intent_construction" / "eval_indices" / "swe_bench_verified_eval_ids.json"
)
PUBLISHED_TASK_IDS = (
    _REPO / "intent_construction" / "eval_indices" / "swe_bench_verified_task_ids.json"
)


_VALID_SOURCES = ("raw", "paraphrase")


def _load_reusable_mini_result(
    trajectory_path: str | Path,
    *,
    task_id: str,
    instance_id: str,
    model: str,
    user_turns: list[str],
    reasoning_effort: str | None,
    step_limit_per_turn: int | None,
    tool_call_limit_per_turn: int | None,
    use_tool_calling: bool,
) -> tuple[MiniAgentResult, str] | None:
    """Recover a completed agent patch when only the verifier needs retrying."""
    path = Path(trajectory_path)
    if not path.is_file() or path.stem != task_id:
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("trajectory_format") != "mini-swe-agent-1.1":
        return None

    info = payload.get("info")
    messages = payload.get("messages")
    if not isinstance(info, dict) or not isinstance(messages, list):
        return None
    submission = info.get("submission")
    if (
        info.get("exit_status") != "Submitted"
        or not isinstance(submission, str)
        or not submission.strip().startswith("diff --git ")
    ):
        return None

    config = info.get("config")
    if not isinstance(config, dict):
        return None
    try:
        assert_no_output_limit_fields(config, location="reusable SWE trajectory")
    except HardeningError:
        return None
    model_config = config.get("model")
    model_kwargs = model_config.get("model_kwargs") if isinstance(model_config, dict) else None
    if (
        not isinstance(model_config, dict)
        or model_config.get("model_name") != model
        or not isinstance(model_kwargs, dict)
        or model_kwargs.get("reasoning_effort") != reasoning_effort
    ):
        return None
    expected_model_type = ".LLMToolModel" if use_tool_calling else ".LLMTextbasedModel"
    if not str(config.get("model_type", "")).endswith(expected_model_type):
        return None

    agent_config = config.get("agent")
    if not isinstance(agent_config, dict):
        return None
    configured_output = agent_config.get("output_path")
    if not isinstance(configured_output, str):
        return None
    try:
        if Path(configured_output).resolve() != path.resolve():
            return None
    except OSError:
        return None
    if agent_config.get("step_limit") != step_limit_per_turn:
        return None

    environment = config.get("environment")
    image = environment.get("image") if isinstance(environment, dict) else ""
    instance_marker = instance_id.lower().replace("__", "_1776_")
    if instance_marker not in str(image).lower():
        return None

    trajectory_users = [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(trajectory_users) != len(user_turns) or any(
        turn not in rendered for turn, rendered in zip(user_turns, trajectory_users)
    ):
        return None

    exit_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "exit"
    ]
    exit_extra = exit_messages[-1].get("extra") if exit_messages else None
    if (
        not isinstance(exit_extra, dict)
        or exit_extra.get("exit_status") != "Submitted"
        or exit_extra.get("submission") != submission
    ):
        return None

    tool_budget = config.get("tool_budget")
    completed_tool_calls = (
        tool_budget.get("completed_turns") if isinstance(tool_budget, dict) else None
    )
    if (
        not isinstance(tool_budget, dict)
        or tool_budget.get("limit_per_turn") != tool_call_limit_per_turn
        or not isinstance(completed_tool_calls, list)
        or len(completed_tool_calls) != len(user_turns)
        or not all(
            isinstance(value, int)
            and value >= 0
            and (
                tool_call_limit_per_turn is None
                or value <= tool_call_limit_per_turn
            )
            for value in completed_tool_calls
        )
    ):
        return None

    per_turn_steps: list[int] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if role == "user":
            per_turn_steps.append(0)
        elif role == "assistant" and per_turn_steps:
            per_turn_steps[-1] += 1
    if len(per_turn_steps) != len(user_turns):
        return None

    model_stats = info.get("model_stats")
    raw_cost = model_stats.get("instance_cost", 0.0) if isinstance(model_stats, dict) else 0.0
    try:
        cost = float(raw_cost or 0.0)
    except (TypeError, ValueError):
        return None
    return (
        MiniAgentResult(
            instance_id=instance_id,
            submission=submission,
            exit_status="Submitted",
            n_steps=sum(per_turn_steps),
            n_user_turns_delivered=len(user_turns),
            cost=cost,
            error=None,
            final_messages=list(messages),
            scaffold_mode="native" if use_tool_calling else "textbased",
            step_limit_per_turn=step_limit_per_turn,
            per_turn_steps=per_turn_steps,
            tool_call_limit_per_turn=tool_call_limit_per_turn,
            per_turn_tool_calls=list(completed_tool_calls),
        ),
        hashlib.sha256(raw).hexdigest(),
    )


def _question_from_sample(sample: dict[str, Any], source: str) -> str:
    if source == "raw":
        return sample.get("question") or ""
    if source == "paraphrase":
        return sample.get("fully_specified_question") or sample.get("question") or ""
    raise ValueError(f"unknown source: {source!r}; expected one of {_VALID_SOURCES}")


# Map EvolvingIntent `_infer_scenario` outputs to mini-agent output subdirectories.
# fully-specified keeps the legacy raw/paraphrase naming for back-compat.
_SCENARIO_TO_SUBDIR: dict[str, str] = {
    "under-specified": "swe_under_specified_mini_agent",
    "argument-revision": "swe_argument_revision_mini_agent",
    "function-switch": "swe_function_switch_mini_agent",
    "combined": "swe_combined_independent_mini_agent",
}


def _experiments_subdir(source: str, scenario: str = "fully-specified") -> str:
    if scenario == "fully-specified":
        return {
            "raw": "swe_original_mini_agent",
            "paraphrase": "swe_paraphrased_mini_agent",
        }[source]
    if scenario in _SCENARIO_TO_SUBDIR:
        return _SCENARIO_TO_SUBDIR[scenario]
    raise ValueError(f"unknown scenario: {scenario!r}")


def _sample_to_dict(gs) -> dict[str, Any]:
    """Adapter: flatten a situated_simulation.IntentSample into the dict shape evaluate_one expects.

    Pre-builds ``user_turns`` from the simulator's user-role messages so the mini-agent
    receives the multi-turn script verbatim. The original raw fields (question,
    fully_specified_question) are NOT carried through — for multi-turn mode the
    ``_user_turns`` key is the source of truth and ``--source`` is ignored.
    """
    md = gs.metadata or {}
    return {
        "task_id": gs.task_id,
        "original_id": md.get("original_id"),
        "swe_bench_metadata": md.get("swe_bench_metadata"),
        # ``_user_turns`` is the new key: when present, evaluate_one uses it
        # directly as the multi-turn script. When absent, evaluate_one falls
        # back to the legacy single-turn ``_question_from_sample(source)``.
        "_user_turns": [t["content"] for t in gs.turns if t.get("role") == "user"],
        # Carry simulation scenario so output dir routing can read it back.
        "_scenario": md.get("scenario", "fully-specified"),
    }


def evaluate_one(
    sample: dict[str, Any],
    model: str,
    source: str,
    harness: SWEHarness,
    *,
    cost_limit: float,
    step_limit: int,
    step_limit_per_turn: int | None = None,
    use_tool_calling: bool = False,
    reasoning_effort: str | None = None,
    tool_call_limit_per_turn: int | None = None,
    environment_class: str = "docker",
    checkpoint_scenario: str | None = None,
    resolved_model: str | None = None,
    trajectory_path: str | Path | None = None,
) -> dict[str, Any]:
    instance_id = sample.get("original_id") or ""
    task_id = sample.get("task_id") or instance_id

    # Multi-turn path: pre-built simulator user_turns list is the source of truth.
    # Single-turn path: build [question] from --source as before.
    pre_built_turns = sample.get("_user_turns")
    if pre_built_turns:
        user_turns = list(pre_built_turns)
        first_turn = user_turns[0] if user_turns else ""
    else:
        first_turn = _question_from_sample(sample, source)
        user_turns = [first_turn]

    if not instance_id or not first_turn:
        return {
            "task_id": task_id, "prediction": None, "correct": False,
            "ground_truth": "", "decoding": [], "user_messages": list(user_turns),
            "success": False, "error": "missing_instance_id_or_question",
            "metadata": {"task_id": task_id, "original_id": instance_id, "source": source},
            "swe_eval": None,
        }

    instance = {"instance_id": instance_id}

    # Inject reasoning_effort into the mini config via overrides → model_kwargs.
    overrides: dict[str, Any] = {}
    if reasoning_effort:
        overrides["model"] = {"model_kwargs": {"reasoning_effort": reasoning_effort}}
    if tool_call_limit_per_turn is not None:
        overrides.setdefault("model", {}).setdefault("model_kwargs", {})[
            "parallel_tool_calls"
        ] = False
    if environment_class:
        overrides["environment"] = {"environment_class": environment_class}
        if environment_class == "swerex_modal":
            overrides["environment"].update(
                {
                    "timeout": SWEREX_COMMAND_TIMEOUT_SECONDS,
                    "startup_timeout": 600.0,
                    "runtime_timeout": float(SWEREX_RUNTIME_TIMEOUT_SECONDS),
                    "deployment_timeout": float(SWEREX_RUNTIME_TIMEOUT_SECONDS),
                    "install_pipx": True,
                    "modal_sandbox_kwargs": {},
                }
            )
    assert_no_output_limit_fields(overrides, location="runner overrides")

    error: str | None = None
    success = True
    mini_result: MiniAgentResult | None = None
    reused_trajectory_sha256: str | None = None
    reusable = None
    if trajectory_path is not None and checkpoint_scenario in SCENARIO_BY_NAME:
        reusable = _load_reusable_mini_result(
            trajectory_path,
            task_id=str(task_id),
            instance_id=str(instance_id),
            model=model,
            user_turns=user_turns,
            reasoning_effort=reasoning_effort,
            step_limit_per_turn=step_limit_per_turn,
            tool_call_limit_per_turn=tool_call_limit_per_turn,
            use_tool_calling=use_tool_calling,
        )
    if reusable is not None:
        mini_result, reused_trajectory_sha256 = reusable
        print(f"Reusing completed agent trajectory for verifier retry: {task_id}")
    else:
        try:
            mini_result = run_evolvingintent_with_mini_agent(
                user_turns=user_turns,
                instance=instance,
                model_name=model,
                cost_limit=cost_limit,
                step_limit=step_limit,
                step_limit_per_turn=step_limit_per_turn,
                tool_call_limit_per_turn=tool_call_limit_per_turn,
                output_path=trajectory_path,
                use_tool_calling=use_tool_calling,
                overrides=overrides if overrides else None,
            )
        except Exception as e:
            success = False
            error = f"mini_agent_failed: {type(e).__name__}: {e}"

    if not success or mini_result is None:
        return {
            "task_id": task_id, "prediction": None, "correct": False,
            "ground_truth": "", "decoding": [], "user_messages": list(user_turns),
            "success": False, "error": error,
            "metadata": {"task_id": task_id, "original_id": instance_id, "source": source,
                         "scaffold_mode": "native" if use_tool_calling else "textbased"},
            "swe_eval": None,
        }

    # Score the patch via official harness.
    if mini_result.submission:
        hr = harness.verify_patch(
            instance_id=instance_id,
            patch=mini_result.submission,
            model_name=f"{model}-mini-agent",
        )
        correct = hr.resolved
        swe_eval = {
            "resolved": hr.resolved,
            "patch_extracted": hr.patch_extracted,
            "patch_apply_ok": hr.patch_apply_ok,
            "ftp_pass": hr.ftp_pass, "ftp_fail": hr.ftp_fail,
            "ptp_pass": hr.ptp_pass, "ptp_fail": hr.ptp_fail,
            "harness_error": hr.harness_error,
            "duration_s": hr.duration_s,
            "from_cache": hr.from_cache,
            "instance_id": instance_id,
        }
    else:
        correct = False
        swe_eval = {
            "resolved": False, "patch_extracted": False, "patch_apply_ok": False,
            "ftp_pass": [], "ftp_fail": [], "ptp_pass": [], "ptp_fail": [],
            "harness_error": "no_patch_extracted",
            "duration_s": 0.0, "from_cache": False, "instance_id": instance_id,
        }

    return {
        "task_id": task_id,
        "prediction": mini_result.submission,
        "correct": correct,
        "ground_truth": "",
        "decoding": [],  # full trajectory stored separately if needed
        "user_messages": list(user_turns),
        "success": True,
        "error": mini_result.error,
        "metadata": {
            "task_id": task_id,
            "original_id": instance_id,
            "source": source,
            "repo": (sample.get("swe_bench_metadata") or {}).get("repo"),
            "scaffold": "mini-swe-agent-v2",
            "scaffold_mode": mini_result.scaffold_mode,
            "n_steps": mini_result.n_steps,
            "n_user_turns_delivered": mini_result.n_user_turns_delivered,
            "exit_status": mini_result.exit_status,
            "step_limit_per_turn": mini_result.step_limit_per_turn,
            "per_turn_steps": list(mini_result.per_turn_steps),
            "tool_call_limit_per_turn": mini_result.tool_call_limit_per_turn,
            "per_turn_tool_calls": list(mini_result.per_turn_tool_calls),
            "scenario": sample.get("_scenario", "fully-specified"),
            "checkpoint_scenario": checkpoint_scenario,
            "requested_model": model,
            "resolved_model": resolved_model or resolve_model_name(model),
            "reasoning_effort": reasoning_effort,
            "agent_reused_from_trajectory": reused_trajectory_sha256 is not None,
            "agent_trajectory_sha256": reused_trajectory_sha256,
        },
        "swe_eval": swe_eval,
    }


def run(
    data_path: str,
    model: str,
    source: str,
    *,
    num_samples: int | None = None,
    num_workers: int = 1,
    task_ids_file: str | None = None,
    cost_limit: float = 3.0,
    step_limit: int = 250,
    step_limit_per_turn: int | None = None,
    rerun_failed: bool = False,
    extend: bool = False,
    output_suffix: str | None = None,
    use_tool_calling: bool = False,
    reasoning_effort: str | None = None,
    # Multi-turn flags. When all defaults (num_turns=1, no changes), the
    # legacy single-turn JSON-loading path is used (preserves the validated
    # 64% native baseline byte-for-byte). Otherwise samples are built via
    # EvolvingIntent.
    num_turns: int = 1,
    num_revisions: int = 0,
    num_switches: int = 0,
    ordering: str = "interleaved",
    prefix_style: str = "base",
    naturalizer_model: str | None = None,
    include_evidence: bool = True,
    strict_task_ids: bool = False,
    checkpoint_dir: str | None = None,
    checkpoint_scenario: str | None = None,
    output_path_override: str | None = None,
    tool_call_limit_per_turn: int | None = None,
    environment_class: str = "docker",
    harness_dataset_path: str | None = None,
) -> dict[str, Any]:
    if source not in _VALID_SOURCES:
        raise ValueError(f"--source must be one of {_VALID_SOURCES}")
    strict_resolved_model: str | None = None
    strict_ledger_path: Path | None = None
    strict_ledger_start = 0
    if strict_task_ids:
        validate_requested_models([model])
        if reasoning_effort != EXPECTED_REASONING_EFFORT:
            raise HardeningError(
                "strict SWE mode requires "
                f"--reasoning_effort {EXPECTED_REASONING_EFFORT}"
            )
        strict_resolved_model = validate_runtime_environment(os.environ)
        if resolve_model_name(model) != strict_resolved_model:
            raise HardeningError("resolved Kimi model changed during strict preflight")
        raw_ledger_path = os.environ.get("LLM_USAGE_LEDGER_PATH", "").strip()
        if not raw_ledger_path:
            raise HardeningError("strict SWE mode requires LLM_USAGE_LEDGER_PATH")
        strict_ledger_path = Path(raw_ledger_path).expanduser().resolve()
        strict_ledger_start = ledger_offset(strict_ledger_path)
        if not task_ids_file:
            raise HardeningError("strict SWE mode requires --task_ids_file")
        if num_samples is not None or rerun_failed or extend:
            raise HardeningError(
                "strict SWE mode forbids --num_samples, --rerun_failed, and --extend"
            )
        if naturalizer_model is not None:
            raise HardeningError("strict SWE mode forbids an online naturalizer model")
        if source != "paraphrase" or ordering != "interleaved" or prefix_style != "base":
            raise HardeningError("strict SWE mode requires the fixed prompt/scheduler configuration")
        if not include_evidence or not use_tool_calling:
            raise HardeningError("strict SWE mode requires evidence and native tool calling")
        if checkpoint_dir is None or checkpoint_scenario not in SCENARIO_BY_NAME:
            raise HardeningError(
                "strict SWE mode requires --checkpoint_dir and a known --checkpoint_scenario"
            )
        if tool_call_limit_per_turn != TOOL_CALL_LIMIT_PER_TURN:
            raise HardeningError(
                f"strict SWE mode requires --tool_call_limit_per_turn {TOOL_CALL_LIMIT_PER_TURN}"
            )
        expected_spec = SCENARIO_BY_NAME[checkpoint_scenario]
        if (
            num_turns,
            num_revisions,
            num_switches,
        ) != (
            expected_spec.turns,
            expected_spec.revisions,
            expected_spec.switches,
        ):
            raise HardeningError(
                f"scenario {checkpoint_scenario!r} does not match its fixed turn configuration"
            )
        if step_limit_per_turn != MODEL_STEP_LIMIT_PER_TURN:
            raise HardeningError(
                "strict SWE mode requires an unlimited model-step budget "
                f"(--step_limit_per_turn {MODEL_STEP_LIMIT_PER_TURN})"
            )
        if cost_limit != 0:
            raise HardeningError("strict SWE mode forbids mini-swe-agent's cost gate")
        if environment_class != "swerex_modal":
            raise HardeningError("strict SWE mode requires the swerex_modal environment")
        if not harness_dataset_path:
            raise HardeningError("strict SWE mode requires --harness_dataset_path")

    # Explicit AND gate: only the legacy JSON path counts as fully-specified.
    # Single-turn fully-specified MUST use the JSON path to preserve the
    # validated 64% baseline.
    is_single_turn = (
        num_turns == 1
        and num_revisions == 0
        and num_switches == 0
    )

    if is_single_turn:
        scenario = "fully-specified"
        with open(data_path) as f:
            data: list[dict[str, Any]] = json.load(f)
        requested_task_ids: list[str] | None = None
        if task_ids_file:
            with open(task_ids_file) as f:
                payload = json.load(f)
                requested_task_ids = list(payload.get("task_ids", payload))
                allowed = set(requested_task_ids)
            data = [d for d in data if d.get("task_id") in allowed]
        if num_samples is not None and not rerun_failed and not extend:
            data = data[:num_samples]
    else:
        # Multi-turn path: build samples through EvolvingIntent.
        task_ids_filter: set[str] | None = None
        requested_task_ids = None
        if task_ids_file:
            with open(task_ids_file) as f:
                payload = json.load(f)
                requested_task_ids = list(payload.get("task_ids", payload))
                task_ids_filter = set(requested_task_ids)
        sim = EvolvingIntent(
            data_path=data_path,
            mode="eval",
            domain="swe_bench_verified",
            num_turns=num_turns,
            num_revisions=num_revisions,
            num_switches=num_switches,
            ordering=ordering,
            task_ids=task_ids_filter,
            naturalizer_model=naturalizer_model,
            prefix_style=prefix_style,
            include_evidence=include_evidence,
        )
        scenario = sim.scenario
        samples = list(sim)
        if num_samples is not None and not rerun_failed and not extend:
            samples = samples[:num_samples]
        data = [_sample_to_dict(gs) for gs in samples]

    if strict_task_ids:
        assert requested_task_ids is not None
        canonical_task_ids = load_published_task_ids(
            PUBLISHED_EVAL_MANIFEST,
            PUBLISHED_TASK_IDS,
        )
        validate_exact_ids(
            requested_task_ids,
            canonical_task_ids,
            label="requested task IDs",
            require_order=True,
        )
        validate_exact_ids(
            [str(item.get("task_id") or "") for item in data],
            canonical_task_ids,
            label="scheduled SWE samples",
        )
        canonical_original_ids = load_published_id_map(PUBLISHED_EVAL_MANIFEST)
        mismatched_mappings = [
            item.get("task_id")
            for item in data
            if item.get("original_id")
            != canonical_original_ids.get(str(item.get("task_id") or ""))
        ]
        if mismatched_mappings:
            raise HardeningError(
                "scheduled SWE task/original ID mapping differs from the published set: "
                f"{mismatched_mappings[:5]}"
            )
        harness_rows = read_json(
            harness_dataset_path,
            label="filtered official SWE harness dataset",
        )
        if not isinstance(harness_rows, list):
            raise HardeningError("filtered official SWE harness dataset must be a list")
        validate_exact_ids(
            [
                str(row.get("instance_id") or "")
                for row in harness_rows
                if isinstance(row, dict)
            ],
            [str(item.get("original_id") or "") for item in data],
            label="official SWE harness instances",
        )

    # Default suffix when use_tool_calling=True so on-disk files don't
    # collide with the existing text-based baselines (gpt-5.1.json).
    # Final filename: gpt-5.1_native.json (single underscore).
    if output_suffix is None and use_tool_calling:
        output_suffix = "native"

    save_name = clean_model_name(model)
    output_filename = f"{save_name}.json"
    # For multi-turn scenarios, append _t{T}_g{G}_p{P} so configs that share
    # a scenario (e.g. combined t=5 vs combined t=7) don't collide on disk.
    if not is_single_turn:
        output_suffix = (output_suffix or "") + (
            f"_t{num_turns}_g{num_switches}_p{num_revisions}"
        )
    if output_suffix:
        output_filename = output_filename.replace(".json", f"_{output_suffix}.json")

    if output_path_override:
        output_path = Path(output_path_override).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = EXPERIMENTS_DIR / _experiments_subdir(source, scenario) / "swe_bench_verified"
        exp_dir.mkdir(parents=True, exist_ok=True)
        output_path = exp_dir / output_filename

    existing: dict[str, Any] = {}
    failed_ids: list[str] | None = None
    if output_path.exists() and checkpoint_dir is None:
        if extend:
            with open(output_path) as f:
                existing = json.load(f)
            already_done = set(existing.keys())
            data = [d for d in data if d.get("task_id") not in already_done]
            if not data:
                total = len(existing)
                correct = sum(1 for r in existing.values() if r.get("correct"))
                print(f"✅ All requested task_ids already in {output_path} ({correct}/{total}). Skip.")
                return {"model": model, "status": "skipped", "path": str(output_path)}
            print(f"🧩 Extend mode: keeping {len(already_done)} existing, running {len(data)} new in {output_path}")
        elif rerun_failed:
            with open(output_path) as f:
                existing = json.load(f)
            failed_ids = [
                tid for tid, r in existing.items()
                if not r.get("success", True) or r.get("error") is not None
                or (r.get("swe_eval") or {}).get("harness_error") is not None
            ]
            if not failed_ids:
                total = len(existing)
                correct = sum(1 for r in existing.values() if r.get("correct"))
                print(f"✅ No failed samples in {output_path} ({correct}/{total}). Skip.")
                return {"model": model, "status": "skipped", "path": str(output_path)}
            print(f"🔄 Rerunning {len(failed_ids)} failed samples in {output_path}")
        else:
            print(f"⏭️  Skipping (already exists): {output_path}")
            return {"model": model, "status": "skipped", "path": str(output_path)}

    if failed_ids is not None:
        data = [d for d in data if d.get("task_id") in set(failed_ids)]

    print("=" * 60)
    print(f"SWE-bench Verified mini-agent baseline ({source})")
    print(f"Scenario: {scenario}")
    if not is_single_turn:
        print(f"Multi-turn config: t={num_turns} g={num_switches} p={num_revisions} ordering={ordering}")
    print(f"Model: {model}")
    print(f"Samples: {len(data)}")
    print(f"step_limit={step_limit}, cost_limit={cost_limit}")
    if step_limit_per_turn is not None:
        print(f"step_limit_per_turn={step_limit_per_turn} (per-turn cap; legacy step_limit ignored)")
    print(f"Scaffold mode: {'native (function calling)' if use_tool_calling else 'textbased (regex parsing)'}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Per-model SWEHarness instance + per-model run_id namespace. This
    # gives each concurrent model its own threading.Lock so verify-time
    # contention is per-model rather than global within the launcher
    # process. Only relevant when multiple models share one launcher
    # process (multi-process launching gives the same isolation
    # automatically, since each process has its own SWEHarness).
    harness_kwargs: dict[str, Any] = {
        "modal": environment_class == "swerex_modal",
    }
    if harness_dataset_path:
        harness_kwargs["dataset_name"] = str(Path(harness_dataset_path).resolve())
    harness = SWEHarness(**harness_kwargs)
    harness_run_id = f"swe_{clean_model_name(model)}"
    resolved_model = strict_resolved_model or resolve_model_name(model)
    trajectory_dir: Path | None = None
    if checkpoint_scenario in SCENARIO_BY_NAME:
        run_root = output_path.parent.parent
        trajectory_dir = run_root / "trajectories" / str(checkpoint_scenario)

    def _trajectory_path(sample: dict[str, Any]) -> Path | None:
        if trajectory_dir is None:
            return None
        task_id = str(sample.get("task_id") or "unknown").replace("/", "_")
        return trajectory_dir / f"{task_id}.json"

    # Results buffer: starts with whatever extend/rerun_failed loaded so
    # incremental saves always reflect the full merged state. The
    # ``existing`` dict has already been used to filter ``data``; from
    # here on it's just the seed for ``results``.
    results: dict[str, Any] = dict(existing)
    checkpoint_store: TaskCheckpointStore | None = None
    resumed_count = 0
    pending_at_start = len(data)
    if checkpoint_dir is not None:
        if checkpoint_scenario not in SCENARIO_BY_NAME:
            raise HardeningError(f"unknown checkpoint scenario: {checkpoint_scenario!r}")
        checkpoint_store = TaskCheckpointStore(
            checkpoint_dir,
            scenario=SCENARIO_BY_NAME[checkpoint_scenario],
            requested_model=model,
            resolved_model=resolved_model,
        )
        pending: list[dict[str, Any]] = []
        for sample in data:
            task_id = str(sample.get("task_id") or "")
            if checkpoint_store.exists(task_id):
                results[task_id] = checkpoint_store.load(task_id)
                resumed_count += 1
            else:
                pending.append(sample)
        data = pending
        pending_at_start = len(data)
        print(
            f"Checkpoint resume: valid={resumed_count}, pending={len(data)}, "
            f"directory={checkpoint_store.directory}"
        )
    pending_task_ids_at_start = [str(sample.get("task_id") or "") for sample in data]
    write_lock = threading.Lock()

    def _save_incremental() -> None:
        """Atomically dump the current results dict to ``output_path``.

        Uses a tmp + rename so an interruption (SIGINT, SIGKILL, OOM,
        host reboot) can never leave a half-written JSON on disk: the
        previous file remains intact until rename succeeds.
        """
        with write_lock:
            atomic_write_json(output_path, results)

    # Make sure the file exists from the start so other tooling can
    # poll it while the run is in progress (and `--extend` on a later
    # invocation finds something even if we crash before the first
    # task completes).
    if results:
        _save_incremental()

    failures: list[str] = []

    def _accept_result(result: dict[str, Any]) -> None:
        task_id = str(result.get("task_id") or "")
        if checkpoint_store is not None:
            checkpoint_store.write(task_id, result)
        results[task_id] = result
        _save_incremental()

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {
                ex.submit(
                    evaluate_one, sample, model, source, harness,
                    cost_limit=cost_limit, step_limit=step_limit,
                    step_limit_per_turn=step_limit_per_turn,
                    tool_call_limit_per_turn=tool_call_limit_per_turn,
                    use_tool_calling=use_tool_calling,
                    reasoning_effort=reasoning_effort,
                    environment_class=environment_class,
                    checkpoint_scenario=checkpoint_scenario,
                    resolved_model=resolved_model,
                    trajectory_path=_trajectory_path(sample),
                ): sample
                for sample in data
            }
            from tqdm import tqdm
            for fut in tqdm(as_completed(futs), total=len(data), desc=model):
                try:
                    r = fut.result()
                    _accept_result(r)
                except Exception as e:
                    s = futs[fut]
                    print(f"\n❌ {s.get('task_id')}: {e}")
                    failures.append(f"{s.get('task_id')}: {type(e).__name__}: {e}")
    else:
        from tqdm import tqdm
        for sample in tqdm(data, desc=model):
            try:
                r = evaluate_one(
                    sample, model, source, harness,
                    cost_limit=cost_limit, step_limit=step_limit,
                    step_limit_per_turn=step_limit_per_turn,
                    tool_call_limit_per_turn=tool_call_limit_per_turn,
                    use_tool_calling=use_tool_calling,
                    reasoning_effort=reasoning_effort,
                    environment_class=environment_class,
                    checkpoint_scenario=checkpoint_scenario,
                    resolved_model=resolved_model,
                    trajectory_path=_trajectory_path(sample),
                )
                _accept_result(r)
            except Exception as e:
                print(f"\n❌ {sample.get('task_id')}: {e}")
                failures.append(
                    f"{sample.get('task_id')}: {type(e).__name__}: {e}"
                )

    # Final save (no-op if we've been saving incrementally, but cheap
    # insurance and writes the same atomic file).
    if results:
        _save_incremental()

    if strict_task_ids:
        assert requested_task_ids is not None
        assert checkpoint_store is not None
        validate_aggregate_results(
            results,
            requested_task_ids,
            store=checkpoint_store,
        )
    if failures:
        raise RuntimeError(
            f"{len(failures)} SWE task(s) failed; first failure: {failures[0]}"
        )
    if strict_task_ids:
        assert strict_ledger_path is not None
        all_usage_events = read_usage_events(strict_ledger_path)
        validate_usage_events(
            all_usage_events,
            requested_model=model,
            resolved_model=resolved_model,
        )
        new_usage_events = read_usage_events(
            strict_ledger_path,
            start_offset=strict_ledger_start,
        )
        validate_usage_events(
            new_usage_events,
            requested_model=model,
            resolved_model=resolved_model,
        )
        if pending_at_start and not new_usage_events:
            verifier_only_retry = all(
                (results.get(task_id, {}).get("metadata") or {}).get(
                    "agent_reused_from_trajectory"
                )
                is True
                for task_id in pending_task_ids_at_start
            )
            if not verifier_only_retry:
                raise HardeningError(
                    "strict SWE tasks completed without provider usage records"
                )

    total = len(results)
    correct = sum(1 for r in results.values() if r.get("correct"))
    failed = sum(1 for r in results.values() if not r.get("success", True))
    accuracy = correct / total if total else 0.0
    print(f"\n📊 {model} ({source}): {correct}/{total} = {accuracy:.1%}  failed={failed}")

    return {
        "model": model, "source": source,
        "total": total, "correct": correct, "failed": failed,
        "accuracy": accuracy, "output_path": str(output_path),
        "resumed": resumed_count,
        "executed": len(results) - resumed_count,
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--source", choices=_VALID_SOURCES, default="paraphrase",
                   help="'raw' (problem_statement) or 'paraphrase' (fully_specified_question, default).")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=1,
                   help="Parallel agent runs. Each spins up its own docker container.")
    p.add_argument("--task_ids_file", type=str, default=None)
    p.add_argument("--cost_limit", type=float, default=3.0,
                   help="Per-instance cost cap (USD). Default 3.0 matches mini-swe-agent v2 swebench.yaml. "
                        "Note: the API doesn't surface per-call cost, so this guard never fires; "
                        "step_limit is the binding constraint.")
    p.add_argument("--step_limit", type=int, default=250,
                   help="Legacy per-instance global step cap. Default 250 matches mini-swe-agent v2 swebench.yaml leaderboard config. "
                        "When --step_limit_per_turn is also passed, this value is ignored (per-turn cap wins).")
    p.add_argument("--step_limit_per_turn", type=int, default=50,
                   help="Per-turn step cap for multi-turn EvolvingIntent scenarios. Each intent turn gets a fresh "
                        "budget of N steps (mini's agent.n_calls is reset to 0 at each turn boundary). "
                        "Default 50; empirically fully-specified GPT-5.1 used median 24 / max 61 steps so "
                        "50 covers ~94%% of single-turn trajectories. Pass an explicit value or 0 (= unlimited "
                        "per turn, mirrors mini's step_limit=0) to override. Set --step_limit_per_turn -1 to "
                        "fall back to legacy single-cap mode using --step_limit.")
    p.add_argument("--rerun_failed", action="store_true")
    p.add_argument("--extend", action="store_true",
                   help="Keep existing per-sample results; only run task_ids not yet present.")
    p.add_argument("--output_suffix", type=str, default=None)
    p.add_argument("--reasoning_effort", type=str, default=None,
                   choices=["minimal", "low", "medium", "high"],
                   help="Optional reasoning effort for GPT-5.x reasoning models. "
                        "matches mini-swe-agent v2 leaderboard config (metadata.yaml: "
                        "'GPT-5.1 (2025-11-13) (medium reasoning)').")
    p.add_argument("--use_tool_calling", action="store_true",
                   help="Route LLM calls through native OpenAI function calling "
                        "(LLMToolModel + swebench.yaml). Default off — uses "
                        "the regex-on-fenced-blocks scaffold (LLMTextbasedModel + "
                        "swebench_backticks.yaml). When set, output_suffix defaults to "
                        "'native' so files don't collide with text-based baselines.")
    # Multi-turn EvolvingIntent scenario flags (mirrors run_experiment.py). Default
    # 0/0/0 keeps the legacy single-turn JSON-loading path → preserves the
    # validated 64% native baseline byte-for-byte. Any non-default flips the
    # runner into EvolvingIntent sample-generation mode.
    p.add_argument("--num_turns", type=int, default=1,
                   help="Number of conversation turns. 1 = fully-specified (default).")
    p.add_argument("--num_revisions", type=int, default=0,
                   help="Number of revisions (revision scenario when >0).")
    p.add_argument("--num_switches", type=int, default=0,
                   help="Number of function switches (switch scenario when >0).")
    p.add_argument("--ordering", type=str, default="interleaved",
                   choices=["sequential", "interleaved", "mixed", "random"],
                   help="EvolvingIntent turn-ordering policy.")
    p.add_argument("--prefix_style", type=str, default="base",
                   choices=["base", "function-naturalized", "function-naturalized-v2"])
    p.add_argument("--naturalizer_model", type=str, default=None,
                   help="Optional online naturalizer model.")
    p.add_argument("--tool_call_limit_per_turn", type=int, default=None,
                   help="Cap actual environment execute calls per user turn. The hardened path requires 200.")
    p.add_argument("--environment_class", type=str, default="docker",
                   choices=["docker", "swerex_modal"],
                   help="mini-swe-agent execution environment.")
    p.add_argument("--strict_task_ids", action="store_true",
                   help="Fail closed unless the fixed 50-task Kimi run is complete.")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Directory for one atomic checkpoint per task ID.")
    p.add_argument("--checkpoint_scenario", choices=sorted(SCENARIO_BY_NAME), default=None)
    p.add_argument("--output_path", type=str, default=None,
                   help="Explicit aggregate result path (checkpoint mode is the resume source).")
    p.add_argument("--harness_dataset_path", type=str, default=None,
                   help="Local filtered official SWE dataset for the verifier (.json).")
    args = p.parse_args()
    if args.strict_task_ids:
        validate_requested_models(args.models)

    summaries = []
    # Sentinel -1 means "fall back to legacy single-cap mode" (use --step_limit
    # as a single global trajectory cap). Any other value (including 0) is
    # treated as the per-turn cap.
    step_limit_per_turn = None if args.step_limit_per_turn == -1 else args.step_limit_per_turn
    for model in args.models:
        s = run(
            data_path=args.data_path,
            model=model,
            source=args.source,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            task_ids_file=args.task_ids_file,
            cost_limit=args.cost_limit,
            step_limit=args.step_limit,
            step_limit_per_turn=step_limit_per_turn,
            rerun_failed=args.rerun_failed,
            extend=args.extend,
            output_suffix=args.output_suffix,
            use_tool_calling=args.use_tool_calling,
            reasoning_effort=args.reasoning_effort,
            num_turns=args.num_turns,
            num_revisions=args.num_revisions,
            num_switches=args.num_switches,
            ordering=args.ordering,
            prefix_style=args.prefix_style,
            naturalizer_model=args.naturalizer_model,
            strict_task_ids=args.strict_task_ids,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_scenario=args.checkpoint_scenario,
            output_path_override=args.output_path,
            tool_call_limit_per_turn=args.tool_call_limit_per_turn,
            environment_class=args.environment_class,
            harness_dataset_path=args.harness_dataset_path,
        )
        summaries.append(s)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for s in summaries:
        if s.get("status") == "skipped":
            print(f"  {s['model']:30s}  skipped ({s['path']})")
        else:
            print(f"  {s['model']:30s}  {s['source']:11s}  {s['correct']}/{s['total']} = {s['accuracy']:.1%}")


if __name__ == "__main__":
    main()
