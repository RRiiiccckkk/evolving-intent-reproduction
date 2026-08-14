#!/usr/bin/env python3
"""Run the small Plan A reproduction end to end.

The offline ``dry-run`` command validates selection, settings, paths, and
secret hygiene without importing model or dataset libraries.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .workflow import (
    DEFAULT_CONFIG_PATH,
    PLAN_A_SETTING_NAMES,
    REPO_ROOT,
    DeadlineReached,
    WorkflowError,
    aggregate_results,
    append_ledger,
    atomic_write_json,
    build_manifest,
    canonicalize_stage_file,
    choose_sample_count,
    contains_secret_material,
    enforce_deadline,
    file_sha256,
    inspect_stage,
    iso_now,
    iter_result_files,
    load_config,
    load_list,
    parse_deadline,
    read_json,
    repeat_last_user_turn,
    selected_original_ids,
    selected_task_ids,
    unwrap_results,
    validate_setting_sample,
    write_summary_csv,
    write_summary_html,
)


def _run_dir(run_id: str, override: str | None) -> Path:
    return Path(override).resolve() if override else REPO_ROOT / "reproduction" / "runs" / run_id


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _config_models(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    construction = args.construction_model or config["construction"]["default_model"]
    evaluation = args.evaluation_model or config["evaluation"]["default_model"]
    if not construction or not evaluation:
        raise WorkflowError("construction and evaluation model IDs cannot be empty")
    return construction, evaluation


def _deadline(config: Mapping[str, Any], args: argparse.Namespace):
    value = args.deadline
    if value is None:
        value = config["deadline"].get("operational_cutoff")
    return parse_deadline(value)


def _create_or_load_context(
    args: argparse.Namespace,
    *,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    config = load_config(args.config)
    construction_model, evaluation_model = _config_models(config, args)
    deadline = _deadline(config, args)
    run_id = args.run_id or ("plan-a-dry-run" if dry_run else "plan-a-20260814")
    run_dir = _run_dir(run_id, args.run_dir)
    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("run_id") != run_id:
            raise WorkflowError("existing manifest run_id does not match the requested run")
        if args.sample_count is not None:
            existing_count = int(manifest["dataset"]["requested_count"])
            if existing_count != args.sample_count:
                raise WorkflowError(
                    f"existing run uses N={existing_count}; choose a new run_id for N={args.sample_count}"
                )
        if args.construction_model is not None:
            manifest["models"]["construction"] = construction_model
        if args.evaluation_model is not None:
            manifest["models"]["evaluation"] = evaluation_model
        manifest["deadline"] = deadline.isoformat() if deadline else None
        manifest["budget"] = dict(config["budget"])
    else:
        sample_count, reason = choose_sample_count(
            config,
            args.sample_count,
            auto_fallback=bool(args.auto_fallback),
            deadline=deadline,
        )
        manifest = build_manifest(
            config,
            run_id=run_id,
            sample_count=sample_count,
            selection_reason=reason,
            construction_model=construction_model,
            evaluation_model=evaluation_model,
            deadline=deadline,
            dry_run=dry_run,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    return config, manifest, run_dir, manifest_path


def _save_manifest(manifest_path: Path, manifest: dict[str, Any], status: str | None = None) -> None:
    if status:
        manifest["status"] = status
    manifest["updated_at"] = iso_now()
    atomic_write_json(manifest_path, manifest)


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    data_dir = run_dir / "data"
    return {
        "stage1": data_dir / "stage1_extracted.json",
        "stage1_failures": data_dir / "stage1_failures.json",
        "stage2": data_dir / "stage2_counterfactual.json",
        "stage3": data_dir / "stage3_predecessor.json",
        "ledger": run_dir / "cost_ledger.jsonl",
        "summary": run_dir / "summary.json",
        "summary_csv": run_dir / "summary.csv",
        "summary_html": run_dir / "summary.html",
    }


def _ensure_credentials() -> None:
    backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if backend in {"generic", "openai-compatible", "openai_compatible"}:
        backend = "compatible"
    if backend and backend not in {"openai", "azure", "compatible"}:
        raise WorkflowError("LLM_BACKEND must be openai, azure, or compatible")
    openai_ready = bool(os.environ.get("OPENAI_API_KEY"))
    azure_ready = bool(
        os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    compatible_ready = bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"))
    ready = {
        "openai": openai_ready,
        "azure": azure_ready,
        "compatible": compatible_ready,
    }
    if (backend and not ready[backend]) or (not backend and not any(ready.values())):
        raise WorkflowError(
            "No credentials found for the selected backend. Use OPENAI_API_KEY, "
            "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT, or "
            "LLM_API_KEY + LLM_BASE_URL. Do not pass keys as CLI arguments."
        )


def _configure_accounting(run_dir: Path, config: Mapping[str, Any]) -> None:
    raw_prices = os.environ.get("LLM_PRICE_MAP", "").strip()
    if not raw_prices:
        raise WorkflowError(
            "LLM_PRICE_MAP is required for a live run; use provider-documented USD-per-million-token prices"
        )
    try:
        prices = json.loads(raw_prices)
    except json.JSONDecodeError as exc:
        raise WorkflowError("LLM_PRICE_MAP must be a JSON object") from exc
    if not isinstance(prices, dict) or not prices:
        raise WorkflowError("LLM_PRICE_MAP must be a non-empty JSON object")
    os.environ["LLM_USAGE_LEDGER_PATH"] = str(_artifact_paths(run_dir)["ledger"])
    hard_cap = config["budget"].get("hard_cap_usd")
    if hard_cap is None:
        os.environ.pop("LLM_COST_HARD_CAP_USD", None)
    else:
        os.environ["LLM_COST_HARD_CAP_USD"] = str(hard_cap)

    default_output = config["construction"].get("default_max_output_tokens")
    if default_output is None:
        os.environ.pop("LLM_DEFAULT_MAX_OUTPUT_TOKENS", None)
    else:
        os.environ["LLM_DEFAULT_MAX_OUTPUT_TOKENS"] = str(default_output)

    if bool(config.get("runtime", {}).get("disable_output_limits", False)):
        if hard_cap is not None:
            raise WorkflowError("output limits cannot be disabled with a hard cost cap")
        os.environ["LLM_DISABLE_OUTPUT_LIMITS"] = "1"
    else:
        os.environ.pop("LLM_DISABLE_OUTPUT_LIMITS", None)


def _ensure_live_dependencies(*, construction: bool) -> None:
    modules = ["openai", "tqdm"]
    if construction:
        modules.append("datasets")
    missing = []
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise WorkflowError(
            "Missing live-run dependencies: "
            + ", ".join(missing)
            + ". Install reproduction/requirements.txt in a Python 3.10+ environment."
        )


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    ledger_path: Path,
    stage: str,
    model: str,
    deadline,
    deadline_buffer: int,
    ignore_deadline: bool,
) -> None:
    enforce_deadline(
        deadline,
        buffer_minutes=deadline_buffer,
        ignore=ignore_deadline,
    )
    print(f"[{stage}] {shlex.join(command)}")
    start = time.monotonic()
    append_ledger(
        ledger_path,
        {"event": "stage_start", "stage": stage, "model": model},
    )
    try:
        subprocess.run(
            list(command),
            cwd=cwd,
            env=_subprocess_env(),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        append_ledger(
            ledger_path,
            {
                "event": "stage_end",
                "stage": stage,
                "model": model,
                "status": "failed",
                "returncode": exc.returncode,
                "elapsed_seconds": round(time.monotonic() - start, 3),
            },
        )
        raise WorkflowError(f"{stage} exited with code {exc.returncode}") from exc
    append_ledger(
        ledger_path,
        {
            "event": "stage_end",
            "stage": stage,
            "model": model,
            "status": "complete",
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )


def _run_stage1(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    model: str,
    workers: int,
    deadline,
    deadline_buffer: int,
    ignore_deadline: bool,
) -> dict[str, Any]:
    selected_ids = selected_task_ids(manifest)
    existing: list[dict[str, Any]] = []
    if paths["stage1"].exists():
        existing = load_list(paths["stage1"])
    by_id = {
        row.get("task_id"): row
        for row in existing
        if row.get("task_id") in set(selected_ids)
        and row.get("function")
        and row.get("arguments")
    }
    missing_entries = [
        entry
        for entry in manifest["dataset"]["selected_samples"]
        if entry["task_id"] not in by_id
    ]
    if not missing_entries:
        canonicalize_stage_file(paths["stage1"], selected_ids)
        return inspect_stage(paths["stage1"], selected_ids, "stage1")

    enforce_deadline(
        deadline,
        buffer_minutes=deadline_buffer,
        ignore=ignore_deadline,
    )
    from datasets import load_dataset
    from intent_construction.intent_extraction.registry import get_extractor

    print(f"[stage1] loading GSM8K test split; {len(missing_entries)} selected samples pending")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    original_ids = selected_original_ids(manifest)
    if max(original_ids) >= len(dataset):
        raise WorkflowError("published original_id falls outside the GSM8K test split")

    samples: dict[int, dict[str, Any]] = {}
    for original_id in original_ids:
        source = dict(dataset[original_id])
        source.update({"id": original_id, "task": "math", "split": "test"})
        samples[original_id] = source

    local = threading.local()
    construction = config["construction"]

    def process(entry: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            extractor = getattr(local, "extractor", None)
            if extractor is None:
                extractor = get_extractor(
                    "gsm8k",
                    model=model,
                    verif_model=model,
                    enable_model_verification=bool(construction["model_verification"]),
                    max_verification_attempts=int(construction["max_verification_attempts"]),
                )
                local.extractor = extractor
            result = extractor.extract(samples[int(entry["original_id"])])
            return str(entry["task_id"]), result, None
        except Exception as exc:  # the failure is checkpointed and can be retried
            return str(entry["task_id"]), None, f"{type(exc).__name__}: {exc}"

    failures: dict[str, str] = {}
    start = time.monotonic()
    append_ledger(paths["ledger"], {"event": "stage_start", "stage": "stage1", "model": model})
    for offset in range(0, len(missing_entries), workers):
        enforce_deadline(
            deadline,
            buffer_minutes=deadline_buffer,
            ignore=ignore_deadline,
        )
        batch = missing_entries[offset : offset + workers]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, entry) for entry in batch]
            for future in as_completed(futures):
                task_id, result, error = future.result()
                if result is not None:
                    by_id[task_id] = result
                    failures.pop(task_id, None)
                else:
                    failures[task_id] = error or "unknown extraction failure"
                atomic_write_json(
                    paths["stage1"],
                    [by_id[task_id] for task_id in selected_ids if task_id in by_id],
                )
                atomic_write_json(paths["stage1_failures"], failures)

    status = inspect_stage(paths["stage1"], selected_ids, "stage1")
    append_ledger(
        paths["ledger"],
        {
            "event": "stage_end",
            "stage": "stage1",
            "model": model,
            "status": "complete" if status["complete"] else "partial",
            "completed_samples": status["count"],
            "elapsed_seconds": round(time.monotonic() - start, 3),
        },
    )
    return status


def _run_stage2(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    model: str,
    workers: int,
    deadline,
    deadline_buffer: int,
    ignore_deadline: bool,
) -> dict[str, Any]:
    selected_ids = selected_task_ids(manifest)
    required = int(config["construction"]["counterfactuals_per_argument"])
    status = inspect_stage(
        paths["stage2"], selected_ids, "stage2", required_revisions=2
    )
    directory = REPO_ROOT / "intent_construction" / "retrospective_expansion" / "counterfactual"
    if not status["exists"]:
        command = [
            sys.executable,
            "generate_counterfactuals.py",
            "--input",
            str(paths["stage1"]),
            "--output",
            str(paths["stage2"]),
            "--num_counterfactuals",
            str(required),
            "--model",
            model,
            "--dataset_type",
            "math",
            "--seed",
            str(manifest["seed"]),
            "--checkpoint_interval",
            str(workers),
            "--resume",
            "--batch",
            "--batch_size",
            str(workers),
        ]
        _run_command(
            command,
            cwd=directory,
            ledger_path=paths["ledger"],
            stage="stage2",
            model=model,
            deadline=deadline,
            deadline_buffer=deadline_buffer,
            ignore_deadline=ignore_deadline,
        )

    canonicalize_stage_file(paths["stage2"], selected_ids)
    status = inspect_stage(paths["stage2"], selected_ids, "stage2", required_revisions=2)
    if not status["complete"]:
        if status["invalid_task_ids"]:
            invalid = set(status["invalid_task_ids"])
            valid_rows = [
                row
                for row in load_list(paths["stage2"])
                if row.get("task_id") not in invalid
            ]
            atomic_write_json(paths["stage2"], valid_rows)
        retry = [
            sys.executable,
            "retry_failed.py",
            "--input",
            str(paths["stage1"]),
            "--output",
            str(paths["stage2"]),
            "--model",
            model,
            "--dataset_type",
            "math",
            "--num_counterfactuals",
            str(required),
            "--batch_size",
            str(workers),
        ]
        _run_command(
            retry,
            cwd=directory,
            ledger_path=paths["ledger"],
            stage="stage2_retry",
            model=model,
            deadline=deadline,
            deadline_buffer=deadline_buffer,
            ignore_deadline=ignore_deadline,
        )
        canonicalize_stage_file(paths["stage2"], selected_ids)
        status = inspect_stage(paths["stage2"], selected_ids, "stage2", required_revisions=2)
    return status


def _merge_stage3_retry(primary: Path, retry: Path, selected_ids: Sequence[str]) -> None:
    rows = load_list(primary) if primary.exists() else []
    if retry.exists():
        rows.extend(load_list(retry))
    by_id = {row.get("task_id"): row for row in rows if row.get("task_id")}
    atomic_write_json(primary, [by_id[task_id] for task_id in selected_ids if task_id in by_id])


def _stage3_command(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    workers: int,
) -> list[str]:
    construction = config["construction"]
    return [
        sys.executable,
        "generate_predecessors.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--dataset_type",
        "gsm8k",
        "--num_predecessors",
        str(construction["predecessors_per_sample"]),
        "--model",
        model,
        "--fallback_model",
        model,
        "--judge_model",
        model,
        "--seed",
        str(manifest["seed"]),
        "--parallel",
        str(workers),
        "--checkpoint_interval",
        str(workers),
        "--resume",
        "--independence_runs",
        str(construction["independence_runs"]),
        "--max_independence_retries",
        str(construction["max_independence_retries"]),
    ]


def _run_stage3(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    model: str,
    workers: int,
    deadline,
    deadline_buffer: int,
    ignore_deadline: bool,
) -> dict[str, Any]:
    selected_ids = selected_task_ids(manifest)
    directory = REPO_ROOT / "intent_construction" / "retrospective_expansion" / "predecessor"
    status = inspect_stage(paths["stage3"], selected_ids, "stage3")
    if not status["exists"]:
        _run_command(
            _stage3_command(
                config,
                manifest,
                input_path=paths["stage2"],
                output_path=paths["stage3"],
                model=model,
                workers=workers,
            ),
            cwd=directory,
            ledger_path=paths["ledger"],
            stage="stage3",
            model=model,
            deadline=deadline,
            deadline_buffer=deadline_buffer,
            ignore_deadline=ignore_deadline,
        )

    canonicalize_stage_file(paths["stage3"], selected_ids)
    status = inspect_stage(paths["stage3"], selected_ids, "stage3")
    if not status["complete"]:
        invalid = set(status["invalid_task_ids"])
        done = (
            {
                row.get("task_id")
                for row in load_list(paths["stage3"])
                if row.get("task_id") not in invalid
            }
            if paths["stage3"].exists()
            else set()
        )
        missing_rows = [
            row for row in load_list(paths["stage2"]) if row.get("task_id") not in done
        ]
        if missing_rows:
            retry_input = paths["stage3"].with_name("stage3_retry_input.json")
            retry_output = paths["stage3"].with_name("stage3_retry_output.json")
            atomic_write_json(retry_input, missing_rows)
            _run_command(
                _stage3_command(
                    config,
                    manifest,
                    input_path=retry_input,
                    output_path=retry_output,
                    model=model,
                    workers=workers,
                ),
                cwd=directory,
                ledger_path=paths["ledger"],
                stage="stage3_retry",
                model=model,
                deadline=deadline,
                deadline_buffer=deadline_buffer,
                ignore_deadline=ignore_deadline,
            )
            _merge_stage3_retry(paths["stage3"], retry_output, selected_ids)
            retry_input.unlink(missing_ok=True)
            retry_output.unlink(missing_ok=True)
        status = inspect_stage(paths["stage3"], selected_ids, "stage3")
    return status


def run_construction(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config, manifest, run_dir, manifest_path = _create_or_load_context(args)
    _configure_accounting(run_dir, config)
    _ensure_credentials()
    _ensure_live_dependencies(construction=True)
    paths = _artifact_paths(run_dir)
    deadline = parse_deadline(manifest.get("deadline"))
    buffer_minutes = int(config["deadline"]["guard_minutes"])
    workers = args.workers or int(config["construction"]["workers"])
    if workers < 1:
        raise WorkflowError("workers must be at least 1")
    model = manifest["models"]["construction"]
    selected_ids = selected_task_ids(manifest)

    stage1 = _run_stage1(
        config,
        manifest,
        paths,
        model=model,
        workers=workers,
        deadline=deadline,
        deadline_buffer=buffer_minutes,
        ignore_deadline=args.ignore_deadline,
    )
    manifest["coverage"]["stage1"] = stage1
    _save_manifest(manifest_path, manifest, "constructing")
    if not stage1["complete"]:
        raise WorkflowError(f"stage1 incomplete: {stage1['missing_task_ids'] + stage1['invalid_task_ids']}")

    stage2 = _run_stage2(
        config,
        manifest,
        paths,
        model=model,
        workers=workers,
        deadline=deadline,
        deadline_buffer=buffer_minutes,
        ignore_deadline=args.ignore_deadline,
    )
    manifest["coverage"]["stage2"] = stage2
    _save_manifest(manifest_path, manifest, "constructing")
    if not stage2["complete"]:
        raise WorkflowError(f"stage2 incomplete: {stage2['missing_task_ids'] + stage2['invalid_task_ids']}")

    stage3 = _run_stage3(
        config,
        manifest,
        paths,
        model=model,
        workers=workers,
        deadline=deadline,
        deadline_buffer=buffer_minutes,
        ignore_deadline=args.ignore_deadline,
    )
    manifest["coverage"]["stage3"] = stage3
    eligible = [
        task_id
        for task_id in selected_ids
        if task_id not in set(stage3["missing_task_ids"] + stage3["invalid_task_ids"])
    ]
    manifest["dataset"]["eligible_task_ids"] = eligible
    for stage in ("stage1", "stage2", "stage3"):
        path = paths[stage]
        manifest["artifacts"][stage] = {
            "path": _rel(path),
            "sha256": file_sha256(path) if path.exists() else None,
        }
    _save_manifest(manifest_path, manifest, "constructed" if stage3["complete"] else "partial")
    if not stage3["complete"] and not args.allow_partial:
        raise WorkflowError(
            "stage3 is incomplete; rerun construct to retry, use the N=10 fallback, "
            "or pass --allow-partial and accept reduced coverage"
        )
    print(f"Construction ready: {len(eligible)}/{len(selected_ids)} samples")
    return manifest, run_dir


def _setting_by_name(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    for setting in manifest["settings"]:
        if setting["name"] == name:
            return dict(setting)
    raise WorkflowError(f"setting not found in manifest: {name}")


def build_setting_samples(
    data_path: Path,
    task_ids: Sequence[str],
    setting: Mapping[str, Any],
    *,
    seed: int,
    require_all: bool = True,
) -> list[Any]:
    """Build samples for a standard or turn-matched repeat-control setting."""
    from situated_simulation.user_simulation import EvolvingIntent

    if setting["kind"] == "repeat_control":
        params = setting["base"]
    else:
        params = setting
    simulator = EvolvingIntent(
        data_path=data_path,
        mode="eval",
        domain="math",
        ordering="interleaved",
        num_turns=int(params["turns"]),
        num_revisions=int(params["revisions"]),
        num_switches=int(params["switches"]),
        seed=seed,
        task_ids=list(task_ids),
    )
    by_id = {sample.task_id: sample for sample in simulator}
    samples = [by_id[task_id] for task_id in task_ids if task_id in by_id]
    base_samples: dict[str, Any] = {}
    if setting["kind"] == "repeat_control":
        base_samples = {sample.task_id: copy.deepcopy(sample) for sample in samples}
        samples = [repeat_last_user_turn(sample, int(setting["repeat_turns"])) for sample in samples]

    expected_turns = int(setting["turns"])
    valid_samples = []
    for sample in samples:
        try:
            validate_setting_sample(
                sample,
                setting,
                base_sample=base_samples.get(sample.task_id),
            )
        except WorkflowError:
            if require_all:
                raise
            continue
        sample.metadata = {
            **dict(sample.metadata),
            "reproduction_setting": setting["name"],
        }
        valid_samples.append(sample)
    return valid_samples


def _load_result_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}, unwrap_results(payload)


def _validate_result_checkpoint(
    path: Path,
    header: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    model: str,
    task_ids: Sequence[str],
) -> None:
    if not header:
        return
    mismatches = []
    if header.get("run_id") != manifest["run_id"]:
        mismatches.append("run_id")
    if header.get("model") != model:
        mismatches.append("model")
    checkpoint_setting = header.get("setting")
    if not isinstance(checkpoint_setting, Mapping) or checkpoint_setting.get(
        "name"
    ) != setting.get("name"):
        mismatches.append("setting")
    if header.get("selected_task_ids") != list(task_ids):
        mismatches.append("selected_task_ids")
    if mismatches:
        raise WorkflowError(
            f"result checkpoint {path} does not match the requested run "
            f"({', '.join(mismatches)}); use a new run_id"
        )


def _save_result_checkpoint(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    model: str,
    task_ids: Sequence[str],
    results: Mapping[str, Mapping[str, Any]],
) -> None:
    ordered = {task_id: results[task_id] for task_id in task_ids if task_id in results}
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "run_id": manifest["run_id"],
            "setting": dict(setting),
            "model": model,
            "selected_task_ids": list(task_ids),
            "updated_at": iso_now(),
            "results": ordered,
        },
    )


def _run_evaluation_setting(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run_dir: Path,
    setting: Mapping[str, Any],
    *,
    task_ids: Sequence[str],
    model: str,
    workers: int,
    retry_failed: bool,
    deadline,
    deadline_buffer: int,
    ignore_deadline: bool,
    only_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    from evaluation.runners.run_experiment import evaluate_sample

    result_path = run_dir / "results" / f"{setting['name']}.json"
    header, results = _load_result_checkpoint(result_path)
    _validate_result_checkpoint(
        result_path,
        header,
        manifest=manifest,
        setting=setting,
        model=model,
        task_ids=task_ids,
    )
    completed = set(results)
    if retry_failed:
        completed = {
            task_id
            for task_id, result in results.items()
            if bool(result.get("success", True)) and result.get("error") is None
        }
    pending_ids = [task_id for task_id in task_ids if task_id not in completed]
    if only_task_ids is not None:
        allowed = set(only_task_ids)
        pending_ids = [task_id for task_id in pending_ids if task_id in allowed]
    if not pending_ids:
        print(f"[{setting['name']}] already complete")
        return {"path": result_path, "new_results": {}}

    all_samples = build_setting_samples(
        _artifact_paths(run_dir)["stage3"],
        task_ids,
        setting,
        seed=int(manifest["seed"]),
    )
    pending = set(pending_ids)
    samples = [sample for sample in all_samples if sample.task_id in pending]
    if len(samples) != len(pending_ids):
        found = {sample.task_id for sample in samples}
        raise WorkflowError(
            f"{setting['name']} scheduler rejected task IDs: "
            + ", ".join(task_id for task_id in pending_ids if task_id not in found)
        )

    evaluation = config["evaluation"]
    raw_temperature = evaluation.get("temperature")
    temperature = None if raw_temperature is None else float(raw_temperature)
    raw_max_tokens = evaluation.get("max_tokens")
    max_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
    reasoning_effort = evaluation.get("reasoning_effort")
    newly_completed: dict[str, dict[str, Any]] = {}

    def process(sample: Any) -> tuple[str, dict[str, Any]]:
        try:
            result = evaluate_sample(
                sample,
                model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            result = {
                "task_id": sample.task_id,
                "prediction": None,
                "correct": False,
                "ground_truth": sample.label,
                "decoding": [],
                "user_messages": [
                    turn["content"] for turn in sample.turns if turn.get("role") == "user"
                ],
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "metadata": sample.metadata,
            }
        if bool(result.get("success", False)):
            user_turn_count = sum(turn.get("role") == "user" for turn in sample.turns)
            response_count = len(result.get("decoding") or [])
            if response_count != user_turn_count:
                result.update(
                    {
                        "success": False,
                        "correct": False,
                        "error": (
                            f"assistant response count {response_count} does not match "
                            f"user turn count {user_turn_count}"
                        ),
                    }
                )
        return sample.task_id, result

    print(f"[{setting['name']}] evaluating {len(samples)} samples with {model}")
    for offset in range(0, len(samples), workers):
        enforce_deadline(
            deadline,
            buffer_minutes=deadline_buffer,
            ignore=ignore_deadline,
        )
        batch = samples[offset : offset + workers]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, sample) for sample in batch]
            for future in as_completed(futures):
                task_id, result = future.result()
                results[task_id] = result
                newly_completed[task_id] = result
                _save_result_checkpoint(
                    result_path,
                    manifest=manifest,
                    setting=setting,
                    model=model,
                    task_ids=task_ids,
                    results=results,
                )

    return {"path": result_path, "new_results": newly_completed}


def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config, manifest, run_dir, manifest_path = _create_or_load_context(args)
    _configure_accounting(run_dir, config)
    _ensure_credentials()
    _ensure_live_dependencies(construction=False)
    paths = _artifact_paths(run_dir)
    selected_ids = selected_task_ids(manifest)
    stage3 = inspect_stage(paths["stage3"], selected_ids, "stage3")
    eligible_ids = [
        task_id
        for task_id in selected_ids
        if task_id not in set(stage3["missing_task_ids"] + stage3["invalid_task_ids"])
    ]
    if not eligible_ids:
        raise WorkflowError("no eligible Stage 3 samples; run construct first")
    if len(eligible_ids) != len(selected_ids) and not args.allow_partial:
        raise WorkflowError("Stage 3 coverage is incomplete; rerun construct or pass --allow-partial")
    manifest["dataset"]["eligible_task_ids"] = eligible_ids

    scheduler_coverage: dict[str, list[str]] = {}
    common_ids = set(eligible_ids)
    for name in PLAN_A_SETTING_NAMES:
        setting = _setting_by_name(manifest, name)
        schedulable = build_setting_samples(
            paths["stage3"],
            eligible_ids,
            setting,
            seed=int(manifest["seed"]),
            require_all=False,
        )
        setting_ids = [sample.task_id for sample in schedulable]
        scheduler_coverage[name] = setting_ids
        common_ids &= set(setting_ids)
    paired_ids = [task_id for task_id in eligible_ids if task_id in common_ids]
    manifest["coverage"]["scheduler"] = {
        "by_setting": scheduler_coverage,
        "paired_task_ids": paired_ids,
        "paired_count": len(paired_ids),
    }
    if len(paired_ids) != len(selected_ids) and not args.allow_partial:
        raise WorkflowError(
            f"only {len(paired_ids)}/{len(selected_ids)} samples support all four settings; "
            "rerun construction or pass --allow-partial"
        )
    eligible_ids = paired_ids
    manifest["dataset"]["eligible_task_ids"] = eligible_ids

    deadline = parse_deadline(manifest.get("deadline"))
    workers = args.workers or int(config["evaluation"]["workers"])
    if workers < 1:
        raise WorkflowError("workers must be at least 1")
    model = manifest["models"]["evaluation"]
    requested_settings = args.setting or list(PLAN_A_SETTING_NAMES)
    paired_first = bool(config["evaluation"].get("paired_first", False))
    task_batches: list[Sequence[str] | None]
    if paired_first and len(requested_settings) > 1:
        task_batches = [
            eligible_ids[offset : offset + workers]
            for offset in range(0, len(eligible_ids), workers)
        ]
    else:
        task_batches = [None]

    for task_batch in task_batches:
        for name in requested_settings:
            setting = _setting_by_name(manifest, name)
            outcome = _run_evaluation_setting(
                config,
                manifest,
                run_dir,
                setting,
                task_ids=eligible_ids,
                model=model,
                workers=workers,
                retry_failed=args.retry_failed,
                deadline=deadline,
                deadline_buffer=int(config["deadline"]["guard_minutes"]),
                ignore_deadline=args.ignore_deadline,
                only_task_ids=task_batch,
            )
            path = outcome["path"]
            manifest["artifacts"][f"result_{name}"] = {
                "path": _rel(path),
                "sha256": file_sha256(path),
            }
            _save_manifest(manifest_path, manifest, "evaluating")

    checkpoints = {
        name: unwrap_results(read_json(path)) if path.exists() else {}
        for name, path in iter_result_files(run_dir)
    }
    attempted = all(
        set(results) == set(eligible_ids) for results in checkpoints.values()
    )
    successful = attempted and all(
        bool(result.get("success", True)) and result.get("error") is None
        for results in checkpoints.values()
        for result in results.values()
    )
    status = (
        "evaluated"
        if successful
        else "evaluated_with_failures"
        if attempted
        else "evaluating"
    )
    _save_manifest(manifest_path, manifest, status)
    return manifest, run_dir


def run_aggregation(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    _, manifest, run_dir, manifest_path = _create_or_load_context(args)
    paths = _artifact_paths(run_dir)
    payloads = {
        name: read_json(path) if path.exists() else {}
        for name, path in iter_result_files(run_dir)
    }
    summary = aggregate_results(manifest, payloads, ledger_path=paths["ledger"])
    atomic_write_json(paths["summary"], summary)
    write_summary_csv(summary, paths["summary_csv"])
    write_summary_html(summary, paths["summary_html"])
    manifest["artifacts"]["summary"] = {
        "path": _rel(paths["summary"]),
        "sha256": file_sha256(paths["summary"]),
    }
    manifest["artifacts"]["summary_html"] = {
        "path": _rel(paths["summary_html"]),
        "sha256": file_sha256(paths["summary_html"]),
    }
    all_complete = all(
        summary["settings"][name]["completed"]
        == summary["settings"][name]["requested"]
        for name in PLAN_A_SETTING_NAMES
    )
    _save_manifest(manifest_path, manifest, "complete" if all_complete else "partial")
    print(f"Summary: {paths['summary_html']}")
    return summary, run_dir


def run_dry_run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config, manifest, run_dir, manifest_path = _create_or_load_context(args, dry_run=True)
    paths = _artifact_paths(run_dir)
    python = "python3"
    common = ["--run-id", manifest["run_id"], "--config", _rel(Path(args.config))]
    planned = {
        "construct": [python, "-m", "reproduction.plan_a", "construct", *common],
        "evaluate": [python, "-m", "reproduction.plan_a", "evaluate", *common],
        "aggregate": [python, "-m", "reproduction.plan_a", "aggregate", *common],
    }
    report = {
        "schema_version": 1,
        "status": "ok",
        "offline": True,
        "network_calls": 0,
        "model_calls": 0,
        "selected_count": manifest["dataset"]["requested_count"],
        "selected_task_ids": selected_task_ids(manifest),
        "settings": [setting["name"] for setting in manifest["settings"]],
        "planned_commands": planned,
        "checks": {
            "config_valid": True,
            "published_ids_explicit": True,
            "repeat_control_is_exact_copy": True,
            "commands_contain_secrets": contains_secret_material(planned),
        },
    }
    if report["checks"]["commands_contain_secrets"]:
        raise WorkflowError("secret-like material found in planned commands")
    report_path = run_dir / "dry_run_report.json"
    atomic_write_json(report_path, report)
    manifest["artifacts"]["dry_run_report"] = {
        "path": _rel(report_path),
        "sha256": file_sha256(report_path),
    }
    manifest["dry_run"] = report
    _save_manifest(manifest_path, manifest, "dry_run")
    print(f"Offline dry-run passed: {report_path}")
    return report, run_dir


def run_inspect(args: argparse.Namespace) -> None:
    _, manifest, run_dir, _ = _create_or_load_context(args)
    paths = _artifact_paths(run_dir)
    ids = selected_task_ids(manifest)
    report = {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "sample_count": len(ids),
        "stages": {
            stage: inspect_stage(paths[stage], ids, stage)
            for stage in ("stage1", "stage2", "stage3")
        },
        "results": {
            name: (
                len(unwrap_results(read_json(path))) if path.exists() else 0
            )
            for name, path in iter_result_files(run_dir)
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--sample-count", type=int, choices=[10, 20], default=None)
    parser.add_argument("--auto-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--construction-model", default=None)
    parser.add_argument("--evaluation-model", default=None)
    parser.add_argument("--deadline", default=None, help="ISO 8601 cutoff with UTC offset")
    parser.add_argument("--ignore-deadline", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan A GSM8K reproduction workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser("dry-run", help="offline validation; no network or model calls")
    add_common_arguments(dry)

    construct = subparsers.add_parser("construct", help="run resumable construction stages")
    add_common_arguments(construct)
    construct.add_argument("--workers", type=int, default=None)
    construct.add_argument("--allow-partial", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="run the four evaluation settings")
    add_common_arguments(evaluate)
    evaluate.add_argument("--workers", type=int, default=None)
    evaluate.add_argument("--allow-partial", action="store_true")
    evaluate.add_argument("--retry-failed", action="store_true")
    evaluate.add_argument("--setting", action="append", choices=PLAN_A_SETTING_NAMES)

    aggregate = subparsers.add_parser("aggregate", help="write JSON, CSV, and HTML summaries")
    add_common_arguments(aggregate)

    all_parser = subparsers.add_parser("all", help="construct, evaluate, and aggregate")
    add_common_arguments(all_parser)
    all_parser.add_argument("--workers", type=int, default=None)
    all_parser.add_argument("--allow-partial", action="store_true")
    all_parser.add_argument("--retry-failed", action="store_true")
    all_parser.add_argument("--setting", action="append", choices=PLAN_A_SETTING_NAMES)

    inspect_parser = subparsers.add_parser("inspect", help="show resumable progress")
    add_common_arguments(inspect_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            run_dry_run(args)
        elif args.command == "construct":
            run_construction(args)
        elif args.command == "evaluate":
            run_evaluation(args)
        elif args.command == "aggregate":
            run_aggregation(args)
        elif args.command == "inspect":
            run_inspect(args)
        elif args.command == "all":
            run_construction(args)
            run_evaluation(args)
            run_aggregation(args)
        else:  # pragma: no cover
            raise WorkflowError(f"unsupported command: {args.command}")
        return 0
    except DeadlineReached as exc:
        print(f"Stopped by deadline guard: {exc}", file=sys.stderr)
        return 3
    except WorkflowError as exc:
        print(f"Workflow error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
