#!/usr/bin/env python3
"""Build the fixed 50-task SWE dataset from the local Verified parquet."""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .reproduction import (
    COUNTERFACTUALS_PER_ARGUMENT,
    DEFAULT_EVAL_MANIFEST_PATH,
    DEFAULT_FINAL_OUTPUT,
    DEFAULT_RUN_DIR,
    DEFAULT_SOURCE_PARQUET,
    DEFAULT_TASK_IDS_PATH,
    ELIGIBILITY_REPAIR_PROMPT_SHA256,
    ELIGIBILITY_REPAIR_VERIFICATION_PROMPT_SHA256,
    PUBLISHED_TASK_COUNT,
    REQUIRED_MODEL,
    REQUIRED_REASONING_EFFORT,
    SOURCE_PARQUET_SHA256,
    SWEConstructionError,
    StageCheckpointStore,
    assert_build_policy,
    atomic_write_json,
    load_published_manifest,
    load_source_pool,
    read_json,
    select_build_rows,
    source_row_to_sample,
    task_id_for,
    validate_counterfactual_row,
    validate_eligibility_repair_delta,
    validate_exact_rows,
    validate_extraction_row,
    validate_final_row,
    validate_g1_row,
    validate_pair_row,
)
from intent_construction.retrospective_expansion.predecessor.pair_swe_bugs import (
    attach_pair,
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class StageProcessors:
    extract: Callable[[dict[str, Any]], dict[str, Any]]
    counterfactual: Callable[[dict[str, Any]], dict[str, Any]]
    g1: Callable[[dict[str, Any], str], dict[str, Any]]
    impl: Callable[[dict[str, Any]], dict[str, Any]]
    repair_eligibility: (
        Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None
    ) = None


def _default_processors() -> StageProcessors:
    from .extractor import SWEBenchVerifiedExtractor
    from intent_construction.intent_extraction.core.llm_utils import load_prompt
    from intent_construction.retrospective_expansion.counterfactual.generate_counterfactuals_swe import (
        COUNTERFACTUAL_ELIGIBLE_CATEGORIES,
        PROMPT_FILES,
        generate_counterfactuals,
    )
    from intent_construction.retrospective_expansion.predecessor.generate_g1_swe import (
        PROMPT_PATH as G1_PROMPT_PATH,
        generate_g1_for_sample,
    )
    from intent_construction.retrospective_expansion.predecessor.generate_impl_precursors_swe import (
        PROMPT_PATH as IMPL_PROMPT_PATH,
        generate_impl_precursor_for_sample,
    )

    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    counterfactual_templates = {
        category: load_prompt(str(PROMPT_FILES[category]))
        for category in COUNTERFACTUAL_ELIGIBLE_CATEGORIES
    }
    g1_template = load_prompt(str(G1_PROMPT_PATH))
    impl_template = load_prompt(str(IMPL_PROMPT_PATH))

    def extract(sample: dict[str, Any]) -> dict[str, Any]:
        result = extractor.extract(sample)
        if not isinstance(result, dict):
            raise SWEConstructionError(
                f"stage1 returned no result for {task_id_for(sample['instance_id'])}"
            )
        return result

    def counterfactual(sample: dict[str, Any]) -> dict[str, Any]:
        return generate_counterfactuals(
            sample,
            counterfactual_templates,
            REQUIRED_MODEL,
            COUNTERFACTUALS_PER_ARGUMENT,
            reasoning_effort=REQUIRED_REASONING_EFFORT,
        )

    def g1(sample: dict[str, Any], archetype: str) -> dict[str, Any]:
        return generate_g1_for_sample(
            sample,
            template=g1_template,
            model=REQUIRED_MODEL,
            archetype=archetype,
            reasoning_effort=REQUIRED_REASONING_EFFORT,
        )

    def impl(sample: dict[str, Any]) -> dict[str, Any]:
        return generate_impl_precursor_for_sample(
            sample,
            template=impl_template,
            model=REQUIRED_MODEL,
            reasoning_effort=REQUIRED_REASONING_EFFORT,
        )

    return StageProcessors(
        extract=extract,
        counterfactual=counterfactual,
        g1=g1,
        impl=impl,
        repair_eligibility=extractor.ensure_counterfactual_eligibility,
    )


def _run_checkpointed_stage(
    samples: list[dict[str, Any]],
    *,
    store: StageCheckpointStore,
    process: Callable[[dict[str, Any]], dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], int]:
    completed = store.load_available()
    pending = [sample for sample in samples if sample["task_id"] not in completed]
    if not pending:
        return store.require_complete(), 0

    if workers <= 1:
        for sample in pending:
            store.write(sample["task_id"], process(sample))
        return store.require_complete(), len(pending)

    executor = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
        executor.submit(process, sample): sample for sample in pending
    }
    first_failure: BaseException | None = None
    try:
        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                store.write(sample["task_id"], result)
            except CancelledError:
                continue
            except BaseException as exc:
                if first_failure is None:
                    first_failure = exc
                    for other in futures:
                        if other is not future:
                            other.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if first_failure is not None:
        raise SWEConstructionError(
            f"{store.stage} failed; valid completed task checkpoints were preserved"
        ) from first_failure
    return store.require_complete(), len(pending)


def _configure_live_environment(run_dir: Path) -> str:
    assert_build_policy(
        model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
        environment=os.environ,
        require_credentials=False,
    )
    os.environ["LLM_REASONING_EFFORT"] = REQUIRED_REASONING_EFFORT
    os.environ["LLM_DISABLE_OUTPUT_LIMITS"] = "1"
    os.environ["LLM_REQUIRE_USAGE_ACCOUNTING"] = "1"
    os.environ["LLM_USAGE_LEDGER_PATH"] = str((run_dir / "usage.jsonl").resolve())
    return assert_build_policy(
        model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
        environment=os.environ,
        require_credentials=True,
    )


def _write_plan(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() and read_json(path, label="SWE construction plan") != payload:
        raise SWEConstructionError(
            f"existing construction plan differs from this run: {path}"
        )
    atomic_write_json(path, payload)


def build_published_dataset(
    *,
    source_parquet: str | Path,
    run_dir: str | Path,
    final_output: str | Path,
    manifest_path: str | Path = DEFAULT_EVAL_MANIFEST_PATH,
    task_ids_path: str | Path = DEFAULT_TASK_IDS_PATH,
    workers: int = 4,
    seed: int = 42,
    expected_source_sha256: str | None = SOURCE_PARQUET_SHA256,
    plan_only: bool = False,
    processors: StageProcessors | None = None,
    require_live_environment: bool = True,
) -> dict[str, Any]:
    if workers < 1:
        raise SWEConstructionError("workers must be positive")
    run_path = Path(run_dir).resolve()
    final_path = Path(final_output).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    published = load_published_manifest(manifest_path, task_ids_path)
    source_rows, source_sha = load_source_pool(
        source_parquet, expected_sha256=expected_source_sha256
    )
    selection = select_build_rows(source_rows, published)
    selection_payload = selection.manifest(
        source_path=source_parquet,
        source_sha256=source_sha,
    )
    _write_plan(run_path / "selection.json", selection_payload)
    atomic_write_json(
        run_path / "selected_sources.json",
        [dict(row) for row in selection.extraction_rows],
    )
    if plan_only:
        return {
            "status": "planned",
            "run_dir": str(run_path),
            "target_count": PUBLISHED_TASK_COUNT,
            "extra_candidate_count": len(selection.extra_rows),
            "extraction_count": len(selection.extraction_rows),
        }

    resolved_model = REQUIRED_MODEL
    if require_live_environment:
        resolved_model = _configure_live_environment(run_path)
    else:
        assert_build_policy(
            model=REQUIRED_MODEL,
            reasoning_effort=REQUIRED_REASONING_EFFORT,
            environment={},
            require_credentials=False,
        )
    active_processors = processors or _default_processors()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "model": REQUIRED_MODEL,
        "resolved_model": resolved_model,
        "reasoning_effort": REQUIRED_REASONING_EFFORT,
        "output_token_limit": None,
        "cost_hard_cap_usd": None,
        "selection": selection_payload,
        "stages": {},
        "final_output": str(final_path),
    }
    atomic_write_json(run_path / "manifest.json", manifest)

    extraction_samples = [source_row_to_sample(row) for row in selection.extraction_rows]
    extraction_inputs = {
        task_id_for(sample["instance_id"]): sample for sample in extraction_samples
    }
    for sample in extraction_samples:
        sample["task_id"] = task_id_for(sample["instance_id"])
    stage1_store = StageCheckpointStore(
        run_path / "checkpoints/stage1_extraction",
        stage="stage1_extraction",
        inputs_by_id=extraction_inputs,
        validator=validate_extraction_row,
    )
    stage1_rows, executed = _run_checkpointed_stage(
        extraction_samples,
        store=stage1_store,
        process=active_processors.extract,
        workers=workers,
    )
    validate_exact_rows(
        stage1_rows,
        expected_task_ids=selection.extraction_task_ids,
        stage="stage1_extraction_pool",
        validator=validate_extraction_row,
    )
    by_stage1 = {row["task_id"]: row for row in stage1_rows}
    stage1_targets = [by_stage1[task_id] for task_id in selection.target_task_ids]
    validate_exact_rows(
        stage1_targets,
        expected_task_ids=selection.target_task_ids,
        stage="stage1_published_targets",
        validator=validate_extraction_row,
    )
    atomic_write_json(run_path / "artifacts/stage1_extraction_pool.json", stage1_rows)
    atomic_write_json(run_path / "artifacts/stage1_targets.json", stage1_targets)
    manifest["stages"]["stage1"] = {
        "coverage": len(stage1_targets),
        "extracted_with_candidates": len(stage1_rows),
        "executed": executed,
    }
    atomic_write_json(run_path / "manifest.json", manifest)

    stage1b_inputs = {row["task_id"]: row for row in stage1_targets}

    def validate_stage1b(row: Any, *, task_id: str) -> dict[str, Any]:
        return validate_eligibility_repair_delta(
            row,
            task_id=task_id,
            source_row=stage1b_inputs[task_id],
        )

    stage1b_store = StageCheckpointStore(
        run_path / "checkpoints/stage1b_eligible_argument_v1",
        stage="stage1b_eligible_argument_v1",
        inputs_by_id=stage1b_inputs,
        validator=validate_stage1b,
    )
    repair_eligibility = active_processors.repair_eligibility

    def process_stage1b(sample: dict[str, Any]) -> dict[str, Any]:
        if repair_eligibility is None:
            return sample
        return repair_eligibility(sample, extraction_inputs[sample["task_id"]])

    stage1b_rows, executed = _run_checkpointed_stage(
        stage1_targets,
        store=stage1b_store,
        process=process_stage1b,
        workers=workers,
    )
    stage1b_rows = validate_exact_rows(
        stage1b_rows,
        expected_task_ids=selection.target_task_ids,
        stage="stage1b_eligible_argument_v1",
        validator=validate_stage1b,
    )
    atomic_write_json(
        run_path / "artifacts/stage1b_eligible_targets.json", stage1b_rows
    )
    repaired_ids = [
        row["task_id"] for row in stage1b_rows if "eligibility_repair_info" in row
    ]
    manifest["stages"]["stage1b"] = {
        "coverage": len(stage1b_rows),
        "executed": executed,
        "repaired_count": len(repaired_ids),
        "repaired_task_ids": repaired_ids,
        "policy": {
            "model": REQUIRED_MODEL,
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
            "only_when_no_eligible_argument": True,
            "prompt_sha256": ELIGIBILITY_REPAIR_PROMPT_SHA256,
            "verification_prompt_sha256": (
                ELIGIBILITY_REPAIR_VERIFICATION_PROMPT_SHA256
            ),
            "repair_version": "eligible_argument_v1",
        },
    }
    atomic_write_json(run_path / "manifest.json", manifest)

    stage2_inputs = {row["task_id"]: row for row in stage1b_rows}
    stage2_store = StageCheckpointStore(
        run_path / "checkpoints/stage2_counterfactual",
        stage="stage2_counterfactual",
        inputs_by_id=stage2_inputs,
        validator=validate_counterfactual_row,
    )
    stage2_rows, executed = _run_checkpointed_stage(
        stage1b_rows,
        store=stage2_store,
        process=active_processors.counterfactual,
        workers=workers,
    )
    stage2_rows = validate_exact_rows(
        stage2_rows,
        expected_task_ids=selection.target_task_ids,
        stage="stage2_counterfactual",
        validator=validate_counterfactual_row,
    )
    atomic_write_json(run_path / "artifacts/stage2_counterfactual.json", stage2_rows)
    manifest["stages"]["stage2"] = {"coverage": len(stage2_rows), "executed": executed}
    atomic_write_json(run_path / "manifest.json", manifest)

    pairing_by_target = {pairing["target_task_id"]: pairing for pairing in selection.pairings}
    stage3_rows: list[dict[str, Any]] = []
    for target in stage2_rows:
        pairing = pairing_by_target[target["task_id"]]
        paired = by_stage1[pairing["paired_task_id"]]
        output = attach_pair(target, paired, pairing["match_level"])
        stage3_rows.append(
            validate_pair_row(output, task_id=target["task_id"], pairing=pairing)
        )
    if [row["task_id"] for row in stage3_rows] != selection.target_task_ids:
        raise SWEConstructionError("stage3 does not preserve exact published order")
    atomic_write_json(run_path / "artifacts/stage3_real_bug_pairs.json", stage3_rows)
    manifest["stages"]["stage3"] = {"coverage": len(stage3_rows), "executed": len(stage3_rows)}
    atomic_write_json(run_path / "manifest.json", manifest)

    from intent_construction.retrospective_expansion.predecessor.generate_g1_swe import (
        ARCHETYPES,
    )

    indices = list(range(PUBLISHED_TASK_COUNT))
    random.Random(seed).shuffle(indices)
    archetypes = {
        stage3_rows[index]["task_id"]: ARCHETYPES[position % len(ARCHETYPES)]
        for position, index in enumerate(indices)
    }
    stage4_inputs = {row["task_id"]: row for row in stage3_rows}
    stage4_store = StageCheckpointStore(
        run_path / "checkpoints/stage4_g1",
        stage="stage4_g1",
        inputs_by_id=stage4_inputs,
        validator=validate_g1_row,
    )
    stage4_rows, executed = _run_checkpointed_stage(
        stage3_rows,
        store=stage4_store,
        process=lambda sample: active_processors.g1(sample, archetypes[sample["task_id"]]),
        workers=workers,
    )
    stage4_rows = validate_exact_rows(
        stage4_rows,
        expected_task_ids=selection.target_task_ids,
        stage="stage4_g1",
        validator=validate_g1_row,
    )
    atomic_write_json(run_path / "artifacts/stage4_g1.json", stage4_rows)
    manifest["stages"]["stage4"] = {"coverage": len(stage4_rows), "executed": executed}
    atomic_write_json(run_path / "manifest.json", manifest)

    stage5_inputs = {row["task_id"]: row for row in stage4_rows}
    stage5_store = StageCheckpointStore(
        run_path / "checkpoints/stage5_impl_precursor",
        stage="stage5_impl_precursor",
        inputs_by_id=stage5_inputs,
        validator=validate_final_row,
    )
    stage5_rows, executed = _run_checkpointed_stage(
        stage4_rows,
        store=stage5_store,
        process=active_processors.impl,
        workers=workers,
    )
    final_rows = validate_exact_rows(
        stage5_rows,
        expected_task_ids=selection.target_task_ids,
        stage="stage5_final",
        validator=validate_final_row,
    )
    atomic_write_json(run_path / "artifacts/stage5_final.json", final_rows)
    atomic_write_json(final_path, final_rows)
    manifest["stages"]["stage5"] = {"coverage": len(final_rows), "executed": executed}

    if require_live_environment:
        from evaluation.swe_bench.state import read_usage_events, validate_usage_events

        events = read_usage_events(run_path / "usage.jsonl")
        if not events:
            raise SWEConstructionError("construction completed without provider usage records")
        manifest["usage"] = validate_usage_events(
            events,
            requested_model=REQUIRED_MODEL,
            resolved_model=resolved_model,
        )
    manifest["status"] = "complete"
    manifest["updated_at"] = _now()
    manifest["completed_at"] = _now()
    manifest["final_coverage"] = len(final_rows)
    atomic_write_json(run_path / "manifest.json", manifest)
    return manifest


def build_canary_dataset(
    *,
    source_parquet: str | Path,
    run_dir: str | Path,
    canary_task_id: str,
    manifest_path: str | Path = DEFAULT_EVAL_MANIFEST_PATH,
    task_ids_path: str | Path = DEFAULT_TASK_IDS_PATH,
    workers: int = 1,
    expected_source_sha256: str | None = SOURCE_PARQUET_SHA256,
    processors: StageProcessors | None = None,
    require_live_environment: bool = True,
) -> dict[str, Any]:
    """Run one target end to end in a run directory isolated from the 50-task build."""
    run_path = Path(run_dir).resolve()
    formal_run_path = Path(DEFAULT_RUN_DIR).resolve()
    if run_path == formal_run_path or formal_run_path in run_path.parents:
        raise SWEConstructionError("canary must use a run-dir separate from the full build")
    run_path.mkdir(parents=True, exist_ok=True)
    published = load_published_manifest(manifest_path, task_ids_path)
    source_rows, source_sha = load_source_pool(
        source_parquet, expected_sha256=expected_source_sha256
    )
    selection = select_build_rows(source_rows, published)
    if not canary_task_id.startswith("extracted-"):
        canary_task_id = task_id_for(canary_task_id)
    pairing = next(
        (
            item
            for item in selection.pairings
            if item["target_task_id"] == canary_task_id
        ),
        None,
    )
    if pairing is None:
        raise SWEConstructionError("canary task must be one of the published 50 IDs")
    source_by_task = {
        task_id_for(row["instance_id"]): row for row in selection.extraction_rows
    }
    selected_task_ids = [canary_task_id, pairing["paired_task_id"]]
    selected_task_ids = list(dict.fromkeys(selected_task_ids))
    samples = [source_row_to_sample(source_by_task[task_id]) for task_id in selected_task_ids]
    for sample in samples:
        sample["task_id"] = task_id_for(sample["instance_id"])

    canary_plan = {
        **selection.manifest(source_path=source_parquet, source_sha256=source_sha),
        "mode": "one_task_canary",
        "canary_target_task_id": canary_task_id,
        "canary_pairing": dict(pairing),
        "canary_extraction_task_ids": selected_task_ids,
    }
    _write_plan(run_path / "selection.json", canary_plan)
    resolved_model = REQUIRED_MODEL
    if require_live_environment:
        resolved_model = _configure_live_environment(run_path)
    active_processors = processors or _default_processors()

    stage1_inputs = {sample["task_id"]: sample for sample in samples}
    stage1_store = StageCheckpointStore(
        run_path / "checkpoints/stage1_extraction",
        stage="canary_stage1_extraction",
        inputs_by_id=stage1_inputs,
        validator=validate_extraction_row,
    )
    stage1_rows, _ = _run_checkpointed_stage(
        samples,
        store=stage1_store,
        process=active_processors.extract,
        workers=workers,
    )
    by_stage1 = {row["task_id"]: row for row in stage1_rows}
    target = by_stage1[canary_task_id]

    def validate_canary_stage1b(row: Any, *, task_id: str) -> dict[str, Any]:
        return validate_eligibility_repair_delta(
            row,
            task_id=task_id,
            source_row=target,
        )

    stage1b_store = StageCheckpointStore(
        run_path / "checkpoints/stage1b_eligible_argument_v1",
        stage="canary_stage1b_eligible_argument_v1",
        inputs_by_id={canary_task_id: target},
        validator=validate_canary_stage1b,
    )
    repair_eligibility = active_processors.repair_eligibility

    def process_canary_stage1b(sample: dict[str, Any]) -> dict[str, Any]:
        if repair_eligibility is None:
            return sample
        return repair_eligibility(sample, stage1_inputs[sample["task_id"]])

    stage1b_rows, _ = _run_checkpointed_stage(
        [target],
        store=stage1b_store,
        process=process_canary_stage1b,
        workers=1,
    )
    target = stage1b_rows[0]
    atomic_write_json(
        run_path / "artifacts/stage1b_eligible_target.json", stage1b_rows
    )

    stage2_store = StageCheckpointStore(
        run_path / "checkpoints/stage2_counterfactual",
        stage="canary_stage2_counterfactual",
        inputs_by_id={canary_task_id: target},
        validator=validate_counterfactual_row,
    )
    stage2_rows, _ = _run_checkpointed_stage(
        [target],
        store=stage2_store,
        process=active_processors.counterfactual,
        workers=1,
    )
    paired = by_stage1[pairing["paired_task_id"]]
    stage3 = validate_pair_row(
        attach_pair(stage2_rows[0], paired, pairing["match_level"]),
        task_id=canary_task_id,
        pairing=pairing,
    )
    atomic_write_json(run_path / "artifacts/stage3_real_bug_pair.json", [stage3])

    from intent_construction.retrospective_expansion.predecessor.generate_g1_swe import (
        ARCHETYPES,
    )

    stage4_store = StageCheckpointStore(
        run_path / "checkpoints/stage4_g1",
        stage="canary_stage4_g1",
        inputs_by_id={canary_task_id: stage3},
        validator=validate_g1_row,
    )
    stage4_rows, _ = _run_checkpointed_stage(
        [stage3],
        store=stage4_store,
        process=lambda sample: active_processors.g1(sample, ARCHETYPES[0]),
        workers=1,
    )
    stage5_store = StageCheckpointStore(
        run_path / "checkpoints/stage5_impl_precursor",
        stage="canary_stage5_impl_precursor",
        inputs_by_id={canary_task_id: stage4_rows[0]},
        validator=validate_final_row,
    )
    final_rows, _ = _run_checkpointed_stage(
        stage4_rows,
        store=stage5_store,
        process=active_processors.impl,
        workers=1,
    )
    output_path = run_path / "canary_final.json"
    atomic_write_json(output_path, final_rows)
    result = {
        "schema_version": 1,
        "status": "canary_complete",
        "model": REQUIRED_MODEL,
        "resolved_model": resolved_model,
        "reasoning_effort": REQUIRED_REASONING_EFFORT,
        "output_token_limit": None,
        "target_task_id": canary_task_id,
        "paired_task_id": pairing["paired_task_id"],
        "coverage": 1,
        "output": str(output_path),
        "completed_at": _now(),
    }
    if require_live_environment:
        from evaluation.swe_bench.state import read_usage_events, validate_usage_events

        events = read_usage_events(run_path / "usage.jsonl")
        if not events:
            raise SWEConstructionError("canary completed without provider usage records")
        result["usage"] = validate_usage_events(
            events,
            requested_model=REQUIRED_MODEL,
            resolved_model=resolved_model,
        )
    atomic_write_json(run_path / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--final-output", default=str(DEFAULT_FINAL_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_EVAL_MANIFEST_PATH))
    parser.add_argument("--task-ids-file", default=str(DEFAULT_TASK_IDS_PATH))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=REQUIRED_MODEL)
    parser.add_argument("--reasoning-effort", default=REQUIRED_REASONING_EFFORT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--canary-task-id",
        default=None,
        help="Build one published target plus its pair in an isolated --run-dir.",
    )
    args = parser.parse_args()
    assert_build_policy(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        environment=os.environ,
        require_credentials=False,
    )
    if args.canary_task_id:
        if args.plan_only:
            parser.error("--canary-task-id and --plan-only are mutually exclusive")
        summary = build_canary_dataset(
            source_parquet=args.source_parquet,
            run_dir=args.run_dir,
            canary_task_id=args.canary_task_id,
            manifest_path=args.manifest,
            task_ids_path=args.task_ids_file,
            workers=args.workers,
        )
    else:
        summary = build_published_dataset(
            source_parquet=args.source_parquet,
            run_dir=args.run_dir,
            final_output=args.final_output,
            manifest_path=args.manifest,
            task_ids_path=args.task_ids_file,
            workers=args.workers,
            seed=args.seed,
            plan_only=args.plan_only,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
