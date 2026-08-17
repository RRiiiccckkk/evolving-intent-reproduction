#!/usr/bin/env python3
"""Low-disk orchestrator for the fixed 100-task BIRD construction run.

Only one SQLite database is resident at a time.  Every task is checkpointed
after each completed construction stage; a database is evicted only after all
of its published tasks have reached the predecessor stage.
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from intent_construction.intent_extraction.dataset_impl.bird_sql.download_required import (
    RequiredFile,
    build_selected_file,
    download_required_files,
    evict_required_database,
    required_files,
)
from intent_construction.intent_extraction.dataset_impl.bird_sql.extractor import (
    BirdSqlExtractor,
)
from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import (
    DEFAULT_EVAL_MANIFEST_PATH,
    DEFAULT_TASK_IDS_PATH,
    REQUIRED_MODEL,
    REPO_ROOT,
    BirdReproductionError,
    TaskCheckpoint,
    acquire_exclusive_run_lock,
    assert_required_model,
    atomic_write_json,
    checkpoint_path_for,
    load_published_manifest,
    load_published_task_ids,
    validate_stage_rows,
)
from intent_construction.retrospective_expansion.counterfactual.generate_counterfactuals_sql import (
    SQLCounterfactualGenerator,
    _validate_completed_sample,
)
from intent_construction.retrospective_expansion.predecessor.generate_predecessors_sql_llm import (
    SQLPredecessorGeneratorLLM,
)


DEFAULT_RUN_DIR = REPO_ROOT / "reproduction" / "runs" / "bird-sql-kimi-k2.6"
MAX_TEMP_BYTES = 8 * 1024**3


def _ordered_database_groups(
    manifest: list[dict[str, Any]],
) -> list[tuple[str, str, list[str]]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []
    for row in manifest:
        split = row["task_id"].split("_")[2]
        key = (split, row["db_id"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row["task_id"])
    return [(split, db_id, grouped[(split, db_id)]) for split, db_id in order]


def _checkpoint_with_output_seed(
    *,
    path: Path,
    output_path: Path,
    stage: str,
    required_ids: list[str],
    resume: bool,
    min_predecessors: int = 0,
) -> TaskCheckpoint:
    existed = path.exists()
    checkpoint = TaskCheckpoint(
        path,
        stage=stage,
        required_ids=required_ids,
        model=REQUIRED_MODEL,
        resume=resume,
    )
    if resume and not existed and output_path.exists():
        from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import read_json

        rows = validate_stage_rows(
            read_json(output_path),
            stage=f"{stage}_resume_seed",
            required_ids=required_ids,
            require_model=True,
            min_predecessors=min_predecessors,
        )
        for row in rows:
            checkpoint.record_success(row)
        checkpoint.mark_complete()
    return checkpoint


def _run_checkpointed_stage(
    samples: list[dict[str, Any]],
    *,
    checkpoint: TaskCheckpoint,
    process: Callable[[dict[str, Any]], dict[str, Any]],
    workers: int,
    description: str,
) -> None:
    pending = [
        sample for sample in samples if sample["task_id"] in set(checkpoint.pending_ids)
    ]
    if not pending:
        return

    if workers <= 1:
        for sample in tqdm(pending, desc=description):
            try:
                checkpoint.record_success(process(sample))
            except BaseException as exc:
                checkpoint.record_failure(sample["task_id"], exc)
                raise
        return

    executor = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future, dict[str, Any]] = {
        executor.submit(process, sample): sample for sample in pending
    }
    first_failure: BaseException | None = None
    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            sample = futures[future]
            try:
                result = future.result()
            except CancelledError:
                continue
            except BaseException as exc:
                checkpoint.record_failure(sample["task_id"], exc)
                if first_failure is None:
                    first_failure = exc
                    for other in futures:
                        if other is not future:
                            other.cancel()
                continue
            checkpoint.record_success(result)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if first_failure is not None:
        raise first_failure


def _cache_database_ids(data_dir: Path) -> set[str]:
    ids = {path.parent.name for path in data_dir.rglob("*.sqlite")}
    ids.update(path.parent.name for path in data_dir.rglob("*.sqlite.part"))
    return ids


def _validate_predecessor_result(
    result: object,
    *,
    task_id: str,
    expected_count: int,
) -> dict[str, Any]:
    if not isinstance(result, dict) or result.get("task_id") != task_id:
        raise BirdReproductionError(
            f"{task_id}: predecessor result is empty or misidentified"
        )
    predecessors = result.get("predecessor_functions")
    if not isinstance(predecessors, list) or len(predecessors) != expected_count:
        raise BirdReproductionError(f"{task_id}: incomplete predecessor result")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=REQUIRED_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--counterfactuals", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--task-ids-file", default=str(DEFAULT_TASK_IDS_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_EVAL_MANIFEST_PATH))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--database",
        default=None,
        help="Process one published database, then exit with partial checkpoints.",
    )
    parser.add_argument(
        "--extracted-output",
        default=str(REPO_ROOT / "intent_construction/intent_extraction/output/bird_sql/extracted.json"),
    )
    parser.add_argument(
        "--counterfactual-output",
        default=str(
            REPO_ROOT
            / "intent_construction/retrospective_expansion/counterfactual/output/bird_sql/argument_counterfactual.json"
        ),
    )
    parser.add_argument(
        "--predecessor-output",
        default=str(
            REPO_ROOT
            / "intent_construction/retrospective_expansion/predecessor/output/bird_sql/predecessor.json"
        ),
    )
    parser.add_argument(
        "--final-output",
        default=str(REPO_ROOT / "final_dataset/bird_sql_final.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    assert_required_model(args.model, context="BIRD low-disk construction model")
    if args.counterfactuals < 2:
        raise BirdReproductionError("the paper's g=2/p=2 setting needs at least two variants")
    if args.workers < 1:
        raise BirdReproductionError("workers must be positive")

    run_dir = Path(args.run_dir).resolve()
    _run_lock = acquire_exclusive_run_lock(run_dir)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else run_dir / "bird_data"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    selected_path = run_dir / "selected_published_n100.json"
    extracted_output = Path(args.extracted_output)
    counterfactual_output = Path(args.counterfactual_output)
    predecessor_output = Path(args.predecessor_output)
    final_output = Path(args.final_output)

    required_ids = load_published_task_ids(args.task_ids_file)
    manifest = load_published_manifest(
        args.manifest,
        task_ids_path=args.task_ids_file,
    )
    specs = required_files(manifest, data_dir=data_dir)
    json_specs = [spec for spec in specs if spec.kind == "json"]
    db_specs = {
        (spec.split, spec.db_id): spec
        for spec in specs
        if spec.kind == "sqlite" and spec.db_id is not None
    }
    download_required_files(
        json_specs,
        chunk_size=8 * 1024 * 1024,
        max_file_bytes=MAX_TEMP_BYTES,
    )
    selected = build_selected_file(
        manifest,
        data_dir=data_dir,
        output_path=selected_path,
    )
    if [sample["task_id"] for sample in selected] != required_ids:
        raise BirdReproductionError("selected samples differ from published task order")
    selected_by_id = {sample["task_id"]: sample for sample in selected}

    extraction_cp = _checkpoint_with_output_seed(
        path=checkpoint_path_for(extracted_output),
        output_path=extracted_output,
        stage="bird_extraction",
        required_ids=required_ids,
        resume=args.resume,
    )
    counterfactual_cp = _checkpoint_with_output_seed(
        path=checkpoint_path_for(counterfactual_output),
        output_path=counterfactual_output,
        stage="bird_counterfactual",
        required_ids=required_ids,
        resume=args.resume,
    )
    predecessor_cp = _checkpoint_with_output_seed(
        path=checkpoint_path_for(predecessor_output),
        output_path=predecessor_output,
        stage="bird_predecessor",
        required_ids=required_ids,
        resume=args.resume,
        min_predecessors=args.counterfactuals,
    )
    if not set(predecessor_cp.processed_ids).issubset(counterfactual_cp.processed_ids):
        raise BirdReproductionError("predecessor checkpoint is ahead of counterfactual checkpoint")
    if not set(counterfactual_cp.processed_ids).issubset(extraction_cp.processed_ids):
        raise BirdReproductionError("counterfactual checkpoint is ahead of extraction checkpoint")

    extractor = BirdSqlExtractor(
        model=args.model,
        verif_model=args.model,
        strip_model=args.model,
        data_root=data_dir,
    )
    counterfactual_generator = SQLCounterfactualGenerator(
        num_counterfactuals=args.counterfactuals,
        sql_timeout=30,
    )
    predecessor_generator = SQLPredecessorGeneratorLLM(
        num_predecessors=args.counterfactuals,
        max_attempts=args.max_attempts,
        model=args.model,
        naturalizer_model=args.model,
        temperature=None,
        reasoning_effort=None,
        sql_timeout=30,
        schema_max_tables=12,
    )

    groups = _ordered_database_groups(manifest)
    if args.database is not None:
        groups = [group for group in groups if group[1] == args.database]
        if len(groups) != 1:
            raise BirdReproductionError(
                f"{args.database!r} is not a unique published BIRD database"
            )
    cached = _cache_database_ids(data_dir)
    if len(cached) > 1:
        raise BirdReproductionError(
            f"low-disk cache contains more than one database: {sorted(cached)}"
        )
    if cached:
        cached_id = next(iter(cached))
        groups.sort(key=lambda group: 0 if group[1] == cached_id else 1)

    for group_index, (split, db_id, group_ids) in enumerate(groups, start=1):
        group_samples = [selected_by_id[task_id] for task_id in group_ids]
        remaining = set(group_ids) & (
            set(extraction_cp.pending_ids)
            | set(counterfactual_cp.pending_ids)
            | set(predecessor_cp.pending_ids)
        )
        spec = db_specs[(split, db_id)]
        if not remaining:
            evict_required_database(spec, data_dir=data_dir)
            continue

        resident = _cache_database_ids(data_dir)
        if resident and resident != {db_id}:
            raise BirdReproductionError(
                f"refusing to load {db_id}; cache still contains {sorted(resident)}"
            )
        print(
            f"[DB {group_index}/{len(groups)}] {split}/{db_id}: "
            f"{len(group_ids)} published tasks"
        )
        download_required_files(
            [spec],
            chunk_size=8 * 1024 * 1024,
            max_file_bytes=MAX_TEMP_BYTES,
        )

        def extract_one(sample: dict[str, Any]) -> dict[str, Any]:
            result = extractor.extract(sample)
            if not isinstance(result, dict) or result.get("task_id") != sample["task_id"]:
                raise BirdReproductionError(
                    f"{sample['task_id']}: incomplete extraction result"
                )
            return result

        _run_checkpointed_stage(
            group_samples,
            checkpoint=extraction_cp,
            process=extract_one,
            workers=args.workers,
            description=f"extract:{db_id}",
        )

        extraction_rows = extraction_cp.results_by_id

        def counterfactual_one(sample: dict[str, Any]) -> dict[str, Any]:
            task_id = sample["task_id"]
            random.seed(f"42:{task_id}")
            return _validate_completed_sample(
                counterfactual_generator.generate_counterfactuals(
                    extraction_rows[task_id]
                ),
                minimum=args.counterfactuals,
            )

        _run_checkpointed_stage(
            group_samples,
            checkpoint=counterfactual_cp,
            process=counterfactual_one,
            workers=1,
            description=f"counterfactual:{db_id}",
        )

        counterfactual_rows = counterfactual_cp.results_by_id

        def predecessor_one(sample: dict[str, Any]) -> dict[str, Any]:
            task_id = sample["task_id"]
            return _validate_predecessor_result(
                predecessor_generator.generate_predecessors(
                    counterfactual_rows[task_id]
                ),
                task_id=task_id,
                expected_count=args.counterfactuals,
            )

        _run_checkpointed_stage(
            group_samples,
            checkpoint=predecessor_cp,
            process=predecessor_one,
            workers=args.workers,
            description=f"predecessor:{db_id}",
        )
        if not set(group_ids).issubset(predecessor_cp.processed_ids):
            raise BirdReproductionError(
                f"refusing to evict {db_id}; its predecessor tasks are incomplete"
            )
        evict_required_database(spec, data_dir=data_dir)

    if args.database is not None:
        print(
            f"BIRD construction smoke complete for {args.database}: "
            f"{len(predecessor_cp.processed_ids)}/{len(required_ids)} tasks checkpointed; "
            "final output was not written"
        )
        return

    extracted_rows = validate_stage_rows(
        extraction_cp.results,
        stage="bird_extraction",
        required_ids=required_ids,
        require_model=True,
    )
    counterfactual_rows_final = validate_stage_rows(
        counterfactual_cp.results,
        stage="bird_counterfactual",
        required_ids=required_ids,
        require_model=True,
    )
    predecessor_rows = validate_stage_rows(
        predecessor_cp.results,
        stage="bird_predecessor",
        required_ids=required_ids,
        require_model=True,
        min_predecessors=args.counterfactuals,
    )
    extraction_cp.mark_complete()
    counterfactual_cp.mark_complete()
    predecessor_cp.mark_complete()
    atomic_write_json(extracted_output, extracted_rows)
    atomic_write_json(counterfactual_output, counterfactual_rows_final)
    atomic_write_json(predecessor_output, predecessor_rows)
    atomic_write_json(final_output, predecessor_rows)
    assert_required_model(args.model, context="completed low-disk construction model")
    print(f"BIRD construction complete: {len(predecessor_rows)}/100 -> {final_output}")


if __name__ == "__main__":
    main()
