"""Extract the fixed 100 published BIRD-SQL samples with durable resume."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from intent_construction.intent_extraction.dataset_impl.bird_sql.extractor import (
    BirdSqlExtractor,
)
from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import (
    DEFAULT_TASK_IDS_PATH,
    REQUIRED_MODEL,
    BirdReproductionError,
    TaskCheckpoint,
    assert_required_model,
    atomic_write_json,
    checkpoint_path_for,
    load_published_task_ids,
    read_json,
    task_id_for,
    validate_stage_rows,
)


def _sample_task_id(sample: dict) -> str:
    derived = task_id_for(
        sample.get("source", "train"),
        sample.get("index", sample.get("id")),
    )
    supplied = sample.get("task_id")
    if supplied is not None and supplied != derived:
        raise BirdReproductionError(
            f"selected sample task_id mismatch: supplied={supplied!r}, derived={derived!r}"
        )
    return derived


def _ordered_selected_samples(
    samples: object, required_ids: list[str]
) -> list[dict]:
    if not isinstance(samples, list) or not all(isinstance(row, dict) for row in samples):
        raise BirdReproductionError("selected BIRD input must be a JSON list of objects")
    indexed: dict[str, dict] = {}
    for sample in samples:
        task_id = _sample_task_id(sample)
        if task_id in indexed:
            raise BirdReproductionError(f"selected input duplicates {task_id}")
        indexed[task_id] = sample
    allowed = set(required_ids)
    missing = [task_id for task_id in required_ids if task_id not in indexed]
    extras = [task_id for task_id in indexed if task_id not in allowed]
    if missing or extras:
        raise BirdReproductionError(
            f"selected input must exactly cover the published IDs; "
            f"missing={missing[:5]}, extras={extras[:5]}"
        )
    return [indexed[task_id] for task_id in required_ids]


def _process_one(extractor: BirdSqlExtractor, sample: dict) -> dict:
    result = extractor.extract(sample)
    if not isinstance(result, dict):
        raise BirdReproductionError(
            f"extractor returned an incomplete result for {_sample_task_id(sample)}"
        )
    if result.get("task_id") != _sample_task_id(sample):
        raise BirdReproductionError(
            f"extractor returned the wrong task ID for {_sample_task_id(sample)}"
        )
    return result


def _record_future(
    future: Future,
    sample: dict,
    checkpoint: TaskCheckpoint,
) -> None:
    task_id = _sample_task_id(sample)
    try:
        result = future.result()
        checkpoint.record_success(result)
    except BaseException as exc:
        checkpoint.record_failure(task_id, exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=REQUIRED_MODEL)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--task_ids_file", default=str(DEFAULT_TASK_IDS_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    assert_required_model(args.model, context="BIRD extraction CLI model")
    required_ids = load_published_task_ids(args.task_ids_file)
    if args.num_samples not in {None, len(required_ids)}:
        raise BirdReproductionError(
            "published BIRD extraction cannot truncate the fixed 100-task subset"
        )
    samples = _ordered_selected_samples(read_json(args.input), required_ids)
    samples_by_id = {_sample_task_id(sample): sample for sample in samples}

    output_path = Path(args.output)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else checkpoint_path_for(output_path)
    )
    checkpoint = TaskCheckpoint(
        checkpoint_path,
        stage="bird_extraction",
        required_ids=required_ids,
        model=args.model,
        resume=args.resume,
    )
    extractor = BirdSqlExtractor(
        model=args.model,
        verif_model=args.model,
        strip_model=args.model,
    )

    pending = [samples_by_id[task_id] for task_id in checkpoint.pending_ids]
    print(
        f"BIRD extraction: {len(checkpoint.processed_ids)} complete, "
        f"{len(pending)} pending, model={args.model}"
    )
    if args.num_workers > 1 and pending:
        executor = ThreadPoolExecutor(max_workers=args.num_workers)
        futures = {
            executor.submit(_process_one, extractor, sample): sample
            for sample in pending
        }
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="extract"):
                _record_future(future, futures[future], checkpoint)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    else:
        for sample in tqdm(pending, desc="extract"):
            task_id = _sample_task_id(sample)
            try:
                checkpoint.record_success(_process_one(extractor, sample))
            except BaseException as exc:
                checkpoint.record_failure(task_id, exc)
                raise

    ordered = validate_stage_rows(
        checkpoint.results,
        stage="bird_extraction",
        required_ids=required_ids,
        require_model=True,
    )
    assert_required_model(args.model, context="BIRD extraction completed model")
    checkpoint.mark_complete()
    atomic_write_json(output_path, ordered)
    print(f"extracted {len(ordered)}/{len(required_ids)} samples to {output_path}")


if __name__ == "__main__":
    main()
