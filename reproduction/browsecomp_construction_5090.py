"""Resume BrowseComp+ construction on one RTX 5090 host.

This entrypoint never creates or contacts Modal compute. It requires the
migrated Stage 1/2 artifacts and checkpoints, then resumes only missing work.
Credentials must already be present in the process environment; the launcher
is responsible for loading them from CC Switch without writing them to disk.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TextIO

from reproduction.browsecomp_construction_modal import (
    STAGE3_READ_TIMEOUT_SECONDS,
    STAGE3_WORKERS,
    _execute_remote,
    _retry_incomplete_stage3_processor,
    _validate_secret_environment,
)
from reproduction.browsecomp_plan_a import (
    BM25_CORPUS_DATASET,
    BM25_CORPUS_REVISION,
    DEFAULT_MANIFEST,
    HF_DATASET_REVISION,
    PLAN_A_MODEL,
    PREDECESSOR_PROMPTS,
    REMOTE_PRIVATE_SOURCE_COLUMNS,
    WorkflowError,
    _checkpoint_envelope,
    _checkpoint_path,
    _load_selected_samples,
    _load_stage_results,
    _stage_result_is_complete,
    atomic_write_json,
    audit_completed_run,
    validate_prepared_data,
)


REQUIRED_STAGE_COUNTS = {
    "stage1_extracted": 100,
    "stage2_counterfactual": 100,
}
STAGE3_SHARDS = 10
STAGE3_SHARD_WORKERS = 2
CONSTRUCTION_MODE_LOCK = "construction-mode.lock"


def _json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise WorkflowError(f"required migrated artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise WorkflowError(f"migrated artifact is not a JSON list: {path}")
    return payload


def checkpoint_inventory(root: Path) -> dict[str, int]:
    """Return counts only; never expose private task contents."""
    run_dir = root / "run"
    inventory: dict[str, int] = {}
    for stage in (
        "stage1_extracted",
        "stage2_counterfactual",
        "stage3_predecessor",
    ):
        checkpoint_dir = run_dir / f"{stage}.checkpoints"
        inventory[stage] = (
            len(list(checkpoint_dir.glob("*.json")))
            if checkpoint_dir.is_dir()
            else 0
        )
    return inventory


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WorkflowError(f"required migrated artifact is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise WorkflowError(f"migrated artifact is not a JSON object: {path}")
    return payload


def _ordered_checkpoint_results(
    name: str,
    inputs: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    checkpoint_dir = output_path.parent / f"{output_path.stem}.checkpoints"
    results_by_id = _load_stage_results(
        name,
        inputs,
        output_path,
        checkpoint_dir,
    )
    expected_ids = [str(item["task_id"]) for item in inputs]
    if set(results_by_id) != set(expected_ids):
        raise WorkflowError(
            f"{name} checkpoint IDs do not cover the fixed 100-task manifest"
        )
    ordered = [results_by_id[task_id] for task_id in expected_ids]
    if not all(_stage_result_is_complete(name, item) for item in ordered):
        raise WorkflowError(f"{name} contains an incomplete checkpoint result")
    aggregate = _json_list(output_path)
    if aggregate != ordered:
        raise WorkflowError(f"{name} aggregate does not match atomic checkpoints")
    return ordered


def validate_migrated_root(
    root: Path,
    *,
    expected_manifest_path: Path = DEFAULT_MANIFEST,
    require_stage3_aggregate: bool = True,
) -> dict[str, Any]:
    required_files = (
        root / "manifest.json",
        root / "private" / "raw_data_with_gold_docs.jsonl",
        root / "private" / "raw_data_with_gold_docs.manifest.json",
    )
    for path in required_files:
        if not path.is_file():
            raise WorkflowError(f"required migrated artifact is missing: {path}")

    migrated_manifest = _read_json_object(root / "manifest.json")
    expected_manifest = _read_json_object(expected_manifest_path)
    if migrated_manifest != expected_manifest:
        raise WorkflowError("migrated manifest does not match the fixed manifest")

    private_raw = root / "private" / "raw_data_with_gold_docs.jsonl"
    preparation = _read_json_object(
        root / "private" / "raw_data_with_gold_docs.manifest.json"
    )
    if preparation.get("source_revision") != HF_DATASET_REVISION:
        raise WorkflowError("migrated private source revision does not match")
    if preparation.get("source_columns") != list(REMOTE_PRIVATE_SOURCE_COLUMNS):
        raise WorkflowError("migrated private source columns do not match")
    prepared = validate_prepared_data(
        private_raw,
        root / "manifest.json",
        require_gold_docs=True,
        expected_sample_count=100,
    )
    raw_samples = _load_selected_samples(
        private_raw,
        root / "manifest.json",
        expected_sample_count=100,
    )

    run_dir = root / "run"
    inventory = checkpoint_inventory(root)
    for stage, expected in REQUIRED_STAGE_COUNTS.items():
        if inventory[stage] != expected:
            raise WorkflowError(
                f"{stage} checkpoint count is {inventory[stage]}, expected {expected}"
            )

    stage1 = _ordered_checkpoint_results(
        "stage1",
        raw_samples,
        run_dir / "stage1_extracted.json",
    )
    stage2 = _ordered_checkpoint_results(
        "stage2",
        stage1,
        run_dir / "stage2_counterfactual.json",
    )
    stage3_output = run_dir / "stage3_predecessor.json"
    aggregate_matches_checkpoints = True
    stage3_results = _load_stage_results(
        "stage3",
        stage2,
        stage3_output,
        run_dir / "stage3_predecessor.checkpoints",
    )
    # Atomic checkpoint files can appear while ten shards preflight. Treat one
    # validated directory snapshot as authoritative instead of comparing it to
    # an earlier glob count that may already be stale.
    inventory["stage3_predecessor"] = len(stage3_results)
    if inventory["stage3_predecessor"] > 100:
        raise WorkflowError("Stage 3 checkpoint count exceeds the fixed 100 IDs")
    if stage3_results:
        if not all(
            _stage_result_is_complete("stage3", item)
            for item in stage3_results.values()
        ):
            raise WorkflowError("Stage 3 contains an incomplete checkpoint result")
        ordered_partial = [
            stage3_results[item["task_id"]]
            for item in stage2
            if item["task_id"] in stage3_results
        ]
        aggregate_matches_checkpoints = (
            stage3_output.is_file()
            and _json_list(stage3_output) == ordered_partial
        )
        if require_stage3_aggregate and not aggregate_matches_checkpoints:
            raise WorkflowError(
                "Stage 3 aggregate does not match atomic checkpoints"
            )
    elif stage3_output.exists() and _json_list(stage3_output):
        raise WorkflowError("Stage 3 aggregate exists without atomic checkpoints")
    return {
        "root": str(root.resolve()),
        "stage_counts": inventory,
        "manifest_ids": len(expected_manifest["samples"]),
        "private_rows": prepared["sample_count"],
        "policies_and_hashes": "verified",
        "aggregate_matches_checkpoints": aggregate_matches_checkpoints,
        "safe_to_resume": True,
    }


def validate_worker_configuration(workers: int) -> None:
    if workers != STAGE3_WORKERS:
        raise WorkflowError(
            f"5090 Stage 3 requires exactly {STAGE3_WORKERS} internal workers"
        )


def validate_shard_configuration(
    shard_index: int,
    shard_count: int,
    workers: int,
) -> None:
    if shard_count != STAGE3_SHARDS:
        raise WorkflowError(
            f"formal 5090 Stage 3 requires exactly {STAGE3_SHARDS} shards"
        )
    if shard_index < 0 or shard_index >= shard_count:
        raise WorkflowError("Stage 3 shard index is outside the configured range")
    if workers != STAGE3_SHARD_WORKERS:
        raise WorkflowError(
            f"formal 5090 Stage 3 shards require exactly {STAGE3_SHARD_WORKERS} workers"
        )


def stage3_shard_inputs(
    inputs: list[dict[str, Any]],
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Split the fixed manifest order into deterministic, disjoint shards."""
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise WorkflowError("invalid Stage 3 shard configuration")
    return [
        item
        for position, item in enumerate(inputs)
        if position % shard_count == shard_index
    ]


def acquire_stage3_shard_lock(root: Path, shard_index: int) -> TextIO:
    lock_path = root / "stage3-shards" / f"shard-{shard_index:02d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise WorkflowError(f"Stage 3 shard {shard_index} already has an owner") from exc
    return handle


def acquire_construction_mode_lock(root: Path, *, exclusive: bool) -> TextIO:
    """Prevent monolithic and sharded construction modes from overlapping."""
    lock_path = root / CONSTRUCTION_MODE_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        mode = "monolithic/finalize" if exclusive else "sharded"
        raise WorkflowError(
            f"BrowseComp+ construction mode conflicts with an active {mode} owner"
        ) from exc
    return handle


def _load_stage2_inputs(root: Path) -> list[dict[str, Any]]:
    private_raw = root / "private" / "raw_data_with_gold_docs.jsonl"
    manifest_path = root / "manifest.json"
    run_dir = root / "run"
    raw_samples = _load_selected_samples(
        private_raw,
        manifest_path,
        expected_sample_count=100,
    )
    stage1 = _ordered_checkpoint_results(
        "stage1",
        raw_samples,
        run_dir / "stage1_extracted.json",
    )
    return _ordered_checkpoint_results(
        "stage2",
        stage1,
        run_dir / "stage2_counterfactual.json",
    )


def run_stage3_shard(
    root: Path,
    *,
    shard_index: int,
    shard_count: int,
    workers: int,
    expected_manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Run one non-overlapping Stage 3 shard without writing the aggregate."""
    validate_shard_configuration(shard_index, shard_count, workers)
    mode_lock = acquire_construction_mode_lock(root, exclusive=False)
    try:
        shard_lock = acquire_stage3_shard_lock(root, shard_index)
    except BaseException:
        mode_lock.close()
        raise
    ledger = root / "usage" / "llm_usage.jsonl"
    try:
        preflight = validate_migrated_root(
            root,
            expected_manifest_path=expected_manifest_path,
            require_stage3_aggregate=False,
        )
        _validate_secret_environment(root=root, ledger=ledger)
        from intent_construction.intent_extraction.core import llm_utils
        from intent_construction.retrospective_expansion.predecessor.generate_predecessors import (
            PredecessorGenerator,
        )

        llm_utils._CLIENT_TIMEOUT = llm_utils.httpx.Timeout(
            connect=30.0,
            read=float(STAGE3_READ_TIMEOUT_SECONDS),
            write=600.0,
            pool=600.0,
        )
        stage2 = _load_stage2_inputs(root)
        assigned = stage3_shard_inputs(stage2, shard_index, shard_count)
        output_path = root / "run" / "stage3_predecessor.json"
        checkpoint_dir = root / "run" / "stage3_predecessor.checkpoints"
        existing = _load_stage_results(
            "stage3",
            stage2,
            output_path,
            checkpoint_dir,
        )
        pending = [item for item in assigned if item["task_id"] not in existing]
        generator = PredecessorGenerator(
            model=PLAN_A_MODEL,
            prompts_dir=str(PREDECESSOR_PROMPTS),
            fallback_model=PLAN_A_MODEL,
            judge_model=PLAN_A_MODEL,
            dataset_type="browsecomp",
            num_predecessors=3,
            reasoning_effort=None,
            corpus_dataset=BM25_CORPUS_DATASET,
            corpus_revision=BM25_CORPUS_REVISION,
        )
        processor = _retry_incomplete_stage3_processor(
            generator.generate_predecessors
        )
        completed = 0
        completed_lock = threading.Lock()
        task_lock_dir = root / "run" / "stage3_predecessor.locks"
        task_lock_dir.mkdir(parents=True, exist_ok=True)

        def process(item: dict[str, Any]) -> None:
            nonlocal completed
            checkpoint_path = _checkpoint_path(checkpoint_dir, str(item["task_id"]))
            task_lock_path = task_lock_dir / f"{checkpoint_path.name}.lock"
            task_lock = task_lock_path.open("a+")
            try:
                try:
                    fcntl.flock(
                        task_lock.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError as exc:
                    raise WorkflowError(
                        f"Stage 3 task already has an owner: {item['task_id']}"
                    ) from exc
                if checkpoint_path.exists():
                    return
                result = processor(item)
                if (
                    not _stage_result_is_complete("stage3", result)
                    or result.get("task_id") != item.get("task_id")
                ):
                    raise WorkflowError(
                        f"Stage 3 shard produced an incomplete result for {item['task_id']}"
                    )
                atomic_write_json(
                    checkpoint_path,
                    _checkpoint_envelope("stage3", item, result),
                )
                with completed_lock:
                    completed += 1
            finally:
                try:
                    fcntl.flock(task_lock.fileno(), fcntl.LOCK_UN)
                finally:
                    task_lock.close()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, item) for item in pending]
            for future in as_completed(futures):
                future.result()
        return {
            "runtime": "rtx-5090",
            "model": PLAN_A_MODEL,
            "reasoning_effort": "medium",
            "shard_index": shard_index,
            "shard_count": shard_count,
            "workers": workers,
            "assigned": len(assigned),
            "already_complete": len(assigned) - len(pending),
            "completed": completed,
            "preflight_stage_counts": preflight["stage_counts"],
        }
    finally:
        shard_lock.close()
        mode_lock.close()


def finalize_sharded_run(
    root: Path,
    *,
    expected_manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Build ordered aggregates only after all Stage 3 shards have exited."""
    mode_lock = acquire_construction_mode_lock(root, exclusive=True)
    shard_locks: list[TextIO] = []
    try:
        for shard_index in range(STAGE3_SHARDS):
            shard_locks.append(acquire_stage3_shard_lock(root, shard_index))
        validate_migrated_root(
            root,
            expected_manifest_path=expected_manifest_path,
            require_stage3_aggregate=False,
        )
        stage2 = _load_stage2_inputs(root)
        run_dir = root / "run"
        output_path = run_dir / "stage3_predecessor.json"
        checkpoint_dir = run_dir / "stage3_predecessor.checkpoints"
        results = _load_stage_results("stage3", stage2, output_path, checkpoint_dir)
        expected_ids = [str(item["task_id"]) for item in stage2]
        if set(results) != set(expected_ids):
            raise WorkflowError(
                "Stage 3 shards are incomplete: "
                f"{len(expected_ids) - len(results)} tasks missing"
            )
        ordered = [results[task_id] for task_id in expected_ids]
        if not all(_stage_result_is_complete("stage3", item) for item in ordered):
            raise WorkflowError("Stage 3 shard output contains an incomplete result")
        final_path = root / "final_dataset" / "browsecomp_plus_final.json"
        ledger = root / "usage" / "llm_usage.jsonl"
        # Audit the complete ordered data and usage before publishing any
        # completion artifact. Stage 1/2 are immutable validated inputs.
        with tempfile.TemporaryDirectory(
            prefix=".stage3-finalize-",
            dir=run_dir,
        ) as staging_raw:
            staging_run = Path(staging_raw)
            for name in ("stage1_extracted.json", "stage2_counterfactual.json"):
                os.symlink(run_dir / name, staging_run / name)
            staging_stage3 = staging_run / "stage3_predecessor.json"
            staging_final = staging_run / "browsecomp_plus_final.json"
            atomic_write_json(staging_stage3, ordered)
            atomic_write_json(staging_final, ordered)
            audit = audit_completed_run(
                root / "manifest.json",
                staging_run,
                staging_final,
                ledger,
                expected_sample_count=100,
            )
        atomic_write_json(output_path, ordered)
        atomic_write_json(final_path, ordered)
        summary = {
            "status": "complete",
            "model": PLAN_A_MODEL,
            "sample_count": len(ordered),
            "stage1": str((run_dir / "stage1_extracted.json").resolve()),
            "stage2": str((run_dir / "stage2_counterfactual.json").resolve()),
            "stage3": str(output_path.resolve()),
            "final_dataset": str(final_path.resolve()),
            "construction_shards": STAGE3_SHARDS,
            "workers_per_shard": STAGE3_SHARD_WORKERS,
        }
        atomic_write_json(run_dir / "construction_summary.json", summary)
        preparation = _read_json_object(
            root / "private" / "raw_data_with_gold_docs.manifest.json"
        )
        safe_report = {
            "status": "complete",
            "model": PLAN_A_MODEL,
            "sample_count": audit["sample_count"],
            "source_revision": preparation["source_revision"],
            "source_columns": preparation["source_columns"],
            "stage_counts": audit["stage_counts"],
            "usage": audit["usage"],
            "independence_verification": True,
            "remote_final": summary["final_dataset"],
            "construction_shards": STAGE3_SHARDS,
            "workers_per_shard": STAGE3_SHARD_WORKERS,
        }
        atomic_write_json(run_dir / "remote_audit.json", safe_report)
        return safe_report
    finally:
        for shard_lock in reversed(shard_locks):
            shard_lock.close()
        mode_lock.close()


def run(root: Path, workers: int) -> dict[str, Any]:
    validate_worker_configuration(workers)
    mode_lock = acquire_construction_mode_lock(root, exclusive=True)
    try:
        preflight = validate_migrated_root(root)
        report = _execute_remote(
            DEFAULT_MANIFEST.read_text(encoding="utf-8"),
            workers=workers,
            root=root,
            expected_sample_count=100,
            commit=lambda: None,
        )
        return {"preflight": preflight, "result": report, "runtime": "rtx-5090"}
    finally:
        mode_lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reproduction/runs/browsecomp-plan-a-n100"),
    )
    parser.add_argument("--workers", type=int, default=STAGE3_WORKERS)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=STAGE3_SHARDS)
    parser.add_argument("--finalize-shards", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inspect_only:
        report = validate_migrated_root(
            args.root,
            require_stage3_aggregate=False,
        )
    elif args.finalize_shards:
        report = finalize_sharded_run(args.root)
    elif args.shard_index is not None:
        report = run_stage3_shard(
            args.root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            workers=args.workers,
        )
    else:
        validate_worker_configuration(args.workers)
        report = run(args.root, args.workers)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
