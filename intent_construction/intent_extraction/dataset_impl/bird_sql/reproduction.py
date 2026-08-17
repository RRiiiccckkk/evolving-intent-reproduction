"""Strict invariants for the published BIRD-SQL reproduction.

This module is intentionally BIRD-specific.  It keeps the fixed evaluation
subset, model lock, and durable per-task checkpoints out of shared pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_MODEL = "kimi-k2.6"
PUBLISHED_TASK_COUNT = 100
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TASK_IDS_PATH = (
    REPO_ROOT / "intent_construction" / "eval_indices" / "bird_sql_task_ids.json"
)
DEFAULT_EVAL_MANIFEST_PATH = (
    REPO_ROOT / "intent_construction" / "eval_indices" / "bird_sql_eval_ids.json"
)
_TASK_ID_RE = re.compile(r"^bird_sql_(train|dev)_(0|[1-9][0-9]*)$")


class BirdReproductionError(RuntimeError):
    """Raised when a published-run invariant is violated."""


def acquire_exclusive_run_lock(run_dir: str | Path):
    """Hold one advisory construction lock for the lifetime of the handle."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - the formal run is POSIX-only
        raise BirdReproductionError("BIRD construction requires POSIX file locking") from exc

    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "construction.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise BirdReproductionError(
            f"another BIRD construction process already holds {root / 'construction.lock'}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically replace a JSON file after flushing it to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def assert_required_model(model: str | None, *, context: str) -> str:
    """Reject every logical model except the paper's Kimi K2.6 model."""
    if model != REQUIRED_MODEL:
        raise BirdReproductionError(
            f"{context} must use exactly {REQUIRED_MODEL!r}; received {model!r}"
        )
    return model


def task_id_for(source: str, index: int | str) -> str:
    if source not in {"train", "dev"}:
        raise BirdReproductionError(f"unsupported BIRD split: {source!r}")
    try:
        numeric_index = int(index)
    except (TypeError, ValueError) as exc:
        raise BirdReproductionError(f"invalid BIRD row index: {index!r}") from exc
    if numeric_index < 0:
        raise BirdReproductionError(f"invalid BIRD row index: {numeric_index}")
    return f"bird_sql_{source}_{numeric_index}"


def resolve_db_path(value: str | Path) -> Path:
    """Resolve a relocatable repository-relative BIRD database path."""
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_task_ids(ids: Sequence[Any], *, source: str) -> list[str]:
    if len(ids) != PUBLISHED_TASK_COUNT:
        raise BirdReproductionError(
            f"{source} must contain exactly {PUBLISHED_TASK_COUNT} task IDs; "
            f"found {len(ids)}"
        )
    if not all(isinstance(task_id, str) for task_id in ids):
        raise BirdReproductionError(f"{source} contains a non-string task ID")
    values = list(ids)
    if len(set(values)) != len(values):
        raise BirdReproductionError(f"{source} contains duplicate task IDs")
    malformed = [task_id for task_id in values if not _TASK_ID_RE.fullmatch(task_id)]
    if malformed:
        raise BirdReproductionError(
            f"{source} contains malformed task IDs: {malformed[:3]}"
        )
    return values


def load_published_task_ids(
    path: str | Path = DEFAULT_TASK_IDS_PATH,
) -> list[str]:
    payload = read_json(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("task_ids"), list):
        raise BirdReproductionError(f"published ID file is malformed: {path}")
    ids = _validate_task_ids(payload["task_ids"], source=str(path))
    if payload.get("num_samples") != PUBLISHED_TASK_COUNT:
        raise BirdReproductionError(
            f"published ID file declares num_samples={payload.get('num_samples')!r}"
        )
    return ids


def load_published_manifest(
    path: str | Path = DEFAULT_EVAL_MANIFEST_PATH,
    *,
    task_ids_path: str | Path = DEFAULT_TASK_IDS_PATH,
) -> list[dict[str, Any]]:
    payload = read_json(path)
    samples = payload.get("samples") if isinstance(payload, Mapping) else None
    if not isinstance(samples, list) or not all(isinstance(row, dict) for row in samples):
        raise BirdReproductionError(f"published BIRD manifest is malformed: {path}")
    ids = _validate_task_ids(
        [row.get("task_id") for row in samples], source=str(path)
    )
    if payload.get("num_samples") != PUBLISHED_TASK_COUNT:
        raise BirdReproductionError(
            f"published manifest declares num_samples={payload.get('num_samples')!r}"
        )
    if payload.get("split") != "train+dev":
        raise BirdReproductionError(
            "published BIRD manifest split must be 'train+dev'"
        )
    expected_ids = load_published_task_ids(task_ids_path)
    if ids != expected_ids:
        raise BirdReproductionError(
            "published manifest and task-ID file differ in membership or order"
        )
    for row in samples:
        match = _TASK_ID_RE.fullmatch(row["task_id"])
        assert match is not None
        expected_index = int(row["task_id"].rsplit("_", 1)[1])
        if row.get("original_id") != expected_index or row.get("original_index") != expected_index:
            raise BirdReproductionError(
                f"manifest indices do not match {row['task_id']}"
            )
        if not isinstance(row.get("db_id"), str) or not row["db_id"]:
            raise BirdReproductionError(f"manifest entry has no db_id: {row['task_id']}")
    return [dict(row) for row in samples]


def _rows_by_id(
    rows: Iterable[Mapping[str, Any]], *, stage: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise BirdReproductionError(
                f"{stage} row {position} is not a JSON object"
            )
        row = dict(raw)
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            raise BirdReproductionError(f"{stage} row {position} has no task_id")
        if task_id in indexed:
            raise BirdReproductionError(f"{stage} contains duplicate task ID {task_id}")
        indexed[task_id] = row
    return indexed


def assert_row_model(row: Mapping[str, Any], *, stage: str) -> None:
    assert_required_model(row.get("model_name"), context=f"{stage} row model_name")
    predecessor = row.get("predecessor_info")
    if predecessor is not None:
        if not isinstance(predecessor, Mapping):
            raise BirdReproductionError(f"{stage} predecessor_info is malformed")
        assert_required_model(
            predecessor.get("model"), context=f"{stage} predecessor model"
        )
        assert_required_model(
            predecessor.get("naturalizer_model"),
            context=f"{stage} naturalizer model",
        )


def validate_stage_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    required_ids: Sequence[str] | None = None,
    require_model: bool = True,
    min_predecessors: int = 0,
) -> list[dict[str, Any]]:
    """Validate exact published coverage and return rows in published order."""
    expected = list(required_ids or load_published_task_ids())
    indexed = _rows_by_id(rows, stage=stage)
    missing = [task_id for task_id in expected if task_id not in indexed]
    extras = [task_id for task_id in indexed if task_id not in set(expected)]
    if missing or extras or len(indexed) != len(expected):
        raise BirdReproductionError(
            f"{stage} coverage failure: expected={len(expected)}, found={len(indexed)}, "
            f"missing={missing[:5]}, extras={extras[:5]}"
        )
    ordered = [indexed[task_id] for task_id in expected]
    for row in ordered:
        if require_model:
            assert_row_model(row, stage=stage)
        if min_predecessors:
            predecessors = row.get("predecessor_functions")
            if not isinstance(predecessors, list) or len(predecessors) < min_predecessors:
                raise BirdReproductionError(
                    f"{stage} task {row['task_id']} has fewer than "
                    f"{min_predecessors} complete predecessors"
                )
            if any(
                not isinstance(item, Mapping)
                or not str(item.get("predecessor_function", "")).strip()
                or not str(item.get("counterfactual_sql", "")).strip()
                for item in predecessors
            ):
                raise BirdReproductionError(
                    f"{stage} task {row['task_id']} has an incomplete predecessor"
                )
    return ordered


def checkpoint_path_for(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.stem}_checkpoint{output.suffix or '.json'}")


class TaskCheckpoint:
    """Atomic, per-task checkpoint whose completed IDs are safe to resume."""

    def __init__(
        self,
        path: str | Path,
        *,
        stage: str,
        required_ids: Sequence[str],
        model: str | None = None,
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.stage = stage
        self.required_ids = list(required_ids)
        self.model = model
        if len(set(self.required_ids)) != len(self.required_ids):
            raise BirdReproductionError("checkpoint required IDs contain duplicates")
        if model is not None:
            assert_required_model(model, context=f"{stage} checkpoint model")

        if resume and self.path.exists():
            payload = read_json(self.path)
            self._load(payload)
        else:
            self.results_by_id: dict[str, dict[str, Any]] = {}
            self.failures: dict[str, dict[str, str]] = {}
            self.complete = False
            self.flush()

    def _load(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise BirdReproductionError(f"checkpoint is malformed: {self.path}")
        if payload.get("schema_version") != 1 or payload.get("stage") != self.stage:
            raise BirdReproductionError(
                f"checkpoint stage/schema mismatch: {self.path}"
            )
        if payload.get("required_task_ids") != self.required_ids:
            raise BirdReproductionError(
                "checkpoint task IDs differ from the fixed published subset"
            )
        if payload.get("model") != self.model:
            raise BirdReproductionError(
                f"checkpoint model mismatch: expected {self.model!r}, "
                f"found {payload.get('model')!r}"
            )
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise BirdReproductionError("checkpoint results must be a list")
        self.results_by_id = _rows_by_id(raw_results, stage=self.stage)
        allowed = set(self.required_ids)
        extras = set(self.results_by_id) - allowed
        if extras:
            raise BirdReproductionError(
                f"checkpoint contains non-published task IDs: {sorted(extras)[:5]}"
            )
        processed = payload.get("processed_ids")
        expected_processed = [
            task_id for task_id in self.required_ids if task_id in self.results_by_id
        ]
        if processed != expected_processed:
            raise BirdReproductionError(
                "checkpoint processed_ids do not exactly match successful results"
            )
        raw_failures = payload.get("failures", {})
        if not isinstance(raw_failures, Mapping):
            raise BirdReproductionError("checkpoint failures must be an object")
        self.failures = {
            str(task_id): dict(detail)
            for task_id, detail in raw_failures.items()
            if isinstance(detail, Mapping)
        }
        self.complete = bool(payload.get("complete", False))
        if self.complete and len(self.results_by_id) != len(self.required_ids):
            raise BirdReproductionError("checkpoint claims complete coverage but is partial")

    @property
    def processed_ids(self) -> list[str]:
        return [
            task_id for task_id in self.required_ids if task_id in self.results_by_id
        ]

    @property
    def results(self) -> list[dict[str, Any]]:
        return [self.results_by_id[task_id] for task_id in self.processed_ids]

    @property
    def pending_ids(self) -> list[str]:
        return [
            task_id for task_id in self.required_ids if task_id not in self.results_by_id
        ]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "model": self.model,
            "required_task_ids": self.required_ids,
            "processed_ids": self.processed_ids,
            "results": self.results,
            "failures": self.failures,
            "complete": self.complete,
        }

    def flush(self) -> None:
        atomic_write_json(self.path, self.payload())

    def record_success(self, row: Mapping[str, Any]) -> None:
        task_id = row.get("task_id")
        if task_id not in set(self.required_ids):
            raise BirdReproductionError(
                f"{self.stage} produced unexpected task ID {task_id!r}"
            )
        materialized = dict(row)
        if self.model is not None:
            assert_row_model(materialized, stage=self.stage)
        self.results_by_id[str(task_id)] = materialized
        self.failures.pop(str(task_id), None)
        self.complete = False
        self.flush()

    def record_failure(self, task_id: str, error: BaseException | str) -> None:
        if task_id not in set(self.required_ids):
            raise BirdReproductionError(
                f"{self.stage} failed unexpected task ID {task_id!r}"
            )
        self.failures[task_id] = {
            "error_type": type(error).__name__ if isinstance(error, BaseException) else "error",
            "message": str(error),
        }
        self.complete = False
        self.flush()

    def mark_complete(self) -> None:
        if self.pending_ids:
            raise BirdReproductionError(
                f"{self.stage} cannot complete; missing {self.pending_ids[:5]}"
            )
        self.complete = True
        self.flush()


def _cli_validate(args: argparse.Namespace) -> None:
    payload = read_json(args.path)
    if not isinstance(payload, list):
        raise BirdReproductionError(f"{args.path} must contain a JSON list")
    required_ids = load_published_task_ids(args.task_ids_file)
    validate_stage_rows(
        payload,
        stage=args.stage,
        required_ids=required_ids,
        require_model=not args.no_model_check,
        min_predecessors=args.min_predecessors,
    )
    print(f"validated {len(required_ids)}/{len(required_ids)} published task IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a BIRD stage artifact")
    validate.add_argument("--path", required=True)
    validate.add_argument("--stage", required=True)
    validate.add_argument("--task-ids-file", default=str(DEFAULT_TASK_IDS_PATH))
    validate.add_argument("--min-predecessors", type=int, default=0)
    validate.add_argument("--no-model-check", action="store_true")
    validate.set_defaults(handler=_cli_validate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
