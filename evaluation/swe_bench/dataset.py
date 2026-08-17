"""Prepare a local 50-row SWE-bench verifier dataset without HF loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .state import (
    HardeningError,
    atomic_write_json,
    file_sha256,
    read_json,
    validate_exact_ids,
)


REQUIRED_OFFICIAL_FIELDS = frozenset(
    {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "version",
    }
)


def _read_filtered_parquet(path: Path, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
    try:
        import pyarrow.dataset as arrow_dataset
    except ImportError as exc:
        raise HardeningError(
            "reading the local SWE parquet requires pyarrow (installed by datasets)"
        ) from exc
    try:
        dataset = arrow_dataset.dataset(path, format="parquet")
        missing_columns = REQUIRED_OFFICIAL_FIELDS - set(dataset.schema.names)
        if missing_columns:
            raise HardeningError(
                f"official SWE parquet is missing fields: {sorted(missing_columns)}"
            )
        table = dataset.scanner(
            filter=arrow_dataset.field("instance_id").isin(list(instance_ids))
        ).to_table()
        return table.to_pylist()
    except HardeningError:
        raise
    except Exception as exc:
        raise HardeningError(f"cannot filter official SWE parquet {path}: {exc}") from exc


def _read_filtered_json(path: Path, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: Any = []
        try:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise HardeningError(
                        f"official SWE JSONL line {line_number} is not an object"
                    )
                rows.append(row)
        except json.JSONDecodeError as exc:
            raise HardeningError(f"official SWE JSONL is invalid: {exc}") from exc
    else:
        rows = read_json(path, label="official SWE JSON dataset")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise HardeningError("official SWE JSON dataset must be a list of objects")
    allowed = set(instance_ids)
    return [dict(row) for row in rows if row.get("instance_id") in allowed]


def prepare_filtered_official_dataset(
    source_path: str | Path,
    *,
    instance_ids: Sequence[str],
    output_path: str | Path,
) -> dict[str, Any]:
    """Predicate-filter the official source and atomically write 50 JSON rows."""
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise HardeningError(f"official SWE dataset source does not exist: {source}")
    if source.suffix == ".parquet":
        rows = _read_filtered_parquet(source, instance_ids)
    elif source.suffix in {".json", ".jsonl"}:
        rows = _read_filtered_json(source, instance_ids)
    else:
        raise HardeningError("official SWE dataset source must be .parquet, .json, or .jsonl")

    actual_ids = [row.get("instance_id") for row in rows]
    if not all(isinstance(instance_id, str) for instance_id in actual_ids):
        raise HardeningError("official SWE dataset contains a row without instance_id")
    validate_exact_ids(
        actual_ids,
        instance_ids,
        label="filtered official SWE dataset",
    )
    by_id = {row["instance_id"]: row for row in rows}
    ordered = [by_id[instance_id] for instance_id in instance_ids]
    for row in ordered:
        missing = REQUIRED_OFFICIAL_FIELDS - set(row)
        if missing:
            raise HardeningError(
                f"official SWE row {row['instance_id']} is missing fields: {sorted(missing)}"
            )
        empty = [
            field
            for field in REQUIRED_OFFICIAL_FIELDS
            if row.get(field) is None
        ]
        if empty:
            raise HardeningError(
                f"official SWE row {row['instance_id']} has null fields: {sorted(empty)}"
            )

    atomic_write_json(output, ordered)
    return {
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "filtered_path": str(output),
        "filtered_sha256": file_sha256(output),
        "rows": len(ordered),
        "instance_ids": list(instance_ids),
    }
