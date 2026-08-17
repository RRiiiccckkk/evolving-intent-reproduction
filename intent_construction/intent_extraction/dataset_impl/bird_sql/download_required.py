"""Download only the files needed by the fixed BIRD-SQL evaluation subset.

The official train archive nests a compressed database archive, so individual
SQLite files cannot be sought inside it.  The pinned public Kaggle mirrors used
here expose the same BIRD files individually.  Every transfer is an HTTP Range
request, is resumable through a ``.part`` file, and is atomically promoted only
after the advertised byte count is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import (
    DEFAULT_EVAL_MANIFEST_PATH,
    BirdReproductionError,
    REPO_ROOT,
    atomic_write_json,
    load_published_manifest,
    task_id_for,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_DATASET = "himanshusinghal19/bird-dataset"
TRAIN_DATASET_VERSION = 1
DEV_DATASET = "jenishk/bird-dev"
DEV_DATASET_VERSION = 1
KAGGLE_DOWNLOAD_API = "https://www.kaggle.com/api/v1/datasets/download"
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


@dataclass(frozen=True)
class RequiredFile:
    split: str
    kind: str
    db_id: str | None
    dataset: str
    version: int
    remote_path: str
    local_path: Path
    expected_sha256: str | None = None

    @property
    def url(self) -> str:
        query = urllib.parse.urlencode(
            {
                "datasetVersionNumber": self.version,
                "fileName": self.remote_path,
            }
        )
        return f"{KAGGLE_DOWNLOAD_API}/{self.dataset}?{query}"


@dataclass(frozen=True)
class RangeChunk:
    data: bytes
    start: int
    end: int
    total: int
    etag: str | None


def fetch_range(
    url: str,
    start: int,
    end: int,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: int = 120,
) -> RangeChunk:
    """Fetch one exact byte range and reject whole-file responses."""
    if start < 0 or end < start:
        raise ValueError(f"invalid byte range {start}-{end}")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "evolving-intent-bird-reproduction/1",
        },
    )
    with opener(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise BirdReproductionError(
                f"server ignored HTTP Range for {url!r}: status={status}"
            )
        content_range = response.headers.get("Content-Range", "")
        match = _CONTENT_RANGE_RE.fullmatch(content_range.strip())
        if match is None:
            raise BirdReproductionError(
                f"invalid Content-Range for {url!r}: {content_range!r}"
            )
        actual_start, actual_end, total = map(int, match.groups())
        if actual_start != start or actual_end > end or actual_end >= total:
            raise BirdReproductionError(
                f"unexpected Content-Range {content_range!r} for bytes={start}-{end}"
            )
        data = response.read()
        if len(data) != actual_end - actual_start + 1:
            raise BirdReproductionError(
                f"short ranged response for {url!r}: expected "
                f"{actual_end - actual_start + 1}, got {len(data)}"
            )
        return RangeChunk(
            data=data,
            start=actual_start,
            end=actual_end,
            total=total,
            etag=response.headers.get("ETag"),
        )


def download_range_file(
    url: str,
    destination: str | Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
    opener: Callable[..., Any] = urllib.request.urlopen,
    progress: Callable[[int, int], None] | None = None,
    max_bytes: int | None = None,
) -> int:
    """Resume a file with Range requests and atomically install it."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    target = Path(destination)
    if target.exists():
        return target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")

    probe = fetch_range(url, 0, 0, opener=opener)
    total = probe.total
    if max_bytes is not None and total > max_bytes:
        raise BirdReproductionError(
            f"remote file exceeds temporary-space budget: {total} > {max_bytes}"
        )
    current = partial.stat().st_size if partial.exists() else 0
    if current > total:
        raise BirdReproductionError(
            f"partial file is larger than remote object: {partial}"
        )
    remaining = total - current
    free = shutil.disk_usage(target.parent).free
    reserve = 256 * 1024 * 1024
    if remaining + reserve > free:
        raise BirdReproductionError(
            f"insufficient free space for {target.name}: need {remaining + reserve}, "
            f"available {free}"
        )
    if current == 0:
        with partial.open("wb") as handle:
            handle.write(probe.data)
            handle.flush()
            os.fsync(handle.fileno())
        current = 1

    while current < total:
        end = min(current + chunk_size - 1, total - 1)
        chunk = fetch_range(url, current, end, opener=opener)
        if chunk.total != total:
            raise BirdReproductionError(
                f"remote object size changed while downloading {url!r}"
            )
        with partial.open("ab") as handle:
            handle.write(chunk.data)
            handle.flush()
            os.fsync(handle.fileno())
        current = chunk.end + 1
        if progress is not None:
            progress(current, total)

    if partial.stat().st_size != total:
        raise BirdReproductionError(
            f"completed file has wrong size: {partial.stat().st_size} != {total}"
        )
    os.replace(partial, target)
    return total


def required_files(
    manifest: list[dict[str, Any]],
    *,
    data_dir: Path = DATA_DIR,
) -> list[RequiredFile]:
    """Return the two split JSON files and 35 required SQLite files."""
    files = [
        RequiredFile(
            split="train",
            kind="json",
            db_id=None,
            dataset=TRAIN_DATASET,
            version=TRAIN_DATASET_VERSION,
            remote_path="train/train.json",
            local_path=data_dir / "train_extracted" / "train" / "train.json",
            expected_sha256="abf17d3def22f56f41b3e139b94bba3fa4a986280abcbf6df2d7f30150695520",
        ),
        RequiredFile(
            split="dev",
            kind="json",
            db_id=None,
            dataset=DEV_DATASET,
            version=DEV_DATASET_VERSION,
            remote_path="dev_20240627/dev.json",
            local_path=data_dir / "dev_extracted" / "dev_20240627" / "dev.json",
            expected_sha256="630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06",
        ),
    ]
    databases: dict[str, set[str]] = {"train": set(), "dev": set()}
    for row in manifest:
        split = row["task_id"].split("_")[2]
        databases[split].add(row["db_id"])

    for db_id in sorted(databases["train"]):
        files.append(
            RequiredFile(
                split="train",
                kind="sqlite",
                db_id=db_id,
                dataset=TRAIN_DATASET,
                version=TRAIN_DATASET_VERSION,
                remote_path=(
                    f"train/train_databases/train_databases/{db_id}/{db_id}.sqlite"
                ),
                local_path=(
                    data_dir
                    / "train_extracted"
                    / "train"
                    / "train_databases"
                    / db_id
                    / f"{db_id}.sqlite"
                ),
            )
        )
    for db_id in sorted(databases["dev"]):
        files.append(
            RequiredFile(
                split="dev",
                kind="sqlite",
                db_id=db_id,
                dataset=DEV_DATASET,
                version=DEV_DATASET_VERSION,
                remote_path=f"dev_20240627/dev_databases/{db_id}/{db_id}.sqlite",
                local_path=(
                    data_dir
                    / "dev_extracted"
                    / "dev_20240627"
                    / "dev_databases"
                    / db_id
                    / f"{db_id}.sqlite"
                ),
            )
        )
    return files


def _partial_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.part")


def resident_database_ids(specs: list[RequiredFile]) -> set[str]:
    """Return required database IDs with a complete or partial local file."""
    resident: set[str] = set()
    for spec in specs:
        if spec.kind != "sqlite" or spec.db_id is None:
            continue
        if spec.local_path.exists() or _partial_path(spec.local_path).exists():
            resident.add(spec.db_id)
    return resident


def ordered_database_ids(
    manifest: list[dict[str, Any]],
    *,
    data_dir: Path,
) -> list[str]:
    """Keep manifest order, but resume a single resident database first."""
    ordered: list[str] = []
    seen: set[str] = set()
    for row in manifest:
        db_id = row["db_id"]
        if db_id not in seen:
            seen.add(db_id)
            ordered.append(db_id)

    specs = required_files(manifest, data_dir=data_dir)
    resident = resident_database_ids(specs)
    if len(resident) > 1:
        raise BirdReproductionError(
            f"low-disk cache contains more than one database: {sorted(resident)}"
        )
    if resident:
        resident_id = next(iter(resident))
        ordered.sort(key=lambda db_id: 0 if db_id == resident_id else 1)
    return ordered


def assert_single_database_cache(
    specs: list[RequiredFile],
    *,
    requested_db_id: str,
) -> None:
    """Refuse a download that would make two databases resident."""
    resident = resident_database_ids(specs)
    if resident and resident != {requested_db_id}:
        raise BirdReproductionError(
            f"refusing to load {requested_db_id}; cache still contains "
            f"{sorted(resident)}"
        )


def evict_required_database(spec: RequiredFile, *, data_dir: Path) -> None:
    """Delete only one exact required SQLite file and its resumable partial."""
    if spec.kind != "sqlite" or spec.db_id is None:
        raise BirdReproductionError("refusing to evict a non-database file")
    root = data_dir.resolve()
    target = spec.local_path.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise BirdReproductionError(
            f"refusing to evict database outside the run cache: {target}"
        ) from exc
    if target.name != f"{spec.db_id}.sqlite" or target.parent.name != spec.db_id:
        raise BirdReproductionError(f"refusing to evict unexpected file: {target}")
    for path in (target, _partial_path(target)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _validate_json(path: Path, *, split: str) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BirdReproductionError(f"invalid BIRD JSON file: {path}") from exc
    expected_count = 9428 if split == "train" else 1534
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise BirdReproductionError(
            f"{split} JSON must contain {expected_count} rows; found "
            f"{len(rows) if isinstance(rows, list) else 'non-list'}"
        )
    return rows


def _validate_sqlite(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError as exc:
        raise BirdReproductionError(f"cannot read SQLite file: {path}") from exc
    if header != b"SQLite format 3\x00":
        raise BirdReproductionError(f"invalid SQLite header: {path}")
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        try:
            connection.execute("PRAGMA schema_version").fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise BirdReproductionError(f"cannot open SQLite file: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_valid_existing(spec: RequiredFile) -> bool:
    if not spec.local_path.exists():
        return False
    if spec.kind == "json":
        _validate_json(spec.local_path, split=spec.split)
        if spec.expected_sha256 and _sha256(spec.local_path) != spec.expected_sha256:
            raise BirdReproductionError(
                f"pinned SHA-256 mismatch for {spec.local_path}"
            )
    else:
        _validate_sqlite(spec.local_path)
    return True


def download_required_files(
    specs: list[RequiredFile],
    *,
    chunk_size: int,
    max_file_bytes: int | None = None,
) -> None:
    for position, spec in enumerate(specs, start=1):
        label = spec.db_id or f"{spec.split}.json"
        if _is_valid_existing(spec):
            print(f"[{position}/{len(specs)}] present: {label}")
            continue
        print(f"[{position}/{len(specs)}] range download: {label}")

        last_percent = -1

        def show_progress(done: int, total: int) -> None:
            nonlocal last_percent
            percent = int(done * 100 / total)
            if percent >= last_percent + 10 or done == total:
                print(f"  {done}/{total} bytes ({percent}%)")
                last_percent = percent

        download_range_file(
            spec.url,
            spec.local_path,
            chunk_size=chunk_size,
            progress=show_progress,
            max_bytes=max_file_bytes,
        )
        if not _is_valid_existing(spec):
            raise BirdReproductionError(f"download validation failed: {spec.local_path}")


def build_selected_file(
    manifest: list[dict[str, Any]],
    *,
    data_dir: Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    split_rows = {
        "train": _validate_json(
            data_dir / "train_extracted" / "train" / "train.json",
            split="train",
        ),
        "dev": _validate_json(
            data_dir / "dev_extracted" / "dev_20240627" / "dev.json",
            split="dev",
        ),
    }
    selected: list[dict[str, Any]] = []
    for entry in manifest:
        _, _, split, raw_index = entry["task_id"].split("_", 3)
        index = int(raw_index)
        raw = split_rows[split][index]
        if raw.get("db_id") != entry["db_id"]:
            raise BirdReproductionError(
                f"db_id mismatch for {entry['task_id']}: manifest={entry['db_id']!r}, "
                f"JSON={raw.get('db_id')!r}"
            )
        sample = {
            "task_id": task_id_for(split, index),
            "db_id": raw["db_id"],
            "question": raw["question"],
            "evidence": raw.get("evidence", ""),
            "gold_sql": raw["SQL"],
            "difficulty": raw.get("difficulty", ""),
            "source": split,
            "index": index,
            "question_id": raw.get("question_id", index),
        }
        selected.append(sample)
    atomic_write_json(output_path, selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(DEFAULT_EVAL_MANIFEST_PATH)
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "selected_published_n100.json"),
        help="Selected 100-row JSON passed to extraction.",
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--chunk-size-mib", type=int, default=8)
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument(
        "--only-json",
        action="store_true",
        help="Download and validate only train.json and dev.json.",
    )
    filters.add_argument(
        "--database",
        help="Download one required database by db_id and nothing else.",
    )
    filters.add_argument(
        "--evict-database",
        help="Delete one exact required database and its partial from data-dir.",
    )
    filters.add_argument(
        "--ordered-databases",
        action="store_true",
        help="Print database IDs in manifest order, with a resident partial first.",
    )
    parser.add_argument("--max-file-gib", type=float, default=8.0)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print exact remote/local file mappings without downloading.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Build the selected file from already validated local data.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    manifest = load_published_manifest(args.manifest)
    all_specs = required_files(manifest, data_dir=data_dir)
    if len(all_specs) != 37:
        raise BirdReproductionError(
            f"published manifest must resolve to 2 JSON + 35 DB files; got {len(all_specs)}"
        )
    if args.ordered_databases:
        for db_id in ordered_database_ids(manifest, data_dir=data_dir):
            print(db_id)
        return

    database_filter = args.database or args.evict_database
    specs = all_specs
    if args.only_json:
        specs = [spec for spec in all_specs if spec.kind == "json"]
    elif database_filter:
        specs = [
            spec
            for spec in all_specs
            if spec.kind == "sqlite" and spec.db_id == database_filter
        ]
        if len(specs) != 1:
            raise BirdReproductionError(
                f"database {database_filter!r} is not a unique published database"
            )
    if args.evict_database:
        evict_required_database(specs[0], data_dir=data_dir)
        print(f"evicted required database {args.evict_database} from {data_dir}")
        return
    if args.plan:
        for spec in specs:
            print(f"{spec.kind}\t{spec.url}\t{spec.local_path}")
        return
    if args.database:
        assert_single_database_cache(
            all_specs,
            requested_db_id=args.database,
        )
    if not args.skip_download:
        download_required_files(
            specs,
            chunk_size=args.chunk_size_mib * 1024 * 1024,
            max_file_bytes=int(args.max_file_gib * 1024**3),
        )
    else:
        missing = [str(spec.local_path) for spec in specs if not _is_valid_existing(spec)]
        if missing:
            raise BirdReproductionError(
                f"--skip-download requires every selected file; missing={missing[:3]}"
            )
    if args.database:
        print(f"prepared required database {args.database} in {data_dir}")
    else:
        selected = build_selected_file(
            manifest,
            data_dir=data_dir,
            output_path=args.output,
        )
        if [row["task_id"] for row in selected] != [row["task_id"] for row in manifest]:
            raise BirdReproductionError("selected file does not preserve published ID order")
        print(f"prepared {len(selected)} fixed published samples at {args.output}")


if __name__ == "__main__":
    try:
        main()
    except BirdReproductionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
