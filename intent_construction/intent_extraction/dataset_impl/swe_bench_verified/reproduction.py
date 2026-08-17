"""Strict state and selection rules for the published SWE construction run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REQUIRED_MODEL = "kimi-k2.6"
REQUIRED_REASONING_EFFORT = "medium"
PUBLISHED_TASK_COUNT = 50
SOURCE_POOL_COUNT = 500
COUNTERFACTUALS_PER_ARGUMENT = 2
SOURCE_PARQUET_SHA256 = (
    "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
)
CHECKPOINT_SCHEMA_VERSION = 1
ELIGIBILITY_REPAIR_PROMPT_SHA256 = hashlib.sha256(
    (
        Path(__file__).resolve().parent / "prompts/eligibility_repair.txt"
    ).read_bytes()
).hexdigest()
ELIGIBILITY_REPAIR_VERIFICATION_PROMPT_SHA256 = hashlib.sha256(
    (
        Path(__file__).resolve().parent
        / "prompts/eligibility_repair_verification.txt"
    ).read_bytes()
).hexdigest()

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVAL_MANIFEST_PATH = (
    REPO_ROOT / "intent_construction/eval_indices/swe_bench_verified_eval_ids.json"
)
DEFAULT_TASK_IDS_PATH = (
    REPO_ROOT / "intent_construction/eval_indices/swe_bench_verified_task_ids.json"
)
DEFAULT_SOURCE_PARQUET = (
    REPO_ROOT / "tmp/source-data/swe_bench_verified_princeton.parquet"
)
DEFAULT_RUN_DIR = REPO_ROOT / "reproduction/runs/swe-construction-kimi-k2.6"
DEFAULT_FINAL_OUTPUT = REPO_ROOT / "final_dataset/swe_bench_verified_final.json"

_TASK_PREFIX = "extracted-swe_bench_verified-test-"
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_ARGUMENT_CATEGORIES = frozenset(
    {"symptom", "trigger", "location", "approach", "scope", "constraint"}
)
_COUNTERFACTUAL_ELIGIBLE_CATEGORIES = frozenset(
    {"location", "approach", "scope", "constraint"}
)
_FORBIDDEN_LIMIT_ENV = (
    "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
    "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
    "LLM_COST_HARD_CAP_USD",
)
_REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "repo",
        "instance_id",
        "base_commit",
        "patch",
        "test_patch",
        "problem_statement",
        "version",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "difficulty",
    }
)


class SWEConstructionError(RuntimeError):
    """Raised when a published SWE construction invariant is violated."""


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Flush a JSON file and atomically replace its prior version."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: str | Path, *, label: str = "JSON file") -> Any:
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise SWEConstructionError(f"cannot read {label} {target}: {exc}") from exc
    if not raw.strip():
        raise SWEConstructionError(f"{label} is empty: {target}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SWEConstructionError(f"{label} is invalid JSON: {target}: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SWEConstructionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_id_for(instance_id: str) -> str:
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise SWEConstructionError(f"invalid SWE instance ID: {instance_id!r}")
    task_id = f"{_TASK_PREFIX}{instance_id}"
    if not _SAFE_TASK_ID.fullmatch(task_id):
        raise SWEConstructionError(f"unsafe SWE task ID: {task_id!r}")
    return task_id


def instance_id_for(task_id: str) -> str:
    if not isinstance(task_id, str) or not task_id.startswith(_TASK_PREFIX):
        raise SWEConstructionError(f"invalid SWE task ID: {task_id!r}")
    instance_id = task_id[len(_TASK_PREFIX) :]
    if task_id_for(instance_id) != task_id:
        raise SWEConstructionError(f"invalid SWE task ID: {task_id!r}")
    return instance_id


def assert_build_policy(
    *,
    model: str,
    reasoning_effort: str,
    environment: Mapping[str, str],
    require_credentials: bool,
) -> str:
    """Validate the fixed model/effort/unbounded-output provider policy."""
    if model != REQUIRED_MODEL:
        raise SWEConstructionError(
            f"SWE construction requires exactly {REQUIRED_MODEL!r}; got {model!r}"
        )
    if reasoning_effort != REQUIRED_REASONING_EFFORT:
        raise SWEConstructionError(
            "SWE construction reasoning effort must be exactly "
            f"{REQUIRED_REASONING_EFFORT!r}"
        )
    configured_effort = environment.get("LLM_REASONING_EFFORT", "").strip()
    if configured_effort and configured_effort != REQUIRED_REASONING_EFFORT:
        raise SWEConstructionError(
            "LLM_REASONING_EFFORT conflicts with the fixed medium policy"
        )
    disabled = environment.get("LLM_DISABLE_OUTPUT_LIMITS", "").strip().lower()
    if disabled and disabled not in _TRUE_VALUES:
        raise SWEConstructionError("LLM_DISABLE_OUTPUT_LIMITS must be enabled")
    forbidden = [
        key for key in _FORBIDDEN_LIMIT_ENV if environment.get(key, "").strip()
    ]
    if forbidden:
        raise SWEConstructionError(
            "output limits and cost gates are forbidden: " + ", ".join(forbidden)
        )

    raw_map = environment.get("LLM_MODEL_MAP", "").strip()
    model_map: dict[str, Any] = {}
    if raw_map:
        try:
            parsed = json.loads(raw_map)
        except json.JSONDecodeError as exc:
            raise SWEConstructionError("LLM_MODEL_MAP must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise SWEConstructionError("LLM_MODEL_MAP must be a JSON object")
        model_map = parsed
    resolved = model_map.get(REQUIRED_MODEL, REQUIRED_MODEL)
    if not isinstance(resolved, str) or not resolved.strip():
        raise SWEConstructionError(
            f"LLM_MODEL_MAP[{REQUIRED_MODEL!r}] must be a non-empty string"
        )
    if not require_credentials:
        return resolved

    backend = environment.get("LLM_BACKEND", "").strip().lower()
    if backend not in {
        "compatible",
        "openai-compatible",
        "openai_compatible",
        "generic",
    }:
        raise SWEConstructionError("live SWE construction requires LLM_BACKEND=compatible")
    if not environment.get("LLM_API_KEY", "").strip() or not environment.get(
        "LLM_BASE_URL", ""
    ).strip():
        raise SWEConstructionError("live SWE construction needs LLM_API_KEY and LLM_BASE_URL")
    if disabled not in _TRUE_VALUES:
        raise SWEConstructionError("live SWE construction requires LLM_DISABLE_OUTPUT_LIMITS=1")
    if environment.get("LLM_REQUIRE_USAGE_ACCOUNTING", "").strip().lower() not in _TRUE_VALUES:
        raise SWEConstructionError("live SWE construction requires usage accounting")
    if not environment.get("LLM_USAGE_LEDGER_PATH", "").strip():
        raise SWEConstructionError("live SWE construction requires LLM_USAGE_LEDGER_PATH")

    raw_prices = environment.get("LLM_PRICE_MAP", "").strip()
    try:
        price_map = json.loads(raw_prices)
    except json.JSONDecodeError as exc:
        raise SWEConstructionError("LLM_PRICE_MAP must be valid JSON") from exc
    if not isinstance(price_map, dict):
        raise SWEConstructionError("LLM_PRICE_MAP must be a JSON object")
    prices = next(
        (
            price_map[key]
            for key in (REQUIRED_MODEL, resolved, "*", "default")
            if key in price_map
        ),
        None,
    )
    if not isinstance(prices, Mapping):
        raise SWEConstructionError("LLM_PRICE_MAP has no Kimi price entry")
    for field in ("input", "output"):
        try:
            value = float(prices[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise SWEConstructionError(
                f"LLM_PRICE_MAP requires a numeric {field!r} price"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise SWEConstructionError(
                f"LLM_PRICE_MAP {field!r} price must be finite and non-negative"
            )
    return resolved


def load_published_manifest(
    manifest_path: str | Path = DEFAULT_EVAL_MANIFEST_PATH,
    task_ids_path: str | Path = DEFAULT_TASK_IDS_PATH,
) -> list[dict[str, str]]:
    manifest = read_json(manifest_path, label="published SWE eval manifest")
    samples = manifest.get("samples") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(samples, list)
        or len(samples) != PUBLISHED_TASK_COUNT
        or manifest.get("num_samples") != PUBLISHED_TASK_COUNT
    ):
        raise SWEConstructionError(
            f"published SWE manifest must contain exactly {PUBLISHED_TASK_COUNT} samples"
        )
    normalized: list[dict[str, str]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise SWEConstructionError(f"published SWE sample {index} is not an object")
        task_id = sample.get("task_id")
        original_id = sample.get("original_id")
        if not isinstance(task_id, str) or not isinstance(original_id, str):
            raise SWEConstructionError(f"published SWE sample {index} has invalid IDs")
        if task_id != task_id_for(original_id):
            raise SWEConstructionError(
                f"published task/original ID mapping is invalid for {task_id!r}"
            )
        normalized.append({"task_id": task_id, "original_id": original_id})
    task_ids = [sample["task_id"] for sample in normalized]
    original_ids = [sample["original_id"] for sample in normalized]
    if len(set(task_ids)) != PUBLISHED_TASK_COUNT or len(set(original_ids)) != PUBLISHED_TASK_COUNT:
        raise SWEConstructionError("published SWE manifest contains duplicate IDs")

    task_payload = read_json(task_ids_path, label="published SWE task-ID file")
    expected = task_payload.get("task_ids") if isinstance(task_payload, Mapping) else None
    if expected != task_ids or task_payload.get("n_total") != PUBLISHED_TASK_COUNT:
        raise SWEConstructionError(
            "published SWE task-ID file differs from the eval manifest"
        )
    return normalized


def load_source_pool(
    path: str | Path,
    *,
    expected_sha256: str | None = SOURCE_PARQUET_SHA256,
) -> tuple[list[dict[str, Any]], str]:
    source = Path(path).resolve()
    if not source.is_file() or source.suffix != ".parquet":
        raise SWEConstructionError(f"SWE source must be a local parquet file: {source}")
    digest = file_sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SWEConstructionError(
            f"SWE source SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )
    try:
        import pyarrow.dataset as arrow_dataset
    except ImportError as exc:
        raise SWEConstructionError("local SWE parquet selection requires pyarrow") from exc
    try:
        dataset = arrow_dataset.dataset(source, format="parquet")
        missing = _REQUIRED_SOURCE_FIELDS - set(dataset.schema.names)
        if missing:
            raise SWEConstructionError(
                f"SWE source parquet is missing fields: {sorted(missing)}"
            )
        rows = dataset.scanner().to_table().to_pylist()
    except SWEConstructionError:
        raise
    except Exception as exc:
        raise SWEConstructionError(f"cannot read SWE source parquet: {exc}") from exc
    if len(rows) != SOURCE_POOL_COUNT:
        raise SWEConstructionError(
            f"SWE source pool must contain exactly {SOURCE_POOL_COUNT} rows; found {len(rows)}"
        )
    ids = [row.get("instance_id") for row in rows]
    if not all(isinstance(instance_id, str) and instance_id for instance_id in ids):
        raise SWEConstructionError("SWE source pool contains a missing instance_id")
    if len(set(ids)) != SOURCE_POOL_COUNT:
        raise SWEConstructionError("SWE source pool contains duplicate instance IDs")
    return [dict(row) for row in rows], digest


def affected_files(row: Mapping[str, Any]) -> list[str]:
    patch = row.get("patch", "")
    if not isinstance(patch, str):
        return []
    return sorted(set(re.findall(r"^diff --git a/(.*?) b/", patch, re.MULTILINE)))


def _match_key(row: Mapping[str, Any], level: int) -> tuple[str, str]:
    repo = str(row.get("repo") or "")
    if level == 0:
        return repo, ""
    files = affected_files(row)
    if not files:
        return repo, ""
    return repo, "/".join(files[0].split("/")[:level])


@dataclass(frozen=True)
class BuildSelection:
    target_rows: tuple[dict[str, Any], ...]
    extra_rows: tuple[dict[str, Any], ...]
    extraction_rows: tuple[dict[str, Any], ...]
    pairings: tuple[dict[str, Any], ...]

    @property
    def target_task_ids(self) -> list[str]:
        return [task_id_for(row["instance_id"]) for row in self.target_rows]

    @property
    def extraction_task_ids(self) -> list[str]:
        return [task_id_for(row["instance_id"]) for row in self.extraction_rows]

    def manifest(self, *, source_path: str | Path, source_sha256: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": {
                "path": str(Path(source_path).resolve()),
                "sha256": source_sha256,
                "pool_rows": SOURCE_POOL_COUNT,
            },
            "model": REQUIRED_MODEL,
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
            "output_token_limit": None,
            "published_target_count": len(self.target_rows),
            "extra_candidate_count": len(self.extra_rows),
            "extraction_count": len(self.extraction_rows),
            "target_instance_ids": [row["instance_id"] for row in self.target_rows],
            "target_task_ids": self.target_task_ids,
            "extra_candidate_instance_ids": [row["instance_id"] for row in self.extra_rows],
            "extraction_instance_ids": [row["instance_id"] for row in self.extraction_rows],
            "pairings": [dict(pairing) for pairing in self.pairings],
        }


def select_build_rows(
    source_rows: Sequence[Mapping[str, Any]],
    published: Sequence[Mapping[str, str]],
) -> BuildSelection:
    """Select 50 targets plus one minimal external candidate per uncovered area."""
    if len(source_rows) != SOURCE_POOL_COUNT:
        raise SWEConstructionError(
            f"candidate selection requires the complete {SOURCE_POOL_COUNT}-row source pool"
        )
    rows_by_id = {str(row.get("instance_id")): dict(row) for row in source_rows}
    if len(rows_by_id) != SOURCE_POOL_COUNT:
        raise SWEConstructionError("candidate selection source IDs are not unique")
    target_ids = [sample["original_id"] for sample in published]
    missing = [instance_id for instance_id in target_ids if instance_id not in rows_by_id]
    if missing:
        raise SWEConstructionError(f"published targets are missing from parquet: {missing[:5]}")
    target_rows = [rows_by_id[instance_id] for instance_id in target_ids]
    target_set = set(target_ids)

    extras_by_id: dict[str, dict[str, Any]] = {}
    # A singleton published area gets one external area match when the 500-pool
    # has one. Areas already represented twice need no extra extraction.
    for target in target_rows:
        target_id = target["instance_id"]
        area_key = _match_key(target, 2)
        if any(
            other["instance_id"] != target_id and _match_key(other, 2) == area_key
            for other in target_rows
        ):
            continue
        candidates = sorted(
            (
                row
                for row in source_rows
                if row["instance_id"] not in target_set
                and _match_key(row, 2) == area_key
            ),
            key=lambda row: row["instance_id"],
        )
        if candidates:
            candidate = dict(candidates[0])
            extras_by_id[candidate["instance_id"]] = candidate

    # If an entire repo has only one published target (seaborn in the fixed
    # subset), add the narrowest available external match even when no area
    # candidate was selected above.
    available = target_rows + list(extras_by_id.values())
    for target in target_rows:
        target_id = target["instance_id"]
        if any(
            other["instance_id"] != target_id
            and _match_key(other, level) == _match_key(target, level)
            for level in (2, 1, 0)
            for other in available
        ):
            continue
        for level in (2, 1, 0):
            candidates = sorted(
                (
                    row
                    for row in source_rows
                    if row["instance_id"] not in target_set
                    and _match_key(row, level) == _match_key(target, level)
                ),
                key=lambda row: row["instance_id"],
            )
            if candidates:
                candidate = dict(candidates[0])
                extras_by_id[candidate["instance_id"]] = candidate
                available = target_rows + list(extras_by_id.values())
                break
        else:
            raise SWEConstructionError(f"no real-bug pair exists for {target_id}")

    extra_rows = [extras_by_id[instance_id] for instance_id in sorted(extras_by_id)]
    available = target_rows + extra_rows
    pairings: list[dict[str, Any]] = []
    labels = {2: "area", 1: "folder", 0: "repo"}
    for target in target_rows:
        target_id = target["instance_id"]
        for level in (2, 1, 0):
            candidates = sorted(
                (
                    row
                    for row in available
                    if row["instance_id"] != target_id
                    and _match_key(row, level) == _match_key(target, level)
                ),
                key=lambda row: row["instance_id"],
            )
            if candidates:
                paired = candidates[0]
                pairings.append(
                    {
                        "target_instance_id": target_id,
                        "target_task_id": task_id_for(target_id),
                        "paired_instance_id": paired["instance_id"],
                        "paired_task_id": task_id_for(paired["instance_id"]),
                        "match_level": labels[level],
                        "paired_is_published_target": paired["instance_id"] in target_set,
                    }
                )
                break
        else:
            raise SWEConstructionError(f"selected pool has no pair for {target_id}")
    if len(pairings) != PUBLISHED_TASK_COUNT:
        raise SWEConstructionError("pairing plan does not cover exactly 50 targets")
    seaborn = [pairing for pairing in pairings if "seaborn" in pairing["target_instance_id"]]
    if len(seaborn) != 1 or seaborn[0]["paired_instance_id"] == seaborn[0]["target_instance_id"]:
        raise SWEConstructionError("the unique seaborn target has no distinct pair")

    extraction_rows = target_rows + extra_rows
    if len({row["instance_id"] for row in extraction_rows}) != len(extraction_rows):
        raise SWEConstructionError("target/candidate extraction selection is not deduplicated")
    return BuildSelection(
        target_rows=tuple(target_rows),
        extra_rows=tuple(extra_rows),
        extraction_rows=tuple(extraction_rows),
        pairings=tuple(pairings),
    )


def source_row_to_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    instance_id = str(row.get("instance_id") or "")
    if not instance_id:
        raise SWEConstructionError("selected source row has no instance_id")
    for field in ("repo", "base_commit", "patch", "problem_statement"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise SWEConstructionError(f"{instance_id}: source field {field!r} is empty")
    return {
        "id": instance_id,
        "instance_id": instance_id,
        "question": row["problem_statement"],
        "answer": "",
        "task": "swe_bench",
        "split": "test",
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "patch": row["patch"],
        "test_patch": row.get("test_patch", ""),
        "version": row.get("version", ""),
        "difficulty": row.get("difficulty", ""),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", "[]"),
        "PASS_TO_PASS": row.get("PASS_TO_PASS", "[]"),
        "hints_text": row.get("hints_text", ""),
        "environment_setup_commit": row.get("environment_setup_commit", ""),
    }


def _require_nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SWEConstructionError(f"{label} is empty")
    return value


def validate_extraction_row(row: Any, *, task_id: str) -> dict[str, Any]:
    if not isinstance(row, Mapping) or not row:
        raise SWEConstructionError(f"stage1 result is empty for {task_id}")
    result = dict(row)
    if result.get("task_id") != task_id or result.get("original_id") != instance_id_for(task_id):
        raise SWEConstructionError(f"stage1 IDs do not match {task_id}")
    if result.get("model_name") != REQUIRED_MODEL:
        raise SWEConstructionError(f"stage1 model mismatch for {task_id}")
    _require_nonempty_text(result.get("function"), label=f"stage1 function for {task_id}")
    _require_nonempty_text(
        result.get("fully_specified_question"),
        label=f"stage1 fully specified question for {task_id}",
    )
    arguments = result.get("arguments")
    if not isinstance(arguments, list) or not arguments:
        raise SWEConstructionError(f"stage1 arguments are empty for {task_id}")
    argument_ids: list[int] = []
    for argument in arguments:
        if not isinstance(argument, Mapping):
            raise SWEConstructionError(f"stage1 argument is malformed for {task_id}")
        argument_id = argument.get("argument_id")
        if not isinstance(argument_id, int):
            raise SWEConstructionError(f"stage1 argument ID is invalid for {task_id}")
        argument_ids.append(argument_id)
        _require_nonempty_text(
            argument.get("argument"), label=f"stage1 argument text for {task_id}"
        )
        category = argument.get("category")
        if category not in _ARGUMENT_CATEGORIES:
            raise SWEConstructionError(f"stage1 argument category is invalid for {task_id}")
        if argument.get("counterfactual_eligible") is not (
            category in _COUNTERFACTUAL_ELIGIBLE_CATEGORIES
        ):
            raise SWEConstructionError(
                f"stage1 argument eligibility is inconsistent for {task_id}"
            )
    if len(set(argument_ids)) != len(argument_ids):
        raise SWEConstructionError(f"stage1 argument IDs are duplicated for {task_id}")
    if result.get("num_arguments") != len(arguments):
        raise SWEConstructionError(f"stage1 argument count is inconsistent for {task_id}")
    metadata = result.get("swe_bench_metadata")
    if not isinstance(metadata, Mapping):
        raise SWEConstructionError(f"stage1 SWE metadata is missing for {task_id}")
    _require_nonempty_text(metadata.get("repo"), label=f"stage1 repo for {task_id}")
    files = metadata.get("affected_files")
    if not isinstance(files, list) or not all(isinstance(path, str) for path in files):
        raise SWEConstructionError(f"stage1 affected files are malformed for {task_id}")
    return result


def validate_eligible_extraction_row(row: Any, *, task_id: str) -> dict[str, Any]:
    """Validate the explicit Stage 1b output used by argument revision."""
    result = validate_extraction_row(row, task_id=task_id)
    eligible = [
        argument
        for argument in result["arguments"]
        if argument["category"] in _COUNTERFACTUAL_ELIGIBLE_CATEGORIES
    ]
    if not eligible:
        raise SWEConstructionError(f"stage1b has no eligible argument for {task_id}")

    repair = result.get("eligibility_repair_info")
    if repair is None:
        return result
    if not isinstance(repair, Mapping):
        raise SWEConstructionError(f"stage1b repair metadata is malformed for {task_id}")
    if (
        repair.get("model") != REQUIRED_MODEL
        or repair.get("reasoning_effort") != REQUIRED_REASONING_EFFORT
        or repair.get("source") != "problem_statement"
        or repair.get("category") not in _COUNTERFACTUAL_ELIGIBLE_CATEGORIES
        or repair.get("reason") != "no_counterfactual_eligible_argument"
        or repair.get("repair_version") != "eligible_argument_v1"
        or repair.get("prompt_sha256") != ELIGIBILITY_REPAIR_PROMPT_SHA256
        or repair.get("verifier_decision") != "full_pipeline_passed"
        or repair.get("coverage_verification") is not True
        or repair.get("patch_alignment_verification") is not True
    ):
        raise SWEConstructionError(f"stage1b repair policy mismatch for {task_id}")
    grounding_verifier = repair.get("grounding_verifier")
    if (
        not isinstance(grounding_verifier, Mapping)
        or grounding_verifier.get("decision") != "passed"
        or grounding_verifier.get("model") != REQUIRED_MODEL
        or grounding_verifier.get("reasoning_effort")
        != REQUIRED_REASONING_EFFORT
        or grounding_verifier.get("prompt_sha256")
        != ELIGIBILITY_REPAIR_VERIFICATION_PROMPT_SHA256
        or not isinstance(grounding_verifier.get("reasoning"), str)
    ):
        raise SWEConstructionError(
            f"stage1b grounding verifier metadata mismatch for {task_id}"
        )
    added_argument_id = repair.get("added_argument_id")
    added = next(
        (
            argument
            for argument in result["arguments"]
            if argument.get("argument_id") == added_argument_id
        ),
        None,
    )
    if not isinstance(added, Mapping) or added.get("eligibility_repair") is not True:
        raise SWEConstructionError(f"stage1b repaired argument is missing for {task_id}")
    if added.get("category") != repair.get("category"):
        raise SWEConstructionError(f"stage1b repaired category mismatch for {task_id}")
    grounding_quote = _require_nonempty_text(
        repair.get("grounding_quote"),
        label=f"stage1b grounding quote for {task_id}",
    )
    question = str(result.get("question") or "")
    if repair.get("source_sha256") != hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest():
        raise SWEConstructionError(f"stage1b source hash mismatch for {task_id}")
    quote_start = repair.get("grounding_quote_start")
    quote_end = repair.get("grounding_quote_end")
    if (
        not isinstance(quote_start, int)
        or not isinstance(quote_end, int)
        or quote_start < 0
        or quote_end <= quote_start
        or question[quote_start:quote_end] != grounding_quote
    ):
        raise SWEConstructionError(f"stage1b grounding quote is not in the issue for {task_id}")
    mutable_span = _require_nonempty_text(
        repair.get("mutable_span"),
        label=f"stage1b mutable span for {task_id}",
    )
    if mutable_span not in str(added.get("argument") or ""):
        raise SWEConstructionError(f"stage1b mutable span is not in the argument for {task_id}")
    return result


def validate_eligibility_repair_delta(
    row: Any,
    *,
    task_id: str,
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Restrict Stage 1b to identity or one audited appended argument."""
    source = validate_extraction_row(source_row, task_id=task_id)
    result = validate_eligible_extraction_row(row, task_id=task_id)
    source_has_eligible = any(
        argument["category"] in _COUNTERFACTUAL_ELIGIBLE_CATEGORIES
        for argument in source["arguments"]
    )
    if source_has_eligible:
        if result != source:
            raise SWEConstructionError(
                f"stage1b changed an already-eligible extraction for {task_id}"
            )
        return result

    allowed_changes = {
        "arguments",
        "num_arguments",
        "fully_specified_question",
        "eligibility_repair_info",
    }
    for key in set(source) - allowed_changes:
        if key not in result or result[key] != source[key]:
            raise SWEConstructionError(f"stage1b changed {key!r} for {task_id}")
    unexpected = set(result) - set(source) - {"eligibility_repair_info"}
    if unexpected:
        raise SWEConstructionError(
            f"stage1b added unexpected fields for {task_id}: {sorted(unexpected)}"
        )
    if len(result["arguments"]) != len(source["arguments"]) + 1:
        raise SWEConstructionError(
            f"stage1b must append exactly one argument for {task_id}"
        )
    if result["arguments"][:-1] != source["arguments"]:
        raise SWEConstructionError(f"stage1b changed existing arguments for {task_id}")
    added = result["arguments"][-1]
    source_ids = [argument["argument_id"] for argument in source["arguments"]]
    if added.get("argument_id") != max(source_ids, default=0) + 1:
        raise SWEConstructionError(f"stage1b argument ID is not append-only for {task_id}")
    if sum(
        argument.get("eligibility_repair") is True
        for argument in result["arguments"]
    ) != 1:
        raise SWEConstructionError(f"stage1b repair marker count is invalid for {task_id}")
    expected_fully_specified = " ".join(
        [result["function"]]
        + [argument["argument"] for argument in result["arguments"]]
    ).strip()
    if result["fully_specified_question"] != expected_fully_specified:
        raise SWEConstructionError(
            f"stage1b fully specified question is inconsistent for {task_id}"
        )
    return result


def validate_counterfactual_row(
    row: Any,
    *,
    task_id: str,
    required_count: int = COUNTERFACTUALS_PER_ARGUMENT,
) -> dict[str, Any]:
    result = validate_eligible_extraction_row(row, task_id=task_id)
    info = result.get("counterfactual_info")
    if not isinstance(info, Mapping):
        raise SWEConstructionError(f"stage2 metadata is missing for {task_id}")
    if (
        info.get("model") != REQUIRED_MODEL
        or info.get("reasoning_effort") != REQUIRED_REASONING_EFFORT
        or info.get("num_counterfactuals_requested") != required_count
    ):
        raise SWEConstructionError(f"stage2 policy metadata mismatch for {task_id}")
    eligible_count = 0
    for argument in result["arguments"]:
        eligible = bool(argument.get("counterfactual_eligible"))
        counterfactuals = argument.get("counterfactual_arguments")
        if not isinstance(counterfactuals, list):
            raise SWEConstructionError(f"stage2 counterfactual list is missing for {task_id}")
        if not eligible:
            if counterfactuals:
                raise SWEConstructionError(
                    f"stage2 changed an ineligible argument for {task_id}"
                )
            continue
        eligible_count += 1
        if len(counterfactuals) != required_count:
            raise SWEConstructionError(
                f"stage2 has incomplete counterfactual coverage for {task_id}"
            )
        for counterfactual in counterfactuals:
            if not isinstance(counterfactual, Mapping):
                raise SWEConstructionError(f"stage2 counterfactual is malformed for {task_id}")
            _require_nonempty_text(
                counterfactual.get("counterfactual_argument"),
                label=f"stage2 counterfactual text for {task_id}",
            )
    if eligible_count == 0:
        raise SWEConstructionError(f"stage2 has no eligible argument for {task_id}")
    return result


def validate_pair_row(
    row: Any,
    *,
    task_id: str,
    pairing: Mapping[str, Any],
) -> dict[str, Any]:
    result = validate_counterfactual_row(row, task_id=task_id)
    predecessors = result.get("predecessor_functions")
    if not isinstance(predecessors, list) or len(predecessors) != 1:
        raise SWEConstructionError(f"stage3 pair is missing for {task_id}")
    pair = predecessors[0]
    if not isinstance(pair, Mapping) or pair.get("transition_type") != "real_bug_pair":
        raise SWEConstructionError(f"stage3 pair is malformed for {task_id}")
    _require_nonempty_text(
        pair.get("predecessor_function"), label=f"stage3 paired function for {task_id}"
    )
    metadata = pair.get("_pair_metadata")
    if not isinstance(metadata, Mapping):
        raise SWEConstructionError(f"stage3 pair metadata is missing for {task_id}")
    expected = {
        "match_level": pairing["match_level"],
        "paired_task_id": pairing["paired_task_id"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SWEConstructionError(f"stage3 pair {key} mismatch for {task_id}")
    return result


def validate_g1_row(row: Any, *, task_id: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise SWEConstructionError(f"stage4 result is empty for {task_id}")
    if row.get("task_id") != task_id:
        raise SWEConstructionError(f"stage4 task ID mismatch for {task_id}")
    predecessors = row.get("predecessor_functions")
    if not isinstance(predecessors, list) or len(predecessors) != 2:
        raise SWEConstructionError(f"stage4 predecessor chain is incomplete for {task_id}")
    if predecessors[0].get("transition_type") != "real_bug_pair":
        raise SWEConstructionError(f"stage4 lost the real-bug pair for {task_id}")
    if predecessors[1].get("transition_type") != "exploration":
        raise SWEConstructionError(f"stage4 exploration entry is missing for {task_id}")
    _require_nonempty_text(
        predecessors[1].get("predecessor_function"),
        label=f"stage4 exploration function for {task_id}",
    )
    info = row.get("g1_info")
    if not isinstance(info, Mapping) or info.get("model") != REQUIRED_MODEL or info.get(
        "reasoning_effort"
    ) != REQUIRED_REASONING_EFFORT:
        raise SWEConstructionError(f"stage4 policy metadata mismatch for {task_id}")
    return dict(row)


def validate_final_row(row: Any, *, task_id: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise SWEConstructionError(f"stage5 result is empty for {task_id}")
    if row.get("task_id") != task_id:
        raise SWEConstructionError(f"stage5 task ID mismatch for {task_id}")
    predecessors = row.get("predecessor_functions")
    if not isinstance(predecessors, list) or len(predecessors) != 2:
        raise SWEConstructionError(f"stage5 predecessor chain is incomplete for {task_id}")
    impl = predecessors[0]
    if impl.get("transition_type") != "impl_precursor":
        raise SWEConstructionError(f"stage5 implementation precursor is missing for {task_id}")
    if predecessors[1].get("transition_type") != "exploration":
        raise SWEConstructionError(f"stage5 exploration entry is missing for {task_id}")
    _require_nonempty_text(
        impl.get("predecessor_function"),
        label=f"stage5 implementation request for {task_id}",
    )
    arguments = impl.get("counterfactual_arguments")
    if not isinstance(arguments, list) or not arguments:
        raise SWEConstructionError(f"stage5 implementation arguments are empty for {task_id}")
    if any(
        not isinstance(argument, Mapping)
        or not str(argument.get("argument", "")).strip()
        for argument in arguments
    ):
        raise SWEConstructionError(f"stage5 implementation arguments are malformed for {task_id}")
    info = row.get("impl_precursor_info")
    if not isinstance(info, Mapping) or info.get("model") != REQUIRED_MODEL or info.get(
        "reasoning_effort"
    ) != REQUIRED_REASONING_EFFORT:
        raise SWEConstructionError(f"stage5 policy metadata mismatch for {task_id}")
    return dict(row)


def validate_exact_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_task_ids: Sequence[str],
    stage: str,
    validator: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SWEConstructionError(f"{stage} row {position} is not an object")
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            raise SWEConstructionError(f"{stage} row {position} has no task_id")
        if task_id in indexed:
            raise SWEConstructionError(f"{stage} contains duplicate task ID {task_id}")
        indexed[task_id] = row
    expected = list(expected_task_ids)
    missing = [task_id for task_id in expected if task_id not in indexed]
    extras = [task_id for task_id in indexed if task_id not in set(expected)]
    if missing or extras or len(indexed) != len(expected):
        raise SWEConstructionError(
            f"{stage} coverage failure: expected={len(expected)}, found={len(indexed)}, "
            f"missing={missing[:5]}, extras={extras[:5]}"
        )
    return [validator(indexed[task_id], task_id=task_id) for task_id in expected]


class StageCheckpointStore:
    """One validated, atomic checkpoint file per task ID."""

    def __init__(
        self,
        directory: str | Path,
        *,
        stage: str,
        inputs_by_id: Mapping[str, Mapping[str, Any]],
        validator: Callable[..., dict[str, Any]],
        model: str = REQUIRED_MODEL,
        reasoning_effort: str = REQUIRED_REASONING_EFFORT,
    ) -> None:
        if model != REQUIRED_MODEL or reasoning_effort != REQUIRED_REASONING_EFFORT:
            raise SWEConstructionError("checkpoint policy differs from fixed Kimi/medium")
        self.directory = Path(directory)
        self.stage = stage
        self.inputs_by_id = {task_id: dict(value) for task_id, value in inputs_by_id.items()}
        self.expected_ids = list(self.inputs_by_id)
        self.validator = validator
        self.model = model
        self.reasoning_effort = reasoning_effort
        if not self.expected_ids or len(set(self.expected_ids)) != len(self.expected_ids):
            raise SWEConstructionError(f"{stage} checkpoint expected IDs are empty or duplicated")
        for task_id in self.expected_ids:
            if not _SAFE_TASK_ID.fullmatch(task_id):
                raise SWEConstructionError(f"unsafe checkpoint task ID: {task_id!r}")
        self.directory.mkdir(parents=True, exist_ok=True)
        expected_names = {f"{task_id}.json" for task_id in self.expected_ids}
        extras = sorted(
            path.name
            for path in self.directory.glob("*.json")
            if path.name not in expected_names
        )
        if extras:
            raise SWEConstructionError(
                f"{stage} checkpoint directory contains unexpected files: {extras[:5]}"
            )

    def path_for(self, task_id: str) -> Path:
        if task_id not in self.inputs_by_id:
            raise SWEConstructionError(f"{self.stage} received unexpected task ID {task_id!r}")
        return self.directory / f"{task_id}.json"

    def _input_hash(self, task_id: str) -> str:
        return payload_sha256(self.inputs_by_id[task_id])

    def load(self, task_id: str) -> dict[str, Any]:
        path = self.path_for(task_id)
        envelope = read_json(path, label=f"{self.stage} task checkpoint")
        if not isinstance(envelope, Mapping) or not envelope:
            raise SWEConstructionError(f"{self.stage} checkpoint is empty: {path}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": self.stage,
            "task_id": task_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "input_sha256": self._input_hash(task_id),
        }
        for key, value in expected.items():
            if envelope.get(key) != value:
                raise SWEConstructionError(
                    f"{self.stage} checkpoint {key} mismatch for {task_id}"
                )
        return self.validator(envelope.get("result"), task_id=task_id)

    def write(self, task_id: str, result: Mapping[str, Any]) -> Path:
        validated = self.validator(result, task_id=task_id)
        path = self.path_for(task_id)
        atomic_write_json(
            path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "stage": self.stage,
                "task_id": task_id,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "input_sha256": self._input_hash(task_id),
                "result": validated,
            },
        )
        return path

    def load_available(self) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        for task_id in self.expected_ids:
            if self.path_for(task_id).exists():
                completed[task_id] = self.load(task_id)
        return completed

    def require_complete(self) -> list[dict[str, Any]]:
        completed = self.load_available()
        missing = [task_id for task_id in self.expected_ids if task_id not in completed]
        if missing:
            raise SWEConstructionError(
                f"{self.stage} checkpoint coverage is incomplete: {missing[:5]}"
            )
        return [completed[task_id] for task_id in self.expected_ids]
