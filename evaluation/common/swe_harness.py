"""
SWE-bench Verified harness wrapper.

Stateless, side-effect-isolated wrapper around the official `swebench` library.

Public surface:
    HarnessResult        — dataclass returned by verify_patch
    SWEHarness          — wrapper class, holds workspace paths + dataset name
    verify_patch(...)   — convenience module-level function (uses default harness)

Design notes:
- All Docker / image / log artifacts live under `evaluation/swe_workspace/`.
  The harness library writes to `RUN_EVALUATION_LOG_DIR` (= "logs/run_evaluation")
  *relative to cwd*; we chdir into the workspace for the duration of a call so
  nothing leaks into the repo root.
- Results are cached by (instance_id, sha1(patch)) under
  `evaluation/swe_workspace/cache/`. A cache hit short-circuits the harness
  call entirely.
- We never raise on harness failures; we record them in
  HarnessResult.harness_error and return correct=False. The caller is
  responsible for deciding whether to retry.
- Concurrency: the wrapper itself is NOT thread-safe (it chdirs). Callers
  that need parallel scoring should use a pool of harness instances or
  serialize through a single instance. `evaluate_sample` is currently
  serial w.r.t. scoring even when LLM calls are parallel; that matches
  the existing IF / SQL paths.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default workspace directory (sibling to common/, runners/, experiments/).
_DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent / "swe_workspace"

# Default HuggingFace dataset name for SWE-bench Verified.
_DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
_DEFAULT_SPLIT = "test"


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class HarnessResult:
    """Result of running a single (instance_id, patch) through the harness."""

    instance_id: str
    resolved: bool
    # Patch lifecycle flags. Lets analysis tooling distinguish "model produced
    # no patch" from "patch didn't apply" from "tests failed".
    patch_extracted: bool = True
    patch_apply_ok: bool | None = None  # None if not yet attempted
    # Test outcomes (raw lists from swebench report).
    ftp_pass: list[str] = field(default_factory=list)
    ftp_fail: list[str] = field(default_factory=list)
    ptp_pass: list[str] = field(default_factory=list)
    ptp_fail: list[str] = field(default_factory=list)
    # Diagnostics.
    harness_error: str | None = None
    duration_s: float = 0.0
    # Cache provenance.
    from_cache: bool = False
    # Raw report (for debugging / future re-grading); may be None on error.
    raw_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HarnessResult:
        # Allow old/forward-compat fields without crashing.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def empty_no_patch(cls, instance_id: str) -> HarnessResult:
        """Construct a no-op result for an empty / missing patch."""
        return cls(
            instance_id=instance_id,
            resolved=False,
            patch_extracted=False,
            patch_apply_ok=False,
            harness_error="no_patch_extracted",
        )


# =============================================================================
# Cache layer
# =============================================================================


def _cache_key(instance_id: str, patch: str) -> str:
    h = hashlib.sha1(patch.encode("utf-8")).hexdigest()[:16]
    # Filesystem-safe instance id (covers `org/repo-NNN` shapes if any).
    safe = instance_id.replace("/", "__")
    return f"{safe}__{h}.json"


# =============================================================================
# Harness wrapper
# =============================================================================


class SWEHarness:
    """Stateful workspace; stateless verify_patch (per-call cache lookup)."""

    def __init__(
        self,
        workspace: Path | str | None = None,
        dataset_name: str = _DEFAULT_DATASET,
        split: str = _DEFAULT_SPLIT,
        timeout_s: int = 1800,
        cache_level: str = "env",  # "none" | "base" | "env" | "instance"
        use_cache: bool = True,
        namespace: str | None = "swebench",  # canonical leaderboard default
    ) -> None:
        self.workspace = Path(workspace) if workspace else _DEFAULT_WORKSPACE
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.workspace / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.dataset_name = dataset_name
        self.split = split
        self.timeout_s = timeout_s
        self.cache_level = cache_level
        self.use_cache = use_cache
        # Process-wide lock around the chdir + swebench_main call. swebench's
        # run_evaluation uses relative paths and per-image build directories
        # that are not safe to invoke concurrently from multiple threads in
        # the same process. The cache layer means most calls short-circuit;
        # actual harness work is the bottleneck regardless.
        self._harness_lock = threading.Lock()

    # ---- cache I/O ---------------------------------------------------------

    def _cache_path(self, instance_id: str, patch: str) -> Path:
        return self.cache_dir / _cache_key(instance_id, patch)

    def _read_cache(self, instance_id: str, patch: str) -> HarnessResult | None:
        p = self._cache_path(instance_id, patch)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except Exception:
            logger.warning("Corrupt cache file %s; ignoring", p)
            return None
        result = HarnessResult.from_dict(data)
        result.from_cache = True
        return result

    def _write_cache(self, result: HarnessResult, patch: str) -> None:
        p = self._cache_path(result.instance_id, patch)
        # Don't cache results that came from cache (no-op write).
        try:
            p.write_text(json.dumps(result.to_dict(), indent=2))
        except Exception:
            logger.warning("Failed to write cache %s", p, exc_info=True)

    # ---- main entry --------------------------------------------------------

    def verify_patch(
        self,
        instance_id: str,
        patch: str,
        model_name: str = "evolvingintent",
        run_id: str | None = None,
    ) -> HarnessResult:
        """Apply `patch` for `instance_id` and run FAIL_TO_PASS / PASS_TO_PASS.

        Returns a HarnessResult. Never raises; harness errors are captured in
        `harness_error` and `resolved=False`.

        `model_name` only affects the per-run log subdirectory; correctness
        is independent of it. `run_id` defaults to a stable hash so reruns
        of the same patch land in the same log dir (and the swebench
        in-tree report.json cache short-circuits).
        """
        if not patch or not patch.strip():
            return HarnessResult.empty_no_patch(instance_id)

        # Cache lookup (short-circuit if seen before).
        if self.use_cache:
            cached = self._read_cache(instance_id, patch)
            if cached is not None:
                return cached

        # Default run_id ties to patch hash so swebench's own report.json
        # cache lines up with our cache key (safe to re-call with identical
        # input).
        if run_id is None:
            run_id = "ei_" + hashlib.sha1(patch.encode("utf-8")).hexdigest()[:12]

        start = time.time()
        result = self._run_harness(instance_id, patch, model_name, run_id)
        result.duration_s = time.time() - start

        if self.use_cache:
            self._write_cache(result, patch)
        return result

    # ---- harness invocation ------------------------------------------------

    def _run_harness(
        self,
        instance_id: str,
        patch: str,
        model_name: str,
        run_id: str,
    ) -> HarnessResult:
        """Invoke swebench.harness.run_evaluation.main and parse its report.

        We chdir into self.workspace so all relative-path artifacts
        ("logs/run_evaluation/...", final report file) land there.
        """
        # Lazy import: avoid swebench import cost on non-SWE eval paths.
        try:
            from swebench.harness.run_evaluation import main as swebench_main
            from swebench.harness.constants import (
                RUN_EVALUATION_LOG_DIR,
                LOG_REPORT,
            )
        except Exception as e:  # pragma: no cover - dependency missing
            return HarnessResult(
                instance_id=instance_id,
                resolved=False,
                patch_extracted=True,
                patch_apply_ok=None,
                harness_error=f"swebench_import_failed: {e}",
            )

        # Build a one-row predictions file.
        # Schema: list of {instance_id, model_name_or_path, model_patch}.
        # `model_name_or_path` becomes the log subdir; "/" gets replaced with "__".
        safe_model = model_name.replace("/", "__")
        preds_dir = self.workspace / "predictions" / run_id
        preds_dir.mkdir(parents=True, exist_ok=True)
        preds_path = preds_dir / "preds.json"
        preds_path.write_text(
            json.dumps(
                [
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": safe_model,
                        "model_patch": patch,
                    }
                ]
            )
        )

        report_path_rel = (
            RUN_EVALUATION_LOG_DIR / run_id / safe_model / instance_id / LOG_REPORT
        )

        with self._harness_lock, _pushd(self.workspace):
            try:
                swebench_main(
                    dataset_name=self.dataset_name,
                    split=self.split,
                    instance_ids=[instance_id],
                    predictions_path=str(preds_path),
                    max_workers=1,
                    force_rebuild=False,
                    cache_level=self.cache_level,
                    clean=False,
                    open_file_limit=4096,
                    run_id=run_id,
                    timeout=self.timeout_s,
                    namespace=self.namespace,
                    rewrite_reports=False,
                    modal=False,
                )
            except SystemExit as e:
                # swebench occasionally calls sys.exit on bad config; trap it.
                return HarnessResult(
                    instance_id=instance_id,
                    resolved=False,
                    patch_extracted=True,
                    harness_error=f"harness_system_exit: {e}",
                )
            except Exception as e:
                tb = traceback.format_exc(limit=10)
                return HarnessResult(
                    instance_id=instance_id,
                    resolved=False,
                    patch_extracted=True,
                    harness_error=f"harness_exception: {e}\n{tb}",
                )

            report_abs = (self.workspace / report_path_rel).resolve()

        # Parse report.json. swebench writes:
        #   { instance_id: {
        #         "patch_is_None": bool,
        #         "patch_exists": bool,
        #         "patch_successfully_applied": bool,
        #         "resolved": bool,
        #         "tests_status": {
        #             "FAIL_TO_PASS": {"success": [...], "failure": [...]},
        #             "PASS_TO_PASS": {"success": [...], "failure": [...]},
        #             ...
        #         }
        #     }
        #   }
        # When apply fails or other harness error, the report may be missing.
        if not report_abs.exists():
            return HarnessResult(
                instance_id=instance_id,
                resolved=False,
                patch_extracted=True,
                patch_apply_ok=False,
                harness_error="report_not_written",
            )

        try:
            report = json.loads(report_abs.read_text())
        except Exception as e:
            return HarnessResult(
                instance_id=instance_id,
                resolved=False,
                patch_extracted=True,
                harness_error=f"report_parse_failed: {e}",
            )

        inst_report = report.get(instance_id, {})
        tests_status = inst_report.get("tests_status", {}) or {}
        ftp = tests_status.get("FAIL_TO_PASS", {}) or {}
        ptp = tests_status.get("PASS_TO_PASS", {}) or {}

        return HarnessResult(
            instance_id=instance_id,
            resolved=bool(inst_report.get("resolved", False)),
            patch_extracted=True,
            patch_apply_ok=bool(inst_report.get("patch_successfully_applied", False)),
            ftp_pass=list(ftp.get("success", []) or []),
            ftp_fail=list(ftp.get("failure", []) or []),
            ptp_pass=list(ptp.get("success", []) or []),
            ptp_fail=list(ptp.get("failure", []) or []),
            harness_error=None,
            raw_report=inst_report,
        )


# =============================================================================
# Module-level convenience
# =============================================================================


_default_harness: SWEHarness | None = None


def get_default_harness() -> SWEHarness:
    global _default_harness
    if _default_harness is None:
        _default_harness = SWEHarness()
    return _default_harness


def verify_patch(
    instance_id: str,
    patch: str,
    model_name: str = "evolvingintent",
) -> HarnessResult:
    """Module-level convenience that uses the default harness instance."""
    return get_default_harness().verify_patch(
        instance_id=instance_id, patch=patch, model_name=model_name
    )


# =============================================================================
# Helpers
# =============================================================================


@contextlib.contextmanager
def _pushd(target: Path):
    """Temporarily chdir into `target`."""
    prev = Path.cwd()
    target.mkdir(parents=True, exist_ok=True)
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(prev)
