"""Modal-only BrowseComp+ Plan A construction and local artifact audit.

The remote function prepares a private 100-row input with ``gold_docs`` and
runs all three construction stages on a high-memory CPU. The local entrypoint
downloads only derived stage outputs, the final dataset, summary, and usage
ledger; decrypted source documents remain on the private Modal Volume.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from reproduction.browsecomp_plan_a import (
    DEFAULT_MANIFEST,
    HF_DATASET_REVISION,
    PLAN_A_MODEL,
    REMOTE_PRIVATE_SOURCE_COLUMNS,
    REPOSITORY_ROOT,
    WorkflowError,
    atomic_write_json,
    audit_completed_run,
    construct,
    prepare_selected_data,
    validate_prepared_data,
)

try:
    import modal
except ImportError:  # Local unit tests can inspect helpers without Modal.
    modal = None


APP_NAME = os.environ.get(
    "BROWSECOMP_CONSTRUCTION_APP_NAME",
    "evolving-intent-browsecomp-construction",
)
SECRET_NAME = "evolving-intent-kimi-k2-6"
VOLUME_NAME = "browsecomp-plan-a-construction"
CANARY_VOLUME_NAME = "browsecomp-plan-a-construction-canary-v6"
VOLUME_MOUNT = Path("/artifacts")
CANARY_VOLUME_MOUNT = Path("/canary-artifacts")
VOLUME_RUN_ROOT = "browsecomp-plan-a-n100"
CANARY_RUN_ROOT = "browsecomp-plan-a-canary-v6"
REMOTE_ROOT = VOLUME_MOUNT / VOLUME_RUN_ROOT
REMOTE_PRIVATE_RAW = REMOTE_ROOT / "private" / "raw_data_with_gold_docs.jsonl"
REMOTE_MANIFEST = REMOTE_ROOT / "manifest.json"
REMOTE_RUN_DIR = REMOTE_ROOT / "run"
REMOTE_FINAL = REMOTE_ROOT / "final_dataset" / "browsecomp_plus_final.json"
REMOTE_LEDGER = REMOTE_ROOT / "usage" / "llm_usage.jsonl"

DEFAULT_LOCAL_RUN_DIR = REPOSITORY_ROOT / "reproduction" / "runs" / VOLUME_RUN_ROOT
DEFAULT_LOCAL_FINAL = REPOSITORY_ROOT / "final_dataset" / "browsecomp_plus_final.json"
DEFAULT_CANARY_TASK_ID = "extracted-browsecomp_plus-test-1019"
PLAN_A_PRICE_MAP = {
    PLAN_A_MODEL: {
        "input": 0.95,
        "output": 4.0,
        "cached_input": 0.16,
        "reasoning": 4.0,
    }
}
STAGE3_RECOVERABLE_SAMPLE_ATTEMPTS = 8
STAGE3_WORKERS = 16
STAGE3_READ_TIMEOUT_SECONDS = 1800
DEFAULT_LOCAL_CANARY_DIR = (
    REPOSITORY_ROOT / "reproduction" / "runs" / CANARY_RUN_ROOT
)


def _download_keys(run_root: str) -> dict[str, str]:
    return {
        f"{run_root}/run/stage1_extracted.json": "stage1_extracted.json",
        f"{run_root}/run/stage2_counterfactual.json": "stage2_counterfactual.json",
        f"{run_root}/run/stage3_predecessor.json": "stage3_predecessor.json",
        f"{run_root}/run/construction_summary.json": "construction_summary.json",
        f"{run_root}/run/remote_audit.json": "remote_audit.json",
        f"{run_root}/private/raw_data_with_gold_docs.manifest.json": (
            "preparation_manifest.json"
        ),
        f"{run_root}/usage/llm_usage.jsonl": "llm_usage.jsonl",
    }


DOWNLOADS = _download_keys(VOLUME_RUN_ROOT)
REMOTE_FINAL_KEY = f"{VOLUME_RUN_ROOT}/final_dataset/browsecomp_plus_final.json"


def _validate_secret_environment(*, root: Path, ledger: Path) -> None:
    configured_backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if configured_backend and configured_backend not in {
        "compatible",
        "generic",
        "openai-compatible",
        "openai_compatible",
    }:
        raise WorkflowError("Modal secret must set LLM_BACKEND=compatible")
    for name in ("LLM_API_KEY", "LLM_BASE_URL"):
        if not os.environ.get(name, "").strip():
            raise WorkflowError(f"Modal secret must set {name}")
    if os.environ.get("LLM_COST_HARD_CAP_USD", "").strip():
        raise WorkflowError("Plan A records usage only; remove LLM_COST_HARD_CAP_USD")
    configured_effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
    if configured_effort and configured_effort != "medium":
        raise WorkflowError("Plan A requires LLM_REASONING_EFFORT=medium")

    os.environ["LLM_BACKEND"] = "compatible"
    os.environ["LLM_LOCKED_MODEL"] = PLAN_A_MODEL
    os.environ["LLM_REASONING_EFFORT"] = "medium"
    os.environ["LLM_DISABLE_OUTPUT_LIMITS"] = "1"
    os.environ["LLM_REQUIRE_USAGE_ACCOUNTING"] = "1"
    os.environ["LLM_USAGE_LEDGER_PATH"] = str(ledger)
    os.environ["LLM_PRICE_MAP"] = json.dumps(PLAN_A_PRICE_MAP)
    os.environ["HF_HOME"] = str(root / "cache" / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(
        root / "cache" / "huggingface" / "datasets"
    )
    if os.environ.get("LLM_REASONING_EFFORT") != "medium":
        raise WorkflowError("failed to lock LLM_REASONING_EFFORT=medium")

    from intent_construction.intent_extraction.core import llm_utils

    # Normal Kimi calls finish within minutes; recycle stalled streams without
    # imposing an output-token limit.
    llm_utils._CLIENT_TIMEOUT = llm_utils.httpx.Timeout(
        connect=30.0,
        read=600.0,
        write=600.0,
        pool=600.0,
    )

    resolved = llm_utils.resolve_model_name(PLAN_A_MODEL)
    if resolved != PLAN_A_MODEL:
        raise WorkflowError(
            f"LLM_MODEL_MAP resolves {PLAN_A_MODEL!r} to {resolved!r}; "
            "Plan A requires requested and resolved model IDs to match"
        )
    if llm_utils._model_prices(PLAN_A_MODEL, resolved) is None:
        raise WorkflowError(f"LLM_PRICE_MAP has no entry for {PLAN_A_MODEL}")


def _retry_incomplete_stage3_processor(
    processor: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    """Retry one Stage 3 sample after a recoverable incomplete attempt.

    A returned ``None`` or same-sample candidate that fails the Stage 3
    acceptance gate means the full sample must be regenerated. Exceptions are
    deliberately not treated as candidate rejection: only a provider-truncated
    response is recoverable here; accounting, model, policy, and other failures
    still escape immediately.
    """
    from intent_construction.intent_extraction.core.llm_utils import (
        LLMIncompleteResponse,
    )
    from reproduction import browsecomp_plan_a as plan_a

    def needs_full_regeneration(
        item: dict[str, Any], result: dict[str, Any] | None
    ) -> bool:
        if plan_a._stage_result_is_complete("stage3", result):
            return False
        if isinstance(result, dict):
            result_task_id = result.get("task_id")
            if result_task_id is not None and result_task_id != item.get("task_id"):
                return False
            if result.get("success") is False or result.get("error"):
                return False
        return True

    def resilient_processor(item: dict[str, Any]) -> dict[str, Any] | None:
        for attempt in range(STAGE3_RECOVERABLE_SAMPLE_ATTEMPTS):
            try:
                result = processor(item)
            except LLMIncompleteResponse:
                if attempt == STAGE3_RECOVERABLE_SAMPLE_ATTEMPTS - 1:
                    raise
                reason = "Incomplete Kimi response"
            else:
                if not needs_full_regeneration(item, result):
                    return result
                if attempt == STAGE3_RECOVERABLE_SAMPLE_ATTEMPTS - 1:
                    return result
                reason = "Incomplete or rejected Stage 3 result"

            delay = min(60, 5 * (attempt + 1))
            print(
                f"[stage3-recovery] {reason} for "
                f"{item.get('task_id', 'unknown')}; regenerating full sample "
                f"({attempt + 2}/{STAGE3_RECOVERABLE_SAMPLE_ATTEMPTS}) "
                f"after {delay}s",
                flush=True,
            )
            time.sleep(delay)
        raise RuntimeError("unreachable Stage 3 retry state")

    return resilient_processor


def _install_recoverable_stage3_retries() -> None:
    """Keep a transient incomplete response from cancelling other Stage 3 work.

    Stage 3 also runs with two throughput overrides, applied only to the
    Stage 3 stage call and outside the checkpoint source bundles:

    - ``STAGE3_WORKERS``: predecessor chains are latency-bound (each
      sample needs a long sequence of dependent Kimi calls), so the default
      cap of 4 workers starves the stage; raise it instead of changing
      construction semantics.
    - ``STAGE3_READ_TIMEOUT_SECONDS``: observed 2026-08-17, >50% of Stage 3
      calls exceed the 600s default read timeout while small calls on the
      same account return in ~2s, so these are genuinely long generations,
      not provider congestion. Recycle them at 1800s instead.
    """
    from intent_construction.intent_extraction.core import llm_utils
    from reproduction import browsecomp_plan_a as plan_a

    if getattr(plan_a, "_modal_stage3_recovery_installed", False):
        return
    original_run_stage = plan_a._run_stage

    def resilient_run_stage(
        name: str,
        inputs: list[dict[str, Any]],
        output_path: Path,
        processor: Callable[[dict[str, Any]], dict[str, Any] | None],
        workers: int,
        checkpoint_hook: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        if name == "stage3":
            processor = _retry_incomplete_stage3_processor(processor)
            workers = max(workers, STAGE3_WORKERS)
            llm_utils._CLIENT_TIMEOUT = llm_utils.httpx.Timeout(
                connect=30.0,
                read=float(STAGE3_READ_TIMEOUT_SECONDS),
                write=600.0,
                pool=600.0,
            )
            print(
                f"[stage3-recovery] stage3 overrides: workers={workers} "
                f"read_timeout={STAGE3_READ_TIMEOUT_SECONDS}s",
                flush=True,
            )
        return original_run_stage(
            name,
            inputs,
            output_path,
            processor,
            workers,
            checkpoint_hook,
        )

    plan_a._run_stage = resilient_run_stage
    plan_a._modal_stage3_recovery_installed = True


def _ensure_private_input(
    manifest_json: str,
    *,
    manifest_path: Path,
    raw_path: Path,
    expected_sample_count: int,
) -> dict[str, Any]:
    try:
        manifest_payload = json.loads(manifest_json)
    except json.JSONDecodeError as exc:
        raise WorkflowError("manifest payload is not valid JSON") from exc
    atomic_write_json(manifest_path, manifest_payload)

    preparation_manifest = raw_path.with_suffix(".manifest.json")
    if raw_path.exists() and preparation_manifest.exists():
        with preparation_manifest.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if (
            report.get("source_revision") == HF_DATASET_REVISION
            and report.get("source_columns") == list(REMOTE_PRIVATE_SOURCE_COLUMNS)
        ):
            validate_prepared_data(
                raw_path,
                manifest_path,
                require_gold_docs=True,
                expected_sample_count=expected_sample_count,
            )
            return report

    report = prepare_selected_data(
        manifest_path,
        raw_path,
        source_columns=REMOTE_PRIVATE_SOURCE_COLUMNS,
        max_request_bytes=24 * 1024 * 1024,
        max_total_bytes=40 * 1024 * 1024,
        expected_sample_count=expected_sample_count,
    )
    validate_prepared_data(
        raw_path,
        manifest_path,
        require_gold_docs=True,
        expected_sample_count=expected_sample_count,
    )
    return report


def _single_sample_manifest(manifest_json: str, task_id: str) -> str:
    try:
        payload = json.loads(manifest_json)
    except json.JSONDecodeError as exc:
        raise WorkflowError("manifest payload is not valid JSON") from exc
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or len(samples) != 100:
        raise WorkflowError("canary source must be the fixed 100-sample manifest")
    matches = [item for item in samples if item.get("task_id") == task_id]
    if len(matches) != 1:
        raise WorkflowError(f"canary task ID is not in the fixed manifest: {task_id}")
    canary = {
        "dataset": payload.get("dataset", "browsecomp_plus"),
        "split": payload.get("split", "test"),
        "num_samples": 1,
        "samples": matches,
    }
    return json.dumps(canary, ensure_ascii=False)


def _write_chunks_atomically(path: Path, chunks: Iterable[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise WorkflowError("Modal Volume returned a non-bytes chunk")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_and_audit(
    read_remote: Callable[[str], Iterable[bytes]],
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    local_run_dir: Path = DEFAULT_LOCAL_RUN_DIR,
    local_final_path: Path = DEFAULT_LOCAL_FINAL,
    downloads: dict[str, str] = DOWNLOADS,
    remote_final_key: str = REMOTE_FINAL_KEY,
    expected_sample_count: int = 100,
) -> dict[str, Any]:
    """Download the approved export set and audit it before publication use."""
    local_run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=local_run_dir.parent, prefix=".browsecomp-download-"
    ) as temporary_dir:
        staging = Path(temporary_dir)
        staged_run = staging / "run"
        for remote_key, local_name in downloads.items():
            _write_chunks_atomically(staged_run / local_name, read_remote(remote_key))
        staged_final = staging / "final_dataset" / local_final_path.name
        _write_chunks_atomically(staged_final, read_remote(remote_final_key))

        audit = audit_completed_run(
            manifest_path,
            staged_run,
            staged_final,
            staged_run / "llm_usage.jsonl",
            expected_sample_count=expected_sample_count,
        )
        for local_name in downloads.values():
            destination = local_run_dir / local_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_run / local_name, destination)
        local_final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_final, local_final_path)

    audit["final_dataset"] = str(local_final_path.resolve())
    atomic_write_json(local_run_dir / "local_audit.json", audit)
    return audit


def _safe_task_component(task_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in task_id)
    if not safe:
        raise WorkflowError("canary task ID is empty")
    return safe[:120]


def _execute_remote(
    manifest_json: str,
    *,
    workers: int,
    root: Path,
    expected_sample_count: int,
    commit: Callable[[], None],
) -> dict[str, Any]:
    if workers < 1 or workers > 16:
        raise WorkflowError("workers must be between 1 and 16")
    manifest_path = root / "manifest.json"
    private_raw = root / "private" / "raw_data_with_gold_docs.jsonl"
    run_dir = root / "run"
    final_path = root / "final_dataset" / "browsecomp_plus_final.json"
    ledger_path = root / "usage" / "llm_usage.jsonl"
    _validate_secret_environment(root=root, ledger=ledger_path)
    preparation = _ensure_private_input(
        manifest_json,
        manifest_path=manifest_path,
        raw_path=private_raw,
        expected_sample_count=expected_sample_count,
    )
    commit()
    _install_recoverable_stage3_retries()
    summary = construct(
        private_raw,
        manifest_path,
        run_dir,
        workers,
        final_path=final_path,
        checkpoint_hook=commit,
        expected_sample_count=expected_sample_count,
    )
    audit = audit_completed_run(
        manifest_path,
        run_dir,
        final_path,
        ledger_path,
        expected_sample_count=expected_sample_count,
    )
    safe_report = {
        "model": PLAN_A_MODEL,
        "sample_count": audit["sample_count"],
        "source_revision": preparation["source_revision"],
        "source_columns": preparation["source_columns"],
        "stage_counts": audit["stage_counts"],
        "usage": audit["usage"],
        "independence_verification": True,
        "remote_final": summary["final_dataset"],
    }
    atomic_write_json(run_dir / "remote_audit.json", safe_report)
    commit()
    return safe_report


if modal is not None:
    app = modal.App(APP_NAME)
    artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    canary_volume = modal.Volume.from_name(
        CANARY_VOLUME_NAME, create_if_missing=True
    )
    kimi_secret = modal.Secret.from_name(SECRET_NAME)
    construction_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "datasets>=4,<5",
            "numpy>=2,<3",
            "openai>=2,<3",
            "pyarrow>=18,<26",
            "rank-bm25>=0.2.2,<1",
            "tqdm>=4.67,<5",
        )
        .add_local_python_source("intent_construction", "reproduction")
        .add_local_dir(
            REPOSITORY_ROOT
            / "intent_construction"
            / "intent_extraction"
            / "dataset_impl"
            / "browsecomp_plus"
            / "prompts",
            "/root/intent_construction/intent_extraction/dataset_impl/"
            "browsecomp_plus/prompts",
        )
        .add_local_dir(
            REPOSITORY_ROOT
            / "intent_construction"
            / "retrospective_expansion"
            / "counterfactual"
            / "prompts",
            "/root/intent_construction/retrospective_expansion/"
            "counterfactual/prompts",
        )
        .add_local_dir(
            REPOSITORY_ROOT
            / "intent_construction"
            / "retrospective_expansion"
            / "predecessor"
            / "prompts",
            "/root/intent_construction/retrospective_expansion/predecessor/prompts",
        )
    )

    @app.function(
        image=construction_image,
        cpu=8.0,
        memory=65536,
        volumes={str(VOLUME_MOUNT): artifact_volume},
        secrets=[kimi_secret],
        timeout=24 * 60 * 60,
    )
    def construct_remote(manifest_json: str, workers: int = 8) -> dict[str, Any]:
        return _execute_remote(
            manifest_json,
            workers=workers,
            root=REMOTE_ROOT,
            expected_sample_count=100,
            commit=artifact_volume.commit,
        )

    @app.function(
        image=construction_image,
        cpu=2.0,
        memory=8192,
        volumes={str(VOLUME_MOUNT): artifact_volume},
        timeout=1800,
    )
    def migrate_stage1_policy() -> dict[str, Any]:
        """Re-fingerprint stage-1 checkpoints after a deliberate code change.

        The 2026-08-16 hardening (bounded verification timeouts + explicit
        accept-on-infra-timeout fallback in base_extractor/extractor) changed
        the stage-1 source bundle, so existing checkpoints fail the policy
        equality check even though their inputs and results are unchanged.
        This migration verifies ``input_sha256`` still matches and that the
        policy differs ONLY in ``source_bundle_sha256``, then rewrites the
        policy field. Anything else is refused.
        """
        from reproduction.browsecomp_plan_a import (
            WorkflowError,
            _checkpoint_envelope,
            _load_selected_samples,
            atomic_write_json as plan_a_atomic_write,
        )

        manifest_path = REMOTE_ROOT / "manifest.json"
        private_raw = REMOTE_ROOT / "private" / "raw_data_with_gold_docs.jsonl"
        checkpoint_dir = REMOTE_ROOT / "run" / "stage1_extracted.checkpoints"
        inputs = _load_selected_samples(
            private_raw, manifest_path, expected_sample_count=100
        )
        inputs_by_id = {str(item["task_id"]): item for item in inputs}
        migrated: list[str] = []
        refusals: list[str] = []
        for path in sorted(checkpoint_dir.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            task_id = envelope.get("task_id")
            if task_id not in inputs_by_id:
                refusals.append(f"{path.name}: unknown task id")
                continue
            expected = _checkpoint_envelope(
                "stage1", inputs_by_id[task_id], envelope.get("result", {})
            )
            if envelope.get("input_sha256") != expected["input_sha256"]:
                refusals.append(f"{path.name}: input_sha256 changed")
                continue
            old_policy = envelope.get("policy")
            new_policy = expected["policy"]
            if not isinstance(old_policy, dict):
                refusals.append(f"{path.name}: policy missing")
                continue
            differing = {
                key
                for key in set(old_policy) | set(new_policy)
                if old_policy.get(key) != new_policy.get(key)
            }
            if differing != {"source_bundle_sha256"}:
                refusals.append(
                    f"{path.name}: policy differs beyond source bundle: {sorted(differing)}"
                )
                continue
            envelope["policy"] = new_policy
            plan_a_atomic_write(path, envelope)
            migrated.append(str(task_id))
        artifact_volume.commit()
        if refusals:
            raise WorkflowError(
                f"refused to migrate {len(refusals)} checkpoints: {refusals[:5]}"
            )
        return {"migrated": len(migrated), "task_ids": migrated}

    @app.local_entrypoint()
    def migrate() -> None:
        report = migrate_stage1_policy.remote()
        print(json.dumps({"migrated": report["migrated"]}, indent=2))

    @app.function(
        image=construction_image,
        cpu=8.0,
        memory=65536,
        volumes={str(CANARY_VOLUME_MOUNT): canary_volume},
        secrets=[kimi_secret],
        timeout=24 * 60 * 60,
    )
    def construct_canary_remote(
        manifest_json: str,
        task_id: str = DEFAULT_CANARY_TASK_ID,
        workers: int = 1,
    ) -> dict[str, Any]:
        canary_manifest = _single_sample_manifest(manifest_json, task_id)
        root = CANARY_VOLUME_MOUNT / CANARY_RUN_ROOT / _safe_task_component(task_id)
        return _execute_remote(
            canary_manifest,
            workers=workers,
            root=root,
            expected_sample_count=1,
            commit=canary_volume.commit,
        )

    @app.local_entrypoint()
    def main(workers: int = 8) -> None:
        manifest_json = DEFAULT_MANIFEST.read_text(encoding="utf-8")
        remote_report = construct_remote.remote(manifest_json, workers)
        artifact_volume.reload()
        audit = download_and_audit(artifact_volume.read_file)
        print(json.dumps({"remote": remote_report, "local": audit}, indent=2))

    @app.local_entrypoint()
    def canary(task_id: str = DEFAULT_CANARY_TASK_ID, workers: int = 1) -> None:
        manifest_json = DEFAULT_MANIFEST.read_text(encoding="utf-8")
        canary_manifest = _single_sample_manifest(manifest_json, task_id)
        remote_report = construct_canary_remote.remote(
            manifest_json, task_id, workers
        )
        canary_volume.reload()
        safe_task = _safe_task_component(task_id)
        run_root = f"{CANARY_RUN_ROOT}/{safe_task}"
        local_run = DEFAULT_LOCAL_CANARY_DIR / safe_task
        local_manifest = local_run / "manifest.json"
        atomic_write_json(local_manifest, json.loads(canary_manifest))
        audit = download_and_audit(
            canary_volume.read_file,
            manifest_path=local_manifest,
            local_run_dir=local_run,
            local_final_path=local_run / "canary_final.json",
            downloads=_download_keys(run_root),
            remote_final_key=f"{run_root}/final_dataset/browsecomp_plus_final.json",
            expected_sample_count=1,
        )
        print(json.dumps({"remote": remote_report, "local": audit}, indent=2))

else:
    app = None


def cli() -> None:
    if modal is None:
        raise SystemExit(
            "Modal is not installed. Install reproduction/requirements-browsecomp.txt, "
            "then run `modal run -m reproduction.browsecomp_construction_modal`."
        )
    print(
        "Run full: modal run -m reproduction.browsecomp_construction_modal::main\n"
        "Run 1-ID canary: modal run -m "
        "reproduction.browsecomp_construction_modal::canary"
    )


if __name__ == "__main__":
    cli()
