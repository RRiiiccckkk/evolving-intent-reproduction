"""Run the remaining BrowseComp+ construction and evaluation on one 5090.

The pipeline has one non-blocking owner lock and no automatic relaunch loop.
Re-running after a failure is safe because construction and evaluation both
resume from task-bound atomic checkpoints.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO
from urllib import request as urllib_request

from reproduction.browsecomp_modal import RETRIEVER_REVISION
from reproduction.browsecomp_construction_5090 import (
    STAGE3_SHARDS,
    STAGE3_SHARD_WORKERS,
    STAGE3_WORKERS,
    finalize_sharded_run,
)
from reproduction.browsecomp_plan_a import WorkflowError
from reproduction.run_with_cc_switch import assert_model_available


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCKED_MODEL = "kimi-k2.6"
REASONING_EFFORT = "medium"
FORBIDDEN_ENVIRONMENT = {
    "LLM_COST_HARD_CAP_USD",
    "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
    "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
}
RETRIEVER_SECRET_ENVIRONMENT = {
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_runtime_environment(environment: dict[str, str]) -> None:
    if environment.get("LLM_LOCKED_MODEL") != LOCKED_MODEL:
        raise WorkflowError(f"pipeline requires LLM_LOCKED_MODEL={LOCKED_MODEL}")
    if environment.get("LLM_REASONING_EFFORT") != REASONING_EFFORT:
        raise WorkflowError("pipeline requires LLM_REASONING_EFFORT=medium")
    if environment.get("LLM_DISABLE_OUTPUT_LIMITS", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise WorkflowError("pipeline requires output limits to be disabled")
    if not environment.get("LLM_API_KEY", "").strip():
        raise WorkflowError("pipeline requires an in-memory LLM_API_KEY")
    present = sorted(name for name in FORBIDDEN_ENVIRONMENT if environment.get(name))
    if present:
        raise WorkflowError(f"pipeline forbids environment limits: {present}")


def read_ready(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib_request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise WorkflowError("retriever ready endpoint returned a non-object")
    if payload.get("retriever_revision") != RETRIEVER_REVISION:
        raise WorkflowError("retriever revision does not match the fixed Plan A revision")
    if payload.get("runtime") != "rtx-5090":
        raise WorkflowError("retriever is not the 5090 runtime")
    return payload


def wait_for_retriever(url: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise WorkflowError(
                f"5090 retriever exited during startup with code {process.returncode}"
            )
        try:
            read_ready(url)
            return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(5)
    raise WorkflowError(f"5090 retriever did not become ready: {last_error}")


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def terminate_on_parent_death(expected_parent_pid: int) -> None:
    """Ask Linux to terminate a shard if its coordinator disappears."""
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def acquire_pipeline_owner_lock(root: Path) -> TextIO:
    lock_handle = (root / "pipeline.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise WorkflowError("another 5090 BrowseComp+ pipeline owns this root") from exc
    return lock_handle


def construction_command(root: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "reproduction.browsecomp_construction_5090",
        "--root",
        str(root),
        "--workers",
        str(STAGE3_WORKERS),
    ]


def construction_shard_command(root: Path, shard_index: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "reproduction.browsecomp_construction_5090",
        "--root",
        str(root),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(STAGE3_SHARDS),
        "--workers",
        str(STAGE3_SHARD_WORKERS),
    ]


def run_construction_shards(
    root: Path,
    state_path: Path,
    started_at: float,
) -> dict[str, Any]:
    """Run ten disjoint Stage 3 subprocesses and aggregate after all exit."""
    processes: list[subprocess.Popen[Any]] = []
    expected_parent_pid = os.getpid()
    try:
        for shard_index in range(STAGE3_SHARDS):
            processes.append(
                subprocess.Popen(
                    construction_shard_command(root, shard_index),
                    cwd=REPOSITORY_ROOT,
                    start_new_session=True,
                    preexec_fn=lambda: terminate_on_parent_death(
                        expected_parent_pid
                    ),
                )
            )
        atomic_write_json(
            state_path,
            {
                "status": "construction",
                "started_at": started_at,
                "pid": os.getpid(),
                "runtime": "rtx-5090",
                "construction_shards": STAGE3_SHARDS,
                "workers_per_shard": STAGE3_SHARD_WORKERS,
                "total_construction_workers": (
                    STAGE3_SHARDS * STAGE3_SHARD_WORKERS
                ),
                "shard_pids": [process.pid for process in processes],
            },
        )
        returncodes: dict[int, int] = {}
        remaining = set(range(len(processes)))
        while remaining:
            for shard_index in tuple(remaining):
                returncode = processes[shard_index].poll()
                if returncode is None:
                    continue
                remaining.remove(shard_index)
                returncodes[shard_index] = returncode
            if remaining:
                time.sleep(1)
        failures = {
            str(index): returncode
            for index, returncode in returncodes.items()
            if returncode != 0
        }
        if failures:
            raise WorkflowError(f"BrowseComp+ Stage 3 shards failed: {failures}")
        return finalize_sharded_run(root)
    finally:
        for process in processes:
            stop_process(process)


def run_pipeline(root: Path, cache_root: Path, port: int) -> dict[str, Any]:
    print(
        "[5090-pipeline] starting locked kimi-k2.6 / medium run",
        flush=True,
    )
    validate_runtime_environment(dict(os.environ))
    assert_model_available(os.environ["LLM_API_KEY"])
    root = root.resolve()
    cache_root = cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    shared_hf_home = root / "cache" / "huggingface"
    shared_hf_home.mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_pipeline_owner_lock(root)
    try:
        state_path = root / "pipeline_state.json"
        started_at = time.time()
        atomic_write_json(
            state_path,
            {
                "status": "construction",
                "started_at": started_at,
                "pid": os.getpid(),
                "runtime": "rtx-5090",
                "construction_shards": STAGE3_SHARDS,
                "workers_per_shard": STAGE3_SHARD_WORKERS,
                "total_construction_workers": (
                    STAGE3_SHARDS * STAGE3_SHARD_WORKERS
                ),
            },
        )
        construction_report = run_construction_shards(
            root,
            state_path,
            started_at,
        )

        retriever_log_path = root / "retriever_5090.log"
        retriever_log = retriever_log_path.open("ab", buffering=0)
        retriever: subprocess.Popen[Any] | None = None
        try:
            retriever_environment = os.environ.copy()
            for name in RETRIEVER_SECRET_ENVIRONMENT:
                retriever_environment.pop(name, None)
            retriever_environment["HF_HOME"] = str(shared_hf_home)
            retriever = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "reproduction.browsecomp_retriever_5090",
                    "--cache-root",
                    str(cache_root),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--device",
                    "cuda:0",
                ],
                cwd=REPOSITORY_ROOT,
                env=retriever_environment,
                stdout=retriever_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ready_url = f"http://127.0.0.1:{port}/ready"
            search_url = f"http://127.0.0.1:{port}/search"
            wait_for_retriever(ready_url, retriever, timeout=3600)
            atomic_write_json(
                state_path,
                {
                    "status": "evaluation",
                    "started_at": started_at,
                    "pid": os.getpid(),
                    "retriever_pid": retriever.pid,
                    "runtime": "rtx-5090",
                    "retriever_revision": RETRIEVER_REVISION,
                    "construction_shards": STAGE3_SHARDS,
                    "workers_per_shard": STAGE3_SHARD_WORKERS,
                    "total_construction_workers": (
                        STAGE3_SHARDS * STAGE3_SHARD_WORKERS
                    ),
                },
            )

            evaluation_environment = os.environ.copy()
            for name in FORBIDDEN_ENVIRONMENT:
                evaluation_environment.pop(name, None)
            evaluation_environment.update(
                {
                    "DATA_PATH": str(
                        root / "final_dataset" / "browsecomp_plus_final.json"
                    ),
                    "RETRIEVER_URL": search_url,
                    "LLM_USAGE_LEDGER_PATH": str(root / "evaluation_usage.jsonl"),
                    "NUM_SAMPLES": "100",
                    "NUM_WORKERS": "8",
                    "REASONING_EFFORT": REASONING_EFFORT,
                }
            )
            evaluation = subprocess.run(
                [
                    "bash",
                    "evaluation/scripts/run_browsecomp.sh",
                    LOCKED_MODEL,
                ],
                cwd=REPOSITORY_ROOT,
                env=evaluation_environment,
                check=False,
            )
            if evaluation.returncode != 0:
                raise WorkflowError(
                    f"BrowseComp+ evaluation failed with code {evaluation.returncode}"
                )
        finally:
            stop_process(retriever)
            retriever_log.close()

        report = {
            "status": "complete",
            "started_at": started_at,
            "completed_at": time.time(),
            "runtime": "rtx-5090",
            "model": LOCKED_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "retriever_revision": RETRIEVER_REVISION,
            "construction": construction_report,
            "construction_shards": STAGE3_SHARDS,
            "workers_per_shard": STAGE3_SHARD_WORKERS,
            "total_construction_workers": (
                STAGE3_SHARDS * STAGE3_SHARD_WORKERS
            ),
        }
        atomic_write_json(state_path, report)
        return report
    except Exception as exc:
        atomic_write_json(
            root / "pipeline_state.json",
            {
                "status": "failed",
                "failed_at": time.time(),
                "runtime": "rtx-5090",
                "construction_shards": STAGE3_SHARDS,
                "workers_per_shard": STAGE3_SHARD_WORKERS,
                "total_construction_workers": (
                    STAGE3_SHARDS * STAGE3_SHARD_WORKERS
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_pipeline(args.root, args.cache_root, args.port)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
