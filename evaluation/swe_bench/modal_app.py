"""Modal launcher for the hardened SWE reproduction.

The CLI accepts a Modal Secret name, never a key value. The named secret must
provide the compatible-provider variables documented in ``.env.local.example``.
When nested Modal authentication is not inherited automatically, include
``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` in the same secret so both the
mini-swe-agent ``swerex_modal`` environment and SWE-bench's official Modal
harness can create their sandboxes.
"""

from __future__ import annotations

import inspect
import re
import threading
from pathlib import Path
try:
    import modal
except ModuleNotFoundError:  # Offline guard tests do not install Modal.
    modal = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_REPO = Path("/workspace")
MINI_SWE_AGENT_COMMIT = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
DEFAULT_OFFICIAL_DATASET = "tmp/source-data/swe_bench_verified_princeton.parquet"
DEFAULT_CANARY_DATA = (
    "reproduction/runs/canary/swe-build-seaborn/canary_final.json"
)
DEFAULT_CANARY_TASK_ID = (
    "extracted-swe_bench_verified-test-mwaskom__seaborn-3069"
)
DEFAULT_CANARY_RUN_NAME = "canary-seaborn-3069-agent-verifier-v1"
DEFAULT_CANARY_VOLUME = "evolving-intent-swe-canary-seaborn-3069-v1"
FORMAL_RUN_NAME = "kimi-k2.6"
FORMAL_VOLUME_NAME = "evolving-intent-swe-results"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


app = None
image = None
if modal is not None:
    app = modal.App("evolving-intent-swe-kimi")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "openai>=1.0.0",
            "datasets>=2.14.0",
            "tqdm>=4.65.0",
            "PyYAML>=6.0",
            "swebench==4.1.0",
            "swe-rex==1.4.0",
            "modal>=1.5.0",
            (
                "mini-swe-agent[modal] @ "
                "git+https://github.com/SWE-agent/mini-swe-agent.git@"
                + MINI_SWE_AGENT_COMMIT
            ),
        )
    )
    for source_name in ("evaluation", "intent_construction", "situated_simulation"):
        image = image.add_local_dir(
            REPO_ROOT / source_name,
            str(REMOTE_REPO / source_name),
            copy=True,
            ignore=[
                "**/__pycache__/**",
                "experiments/**",
                "logs/**",
                "**/artifacts/**",
                "**/checkpoints/**",
                "**/output/**",
                "swe_runs/**",
                "swe_workspace/**",
            ],
        )
    if (REPO_ROOT / "final_dataset").is_dir():
        image = image.add_local_dir(
            REPO_ROOT / "final_dataset",
            str(REMOTE_REPO / "final_dataset"),
            copy=True,
        )
    official_source = REPO_ROOT / DEFAULT_OFFICIAL_DATASET
    if official_source.is_file():
        image = image.add_local_file(
            official_source,
            str(REMOTE_REPO / DEFAULT_OFFICIAL_DATASET),
            copy=True,
        )
    canary_source = REPO_ROOT / DEFAULT_CANARY_DATA
    if canary_source.is_file():
        image = image.add_local_file(
            canary_source,
            str(REMOTE_REPO / DEFAULT_CANARY_DATA),
            copy=True,
        )


def build_remote_command(
    *,
    data_path: str,
    run_name: str,
    workers: int,
    official_dataset_path: str = DEFAULT_OFFICIAL_DATASET,
    dry_run: bool = False,
) -> list[str]:
    if not _SAFE_NAME.fullmatch(run_name):
        raise ValueError("run_name may contain only letters, digits, '.', '_' and '-'")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    local_data = Path(data_path)
    if local_data.is_absolute() or ".." in local_data.parts:
        raise ValueError("data_path must be a repository-relative path")
    if not local_data.parts or local_data.parts[0] != "final_dataset":
        raise ValueError("Modal data_path must be under final_dataset/")
    local_official = Path(official_dataset_path)
    if local_official.is_absolute() or ".." in local_official.parts:
        raise ValueError("official_dataset_path must be repository-relative")
    if str(local_official) != DEFAULT_OFFICIAL_DATASET:
        raise ValueError(
            f"Modal official_dataset_path is fixed to {DEFAULT_OFFICIAL_DATASET}"
        )
    command = [
        "python",
        "-m",
        "evaluation.swe_bench.run",
        "--data-path",
        str(REMOTE_REPO / local_data),
        "--official-dataset-path",
        str(REMOTE_REPO / local_official),
        "--run-dir",
        f"/results/{run_name}",
        "--workers",
        str(workers),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def build_canary_remote_command(
    *,
    task_id: str = DEFAULT_CANARY_TASK_ID,
    data_path: str = DEFAULT_CANARY_DATA,
    run_name: str = DEFAULT_CANARY_RUN_NAME,
    official_dataset_path: str = DEFAULT_OFFICIAL_DATASET,
) -> list[str]:
    """Build the fixed one-ID command executed by the outer Modal sandbox."""
    if not _SAFE_TASK_ID.fullmatch(task_id):
        raise ValueError("canary task_id contains unsafe characters")
    if task_id != DEFAULT_CANARY_TASK_ID:
        raise ValueError(f"canary task_id is fixed to {DEFAULT_CANARY_TASK_ID}")
    if not _SAFE_NAME.fullmatch(run_name) or not run_name.startswith("canary-"):
        raise ValueError("canary run_name must be safe and start with 'canary-'")
    if run_name == FORMAL_RUN_NAME:
        raise ValueError("canary run_name must not equal the formal run name")

    local_data = Path(data_path)
    if local_data.is_absolute() or ".." in local_data.parts:
        raise ValueError("canary data_path must be repository-relative")
    if str(local_data) != DEFAULT_CANARY_DATA:
        raise ValueError(f"canary data_path is fixed to {DEFAULT_CANARY_DATA}")

    local_official = Path(official_dataset_path)
    if local_official.is_absolute() or ".." in local_official.parts:
        raise ValueError("official_dataset_path must be repository-relative")
    if str(local_official) != DEFAULT_OFFICIAL_DATASET:
        raise ValueError(
            f"Modal official_dataset_path is fixed to {DEFAULT_OFFICIAL_DATASET}"
        )

    return [
        "python",
        "-u",
        "-m",
        "evaluation.swe_bench.canary",
        "--task-id",
        task_id,
        "--data-path",
        str(REMOTE_REPO / local_data),
        "--official-source-path",
        str(REMOTE_REPO / local_official),
        "--run-dir",
        f"/results/{run_name}",
    ]


def validate_canary_isolation(*, run_name: str, volume_name: str) -> None:
    if not _SAFE_NAME.fullmatch(volume_name):
        raise ValueError("volume_name contains unsafe characters")
    if not volume_name.startswith("evolving-intent-swe-canary-"):
        raise ValueError(
            "canary volume_name must start with 'evolving-intent-swe-canary-'"
        )
    if volume_name == FORMAL_VOLUME_NAME:
        raise ValueError("canary must not use the formal results volume")
    if not _SAFE_NAME.fullmatch(run_name) or not run_name.startswith("canary-"):
        raise ValueError("canary run_name must be safe and start with 'canary-'")
    if run_name == FORMAL_RUN_NAME:
        raise ValueError("canary must not use the formal run directory")


def _stream(stream, *, prefix: str = "") -> None:
    for line in stream:
        print(prefix + line, end="")


def _launch_remote(
    *,
    command: list[str],
    secret_name: str,
    volume_name: str,
) -> None:
    if modal is None or app is None or image is None:
        raise RuntimeError("Modal is not installed; install mini-swe-agent[modal]")
    if not secret_name.strip():
        raise ValueError("secret_name is required")
    if not _SAFE_NAME.fullmatch(volume_name):
        raise ValueError("volume_name contains unsafe characters")
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    sandbox = modal.Sandbox.create(
        *command,
        app=app,
        image=image,
        secrets=[modal.Secret.from_name(secret_name)],
        volumes={"/results": volume},
        timeout=24 * 60 * 60,
        workdir=str(REMOTE_REPO),
        env={"PYTHONPATH": str(REMOTE_REPO)},
    )
    stdout_thread = threading.Thread(target=_stream, args=(sandbox.stdout,))
    stderr_thread = threading.Thread(
        target=_stream,
        args=(sandbox.stderr,),
        kwargs={"prefix": "[stderr] "},
    )
    stdout_thread.start()
    stderr_thread.start()
    sandbox.wait(raise_on_termination=False)
    stdout_thread.join()
    stderr_thread.join()
    if sandbox.returncode != 0:
        raise SystemExit(sandbox.returncode or 1)


def _main_impl(
    secret_name: str,
    data_path: str = "final_dataset/swe_bench_verified_final.json",
    official_dataset_path: str = DEFAULT_OFFICIAL_DATASET,
    run_name: str = "kimi-k2.6",
    workers: int = 8,
    volume_name: str = FORMAL_VOLUME_NAME,
    dry_run: bool = False,
) -> None:
    """Launch a remote orchestrator with credentials injected by Secret name."""
    command = build_remote_command(
        data_path=data_path,
        run_name=run_name,
        workers=workers,
        official_dataset_path=official_dataset_path,
        dry_run=dry_run,
    )
    _launch_remote(
        command=command,
        secret_name=secret_name,
        volume_name=volume_name,
    )
    print(
        f"Results are in Modal Volume {volume_name!r} under {run_name!r}. "
        f"Download with: modal volume get {volume_name} {run_name}"
    )


def _canary_impl(
    secret_name: str,
    task_id: str = DEFAULT_CANARY_TASK_ID,
    data_path: str = DEFAULT_CANARY_DATA,
    official_dataset_path: str = DEFAULT_OFFICIAL_DATASET,
    run_name: str = DEFAULT_CANARY_RUN_NAME,
    volume_name: str = DEFAULT_CANARY_VOLUME,
) -> None:
    """Launch one isolated evolve task through agent and official verifier."""
    validate_canary_isolation(run_name=run_name, volume_name=volume_name)
    command = build_canary_remote_command(
        task_id=task_id,
        data_path=data_path,
        run_name=run_name,
        official_dataset_path=official_dataset_path,
    )
    _launch_remote(
        command=command,
        secret_name=secret_name,
        volume_name=volume_name,
    )
    print(
        f"Canary results are in Modal Volume {volume_name!r} under {run_name!r}."
    )


main = app.local_entrypoint()(_main_impl) if app is not None else _main_impl
canary = app.local_entrypoint()(_canary_impl) if app is not None else _canary_impl
if app is not None:
    main.__signature__ = inspect.signature(_main_impl)
    canary.__signature__ = inspect.signature(_canary_impl)
