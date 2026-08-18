from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reproduction import fetch_browsecomp_5090 as fetch_module
from reproduction.fetch_browsecomp_5090 import (
    FetchError,
    construction_rsync_command,
    evaluation_rsync_command,
    validate_remote_path,
)


def test_construction_fetch_is_selective_and_excludes_cache(tmp_path: Path) -> None:
    command = construction_rsync_command(
        "fanruibo-5090-1",
        "/nvme/home/fanruibo/evolving-intent-private/browsecomp-n100",
        tmp_path,
    )

    rendered = " ".join(command)
    assert "/run/***" in rendered
    assert "/usage/***" in rendered
    assert "/final_dataset/***" in rendered
    assert "--exclude *" in rendered
    assert "/cache/" not in rendered
    assert "/private/" not in command
    assert "--delete" not in command


def test_evaluation_fetch_accepts_only_paper_scenarios(tmp_path: Path) -> None:
    command = evaluation_rsync_command(
        "fanruibo-5090-1",
        "/nvme/home/fanruibo/evolving-intent-reproduction",
        "fully_specified",
        tmp_path,
    )
    assert "browsecomp_plus_n100" in command[-2]
    assert "fully_specified" in command[-2]

    with pytest.raises(FetchError, match="unsupported"):
        evaluation_rsync_command(
            "fanruibo-5090-1",
            "/nvme/home/fanruibo/evolving-intent-reproduction",
            "other",
            tmp_path,
        )


def test_fetch_rejects_remote_path_injection() -> None:
    assert validate_remote_path(
        "/nvme/home/fanruibo/evolving-intent-private/browsecomp-n100",
        "remote root",
    ).endswith("browsecomp-n100")
    with pytest.raises(FetchError, match="unsupported characters"):
        validate_remote_path(
            "/nvme/home/fanruibo/private;touch-bad",
            "remote root",
        )
    with pytest.raises(FetchError, match="must be under"):
        validate_remote_path("/tmp/private", "remote root")


def test_fetch_refuses_incomplete_state_before_any_rsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        arguments: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(arguments))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout='{"status":"construction","runtime":"rtx-5090"}',
            stderr="",
        )

    monkeypatch.setattr(fetch_module, "_run", fake_run)

    with pytest.raises(FetchError, match="refusing to fetch an incomplete"):
        fetch_module.fetch(
            host="fanruibo-5090-1",
            remote_root=(
                "/nvme/home/fanruibo/evolving-intent-private/browsecomp-n100"
            ),
            remote_repo="/nvme/home/fanruibo/evolving-intent-reproduction",
            repo_root=tmp_path,
        )

    assert len(calls) == 1
    assert calls[0][0] == "ssh"
    assert all(call[0] != "rsync" for call in calls)
