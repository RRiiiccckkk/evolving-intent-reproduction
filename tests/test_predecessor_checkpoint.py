import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from intent_construction.retrospective_expansion.predecessor import (
    generate_predecessors,
)


class DeferredFuture:
    def __init__(self, function, *args):
        self.function = function
        self.args = args

    def result(self):
        return self.function(*self.args)


class DeferredExecutor:
    def __init__(self, **_kwargs):
        self.futures = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, function, *args):
        future = DeferredFuture(function, *args)
        self.futures.append(future)
        return future


class FakeGenerator:
    calls = []

    def __init__(self, **_kwargs):
        pass

    def generate_predecessors(self, sample):
        self.calls.append(sample["task_id"])
        return {
            **sample,
            "verification_passed": True,
            "independence_passed": True,
        }


def _run_main(monkeypatch, input_path, output_path, *, resume=False):
    args = [
        "generate_predecessors.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--parallel",
        "3",
        "--checkpoint_interval",
        "999",
    ]
    if resume:
        args.append("--resume")

    FakeGenerator.calls = []
    monkeypatch.setattr(
        generate_predecessors,
        "PredecessorGenerator",
        FakeGenerator,
    )
    monkeypatch.setattr(
        generate_predecessors,
        "ThreadPoolExecutor",
        DeferredExecutor,
    )
    monkeypatch.setattr(sys, "argv", args)
    generate_predecessors.main()


def test_parallel_future_completion_is_checkpointed_immediately(
    monkeypatch, tmp_path
):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "output_checkpoint.json"
    samples = [{"task_id": f"task-{idx}"} for idx in range(3)]
    input_path.write_text(json.dumps(samples))

    checkpoint_snapshots = []
    real_atomic_write = generate_predecessors.atomic_write_json

    def track_atomic_write(path, payload, **kwargs):
        if Path(path) == checkpoint_path:
            checkpoint_snapshots.append(deepcopy(payload))
        real_atomic_write(path, payload, **kwargs)

    def reverse_completion(futures):
        yield from reversed(list(futures))

    monkeypatch.setattr(
        generate_predecessors,
        "atomic_write_json",
        track_atomic_write,
    )
    monkeypatch.setattr(
        generate_predecessors,
        "as_completed",
        reverse_completion,
    )

    _run_main(monkeypatch, input_path, output_path)

    immediate = checkpoint_snapshots[:3]
    assert [snapshot["processed_ids"] for snapshot in immediate] == [
        ["task-2"],
        ["task-1", "task-2"],
        ["task-0", "task-1", "task-2"],
    ]
    assert [snapshot["next_idx"] for snapshot in immediate] == [0, 0, 3]
    assert [row["task_id"] for row in json.loads(output_path.read_text())] == [
        "task-0",
        "task-1",
        "task-2",
    ]


def test_resume_ignores_unsafe_next_idx_and_processes_missing_lower_ids(
    monkeypatch, tmp_path
):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "output_checkpoint.json"
    samples = [{"task_id": f"task-{idx}"} for idx in range(3)]
    input_path.write_text(json.dumps(samples))
    checkpoint_path.write_text(
        json.dumps(
            {
                "results": [{"task_id": "task-2"}],
                "failed": 0,
                "next_idx": 3,
                "processed_ids": ["task-2"],
            }
        )
    )
    monkeypatch.setattr(
        generate_predecessors,
        "as_completed",
        lambda futures: iter(futures),
    )

    _run_main(monkeypatch, input_path, output_path, resume=True)

    assert FakeGenerator.calls == ["task-0", "task-1"]
    assert [row["task_id"] for row in json.loads(output_path.read_text())] == [
        "task-0",
        "task-1",
        "task-2",
    ]
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["processed_ids"] == ["task-0", "task-1", "task-2"]
    assert checkpoint["next_idx"] == 3


def test_keyboard_interrupt_preserves_completed_future_checkpoint(
    monkeypatch, tmp_path
):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    checkpoint_path = tmp_path / "output_checkpoint.json"
    samples = [{"task_id": f"task-{idx}"} for idx in range(3)]
    input_path.write_text(json.dumps(samples))
    checkpoint_path.write_text(
        json.dumps(
            {
                "results": [{"task_id": "task-2"}],
                "failed": 0,
                "next_idx": 0,
                "processed_ids": ["task-2"],
            }
        )
    )

    def interrupt_after_first(futures):
        yield next(iter(futures))
        raise KeyboardInterrupt

    monkeypatch.setattr(
        generate_predecessors,
        "as_completed",
        interrupt_after_first,
    )

    _run_main(monkeypatch, input_path, output_path, resume=True)

    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["processed_ids"] == ["task-0", "task-2"]
    assert [row["task_id"] for row in checkpoint["results"]] == [
        "task-0",
        "task-2",
    ]
    assert checkpoint["next_idx"] == 1
    assert not output_path.exists()


def test_atomic_checkpoint_write_survives_keyboard_interrupt(
    monkeypatch, tmp_path
):
    checkpoint_path = tmp_path / "checkpoint.json"
    original = {"processed_ids": ["stable"]}
    checkpoint_path.write_text(json.dumps(original))

    def interrupted_dump(_payload, file_obj, **_kwargs):
        file_obj.write('{"partial":')
        raise KeyboardInterrupt

    monkeypatch.setattr(generate_predecessors.json, "dump", interrupted_dump)

    with pytest.raises(KeyboardInterrupt):
        generate_predecessors.atomic_write_json(
            checkpoint_path,
            {"processed_ids": ["replacement"]},
        )

    assert json.loads(checkpoint_path.read_text()) == original
    assert list(tmp_path.glob(f".{checkpoint_path.name}.*.tmp")) == []
