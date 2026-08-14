import json
import sys
from unittest.mock import Mock

import pytest

from intent_construction.retrospective_expansion.counterfactual import retry_failed


def test_completed_future_is_merged_before_later_future_finishes(
    monkeypatch, tmp_path
):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps([{"task_id": "task-1"}, {"task_id": "task-2"}])
    )
    output_path.write_text(json.dumps([{"task_id": "already-there"}]))

    generator = Mock()
    generator.generate_counterfactuals.side_effect = lambda sample, **_kwargs: {
        **sample,
        "generated": True,
    }
    monkeypatch.setattr(
        retry_failed, "CounterfactualGenerator", Mock(return_value=generator)
    )

    submitted = []

    class DeferredFuture:
        def __init__(self, function, *args):
            self.function = function
            self.args = args
            self.result_called = False

        def result(self):
            self.result_called = True
            return self.function(*self.args)

    class DeferredExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            future = DeferredFuture(function, *args)
            submitted.append(future)
            return future

    def interrupt_after_first(futures):
        yield next(iter(futures))
        raise KeyboardInterrupt

    monkeypatch.setattr(retry_failed, "ThreadPoolExecutor", DeferredExecutor)
    monkeypatch.setattr(retry_failed, "as_completed", interrupt_after_first)
    monkeypatch.setattr(retry_failed, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retry_failed.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--task_ids",
            "task-1",
            "task-2",
        ],
    )

    with pytest.raises(KeyboardInterrupt):
        retry_failed.main()

    assert json.loads(output_path.read_text()) == [
        {"task_id": "already-there"},
        {"task_id": "task-1", "generated": True},
    ]
    assert [future.result_called for future in submitted] == [True, False]


def test_atomic_write_keeps_original_and_removes_partial_temp_file(tmp_path):
    output_path = tmp_path / "output.json"
    original = [{"task_id": "stable"}]
    output_path.write_text(json.dumps(original))

    with pytest.raises(TypeError):
        retry_failed.atomic_write_json(
            output_path,
            [{"task_id": "replacement"}, {"not_json": object()}],
        )

    assert json.loads(output_path.read_text()) == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []
