from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reproduction import browsecomp_plan_a as plan_a
from reproduction import browsecomp_construction_modal as construction_modal
from reproduction import browsecomp_construction_5090 as construction_5090
from reproduction import browsecomp_pipeline_5090 as pipeline_5090
from reproduction.browsecomp_construction_5090 import (
    STAGE3_SHARDS,
    STAGE3_SHARD_WORKERS,
    STAGE3_WORKERS,
    acquire_construction_mode_lock,
    acquire_stage3_shard_lock,
    build_parser as build_construction_parser,
    checkpoint_inventory,
    finalize_sharded_run,
    run_stage3_shard,
    stage3_shard_inputs,
    validate_migrated_root,
    validate_shard_configuration,
    validate_worker_configuration,
)
from reproduction.browsecomp_pipeline_5090 import (
    RETRIEVER_SECRET_ENVIRONMENT,
    acquire_pipeline_owner_lock,
    construction_command,
    construction_shard_command,
    run_construction_shards,
    validate_runtime_environment,
)
from reproduction.browsecomp_plan_a import WorkflowError
from reproduction.browsecomp_retriever_5090 import (
    ready_payload,
    validate_search_payload,
)
from reproduction.run_5090_with_cc_switch import (
    build_launch_payload,
    build_remote_environment,
    validate_remote_path,
)
from reproduction.run_with_cc_switch import CredentialError


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_migrated_root(root: Path) -> None:
    manifest_samples = [
        {"query_id": f"query-{index}", "task_id": f"task-{index}"}
        for index in range(100)
    ]
    _write_json(root / "manifest.json", {"samples": manifest_samples})
    private = root / "private"
    private.mkdir(parents=True)
    rows = [
        {
            "query_id": item["query_id"],
            "query": f"Question {index}",
            "answer": f"Answer {index}",
            "gold_docs": [{"docid": f"doc-{index}"}],
        }
        for index, item in enumerate(manifest_samples)
    ]
    (private / "raw_data_with_gold_docs.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(
        private / "raw_data_with_gold_docs.manifest.json",
        {
            "source_revision": plan_a.HF_DATASET_REVISION,
            "source_columns": list(plan_a.REMOTE_PRIVATE_SOURCE_COLUMNS),
        },
    )

    raw_samples = [
        {
            "id": item["query_id"],
            "task_id": item["task_id"],
            "query_id": item["query_id"],
            "question": rows[index]["query"],
            "answer": rows[index]["answer"],
            "task": "search",
            "split": "test",
            "evidence_docs": [],
            "gold_docs": rows[index]["gold_docs"],
        }
        for index, item in enumerate(manifest_samples)
    ]
    stage1 = [
        {
            "task_id": sample["task_id"],
            "function": f"Find answer {index}",
            "arguments": [{"argument_id": 1, "argument": "A clue"}],
        }
        for index, sample in enumerate(raw_samples)
    ]
    stage2 = [
        {
            "task_id": item["task_id"],
            "arguments": [
                {
                    "argument_id": 1,
                    "argument": "A clue",
                    "counterfactual_arguments": [
                        {"counterfactual_argument": f"Alternative {candidate}"}
                        for candidate in range(4)
                    ],
                }
            ],
            "counterfactual_info": {
                "num_counterfactuals_requested": 4,
                "total_arguments": 1,
                "successful_counterfactuals": 4,
            },
        }
        for item in stage1
    ]

    run_dir = root / "run"
    _write_json(run_dir / "stage1_extracted.json", stage1)
    _write_json(run_dir / "stage2_counterfactual.json", stage2)
    stage1_policy = plan_a._stage_policy("stage1")
    stage2_policy = plan_a._stage_policy("stage2")
    for name, inputs, outputs, policy in (
        ("stage1", raw_samples, stage1, stage1_policy),
        ("stage2", stage1, stage2, stage2_policy),
    ):
        checkpoint_dir = run_dir / (
            "stage1_extracted.checkpoints"
            if name == "stage1"
            else "stage2_counterfactual.checkpoints"
        )
        checkpoint_dir.mkdir(parents=True)
        for index, (stage_input, result) in enumerate(zip(inputs, outputs)):
            _write_json(
                checkpoint_dir / f"{index}.json",
                {
                    "schema_version": plan_a.CHECKPOINT_SCHEMA_VERSION,
                    "stage": name,
                    "task_id": stage_input["task_id"],
                    "input_sha256": plan_a._canonical_sha256(stage_input),
                    "policy": policy,
                    "result": result,
                },
            )


def test_5090_preflight_requires_complete_stage1_and_stage2(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)

    report = validate_migrated_root(
        root,
        expected_manifest_path=root / "manifest.json",
    )

    assert report["safe_to_resume"] is True
    assert report["stage_counts"] == {
        "stage1_extracted": 100,
        "stage2_counterfactual": 100,
        "stage3_predecessor": 0,
    }
    assert report["policies_and_hashes"] == "verified"
    assert report["aggregate_matches_checkpoints"] is True


def test_5090_preflight_rejects_incomplete_checkpoint_set(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)
    (root / "run" / "stage2_counterfactual.checkpoints" / "99.json").unlink()

    with pytest.raises(WorkflowError, match="checkpoint count is 99"):
        validate_migrated_root(
            root,
            expected_manifest_path=root / "manifest.json",
        )


def test_5090_preflight_rejects_checkpoint_policy_tampering(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)
    checkpoint = root / "run" / "stage2_counterfactual.checkpoints" / "0.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["policy"]["requested_model"] = "another-model"
    _write_json(checkpoint, payload)

    with pytest.raises(WorkflowError, match="checkpoint policy does not match"):
        validate_migrated_root(
            root,
            expected_manifest_path=root / "manifest.json",
        )


def test_5090_preflight_rejects_manifest_drift(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)
    expected = tmp_path / "expected.json"
    expected_payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_payload["samples"][0]["task_id"] = "different-task"
    _write_json(expected, expected_payload)

    with pytest.raises(WorkflowError, match="does not match the fixed manifest"):
        validate_migrated_root(root, expected_manifest_path=expected)


def test_checkpoint_inventory_does_not_require_private_contents(tmp_path: Path) -> None:
    root = tmp_path / "run"
    checkpoint_dir = root / "run" / "stage3_predecessor.checkpoints"
    checkpoint_dir.mkdir(parents=True)
    _write_json(checkpoint_dir / "opaque.json", {"private": "not returned"})

    assert checkpoint_inventory(root)["stage3_predecessor"] == 1


def test_5090_stage3_uses_16_internal_workers_from_one_configuration() -> None:
    assert STAGE3_WORKERS == 16
    assert construction_modal.STAGE3_WORKERS == STAGE3_WORKERS
    assert build_construction_parser().parse_args([]).workers == STAGE3_WORKERS
    validate_worker_configuration(STAGE3_WORKERS)

    command = construction_command(Path("/private/browsecomp-run"))
    assert command == [
        sys.executable,
        "-u",
        "-m",
        "reproduction.browsecomp_construction_5090",
        "--root",
        "/private/browsecomp-run",
        "--workers",
        "16",
    ]


def test_5090_stage3_rejects_worker_configuration_drift() -> None:
    with pytest.raises(WorkflowError, match="exactly 16 internal workers"):
        validate_worker_configuration(12)


def test_5090_stage3_uses_ten_disjoint_process_shards() -> None:
    inputs = [{"task_id": f"task-{index}"} for index in range(100)]
    shards = [
        stage3_shard_inputs(inputs, index, STAGE3_SHARDS)
        for index in range(STAGE3_SHARDS)
    ]

    assert STAGE3_SHARDS == 10
    assert STAGE3_SHARD_WORKERS == 2
    assert all(len(shard) == 10 for shard in shards)
    flattened = [item["task_id"] for shard in shards for item in shard]
    assert len(flattened) == len(set(flattened)) == 100
    assert set(flattened) == {item["task_id"] for item in inputs}
    for index in range(STAGE3_SHARDS):
        validate_shard_configuration(
            index,
            STAGE3_SHARDS,
            STAGE3_SHARD_WORKERS,
        )


def test_5090_stage3_rejects_shard_configuration_drift() -> None:
    with pytest.raises(WorkflowError, match="exactly 10 shards"):
        validate_shard_configuration(0, 9, STAGE3_SHARD_WORKERS)
    with pytest.raises(WorkflowError, match="outside"):
        validate_shard_configuration(10, STAGE3_SHARDS, STAGE3_SHARD_WORKERS)
    with pytest.raises(WorkflowError, match="exactly 2 workers"):
        validate_shard_configuration(0, STAGE3_SHARDS, 3)


def test_5090_stage3_shard_commands_are_explicit_and_disjoint() -> None:
    commands = [
        construction_shard_command(Path("/private/browsecomp-run"), index)
        for index in range(STAGE3_SHARDS)
    ]

    assert len(commands) == 10
    assert len({tuple(command) for command in commands}) == 10
    for index, command in enumerate(commands):
        assert command[-6:] == [
            "--shard-index",
            str(index),
            "--shard-count",
            "10",
            "--workers",
            "2",
        ]


def test_5090_pipeline_launches_ten_shards_before_finalizing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
            self.pid = 8000 + len(commands)

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(pipeline_5090.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        pipeline_5090,
        "finalize_sharded_run",
        lambda root: {"stage_counts": {"stage3": 100}, "root": str(root)},
    )
    state_path = tmp_path / "pipeline_state.json"

    report = run_construction_shards(tmp_path, state_path, 123.0)

    assert len(commands) == STAGE3_SHARDS == 10
    assert len({tuple(command) for command, _kwargs in commands}) == 10
    assert all(kwargs["cwd"] == pipeline_5090.REPOSITORY_ROOT for _, kwargs in commands)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["construction_shards"] == 10
    assert state["workers_per_shard"] == 2
    assert state["total_construction_workers"] == 20
    assert len(state["shard_pids"]) == 10
    assert report["stage_counts"]["stage3"] == 100
    assert all(kwargs["start_new_session"] is True for _, kwargs in commands)
    assert all(callable(kwargs["preexec_fn"]) for _, kwargs in commands)


def test_5090_pipeline_rejects_any_failed_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeProcess:
        next_pid = 9000

        def __init__(self, _command, **_kwargs):
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1

        def wait(self):
            return 7 if self.pid == 9004 else 0

        def poll(self):
            return 7 if self.pid == 9004 else 0

    monkeypatch.setattr(pipeline_5090.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        pipeline_5090,
        "finalize_sharded_run",
        lambda _root: pytest.fail("failed shards must not be finalized"),
    )

    with pytest.raises(WorkflowError, match="'4': 7"):
        run_construction_shards(
            tmp_path,
            tmp_path / "pipeline_state.json",
            123.0,
        )


def test_5090_pipeline_lets_healthy_shards_finish_after_one_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processes = []

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            self.pid = 9050 + len(processes)
            self.poll_count = 0
            processes.append(self)

        def poll(self):
            self.poll_count += 1
            if self.pid == 9053:
                return 9
            return 0 if self.poll_count >= 2 else None

    monkeypatch.setattr(pipeline_5090.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(pipeline_5090.time, "sleep", lambda _seconds: None)

    with pytest.raises(WorkflowError, match="'3': 9"):
        run_construction_shards(
            tmp_path,
            tmp_path / "pipeline_state.json",
            123.0,
        )

    assert len(processes) == STAGE3_SHARDS
    assert all(process.poll_count >= 2 for process in processes if process.pid != 9053)


def test_5090_stage3_shard_lock_rejects_duplicate_owner(tmp_path: Path) -> None:
    first_owner = acquire_stage3_shard_lock(tmp_path, 3)
    try:
        with pytest.raises(WorkflowError, match="already has an owner"):
            acquire_stage3_shard_lock(tmp_path, 3)
    finally:
        first_owner.close()

    resumed_owner = acquire_stage3_shard_lock(tmp_path, 3)
    resumed_owner.close()


def test_5090_construction_mode_lock_separates_monolithic_and_sharded(
    tmp_path: Path,
) -> None:
    first_shard = acquire_construction_mode_lock(tmp_path, exclusive=False)
    second_shard = acquire_construction_mode_lock(tmp_path, exclusive=False)
    try:
        with pytest.raises(WorkflowError, match="construction mode conflicts"):
            acquire_construction_mode_lock(tmp_path, exclusive=True)
    finally:
        second_shard.close()
        first_shard.close()

    monolithic = acquire_construction_mode_lock(tmp_path, exclusive=True)
    try:
        with pytest.raises(WorkflowError, match="construction mode conflicts"):
            acquire_construction_mode_lock(tmp_path, exclusive=False)
    finally:
        monolithic.close()


def test_5090_pipeline_cleans_started_shards_when_spawn_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processes = []

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            if len(processes) == 3:
                raise OSError("simulated spawn failure")
            self.pid = 9100 + len(processes)
            self.returncode = None
            processes.append(self)

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = -15

        def wait(self, timeout=None):
            assert timeout == 30
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(pipeline_5090.subprocess, "Popen", FakeProcess)

    with pytest.raises(OSError, match="simulated spawn failure"):
        run_construction_shards(
            tmp_path,
            tmp_path / "pipeline_state.json",
            123.0,
        )

    assert len(processes) == 3
    assert all(process.returncode == -15 for process in processes)


def test_5090_pipeline_cleans_shards_when_state_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processes = []

    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            self.pid = 9200 + len(processes)
            self.returncode = None
            processes.append(self)

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = -15

        def wait(self, timeout=None):
            assert timeout == 30
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(pipeline_5090.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        pipeline_5090,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("state write failed")),
    )

    with pytest.raises(OSError, match="state write failed"):
        run_construction_shards(
            tmp_path,
            tmp_path / "pipeline_state.json",
            123.0,
        )

    assert len(processes) == STAGE3_SHARDS
    assert all(process.returncode == -15 for process in processes)


def test_5090_preflight_uses_one_stage3_checkpoint_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)
    _write_json(root / "run" / "stage3_predecessor.json", [])
    original_inventory = construction_5090.checkpoint_inventory

    def inventory_then_publish(target: Path):
        inventory = original_inventory(target)
        stage2 = json.loads(
            (target / "run" / "stage2_counterfactual.json").read_text(
                encoding="utf-8"
            )
        )
        result = {
            "task_id": stage2[0]["task_id"],
            "predecessors": [{}, {}, {}],
            "predecessor_functions": [
                {"predecessor_function": f"predecessor-{index}"}
                for index in range(3)
            ],
            "verification_passed": True,
            "independence_passed": True,
        }
        checkpoint_dir = target / "run" / "stage3_predecessor.checkpoints"
        _write_json(
            plan_a._checkpoint_path(checkpoint_dir, stage2[0]["task_id"]),
            plan_a._checkpoint_envelope("stage3", stage2[0], result),
        )
        return inventory

    monkeypatch.setattr(
        construction_5090,
        "checkpoint_inventory",
        inventory_then_publish,
    )

    report = validate_migrated_root(
        root,
        expected_manifest_path=root / "manifest.json",
        require_stage3_aggregate=False,
    )

    assert report["stage_counts"]["stage3_predecessor"] == 1


def test_5090_relaxed_preflight_accepts_checkpoint_only_recovery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)
    stage2 = json.loads(
        (root / "run" / "stage2_counterfactual.json").read_text(encoding="utf-8")
    )
    result = {
        "task_id": stage2[0]["task_id"],
        "predecessors": [{}, {}, {}],
        "predecessor_functions": [
            {"predecessor_function": f"predecessor-{index}"}
            for index in range(3)
        ],
        "verification_passed": True,
        "independence_passed": True,
    }
    checkpoint_dir = root / "run" / "stage3_predecessor.checkpoints"
    _write_json(
        plan_a._checkpoint_path(checkpoint_dir, stage2[0]["task_id"]),
        plan_a._checkpoint_envelope("stage3", stage2[0], result),
    )

    report = validate_migrated_root(
        root,
        expected_manifest_path=root / "manifest.json",
        require_stage3_aggregate=False,
    )

    assert report["stage_counts"]["stage3_predecessor"] == 1
    assert report["aggregate_matches_checkpoints"] is False


def test_5090_sharded_finalize_rebuilds_ordered_aggregate(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)
    stage2 = json.loads(
        (root / "run" / "stage2_counterfactual.json").read_text(encoding="utf-8")
    )
    checkpoint_dir = root / "run" / "stage3_predecessor.checkpoints"
    for item in reversed(stage2):
        result = {
            "task_id": item["task_id"],
            "predecessors": [{}, {}, {}],
            "predecessor_functions": [
                {"predecessor_function": f"predecessor-{index}"}
                for index in range(3)
            ],
            "verification_passed": True,
            "independence_passed": True,
        }
        _write_json(
            plan_a._checkpoint_path(checkpoint_dir, item["task_id"]),
            plan_a._checkpoint_envelope("stage3", item, result),
        )
    _write_json(root / "run" / "stage3_predecessor.json", [])
    usage = root / "usage" / "llm_usage.jsonl"
    usage.parent.mkdir(parents=True)
    usage.write_text(
        json.dumps(
            {
                "event": "usage",
                "requested_model": plan_a.PLAN_A_MODEL,
                "resolved_model": plan_a.PLAN_A_MODEL,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    relaxed = validate_migrated_root(
        root,
        expected_manifest_path=root / "manifest.json",
        require_stage3_aggregate=False,
    )
    assert relaxed["aggregate_matches_checkpoints"] is False
    with pytest.raises(WorkflowError, match="aggregate does not match"):
        validate_migrated_root(
            root,
            expected_manifest_path=root / "manifest.json",
        )

    report = finalize_sharded_run(
        root,
        expected_manifest_path=root / "manifest.json",
    )

    assert report["stage_counts"] == {
        "stage1": 100,
        "stage2": 100,
        "stage3": 100,
    }
    aggregate = json.loads(
        (root / "run" / "stage3_predecessor.json").read_text(encoding="utf-8")
    )
    assert [item["task_id"] for item in aggregate] == [
        item["task_id"] for item in stage2
    ]
    assert json.loads(
        (root / "final_dataset" / "browsecomp_plus_final.json").read_text(
            encoding="utf-8"
        )
    ) == aggregate


def test_5090_sharded_finalize_rejects_an_active_shard(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _build_migrated_root(root)
    active_shard = acquire_stage3_shard_lock(root, 6)
    try:
        with pytest.raises(WorkflowError, match="already has an owner"):
            finalize_sharded_run(
                root,
                expected_manifest_path=root / "manifest.json",
            )
    finally:
        active_shard.close()


def test_5090_stage3_shard_writes_only_its_assigned_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from intent_construction.retrospective_expansion.predecessor.generate_predecessors import (
        PredecessorGenerator,
    )

    root = tmp_path / "run"
    _build_migrated_root(root)
    (root / "run" / "stage3_predecessor.checkpoints").mkdir(parents=True)
    _write_json(root / "run" / "stage3_predecessor.json", [])
    monkeypatch.setattr(
        construction_5090,
        "_validate_secret_environment",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        PredecessorGenerator,
        "__init__",
        lambda self, **_kwargs: None,
    )

    def complete(self, item):
        return {
            "task_id": item["task_id"],
            "predecessors": [{}, {}, {}],
            "predecessor_functions": [
                {"predecessor_function": f"predecessor-{index}"}
                for index in range(3)
            ],
            "verification_passed": True,
            "independence_passed": True,
        }

    monkeypatch.setattr(PredecessorGenerator, "generate_predecessors", complete)

    first = run_stage3_shard(
        root,
        shard_index=4,
        shard_count=STAGE3_SHARDS,
        workers=STAGE3_SHARD_WORKERS,
        expected_manifest_path=root / "manifest.json",
    )
    resumed = run_stage3_shard(
        root,
        shard_index=4,
        shard_count=STAGE3_SHARDS,
        workers=STAGE3_SHARD_WORKERS,
        expected_manifest_path=root / "manifest.json",
    )

    assert first["assigned"] == first["completed"] == 10
    assert first["already_complete"] == 0
    assert resumed["assigned"] == resumed["already_complete"] == 10
    assert resumed["completed"] == 0
    assert checkpoint_inventory(root)["stage3_predecessor"] == 10
    assert json.loads(
        (root / "run" / "stage3_predecessor.json").read_text(encoding="utf-8")
    ) == []


def test_5090_pipeline_owner_lock_rejects_a_second_owner(tmp_path: Path) -> None:
    first_owner = acquire_pipeline_owner_lock(tmp_path)
    try:
        with pytest.raises(WorkflowError, match="another 5090 BrowseComp"):
            acquire_pipeline_owner_lock(tmp_path)
    finally:
        first_owner.close()

    resumed_owner = acquire_pipeline_owner_lock(tmp_path)
    resumed_owner.close()


def test_5090_retriever_identity_is_pinned() -> None:
    payload = ready_payload()
    assert payload["runtime"] == "rtx-5090"
    assert payload["retriever_revision"].startswith("browsecomp-retriever-")
    assert len(payload["model_revision"]) == 40
    assert len(payload["corpus_revision"]) == 40
    assert len(payload["index_revision"]) == 40


def test_5090_retriever_accepts_only_fixed_search_contract() -> None:
    assert validate_search_payload({"query": "public smoke query", "k": 5}) == (
        "public smoke query",
        5,
    )
    with pytest.raises(ValueError, match="non-empty"):
        validate_search_payload({"query": "", "k": 5})
    with pytest.raises(ValueError, match="search_k at 5"):
        validate_search_payload({"query": "public smoke query", "k": 10})


def test_5090_pipeline_requires_locked_uncapped_runtime() -> None:
    environment = build_remote_environment(
        "secret-only-in-test-memory",
        "/nvme/home/fanruibo/evolving-intent-private/evaluation.jsonl",
    )
    validate_runtime_environment(environment)
    environment["LLM_COST_HARD_CAP_USD"] = "30"
    with pytest.raises(WorkflowError, match="forbids environment limits"):
        validate_runtime_environment(environment)


def test_5090_retriever_subprocess_scrubs_model_credentials() -> None:
    assert RETRIEVER_SECRET_ENVIRONMENT == {
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    }


def test_5090_launcher_keeps_secret_out_of_command_and_paths() -> None:
    secret = "secret-only-in-test-memory"
    payload = build_launch_payload(
        api_key=secret,
        remote_cwd="/nvme/home/fanruibo/evolving-intent-reproduction",
        remote_root="/nvme/home/fanruibo/evolving-intent-private/browsecomp-n100",
        remote_cache="/nvme/home/fanruibo/evolving-intent-cache",
        remote_log="/nvme/home/fanruibo/evolving-intent-private/pipeline.log",
        remote_python=".venv/bin/python",
    )
    assert secret not in json.dumps(payload["command"])
    assert secret not in payload["cwd"]
    assert secret not in payload["log_path"]
    assert payload["environment"]["LLM_API_KEY"] == secret
    assert payload["environment"]["OPENAI_API_KEY"] == secret
    assert payload["environment"]["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert "LLM_COST_HARD_CAP_USD" not in payload["environment"]


def test_5090_launcher_rejects_broad_or_external_paths() -> None:
    with pytest.raises(CredentialError, match="specific absolute path"):
        validate_remote_path("/nvme/home/fanruibo", "remote root")
    with pytest.raises(CredentialError, match="must be under"):
        validate_remote_path("/tmp/evolving-intent", "remote root")


def test_5090_launcher_is_locked_to_selected_node() -> None:
    from reproduction.run_5090_with_cc_switch import ALLOWED_HOSTS

    assert ALLOWED_HOSTS == {
        "fanruibo-5090-1",
        "fanruibo-10-123-4-13",
    }
    assert "fanruibo" not in ALLOWED_HOSTS
    assert "fanruibo-10-123-4-14" not in ALLOWED_HOSTS


def test_5090_launcher_supports_direct_help_invocation() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "reproduction/run_5090_with_cc_switch.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--host" in result.stdout


def test_5090_direct_dependencies_are_exactly_pinned() -> None:
    requirements = Path("reproduction/requirements-browsecomp-5090.txt")
    lines = [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    assert all(line.count("==") == 1 for line in lines)
    assert "torch==2.7.1" in lines


def test_browsecomp_evaluation_uses_the_pinned_virtualenv_python() -> None:
    script = Path("evaluation/scripts/run_browsecomp.sh").read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"' in script
    assert (
        '"$PYTHON_BIN" -u -m evaluation.runners.run_browsecomp_experiment'
        in script
    )
    assert '"$PYTHON_BIN" -u evaluation/runners/run_browsecomp_experiment.py' not in script
    assert "\n    python -u evaluation/runners/run_browsecomp_experiment.py" not in script
