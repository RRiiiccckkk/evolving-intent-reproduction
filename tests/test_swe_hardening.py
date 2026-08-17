import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import Mock, patch

from evaluation.swe_bench import modal_app
from evaluation.swe_bench.canary import (
    _validate_reusable_agent_result,
    build_canary_runner_command,
)
from evaluation.common.swe_harness import (
    SWEHarness,
    _CURRENT_MODAL_CGROUP_WRITE,
    _CURRENT_MODAL_REPO_SCRIPT,
    _CURRENT_MODAL_WRITE,
    _LEGACY_MODAL_CGROUP_WRITE,
    _LEGACY_MODAL_REPO_SCRIPT,
    _LEGACY_MODAL_WRITE,
    _PREVIOUS_MODAL_REPO_SCRIPT,
    _modal_credential_marker,
    _patch_swebench_modal_source,
)
from evaluation.common.swe_minisweagent_scaffold import (
    EnvironmentExecutionError,
    _ToolBudgetEnvironment,
    _environment_failure,
)
from evaluation.runners import run_swe_mini_agent as swe_runner
from evaluation.runners.run_swe_mini_agent import _load_reusable_mini_result
from intent_construction.intent_extraction.core import llm_utils
from evaluation.swe_bench.run import _run_subprocess, build_runner_command, run_hardened
from evaluation.swe_bench.state import (
    EXPECTED_MODEL,
    EXPECTED_REASONING_EFFORT,
    MODEL_STEP_LIMIT_PER_TURN,
    PUBLISHED_TASK_COUNT,
    SCENARIOS,
    TOOL_CALL_LIMIT_PER_TURN,
    HardeningError,
    TaskCheckpointStore,
    ToolCallCounter,
    assert_kimi_tool_policy,
    assert_no_output_limit_fields,
    load_published_task_ids,
    load_published_id_map,
    read_usage_events,
    validate_aggregate_results,
    validate_requested_models,
    validate_runtime_environment,
    validate_usage_events,
)


def _runtime_environment(**overrides):
    values = {
        "LLM_BACKEND": "compatible",
        "LLM_API_KEY": "fake-key",
        "LLM_BASE_URL": "https://example.invalid/v1",
        "LLM_MODEL_MAP": json.dumps({EXPECTED_MODEL: EXPECTED_MODEL}),
        "LLM_PRICE_MAP": json.dumps(
            {EXPECTED_MODEL: {"input": 1.0, "output": 2.0}}
        ),
        "LLM_LOCKED_MODEL": EXPECTED_MODEL,
        "LLM_REASONING_EFFORT": EXPECTED_REASONING_EFFORT,
        "LLM_DISABLE_OUTPUT_LIMITS": "1",
    }
    values.update(overrides)
    return values


def _valid_result(task_id, *, scenario, resolved=EXPECTED_MODEL):
    return {
        "task_id": task_id,
        "prediction": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
        "correct": False,
        "success": True,
        "error": None,
        "metadata": {
            "requested_model": EXPECTED_MODEL,
            "resolved_model": resolved,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "checkpoint_scenario": scenario.name,
            "tool_call_limit_per_turn": TOOL_CALL_LIMIT_PER_TURN,
            "n_user_turns_delivered": scenario.turns,
            "per_turn_tool_calls": [7] * scenario.turns,
        },
        "swe_eval": {
            "resolved": False,
            "patch_extracted": True,
            "patch_apply_ok": True,
            "harness_error": None,
        },
    }


class RuntimePolicyTests(unittest.TestCase):
    def test_compatible_kimi_environment_is_accepted(self):
        resolved = validate_runtime_environment(_runtime_environment())
        self.assertEqual(resolved, EXPECTED_MODEL)

    def test_only_one_requested_kimi_model_is_accepted(self):
        validate_requested_models([EXPECTED_MODEL])
        for models in ([], ["gpt-5.1"], [EXPECTED_MODEL, "gpt-5.1"]):
            with self.subTest(models=models):
                with self.assertRaises(HardeningError):
                    validate_requested_models(models)

    def test_output_limit_and_cost_gate_settings_are_rejected(self):
        for name in (
            "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
            "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
            "LLM_COST_HARD_CAP_USD",
        ):
            with self.subTest(name=name):
                with self.assertRaises(HardeningError):
                    validate_runtime_environment(_runtime_environment(**{name: "1"}))
        with self.assertRaises(HardeningError):
            validate_runtime_environment(
                _runtime_environment(LLM_DISABLE_OUTPUT_LIMITS="0")
            )

    def test_model_reasoning_and_resolution_locks_are_required(self):
        for overrides in (
            {"LLM_LOCKED_MODEL": "gpt-5.1"},
            {"LLM_REASONING_EFFORT": "high"},
            {"LLM_MODEL_MAP": json.dumps({EXPECTED_MODEL: "provider-alias"})},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(HardeningError):
                    validate_runtime_environment(_runtime_environment(**overrides))

    def test_model_configuration_cannot_contain_output_limit_fields(self):
        for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            with self.subTest(field=field):
                with self.assertRaises(HardeningError):
                    assert_no_output_limit_fields({"model": {field: None}})

    def test_every_scenario_has_the_fixed_200_call_budget(self):
        self.assertEqual({scenario.name for scenario in SCENARIOS}, {"single", "evolve"})
        self.assertTrue(
            all(
                scenario.tool_call_limit_per_turn == TOOL_CALL_LIMIT_PER_TURN == 200
                for scenario in SCENARIOS
            )
        )

    def test_actual_tool_counter_stops_at_200_and_resets_per_turn(self):
        counter = ToolCallCounter(200)
        for expected in range(1, 201):
            self.assertEqual(counter.consume(), expected)
        with self.assertRaises(HardeningError):
            counter.consume()
        self.assertEqual(counter.finish_turn(), 200)
        self.assertEqual(counter.consume(), 1)

    def test_model_step_cap_is_disabled_independently_of_tool_budget(self):
        self.assertEqual(MODEL_STEP_LIMIT_PER_TURN, 0)
        self.assertEqual(TOOL_CALL_LIMIT_PER_TURN, 200)

    def test_swerex_timeout_and_dead_runtime_fail_before_another_model_step(self):
        timeout_output = {
            "output": "Timeout (60.0s) exceeded while running command",
            "returncode": -1,
            "exception_info": "An error occurred while executing the command",
            "extra": {
                "exception_type": "CommandTimeoutError",
                "exception": "Timeout (60.0s) exceeded while running command",
            },
        }
        transport_output = {
            "output": "Cannot connect to host dead.modal.host:443",
            "returncode": -1,
            "exception_info": "An error occurred while executing the command",
            "extra": {
                "exception_type": "ClientConnectorError",
                "exception": "Cannot connect to host dead.modal.host:443",
            },
        }
        for output in (timeout_output, transport_output):
            with self.subTest(output=output["extra"]["exception_type"]):
                counter = ToolCallCounter(200)
                environment = SimpleNamespace(execute=Mock(return_value=output))
                guarded = _ToolBudgetEnvironment(environment, counter)
                with self.assertRaises(EnvironmentExecutionError):
                    guarded.execute({"command": "pytest -q"})
                self.assertEqual(counter.current, 1)
                environment.execute.assert_called_once()
                self.assertIsNotNone(_environment_failure(output))

    def test_normal_nonzero_command_is_not_a_transport_failure(self):
        output = {
            "output": "tests failed",
            "returncode": 1,
            "exception_info": "",
        }
        environment = SimpleNamespace(execute=Mock(return_value=output))
        guarded = _ToolBudgetEnvironment(environment, ToolCallCounter(200))
        self.assertEqual(guarded.execute({"command": "pytest -q"}), output)
        self.assertIsNone(_environment_failure(output))

    def test_swerex_timeouts_outlive_long_test_commands(self):
        self.assertGreaterEqual(swe_runner.SWEREX_COMMAND_TIMEOUT_SECONDS, 30 * 60)
        self.assertGreater(
            swe_runner.SWEREX_RUNTIME_TIMEOUT_SECONDS,
            swe_runner.SWEREX_COMMAND_TIMEOUT_SECONDS,
        )

    def test_modal_marker_uses_env_auth_without_persisting_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            environment = {
                "HOME": str(home),
                "MODAL_TOKEN_ID": "token-id-secret",
                "MODAL_TOKEN_SECRET": "token-secret-secret",
            }
            with patch.dict(os.environ, environment, clear=True):
                with _modal_credential_marker(True):
                    marker = home / ".modal.toml"
                    self.assertTrue(marker.exists())
                    rendered = marker.read_text(encoding="utf-8")
                    self.assertNotIn(environment["MODAL_TOKEN_ID"], rendered)
                    self.assertNotIn(environment["MODAL_TOKEN_SECRET"], rendered)
                    self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
                self.assertFalse(marker.exists())

    def test_swebench_modal_compat_applies_runtime_compatibility_fixes(self):
        source = (
            f"def __init__(self):\n    {_LEGACY_MODAL_CGROUP_WRITE}\n"
            f"def write_file(self):\n    {_LEGACY_MODAL_WRITE}\n"
            f"def get_instance_image(self):\n{_LEGACY_MODAL_REPO_SCRIPT}\n"
        )
        patched_source, changed = _patch_swebench_modal_source(source)
        self.assertTrue(changed)
        self.assertNotIn(_LEGACY_MODAL_CGROUP_WRITE, patched_source)
        self.assertNotIn(_LEGACY_MODAL_WRITE, patched_source)
        self.assertIn(_CURRENT_MODAL_CGROUP_WRITE, patched_source)
        self.assertIn(_CURRENT_MODAL_REPO_SCRIPT, patched_source)
        self.assertIn("pip<25.1", patched_source)
        self.assertIn("vcs-versioning", patched_source)
        self.assertIn(_CURRENT_MODAL_WRITE, patched_source)

        same_source, changed = _patch_swebench_modal_source(patched_source)
        self.assertFalse(changed)
        self.assertEqual(same_source, patched_source)

        upgraded_source, changed = _patch_swebench_modal_source(
            patched_source.replace(
                _CURRENT_MODAL_REPO_SCRIPT,
                _PREVIOUS_MODAL_REPO_SCRIPT,
            )
        )
        self.assertTrue(changed)
        self.assertIn(_CURRENT_MODAL_REPO_SCRIPT, upgraded_source)
        self.assertNotIn(_PREVIOUS_MODAL_REPO_SCRIPT, upgraded_source)

        with self.assertRaises(RuntimeError):
            _patch_swebench_modal_source("def write_file(self): pass\n")

    def test_canary_agent_result_can_resume_only_the_official_verifier(self):
        scenario = next(item for item in SCENARIOS if item.name == "evolve")
        task_id = "extracted-swe_bench_verified-test-repo__repo-1"
        original_id = "repo__repo-1"
        candidate = _valid_result(task_id, scenario=scenario)
        candidate["metadata"]["original_id"] = original_id
        candidate["metadata"]["exit_status"] = "Submitted"
        self.assertEqual(
            _validate_reusable_agent_result(
                candidate,
                task_id=task_id,
                original_id=original_id,
            ),
            candidate["prediction"],
        )
        for mutate in (
            lambda row: row.update(error="agent failed"),
            lambda row: row["metadata"].update(resolved_model="other-model"),
            lambda row: row["metadata"]["per_turn_tool_calls"].__setitem__(0, 201),
        ):
            broken = json.loads(json.dumps(candidate))
            mutate(broken)
            with self.assertRaises(HardeningError):
                _validate_reusable_agent_result(
                    broken,
                    task_id=task_id,
                    original_id=original_id,
                )

    def test_completed_formal_trajectory_can_retry_only_the_verifier(self):
        task_id = "extracted-swe_bench_verified-test-repo__repo-1"
        instance_id = "repo__repo-1"
        user_turn = "Fix the regression."
        submission = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "single" / f"{task_id}.json"
            path.parent.mkdir(parents=True)
            payload = {
                "trajectory_format": "mini-swe-agent-1.1",
                "info": {
                    "exit_status": "Submitted",
                    "submission": submission,
                    "model_stats": {"instance_cost": 0.25},
                    "config": {
                        "agent": {
                            "output_path": str(path.resolve()),
                            "step_limit": MODEL_STEP_LIMIT_PER_TURN,
                        },
                        "model": {
                            "model_name": EXPECTED_MODEL,
                            "model_kwargs": {
                                "reasoning_effort": EXPECTED_REASONING_EFFORT,
                                "parallel_tool_calls": False,
                            },
                        },
                        "model_type": (
                            "evaluation.common.swe_minisweagent_scaffold.LLMToolModel"
                        ),
                        "environment": {
                            "image": "docker.io/swebench/sweb.eval.x86_64."
                            "repo_1776_repo-1:latest"
                        },
                        "tool_budget": {
                            "limit_per_turn": TOOL_CALL_LIMIT_PER_TURN,
                            "completed_turns": [2],
                        },
                    },
                },
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": f"Task: {user_turn}"},
                    {"role": "assistant", "content": "working", "extra": {}},
                    {"role": "assistant", "content": "", "extra": {}},
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {
                            "exit_status": "Submitted",
                            "submission": submission,
                        },
                    },
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            reusable = _load_reusable_mini_result(
                path,
                task_id=task_id,
                instance_id=instance_id,
                model=EXPECTED_MODEL,
                user_turns=[user_turn],
                reasoning_effort=EXPECTED_REASONING_EFFORT,
                step_limit_per_turn=MODEL_STEP_LIMIT_PER_TURN,
                tool_call_limit_per_turn=TOOL_CALL_LIMIT_PER_TURN,
                use_tool_calling=True,
            )
            self.assertIsNotNone(reusable)
            result, digest = reusable
            self.assertEqual(result.submission, submission)
            self.assertEqual(result.per_turn_steps, [2])
            self.assertEqual(result.per_turn_tool_calls, [2])
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

            payload["info"]["config"]["model"]["model_name"] = "other-model"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                _load_reusable_mini_result(
                    path,
                    task_id=task_id,
                    instance_id=instance_id,
                    model=EXPECTED_MODEL,
                    user_turns=[user_turn],
                    reasoning_effort=EXPECTED_REASONING_EFFORT,
                    step_limit_per_turn=MODEL_STEP_LIMIT_PER_TURN,
                    tool_call_limit_per_turn=TOOL_CALL_LIMIT_PER_TURN,
                    use_tool_calling=True,
                )
            )

    def test_custom_loop_persists_trajectory_after_each_step(self):
        source = Path("evaluation/common/swe_minisweagent_scaffold.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_evolvingintent_with_mini_agent"
        )
        save_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
        ]
        self.assertTrue(save_calls)
        self.assertTrue(
            any(
                isinstance(node, ast.Try) and node.finalbody
                for node in ast.walk(function)
            )
        )

    def test_custom_loop_preserves_stock_format_error_guard(self):
        source = Path("evaluation/common/swe_minisweagent_scaffold.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_evolvingintent_with_mini_agent"
        )
        handlers = [
            handler
            for node in ast.walk(function)
            if isinstance(node, ast.Try)
            for handler in node.handlers
        ]
        format_handler = next(
            handler
            for handler in handlers
            if isinstance(handler.type, ast.Name) and handler.type.id == "FormatError"
        )
        generic_handler = next(
            handler
            for handler in handlers
            if isinstance(handler.type, ast.Name)
            and handler.type.id == "InterruptAgentFlow"
        )
        self.assertLess(source.index("except FormatError"), source.index("except Submitted"))
        self.assertLess(
            source.index("except EnvironmentExecutionError"),
            source.index("except FormatError"),
        )
        self.assertLess(
            source.index("except FormatError"),
            source.index("except InterruptAgentFlow"),
        )
        rendered = ast.unparse(format_handler)
        self.assertIn("n_consecutive_format_errors += 1", rendered)
        self.assertIn("max_consecutive_format_errors", rendered)
        self.assertIn("RepeatedFormatError", rendered)
        self.assertTrue(any(isinstance(node, ast.Break) for node in ast.walk(format_handler)))
        self.assertIsNotNone(generic_handler)

    def test_kimi_tool_choice_is_omitted_and_parallel_calls_are_disabled(self):
        assert_kimi_tool_policy(
            {"model_kwargs": {"parallel_tool_calls": False}}
        )
        assert_kimi_tool_policy(
            {
                "model_kwargs": {
                    "tool_choice": None,
                    "parallel_tool_calls": False,
                }
            }
        )
        for forced in ("auto", "required", "none"):
            with self.subTest(forced=forced):
                with self.assertRaises(HardeningError):
                    assert_kimi_tool_policy(
                        {
                            "model_kwargs": {
                                "tool_choice": forced,
                                "parallel_tool_calls": False,
                            }
                        }
                    )
        with self.assertRaises(HardeningError):
            assert_kimi_tool_policy(
                {"model_kwargs": {"parallel_tool_calls": True}}
            )

        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace()]))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.dict(os.environ, _runtime_environment(), clear=True):
            with patch.object(llm_utils, "get_client", return_value=client):
                llm_utils.generate_with_tools(
                    [{"role": "user", "content": "use bash"}],
                    model=EXPECTED_MODEL,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    tool_choice=None,
                    parallel_tool_calls=False,
                    max_tokens=None,
                )
        payload = create.call_args.kwargs
        self.assertNotIn("tool_choice", payload)
        self.assertIs(payload["parallel_tool_calls"], False)
        for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            self.assertNotIn(field, payload)

    def test_dry_run_writes_secret_free_manifest_without_model_calls(self):
        published = load_published_id_map(
            "intent_construction/eval_indices/swe_bench_verified_eval_ids.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "data.json"
            data_path.write_text(
                json.dumps(
                    [
                        {"task_id": task_id, "original_id": original_id}
                        for task_id, original_id in published.items()
                    ]
                ),
                encoding="utf-8",
            )
            official_path = root / "official.json"
            official_path.write_text(
                json.dumps(
                    [
                        {
                            "instance_id": original_id,
                            "repo": "org/repo",
                            "base_commit": "abc123",
                            "problem_statement": "fix it",
                            "patch": "gold patch",
                            "test_patch": "test patch",
                            "FAIL_TO_PASS": "[]",
                            "PASS_TO_PASS": "[]",
                            "version": "1.0",
                        }
                        for original_id in published.values()
                    ]
                ),
                encoding="utf-8",
            )
            environment = _runtime_environment()
            with patch.dict(os.environ, environment, clear=True):
                manifest = run_hardened(
                    data_path=data_path,
                    official_dataset_path=official_path,
                    run_dir=root / "run",
                    workers=2,
                    dry_run=True,
                )
            rendered = json.dumps(manifest)
            self.assertEqual(manifest["status"], "validated")
            self.assertNotIn(environment["LLM_API_KEY"], rendered)
            self.assertNotIn(environment["LLM_BASE_URL"], rendered)

            manifest_path = root / "run" / "manifest.json"
            stale_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stale_manifest["failure"] = "HardeningError: stale failure"
            manifest_path.write_text(json.dumps(stale_manifest), encoding="utf-8")
            with patch.dict(os.environ, environment, clear=True):
                resumed = run_hardened(
                    data_path=data_path,
                    official_dataset_path=official_path,
                    run_dir=root / "run",
                    workers=2,
                    dry_run=True,
                )
            self.assertEqual(resumed["status"], "validated")
            self.assertNotIn("failure", resumed)


class PublishedIdTests(unittest.TestCase):
    def _write_files(self, root, task_ids):
        samples = [
            {"task_id": f"task-{index}", "original_id": f"repo__repo-{index}"}
            for index in range(PUBLISHED_TASK_COUNT)
        ]
        manifest = root / "eval.json"
        ids_file = root / "ids.json"
        manifest.write_text(
            json.dumps(
                {
                    "num_samples": PUBLISHED_TASK_COUNT,
                    "samples": samples,
                }
            ),
            encoding="utf-8",
        )
        ids_file.write_text(json.dumps({"task_ids": task_ids}), encoding="utf-8")
        return manifest, ids_file

    def test_exact_50_published_ids_are_required(self):
        canonical = [f"task-{index}" for index in range(PUBLISHED_TASK_COUNT)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, ids_file = self._write_files(root, canonical)
            self.assertEqual(load_published_task_ids(manifest, ids_file), canonical)

            cases = {
                "missing": canonical[:-1],
                "extra": canonical + ["task-extra"],
                "duplicate": canonical[:-1] + [canonical[0]],
            }
            for name, task_ids in cases.items():
                with self.subTest(name=name):
                    _, ids_file = self._write_files(root, task_ids)
                    with self.assertRaises(HardeningError):
                        load_published_task_ids(manifest, ids_file)


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_resume_and_complete_aggregate(self):
        scenario = SCENARIOS[0]
        task_id = "task-1"
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskCheckpointStore(
                temporary,
                scenario=scenario,
                requested_model=EXPECTED_MODEL,
                resolved_model=EXPECTED_MODEL,
            )
            result = _valid_result(task_id, scenario=scenario)
            path = store.write(task_id, result)
            self.assertEqual(store.load(task_id), result)
            self.assertTrue(path.exists())
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])
            self.assertEqual(
                validate_aggregate_results(
                    {task_id: result},
                    [task_id],
                    store=store,
                ),
                {task_id: result},
            )

    def test_empty_corrupt_and_incomplete_checkpoints_fail(self):
        scenario = SCENARIOS[0]
        task_id = "task-1"
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskCheckpointStore(
                temporary,
                scenario=scenario,
                requested_model=EXPECTED_MODEL,
                resolved_model=EXPECTED_MODEL,
            )
            path = store.path_for(task_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            for content in ("", "{", json.dumps({"schema_version": 1})):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(HardeningError):
                        store.load(task_id)

            incomplete = _valid_result(task_id, scenario=scenario)
            incomplete["prediction"] = ""
            with self.assertRaises(HardeningError):
                store.write(task_id, incomplete)


class UsageAndProcessTests(unittest.TestCase):
    def test_ledger_rejects_any_other_requested_or_resolved_model(self):
        good = {
            "event": "usage",
            "requested_model": EXPECTED_MODEL,
            "resolved_model": EXPECTED_MODEL,
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_tokens": 2,
            "reasoning_tokens": 1,
            "cost_usd": 0.001,
        }
        summary = validate_usage_events(
            [good],
            requested_model=EXPECTED_MODEL,
            resolved_model=EXPECTED_MODEL,
        )
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["input_tokens"], 10)
        for field, bad_value in (
            ("requested_model", "gpt-5.1"),
            ("resolved_model", "another-kimi"),
        ):
            bad = dict(good)
            bad[field] = bad_value
            with self.subTest(field=field):
                with self.assertRaises(HardeningError):
                    validate_usage_events(
                        [bad],
                        requested_model=EXPECTED_MODEL,
                        resolved_model=EXPECTED_MODEL,
                    )

    def test_usage_reader_obeys_byte_offset(self):
        first = {
            "event": "usage",
            "requested_model": EXPECTED_MODEL,
            "resolved_model": EXPECTED_MODEL,
        }
        second = dict(first, input_tokens=1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "usage.jsonl"
            first_line = json.dumps(first) + "\n"
            path.write_text(first_line + json.dumps(second) + "\n", encoding="utf-8")
            self.assertEqual(read_usage_events(path, start_offset=len(first_line.encode())), [second])

    def test_failed_child_process_is_not_swallowed(self):
        def failed_runner(*args, **kwargs):
            return SimpleNamespace(returncode=9)

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(HardeningError, "exit code 9"):
                _run_subprocess(
                    ["false"],
                    log_path=Path(temporary) / "child.log",
                    environment={},
                    runner=failed_runner,
                )


class ModalEntryTests(unittest.TestCase):
    def test_modal_entry_accepts_secret_name_but_no_key_arguments(self):
        signature = inspect.signature(modal_app.main)
        parameter_names = set(signature.parameters)
        self.assertIn("secret_name", parameter_names)
        self.assertNotIn("api_key", parameter_names)
        self.assertNotIn("base_url", parameter_names)

        secret_value = "do-not-put-this-in-command"
        with patch.dict(os.environ, {"LLM_API_KEY": secret_value}):
            command = modal_app.build_remote_command(
                data_path="final_dataset/swe_bench_verified_final.json",
                run_name="kimi-k2.6",
                workers=8,
            )
        self.assertNotIn(secret_value, " ".join(command))

    def test_outer_modal_canary_is_fixed_and_isolated(self):
        command = modal_app.build_canary_remote_command()
        rendered = " ".join(command)
        self.assertIn("-m evaluation.swe_bench.canary", rendered)
        self.assertIn(modal_app.DEFAULT_CANARY_TASK_ID, command)
        self.assertIn(
            f"/results/{modal_app.DEFAULT_CANARY_RUN_NAME}",
            command,
        )
        for forbidden in (
            "tool_choice",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
        ):
            self.assertNotIn(forbidden, rendered)

        modal_app.validate_canary_isolation(
            run_name=modal_app.DEFAULT_CANARY_RUN_NAME,
            volume_name=modal_app.DEFAULT_CANARY_VOLUME,
        )
        for run_name, volume_name in (
            (modal_app.FORMAL_RUN_NAME, modal_app.DEFAULT_CANARY_VOLUME),
            (modal_app.DEFAULT_CANARY_RUN_NAME, modal_app.FORMAL_VOLUME_NAME),
            (modal_app.DEFAULT_CANARY_RUN_NAME, "unscoped-volume"),
        ):
            with self.subTest(run_name=run_name, volume_name=volume_name):
                with self.assertRaises(ValueError):
                    modal_app.validate_canary_isolation(
                        run_name=run_name,
                        volume_name=volume_name,
                    )

    def test_modal_canary_entry_accepts_only_secret_name_not_key_values(self):
        signature = inspect.signature(modal_app.canary)
        parameter_names = set(signature.parameters)
        self.assertIn("secret_name", parameter_names)
        self.assertNotIn("api_key", parameter_names)
        self.assertNotIn("base_url", parameter_names)

    def test_runner_command_has_all_hardening_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = build_runner_command(
                data_path=root / "data.json",
                task_ids_path=root / "ids.json",
                run_dir=root / "run",
                official_dataset_path=root / "official.json",
                scenario_name="single",
                workers=2,
            )
        rendered = " ".join(command)
        for expected in (
            "--models kimi-k2.6",
            "--step_limit_per_turn 0",
            "--tool_call_limit_per_turn 200",
            "--cost_limit 0",
            "--environment_class swerex_modal",
            "--harness_dataset_path",
            "--strict_task_ids",
        ):
            self.assertIn(expected, rendered)


class ModalHarnessTests(unittest.TestCase):
    def test_official_harness_receives_modal_true_and_parses_report(self):
        captured = {}
        run_evaluation = ModuleType("swebench.harness.run_evaluation")
        constants = ModuleType("swebench.harness.constants")
        constants.RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")
        constants.LOG_REPORT = "report.json"

        def fake_main(**kwargs):
            captured.update(kwargs)
            report_dir = (
                constants.RUN_EVALUATION_LOG_DIR
                / kwargs["run_id"]
                / "kimi-k2.6-mini-agent"
                / kwargs["instance_ids"][0]
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / constants.LOG_REPORT).write_text(
                json.dumps(
                    {
                        kwargs["instance_ids"][0]: {
                            "resolved": True,
                            "patch_successfully_applied": True,
                            "tests_status": {
                                "FAIL_TO_PASS": {"success": ["f2p"], "failure": []},
                                "PASS_TO_PASS": {"success": ["p2p"], "failure": []},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

        run_evaluation.main = fake_main
        modules = {
            "swebench": ModuleType("swebench"),
            "swebench.harness": ModuleType("swebench.harness"),
            "swebench.harness.run_evaluation": run_evaluation,
            "swebench.harness.constants": constants,
        }
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "evaluation.common.swe_harness."
                "_ensure_swebench_modal_filesystem_compat"
            ), patch.dict(sys.modules, modules):
                result = SWEHarness(
                    workspace=temporary,
                    modal=True,
                    use_cache=False,
                ).verify_patch(
                    instance_id="repo__repo-1",
                    patch="diff --git a/a b/a\n",
                    model_name="kimi-k2.6-mini-agent",
                    run_id="modal-test",
                )
        self.assertTrue(captured["modal"])
        self.assertTrue(result.resolved)
        self.assertEqual(result.ftp_pass, ["f2p"])
        self.assertEqual(result.ptp_pass, ["p2p"])


class CanaryCommandTests(unittest.TestCase):
    def test_canary_pairs_modal_agent_with_official_verifier(self):
        root = Path("/tmp/isolated-swe-canary")
        command = build_canary_runner_command(
            data_path=root / "data.json",
            task_ids_path=root / "ids.json",
            official_dataset_path=root / "official.json",
            run_dir=root,
        )
        rendered = " ".join(command)
        for expected in (
            "--models kimi-k2.6",
            "--step_limit_per_turn 0",
            "--tool_call_limit_per_turn 200",
            "--environment_class swerex_modal",
            "--harness_dataset_path",
            "--checkpoint_scenario evolve",
            "--reasoning_effort medium",
            "--num_workers 1",
            "--rerun_failed",
        ):
            self.assertIn(expected, rendered)
        self.assertNotIn("--strict_task_ids", command)
        self.assertNotIn("--checkpoint_dir", command)


if __name__ == "__main__":
    unittest.main()
