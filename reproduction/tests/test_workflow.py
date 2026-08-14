from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.workflow import (
    PLAN_A_SETTING_NAMES,
    REPO_ROOT,
    WorkflowError,
    aggregate_results,
    build_manifest,
    contains_secret_material,
    load_config,
    repeat_last_user_turn,
    select_official_samples,
    summarize_ledger,
    wilson_interval,
)


class SelectionTests(unittest.TestCase):
    def test_first_twenty_published_ids_are_fixed(self):
        selected = select_official_samples(
            REPO_ROOT / "intent_construction/eval_indices/gsm8k_eval_ids.json",
            20,
        )
        self.assertEqual(
            [item["original_id"] for item in selected],
            [12, 14, 16, 36, 40, 43, 49, 50, 60, 62, 63, 70, 80, 87, 90, 95, 100, 115, 118, 122],
        )

    def test_checked_in_selection_matches_official_manifest(self):
        checked_in = json.loads(
            (REPO_ROOT / "reproduction/config/selected_gsm8k_n20.json").read_text()
        )["samples"]
        official = select_official_samples(
            REPO_ROOT / "intent_construction/eval_indices/gsm8k_eval_ids.json",
            20,
        )
        self.assertEqual(checked_in, official)


class SettingTests(unittest.TestCase):
    def test_plan_has_exact_four_settings_and_compositions(self):
        config = load_config()
        self.assertEqual(config["construction"]["default_model"], "kimi-k2.6")
        self.assertEqual(config["evaluation"]["default_model"], "kimi-k2.6")
        self.assertEqual(config["construction"]["default_max_output_tokens"], 8192)
        self.assertIsNone(config["budget"]["hard_cap_usd"])
        self.assertIsNone(config["evaluation"]["temperature"])
        self.assertIsNone(config["evaluation"]["reasoning_effort"])
        settings = config["evaluation"]["settings"]
        self.assertEqual(tuple(item["name"] for item in settings), PLAN_A_SETTING_NAMES)
        t4 = settings[1]
        t7 = settings[3]
        self.assertEqual(t4["turns"] - 1 - t4["switches"] - t4["revisions"], 1)
        self.assertEqual(t7["turns"] - 1 - t7["switches"] - t7["revisions"], 2)

    def test_plan_accounting_clears_inherited_hard_cap(self):
        from reproduction.plan_a import _configure_accounting

        config = load_config()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "LLM_PRICE_MAP": json.dumps(
                    {"kimi-k2.6": {"input": 1, "output": 1}}
                ),
                "LLM_COST_HARD_CAP_USD": "30",
            },
            clear=True,
        ):
            _configure_accounting(Path(tmp), config)
            self.assertNotIn("LLM_COST_HARD_CAP_USD", os.environ)
            self.assertEqual(os.environ["LLM_DEFAULT_MAX_OUTPUT_TOKENS"], "8192")

    def test_repeat_control_copies_last_turn_without_label_leak(self):
        sample = SimpleNamespace(
            turns=[
                {"role": "system", "content": "solve"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "final update"},
            ],
            metadata={},
            structured_contents=[object(), object()],
        )
        repeat_last_user_turn(sample, 3)
        self.assertEqual(
            [turn["content"] for turn in sample.turns if turn["role"] == "user"],
            ["first", "final update", "final update", "final update", "final update"],
        )
        self.assertIsNone(sample.structured_contents)
        self.assertNotIn("answer", json.dumps(sample.turns).lower())

    def test_current_scheduler_builds_all_four_turn_shapes(self):
        from reproduction.plan_a import build_setting_samples

        arguments = []
        for argument_id in range(1, 5):
            value = argument_id * 10
            arguments.append(
                {
                    "argument_id": argument_id,
                    "argument": f"Fact {argument_id} has value {value}.",
                    "counterfactual_arguments": [
                        {"counterfactual_argument": f"Fact {argument_id} has value {value + 1}."},
                        {"counterfactual_argument": f"Fact {argument_id} has value {value + 2}."},
                    ],
                }
            )
        predecessors = [
            {
                "predecessor_function": f"What is predecessor quantity {index}?",
                "counterfactual_arguments": [
                    {
                        "argument_id": item["argument_id"],
                        "argument": item["argument"],
                        "is_shared": True,
                    }
                    for item in arguments
                ],
                "is_predecessor": True,
            }
            for index in range(2)
        ]
        raw = {
            "task_id": "extracted-gsm8k-test-12",
            "task": "math",
            "function": "What is the final quantity?",
            "arguments": arguments,
            "predecessor_functions": predecessors,
            "answer": "100",
        }
        settings = load_config()["evaluation"]["settings"]
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "stage3.json"
            data_path.write_text(json.dumps([raw]))
            observed = {}
            built = {}
            for setting in settings:
                samples = build_setting_samples(
                    data_path,
                    [raw["task_id"]],
                    setting,
                    seed=42,
                )
                observed[setting["name"]] = sum(
                    turn["role"] == "user" for turn in samples[0].turns
                )
                built[setting["name"]] = samples[0]
        self.assertEqual(
            observed,
            {
                "single_t1": 1,
                "evolve_t4_g1_p1": 4,
                "repeat_control_t7": 7,
                "evolve_t7_g2_p2": 7,
            },
        )
        t4 = built["evolve_t4_g1_p1"]
        repeat = built["repeat_control_t7"]
        t4_users = [turn for turn in t4.turns if turn["role"] == "user"]
        repeat_users = [turn for turn in repeat.turns if turn["role"] == "user"]
        self.assertEqual(repeat_users[:4], t4_users)
        self.assertEqual(repeat_users[4:], [t4_users[-1]] * 3)
        repeat_plan = repeat.metadata["change_plan"]
        self.assertTrue(all(item["type"] == "repeat" for item in repeat_plan["transitions"][-3:]))
        self.assertEqual(repeat_plan["intent_trajectory"][-3:], [repeat_plan["intent_trajectory"][3]] * 3)


class StatisticsTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_accuracy(self):
        low, high = wilson_interval(15, 20)
        self.assertLess(low, 0.75)
        self.assertGreater(high, 0.75)
        self.assertAlmostEqual(low, 0.5313, places=3)
        self.assertAlmostEqual(high, 0.8881, places=3)

    def test_aggregation_uses_paired_task_ids(self):
        config = load_config()
        manifest = build_manifest(
            config,
            run_id="test-run",
            sample_count=10,
            selection_reason="test",
            construction_model="model-a",
            evaluation_model="model-b",
            deadline=None,
            dry_run=True,
        )
        ids = [item["task_id"] for item in manifest["dataset"]["selected_samples"]]
        baseline = {task_id: {"correct": index < 8, "success": True} for index, task_id in enumerate(ids)}
        changed = {task_id: {"correct": index < 6, "success": True} for index, task_id in enumerate(ids)}
        payloads = {
            "single_t1": {"results": baseline},
            "evolve_t4_g1_p1": {"results": changed},
            "repeat_control_t7": {"results": baseline},
            "evolve_t7_g2_p2": {"results": changed},
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = aggregate_results(
                manifest,
                payloads,
                ledger_path=Path(tmp) / "ledger.jsonl",
            )
        paired = summary["paired_differences"]["evolve_t4_g1_p1"]
        self.assertEqual(paired["paired_samples"], 10)
        self.assertEqual(paired["regressed"], 2)
        self.assertEqual(paired["improved"], 0)
        self.assertAlmostEqual(paired["percentage_point_difference"], -20.0)

    def test_aggregation_excludes_failed_calls_from_every_accuracy(self):
        config = load_config()
        manifest = build_manifest(
            config,
            run_id="failed-call-test",
            sample_count=10,
            selection_reason="test",
            construction_model="model-a",
            evaluation_model="model-b",
            deadline=None,
            dry_run=True,
        )
        ids = [item["task_id"] for item in manifest["dataset"]["selected_samples"]]
        payloads = {}
        for name in PLAN_A_SETTING_NAMES:
            payloads[name] = {
                "results": {
                    task_id: {"correct": True, "success": True, "error": None}
                    for task_id in ids
                }
            }
        payloads["evolve_t7_g2_p2"]["results"][ids[-1]] = {
            "correct": False,
            "success": False,
            "error": "API timeout",
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = aggregate_results(
                manifest, payloads, ledger_path=Path(tmp) / "ledger.jsonl"
            )
        self.assertEqual(len(summary["sample_selection"]["common_successful_task_ids"]), 9)
        self.assertTrue(all(item["completed"] == 9 for item in summary["settings"].values()))
        self.assertEqual(summary["settings"]["evolve_t7_g2_p2"]["raw_completed"], 10)
        self.assertEqual(summary["settings"]["evolve_t7_g2_p2"]["failed_calls"], 1)

    def test_ledger_summary_uses_provider_usage_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "cost.jsonl"
            ledger.write_text(
                json.dumps({"event": "stage_start"}) + "\n"
                + json.dumps(
                    {
                        "event": "usage",
                        "cost_usd": 0.25,
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cached_tokens": 10,
                        "reasoning_tokens": 5,
                    }
                )
                + "\n"
            )
            summary = summarize_ledger(ledger, {"hard_cap_usd": None})
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["actual_usd_recorded"], 0.25)
        self.assertEqual(summary["input_tokens"], 100)
        self.assertEqual(summary["reasoning_tokens"], 5)
        self.assertIsNone(summary["hard_cap_usd"])

    def test_ledger_summary_rejects_unpriced_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "cost.jsonl"
            ledger.write_text(
                json.dumps({"event": "usage", "cost_usd": None}) + "\n"
            )
            with self.assertRaisesRegex(WorkflowError, "unpriced response"):
                summarize_ledger(ledger, {"hard_cap_usd": None})


class ManifestTests(unittest.TestCase):
    def test_manifest_is_explicit_and_has_no_secret_material(self):
        config = load_config()
        manifest = build_manifest(
            config,
            run_id="offline-test",
            sample_count=20,
            selection_reason="explicit",
            construction_model="gpt-test",
            evaluation_model="gpt-test",
            deadline=None,
            dry_run=True,
        )
        self.assertEqual(len(manifest["dataset"]["selected_samples"]), 20)
        self.assertFalse(contains_secret_material(manifest))


if __name__ == "__main__":
    unittest.main()
