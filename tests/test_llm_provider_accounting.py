import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from intent_construction.intent_extraction.core import llm_utils


def _chat_response(
    *,
    content="ok",
    input_tokens=100,
    output_tokens=50,
    cached_tokens=20,
    reasoning_tokens=10,
    finish_reason="stop",
    reasoning_content=None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
        ),
    )


class ProviderConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def test_compatible_client_and_model_map(self):
        os.environ.update(
            {
                "LLM_BACKEND": "compatible",
                "LLM_API_KEY": "not-a-real-key",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_MODEL_MAP": json.dumps({"plan-a": "kimi-k2.5"}),
            }
        )

        with patch.object(llm_utils, "OpenAI") as openai:
            client = llm_utils.get_client()

        self.assertIs(client, openai.return_value)
        openai.assert_called_once_with(
            api_key="not-a-real-key", base_url="https://example.invalid/v1"
        )
        self.assertEqual(llm_utils.resolve_model_name("plan-a"), "kimi-k2.5")

    def test_legacy_backend_precedence_is_unchanged(self):
        os.environ.update(
            {
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_ENDPOINT": "https://azure.invalid",
                "OPENAI_API_KEY": "openai-key",
                "LLM_API_KEY": "compatible-key",
                "LLM_BASE_URL": "https://compatible.invalid/v1",
            }
        )
        self.assertEqual(llm_utils._select_backend(), "azure")

        del os.environ["AZURE_OPENAI_API_KEY"]
        del os.environ["AZURE_OPENAI_ENDPOINT"]
        self.assertEqual(llm_utils._select_backend(), "openai")

    def test_invalid_compatible_model_map_fails_before_client_use(self):
        os.environ.update(
            {
                "LLM_BACKEND": "compatible",
                "LLM_API_KEY": "key",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_MODEL_MAP": "[]",
            }
        )
        with self.assertRaisesRegex(llm_utils.LLMAccountingError, "must map"):
            llm_utils.resolve_model_name("plan-a")

    def test_kimi_k2_chat_payloads_use_provider_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ.update(
                {
                    "LLM_BACKEND": "compatible",
                    "LLM_API_KEY": "key",
                    "LLM_BASE_URL": "https://example.invalid/v1",
                    "LLM_MODEL_MAP": json.dumps({"plan-a": "kimi-k2.6"}),
                    "LLM_USAGE_LEDGER_PATH": str(Path(temp_dir) / "usage.jsonl"),
                    "LLM_COST_HARD_CAP_USD": "10",
                    "LLM_PRICE_MAP": json.dumps(
                        {"plan-a": {"input": 1.0, "output": 1.0}}
                    ),
                }
            )
            create = Mock(return_value=_chat_response(content='{"ok": true}'))
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )

            with patch.object(llm_utils, "get_client", return_value=client):
                llm_utils.generate_json(
                    [{"role": "user", "content": "json"}],
                    model="plan-a",
                    temperature=0.7,
                )
                llm_utils.generate_text(
                    [{"role": "user", "content": "text"}],
                    model="plan-a",
                    temperature=0.7,
                    max_tokens=77,
                )
                llm_utils.generate_multi_turn(
                    [{"role": "user", "content": "multi"}],
                    model="plan-a",
                    temperature=0.7,
                    max_tokens=77,
                )
                llm_utils.generate_with_tools(
                    [{"role": "user", "content": "tool"}],
                    model="plan-a",
                    temperature=0.7,
                    max_tokens=77,
                )

        self.assertEqual(create.call_count, 4)
        for call in create.call_args_list:
            payload = call.kwargs
            self.assertEqual(payload["model"], "kimi-k2.6")
            self.assertNotIn("temperature", payload)
            self.assertNotIn("reasoning_effort", payload)
            self.assertNotIn("max_tokens", payload)
            self.assertIn("max_completion_tokens", payload)
        self.assertEqual(create.call_args_list[0].kwargs["max_completion_tokens"], 4096)
        for call in create.call_args_list[1:]:
            self.assertEqual(call.kwargs["max_completion_tokens"], 77)

    def test_compatible_non_kimi_payload_is_unchanged(self):
        os.environ.update(
            {
                "LLM_BACKEND": "compatible",
                "LLM_API_KEY": "key",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_MODEL_MAP": json.dumps({"plan-a": "glm-4.5"}),
            }
        )
        create = Mock(return_value=_chat_response())
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch.object(llm_utils, "get_client", return_value=client):
            llm_utils.generate_text(
                [{"role": "user", "content": "text"}],
                model="plan-a",
                temperature=0.7,
                max_tokens=77,
            )

        payload = create.call_args.kwargs
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 77)
        self.assertNotIn("max_completion_tokens", payload)

    def test_reasoning_content_is_never_used_as_final_answer(self):
        os.environ.update(
            {
                "LLM_BACKEND": "compatible",
                "LLM_API_KEY": "key",
                "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_MODEL_MAP": json.dumps({"plan-a": "kimi-k2.6"}),
            }
        )
        create = Mock(
            return_value=_chat_response(
                content=None,
                reasoning_content="unfinished reasoning ending in 42",
                finish_reason="stop",
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch.object(llm_utils, "get_client", return_value=client):
            with self.assertRaises(llm_utils.LLMIncompleteResponse):
                llm_utils.generate_text(
                    [{"role": "user", "content": "solve"}],
                    model="plan-a",
                    max_retries=3,
                    max_tokens=8192,
                )

        create.assert_called_once()

    def test_length_truncated_content_is_rejected(self):
        response = _chat_response(content="42", finish_reason="length")
        with self.assertRaises(llm_utils.LLMIncompleteResponse):
            llm_utils._chat_final_content(response)


class UsageAccountingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        os.environ["OPENAI_API_KEY"] = "not-a-real-key"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger = Path(self.temp_dir.name) / "nested" / "usage.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()
        self.environment.stop()

    def _configure_prices(self):
        os.environ["LLM_USAGE_LEDGER_PATH"] = str(self.ledger)
        os.environ["LLM_PRICE_MAP"] = json.dumps(
            {
                "test-model": {
                    "input": 2.0,
                    "output": 4.0,
                    "cached_input": 0.5,
                    "reasoning": 6.0,
                }
            }
        )

    def _read_entries(self):
        return [json.loads(line) for line in self.ledger.read_text().splitlines()]

    def test_chat_usage_is_normalized_costed_and_appended(self):
        self._configure_prices()
        response = _chat_response()

        returned = llm_utils._accounted_api_call(
            lambda **kwargs: response,
            {"model": "test-model", "messages": [{"role": "user", "content": "x"}]},
            requested_model="test-model",
            resolved_model="provider-model",
            api="chat.completions",
        )

        self.assertIs(returned, response)
        entries = self._read_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["input_tokens"], 100)
        self.assertEqual(entry["output_tokens"], 50)
        self.assertEqual(entry["cached_tokens"], 20)
        self.assertEqual(entry["reasoning_tokens"], 10)
        self.assertAlmostEqual(entry["cost_usd"], 0.00039)
        self.assertEqual(entry["requested_model"], "test-model")
        self.assertEqual(entry["resolved_model"], "provider-model")

    def test_default_output_limit_is_independent_of_hard_cap(self):
        self._configure_prices()
        os.environ["LLM_DEFAULT_MAX_OUTPUT_TOKENS"] = "8192"
        create = Mock(return_value=_chat_response())

        llm_utils._accounted_api_call(
            create,
            {"model": "test-model", "messages": []},
            requested_model="test-model",
            resolved_model="test-model",
            api="chat.completions",
        )

        self.assertEqual(create.call_args.kwargs["max_tokens"], 8192)
        self.assertNotIn("reservation_id", self._read_entries()[0])

    def test_ledger_requires_matching_prices_without_hard_cap(self):
        os.environ["LLM_USAGE_LEDGER_PATH"] = str(self.ledger)
        os.environ["LLM_PRICE_MAP"] = json.dumps(
            {"other-model": {"input": 1.0, "output": 1.0}}
        )
        create = Mock(return_value=_chat_response())

        with self.assertRaisesRegex(
            llm_utils.LLMAccountingError, "No LLM_PRICE_MAP entry"
        ):
            llm_utils._accounted_api_call(
                create,
                {"model": "test-model", "messages": []},
                requested_model="test-model",
                resolved_model="test-model",
                api="chat.completions",
            )

        create.assert_not_called()

    def test_responses_usage_shape_is_supported(self):
        usage = {
            "input_tokens": 12,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        }
        normalized = llm_utils._extract_usage({"usage": usage})
        self.assertEqual(
            normalized,
            {
                "input_tokens": 12,
                "output_tokens": 7,
                "cached_tokens": 3,
                "reasoning_tokens": 2,
            },
        )

    def test_hard_cap_blocks_call_during_preflight(self):
        self._configure_prices()
        os.environ["LLM_COST_HARD_CAP_USD"] = "0.000001"
        create = Mock(return_value=_chat_response())

        with self.assertRaises(llm_utils.LLMBudgetExceeded):
            llm_utils._accounted_api_call(
                create,
                {"model": "test-model", "messages": []},
                requested_model="test-model",
                resolved_model="test-model",
                api="chat.completions",
            )

        create.assert_not_called()
        self.assertTrue(self.ledger.exists())
        self.assertEqual(self.ledger.read_text(), "")

    def test_hard_cap_records_then_fails_if_reported_usage_overshoots(self):
        os.environ["LLM_USAGE_LEDGER_PATH"] = str(self.ledger)
        os.environ["LLM_COST_HARD_CAP_USD"] = "0.001"
        os.environ["LLM_PRICE_MAP"] = json.dumps(
            {"test-model": {"input": 1.0, "output": 1.0}}
        )
        response = _chat_response(
            input_tokens=2_000,
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
        )

        with self.assertRaises(llm_utils.LLMBudgetExceeded):
            llm_utils._accounted_api_call(
                lambda **kwargs: response,
                {"model": "test-model", "messages": []},
                requested_model="test-model",
                resolved_model="test-model",
                api="chat.completions",
                max_output_tokens=0,
            )

        entries = self._read_entries()
        self.assertEqual([entry["event"] for entry in entries], ["reservation", "usage"])
        self.assertEqual(entries[1]["reservation_id"], entries[0]["reservation_id"])
        self.assertAlmostEqual(entries[1]["cost_usd"], 0.002)
        self.assertAlmostEqual(entries[1]["cumulative_cost_usd"], 0.002)

    def test_cap_fails_closed_when_provider_omits_usage(self):
        self._configure_prices()
        os.environ["LLM_COST_HARD_CAP_USD"] = "1"
        response = SimpleNamespace(choices=[])

        with self.assertRaisesRegex(
            llm_utils.LLMAccountingError, "no usable token accounting"
        ):
            llm_utils._accounted_api_call(
                lambda **kwargs: response,
                {"model": "test-model", "messages": []},
                requested_model="test-model",
                resolved_model="test-model",
                api="chat.completions",
            )
        entries = self._read_entries()
        self.assertEqual([entry["event"] for entry in entries], ["reservation"])
        with self.ledger.open("a+", encoding="utf-8") as handle:
            _, active = llm_utils._read_ledger_state(handle, strict=True)
        self.assertEqual(set(active), {entries[0]["reservation_id"]})

    def test_public_helper_does_not_retry_accounting_failures(self):
        self._configure_prices()
        os.environ["LLM_COST_HARD_CAP_USD"] = "1"
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        create = Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch.object(llm_utils, "get_client", return_value=client):
            with self.assertRaises(llm_utils.LLMAccountingError):
                llm_utils.generate_text(
                    [{"role": "user", "content": "hello"}],
                    model="test-model",
                    max_retries=3,
                )

        self.assertEqual(create.call_count, 1)
        self.assertEqual(create.call_args.kwargs["max_tokens"], 4096)

    def test_cap_requires_ledger_and_model_prices_before_call(self):
        os.environ["LLM_COST_HARD_CAP_USD"] = "1"
        create = Mock(return_value=_chat_response())
        with self.assertRaisesRegex(
            llm_utils.LLMAccountingError, "requires LLM_USAGE_LEDGER_PATH"
        ):
            llm_utils._accounted_api_call(
                create,
                {"model": "unknown", "messages": []},
                requested_model="unknown",
                resolved_model="unknown",
                api="chat.completions",
            )
        create.assert_not_called()

    def test_persisted_reservation_blocks_another_worker_until_usage(self):
        os.environ["LLM_USAGE_LEDGER_PATH"] = str(self.ledger)
        os.environ["LLM_COST_HARD_CAP_USD"] = "0.0008"
        os.environ["LLM_PRICE_MAP"] = json.dumps(
            {"test-model": {"input": 1.0, "output": 1.0}}
        )
        payload = {"model": "test-model", "messages": []}

        first = llm_utils._reserve_budget(
            "test-model", "test-model", payload, 0, api="chat.completions"
        )
        with self.assertRaises(llm_utils.LLMBudgetExceeded):
            llm_utils._reserve_budget(
                "test-model", "test-model", payload, 0, api="chat.completions"
            )

        llm_utils._record_response_usage(
            _chat_response(
                input_tokens=10,
                output_tokens=0,
                cached_tokens=0,
                reasoning_tokens=0,
            ),
            requested_model="test-model",
            resolved_model="test-model",
            api="chat.completions",
            reservation=first,
        )
        second = llm_utils._reserve_budget(
            "test-model", "test-model", payload, 0, api="chat.completions"
        )
        self.assertIsNotNone(second)

        events = self._read_entries()
        self.assertEqual(
            [entry["event"] for entry in events],
            ["reservation", "usage", "reservation"],
        )
        self.assertEqual(events[1]["reservation_id"], events[0]["reservation_id"])

    def test_persisted_release_unblocks_after_api_error(self):
        os.environ["LLM_USAGE_LEDGER_PATH"] = str(self.ledger)
        os.environ["LLM_COST_HARD_CAP_USD"] = "0.0008"
        os.environ["LLM_PRICE_MAP"] = json.dumps(
            {"test-model": {"input": 1.0, "output": 1.0}}
        )
        payload = {"model": "test-model", "messages": []}

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            llm_utils._accounted_api_call(
                Mock(side_effect=RuntimeError("provider failed")),
                payload,
                requested_model="test-model",
                resolved_model="test-model",
                api="chat.completions",
                max_output_tokens=0,
            )

        replacement = llm_utils._reserve_budget(
            "test-model", "test-model", payload, 0, api="chat.completions"
        )
        self.assertIsNotNone(replacement)
        events = self._read_entries()
        self.assertEqual(
            [entry["event"] for entry in events],
            ["reservation", "release", "reservation"],
        )
        self.assertEqual(events[1]["reservation_id"], events[0]["reservation_id"])


if __name__ == "__main__":
    unittest.main()
