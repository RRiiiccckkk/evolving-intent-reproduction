from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluation.runners import run_browsecomp_experiment as runner
from intent_construction.intent_extraction.core import llm_utils
from reproduction import browsecomp_plan_a as plan_a
from reproduction import browsecomp_construction_modal as construction_modal
from reproduction import browsecomp_modal
from intent_construction.retrospective_expansion.predecessor import (
    bm25_retriever,
    generate_predecessors as predecessor_generation,
)
from situated_simulation import naturalizer


def _encrypt_string(value: str, password: str) -> str:
    raw = value.encode("utf-8")
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    key = digest * (len(raw) // len(digest)) + digest[: len(raw) % len(digest)]
    return base64.b64encode(bytes(left ^ right for left, right in zip(raw, key))).decode()


def test_retriever_health_payload_exposes_pinned_revision():
    assert browsecomp_modal.health_payload() == {
        "status": "ok",
        "retriever_revision": browsecomp_modal.RETRIEVER_REVISION,
    }
    assert browsecomp_modal.ready_payload() == {
        "status": "ready",
        "retriever_revision": browsecomp_modal.RETRIEVER_REVISION,
        "model_revision": browsecomp_modal.MODEL_REVISION,
        "corpus_revision": browsecomp_modal.CORPUS_REVISION,
        "index_revision": browsecomp_modal.INDEX_REVISION,
    }


def test_bm25_retriever_pins_corpus_revision(monkeypatch):
    observed = {}

    def load_corpus(name, revision):
        observed.update(name=name, revision=revision)
        return [{"docid": "1", "text": "one document"}]

    monkeypatch.setattr(bm25_retriever, "_load_corpus", load_corpus)
    monkeypatch.setattr(bm25_retriever, "_build_index", lambda _tokens: object())
    bm25_retriever.BM25Retriever()

    assert observed == {
        "name": bm25_retriever.DEFAULT_CORPUS_DATASET,
        "revision": bm25_retriever.DEFAULT_CORPUS_REVISION,
    }


def test_stage_source_bundle_covers_loaded_prompts_and_bm25():
    stage1 = {path.name for path in plan_a._construction_source_files("stage1")}
    stage2 = {path.name for path in plan_a._construction_source_files("stage2")}
    stage3 = {path.name for path in plan_a._construction_source_files("stage3")}

    assert {"segmentation.txt", "conversational.txt", "verification.txt", "llm_judge.txt"} <= stage1
    assert "generate_counterfactual_search.txt" in stage2
    assert {
        "bm25_retriever.py",
        "generate_predecessor_browsecomp.txt",
        "similarity_check_default.txt",
        "cross_turn_relevance_check.txt",
    } <= stage3
    assert plan_a._stage_policy("stage3")["bm25_corpus_revision"] == plan_a.BM25_CORPUS_REVISION


def test_stage_completion_requires_full_counterfactual_and_verified_predecessors():
    stage2 = {
        "task_id": "one",
        "arguments": [
            {
                "counterfactual_arguments": [
                    {"counterfactual_argument": f"candidate-{index}"}
                    for index in range(4)
                ]
            }
        ],
        "counterfactual_info": {
            "num_counterfactuals_requested": 4,
            "total_arguments": 1,
            "successful_counterfactuals": 4,
        },
    }
    assert plan_a._stage_result_is_complete("stage2", stage2)
    stage2["arguments"][0]["counterfactual_arguments"].pop()
    assert not plan_a._stage_result_is_complete("stage2", stage2)

    stage3 = {
        "task_id": "one",
        "predecessors": [{}, {}, {}],
        "predecessor_functions": [
            {"predecessor_function": f"function-{index}"}
            for index in range(3)
        ],
        "verification_passed": True,
        "independence_passed": True,
    }
    assert plan_a._stage_result_is_complete("stage3", stage3)
    stage3["independence_passed"] = False
    assert not plan_a._stage_result_is_complete("stage3", stage3)


def test_predecessor_retry_includes_rejection_feedback(monkeypatch):
    too_long = " ".join(f"word{index}" for index in range(36))
    responses = [
        {
            "predecessor_function": too_long,
            "entity_sought": "book title",
            "relevant_argument_ids": [1],
            "new_arguments": ["A separate clue."],
            "transition_reason": "The answer motivates the next question.",
            "transition_type": "identify_then_seek",
        },
        {
            "predecessor_function": "Which book matches this separate clue?",
            "entity_sought": "book title",
            "relevant_argument_ids": [1],
            "new_arguments": ["A separate clue."],
            "transition_reason": "The answer motivates the next question.",
            "transition_type": "identify_then_seek",
        },
    ]
    calls = []

    def fake_generate_json(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return responses.pop(0)

    monkeypatch.setattr(predecessor_generation, "generate_json", fake_generate_json)
    generator = predecessor_generation.PredecessorGenerator(
        model=plan_a.PLAN_A_MODEL,
        prompts_dir=str(plan_a.PREDECESSOR_PROMPTS),
        dataset_type="browsecomp",
        max_attempts=2,
        verify_independence=False,
        judge_model=plan_a.PLAN_A_MODEL,
    )

    result = generator._generate_single_predecessor(
        next_function="Which person matches the final clues?",
        next_arguments=[{"argument_id": 1, "argument": "The work appeared in print."}],
        existing_predecessors=[],
        all_functions_in_chain=[],
        future_functions=[],
        chain_entity_types=[],
        chain_type="identify_then_seek",
    )

    assert result["predecessor_function"] == "Which book matches this separate clue?"
    assert len(calls) == 2
    assert [message["role"] for message in calls[1]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    feedback = calls[1]["messages"][-1]["content"]
    assert "36 words" in feedback
    assert "MAX 30 words" in feedback
    assert calls[1]["model"] == plan_a.PLAN_A_MODEL


def test_predecessor_content_filter_restarts_whole_chain(monkeypatch):
    generator = predecessor_generation.PredecessorGenerator(
        model=plan_a.PLAN_A_MODEL,
        prompts_dir=str(plan_a.PREDECESSOR_PROMPTS),
        dataset_type="browsecomp",
        num_predecessors=1,
        max_verify_attempts=2,
        verify_independence=False,
        judge_model=plan_a.PLAN_A_MODEL,
    )
    valid_predecessor = {
        "predecessor_function": "Which book matches the earlier clue?",
        "entity_sought": "book title",
        "full_arguments": [
            {
                "argument_id": 101,
                "argument": "A separate clue.",
                "is_shared": False,
            }
        ],
        "shared_arguments": [],
        "shared_argument_ids": [],
        "new_arguments": ["A separate clue."],
        "new_argument_ids": [101],
        "transition_reason": "The answer motivates the next question.",
        "transition_type": "identify_then_seek",
        "taxonomy_type": "T1",
        "causal_link": "The book identifies the person.",
        "reasoning": "The questions form a causal chain.",
    }
    attempts = []

    def fake_generate_chain(**_kwargs):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise predecessor_generation.PredecessorContentFilterError("filtered")
        return [dict(valid_predecessor)]

    monkeypatch.setattr(generator, "_generate_chain", fake_generate_chain)
    monkeypatch.setattr(generator, "_verify_chain", lambda **_kwargs: None)

    result = generator.generate_predecessors(
        {
            "task_id": "task-1",
            "function": "Who matches the final clues?",
            "arguments": [{"argument_id": 1, "argument": "A final clue."}],
            "answer": "Answer",
        }
    )

    assert attempts == [1, 2]
    assert result["predecessor_functions"][0]["predecessor_function"] == (
        "Which book matches the earlier clue?"
    )


def test_predecessor_content_filter_is_not_retried_with_identical_prompt(monkeypatch):
    calls = []

    def rejected_generate_json(_messages, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("content_filter: prompt considered high risk")

    monkeypatch.setattr(
        predecessor_generation,
        "generate_json",
        rejected_generate_json,
    )
    generator = predecessor_generation.PredecessorGenerator(
        model=plan_a.PLAN_A_MODEL,
        prompts_dir=str(plan_a.PREDECESSOR_PROMPTS),
        dataset_type="browsecomp",
        max_attempts=5,
        verify_independence=False,
        judge_model=plan_a.PLAN_A_MODEL,
    )

    with pytest.raises(predecessor_generation.PredecessorContentFilterError):
        generator._generate_single_predecessor(
            next_function="Which person matches the final clues?",
            next_arguments=[{"argument_id": 1, "argument": "A final clue."}],
            existing_predecessors=[],
            all_functions_in_chain=[],
            future_functions=[],
            chain_entity_types=[],
            chain_type="identify_then_seek",
        )

    assert len(calls) == 1
    assert calls[0]["max_retries"] == 1


def test_browsecomp_independence_uses_structured_answers_without_output_limit(
    monkeypatch,
):
    generator = predecessor_generation.PredecessorGenerator.__new__(
        predecessor_generation.PredecessorGenerator
    )
    generator._bm25_retriever = SimpleNamespace(
        search=lambda _query: json.dumps([{"snippet": "Reference text."}])
    )
    generator.independence_runs = 1
    generator.model = plan_a.PLAN_A_MODEL
    generator.judge_model = plan_a.PLAN_A_MODEL
    generator.temperature = 1.0
    generator.reasoning_effort = None
    responses = [
        llm_utils.LLMIncompleteResponse("truncated"),
        {"answer": "The answer"},
        {"answer": "The answer"},
    ]
    calls = []

    def fake_generate_json(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(predecessor_generation, "generate_json", fake_generate_json)

    passed, reason, feedback = generator._verify_functional_independence_browsecomp(
        function="Which work matches the clues?",
        arguments=[{"argument_id": 1, "argument": "The work was published."}],
        all_new_arguments=[{"argument": "An unrelated clue."}],
        ground_truth="The answer",
    )

    assert passed is True
    assert reason == "Functional independence passed (1/1)"
    assert feedback is None
    assert len(calls) == 3
    assert all(call["model"] == plan_a.PLAN_A_MODEL for call in calls)
    assert all(call["max_retries"] == 1 for call in calls)
    assert all(call["step"] == "functional-independence-browsecomp-answer" for call in calls)
    assert all("max_tokens" not in call for call in calls)


def test_browsecomp_gets_more_in_process_verification_attempts():
    browsecomp = predecessor_generation.PredecessorGenerator(
        model=plan_a.PLAN_A_MODEL,
        prompts_dir=str(plan_a.PREDECESSOR_PROMPTS),
        dataset_type="browsecomp",
        verify_independence=False,
    )
    other = predecessor_generation.PredecessorGenerator(
        model=plan_a.PLAN_A_MODEL,
        prompts_dir=str(plan_a.PREDECESSOR_PROMPTS),
        dataset_type="math",
        verify_independence=False,
    )
    explicit = predecessor_generation.PredecessorGenerator(
        model=plan_a.PLAN_A_MODEL,
        prompts_dir=str(plan_a.PREDECESSOR_PROMPTS),
        dataset_type="browsecomp",
        verify_independence=False,
        max_verify_attempts=2,
    )

    assert browsecomp.max_verify_attempts == 5
    assert other.max_verify_attempts == 2
    assert explicit.max_verify_attempts == 2


class _RangeHandler(BaseHTTPRequestHandler):
    file_path: Path
    ranges: list[str]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        range_header = self.headers.get("Range", "")
        self.ranges.append(range_header)
        _, bounds = range_header.split("=", 1)
        start_text, end_text = bounds.split("-", 1)
        start, end = int(start_text), int(end_text)
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Length", str(length))
        self.send_header(
            "Content-Range", f"bytes {start}-{end}/{self.file_path.stat().st_size}"
        )
        self.end_headers()
        with self.file_path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(length))

    def log_message(self, *_args) -> None:
        return


def test_remote_parquet_reads_only_projected_columns_with_ranges(tmp_path):
    parquet_path = tmp_path / "source.parquet"
    evidence = [os.urandom(512 * 1024) for _ in range(3)]
    pq.write_table(
        pa.table(
            {
                "query_id": ["1", "2", "3"],
                "query": ["q1", "q2", "q3"],
                "answer": ["a1", "a2", "a3"],
                "evidence_docs": evidence,
            }
        ),
        parquet_path,
        compression="NONE",
    )

    handler = type("RangeHandler", (_RangeHandler,), {})
    handler.file_path = parquet_path
    handler.ranges = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        shard = plan_a.RemoteParquetShard(
            "source.parquet",
            parquet_path.stat().st_size,
            f"http://127.0.0.1:{server.server_port}/source.parquet",
        )
        metrics: dict[str, int] = {}
        rows = list(plan_a._iter_remote_parquet_rows([shard], metrics))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert rows == [
        {"query_id": "1", "query": "q1", "answer": "a1"},
        {"query_id": "2", "query": "q2", "answer": "a2"},
        {"query_id": "3", "query": "q3", "answer": "a3"},
    ]
    assert handler.ranges and all(value.startswith("bytes=") for value in handler.ranges)
    assert metrics["range_bytes"] < parquet_path.stat().st_size // 4


def test_prepare_selects_and_decrypts_fixed_ids_in_manifest_order(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "raw_data.jsonl"
    samples = [
        {"query_id": str(index), "task_id": f"task-{index}"}
        for index in range(plan_a.PLAN_A_SAMPLE_COUNT)
    ]
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    source_rows = []
    for index in reversed(range(plan_a.PLAN_A_SAMPLE_COUNT)):
        source_rows.append(
            {
                "query_id": str(index),
                "query": _encrypt_string(f"question {index}", plan_a.DEFAULT_CANARY),
                "answer": _encrypt_string(f"answer {index}", plan_a.DEFAULT_CANARY),
                "evidence_docs": "deliberately-not-encrypted-or-read",
            }
        )
    source_path.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )

    report = plan_a.prepare_selected_data(
        manifest_path, output_path, source_path, plan_a.DEFAULT_CANARY
    )
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert report["selected_rows"] == plan_a.PLAN_A_SAMPLE_COUNT
    assert [row["query_id"] for row in rows] == [str(index) for index in range(100)]
    assert rows[0] == {"query_id": "0", "query": "question 0", "answer": "answer 0"}
    assert all(set(row) == set(plan_a.REMOTE_SOURCE_COLUMNS) for row in rows)


def test_private_prepare_keeps_gold_docs_and_omits_evidence(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "private.jsonl"
    samples = [
        {"query_id": str(index), "task_id": f"task-{index}"}
        for index in range(plan_a.PLAN_A_SAMPLE_COUNT)
    ]
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    rows = []
    for index in range(plan_a.PLAN_A_SAMPLE_COUNT):
        rows.append(
            {
                "query_id": str(index),
                "query": _encrypt_string(f"question {index}", plan_a.DEFAULT_CANARY),
                "answer": _encrypt_string(f"answer {index}", plan_a.DEFAULT_CANARY),
                "gold_docs": [
                    {
                        "url": _encrypt_string("https://example.test", plan_a.DEFAULT_CANARY),
                        "text": _encrypt_string(f"evidence {index}", plan_a.DEFAULT_CANARY),
                    }
                ],
                "evidence_docs": "must-not-be-read",
            }
        )
    source_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = plan_a.prepare_selected_data(
        manifest_path,
        output_path,
        source_path,
        source_columns=plan_a.REMOTE_PRIVATE_SOURCE_COLUMNS,
    )
    prepared = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert report["source_columns"] == list(plan_a.REMOTE_PRIVATE_SOURCE_COLUMNS)
    assert prepared[0]["gold_docs"][0]["text"] == "evidence 0"
    assert all("evidence_docs" not in row for row in prepared)
    assert plan_a.validate_prepared_data(
        output_path, manifest_path, require_gold_docs=True
    )["sample_count"] == plan_a.PLAN_A_SAMPLE_COUNT


def test_canary_manifest_is_one_fixed_task_and_uses_separate_run_name():
    manifest_json = plan_a.DEFAULT_MANIFEST.read_text(encoding="utf-8")
    canary = json.loads(
        construction_modal._single_sample_manifest(
            manifest_json, construction_modal.DEFAULT_CANARY_TASK_ID
        )
    )
    assert canary["num_samples"] == 1
    assert canary["samples"][0]["task_id"] == construction_modal.DEFAULT_CANARY_TASK_ID
    assert construction_modal.CANARY_VOLUME_NAME != construction_modal.VOLUME_NAME
    assert construction_modal.CANARY_RUN_ROOT != construction_modal.VOLUME_RUN_ROOT
    with pytest.raises(plan_a.WorkflowError, match="not in the fixed manifest"):
        construction_modal._single_sample_manifest(manifest_json, "not-published")


def test_modal_environment_locks_medium_reasoning(monkeypatch, tmp_path):
    monkeypatch.setattr(
        llm_utils,
        "_CLIENT_TIMEOUT",
        llm_utils.httpx.Timeout(connect=30.0, read=None, write=600.0, pool=600.0),
    )
    monkeypatch.setenv("LLM_BACKEND", "")
    monkeypatch.setenv("LLM_API_KEY", "offline")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_PRICE_MAP", "")
    for name in (
        "LLM_REASONING_EFFORT",
        "LLM_LOCKED_MODEL",
        "LLM_DISABLE_OUTPUT_LIMITS",
        "LLM_REQUIRE_USAGE_ACCOUNTING",
        "LLM_USAGE_LEDGER_PATH",
        "HF_HOME",
        "HF_DATASETS_CACHE",
    ):
        monkeypatch.setenv(name, "")
    construction_modal._validate_secret_environment(
        root=tmp_path, ledger=tmp_path / "usage.jsonl"
    )
    assert os.environ["LLM_BACKEND"] == "compatible"
    assert os.environ["LLM_REASONING_EFFORT"] == "medium"
    assert json.loads(os.environ["LLM_PRICE_MAP"]) == construction_modal.PLAN_A_PRICE_MAP
    assert llm_utils._CLIENT_TIMEOUT.read == 600.0

    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    with pytest.raises(plan_a.WorkflowError, match="requires LLM_REASONING_EFFORT=medium"):
        construction_modal._validate_secret_environment(
            root=tmp_path, ledger=tmp_path / "usage.jsonl"
        )

    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")
    monkeypatch.setenv("LLM_BACKEND", "openai")
    with pytest.raises(plan_a.WorkflowError, match="LLM_BACKEND=compatible"):
        construction_modal._validate_secret_environment(
            root=tmp_path, ledger=tmp_path / "usage.jsonl"
        )


def test_modal_stage3_retries_incomplete_response(monkeypatch):
    calls = []
    sleeps = []

    def processor(item):
        calls.append(item["task_id"])
        if len(calls) < 3:
            raise llm_utils.LLMIncompleteResponse("truncated")
        return {
            "task_id": item["task_id"],
            "predecessors": [{}, {}, {}],
            "predecessor_functions": [
                {"predecessor_function": f"question-{index}"}
                for index in range(3)
            ],
            "verification_passed": True,
            "independence_passed": True,
        }

    monkeypatch.setattr(construction_modal.time, "sleep", sleeps.append)
    resilient = construction_modal._retry_incomplete_stage3_processor(processor)

    assert resilient({"task_id": "sample-1"})["task_id"] == "sample-1"
    assert calls == ["sample-1", "sample-1", "sample-1"]
    assert sleeps == [5, 10]

    accounting_calls = []

    def accounting_failure(item):
        accounting_calls.append(item["task_id"])
        raise llm_utils.LLMAccountingError("missing usage")

    resilient = construction_modal._retry_incomplete_stage3_processor(
        accounting_failure
    )
    with pytest.raises(llm_utils.LLMAccountingError, match="missing usage"):
        resilient({"task_id": "sample-2"})
    assert accounting_calls == ["sample-2"]


def test_modal_stage3_regenerates_returned_rejected_sample(monkeypatch):
    calls = []
    sleeps = []
    rejected = {
        "task_id": "sample-1",
        "predecessors": [{}, {}, {}],
        "predecessor_functions": [
            {"predecessor_function": f"question-{index}"}
            for index in range(3)
        ],
        "verification_passed": True,
        "independence_passed": False,
    }
    complete = {**rejected, "independence_passed": True}
    responses = [None, rejected, complete]

    def processor(item):
        calls.append(item["task_id"])
        return responses.pop(0)

    monkeypatch.setattr(construction_modal.time, "sleep", sleeps.append)
    resilient = construction_modal._retry_incomplete_stage3_processor(processor)

    assert resilient({"task_id": "sample-1"}) == complete
    assert calls == ["sample-1", "sample-1", "sample-1"]
    assert sleeps == [5, 10]


def test_modal_stage3_rejected_sample_retry_is_bounded(monkeypatch):
    calls = []
    sleeps = []
    rejected = {
        "task_id": "sample-1",
        "predecessors": [{}, {}, {}],
        "predecessor_functions": [
            {"predecessor_function": f"question-{index}"}
            for index in range(3)
        ],
        "verification_passed": False,
        "independence_passed": True,
    }

    def processor(item):
        calls.append(item["task_id"])
        return rejected

    monkeypatch.setattr(construction_modal, "STAGE3_RECOVERABLE_SAMPLE_ATTEMPTS", 3)
    monkeypatch.setattr(construction_modal.time, "sleep", sleeps.append)
    resilient = construction_modal._retry_incomplete_stage3_processor(processor)

    assert resilient({"task_id": "sample-1"}) is rejected
    assert calls == ["sample-1", "sample-1", "sample-1"]
    assert sleeps == [5, 10]


@pytest.mark.parametrize(
    "result",
    [
        {"task_id": "different-sample"},
        {"task_id": "sample-1", "success": False},
        {"task_id": "sample-1", "error": "policy failure"},
    ],
)
def test_modal_stage3_does_not_retry_returned_systemic_failure(
    monkeypatch, result
):
    calls = []

    def processor(item):
        calls.append(item["task_id"])
        return result

    monkeypatch.setattr(
        construction_modal.time,
        "sleep",
        lambda _delay: pytest.fail("systemic failure must not be retried"),
    )
    resilient = construction_modal._retry_incomplete_stage3_processor(processor)

    assert resilient({"task_id": "sample-1"}) is result
    assert calls == ["sample-1"]

def test_compatible_kimi_normalization_omits_tool_choice(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "compatible")
    payload = llm_utils._normalize_chat_payload(
        {
            "model": plan_a.PLAN_A_MODEL,
            "messages": [],
            "tools": [runner.SEARCH_TOOL_DEF],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning_effort": "medium",
        },
        plan_a.PLAN_A_MODEL,
    )
    assert "tool_choice" not in payload
    assert payload["parallel_tool_calls"] is False


class _ToolMessage:
    def __init__(self, count: int):
        self.content = None
        self.tool_calls = [
            SimpleNamespace(
                id=f"call-{index}",
                function=SimpleNamespace(
                    name="search", arguments=json.dumps({"query": f"q-{index}"})
                ),
            )
            for index in range(count)
        ]

    def model_dump(self):
        return {"role": "assistant", "content": None, "tool_calls": []}


def _chat_response(message, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=None
    )


def _patch_chat_client(monkeypatch, responses):
    payloads = []

    def create(**payload):
        payloads.append(payload)
        return responses.pop(0)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(runner, "resolve_model_name", lambda model: model)
    monkeypatch.setattr(runner, "_supports_responses_api", lambda _model: False)
    monkeypatch.setattr(runner, "get_client", lambda **_kwargs: client)
    monkeypatch.setattr(
        runner,
        "_accounted_api_call",
        lambda create_fn, payload, **_kwargs: create_fn(**payload),
    )
    return payloads


def test_parallel_tool_batch_cannot_exceed_fifty_and_forced_final_has_no_limit(
    monkeypatch,
):
    final_message = SimpleNamespace(content="Exact Answer: done", tool_calls=[])
    responses = [_chat_response(_ToolMessage(55)), _chat_response(final_message)]
    payloads = _patch_chat_client(monkeypatch, responses)
    retriever = SimpleNamespace(
        calls=[],
        search=lambda query: retriever.calls.append(query)
        or json.dumps([{"docid": query, "score": 1, "snippet": "x"}]),
    )

    result = runner.run_agentic_search(
        [{"role": "user", "content": "question"}],
        runner.PLAN_A_MODEL,
        retriever,
        max_iterations=51,
        max_tool_calls=50,
        reasoning_effort="medium",
        force_final_answer=True,
    )

    assert result["tool_call_count"] == 50
    assert len(retriever.calls) == 50
    assert all("tool_choice" not in payload for payload in payloads)
    assert payloads[0]["parallel_tool_calls"] is False
    assert "parallel_tool_calls" not in payloads[-1]
    assert payloads[-1]["messages"][-1] == {
        "role": "user",
        "content": runner.FORCE_FINAL_ANSWER_NUDGE,
    }
    assert all(
        item.get("content") != runner.FORCE_FINAL_ANSWER_NUDGE
        for item in result["new_messages"]
    )
    for payload in payloads:
        assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & payload.keys()


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[], usage=None),
        _chat_response(SimpleNamespace(content="partial", tool_calls=[]), "length"),
        _chat_response(SimpleNamespace(content="", tool_calls=[]), "stop"),
    ],
)
def test_agent_empty_or_truncated_response_fails_immediately(monkeypatch, response):
    responses = [response]
    payloads = _patch_chat_client(monkeypatch, responses)
    with pytest.raises(llm_utils.LLMIncompleteResponse):
        runner.run_agentic_search(
            [{"role": "user", "content": "question"}],
            runner.PLAN_A_MODEL,
            SimpleNamespace(search=lambda _query: "[]"),
            max_iterations=1,
            max_retries=3,
        )
    assert len(payloads) == 1


def test_judge_and_naturalizer_send_no_output_limit(monkeypatch):
    judge_message = SimpleNamespace(
        content="extracted_final_answer: x\nreasoning: match\ncorrect: yes",
        tool_calls=[],
    )
    payloads = _patch_chat_client(monkeypatch, [_chat_response(judge_message)])
    monkeypatch.setattr(runner, "_is_responses_api_model", lambda _model: False)
    verdict = runner.llm_judge("q", "x", "x", judge_model=runner.PLAN_A_MODEL)
    assert verdict["correct"] is True
    assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & payloads[0].keys()

    naturalizer_kwargs = {}

    def fake_generate_text(**kwargs):
        naturalizer_kwargs.update(kwargs)
        return "natural wording"

    monkeypatch.setattr(llm_utils, "generate_text", fake_generate_text)
    assert (
        naturalizer._call_llm("prompt", runner.PLAN_A_MODEL, None, "fallback")
        == "natural wording"
    )
    assert naturalizer_kwargs["max_tokens"] is None


@pytest.mark.parametrize(
    "error",
    [llm_utils.LLMAccountingError("ledger failed"), llm_utils.LLMIncompleteResponse("empty")],
)
def test_naturalizer_does_not_hide_accounting_or_incomplete_errors(monkeypatch, error):
    monkeypatch.setattr(llm_utils, "generate_text", lambda **_kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(type(error), match=str(error)):
        naturalizer._call_llm("prompt", runner.PLAN_A_MODEL, None, "fallback")


def _complete_result(task_id: str, answer: str = "x") -> dict:
    return {
        "task_id": task_id,
        "prediction": answer,
        "correct": True,
        "success": True,
        "error": None,
        "responses": [{"response": f"Exact Answer: {answer}"}],
        "total_tool_calls": 0,
        "retrieved_docids": [],
        "user_messages": [],
        "metadata": {},
    }


def test_partial_aggregate_resumes_only_incomplete_task(monkeypatch, tmp_path):
    samples = [
        SimpleNamespace(
            task_id=task_id,
            label="x",
            metadata={},
            turns=[{"role": "user", "content": "q"}],
        )
        for task_id in ("done", "retry")
    ]

    class FakeIntent:
        def __init__(self, **_kwargs):
            pass

        def __iter__(self):
            return iter(samples)

    monkeypatch.setattr(runner, "EvolvingIntent", FakeIntent)
    monkeypatch.setattr(runner, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "resolve_model_name", lambda model: model)
    output = tmp_path / "fully_specified" / "unit" / "kimi-k2.6_force-final.json"
    output.parent.mkdir(parents=True)
    runner.atomic_write_json(
        output,
        {
            "done": _complete_result("done"),
            "retry": {
                "task_id": "retry",
                "success": False,
                "prediction": None,
                "responses": [],
            },
        },
    )
    evaluated = []

    def fake_evaluate(sample, *_args, **_kwargs):
        evaluated.append(sample.task_id)
        return _complete_result(sample.task_id)

    monkeypatch.setattr(runner, "evaluate_sample", fake_evaluate)
    summary = runner.run_experiment(
        data_path="unused.json",
        dataset_name="unit",
        model=runner.PLAN_A_MODEL,
        retriever=SimpleNamespace(),
        num_turns=1,
        num_samples=2,
        expected_samples=2,
        force_final_answer=True,
        fail_fast=True,
    )

    assert summary["total_samples"] == 2
    assert evaluated == ["retry"]
    saved = json.loads(output.read_text())
    assert all(runner._result_is_complete(value, 1) for value in saved.values())
    assert len(list(Path(f"{output}.checkpoints").glob("*.json"))) == 1


def test_parallel_fail_fast_keeps_inflight_success_checkpoint(monkeypatch, tmp_path):
    samples = [
        SimpleNamespace(
            task_id=task_id,
            label="x",
            metadata={},
            turns=[{"role": "user", "content": "q"}],
        )
        for task_id in ("fails", "succeeds")
    ]

    class FakeIntent:
        def __init__(self, **_kwargs):
            pass

        def __iter__(self):
            return iter(samples)

    barrier = threading.Barrier(2)

    def fake_evaluate(sample, *_args, **_kwargs):
        barrier.wait(timeout=5)
        if sample.task_id == "fails":
            raise llm_utils.LLMIncompleteResponse("empty provider response")
        return _complete_result(sample.task_id)

    monkeypatch.setattr(runner, "EvolvingIntent", FakeIntent)
    monkeypatch.setattr(runner, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "resolve_model_name", lambda model: model)
    monkeypatch.setattr(runner, "evaluate_sample", fake_evaluate)
    with pytest.raises(llm_utils.LLMIncompleteResponse, match="empty provider response"):
        runner.run_experiment(
            data_path="unused.json",
            dataset_name="parallel",
            model=runner.PLAN_A_MODEL,
            retriever=SimpleNamespace(),
            num_turns=1,
            num_samples=2,
            expected_samples=2,
            num_workers=2,
            force_final_answer=True,
            fail_fast=True,
        )

    output = tmp_path / "fully_specified" / "parallel" / "kimi-k2.6_force-final.json"
    saved = json.loads(output.read_text())
    assert saved["fails"]["success"] is False
    assert runner._result_is_complete(saved["succeeds"], 1)
    assert len(list(Path(f"{output}.checkpoints").glob("*.json"))) == 2


def test_construction_stage_resumes_only_incomplete_checkpoint(tmp_path):
    output = tmp_path / "stage1.json"
    inputs = [{"task_id": "done"}, {"task_id": "retry"}]
    done = {"task_id": "done", "function": "f", "arguments": [{"argument": "a"}]}
    runner.atomic_write_json(
        output,
        [
            done,
            {"task_id": "retry", "function": "", "arguments": []},
        ],
    )
    checkpoint_dir = tmp_path / "stage1.checkpoints"
    runner.atomic_write_json(
        plan_a._checkpoint_path(checkpoint_dir, "done"),
        plan_a._checkpoint_envelope("stage1", inputs[0], done),
    )
    processed = []

    def processor(item):
        processed.append(item["task_id"])
        return {
            "task_id": item["task_id"],
            "function": "f",
            "arguments": [{"argument": "a"}],
        }

    results = plan_a._run_stage("stage1", inputs, output, processor, workers=1)
    assert processed == ["retry"]
    assert [item["task_id"] for item in results] == ["done", "retry"]
    checkpoints = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoints) == 2
    assert not list(tmp_path.rglob("*.tmp"))


def test_construction_checkpoint_rejects_changed_input(tmp_path):
    output = tmp_path / "stage1.json"
    checkpoint_dir = tmp_path / "stage1.checkpoints"
    original = {"task_id": "same", "question": "old"}
    result = {
        "task_id": "same",
        "function": "f",
        "arguments": [{"argument": "a"}],
    }
    runner.atomic_write_json(output, [result])
    runner.atomic_write_json(
        plan_a._checkpoint_path(checkpoint_dir, "same"),
        plan_a._checkpoint_envelope("stage1", original, result),
    )

    with pytest.raises(plan_a.WorkflowError, match="input/policy"):
        plan_a._run_stage(
            "stage1",
            [{"task_id": "same", "question": "changed"}],
            output,
            lambda _item: pytest.fail("stale checkpoint must fail before processing"),
            workers=1,
        )


def test_construction_checkpoint_hook_runs_after_atomic_persist(tmp_path):
    output = tmp_path / "stage1.json"
    checkpoints_seen = []

    def hook():
        checkpoints_seen.append(len(list((tmp_path / "stage1.checkpoints").glob("*.json"))))

    plan_a._run_stage(
        "stage1",
        [{"task_id": "one"}, {"task_id": "two"}],
        output,
        lambda item: {
            "task_id": item["task_id"],
            "function": "f",
            "arguments": [{"argument": "a"}],
        },
        workers=1,
        checkpoint_hook=hook,
    )
    assert checkpoints_seen == [1, 2]


def test_modal_download_boundary_excludes_private_raw_and_audits(tmp_path):
    manifest = tmp_path / "manifest.json"
    samples = [
        {"query_id": str(index), "task_id": f"task-{index}"}
        for index in range(plan_a.PLAN_A_SAMPLE_COUNT)
    ]
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    stage1 = [
        {
            "task_id": item["task_id"],
            "function": "find entity",
            "arguments": [{"argument": "a"}],
        }
        for item in samples
    ]
    stage2 = [
        {
            **item,
            "arguments": [
                {
                    "argument": "a",
                    "counterfactual_arguments": [
                        {"counterfactual_argument": f"b-{index}"}
                        for index in range(4)
                    ],
                },
                {
                    "argument": "c",
                    "counterfactual_arguments": [
                        {"counterfactual_argument": f"d-{index}"}
                        for index in range(4)
                    ],
                },
            ],
            "counterfactual_info": {
                "num_counterfactuals_requested": 4,
                "total_arguments": 2,
                "successful_counterfactuals": 8,
            },
        }
        for item in stage1
    ]
    stage3 = [
        {
            **item,
            "predecessor_functions": [
                {"predecessor_function": "first"},
                {"predecessor_function": "second"},
                {"predecessor_function": "third"},
            ],
            "predecessors": [{}, {}, {}],
            "verification_passed": True,
            "independence_passed": True,
        }
        for item in stage2
    ]
    payloads = {
        remote_key: json.dumps({}).encode()
        for remote_key in construction_modal.DOWNLOADS
    }
    for name, payload in (
        ("stage1_extracted.json", stage1),
        ("stage2_counterfactual.json", stage2),
        ("stage3_predecessor.json", stage3),
    ):
        remote_key = next(
            key for key, local_name in construction_modal.DOWNLOADS.items()
            if local_name == name
        )
        payloads[remote_key] = json.dumps(payload).encode()
    ledger_key = next(
        key for key, local_name in construction_modal.DOWNLOADS.items()
        if local_name == "llm_usage.jsonl"
    )
    payloads[ledger_key] = (
        json.dumps(
            {
                "event": "usage",
                "requested_model": plan_a.PLAN_A_MODEL,
                "resolved_model": plan_a.PLAN_A_MODEL,
            }
        )
        + "\n"
    ).encode()
    payloads[construction_modal.REMOTE_FINAL_KEY] = json.dumps(stage3).encode()
    requested = []

    def read_remote(key):
        requested.append(key)
        yield payloads[key]

    local_run = tmp_path / "run"
    local_final = tmp_path / "final.json"
    report = construction_modal.download_and_audit(
        read_remote,
        manifest_path=manifest,
        local_run_dir=local_run,
        local_final_path=local_final,
    )

    assert report["sample_count"] == plan_a.PLAN_A_SAMPLE_COUNT
    assert json.loads(local_final.read_text()) == stage3
    assert not any("raw_data_with_gold_docs.jsonl" == key.rsplit("/", 1)[-1] for key in requested)
    assert (local_run / "local_audit.json").exists()


def test_usage_audit_rejects_resolved_model_alias(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "event": "usage",
                "requested_model": runner.PLAN_A_MODEL,
                "resolved_model": "different-model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="model lock violated"):
        runner._assert_usage_models_since((ledger, 0), runner.PLAN_A_MODEL)


def test_required_usage_accounting_rejects_response_without_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("LLM_REQUIRE_USAGE_ACCOUNTING", "1")
    monkeypatch.setenv("LLM_USAGE_LEDGER_PATH", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv(
        "LLM_PRICE_MAP",
        json.dumps({runner.PLAN_A_MODEL: {"input": 0, "output": 0}}),
    )
    called = []

    def create(**payload):
        called.append(payload)
        return _chat_response(SimpleNamespace(content="answer", tool_calls=[]))

    with pytest.raises(llm_utils.LLMAccountingError, match="no usable token accounting"):
        llm_utils._accounted_api_call(
            create,
            {"model": runner.PLAN_A_MODEL, "messages": []},
            requested_model=runner.PLAN_A_MODEL,
            resolved_model=runner.PLAN_A_MODEL,
            api="chat.completions",
        )
    assert len(called) == 1


def test_remote_retriever_rejects_invalid_endpoint_item(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"results": [{"snippet": "missing docid"}]}).encode()

    monkeypatch.setattr(runner.urllib_request, "urlopen", lambda *_args, **_kwargs: Response())
    retriever = runner.RemoteRetriever("https://example.invalid/search")
    with pytest.raises(RuntimeError, match="invalid result item"):
        retriever.search("query")


def test_remote_retriever_retries_transient_cold_start_errors(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {
                    "retriever_revision": runner.PLAN_A_RETRIEVER_REVISION,
                    "results": [{"docid": "1", "snippet": "ready"}],
                }
            ).encode()

    calls = []

    def urlopen(_request, *, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise runner.urllib_error.URLError("cold start")
        return Response()

    monkeypatch.setattr(runner.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    retriever = runner.RemoteRetriever(
        "https://example.invalid/search",
        revision=runner.PLAN_A_RETRIEVER_REVISION,
    )

    assert json.loads(retriever.search("query"))[0]["docid"] == "1"
    assert calls == [1800.0, 1800.0, 1800.0]


def test_extractor_accepts_coverage_pass_on_verification_infra_timeout():
    """A pinned-ID sample whose solvability verifier only ever times out is
    accepted with an audit flag instead of being dropped (which would make
    stage 1 miss a published task ID), while a definitive wrong-answer
    verification still skips the sample."""
    from intent_construction.intent_extraction.core.base_extractor import (
        BaseExtractor,
    )

    class StubExtractor(BaseExtractor):
        def __init__(self, mode):
            self.mode = mode
            super().__init__(
                model="kimi-k2.6",
                num_arguments=2,
                max_verification_attempts=2,
                verif_model="kimi-k2.6",
                enable_model_verification=True,
            )

        def get_dataset_name(self):
            return "stub"

        def get_prompts_dir(self):
            return Path("/nonexistent")

        def _load_prompts(self):
            self.prompt_decompose = ""
            self.prompt_conversational = ""
            self.prompt_verification = ""

        def decompose(self, sample):
            return {"function": "f", "arguments": [{"argument_id": 1, "argument": "a"}]}

        def to_conversational(self, sample, decomposed):
            return {"initial_query": "f", "hints": [{"hint": "a"}]}

        def verify_coverage(self, sample, extracted):
            return True

        def verify_solvability(self, sample, extracted):
            if self.mode == "timeout":
                self._verification_infra_error = True
                return False
            self._verification_infra_error = False
            return False

        def build_output(self, sample, extracted):
            return {
                "task_id": "t-1",
                "function": extracted["function"],
                "arguments": extracted["arguments"],
            }

    sample = {"id": "1", "question": "q", "answer": "a"}

    timeout_case = StubExtractor("timeout").extract(sample)
    assert timeout_case is not None
    assert timeout_case["solvability_verification"] == "skipped_infra_timeout"

    wrong_case = StubExtractor("wrong").extract(sample)
    assert wrong_case is None
