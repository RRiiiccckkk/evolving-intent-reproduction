from __future__ import annotations

import ast
import json
import sqlite3
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from intent_construction.intent_extraction.dataset_impl.bird_sql.download_required import (
    assert_single_database_cache,
    download_range_file,
    evict_required_database,
    fetch_range,
    ordered_database_ids,
    required_files,
)
from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import (
    DEFAULT_EVAL_MANIFEST_PATH,
    DEFAULT_TASK_IDS_PATH,
    REQUIRED_MODEL,
    BirdReproductionError,
    TaskCheckpoint,
    acquire_exclusive_run_lock,
    assert_required_model,
    atomic_write_json,
    load_published_manifest,
    load_published_task_ids,
    validate_stage_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_published_bird_ids_are_exact_and_aligned():
    ids = load_published_task_ids()
    manifest = load_published_manifest()

    assert len(ids) == len(set(ids)) == 100
    assert [row["task_id"] for row in manifest] == ids
    assert sum("_train_" in task_id for task_id in ids) == 85
    assert sum("_dev_" in task_id for task_id in ids) == 15
    assert len({row["db_id"] for row in manifest}) == 35
    assert json.loads(DEFAULT_EVAL_MANIFEST_PATH.read_text())["split"] == "train+dev"


def test_only_kimi_k2_6_is_accepted():
    assert assert_required_model(REQUIRED_MODEL, context="test") == REQUIRED_MODEL
    for invalid in (None, "", "gpt-5.1", "kimi-k2.5", "Kimi-K2.6"):
        with pytest.raises(BirdReproductionError):
            assert_required_model(invalid, context="test")


def test_bird_evaluation_runtime_policy_is_fail_closed(monkeypatch, tmp_path):
    pytest.importorskip("openai")
    from evaluation.runners.run_bird_experiment import (
        assert_evaluation_runtime_policy,
    )

    policy = {
        "LLM_LOCKED_MODEL": REQUIRED_MODEL,
        "LLM_REASONING_EFFORT": "medium",
        "LLM_DISABLE_OUTPUT_LIMITS": "1",
        "LLM_REQUIRE_USAGE_ACCOUNTING": "1",
        "LLM_USAGE_LEDGER_PATH": str(tmp_path / "usage.jsonl"),
    }
    for name in (
        "REASONING_EFFORT",
        "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
        "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
        "LLM_COST_HARD_CAP_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in policy.items():
        monkeypatch.setenv(name, value)

    assert_evaluation_runtime_policy()

    invalid = {
        "LLM_LOCKED_MODEL": "kimi-k2.5",
        "LLM_REASONING_EFFORT": "high",
        "LLM_DISABLE_OUTPUT_LIMITS": "0",
        "LLM_REQUIRE_USAGE_ACCOUNTING": "0",
        "LLM_USAGE_LEDGER_PATH": "",
    }
    for name, value in invalid.items():
        monkeypatch.setenv(name, value)
        with pytest.raises(BirdReproductionError):
            assert_evaluation_runtime_policy()
        monkeypatch.setenv(name, policy[name])

    forbidden = {
        "REASONING_EFFORT": "medium",
        "LLM_DEFAULT_MAX_OUTPUT_TOKENS": "4096",
        "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS": "4096",
        "LLM_COST_HARD_CAP_USD": "30",
    }
    for name, value in forbidden.items():
        monkeypatch.setenv(name, value)
        with pytest.raises(BirdReproductionError):
            assert_evaluation_runtime_policy()
        monkeypatch.delenv(name)


def test_checkpoint_saves_each_success_and_retries_failures(tmp_path):
    ids = ["bird_sql_train_1", "bird_sql_dev_2"]
    path = tmp_path / "checkpoint.json"
    checkpoint = TaskCheckpoint(
        path,
        stage="test_stage",
        required_ids=ids,
        model=REQUIRED_MODEL,
    )

    checkpoint.record_failure(ids[0], "temporary failure")
    assert checkpoint.processed_ids == []
    checkpoint.record_success({"task_id": ids[1], "model_name": REQUIRED_MODEL})

    resumed = TaskCheckpoint(
        path,
        stage="test_stage",
        required_ids=ids,
        model=REQUIRED_MODEL,
        resume=True,
    )
    assert resumed.processed_ids == [ids[1]]
    assert resumed.pending_ids == [ids[0]]
    resumed.record_success({"task_id": ids[0], "model_name": REQUIRED_MODEL})
    resumed.mark_complete()

    payload = json.loads(path.read_text())
    assert payload["processed_ids"] == ids
    assert payload["complete"] is True
    assert [row["task_id"] for row in payload["results"]] == ids


def test_bird_construction_lock_rejects_a_second_process(tmp_path):
    first = acquire_exclusive_run_lock(tmp_path)
    try:
        with pytest.raises(BirdReproductionError, match="already holds"):
            acquire_exclusive_run_lock(tmp_path)
    finally:
        first.close()

    resumed = acquire_exclusive_run_lock(tmp_path)
    resumed.close()


def test_atomic_json_write_preserves_previous_file_on_replace_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "state.json"
    target.write_text('{"stable": true}\n')

    def fail_replace(source, destination):
        raise KeyboardInterrupt

    monkeypatch.setattr("os.replace", fail_replace)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_json(target, {"stable": False})

    assert json.loads(target.read_text()) == {"stable": True}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_stage_coverage_rejects_one_missing_id():
    ids = ["bird_sql_train_1", "bird_sql_dev_2"]
    rows = [{"task_id": ids[0], "model_name": REQUIRED_MODEL}]
    with pytest.raises(BirdReproductionError, match="coverage failure"):
        validate_stage_rows(rows, stage="test", required_ids=ids)


def test_required_download_plan_contains_only_two_json_and_35_sqlite_files():
    specs = required_files(load_published_manifest())
    assert len(specs) == 37
    assert sum(spec.kind == "json" for spec in specs) == 2
    assert sum(spec.kind == "sqlite" for spec in specs) == 35
    assert all(".zip" not in spec.url and ".zip" not in spec.remote_path for spec in specs)
    assert all("datasetVersionNumber=1" in spec.url for spec in specs)
    assert all(spec.db_id is None or spec.db_id in spec.remote_path for spec in specs)


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    requests: list[str | None] = []
    force_whole_response = False

    def do_GET(self):
        range_header = self.headers.get("Range")
        type(self).requests.append(range_header)
        if type(self).force_whole_response:
            self.send_response(200)
            self.send_header("Content-Length", str(len(type(self).payload)))
            self.end_headers()
            self.wfile.write(type(self).payload)
            return
        if not range_header or not range_header.startswith("bytes="):
            self.send_error(400)
            return
        start_text, end_text = range_header[6:].split("-", 1)
        start = int(start_text)
        end = min(int(end_text), len(type(self).payload) - 1)
        body = type(self).payload[start : end + 1]
        self.send_response(206)
        self.send_header(
            "Content-Range", f"bytes {start}-{end}/{len(type(self).payload)}"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", '"fixture"')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


@pytest.fixture
def range_server():
    _RangeHandler.payload = (b"range-only-bird-data-" * 19) + b"end"
    _RangeHandler.requests = []
    _RangeHandler.force_whole_response = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/data"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_range_download_is_atomic_and_resumes_partial_file(tmp_path, range_server):
    destination = tmp_path / "database.sqlite"
    partial = tmp_path / ".database.sqlite.part"
    partial.write_bytes(_RangeHandler.payload[:11])

    downloaded = download_range_file(
        range_server,
        destination,
        chunk_size=17,
    )

    assert downloaded == len(_RangeHandler.payload)
    assert destination.read_bytes() == _RangeHandler.payload
    assert not partial.exists()
    assert _RangeHandler.requests[0] == "bytes=0-0"
    assert _RangeHandler.requests[1].startswith("bytes=11-")
    assert all(value and value.startswith("bytes=") for value in _RangeHandler.requests)


def test_range_fetch_rejects_server_that_ignores_range(range_server):
    _RangeHandler.force_whole_response = True
    with pytest.raises(BirdReproductionError, match="ignored HTTP Range"):
        fetch_range(range_server, 0, 0)


def test_range_download_rejects_file_over_space_budget(tmp_path, range_server):
    destination = tmp_path / "too-large.sqlite"
    with pytest.raises(BirdReproductionError, match="temporary-space budget"):
        download_range_file(
            range_server,
            destination,
            max_bytes=len(_RangeHandler.payload) - 1,
        )

    assert not destination.exists()
    assert not (tmp_path / ".too-large.sqlite.part").exists()


def test_low_disk_order_resumes_one_partial_and_rejects_two(tmp_path):
    manifest = load_published_manifest()
    specs = required_files(manifest, data_dir=tmp_path)
    database_specs = [spec for spec in specs if spec.kind == "sqlite"]
    resident = database_specs[-1]
    partial = resident.local_path.with_name(f".{resident.local_path.name}.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    order = ordered_database_ids(manifest, data_dir=tmp_path)
    assert order[0] == resident.db_id
    assert len(order) == len(set(order)) == 35
    assert_single_database_cache(specs, requested_db_id=resident.db_id)
    with pytest.raises(BirdReproductionError, match="cache still contains"):
        assert_single_database_cache(
            specs,
            requested_db_id=database_specs[0].db_id,
        )

    second = database_specs[0]
    second.local_path.parent.mkdir(parents=True)
    second.local_path.write_bytes(b"SQLite format 3\x00")
    with pytest.raises(BirdReproductionError, match="more than one database"):
        ordered_database_ids(manifest, data_dir=tmp_path)


def test_database_eviction_is_exact_and_confined_to_cache(tmp_path):
    manifest = load_published_manifest()
    spec = next(
        spec
        for spec in required_files(manifest, data_dir=tmp_path / "cache")
        if spec.kind == "sqlite"
    )
    partial = spec.local_path.with_name(f".{spec.local_path.name}.part")
    spec.local_path.parent.mkdir(parents=True)
    spec.local_path.write_bytes(b"SQLite format 3\x00")
    partial.write_bytes(b"partial")

    evict_required_database(spec, data_dir=tmp_path / "cache")
    assert not spec.local_path.exists()
    assert not partial.exists()

    outside = tmp_path / spec.db_id / f"{spec.db_id}.sqlite"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"keep")
    escaped = replace(spec, local_path=outside)
    with pytest.raises(BirdReproductionError, match="outside the run cache"):
        evict_required_database(escaped, data_dir=tmp_path / "cache")
    assert outside.read_bytes() == b"keep"


def _api_calls(path: Path, function_name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.id if isinstance(node.func, ast.Name) else None
        if called == function_name:
            calls.append(node)
    return calls


def test_bird_api_calls_do_not_set_output_token_limits():
    files_and_calls = [
        (
            REPO_ROOT
            / "intent_construction/intent_extraction/dataset_impl/bird_sql/extractor.py",
            "generate_text",
        ),
        (
            REPO_ROOT
            / "intent_construction/retrospective_expansion/predecessor/generate_predecessors_sql_llm.py",
            "generate_json",
        ),
        (
            REPO_ROOT
            / "intent_construction/retrospective_expansion/predecessor/sql_naturalizer.py",
            "generate_json",
        ),
    ]
    forbidden = {"max_tokens", "max_output_tokens", "max_completion_tokens"}
    for path, function_name in files_and_calls:
        calls = _api_calls(path, function_name)
        assert calls, f"no {function_name} call found in {path}"
        for call in calls:
            assert forbidden.isdisjoint(
                keyword.arg for keyword in call.keywords if keyword.arg
            )

    for script in (
        REPO_ROOT / "intent_construction/scripts/bird_sql.sh",
        REPO_ROOT / "evaluation/scripts/run_bird.sh",
    ):
        text = script.read_text()
        assert "LLM_DISABLE_OUTPUT_LIMITS=1" in text
        assert "LLM_COST_HARD_CAP_USD" in text


def test_official_bird_result_semantics():
    pytest.importorskip("openai")
    from evaluation.runners.run_bird_experiment import official_results_equal

    def result(rows):
        return SimpleNamespace(success=True, rows=rows)

    assert official_results_equal(result([(1, 2), (1, 2)]), result([(1, 2)]))
    assert official_results_equal(result([(2,), (1,)]), result([(1,), (2,)]))
    assert not official_results_equal(result([(1, 2)]), result([(2, 1)]))


def test_bird_sql_execution_enforces_real_query_timeout(tmp_path):
    from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import (
        execute_sql,
    )

    database = tmp_path / "timeout.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE marker (value INTEGER)")
    connection.commit()
    connection.close()

    started = time.monotonic()
    result = execute_sql(
        database,
        "WITH RECURSIVE forever(x) AS ("
        "SELECT 1 UNION ALL SELECT x + 1 FROM forever"
        ") SELECT SUM(x) FROM forever",
        timeout=0.01,
    )

    assert result.success is False
    assert "interrupted" in str(result.error).lower()
    assert time.monotonic() - started < 2


def test_complete_bird_evaluation_checkpoint_recovers_missing_aggregate(tmp_path):
    pytest.importorskip("openai")
    from evaluation.runners.run_bird_experiment import (
        _bird_verifier_policy,
        _finalize_completed_checkpoint,
    )

    ids = ["bird_sql_train_1", "bird_sql_dev_2"]
    checkpoint = TaskCheckpoint(
        tmp_path / "evaluation_checkpoint.json",
        stage="bird_evaluation_t1_g0_p0",
        required_ids=ids,
        model=REQUIRED_MODEL,
    )
    for task_id in ids:
        checkpoint.record_success(
            {
                "task_id": task_id,
                "success": True,
                "error": None,
                "decoding": ["SELECT 1"],
                "prediction": "SELECT 1",
                "correct": True,
                "model_name": REQUIRED_MODEL,
                "per_turn_results": [
                    {
                        "turn": 0,
                        "correct": True,
                        "gold_sql": "SELECT 1",
                        "model_sql": "SELECT 1",
                        "model_sql_valid": True,
                    }
                ],
                "bird_verifier": _bird_verifier_policy(),
            }
        )

    output = tmp_path / "evaluation.json"
    recovered = _finalize_completed_checkpoint(
        checkpoint,
        output_path=output,
        required_ids=ids,
        turns=1,
        model=REQUIRED_MODEL,
    )

    assert recovered is not None
    assert list(json.loads(output.read_text())) == ids
    assert json.loads(checkpoint.path.read_text())["complete"] is True


def test_bird_checkpoint_upgrades_explicit_gold_timeout(tmp_path):
    pytest.importorskip("openai")
    from evaluation.runners.run_bird_experiment import (
        _bird_verifier_policy,
        _upgrade_checkpoint_verifier_policy,
    )

    task_id = "bird_sql_train_1"
    checkpoint = TaskCheckpoint(
        tmp_path / "evaluation_checkpoint.json",
        stage="bird_evaluation_t7_g2_p2",
        required_ids=[task_id],
        model=REQUIRED_MODEL,
    )
    legacy_policy = _bird_verifier_policy()
    legacy_policy.pop("gold_timeout_seconds")
    checkpoint.record_success(
        {
            "task_id": task_id,
            "model_name": REQUIRED_MODEL,
            "bird_verifier": legacy_policy,
        }
    )

    _upgrade_checkpoint_verifier_policy(checkpoint)

    persisted = json.loads(checkpoint.path.read_text())["results"][0]
    assert persisted["bird_verifier"] == _bird_verifier_policy()
    assert persisted["bird_verifier"]["timeout_seconds"] == 30.0
    assert persisted["bird_verifier"]["gold_timeout_seconds"] == 120.0


def test_database_grouping_uses_published_order_and_one_id_smoke_database():
    pytest.importorskip("openai")
    from evaluation.runners.run_bird_experiment import (
        _database_task_ids,
        _pending_database_ids,
    )

    manifest = load_published_manifest()
    formula_ids = _database_task_ids(manifest, "formula_1")
    assert len(formula_ids) == 1
    assert _pending_database_ids(manifest, formula_ids) == ["formula_1"]


def test_bird_shells_use_single_database_orchestrators():
    construction = (REPO_ROOT / "intent_construction/scripts/bird_sql.sh").read_text()
    evaluation = (REPO_ROOT / "evaluation/scripts/run_bird.sh").read_text()

    assert "run_published_construction" in construction
    assert "--data-dir" in construction
    assert "BIRD_ONLY_DATABASE" in construction
    assert "--database" in evaluation
    assert "--db_id" in evaluation
    assert "--evict-database" in evaluation
    assert "BIRD_ONLY_DATABASE" in evaluation
    assert "+        --" not in construction
    assert "+        --" not in evaluation


def test_counterfactual_resolves_unqualified_column_table(tmp_path):
    from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import (
        execute_sql,
    )
    from intent_construction.retrospective_expansion.counterfactual.generate_counterfactuals_sql import (
        SQLCounterfactualGenerator,
    )

    database = tmp_path / "business.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE Business (city TEXT, stars REAL)")
        connection.executemany(
            "INSERT INTO Business VALUES (?, ?)",
            [("Phoenix", 2), ("Phoenix", 2), ("Tempe", 2), ("Mesa", 3)],
        )
        connection.commit()
    finally:
        connection.close()

    sql = "SELECT COUNT(*) FROM Business WHERE city = 'Phoenix'"
    original = execute_sql(database, sql)
    generated = SQLCounterfactualGenerator(num_counterfactuals=1).generate_counterfactual(
        {
            "argument_id": 1,
            "argument": "The city is Phoenix.",
            "sql_column": "city",
            "sql_table": "",
            "sql_operator": "=",
            "sql_value": "Phoenix",
        },
        sql,
        str(database),
        original,
    )

    assert len(generated) == 1
    assert generated[0]["counterfactual_answer"] == "1"


def test_counterfactual_handles_numeric_threshold_expression(tmp_path):
    from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import (
        execute_sql,
    )
    from intent_construction.retrospective_expansion.counterfactual.generate_counterfactuals_sql import (
        SQLCounterfactualGenerator,
    )

    database = tmp_path / "employees.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE Employee (BirthDate TEXT)")
        connection.executemany(
            "INSERT INTO Employee VALUES (?)",
            [(f"{year}-01-01",) for year in range(1930, 1991)],
        )
        connection.commit()
    finally:
        connection.close()

    sql = "SELECT COUNT(*) FROM Employee WHERE STRFTIME('%Y', BirthDate) < '1960'"
    original = execute_sql(database, sql)
    generated = SQLCounterfactualGenerator(num_counterfactuals=3).generate_counterfactual(
        {
            "argument_id": 1,
            "argument": "The year of BirthDate is less than 1960.",
            "sql_column": "TIME_TO_STR(CAST(BirthDate AS TIMESTAMP), '%Y')",
            "sql_table": "",
            "sql_operator": "<",
            "sql_value": "1960",
        },
        sql,
        str(database),
        original,
    )

    assert len(generated) == 3
    assert all(item["counterfactual_sql"] != sql for item in generated)
    assert all("TIME_TO_STR" not in item["counterfactual_argument"] for item in generated)


def test_naturalizer_function_gate_understands_sql_identifier_words():
    from intent_construction.retrospective_expansion.predecessor.sql_naturalizer import (
        _has_function_mention,
    )

    assert _has_function_mention(
        "Show the businesses with the largest review totals.",
        "Business.business_id, SUM(review_count)",
    )
    assert _has_function_mention(
        "Sort them by stars.",
        "ORDER BY Business.stars DESC",
    )
    assert not _has_function_mention(
        "Give me a different summary.",
        "Business.business_id, SUM(review_count)",
    )


def test_naturalizer_value_leak_uses_whole_words_for_word_literals():
    from intent_construction.retrospective_expansion.predecessor.sql_naturalizer import (
        _check_value_leak,
    )

    assert _check_value_leak(
        "Which categories have the highest average review count?",
        ["High", "US"],
    ) == []
    assert _check_value_leak("Keep the High group.", ["High"]) == ["High"]
    assert _check_value_leak("Keep New York entries.", ["New York"]) == [
        "New York"
    ]
    assert _check_value_leak("How many claims are there?", ["Ms."]) == []
    assert _check_value_leak("How many claims did Ms. have?", ["Ms."]) == [
        "Ms."
    ]
    assert _check_value_leak("Use a 200mg dose.", ["200 MG"]) == ["200 MG"]


def test_naturalizer_ignores_sql_date_format_control_literals():
    from intent_construction.retrospective_expansion.predecessor.sql_naturalizer import (
        _extract_literal_values,
    )

    values = _extract_literal_values(
        "SELECT COUNT(*) FROM visits "
        "WHERE race = 'black' AND strftime('%Y', visited_at) = '2013'"
    )

    assert set(values) == {"black", "2013"}
    assert "%Y" not in values


def test_extractor_value_leak_uses_token_boundaries_for_short_titles():
    from intent_construction.intent_extraction.dataset_impl.bird_sql.extractor import (
        _check_value_leak,
    )

    assert _check_value_leak("Give the number of claims.", ["Ms."]) == []
    assert _check_value_leak("Give the number of claims for Ms.", ["Ms."]) == [
        "Ms."
    ]
    assert _check_value_leak("Which dose is 200mg?", ["200 MG"]) == ["200 MG"]


def test_sparse_select_query_repeats_valid_plan_for_three_candidates():
    from intent_construction.intent_extraction.dataset_impl.bird_sql.sql_parser import (
        parse_sql,
    )
    from intent_construction.retrospective_expansion.predecessor.generate_predecessors_sql_llm import (
        _build_messages,
        _plans_for_candidate_count,
    )

    sql = "SELECT COUNT(business_id) FROM Business WHERE city = 'Phoenix'"
    plans = _plans_for_candidate_count(parse_sql(sql), 3)

    assert len(plans) == 3
    assert [plan.change_set for plan in plans] == [("SELECT",)] * 3
    messages = _build_messages(
        "Business(business_id, city, stars)",
        "How many businesses are in Phoenix?",
        sql,
        plans[1],
        candidate_index=1,
        candidate_count=3,
        avoid_sqls=["SELECT AVG(stars) FROM Business WHERE city = 'Phoenix'"],
    )
    assert "candidate 2 of 3" in messages[0]["content"]
    assert "SELECT AVG(stars)" in messages[0]["content"]


def test_limit_makes_order_by_change_preferable_to_group_by_only():
    from intent_construction.intent_extraction.dataset_impl.bird_sql.sql_parser import (
        parse_sql,
    )
    from intent_construction.retrospective_expansion.predecessor.generate_predecessors_sql_llm import (
        _plan_attempt_queue,
        _plans_for_candidate_count,
    )

    sql = (
        "SELECT AwayTeam FROM matchs WHERE HomeTeam = 'Caen' "
        "AND season = 2010 AND FTR = 'A' GROUP BY AwayTeam "
        "ORDER BY COUNT(AwayTeam) DESC LIMIT 1"
    )
    plans = _plans_for_candidate_count(parse_sql(sql), 3)

    assert [plan.change_set for plan in plans] == [
        ("SELECT", "GROUP_BY", "ORDER_BY"),
        ("SELECT",),
        ("ORDER_BY",),
    ]
    assert [plan.change_set for plan in _plan_attempt_queue(parse_sql(sql), 3)] == [
        ("SELECT", "GROUP_BY", "ORDER_BY"),
        ("SELECT",),
        ("ORDER_BY",),
        ("SELECT", "GROUP_BY"),
        ("SELECT", "ORDER_BY"),
        ("GROUP_BY", "ORDER_BY"),
        ("GROUP_BY",),
    ]


def test_predecessor_validator_detects_from_change_and_duplicate_results(tmp_path):
    from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import (
        execute_sql,
    )
    from intent_construction.retrospective_expansion.predecessor.function_change_planner import (
        ALWAYS_PRESERVED,
        FunctionChangePlan,
    )
    from intent_construction.retrospective_expansion.predecessor.generate_predecessors_sql_llm import (
        _validate_candidate,
        diff_clauses,
    )

    database = tmp_path / "business.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE Business (id INTEGER, city TEXT, stars REAL)")
        connection.execute("CREATE TABLE Other (id INTEGER, city TEXT, stars REAL)")
        rows = [(1, "Phoenix", 2.0), (2, "Phoenix", 4.0)]
        connection.executemany("INSERT INTO Business VALUES (?, ?, ?)", rows)
        connection.executemany("INSERT INTO Other VALUES (?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()

    gold_sql = "SELECT COUNT(id) FROM Business WHERE city = 'Phoenix'"
    assert diff_clauses(
        gold_sql,
        "SELECT AVG(stars) FROM Other WHERE city = 'Phoenix'",
    )["FROM"] is True

    plan = FunctionChangePlan(
        change_set=("SELECT",),
        preserve_set=(),
        always_preserved=ALWAYS_PRESERVED,
        score=3,
    )
    gold_result = execute_sql(database, gold_sql)
    seen_hashes: set[str] = set()
    seen_results = []
    first = _validate_candidate(
        "SELECT AVG(stars) AS metric FROM Business WHERE city = 'Phoenix'",
        plan,
        gold_sql,
        str(database),
        gold_result,
        seen_hashes,
        seen_results,
        5,
    )
    second = _validate_candidate(
        "SELECT SUM(stars) / COUNT(stars) AS metric FROM Business WHERE city = 'Phoenix'",
        plan,
        gold_sql,
        str(database),
        gold_result,
        seen_hashes,
        seen_results,
        5,
    )
    out_of_plan = _validate_candidate(
        "SELECT AVG(stars) FROM Business WHERE city = 'Phoenix' GROUP BY city",
        plan,
        gold_sql,
        str(database),
        gold_result,
        set(),
        [],
        5,
    )

    assert first[0] is True
    assert second[:2] == (False, "duplicate_result")
    assert out_of_plan[:2] == (False, "preserve_violation:GROUP_BY")


def test_gold_execution_failure_grades_turn_and_never_crashes(tmp_path):
    """A turn gold SQLite cannot execute is graded incorrect, not fatal.

    Constructed partial SQL (unrevealed-join Cartesian SUM) can overflow
    int64 or time out. official_results_equal already treats either-side
    execution failure as non-equal, so the runner must record the failed
    gold and continue instead of killing the whole 100-sample run.
    """
    pytest.importorskip("openai")
    from evaluation.runners.run_bird_experiment import evaluate_bird_response

    database = tmp_path / "overflow.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE a (v INTEGER)")
    connection.executemany(
        "INSERT INTO a (v) VALUES (?)",
        [(9_000_000_000_000_000_000,)] * 4,
    )
    connection.commit()
    connection.close()

    overflow_gold = "SELECT SUM(v) FROM a"
    grading = evaluate_bird_response("```sql\nSELECT SUM(v) FROM a\n```", overflow_gold, database)
    assert grading["execution_match"] is False
    assert grading["gold_answer"] is None
    assert "integer overflow" in str(grading["gold_error"])
    assert grading["model_sql"] == "SELECT SUM(v) FROM a"
    assert grading["error"] is not None

    working_gold = "SELECT COUNT(*) FROM a"
    grading_ok = evaluate_bird_response("```sql\nSELECT COUNT(*) FROM a\n```", working_gold, database)
    assert grading_ok["execution_match"] is True
    assert grading_ok["gold_error"] is None
    assert grading_ok["error"] is None
