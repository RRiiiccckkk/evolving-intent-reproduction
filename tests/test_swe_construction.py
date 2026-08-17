from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from intent_construction.intent_extraction.dataset_impl.swe_bench_verified import (
    extractor as swe_extractor_module,
)
from intent_construction.intent_extraction.dataset_impl.swe_bench_verified.extractor import (
    SWEBenchVerifiedExtractor,
)
from intent_construction.intent_extraction.dataset_impl.swe_bench_verified.reproduction import (
    REQUIRED_MODEL,
    REQUIRED_REASONING_EFFORT,
    SOURCE_POOL_COUNT,
    SWEConstructionError,
    StageCheckpointStore,
    affected_files,
    assert_build_policy,
    payload_sha256,
    select_build_rows,
    task_id_for,
    validate_eligible_extraction_row,
    validate_eligibility_repair_delta,
    validate_extraction_row,
)
from intent_construction.intent_extraction.dataset_impl.swe_bench_verified.run_published_construction import (
    StageProcessors,
    build_canary_dataset,
    build_published_dataset,
)


def _source_row(instance_id: str, *, repo: str, area: str) -> dict:
    path = f"{area}/module.py"
    return {
        "repo": repo,
        "instance_id": instance_id,
        "base_commit": "a" * 40,
        "patch": f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n",
        "test_patch": "diff --git a/tests/test_module.py b/tests/test_module.py\n",
        "problem_statement": f"Fix the regression in {instance_id}.",
        "hints_text": "",
        "created_at": "2024-01-01T00:00:00Z",
        "version": "1.0",
        "FAIL_TO_PASS": '["tests/test_module.py::test_regression"]',
        "PASS_TO_PASS": '["tests/test_module.py::test_existing"]',
        "environment_setup_commit": "b" * 40,
        "difficulty": "15 min - 1 hour",
    }


def _synthetic_pool() -> tuple[list[dict], list[dict[str, str]]]:
    rows: list[dict] = []
    published: list[dict[str, str]] = []
    for index in range(49):
        instance_id = f"org__main-{index}"
        rows.append(_source_row(instance_id, repo="org/main", area="pkg/core"))
        published.append(
            {"task_id": task_id_for(instance_id), "original_id": instance_id}
        )
    seaborn_id = "mwaskom__seaborn-3069"
    rows.append(_source_row(seaborn_id, repo="mwaskom/seaborn", area="seaborn/_core"))
    published.append(
        {"task_id": task_id_for(seaborn_id), "original_id": seaborn_id}
    )
    rows.append(
        _source_row(
            "mwaskom__seaborn-3187",
            repo="mwaskom/seaborn",
            area="seaborn/_core",
        )
    )
    for index in range(SOURCE_POOL_COUNT - len(rows)):
        rows.append(
            _source_row(
                f"fill__repo-{index}",
                repo=f"fill/repo-{index}",
                area=f"fill_{index}/area",
            )
        )
    assert len(rows) == SOURCE_POOL_COUNT
    return rows, published


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as parquet

    rows, published = _synthetic_pool()
    parquet_path = tmp_path / "verified.parquet"
    parquet.write_table(pyarrow.Table.from_pylist(rows), parquet_path)
    manifest_path = tmp_path / "eval_ids.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "swe_bench_verified",
                "split": "test",
                "num_samples": 50,
                "samples": published,
            }
        ),
        encoding="utf-8",
    )
    task_ids_path = tmp_path / "task_ids.json"
    task_ids_path.write_text(
        json.dumps(
            {
                "task_ids": [sample["task_id"] for sample in published],
                "n_total": 50,
            }
        ),
        encoding="utf-8",
    )
    return parquet_path, manifest_path, task_ids_path


def _fake_processors(calls: dict[str, int]) -> StageProcessors:
    def extract(sample: dict) -> dict:
        calls["stage1"] += 1
        instance_id = sample["instance_id"]
        argument = {
            "argument_id": 1,
            "argument": "Keep existing behavior while fixing the regression.",
            "category": "constraint",
            "counterfactual_eligible": True,
        }
        return {
            "task_id": task_id_for(instance_id),
            "original_id": instance_id,
            "task": "swe_bench",
            "question": sample["question"],
            "answer": "",
            "fully_specified_question": f"Fix {instance_id}. {argument['argument']}",
            "function": f"Fix {instance_id}",
            "arguments": [argument],
            "num_arguments": 1,
            "model_name": REQUIRED_MODEL,
            "swe_bench_metadata": {
                "repo": sample["repo"],
                "base_commit": sample["base_commit"],
                "version": sample["version"],
                "difficulty": sample["difficulty"],
                "affected_files": affected_files(sample),
                "FAIL_TO_PASS": json.loads(sample["FAIL_TO_PASS"]),
                "PASS_TO_PASS": json.loads(sample["PASS_TO_PASS"]),
            },
        }

    def counterfactual(sample: dict) -> dict:
        calls["stage2"] += 1
        output = deepcopy(sample)
        output["arguments"][0]["counterfactual_arguments"] = [
            {"counterfactual_argument": "Use compatibility mode A."},
            {"counterfactual_argument": "Use compatibility mode B."},
        ]
        output["counterfactual_info"] = {
            "model": REQUIRED_MODEL,
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
            "num_counterfactuals_requested": 2,
        }
        return output

    def g1(sample: dict, archetype: str) -> dict:
        calls["stage4"] += 1
        output = deepcopy(sample)
        output["predecessor_functions"].append(
            {
                "predecessor_function": "Explain this module's public API.",
                "counterfactual_arguments": [],
                "is_predecessor": False,
                "transition_type": "exploration",
                "taxonomy_type": archetype,
                "entity_sought": "explanation",
            }
        )
        output["g1_info"] = {
            "model": REQUIRED_MODEL,
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
            "archetype": archetype,
        }
        return output

    def impl(sample: dict) -> dict:
        calls["stage5"] += 1
        output = deepcopy(sample)
        output["predecessor_functions"][0] = {
            "predecessor_function": "Plan a feature that depends on this API.",
            "counterfactual_arguments": [
                {
                    "argument_id": 2001,
                    "argument": "Preserve backward compatibility.",
                    "category": "constraint",
                    "is_shared": False,
                }
            ],
            "is_predecessor": True,
            "transition_type": "impl_precursor",
            "entity_sought": "implementation_plan",
        }
        output["impl_precursor_info"] = {
            "model": REQUIRED_MODEL,
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
        }
        return output

    return StageProcessors(
        extract=extract,
        counterfactual=counterfactual,
        g1=g1,
        impl=impl,
    )


def _ineligible_source() -> dict:
    return {
        "id": "org__repo-1",
        "instance_id": "org__repo-1",
        "question": "Expected behavior: src/gen files should not be checked.",
        "repo": "org/repo",
        "base_commit": "a" * 40,
        "patch": "diff --git a/pkg/a.py b/pkg/a.py\n",
        "version": "1",
        "difficulty": "easy",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
    }


def _ineligible_extraction() -> dict:
    source = _ineligible_source()
    row = _fake_processors(
        {"stage1": 0, "stage2": 0, "stage4": 0, "stage5": 0}
    ).extract(source)
    row["question"] = "Expected behavior: src/gen files should not be checked."
    row["function"] = "Fix recursive ignore paths."
    row["arguments"] = [
        {
            "argument_id": 1,
            "argument": "Recursive mode currently checks ignored files.",
            "category": "symptom",
            "counterfactual_eligible": False,
        }
    ]
    row["num_arguments"] = 1
    row["fully_specified_question"] = (
        "Fix recursive ignore paths. Recursive mode currently checks ignored files."
    )
    return row


def _mock_successful_repair(monkeypatch, calls=None) -> None:
    def fake_generate_json(messages, **kwargs):
        if calls is not None:
            calls.append({"messages": messages, **kwargs})
        if kwargs["step"] == "extraction-eligibility-repair":
            return {
                "argument": "Files under src/gen should not be checked.",
                "category": "constraint",
                "grounding_quote": "src/gen files should not be checked",
                "mutable_span": "src/gen",
            }
        if kwargs["step"] == "extraction-eligibility-repair-grounding":
            return {
                "supported_by_quote": True,
                "category_correct": True,
                "self_contained": True,
                "localized_revision": True,
                "not_goal_restatement": True,
                "reasoning": "The quote directly states the expected behavior.",
            }
        if kwargs["step"] == "extraction-verification":
            return {"coverage": "complete"}
        if kwargs["step"] == "extraction-patch-alignment":
            return {"aligned": True, "reasoning": "Consistent with the patch."}
        raise AssertionError(kwargs["step"])

    monkeypatch.setattr(swe_extractor_module, "generate_json", fake_generate_json)


def test_stage1b_adds_one_grounded_eligible_argument(monkeypatch) -> None:
    calls = []
    _mock_successful_repair(monkeypatch, calls)
    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    source_row = _ineligible_extraction()
    repaired = extractor.ensure_counterfactual_eligibility(
        source_row,
        _ineligible_source(),
    )

    assert [call["step"] for call in calls] == [
        "extraction-eligibility-repair",
        "extraction-eligibility-repair-grounding",
        "extraction-verification",
        "extraction-patch-alignment",
    ]
    assert all(call["model"] == REQUIRED_MODEL for call in calls)
    assert all(
        call["reasoning_effort"] == REQUIRED_REASONING_EFFORT for call in calls
    )
    assert repaired["num_arguments"] == 2
    assert repaired["arguments"][-1] == {
        "argument_id": 2,
        "argument": "Files under src/gen should not be checked.",
        "category": "constraint",
        "counterfactual_eligible": True,
        "eligibility_repair": True,
    }
    assert repaired["eligibility_repair_info"]["reason"] == (
        "no_counterfactual_eligible_argument"
    )
    validate_eligible_extraction_row(repaired, task_id=repaired["task_id"])
    validate_eligibility_repair_delta(
        repaired,
        task_id=repaired["task_id"],
        source_row=source_row,
    )


def test_stage1b_rejects_non_verbatim_grounding_quote(monkeypatch) -> None:
    monkeypatch.setattr(
        swe_extractor_module,
        "generate_json",
        lambda *args, **kwargs: {
            "argument": "Files under src/gen should not be checked.",
            "category": "constraint",
            "grounding_quote": "SRC/GEN files should not be checked",
            "mutable_span": "src/gen",
        },
    )
    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    with pytest.raises(ValueError, match="verbatim"):
        extractor.ensure_counterfactual_eligibility(
            _ineligible_extraction(),
            _ineligible_source(),
            max_attempts=1,
        )


def test_stage1b_rejects_argument_not_supported_by_quote(monkeypatch) -> None:
    calls = []

    def fake_generate_json(messages, **kwargs):
        calls.append(kwargs["step"])
        if kwargs["step"] == "extraction-eligibility-repair":
            return {
                "argument": "Files under src/gen should not be checked.",
                "category": "constraint",
                "grounding_quote": "Expected behavior",
                "mutable_span": "src/gen",
            }
        if kwargs["step"] == "extraction-eligibility-repair-grounding":
            return {
                "supported_by_quote": False,
                "category_correct": True,
                "self_contained": True,
                "localized_revision": True,
                "not_goal_restatement": True,
                "reasoning": "The quote does not support the proposed path condition.",
            }
        raise AssertionError(kwargs["step"])

    monkeypatch.setattr(swe_extractor_module, "generate_json", fake_generate_json)
    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    with pytest.raises(ValueError, match="supported_by_quote"):
        extractor.ensure_counterfactual_eligibility(
            _ineligible_extraction(),
            _ineligible_source(),
            max_attempts=1,
        )
    assert calls == [
        "extraction-eligibility-repair",
        "extraction-eligibility-repair-grounding",
    ]


def test_stage1b_delta_rejects_mutating_the_goal(monkeypatch) -> None:
    source = _ineligible_extraction()
    _mock_successful_repair(monkeypatch)
    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    forged = extractor.ensure_counterfactual_eligibility(
        source,
        _ineligible_source(),
    )
    forged["function"] = "Change an unrelated function."
    forged["fully_specified_question"] = " ".join(
        [forged["function"]]
        + [argument["argument"] for argument in forged["arguments"]]
    )
    with pytest.raises(SWEConstructionError, match="changed 'function'"):
        validate_eligibility_repair_delta(
            forged,
            task_id=forged["task_id"],
            source_row=source,
        )


def test_stage1b_delta_rejects_mutating_existing_arguments(monkeypatch) -> None:
    source = _ineligible_extraction()
    _mock_successful_repair(monkeypatch)
    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    forged = extractor.ensure_counterfactual_eligibility(
        source,
        _ineligible_source(),
    )
    forged["arguments"][0]["argument"] = "Changed existing symptom."
    forged["fully_specified_question"] = " ".join(
        [forged["function"]]
        + [argument["argument"] for argument in forged["arguments"]]
    )
    with pytest.raises(SWEConstructionError, match="changed existing arguments"):
        validate_eligibility_repair_delta(
            forged,
            task_id=forged["task_id"],
            source_row=source,
        )


def test_stage1b_delta_rejects_appending_multiple_arguments(monkeypatch) -> None:
    source = _ineligible_extraction()
    _mock_successful_repair(monkeypatch)
    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    forged = extractor.ensure_counterfactual_eligibility(
        source,
        _ineligible_source(),
    )
    forged["arguments"].append(
        {
            "argument_id": 3,
            "argument": "Also preserve generated files.",
            "category": "constraint",
            "counterfactual_eligible": True,
        }
    )
    forged["num_arguments"] = len(forged["arguments"])
    forged["fully_specified_question"] = " ".join(
        [forged["function"]]
        + [argument["argument"] for argument in forged["arguments"]]
    )
    with pytest.raises(SWEConstructionError, match="append exactly one"):
        validate_eligibility_repair_delta(
            forged,
            task_id=forged["task_id"],
            source_row=source,
        )


def test_stage1b_does_not_call_model_for_already_eligible_row(monkeypatch) -> None:
    row = _fake_processors(
        {"stage1": 0, "stage2": 0, "stage4": 0, "stage5": 0}
    ).extract(
        {
            "instance_id": "org__repo-2",
            "question": "Fix it while preserving compatibility.",
            "repo": "org/repo",
            "base_commit": "a" * 40,
            "patch": "diff --git a/pkg/a.py b/pkg/a.py\n",
            "version": "1",
            "difficulty": "easy",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
        }
    )
    monkeypatch.setattr(
        swe_extractor_module,
        "generate_json",
        lambda *args, **kwargs: pytest.fail("eligible rows must be identity"),
    )
    extractor = SWEBenchVerifiedExtractor(
        model=REQUIRED_MODEL,
        verif_model=REQUIRED_MODEL,
        reasoning_effort=REQUIRED_REASONING_EFFORT,
    )
    before_hash = payload_sha256(row)
    assert extractor.ensure_counterfactual_eligibility(row) is row
    assert payload_sha256(row) == before_hash


def test_minimal_pair_selection_covers_unique_seaborn() -> None:
    rows, published = _synthetic_pool()
    selection = select_build_rows(rows, published)
    assert len(selection.target_rows) == 50
    assert [row["instance_id"] for row in selection.extra_rows] == [
        "mwaskom__seaborn-3187"
    ]
    assert len(selection.extraction_rows) == 51
    seaborn_pair = next(
        pairing
        for pairing in selection.pairings
        if pairing["target_instance_id"] == "mwaskom__seaborn-3069"
    )
    assert seaborn_pair["paired_instance_id"] == "mwaskom__seaborn-3187"
    assert seaborn_pair["match_level"] == "area"


def test_full_offline_build_is_exact_and_resumes(tmp_path: Path) -> None:
    parquet_path, manifest_path, task_ids_path = _write_inputs(tmp_path)
    calls = {"stage1": 0, "stage2": 0, "stage4": 0, "stage5": 0}
    run_dir = tmp_path / "run"
    final_output = tmp_path / "final.json"
    result = build_published_dataset(
        source_parquet=parquet_path,
        run_dir=run_dir,
        final_output=final_output,
        manifest_path=manifest_path,
        task_ids_path=task_ids_path,
        workers=4,
        expected_source_sha256=None,
        processors=_fake_processors(calls),
        require_live_environment=False,
    )
    assert result["status"] == "complete"
    final_rows = json.loads(final_output.read_text(encoding="utf-8"))
    assert len(final_rows) == 50
    assert {row["original_id"] for row in final_rows} == {
        sample["original_id"] for sample in _synthetic_pool()[1]
    }
    assert calls == {"stage1": 51, "stage2": 50, "stage4": 50, "stage5": 50}
    assert len(list((run_dir / "checkpoints/stage1_extraction").glob("*.json"))) == 51
    assert len(
        list(
            (run_dir / "checkpoints/stage1b_eligible_argument_v1").glob("*.json")
        )
    ) == 50
    for stage in ("stage2_counterfactual", "stage4_g1", "stage5_impl_precursor"):
        assert len(list((run_dir / "checkpoints" / stage).glob("*.json"))) == 50

    no_calls = {"stage1": 0, "stage2": 0, "stage4": 0, "stage5": 0}
    resumed = build_published_dataset(
        source_parquet=parquet_path,
        run_dir=run_dir,
        final_output=final_output,
        manifest_path=manifest_path,
        task_ids_path=task_ids_path,
        workers=4,
        expected_source_sha256=None,
        processors=_fake_processors(no_calls),
        require_live_environment=False,
    )
    assert resumed["status"] == "complete"
    assert no_calls == {"stage1": 0, "stage2": 0, "stage4": 0, "stage5": 0}


def test_empty_checkpoint_fails_closed(tmp_path: Path) -> None:
    task_id = task_id_for("org__repo-1")
    sample = _fake_processors({"stage1": 0, "stage2": 0, "stage4": 0, "stage5": 0}).extract(
        {
            "instance_id": "org__repo-1",
            "question": "Fix it.",
            "repo": "org/repo",
            "base_commit": "a" * 40,
            "patch": "diff --git a/pkg/a.py b/pkg/a.py\n",
            "version": "1",
            "difficulty": "easy",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
        }
    )
    store = StageCheckpointStore(
        tmp_path / "checkpoints",
        stage="stage1_extraction",
        inputs_by_id={task_id: {"task_id": task_id}},
        validator=validate_extraction_row,
    )
    store.path_for(task_id).write_text("", encoding="utf-8")
    with pytest.raises(SWEConstructionError, match="empty"):
        store.load_available()
    assert sample["task_id"] == task_id


def test_one_task_canary_is_isolated_and_outputs_one_row(tmp_path: Path) -> None:
    parquet_path, manifest_path, task_ids_path = _write_inputs(tmp_path)
    calls = {"stage1": 0, "stage2": 0, "stage4": 0, "stage5": 0}
    result = build_canary_dataset(
        source_parquet=parquet_path,
        run_dir=tmp_path / "canary-only",
        canary_task_id="mwaskom__seaborn-3069",
        manifest_path=manifest_path,
        task_ids_path=task_ids_path,
        expected_source_sha256=None,
        processors=_fake_processors(calls),
        require_live_environment=False,
    )
    assert result["status"] == "canary_complete"
    assert result["coverage"] == 1
    assert calls == {"stage1": 2, "stage2": 1, "stage4": 1, "stage5": 1}
    rows = json.loads(Path(result["output"]).read_text(encoding="utf-8"))
    assert [row["original_id"] for row in rows] == ["mwaskom__seaborn-3069"]


def test_policy_rejects_wrong_model_effort_and_limits() -> None:
    with pytest.raises(SWEConstructionError, match="requires exactly"):
        assert_build_policy(
            model="gpt-5.1",
            reasoning_effort="medium",
            environment={},
            require_credentials=False,
        )
    with pytest.raises(SWEConstructionError, match="reasoning effort"):
        assert_build_policy(
            model=REQUIRED_MODEL,
            reasoning_effort="high",
            environment={},
            require_credentials=False,
        )
    with pytest.raises(SWEConstructionError, match="forbidden"):
        assert_build_policy(
            model=REQUIRED_MODEL,
            reasoning_effort=REQUIRED_REASONING_EFFORT,
            environment={"LLM_DEFAULT_MAX_OUTPUT_TOKENS": "4096"},
            require_credentials=False,
        )


def test_construction_script_respects_python_bin() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "intent_construction/scripts/swe_bench_verified.sh"
    ).read_text(encoding="utf-8")
    assert '"${PYTHON_BIN:-python}" -m' in script
