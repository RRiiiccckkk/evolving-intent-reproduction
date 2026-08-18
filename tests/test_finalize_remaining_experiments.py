from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from reproduction.finalize_remaining_experiments import (
    BROWSE_RETRIEVER_REVISION,
    EXPECTED_MODEL,
    FinalizationError,
    ReportLayout,
    build_report,
    validate_browsecomp_evaluation_artifacts,
    write_report,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-{index:03d}" for index in range(count)]


def _policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_lock": {
            "construction": EXPECTED_MODEL,
            "evaluation": EXPECTED_MODEL,
            "judge": EXPECTED_MODEL,
            "naturalizer": EXPECTED_MODEL,
            "reasoning_effort": "medium",
            "allow_other_models": False,
        },
        "api_policy": {
            "forbidden_request_fields": [
                "max_tokens",
                "max_completion_tokens",
                "max_output_tokens",
            ]
        },
        "accounting": {
            "hard_cap_usd": None,
            "reservation_is_confirmed_spend": False,
        },
        "scenarios": {
            "single": {
                "turns": 1,
                "function_switches": 0,
                "argument_revisions": 0,
            },
            "evolve": {
                "turns": 7,
                "function_switches": 2,
                "argument_revisions": 2,
            },
        },
        "benchmarks": {
            "bird_sql": {"sample_count": 100},
            "browsecomp_plus": {"sample_count": 100},
            "swe_bench_verified": {"sample_count": 50},
        },
    }


def _usage(timestamp: str, cost: float) -> dict[str, object]:
    return {
        "event": "usage",
        "timestamp": timestamp,
        "requested_model": EXPECTED_MODEL,
        "resolved_model": EXPECTED_MODEL,
        "input_tokens": 100,
        "output_tokens": 40,
        "cached_tokens": 10,
        "reasoning_tokens": 25,
        "cost_usd": cost,
    }


def _bird_results(ids: list[str], *, turns: int, correct: int) -> dict[str, object]:
    return {
        task_id: {
            "task_id": task_id,
            "success": True,
            "error": None,
            "decoding": ["SELECT 1"] * turns,
            "model_name": EXPECTED_MODEL,
            "correct": index < correct,
            "bird_verifier": {"llm_judge": False},
        }
        for index, task_id in enumerate(ids)
    }


def _browse_results(ids: list[str], *, turns: int, correct: int) -> dict[str, object]:
    return {
        task_id: {
            "task_id": task_id,
            "success": True,
            "error": None,
            "prediction": "redacted aggregate answer",
            "responses": ["answer"] * turns,
            "correct": index < correct,
            "metadata": {
                "query": "DO-NOT-EXPORT-PLAINTEXT-QUERY",
                "gold_docs": ["DO-NOT-EXPORT-GOLD-DOCUMENT"],
                "api_key": "sk-do-not-export",
            },
        }
        for index, task_id in enumerate(ids)
    }


def _browse_checkpoints(
    result_path: Path,
    ids: list[str],
    results: dict[str, object],
    *,
    scenario: str,
    turns: int,
    revisions: int,
    switches: int,
) -> None:
    checkpoint_dir = Path(f"{result_path}.checkpoints")
    for index, task_id in enumerate(ids):
        _write_json(
            checkpoint_dir / f"task-{index:03d}.json",
            {
                "schema_version": 2,
                "task_id": task_id,
                "input_sha256": "a" * 64,
                "policy": {
                    "workflow": "browsecomp-plan-a-evaluation",
                    "schema_version": 2,
                    "requested_model": EXPECTED_MODEL,
                    "resolved_model": EXPECTED_MODEL,
                    "reasoning_effort": "medium",
                    "naturalizer_model": EXPECTED_MODEL,
                    "judge_model": EXPECTED_MODEL,
                    "max_tokens": None,
                    "max_tool_calls": 50,
                    "max_search_iterations": 51,
                    "force_final_answer": True,
                    "no_retrieval_history": False,
                    "recap_method": None,
                    "selected_task_ids": ids,
                    "retriever": {
                        "k": 5,
                        "revision": BROWSE_RETRIEVER_REVISION,
                    },
                    "scenario": {
                        "name": scenario,
                        "num_turns": turns,
                        "num_revisions": revisions,
                        "num_switches": switches,
                        "ordering": "interleaved",
                    },
                },
                "result": results[task_id],
            },
        )


def _swe_results(ids: list[str], *, scenario: str, turns: int, correct: int) -> dict[str, object]:
    return {
        task_id: {
            "task_id": task_id,
            "success": True,
            "error": None,
            "prediction": "diff --git a/x b/x",
            "metadata": {
                "requested_model": EXPECTED_MODEL,
                "resolved_model": EXPECTED_MODEL,
                "reasoning_effort": "medium",
                "checkpoint_scenario": scenario,
                "n_user_turns_delivered": turns,
            },
            "swe_eval": {
                "resolved": index < correct,
                "harness_error": None,
                "patch_extracted": True,
            },
        }
        for index, task_id in enumerate(ids)
    }


def _complete_layout(tmp_path: Path) -> tuple[ReportLayout, Path]:
    root = tmp_path / "repo"
    bird_ids = _ids("bird", 100)
    browse_ids = _ids("browse", 100)
    swe_ids = _ids("swe", 50)
    indices = root / "intent_construction" / "eval_indices"
    _write_json(indices / "bird_sql_task_ids.json", {"num_samples": 100, "task_ids": bird_ids})
    _write_json(
        indices / "browsecomp_plus_task_ids.json",
        {"num_samples": 100, "task_ids": browse_ids},
    )
    _write_json(indices / "swe_bench_verified_task_ids.json", {"task_ids": swe_ids})
    _write_json(root / "reproduction/config/paper_remaining_kimi_k2_6.json", _policy())

    experiments = root / "evaluation" / "experiments"
    bird_single_path = experiments / "fully_specified/bird_sql_n100/kimi-k2.6.json"
    bird_evolve_path = (
        experiments
        / "combined_independent/bird_sql_n100/kimi-k2.6_t7_g2_p2.json"
    )
    _write_json(bird_single_path, _bird_results(bird_ids, turns=1, correct=80))
    _write_json(bird_evolve_path, _bird_results(bird_ids, turns=7, correct=60))
    bird_stage_rows = [
        {"task_id": task_id, "model_name": EXPECTED_MODEL} for task_id in bird_ids
    ]
    bird_predecessor_rows = [
        {
            "task_id": task_id,
            "model_name": EXPECTED_MODEL,
            "predecessor_functions": [{"name": "a"}, {"name": "b"}],
            "predecessor_info": {
                "model": EXPECTED_MODEL,
                "naturalizer_model": EXPECTED_MODEL,
            },
        }
        for task_id in bird_ids
    ]
    bird_checkpoints = (
        (
            root
            / "intent_construction/intent_extraction/output/bird_sql/"
            "extracted_checkpoint.json",
            "bird_extraction",
            bird_stage_rows,
        ),
        (
            root
            / "intent_construction/retrospective_expansion/counterfactual/output/"
            "bird_sql/argument_counterfactual_checkpoint.json",
            "bird_counterfactual",
            bird_stage_rows,
        ),
        (
            root
            / "intent_construction/retrospective_expansion/predecessor/output/"
            "bird_sql/predecessor_checkpoint.json",
            "bird_predecessor",
            bird_predecessor_rows,
        ),
    )
    for checkpoint_path, stage, rows in bird_checkpoints:
        _write_json(
            checkpoint_path,
            {
                "schema_version": 1,
                "stage": stage,
                "model": EXPECTED_MODEL,
                "required_task_ids": bird_ids,
                "processed_ids": bird_ids,
                "results": rows,
                "failures": {},
                "complete": True,
            },
        )
    _write_json(root / "final_dataset/bird_sql_final.json", bird_predecessor_rows)

    browse_single_path = (
        experiments
        / "fully_specified/browsecomp_plus_n100/"
        "kimi-k2.6_naturalized_reasoning-medium_force-final.json"
    )
    browse_evolve_path = (
        experiments
        / "combined_independent/browsecomp_plus_n100/"
        "kimi-k2.6_t7_g2_p2_naturalized_reasoning-medium_force-final.json"
    )
    browse_single = _browse_results(browse_ids, turns=1, correct=70)
    browse_evolve = _browse_results(browse_ids, turns=7, correct=49)
    _write_json(browse_single_path, browse_single)
    _write_json(browse_evolve_path, browse_evolve)
    _browse_checkpoints(
        browse_single_path,
        browse_ids,
        browse_single,
        scenario="fully_specified",
        turns=1,
        revisions=0,
        switches=0,
    )
    _browse_checkpoints(
        browse_evolve_path,
        browse_ids,
        browse_evolve,
        scenario="combined",
        turns=7,
        revisions=2,
        switches=2,
    )

    bird_run = root / "reproduction/runs/bird-sql-kimi-k2.6"
    _write_jsonl(bird_run / "usage.jsonl", [_usage("2026-08-15T01:00:00Z", 0.10)])
    _write_jsonl(
        bird_run / "evaluation_usage.jsonl",
        [_usage("2026-08-15T01:30:00Z", 0.15)],
    )
    browse_run = root / "reproduction/runs/browsecomp-plan-a-n100"
    _write_json(
        browse_run / "local_audit.json",
        {
            "model": EXPECTED_MODEL,
            "sample_count": 100,
            "stage_counts": {"stage1": 100, "stage2": 100, "stage3": 100},
            "independence_verification": True,
            "task_ids": browse_ids,
        },
    )
    _write_jsonl(browse_run / "llm_usage.jsonl", [_usage("2026-08-15T02:00:00Z", 0.20)])
    _write_jsonl(
        browse_run / "evaluation_usage.jsonl",
        [_usage("2026-08-15T02:30:00Z", 0.25)],
    )

    swe_construction = root / "reproduction/runs/swe-bench-verified-kimi-k2.6"
    _write_json(
        swe_construction / "manifest.json",
        {
            "status": "complete",
            "model": EXPECTED_MODEL,
            "resolved_model": EXPECTED_MODEL,
            "reasoning_effort": "medium",
            "output_token_limit": None,
            "cost_hard_cap_usd": None,
            "final_coverage": 50,
        },
    )
    _write_jsonl(
        swe_construction / "usage.jsonl", [_usage("2026-08-15T03:00:00Z", 0.30)]
    )

    swe_run = root / "evaluation/swe_runs/kimi-k2.6"
    scenarios = [
        {
            "name": "single",
            "turns": 1,
            "revisions": 0,
            "switches": 0,
            "tool_call_limit_per_turn": 200,
        },
        {
            "name": "evolve",
            "turns": 7,
            "revisions": 2,
            "switches": 2,
            "tool_call_limit_per_turn": 200,
        },
    ]
    _write_json(
        swe_run / "manifest.json",
        {
            "schema_version": 2,
            "status": "complete",
            "created_at": "2026-08-15T03:00:00Z",
            "updated_at": "2026-08-15T05:00:00Z",
            "models": {"requested": [EXPECTED_MODEL], "resolved": [EXPECTED_MODEL]},
            "dataset": {"expected_count": 50, "task_ids": swe_ids},
            "runtime": {
                "reasoning_effort": "medium",
                "output_token_limit": None,
                "cost_hard_cap_usd": None,
            },
            "scenarios": scenarios,
            "scenario_runs": {
                "single": {"status": "complete", "expected": 50, "completed": 50},
                "evolve": {"status": "complete", "expected": 50, "completed": 50},
            },
        },
    )
    _write_json(swe_run / "results/single.json", _swe_results(swe_ids, scenario="single", turns=1, correct=20))
    _write_json(swe_run / "results/evolve.json", _swe_results(swe_ids, scenario="evolve", turns=7, correct=10))
    _write_jsonl(swe_run / "usage.jsonl", [_usage("2026-08-15T05:00:00Z", 0.40)])

    layout = ReportLayout.defaults(
        root,
        output_json=root / "reproduction/report.json",
        output_html=root / "reproduction/report.html",
    )
    return layout, browse_single_path


def test_complete_report_is_strict_aggregate_and_redacted(tmp_path: Path) -> None:
    layout, browse_source = _complete_layout(tmp_path)
    before = browse_source.read_bytes()

    report = build_report(
        layout,
        generated_at=datetime(2026, 8, 15, 6, tzinfo=timezone.utc),
    )
    write_report(layout, report)

    assert report["status"] == "complete"
    assert report["coverage"] == {
        "bird_sql": {"completed": 100, "expected": 100},
        "browsecomp_plus": {"completed": 100, "expected": 100},
        "swe_bench_verified": {"completed": 50, "expected": 50},
    }
    assert report["benchmarks"]["bird_sql"]["single"]["accuracy"] == pytest.approx(0.8)
    assert report["benchmarks"]["bird_sql"]["evolve"]["accuracy"] == pytest.approx(0.6)
    assert report["benchmarks"]["bird_sql"]["delta"]["relative"] == pytest.approx(-0.25)
    assert report["totals"]["usage"]["confirmed_cost_usd"] == pytest.approx(1.4)
    assert report["totals"]["usage"]["calls"] == 6
    assert browse_source.read_bytes() == before

    rendered = layout.output_json.read_text(encoding="utf-8") + layout.output_html.read_text(
        encoding="utf-8"
    )
    assert "DO-NOT-EXPORT-PLAINTEXT-QUERY" not in rendered
    assert "DO-NOT-EXPORT-GOLD-DOCUMENT" not in rendered
    assert "sk-do-not-export" not in rendered

    protected_layout = ReportLayout.defaults(
        layout.repo_root,
        output_json=browse_source,
        output_html=layout.repo_root / "reproduction/other-report.html",
    )
    with pytest.raises(FinalizationError, match="outside experiment"):
        write_report(protected_layout, report)


def test_copied_usage_ledger_is_counted_once(tmp_path: Path) -> None:
    layout, _ = _complete_layout(tmp_path)
    canonical = layout.browse_run_dir / "llm_usage.jsonl"
    copied = layout.browse_run_dir / "usage/llm_usage.jsonl"
    copied.parent.mkdir(parents=True, exist_ok=True)
    copied.write_bytes(canonical.read_bytes())

    report = build_report(
        layout,
        generated_at=datetime(2026, 8, 15, 6, tzinfo=timezone.utc),
    )

    assert report["benchmarks"]["browsecomp_plus"]["usage"]["calls"] == 2
    assert report["benchmarks"]["browsecomp_plus"]["usage"][
        "confirmed_cost_usd"
    ] == pytest.approx(0.45)
    assert report["totals"]["usage"]["calls"] == 6
    assert report["totals"]["usage"]["confirmed_cost_usd"] == pytest.approx(1.4)


def test_fetched_browsecomp_pair_requires_exact_policy_bound_coverage(
    tmp_path: Path,
) -> None:
    layout, single_path = _complete_layout(tmp_path)
    task_ids = json.loads(
        (
            layout.repo_root
            / "intent_construction/eval_indices/browsecomp_plus_task_ids.json"
        ).read_text(encoding="utf-8")
    )["task_ids"]
    evolve_path = (
        layout.experiments_dir
        / "combined_independent/browsecomp_plus_n100/"
        "kimi-k2.6_t7_g2_p2_naturalized_reasoning-medium_force-final.json"
    )

    audit = validate_browsecomp_evaluation_artifacts(
        repo_root=layout.repo_root,
        task_ids=task_ids,
        single_path=single_path,
        evolve_path=evolve_path,
    )

    assert audit["status"] == "complete"
    assert audit["scenarios"]["single"]["completed"] == 100
    assert audit["scenarios"]["evolve"]["checkpoints"] == 100

    next(Path(f"{evolve_path}.checkpoints").glob("*.json")).unlink()
    with pytest.raises(FinalizationError, match="checkpoint"):
        validate_browsecomp_evaluation_artifacts(
            repo_root=layout.repo_root,
            task_ids=task_ids,
            single_path=single_path,
            evolve_path=evolve_path,
        )


def test_formal_mode_fails_closed_but_preview_writes_status(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    indices = root / "intent_construction/eval_indices"
    _write_json(indices / "bird_sql_task_ids.json", {"num_samples": 100, "task_ids": _ids("bird", 100)})
    _write_json(
        indices / "browsecomp_plus_task_ids.json",
        {"num_samples": 100, "task_ids": _ids("browse", 100)},
    )
    _write_json(indices / "swe_bench_verified_task_ids.json", {"task_ids": _ids("swe", 50)})
    _write_json(root / "reproduction/config/paper_remaining_kimi_k2_6.json", _policy())
    layout = ReportLayout.defaults(
        root,
        output_json=root / "preview.json",
        output_html=root / "preview.html",
    )

    with pytest.raises(FinalizationError, match="failed closed"):
        build_report(layout)
    assert not layout.output_json.exists()
    assert not layout.output_html.exists()

    preview = build_report(layout, allow_incomplete=True)
    write_report(layout, preview)
    assert preview["status"] == "incomplete"
    assert preview["coverage"]["bird_sql"] == {"completed": 0, "expected": 100}
    assert preview["issues"]
    assert layout.output_json.is_file()
    assert layout.output_html.is_file()


def test_wrong_reasoning_checkpoint_is_not_accepted(tmp_path: Path) -> None:
    layout, browse_source = _complete_layout(tmp_path)
    checkpoint = next(Path(f"{browse_source}.checkpoints").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["policy"]["reasoning_effort"] = "low"
    _write_json(checkpoint, payload)

    preview = build_report(layout, allow_incomplete=True)
    assert preview["status"] == "incomplete"
    assert any("model/reasoning/limit" in issue for issue in preview["issues"])
    with pytest.raises(FinalizationError):
        build_report(layout)
