#!/usr/bin/env python3
"""Strict execution-based evaluation for the fixed BIRD-SQL reproduction."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from evaluation.common.sql_evaluator import (
    execute_sql,
    extract_sql_from_response,
    stringify_result,
)
from evaluation.runners.run_experiment import evaluate_sample
from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import (
    DEFAULT_TASK_IDS_PATH,
    REQUIRED_MODEL,
    REPO_ROOT,
    BirdReproductionError,
    TaskCheckpoint,
    assert_required_model,
    atomic_write_json,
    load_published_task_ids,
    read_json,
    resolve_db_path,
    validate_stage_rows,
)
from situated_simulation.user_simulation import EvolvingIntent


EXPERIMENTS_DIR = REPO_ROOT / "evaluation" / "experiments"
PAPER_SETTINGS = {(1, 0, 0), (7, 2, 2)}
BIRD_MODEL_EXECUTION_TIMEOUT_SECONDS = 30.0
BIRD_GOLD_EXECUTION_TIMEOUT_SECONDS = 120.0
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def assert_evaluation_runtime_policy() -> None:
    """Fail before loading checkpoints if the fixed run policy is not active."""
    if os.environ.get("LLM_LOCKED_MODEL", "").strip() != REQUIRED_MODEL:
        raise BirdReproductionError(
            f"BIRD evaluation requires LLM_LOCKED_MODEL={REQUIRED_MODEL}"
        )
    if os.environ.get("LLM_REASONING_EFFORT", "").strip() != "medium":
        raise BirdReproductionError(
            "BIRD evaluation requires LLM_REASONING_EFFORT=medium"
        )
    if os.environ.get("REASONING_EFFORT", "").strip():
        raise BirdReproductionError(
            "BIRD evaluation forbids the legacy REASONING_EFFORT override"
        )
    if os.environ.get("LLM_DISABLE_OUTPUT_LIMITS", "").strip().lower() not in (
        _TRUE_ENV_VALUES
    ):
        raise BirdReproductionError(
            "BIRD evaluation requires LLM_DISABLE_OUTPUT_LIMITS=1"
        )
    configured_limits = [
        name
        for name in (
            "LLM_DEFAULT_MAX_OUTPUT_TOKENS",
            "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS",
        )
        if os.environ.get(name, "").strip()
    ]
    if configured_limits:
        raise BirdReproductionError(
            "BIRD evaluation forbids output-token defaults: "
            + ", ".join(configured_limits)
        )
    if os.environ.get("LLM_COST_HARD_CAP_USD", "").strip():
        raise BirdReproductionError(
            "BIRD evaluation records usage but forbids LLM_COST_HARD_CAP_USD"
        )
    if os.environ.get("LLM_REQUIRE_USAGE_ACCOUNTING", "").strip().lower() not in (
        _TRUE_ENV_VALUES
    ):
        raise BirdReproductionError(
            "BIRD evaluation requires LLM_REQUIRE_USAGE_ACCOUNTING=1"
        )
    if not os.environ.get("LLM_USAGE_LEDGER_PATH", "").strip():
        raise BirdReproductionError(
            "BIRD evaluation requires LLM_USAGE_LEDGER_PATH"
        )


def official_results_equal(result_a: Any, result_b: Any) -> bool:
    """Match BIRD's set(rows) semantics.

    Row order and duplicate rows are ignored.  Tuple position is untouched, so
    column order remains significant.
    """
    if not result_a.success or not result_b.success:
        return False
    if result_a.rows is None or result_b.rows is None:
        return result_a.rows is None and result_b.rows is None
    return set(map(tuple, result_a.rows)) == set(map(tuple, result_b.rows))


def _bird_verifier_policy() -> dict[str, Any]:
    return {
        "type": "execution_set_rows",
        "row_order_sensitive": False,
        "duplicate_rows_sensitive": False,
        "column_order_sensitive": True,
        "timeout_seconds": BIRD_MODEL_EXECUTION_TIMEOUT_SECONDS,
        "gold_timeout_seconds": BIRD_GOLD_EXECUTION_TIMEOUT_SECONDS,
        "llm_judge": False,
    }


def _upgrade_checkpoint_verifier_policy(checkpoint: TaskCheckpoint) -> None:
    """Add the explicit trusted-gold timeout to resumable legacy rows."""
    current = _bird_verifier_policy()
    legacy = dict(current)
    legacy.pop("gold_timeout_seconds")
    changed = False
    for row in checkpoint.results_by_id.values():
        if row.get("bird_verifier") == legacy:
            row["bird_verifier"] = dict(current)
            changed = True
    if changed:
        checkpoint.flush()


def evaluate_bird_response(
    response: str,
    gold_sql: str,
    db_path: str | Path,
) -> dict[str, Any]:
    """Execute one model SQL response using the native BIRD verifier rule."""
    if not isinstance(response, str) or not response.strip():
        raise BirdReproductionError("BIRD evaluation received an empty response")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise BirdReproductionError("BIRD evaluation is missing gold SQL")
    resolved_db = resolve_db_path(db_path)
    if not resolved_db.exists():
        raise BirdReproductionError(f"BIRD database does not exist: {resolved_db}")

    # A turn gold that SQLite cannot execute (constructed partial SQL can
    # overflow int64 aggregates or time out on unrevealed-join Cartesian
    # scans) is graded, not fatal: official_results_equal already defines
    # either-side execution failure as non-equal, matching how model-side
    # SQL errors are handled. The run therefore continues and the failed
    # gold is recorded for audit via gold_error.
    gold_result = execute_sql(
        resolved_db,
        gold_sql,
        timeout=BIRD_GOLD_EXECUTION_TIMEOUT_SECONDS,
    )
    gold_error = None if gold_result.success else gold_result.error
    gold_answer = stringify_result(gold_result) if gold_result.success else None
    model_sql = extract_sql_from_response(response)
    if model_sql is None:
        return {
            "execution_match": False,
            "model_sql_valid": False,
            "model_sql": "",
            "gold_sql": gold_sql,
            "model_answer": None,
            "gold_answer": gold_answer,
            "gold_error": gold_error,
            "error": "could_not_extract_sql",
        }
    model_result = execute_sql(
        resolved_db,
        model_sql,
        timeout=BIRD_MODEL_EXECUTION_TIMEOUT_SECONDS,
    )
    if not model_result.success:
        return {
            "execution_match": False,
            "model_sql_valid": False,
            "model_sql": model_sql,
            "gold_sql": gold_sql,
            "model_answer": None,
            "gold_answer": gold_answer,
            "gold_error": gold_error,
            "error": model_result.error,
        }
    return {
        "execution_match": official_results_equal(model_result, gold_result),
        "model_sql_valid": True,
        "model_sql": model_sql,
        "gold_sql": gold_sql,
        "model_answer": stringify_result(model_result),
        "gold_answer": gold_answer,
        "gold_error": gold_error,
        "error": None,
    }


def _scenario_name(turns: int, revisions: int, switches: int) -> str:
    return "fully_specified" if (turns, revisions, switches) == (1, 0, 0) else "combined_independent"


def _output_path(dataset_name: str, turns: int, revisions: int, switches: int) -> Path:
    directory = EXPERIMENTS_DIR / _scenario_name(turns, revisions, switches) / dataset_name
    filename = (
        f"{REQUIRED_MODEL}.json"
        if turns == 1
        else f"{REQUIRED_MODEL}_t{turns}_g{switches}_p{revisions}.json"
    )
    return directory / filename


def _checkpoint_path(output_path: Path, override: str | None) -> Path:
    return (
        Path(override)
        if override
        else output_path.with_name(f"{output_path.stem}_checkpoint.json")
    )


def _stage_name(turns: int, revisions: int, switches: int) -> str:
    return f"bird_evaluation_t{turns}_g{switches}_p{revisions}"


def _database_task_ids(
    source_rows: Sequence[Mapping[str, Any]],
    db_id: str,
) -> list[str]:
    task_ids = [row["task_id"] for row in source_rows if row.get("db_id") == db_id]
    if not task_ids:
        raise BirdReproductionError(f"{db_id!r} is not a published BIRD database")
    return task_ids


def _pending_database_ids(
    source_rows: Sequence[Mapping[str, Any]],
    pending_ids: Sequence[str],
) -> list[str]:
    pending = set(pending_ids)
    ordered: list[str] = []
    seen: set[str] = set()
    for row in source_rows:
        if row["task_id"] not in pending:
            continue
        db_id = row["db_id"]
        if db_id not in seen:
            seen.add(db_id)
            ordered.append(db_id)
    return ordered


def _preflight_sample(
    sample: Any,
    *,
    turns: int,
    revisions: int,
    switches: int,
) -> None:
    user_turns = [turn for turn in sample.turns if turn.get("role") == "user"]
    if len(user_turns) != turns or any(
        not isinstance(turn.get("content"), str) or not turn["content"].strip()
        for turn in user_turns
    ):
        raise BirdReproductionError(
            f"{sample.task_id} rendered {len(user_turns)}/{turns} complete user turns"
        )
    metadata = sample.metadata
    if metadata.get("num_counterfactual_arguments") != revisions:
        raise BirdReproductionError(
            f"{sample.task_id} rendered the wrong number of revisions"
        )
    if metadata.get("num_predecessor_functions") != switches:
        raise BirdReproductionError(
            f"{sample.task_id} rendered the wrong number of function switches"
        )
    per_turn_gold = metadata.get("per_turn_gold")
    if not isinstance(per_turn_gold, list) or len(per_turn_gold) != turns:
        raise BirdReproductionError(
            f"{sample.task_id} has incomplete per-turn execution gold"
        )
    if any(
        not isinstance(entry, Mapping) or not str(entry.get("sql", "")).strip()
        for entry in per_turn_gold
    ):
        raise BirdReproductionError(
            f"{sample.task_id} has an empty per-turn gold SQL entry"
        )
    if not resolve_db_path(metadata.get("db_path", "")).exists():
        raise BirdReproductionError(
            f"{sample.task_id} database is missing: {metadata.get('db_path')!r}"
        )


def _grade_completed_result(
    result: object,
    sample: Any,
    *,
    turns: int,
    model: str,
) -> dict[str, Any]:
    if not isinstance(result, dict) or result.get("task_id") != sample.task_id:
        raise BirdReproductionError(
            f"{sample.task_id} returned an incomplete or misidentified result"
        )
    if result.get("success") is not True or result.get("error") is not None:
        raise BirdReproductionError(
            f"{sample.task_id} model call failed: {result.get('error')!r}"
        )
    responses = result.get("decoding")
    if not isinstance(responses, list) or len(responses) != turns:
        raise BirdReproductionError(
            f"{sample.task_id} returned {len(responses) if isinstance(responses, list) else 0}/"
            f"{turns} responses"
        )
    if any(not isinstance(response, str) or not response.strip() for response in responses):
        raise BirdReproductionError(f"{sample.task_id} returned an empty response")

    metadata = sample.metadata
    per_turn_gold = metadata["per_turn_gold"]
    graded_turns: list[dict[str, Any]] = []
    for turn_index, (response, gold) in enumerate(zip(responses, per_turn_gold)):
        execution = evaluate_bird_response(
            response,
            gold["sql"],
            metadata["db_path"],
        )
        graded_turns.append(
            {
                "turn": turn_index,
                "correct": execution["execution_match"],
                "gold_sql": gold["sql"],
                "gold_answer": execution["gold_answer"],
                "model_sql": execution["model_sql"],
                "model_answer": execution["model_answer"],
                "model_sql_valid": execution["model_sql_valid"],
                "gold_error": execution["gold_error"],
                "error": execution["error"],
            }
        )
    final_execution = graded_turns[-1]
    materialized = dict(result)
    materialized["prediction"] = final_execution["model_sql"]
    materialized["correct"] = final_execution["correct"]
    materialized["per_turn_results"] = graded_turns
    materialized["model_name"] = model
    materialized["bird_verifier"] = _bird_verifier_policy()
    assert_required_model(model, context="completed BIRD evaluation result model")
    return materialized


def validate_evaluation_results(
    payload: object,
    *,
    required_ids: Sequence[str],
    turns: int,
    model: str,
) -> dict[str, dict[str, Any]]:
    assert_required_model(model, context="BIRD evaluation artifact model")
    if not isinstance(payload, dict):
        raise BirdReproductionError("BIRD evaluation output must be a task-ID mapping")
    missing = [task_id for task_id in required_ids if task_id not in payload]
    extras = [task_id for task_id in payload if task_id not in set(required_ids)]
    if missing or extras or len(payload) != len(required_ids):
        raise BirdReproductionError(
            f"BIRD evaluation coverage failure: missing={missing[:5]}, extras={extras[:5]}"
        )
    ordered: dict[str, dict[str, Any]] = {}
    for task_id in required_ids:
        row = payload[task_id]
        if not isinstance(row, dict) or row.get("task_id") != task_id:
            raise BirdReproductionError(f"malformed evaluation result for {task_id}")
        assert_required_model(row.get("model_name"), context=f"{task_id} result model")
        responses = row.get("decoding")
        if (
            row.get("success") is not True
            or row.get("error") is not None
            or not isinstance(responses, list)
            or len(responses) != turns
            or any(
                not isinstance(response, str) or not response.strip()
                for response in responses
            )
        ):
            raise BirdReproductionError(f"incomplete evaluation result for {task_id}")
        if row.get("bird_verifier") != _bird_verifier_policy():
            raise BirdReproductionError(f"non-native verifier result for {task_id}")
        per_turn = row.get("per_turn_results")
        if not isinstance(per_turn, list) or len(per_turn) != turns:
            raise BirdReproductionError(
                f"incomplete per-turn verifier results for {task_id}"
            )
        for turn_index, turn in enumerate(per_turn):
            if (
                not isinstance(turn, dict)
                or turn.get("turn") != turn_index
                or not isinstance(turn.get("correct"), bool)
                or not isinstance(turn.get("model_sql_valid"), bool)
                or not isinstance(turn.get("gold_sql"), str)
                or not turn["gold_sql"].strip()
                or not isinstance(turn.get("model_sql"), str)
            ):
                raise BirdReproductionError(
                    f"malformed turn {turn_index} verifier result for {task_id}"
                )
        final_turn = per_turn[-1]
        if (
            not isinstance(row.get("correct"), bool)
            or row["correct"] != final_turn["correct"]
            or row.get("prediction") != final_turn["model_sql"]
        ):
            raise BirdReproductionError(
                f"final verifier result mismatch for {task_id}"
            )
        ordered[task_id] = row
    return ordered


def _finalize_completed_checkpoint(
    checkpoint: TaskCheckpoint,
    *,
    output_path: Path,
    required_ids: Sequence[str],
    turns: int,
    model: str,
) -> dict[str, dict[str, Any]] | None:
    """Recover the aggregate after the last task checkpoint was persisted."""
    if checkpoint.pending_ids:
        return None
    result_mapping = {row["task_id"]: row for row in checkpoint.results}
    ordered = validate_evaluation_results(
        result_mapping,
        required_ids=required_ids,
        turns=turns,
        model=model,
    )
    checkpoint.mark_complete()
    atomic_write_json(output_path, ordered)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset_name", default="bird_sql_n100")
    parser.add_argument("--model", default=REQUIRED_MODEL)
    parser.add_argument("--task_ids_file", default=str(DEFAULT_TASK_IDS_PATH))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_turns", type=int, required=True)
    parser.add_argument("--num_revisions", type=int, required=True)
    parser.add_argument("--num_switches", type=int, required=True)
    parser.add_argument("--ordering", default="interleaved")
    parser.add_argument("--db_id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list_pending_databases", action="store_true")
    args = parser.parse_args()

    assert_required_model(args.model, context="BIRD evaluation CLI model")
    assert_evaluation_runtime_policy()
    setting = (args.num_turns, args.num_revisions, args.num_switches)
    if setting not in PAPER_SETTINGS:
        raise BirdReproductionError(
            f"unsupported BIRD paper setting {setting}; expected one of {sorted(PAPER_SETTINGS)}"
        )
    if args.ordering != "interleaved":
        raise BirdReproductionError("published BIRD evolution uses interleaved ordering")
    if args.num_workers < 1:
        raise BirdReproductionError("num_workers must be positive")
    required_ids = load_published_task_ids(args.task_ids_file)
    source_rows = validate_stage_rows(
        read_json(args.data_path),
        stage="bird_evaluation_input",
        required_ids=required_ids,
        require_model=True,
        min_predecessors=2,
    )
    if [row["task_id"] for row in source_rows] != required_ids:
        raise BirdReproductionError("BIRD evaluation input order differs from published order")

    output_path = _output_path(
        args.dataset_name,
        args.num_turns,
        args.num_revisions,
        args.num_switches,
    )
    if output_path.exists():
        validate_evaluation_results(
            read_json(output_path),
            required_ids=required_ids,
            turns=args.num_turns,
            model=args.model,
        )
        if not args.list_pending_databases:
            print(f"complete BIRD result already exists: {output_path}")
        return
    cp_path = _checkpoint_path(output_path, args.checkpoint)
    checkpoint = TaskCheckpoint(
        cp_path,
        stage=_stage_name(args.num_turns, args.num_revisions, args.num_switches),
        required_ids=required_ids,
        model=args.model,
        resume=args.resume or args.list_pending_databases,
    )
    _upgrade_checkpoint_verifier_policy(checkpoint)
    if args.list_pending_databases:
        _finalize_completed_checkpoint(
            checkpoint,
            output_path=output_path,
            required_ids=required_ids,
            turns=args.num_turns,
            model=args.model,
        )
        for db_id in _pending_database_ids(source_rows, checkpoint.pending_ids):
            print(db_id)
        return
    if args.db_id is None:
        raise BirdReproductionError(
            "--db_id is required for the low-disk BIRD evaluation"
        )

    database_ids = _database_task_ids(source_rows, args.db_id)
    pending_id_set = set(checkpoint.pending_ids)
    database_pending_ids = [
        task_id for task_id in database_ids if task_id in pending_id_set
    ]

    samples_by_id: dict[str, Any] = {}
    if database_pending_ids:
        # Build all 100 samples so deterministic prefix cycling matches a
        # monolithic run. Only the current database is preflighted or evaluated.
        simulator = EvolvingIntent(
            data_path=args.data_path,
            mode="eval",
            domain="sql",
            num_turns=args.num_turns,
            num_revisions=args.num_revisions,
            num_switches=args.num_switches,
            ordering=args.ordering,
            task_ids=required_ids,
            naturalizer_model=None,
            include_evidence=True,
        )
        samples = list(simulator)
        if [sample.task_id for sample in samples] != required_ids:
            found = [sample.task_id for sample in samples]
            raise BirdReproductionError(
                f"simulator coverage/order mismatch: expected 100, found {len(found)}"
            )
        samples_by_id = {sample.task_id: sample for sample in samples}
        for task_id in database_ids:
            _preflight_sample(
                samples_by_id[task_id],
                turns=args.num_turns,
                revisions=args.num_revisions,
                switches=args.num_switches,
            )
    pending = [samples_by_id[task_id] for task_id in database_pending_ids]

    def run_one(sample: Any) -> dict[str, Any]:
        raw = evaluate_sample(
            sample,
            args.model,
            temperature=None,
            max_tokens=None,
            reasoning_effort=None,
        )
        return _grade_completed_result(
            raw,
            sample,
            turns=args.num_turns,
            model=args.model,
        )

    print(
        f"Evaluating BIRD {setting} on {args.db_id}: "
        f"{len(database_ids) - len(pending)} complete, "
        f"{len(pending)} pending, model={args.model}"
    )
    if args.num_workers > 1 and pending:
        executor = ThreadPoolExecutor(max_workers=args.num_workers)
        futures = {executor.submit(run_one, sample): sample for sample in pending}
        first_failure: BaseException | None = None
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="bird-eval"):
                sample = futures[future]
                try:
                    result = future.result()
                except CancelledError:
                    continue
                except BaseException as exc:
                    checkpoint.record_failure(sample.task_id, exc)
                    if first_failure is None:
                        first_failure = exc
                        for other in futures:
                            if other is not future:
                                other.cancel()
                    continue
                checkpoint.record_success(result)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if first_failure is not None:
            raise first_failure
    else:
        for sample in tqdm(pending, desc="bird-eval"):
            try:
                checkpoint.record_success(run_one(sample))
            except BaseException as exc:
                checkpoint.record_failure(sample.task_id, exc)
                raise

    database_mapping = {
        task_id: checkpoint.results_by_id[task_id] for task_id in database_ids
    }
    validate_evaluation_results(
        database_mapping,
        required_ids=database_ids,
        turns=args.num_turns,
        model=args.model,
    )
    if checkpoint.pending_ids:
        print(
            f"BIRD database complete: {args.db_id} "
            f"({len(checkpoint.processed_ids)}/{len(required_ids)} tasks overall)"
        )
        return

    ordered = _finalize_completed_checkpoint(
        checkpoint,
        output_path=output_path,
        required_ids=required_ids,
        turns=args.num_turns,
        model=args.model,
    )
    assert ordered is not None
    correct = sum(1 for row in ordered.values() if row.get("correct") is True)
    print(
        f"BIRD evaluation complete: {correct}/{len(ordered)} "
        f"({correct / len(ordered):.1%}) -> {output_path}"
    )


if __name__ == "__main__":
    main()
