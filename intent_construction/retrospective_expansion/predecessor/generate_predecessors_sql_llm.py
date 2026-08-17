"""
SQL Function Predecessor — LLM Executor (Stage 4b of BIRD-SQL redesign).

Takes argument-predecessor BIRD-SQL samples and, for each one, asks an LLM
(default ``gpt-5.1``) to rewrite the SQL according to multiple
deterministically-planned ``(change_set, preserve_set)`` plans (Stage 4a in
``function_change_planner.py``).

Every candidate is validated by:
  1. SQLite parse round-trip
  2. Byte-identical preservation of ``preserve_set ∪ {WHERE, HAVING, LIMIT, FROM}``
  3. Meaningful difference on every clause in ``change_set``
  4. Successful execution against the actual DB
  5. A different result-set from the gold SQL
  6. Deduplication against earlier accepted candidates of the same sample

Naturalization (SQL → episodic follow-up text) is delegated to
``sql_naturalizer.py`` and filled in as ``predecessor_function``.

Usage:
    python generate_predecessors_sql_llm.py \
        --input  ../counterfactual/output/bird_sql/argument_counterfactual.json \
        --output output/bird_sql/function_counterfactual_llm.json \
        --num_predecessors 3 \
        --model gpt-5.1 \
        --num_workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import traceback
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import expressions as exp
from tqdm import tqdm

# Local module wiring
_THIS = Path(__file__).resolve().parent



from intent_construction.intent_extraction.dataset_impl.bird_sql.sql_parser import parse_sql  # noqa: E402
from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import (  # noqa: E402
    execute_sql,
    validate_result,
    compare_results,
    get_schema_text,
)
from intent_construction.intent_extraction.core.llm_utils import generate_json, load_prompt, populate_prompt  # noqa: E402
from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import (  # noqa: E402
    DEFAULT_TASK_IDS_PATH,
    REQUIRED_MODEL,
    BirdReproductionError,
    TaskCheckpoint,
    assert_required_model,
    atomic_write_json,
    checkpoint_path_for,
    load_published_task_ids,
    read_json,
    resolve_db_path,
    validate_stage_rows,
)
from intent_construction.retrospective_expansion.predecessor.function_change_planner import (  # noqa: E402
    GOAL_CLAUSES,
    FunctionChangePlan,
    enumerate_plans,
    select_diverse_plans,
)
from intent_construction.retrospective_expansion.predecessor.sql_naturalizer import naturalize_followup  # noqa: E402


PROMPT_PATH = _THIS / "prompts" / "generate_predecessor_sql.txt"


# -----------------------------------------------------------------------------
# Clause extraction (for byte-identical comparison)
# -----------------------------------------------------------------------------

_CLAUSE_TO_ARG: dict[str, str] = {
    "SELECT":   "expressions",
    "FROM":     "from_",
    "WHERE":    "where",
    "GROUP_BY": "group",
    "HAVING":   "having",
    "ORDER_BY": "order",
    "LIMIT":    "limit",
    "JOIN":     "joins",
}


def _normalize(s: str) -> str:
    """Whitespace + case canonicalization for cross-formatting comparison."""
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _clause_canonical(select_node: exp.Select, clause: str) -> str:
    """Return the canonical SQL string for one clause of a SELECT node."""
    if clause == "SELECT":
        exprs = select_node.expressions or []
        return _normalize(", ".join(e.sql(dialect="sqlite") for e in exprs))
    if clause == "JOIN":
        joins = select_node.args.get("joins") or []
        return _normalize(" ".join(j.sql(dialect="sqlite") for j in joins))
    arg = _CLAUSE_TO_ARG.get(clause)
    node = select_node.args.get(arg) if arg else None
    if node is None:
        return ""
    return _normalize(node.sql(dialect="sqlite"))


def _parse_select(sql: str) -> exp.Select | None:
    """Parse SQL and return the top-level SELECT node (or None on failure)."""
    try:
        node = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return None
    if isinstance(node, exp.Select):
        return node
    inner = node.find(exp.Select) if hasattr(node, "find") else None
    return inner


def diff_clauses(gold_sql: str, new_sql: str) -> dict[str, bool]:
    """Compare each clause; True = the clauses differ (canonical compare)."""
    gold = _parse_select(gold_sql)
    new = _parse_select(new_sql)
    if gold is None or new is None:
        return {c: True for c in _CLAUSE_TO_ARG}
    return {
        clause: _clause_canonical(gold, clause) != _clause_canonical(new, clause)
        for clause in _CLAUSE_TO_ARG
    }


def _result_is_ordered(sql: str) -> bool:
    select_node = _parse_select(sql)
    return bool(select_node is not None and select_node.args.get("order") is not None)


def _results_semantically_equal(
    result_a,
    result_b,
    *,
    ordered_a: bool,
    ordered_b: bool,
) -> bool:
    """Compare output schema, orderedness, and values for candidate deduping."""
    if ordered_a != ordered_b:
        return False
    columns_a = tuple(_normalize(column) for column in (result_a.columns or []))
    columns_b = tuple(_normalize(column) for column in (result_b.columns or []))
    if columns_a != columns_b:
        return False
    return compare_results(
        result_a,
        result_b,
        order_sensitive=ordered_a,
        column_order_sensitive=True,
    )


# -----------------------------------------------------------------------------
# Per-candidate validation
# -----------------------------------------------------------------------------

def _validate_candidate(
    new_sql: str,
    plan: FunctionChangePlan,
    gold_sql: str,
    db_path: str,
    gold_result,
    seen_hashes: set[str],
    seen_results: list[tuple[Any, bool]],
    sql_timeout: int,
) -> tuple[bool, str, Any | None]:
    """Return (ok, failure_reason, counterfactual_result_or_None).

    failure_reason is one of:
        "ok", "parse_fail", "preserve_violation:<clause>",
        "change_violation:<clause>", "exec_error", "empty_result",
        "same_result", "duplicate"
    """
    parsed = parse_sql(new_sql)
    if not parsed.parseable:
        return False, "parse_fail", None

    diffs = diff_clauses(gold_sql, new_sql)

    # Preserve check
    must_preserve = (
        set(GOAL_CLAUSES) - set(plan.change_set)
    ) | set(plan.always_preserved)
    for clause in must_preserve:
        if diffs.get(clause, False):
            return False, f"preserve_violation:{clause}", None

    # Change check
    for clause in plan.change_set:
        if not diffs.get(clause, False):
            return False, f"change_violation:{clause}", None

    # Execute
    counterfactual_result = execute_sql(db_path, new_sql, timeout=sql_timeout)
    if not validate_result(counterfactual_result):
        return False, "empty_result", None

    candidate_ordered = _result_is_ordered(new_sql)
    if _results_semantically_equal(
        gold_result,
        counterfactual_result,
        ordered_a=_result_is_ordered(gold_sql),
        ordered_b=candidate_ordered,
    ):
        return False, "same_result", None

    if any(
        _results_semantically_equal(
            prior_result,
            counterfactual_result,
            ordered_a=prior_ordered,
            ordered_b=candidate_ordered,
        )
        for prior_result, prior_ordered in seen_results
    ):
        return False, "duplicate_result", None

    h = hashlib.md5(_normalize(new_sql).encode()).hexdigest()
    if h in seen_hashes:
        return False, "duplicate", None
    seen_hashes.add(h)
    seen_results.append((counterfactual_result, candidate_ordered))

    return True, "ok", counterfactual_result


# -----------------------------------------------------------------------------
# Result extraction helper
# -----------------------------------------------------------------------------

def _extract_answer(result) -> str:
    if result.rows and len(result.rows) > 0:
        first_row = result.rows[0]
        if len(first_row) == 1:
            val = first_row[0]
            return str(val) if val is not None else "NULL"
        return str(first_row)
    return ""


# -----------------------------------------------------------------------------
# LLM call wrapper
# -----------------------------------------------------------------------------

_PROMPT_TEMPLATE: str | None = None


def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = load_prompt(str(PROMPT_PATH))
    return _PROMPT_TEMPLATE


def _build_messages(
    schema_text: str,
    question: str,
    gold_sql: str,
    plan: FunctionChangePlan,
    *,
    candidate_index: int = 0,
    candidate_count: int = 1,
    avoid_sqls: list[str] | None = None,
) -> list[dict]:
    template = _load_prompt_template()
    prompt_preserve = [
        clause for clause in GOAL_CLAUSES if clause not in plan.change_set
    ]
    body = populate_prompt(template, {
        "SCHEMA_TEXT": schema_text,
        "ORIGINAL_QUESTION": question or "",
        "ORIGINAL_SQL": gold_sql,
        "CHANGE_SET": ", ".join(plan.change_set),
        "PRESERVE_SET": ", ".join(prompt_preserve) or "(none)",
        "ALWAYS_PRESERVED": ", ".join(plan.always_preserved),
    })
    body += (
        f"\n\n# CANDIDATE DIVERSITY\n"
        f"Generate candidate {candidate_index + 1} of {candidate_count}."
    )
    if avoid_sqls:
        body += (
            "\nThe candidate must be semantically different from every "
            "previously accepted SQL below:\n"
            + "\n".join(f"- {sql}" for sql in avoid_sqls)
        )
    return [{"role": "user", "content": body}]


def _plans_for_candidate_count(parsed, count: int) -> list[FunctionChangePlan]:
    """Return exactly ``count`` plans, cycling when SQL has few clauses."""
    base_plans = select_diverse_plans(parsed, n_plans=count)
    if not base_plans:
        return []
    return [base_plans[index % len(base_plans)] for index in range(count)]


def _plan_attempt_queue(parsed, count: int) -> list[FunctionChangePlan]:
    """Return primary plans followed by unused deterministic fallbacks."""
    primary = _plans_for_candidate_count(parsed, count)
    if not primary:
        return []
    queue = list(primary)
    seen = {plan.change_set for plan in primary}
    for plan in enumerate_plans(parsed):
        if plan.change_set not in seen:
            queue.append(plan)
            seen.add(plan.change_set)
    return queue


def _call_llm(
    messages: list[dict],
    model: str,
    step: str,
    temperature: float | None,
    reasoning_effort: str | None,
) -> dict:
    assert_required_model(model, context=f"{step} model")
    if reasoning_effort is not None:
        raise BirdReproductionError(
            "Kimi K2.6 does not accept reasoning_effort in this reproduction"
        )
    response = generate_json(
        messages=messages,
        model=model,
        step=step,
        max_retries=2,
        temperature=temperature,
        reasoning_effort=None,
    )
    if not isinstance(response, dict) or not response:
        raise BirdReproductionError(f"{step} returned an empty JSON response")
    assert_required_model(model, context=f"completed {step} model")
    return response


# -----------------------------------------------------------------------------
# Per-sample orchestration
# -----------------------------------------------------------------------------

class SQLPredecessorGeneratorLLM:
    """LLM-driven multi-clause SQL function predecessor."""

    def __init__(
        self,
        num_predecessors: int = 3,
        max_attempts: int = 3,
        model: str = REQUIRED_MODEL,
        naturalizer_model: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        sql_timeout: int = 30,
        schema_max_tables: int = 12,
    ):
        assert_required_model(model, context="BIRD predecessor model")
        resolved_naturalizer = naturalizer_model or model
        assert_required_model(
            resolved_naturalizer, context="BIRD predecessor naturalizer model"
        )
        if reasoning_effort is not None:
            raise BirdReproductionError(
                "Kimi K2.6 does not accept reasoning_effort in this reproduction"
            )
        self.num_predecessors = num_predecessors
        self.max_attempts = max_attempts
        self.model = model
        self.naturalizer_model = resolved_naturalizer
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.sql_timeout = sql_timeout
        self.schema_max_tables = schema_max_tables

    def generate_predecessors(self, sample: dict) -> dict | None:
        task_id = sample.get("task_id", "unknown")
        gold_sql = sample.get("gold_sql", "")
        stored_db_path = sample.get("db_path", "")
        db_path = str(resolve_db_path(stored_db_path)) if stored_db_path else ""
        question = sample.get("question", "")

        if not gold_sql or not db_path:
            raise BirdReproductionError(f"{task_id}: missing gold_sql or db_path")
        if not Path(db_path).exists():
            raise BirdReproductionError(f"{task_id}: DB not found at {db_path}")

        parsed = parse_sql(gold_sql)
        if not parsed.parseable:
            raise BirdReproductionError(f"{task_id}: gold SQL not parseable")

        gold_result = execute_sql(db_path, gold_sql, timeout=self.sql_timeout)
        if not validate_result(gold_result):
            raise BirdReproductionError(f"{task_id}: gold SQL exec failed/empty")

        primary_plans = _plans_for_candidate_count(parsed, self.num_predecessors)
        if len(primary_plans) < self.num_predecessors:
            raise BirdReproductionError(
                f"{task_id}: planner produced "
                f"{len(primary_plans)}/{self.num_predecessors} plans"
            )
        plan_queue = _plan_attempt_queue(parsed, self.num_predecessors)

        schema_text = get_schema_text(db_path, max_tables=self.schema_max_tables)

        accepted: list[dict] = []
        seen_hashes: set[str] = set()
        seen_results: list[tuple[Any, bool]] = []
        failure_counter: Counter[str] = Counter()

        attempted_plans = 0
        for plan_idx, plan in enumerate(plan_queue):
            if len(accepted) == self.num_predecessors:
                break
            attempted_plans += 1
            candidate_index = len(accepted)
            messages = _build_messages(
                schema_text,
                question,
                gold_sql,
                plan,
                candidate_index=candidate_index,
                candidate_count=self.num_predecessors,
                avoid_sqls=[entry["counterfactual_sql"] for entry in accepted],
            )
            for attempt in range(self.max_attempts):
                step = f"predecessor_sql[{task_id}/p{plan_idx}/a{attempt}]"
                resp = _call_llm(
                    messages=messages,
                    model=self.model,
                    step=step,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )
                if "new_sql" not in resp:
                    raise BirdReproductionError(
                        f"{step} returned incomplete JSON without new_sql"
                    )
                new_sql = (resp.get("new_sql") or "").strip().rstrip(";").strip()
                rationale = (resp.get("rationale") or "").strip()
                if not new_sql:
                    raise BirdReproductionError(f"{step} returned empty new_sql")

                ok, reason, counterfactual_result = _validate_candidate(
                    new_sql=new_sql,
                    plan=plan,
                    gold_sql=gold_sql,
                    db_path=db_path,
                    gold_result=gold_result,
                    seen_hashes=seen_hashes,
                    seen_results=seen_results,
                    sql_timeout=self.sql_timeout,
                )
                if not ok:
                    failure_counter[reason] += 1
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": json.dumps(resp, ensure_ascii=False),
                        },
                        {
                            "role": "user",
                            "content": (
                                "The deterministic validator rejected that candidate "
                                f"because {reason}. Return strict JSON with a corrected, "
                                "different SQL candidate that obeys every original rule."
                            ),
                        },
                    ])
                    continue

                accepted.append({
                    "predecessor_function": "",  # filled by naturalizer below
                    "counterfactual_sql": new_sql,
                    "counterfactual_answer": _extract_answer(counterfactual_result),
                    "is_predecessor": True,
                    "transition_type": "llm_multi_clause",
                    "taxonomy_type": "T2",
                    "transition_reason": rationale,
                    "change_set": list(plan.change_set),
                    "preserve_set": list(plan.preserve_set),
                    "always_preserved": list(plan.always_preserved),
                    "plan_score": plan.score,
                    "plan_risk": plan.risk,
                    "fallback_plan": plan_idx >= len(primary_plans),
                    "counterfactual_arguments": [],
                })
                break  # success on this plan; move to next plan

        if len(accepted) != self.num_predecessors:
            raise BirdReproductionError(
                f"{task_id}: incomplete predecessor set "
                f"{len(accepted)}/{self.num_predecessors}; "
                f"failures={dict(failure_counter)}"
            )

        # ---- Naturalization pass: fill predecessor_function for every accepted entry ----
        nat_failures: Counter[str] = Counter()
        for entry in accepted:
            text, status = naturalize_followup(
                new_sql=entry["counterfactual_sql"],
                gold_sql=gold_sql,
                original_question=question,
                changed_clauses=entry["change_set"],
                preserved_clauses=entry["preserve_set"],
                model=self.naturalizer_model,
                max_attempts=self.max_attempts,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                step=f"naturalize[{task_id}]",
            )
            if text is None:
                nat_failures[status] += 1
                raise BirdReproductionError(
                    f"{task_id}: naturalization failed with {status}"
                )
            else:
                entry["predecessor_function"] = text
                entry["naturalization_failed"] = False

        if len(accepted) != self.num_predecessors:
            raise BirdReproductionError(
                f"{task_id}: expected exactly {self.num_predecessors} predecessors, "
                f"found {len(accepted)}"
            )

        new_sample = deepcopy(sample)
        new_sample["predecessor_functions"] = accepted
        new_sample["predecessor_info"] = {
            "num_plans": self.num_predecessors,
            "num_plan_attempts": attempted_plans,
            "num_accepted": len(accepted),
            "failure_counts": dict(failure_counter),
            "naturalization_failure_counts": dict(nat_failures),
            "model": self.model,
            "naturalizer_model": self.naturalizer_model,
            "method": "llm_multi_clause_v1",
        }
        assert_required_model(self.model, context="completed BIRD predecessor model")
        assert_required_model(
            self.naturalizer_model,
            context="completed BIRD predecessor naturalizer model",
        )
        print(
            f"  ✓ {task_id}: {len(accepted)}/{self.num_predecessors} plans accepted "
            f"(failures: {dict(failure_counter)})"
        )
        return new_sample


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_predecessors", type=int, default=3)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model", default=REQUIRED_MODEL)
    parser.add_argument("--naturalizer_model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--sql_timeout", type=int, default=30)
    parser.add_argument("--schema_max_tables", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="Deprecated; checkpoints are saved after every task")
    parser.add_argument("--task_ids_file", default=str(DEFAULT_TASK_IDS_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    assert_required_model(args.model, context="BIRD predecessor CLI model")
    naturalizer_model = args.naturalizer_model or args.model
    assert_required_model(
        naturalizer_model, context="BIRD predecessor CLI naturalizer model"
    )
    if args.temperature is not None:
        raise BirdReproductionError(
            "do not pass temperature for Kimi K2.6; use the provider default"
        )
    if args.reasoning_effort is not None:
        raise BirdReproductionError(
            "Kimi K2.6 does not accept reasoning_effort in this reproduction"
        )
    required_ids = load_published_task_ids(args.task_ids_file)
    if args.num_samples not in {None, len(required_ids)}:
        raise BirdReproductionError(
            "published BIRD predecessor generation cannot truncate the fixed subset"
        )
    if args.num_predecessors < 2:
        raise BirdReproductionError(
            "the paper's g=2 setting requires at least two predecessors per task"
        )

    random.seed(args.seed)
    data = validate_stage_rows(
        read_json(args.input),
        stage="bird_predecessor_input",
        required_ids=required_ids,
        require_model=True,
    )
    by_id = {sample["task_id"]: sample for sample in data}
    output_path = Path(args.output)
    cp_path = Path(args.checkpoint) if args.checkpoint else checkpoint_path_for(output_path)
    checkpoint = TaskCheckpoint(
        cp_path,
        stage="bird_predecessor",
        required_ids=required_ids,
        model=args.model,
        resume=args.resume,
    )
    generator = SQLPredecessorGeneratorLLM(
        num_predecessors=args.num_predecessors,
        max_attempts=args.max_attempts,
        model=args.model,
        naturalizer_model=naturalizer_model,
        temperature=None,
        reasoning_effort=None,
        sql_timeout=args.sql_timeout,
        schema_max_tables=args.schema_max_tables,
    )
    pending = [by_id[task_id] for task_id in checkpoint.pending_ids]

    def process_sample(sample: dict) -> dict:
        result = generator.generate_predecessors(sample)
        if not isinstance(result, dict) or result.get("task_id") != sample["task_id"]:
            raise BirdReproductionError(
                f"{sample['task_id']}: predecessor result is empty or misidentified"
            )
        predecessors = result.get("predecessor_functions")
        if not isinstance(predecessors, list) or len(predecessors) != args.num_predecessors:
            raise BirdReproductionError(
                f"{sample['task_id']}: incomplete predecessor result"
            )
        return result

    print(
        f"Generating BIRD predecessors: {len(checkpoint.processed_ids)} complete, "
        f"{len(pending)} pending, model={args.model}"
    )
    if args.num_workers > 1 and pending:
        executor = ThreadPoolExecutor(max_workers=args.num_workers)
        futures = {executor.submit(process_sample, sample): sample for sample in pending}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="predecessor"):
                sample = futures[future]
                try:
                    checkpoint.record_success(future.result())
                except BaseException as exc:
                    checkpoint.record_failure(sample["task_id"], exc)
                    raise
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    else:
        for sample in tqdm(pending, desc="predecessor"):
            try:
                checkpoint.record_success(process_sample(sample))
            except BaseException as exc:
                checkpoint.record_failure(sample["task_id"], exc)
                raise

    ordered = validate_stage_rows(
        checkpoint.results,
        stage="bird_predecessor",
        required_ids=required_ids,
        require_model=True,
        min_predecessors=args.num_predecessors,
    )
    assert_required_model(args.model, context="completed BIRD predecessor model")
    assert_required_model(
        naturalizer_model, context="completed BIRD predecessor naturalizer model"
    )
    checkpoint.mark_complete()
    atomic_write_json(output_path, ordered)
    print(f"generated predecessors for {len(ordered)}/{len(required_ids)} samples")


if __name__ == "__main__":
    main()
