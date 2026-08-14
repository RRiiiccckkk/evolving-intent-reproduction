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
from intent_construction.retrospective_expansion.predecessor.function_change_planner import (  # noqa: E402
    select_diverse_plans,
    FunctionChangePlan,
)
from intent_construction.retrospective_expansion.predecessor.sql_naturalizer import naturalize_followup  # noqa: E402


PROMPT_PATH = _THIS / "prompts" / "generate_predecessor_sql.txt"


# -----------------------------------------------------------------------------
# Clause extraction (for byte-identical comparison)
# -----------------------------------------------------------------------------

_CLAUSE_TO_ARG: dict[str, str] = {
    "SELECT":   "expressions",
    "FROM":     "from",
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
    must_preserve = set(plan.preserve_set) | set(plan.always_preserved)
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

    if compare_results(gold_result, counterfactual_result):
        return False, "same_result", None

    h = hashlib.md5(_normalize(new_sql).encode()).hexdigest()
    if h in seen_hashes:
        return False, "duplicate", None
    seen_hashes.add(h)

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
) -> list[dict]:
    template = _load_prompt_template()
    body = populate_prompt(template, {
        "SCHEMA_TEXT": schema_text,
        "ORIGINAL_QUESTION": question or "",
        "ORIGINAL_SQL": gold_sql,
        "CHANGE_SET": ", ".join(plan.change_set),
        "PRESERVE_SET": ", ".join(plan.preserve_set) or "(none)",
        "ALWAYS_PRESERVED": ", ".join(plan.always_preserved),
    })
    return [{"role": "user", "content": body}]


def _call_llm(
    messages: list[dict],
    model: str,
    step: str,
    temperature: float,
    reasoning_effort: str | None,
) -> dict | None:
    try:
        return generate_json(
            messages=messages,
            model=model,
            step=step,
            max_retries=2,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        print(f"  ⚠ LLM call failed [{step}]: {e}")
        return None


# -----------------------------------------------------------------------------
# Per-sample orchestration
# -----------------------------------------------------------------------------

class SQLPredecessorGeneratorLLM:
    """LLM-driven multi-clause SQL function predecessor."""

    def __init__(
        self,
        num_predecessors: int = 3,
        max_attempts: int = 3,
        model: str = "gpt-5.1",
        naturalizer_model: str | None = None,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        sql_timeout: int = 30,
        schema_max_tables: int = 12,
    ):
        self.num_predecessors = num_predecessors
        self.max_attempts = max_attempts
        self.model = model
        self.naturalizer_model = naturalizer_model or model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.sql_timeout = sql_timeout
        self.schema_max_tables = schema_max_tables

    def generate_predecessors(self, sample: dict) -> dict | None:
        task_id = sample.get("task_id", "unknown")
        gold_sql = sample.get("gold_sql", "")
        db_path = sample.get("db_path", "")
        question = sample.get("question", "")

        if not gold_sql or not db_path:
            print(f"  ✗ {task_id}: missing gold_sql or db_path")
            return None
        if not Path(db_path).exists():
            print(f"  ✗ {task_id}: DB not found at {db_path}")
            return None

        parsed = parse_sql(gold_sql)
        if not parsed.parseable:
            print(f"  ✗ {task_id}: gold SQL not parseable")
            return None

        gold_result = execute_sql(db_path, gold_sql, timeout=self.sql_timeout)
        if not validate_result(gold_result):
            print(f"  ✗ {task_id}: gold SQL exec failed/empty")
            return None

        plans = select_diverse_plans(parsed, n_plans=self.num_predecessors)
        if not plans:
            print(f"  ✗ {task_id}: planner produced no plans")
            return None

        schema_text = get_schema_text(db_path, max_tables=self.schema_max_tables)

        accepted: list[dict] = []
        seen_hashes: set[str] = set()
        failure_counter: Counter[str] = Counter()

        for plan_idx, plan in enumerate(plans):
            messages = _build_messages(schema_text, question, gold_sql, plan)
            for attempt in range(self.max_attempts):
                step = f"predecessor_sql[{task_id}/p{plan_idx}/a{attempt}]"
                resp = _call_llm(
                    messages=messages,
                    model=self.model,
                    step=step,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )
                if not resp or "new_sql" not in resp:
                    failure_counter["llm_call_or_format_fail"] += 1
                    continue
                new_sql = (resp.get("new_sql") or "").strip().rstrip(";").strip()
                rationale = (resp.get("rationale") or "").strip()
                if not new_sql:
                    failure_counter["empty_sql"] += 1
                    continue

                ok, reason, counterfactual_result = _validate_candidate(
                    new_sql=new_sql,
                    plan=plan,
                    gold_sql=gold_sql,
                    db_path=db_path,
                    gold_result=gold_result,
                    seen_hashes=seen_hashes,
                    sql_timeout=self.sql_timeout,
                )
                if not ok:
                    failure_counter[reason] += 1
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
                    "counterfactual_arguments": [],
                })
                break  # success on this plan; move to next plan

        if not accepted:
            print(
                f"  ✗ {task_id}: 0 accepted "
                f"(failures: {dict(failure_counter)})"
            )
            return None

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
                entry["predecessor_function"] = ""
                entry["naturalization_failed"] = True
                entry["naturalization_failure_reason"] = status
            else:
                entry["predecessor_function"] = text
                entry["naturalization_failed"] = False

        # Drop any entries that failed naturalization (downstream needs the text).
        accepted = [e for e in accepted if not e.get("naturalization_failed")]
        if not accepted:
            print(
                f"  ✗ {task_id}: 0 accepted after naturalization "
                f"(nat failures: {dict(nat_failures)})"
            )
            return None

        new_sample = deepcopy(sample)
        new_sample["predecessor_functions"] = accepted
        new_sample["predecessor_info"] = {
            "num_plans": len(plans),
            "num_accepted": len(accepted),
            "failure_counts": dict(failure_counter),
            "naturalization_failure_counts": dict(nat_failures),
            "model": self.model,
            "naturalizer_model": self.naturalizer_model,
            "method": "llm_multi_clause_v1",
        }
        print(
            f"  ✓ {task_id}: {len(accepted)}/{len(plans)} plans accepted "
            f"(failures: {dict(failure_counter)})"
        )
        return new_sample


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-based SQL function predecessor (multi-clause).")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_predecessors", type=int, default=3,
                        help="Plans per sample (also caps accepted predecessors).")
    parser.add_argument("--max_attempts", type=int, default=3,
                        help="LLM retries per plan.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--naturalizer_model", type=str, default=None,
                        help="Override the model for the naturalizer step (default: same as --model).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=[None, "low", "medium", "high"])
    parser.add_argument("--sql_timeout", type=int, default=30)
    parser.add_argument("--schema_max_tables", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    checkpoint_path = args.output.replace(".json", "_checkpoint.json")

    print(f"Loading input: {args.input}")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    if args.num_samples is not None:
        data = data[: args.num_samples]
        print(f"Processing first {len(data)} samples")

    print(f"Model: {args.model}")
    print(f"Plans per sample: {args.num_predecessors}, max_attempts: {args.max_attempts}")
    print(f"Workers: {args.num_workers}")

    results: list[dict] = []
    failed = 0
    processed_ids: set[str] = set()

    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            cp = json.load(f)
        results = cp.get("results", [])
        failed = cp.get("failed", 0)
        processed_ids = set(cp.get("processed_ids", []))
        print(f"Resuming with {len(results)} prior results")

    generator = SQLPredecessorGeneratorLLM(
        num_predecessors=args.num_predecessors,
        max_attempts=args.max_attempts,
        model=args.model,
        naturalizer_model=args.naturalizer_model,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        sql_timeout=args.sql_timeout,
        schema_max_tables=args.schema_max_tables,
    )

    def _save_checkpoint():
        with open(checkpoint_path, "w") as f:
            json.dump({
                "results": results,
                "failed": failed,
                "processed_ids": list(processed_ids),
            }, f)

    def _process(sample: dict):
        sid = sample.get("task_id", "unknown")
        if sid in processed_ids:
            return None, sid, "skipped"
        try:
            r = generator.generate_predecessors(sample)
        except Exception as e:
            print(f"  ✗ {sid}: exception: {e}")
            traceback.print_exc()
            return None, sid, "error"
        return r, sid, ("success" if r is not None else "failed")

    print("\nGenerating LLM SQL function predecessors...")
    if args.num_workers > 1:
        chunk = args.checkpoint_interval
        for cstart in range(0, len(data), chunk):
            cend = min(cstart + chunk, len(data))
            todo = data[cstart:cend]
            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                futs = {ex.submit(_process, s): s for s in todo}
                for fut in tqdm(as_completed(futs), total=len(todo),
                                desc=f"chunk {cstart//chunk + 1}"):
                    r, sid, status = fut.result()
                    if status == "success":
                        results.append(r)
                        processed_ids.add(sid)
                    elif status in ("failed", "error"):
                        failed += 1
            _save_checkpoint()
            print(f"  💾 checkpoint @ {cend}: {len(results)} accepted, {failed} failed")
    else:
        for s in tqdm(data, desc="function-llm"):
            r, sid, status = _process(s)
            if status == "success":
                results.append(r)
                processed_ids.add(sid)
            elif status in ("failed", "error"):
                failed += 1

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Accepted: {len(results)}/{len(data)} samples (failed: {failed})")
    if results:
        total_g = sum(len(r.get("predecessor_functions", [])) for r in results)
        print(f"  Total function predecessors: {total_g}")
        with open(args.output.replace(".json", "_example.json"), "w") as f:
            json.dump(results[0], f, indent=2)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


if __name__ == "__main__":
    main()
