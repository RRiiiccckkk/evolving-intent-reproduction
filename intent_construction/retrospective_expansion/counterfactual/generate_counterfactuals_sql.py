"""
SQL Argument Counterfactual Script (Programmatic — No LLM)

Generates counterfactual arguments for BIRD-SQL samples by swapping WHERE clause
values with real alternative values from the database. Every counterfactual is
validated by executing the counterfactual SQL and confirming a non-empty, different result.

Zero LLM dependency — all operations are deterministic.

Usage:
    python generate_counterfactuals_sql.py \
        --input ../../intent_extraction/output/bird_sql/extracted_bird_sql.json \
        --output output/bird_sql/argument_counterfactual.json \
        --num_counterfactuals 3 \
        --num_workers 4
"""

import json
import os
import re
import argparse
import random
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from intent_construction.intent_extraction.dataset_impl.bird_sql.sql_parser import (
    parse_sql,
    swap_where_value,
    swap_having_value,
    swap_limit_value,
    SQLCondition,
    SQLHavingPredicate,
)
from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import (
    compare_results,
    execute_sql,
    get_alternative_values,
    get_table_columns,
    validate_result,
)
from intent_construction.intent_extraction.dataset_impl.bird_sql.reproduction import (
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


# =============================================================================
# NL templates for argument descriptions
# =============================================================================

_OPERATOR_TEMPLATES: dict[str, str] = {
    "=": "The {column} is {value}.",
    "!=": "The {column} is not {value}.",
    ">": "The {column} is greater than {value}.",
    "<": "The {column} is less than {value}.",
    ">=": "The {column} is at least {value}.",
    "<=": "The {column} is at most {value}.",
    "LIKE": "The {column} contains {value}.",
    "IN": "The {column} is one of {value}.",
    "BETWEEN": "The {column} is between {value}.",
}


# NL templates for HAVING (group-level) predicates
_HAVING_COUNT_TEMPLATES: dict[str, str] = {
    "=":  "Each group has exactly {k} records.",
    "!=": "Each group has a number of records other than {k}.",
    ">":  "Each group has more than {k} records.",
    "<":  "Each group has fewer than {k} records.",
    ">=": "Each group has at least {k} records.",
    "<=": "Each group has at most {k} records.",
}

_HAVING_AGG_TEMPLATES: dict[str, str] = {
    "=":  "The {agg} of {column} per group equals {v}.",
    "!=": "The {agg} of {column} per group is not {v}.",
    ">":  "The {agg} of {column} per group is greater than {v}.",
    "<":  "The {agg} of {column} per group is less than {v}.",
    ">=": "The {agg} of {column} per group is at least {v}.",
    "<=": "The {agg} of {column} per group is at most {v}.",
}

_AGG_FRIENDLY: dict[str, str] = {
    "COUNT": "count", "SUM": "sum", "AVG": "average",
    "MAX": "maximum", "MIN": "minimum",
}


def _format_having_nl(predicate: SQLHavingPredicate, new_value: str) -> str:
    """Build a natural-language description for a counterfactual HAVING predicate."""
    agg = (predicate.aggregate or "").upper()
    op = predicate.operator.upper().strip()
    if agg == "COUNT":
        tmpl = _HAVING_COUNT_TEMPLATES.get(op, "Each group has at least {k} records.")
        text = tmpl.format(k=new_value)
        # Singular polish: "1 records" -> "1 record"
        if str(new_value).strip() == "1":
            text = text.replace(" 1 records", " 1 record")
        return text
    tmpl = _HAVING_AGG_TEMPLATES.get(op, "The {agg} of {column} per group is {v}.")
    return tmpl.format(
        agg=_AGG_FRIENDLY.get(agg, agg.lower() or "value"),
        column=predicate.column or "value",
        v=new_value,
    )


def _format_limit_nl(new_limit: int) -> str:
    """Build a natural-language description for a counterfactual LIMIT."""
    if new_limit == 1:
        return "Show only the top result."
    return f"Show only the top {new_limit} results."


def _format_nl_condition(column: str, operator: str, value: str) -> str:
    """Build a natural-language argument description from column, operator, and value."""
    op_upper = operator.upper().strip()

    # Clean LIKE patterns: strip surrounding % signs for display
    display_value = value
    if op_upper == "LIKE":
        display_value = value.strip("%")

    template = _OPERATOR_TEMPLATES.get(op_upper, "The {column} is {value}.")
    return template.format(column=column, value=display_value)


# =============================================================================
# SQL value replacement (AST-based with string fallback)
# =============================================================================

def _build_sql_condition(argument: dict) -> SQLCondition:
    """Build a SQLCondition from a sample argument dict for swap_where_value()."""
    return SQLCondition(
        argument_id=argument.get("argument_id", 0),
        column=argument.get("sql_column", ""),
        operator=argument.get("sql_operator", "="),
        value=argument.get("sql_value", ""),
        table=argument.get("sql_table", ""),
        raw_expression="",
    )


def _is_numeric(value: str) -> bool:
    """Check if a string value represents a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _numeric_expression_candidates(value: str, *, limit: int) -> list[str]:
    """Generate deterministic alternatives for numeric expression thresholds."""
    numeric = _coerce_float(value)
    if numeric is None or limit < 1:
        return []
    is_integer = float(numeric).is_integer() and "." not in str(value)
    candidates: list[str] = []
    seen: set[str] = {str(value)}
    for delta in (-1, 1, -2, 2, -5, 5, -10, 10, -25, 25):
        candidate = numeric + delta
        rendered = str(int(candidate)) if is_integer else str(round(candidate, 4))
        if rendered in seen:
            continue
        seen.add(rendered)
        candidates.append(rendered)
        if len(candidates) >= limit:
            break
    return candidates


def _format_expression_counterfactual(argument: dict, new_value: str) -> str:
    """Prefer the extracted natural-language argument over a raw SQL expression."""
    text = str(argument.get("argument", "")).strip()
    old_value = str(argument.get("sql_value", "")).strip()
    if text and old_value:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(old_value)}(?![A-Za-z0-9_])"
        )
        replaced, count = pattern.subn(str(new_value), text, count=1)
        if count:
            return replaced
    return _format_nl_condition(
        str(argument.get("sql_column", "value")),
        str(argument.get("sql_operator", "=")),
        str(new_value),
    )


def _sql_quote(value) -> str:
    """Quote a value for SQL: strings get quotes, numbers don't."""
    s = str(value)
    if _is_numeric(s):
        return s
    # Escape single quotes inside string values
    escaped = s.replace("'", "''")
    return f"'{escaped}'"


def _swap_value_in_sql(
    gold_sql: str,
    argument: dict,
    new_value: str,
) -> str | None:
    """
    Replace a WHERE condition's value in gold_sql.

    Strategy:
      1. Try AST-based swap_where_value() from sql_parser.
      2. Fall back to targeted string replacement if AST fails.

    Returns the modified SQL string, or None on failure.
    """
    sql_cond = _build_sql_condition(argument)
    old_value = argument.get("sql_value", "")
    operator = argument.get("sql_operator", "=").upper()

    # --- Attempt 1: AST-based swap ---
    try:
        modified = swap_where_value(gold_sql, sql_cond, str(new_value))
        # Verify the swap actually changed the SQL
        if modified != gold_sql:
            return modified
    except Exception:
        pass

    # --- Attempt 2: String-based fallback ---
    try:
        old_quoted = _sql_quote(old_value)
        new_quoted = _sql_quote(new_value)

        # For LIKE, preserve % patterns
        if operator == "LIKE":
            # Try replacing the full LIKE pattern
            old_like = f"'%{old_value}%'"
            new_like = f"'%{new_value}%'"
            if old_like in gold_sql:
                return gold_sql.replace(old_like, new_like, 1)
            old_like = f"'{old_value}%'"
            new_like = f"'{new_value}%'"
            if old_like in gold_sql:
                return gold_sql.replace(old_like, new_like, 1)
            old_like = f"'%{old_value}'"
            new_like = f"'%{new_value}'"
            if old_like in gold_sql:
                return gold_sql.replace(old_like, new_like, 1)

        # Direct replacement of quoted value
        if old_quoted in gold_sql:
            return gold_sql.replace(old_quoted, new_quoted, 1)

        # Try unquoted replacement (for numeric values in expressions)
        if str(old_value) in gold_sql:
            # Be careful: only replace in a WHERE-like context
            # Use word-boundary-aware replacement
            pattern = re.compile(
                r"(?<=[=<>!\s])\s*" + re.escape(str(old_value)) + r"(?=\s|$|[)\s,;])",
                re.IGNORECASE,
            )
            result, count = pattern.subn(str(new_value), gold_sql, count=1)
            if count > 0:
                return result

    except Exception:
        pass

    return None


def _resolve_argument_table(
    argument: dict,
    gold_sql: str,
    db_path: str,
) -> str:
    """Resolve an unqualified WHERE column against the query's source tables."""
    explicit = str(argument.get("sql_table", "")).strip()
    if explicit:
        return explicit
    column = str(argument.get("sql_column", "")).strip()
    if not column:
        return ""
    try:
        parsed = parse_sql(gold_sql)
    except Exception:
        return ""
    matches = [
        table
        for table in parsed.tables
        if any(
            info.name.casefold() == column.casefold()
            for info in get_table_columns(db_path, table)
        )
    ]
    return matches[0] if len(matches) == 1 else ""


# =============================================================================
# Numeric candidate generation (for HAVING / LIMIT)
# =============================================================================

def _coerce_int(s) -> int | None:
    try:
        f = float(s)
        if f != f or abs(f) == float("inf"):
            return None
        return int(round(f))
    except (TypeError, ValueError):
        return None


def _coerce_float(s) -> float | None:
    try:
        f = float(s)
        if f != f or abs(f) == float("inf"):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _having_candidates(predicate: SQLHavingPredicate) -> list[str]:
    """Generate ordered candidate replacement values for a HAVING predicate."""
    agg = (predicate.aggregate or "").upper()
    raw = predicate.value
    if isinstance(raw, (list, tuple)):
        return []
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(val):
        if val is None:
            return
        s = str(val)
        if s in seen or s == str(raw):
            return
        seen.add(s)
        candidates.append(s)

    if agg == "COUNT":
        k = _coerce_int(raw)
        if k is None:
            return []
        # Vary the threshold meaningfully but keep it sensible.
        for delta in (-5, -3, -2, -1, 1, 2, 3, 5, 10):
            v = k + delta
            if v >= 1:
                _add(v)
        for mul in (0.5, 2.0):
            v = max(1, int(round(k * mul)))
            _add(v)
    else:
        f = _coerce_float(raw)
        if f is None:
            return []
        is_int = "." not in str(raw) and "e" not in str(raw).lower()
        for mul in (0.5, 0.8, 1.25, 2.0, 1.5):
            v = f * mul
            if is_int:
                _add(max(0, int(round(v))))
            else:
                _add(round(v, 4))
        for delta in (-1, 1, -10, 10):
            v = f + delta
            if v >= 0:
                _add(int(round(v)) if is_int else round(v, 4))
    return candidates


def _limit_candidates(original: int) -> list[int]:
    """Generate candidate LIMIT values different from the original."""
    base = [1, 3, 5, 10, 20, 50, 100]
    extra = [max(1, original // 2), max(1, original * 2), original + 1, max(1, original - 1)]
    out: list[int] = []
    seen: set[int] = set()
    for v in base + extra:
        if v <= 0 or v == original or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# =============================================================================
# Result extraction helper
# =============================================================================

def _extract_answer(result) -> str:
    """Extract a single answer string from a SQLResult."""
    if result.rows and len(result.rows) > 0:
        first_row = result.rows[0]
        if len(first_row) == 1:
            val = first_row[0]
            return str(val) if val is not None else "NULL"
        # Multiple columns: join them
        return str(first_row)
    return ""


# =============================================================================
# Core counterfactual logic
# =============================================================================

class SQLCounterfactualGenerator:
    """Programmatic argument counterfactual for SQL samples."""

    def __init__(
        self,
        num_counterfactuals: int = 3,
        alt_value_limit: int = 10,
        sql_timeout: int = 30,
    ):
        self.num_counterfactuals = num_counterfactuals
        self.alt_value_limit = alt_value_limit
        self.sql_timeout = sql_timeout

    def generate_counterfactual(
        self,
        argument: dict,
        gold_sql: str,
        db_path: str,
        original_result,
    ) -> list[dict]:
        """
        Generate counterfactual versions of a single argument.

        For each alternative DB value, builds a modified SQL, executes it,
        and keeps only counterfactuals that produce non-empty, different results.

        Returns a list of counterfactual dicts.
        """
        sql_column = argument.get("sql_column", "")
        sql_table = _resolve_argument_table(argument, gold_sql, db_path)
        sql_value = argument.get("sql_value", "")
        sql_operator = argument.get("sql_operator", "=")

        if not sql_column:
            return []

        # Skip complex operators that don't support simple value swaps
        if sql_operator.upper() in ("OR", "NOT", "UNKNOWN", "IS"):
            return []

        expression_fallback = False
        if sql_table:
            alternatives = get_alternative_values(
                db_path, sql_table, sql_column, sql_value, limit=self.alt_value_limit
            )
        elif _is_numeric(sql_value) and sql_operator.upper().strip() in {
            "=", "!=", ">", "<", ">=", "<=",
        }:
            # Extractors may describe a date/year predicate as a SQL expression
            # instead of a physical column. Nearby thresholds can still be
            # substituted and execution-validated without guessing a table.
            expression_fallback = True
            alternatives = _numeric_expression_candidates(
                str(sql_value), limit=self.alt_value_limit
            )
        else:
            alternatives = []

        if not alternatives:
            return []

        # Shuffle to get diverse results across runs
        random.shuffle(alternatives)

        counterfactuals: list[dict] = []

        for alt_val in alternatives:
            if len(counterfactuals) >= self.num_counterfactuals:
                break

            alt_str = str(alt_val)

            # Skip empty or null alternatives
            if not alt_str or alt_str.lower() == "none":
                continue

            # Build modified SQL
            modified_sql = _swap_value_in_sql(gold_sql, argument, alt_str)
            if modified_sql is None or modified_sql == gold_sql:
                continue

            # Execute modified SQL
            counterfactual_result = execute_sql(db_path, modified_sql, timeout=self.sql_timeout)

            # Validate: must succeed with non-empty results
            if not validate_result(counterfactual_result):
                continue

            # Validate: result must differ from original
            if compare_results(original_result, counterfactual_result):
                continue

            # Build NL description for the counterfactual argument
            nl_condition = (
                _format_expression_counterfactual(argument, alt_str)
                if expression_fallback
                else _format_nl_condition(sql_column, sql_operator, alt_str)
            )

            counterfactuals.append({
                "counterfactual_argument": nl_condition,
                "original_value": str(sql_value),
                "counterfactual_value": alt_str,
                "counterfactual_sql": modified_sql,
                "counterfactual_answer": _extract_answer(counterfactual_result),
                "reasoning": f"Swapped {sql_column} value from {sql_value} to {alt_str}",
            })

        return counterfactuals

    def generate_having_counterfactuals(
        self,
        predicate: SQLHavingPredicate,
        gold_sql: str,
        db_path: str,
        original_result,
    ) -> list[dict]:
        """Generate counterfactual versions of a single HAVING predicate."""
        candidates = _having_candidates(predicate)
        if not candidates:
            return []
        random.shuffle(candidates)

        counterfactuals: list[dict] = []
        for new_val in candidates:
            if len(counterfactuals) >= self.num_counterfactuals:
                break
            try:
                modified_sql = swap_having_value(gold_sql, predicate, new_val)
            except Exception:
                continue
            if modified_sql == gold_sql:
                continue
            counterfactual_result = execute_sql(db_path, modified_sql, timeout=self.sql_timeout)
            if not validate_result(counterfactual_result):
                continue
            if compare_results(original_result, counterfactual_result):
                continue
            counterfactuals.append({
                "kind": "having",
                "predicate_id": predicate.predicate_id,
                "aggregate": predicate.aggregate,
                "column": predicate.column,
                "operator": predicate.operator,
                "original_value": str(predicate.value),
                "counterfactual_value": str(new_val),
                "counterfactual_sql": modified_sql,
                "counterfactual_answer": _extract_answer(counterfactual_result),
                "nl_condition": _format_having_nl(predicate, str(new_val)),
                "reasoning": (
                    f"Swapped HAVING {predicate.aggregate or ''}({predicate.column}) "
                    f"{predicate.operator} value from {predicate.value} to {new_val}"
                ),
            })
        return counterfactuals

    def generate_limit_counterfactuals(
        self,
        original_limit: int,
        gold_sql: str,
        db_path: str,
        original_result,
    ) -> list[dict]:
        """Generate counterfactual versions of the LIMIT clause."""
        candidates = _limit_candidates(int(original_limit))
        if not candidates:
            return []
        random.shuffle(candidates)

        counterfactuals: list[dict] = []
        for new_val in candidates:
            if len(counterfactuals) >= self.num_counterfactuals:
                break
            try:
                modified_sql = swap_limit_value(gold_sql, int(new_val))
            except Exception:
                continue
            if modified_sql == gold_sql:
                continue
            counterfactual_result = execute_sql(db_path, modified_sql, timeout=self.sql_timeout)
            if not validate_result(counterfactual_result):
                continue
            if compare_results(original_result, counterfactual_result):
                continue
            counterfactuals.append({
                "kind": "limit",
                "original_value": str(original_limit),
                "counterfactual_value": str(new_val),
                "counterfactual_sql": modified_sql,
                "counterfactual_answer": _extract_answer(counterfactual_result),
                "nl_condition": _format_limit_nl(int(new_val)),
                "reasoning": f"Changed LIMIT from {original_limit} to {new_val}",
            })
        return counterfactuals

    def generate_counterfactuals(self, sample: dict) -> dict | None:
        """
        Generate counterfactual arguments for all arguments in a sample.

        Returns the enriched sample with counterfactual_arguments attached to each
        argument, or None if no counterfactuals could be generated.
        """
        task_id = sample.get("task_id", "unknown")
        gold_sql = sample.get("gold_sql", "")
        stored_db_path = sample.get("db_path", "")
        db_path = str(resolve_db_path(stored_db_path)) if stored_db_path else ""
        arguments = sample.get("arguments", [])

        if not gold_sql or not db_path:
            print(f"  ✗ {task_id}: missing gold_sql or db_path")
            return None

        if not Path(db_path).exists():
            print(f"  ✗ {task_id}: database not found at {db_path}")
            return None

        # Execute original SQL to get baseline result
        original_result = execute_sql(db_path, gold_sql, timeout=self.sql_timeout)
        if not validate_result(original_result):
            print(f"  ✗ {task_id}: original SQL failed or returned empty")
            return None

        new_sample = deepcopy(sample)
        total_arguments = len(arguments)
        successful_counterfactuals = 0

        for i, cond in enumerate(arguments):
            counterfactuals = self.generate_counterfactual(
                cond, gold_sql, db_path, original_result
            )
            if counterfactuals:
                new_sample["arguments"][i]["counterfactual_arguments"] = counterfactuals
                successful_counterfactuals += len(counterfactuals)

        # --- HAVING / LIMIT counterfactuals (top-level keys on the sample) ---
        having_counterfactuals: list[dict] = []
        limit_counterfactuals: list[dict] = []
        n_having_targets = 0
        n_limit_targets = 0
        try:
            parsed = parse_sql(gold_sql)
        except Exception:
            parsed = None

        if parsed is not None and parsed.parseable:
            for predicate in parsed.having_predicates:
                n_having_targets += 1
                having_counterfactuals.extend(
                    self.generate_having_counterfactuals(predicate, gold_sql, db_path, original_result)
                )
            if parsed.limit is not None:
                n_limit_targets = 1
                limit_counterfactuals = self.generate_limit_counterfactuals(
                    parsed.limit, gold_sql, db_path, original_result
                )

        if having_counterfactuals:
            new_sample["having_counterfactuals"] = having_counterfactuals
            successful_counterfactuals += len(having_counterfactuals)
        if limit_counterfactuals:
            new_sample["limit_counterfactuals"] = limit_counterfactuals
            successful_counterfactuals += len(limit_counterfactuals)

        if successful_counterfactuals == 0:
            print(f"  ✗ {task_id}: no valid counterfactuals generated")
            return None

        new_sample["counterfactual_info"] = {
            "num_counterfactuals_requested": self.num_counterfactuals,
            "total_arguments": total_arguments,
            "successful_counterfactuals": successful_counterfactuals,
            "num_having_targets": n_having_targets,
            "num_limit_targets": n_limit_targets,
            "num_having_counterfactuals": len(having_counterfactuals),
            "num_limit_counterfactuals": len(limit_counterfactuals),
            "dataset_type": "sql",
        }

        print(
            f"  ✓ {task_id}: {successful_counterfactuals} counterfactuals "
            f"(WHERE={successful_counterfactuals - len(having_counterfactuals) - len(limit_counterfactuals)}, "
            f"HAVING={len(having_counterfactuals)}, LIMIT={len(limit_counterfactuals)})"
        )
        return new_sample


# =============================================================================
# CLI
# =============================================================================


def _validate_completed_sample(result: object, *, minimum: int) -> dict:
    if not isinstance(result, dict):
        raise BirdReproductionError("counterfactual generator returned no result")
    task_id = result.get("task_id", "unknown")
    generated: list[dict] = []
    arguments = result.get("arguments")
    if not isinstance(arguments, list):
        raise BirdReproductionError(f"{task_id} has no argument list")
    for argument in arguments:
        if isinstance(argument, dict):
            values = argument.get("counterfactual_arguments", [])
            if isinstance(values, list):
                generated.extend(item for item in values if isinstance(item, dict))
    for key in ("having_counterfactuals", "limit_counterfactuals"):
        values = result.get(key, [])
        if isinstance(values, list):
            generated.extend(item for item in values if isinstance(item, dict))
    if len(generated) < minimum:
        raise BirdReproductionError(
            f"{task_id} produced only {len(generated)} complete counterfactuals; "
            f"at least {minimum} are required"
        )
    for item in generated:
        if not str(item.get("counterfactual_sql", "")).strip():
            raise BirdReproductionError(
                f"{task_id} contains a counterfactual without SQL"
            )
    info = result.get("counterfactual_info")
    if not isinstance(info, dict) or info.get("successful_counterfactuals") != len(generated):
        raise BirdReproductionError(
            f"{task_id} counterfactual metadata is incomplete or inconsistent"
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Programmatic SQL argument counterfactual for BIRD-SQL"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input extracted JSON file",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--num_counterfactuals", type=int, default=3,
        help="Max counterfactuals per argument (default: 3)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Parallel workers for sample processing (default: 4)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Must be omitted (the published run is fixed at 100 samples)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=50,
        help="Deprecated; checkpoints are saved after every task",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if exists",
    )
    parser.add_argument(
        "--sql_timeout", type=int, default=30,
        help="SQL execution timeout in seconds (default: 30)",
    )
    parser.add_argument("--model", default=REQUIRED_MODEL)
    parser.add_argument("--task_ids_file", default=str(DEFAULT_TASK_IDS_PATH))
    parser.add_argument("--checkpoint", default=None)

    args = parser.parse_args()

    assert_required_model(args.model, context="BIRD counterfactual stage model")
    random.seed(args.seed)
    required_ids = load_published_task_ids(args.task_ids_file)
    if args.num_samples not in {None, len(required_ids)}:
        raise BirdReproductionError(
            "published BIRD counterfactual generation cannot truncate the fixed subset"
        )
    data = validate_stage_rows(
        read_json(args.input),
        stage="bird_counterfactual_input",
        required_ids=required_ids,
        require_model=True,
    )
    by_id = {sample["task_id"]: sample for sample in data}
    output_path = Path(args.output)
    cp_path = Path(args.checkpoint) if args.checkpoint else checkpoint_path_for(output_path)
    checkpoint = TaskCheckpoint(
        cp_path,
        stage="bird_counterfactual",
        required_ids=required_ids,
        model=args.model,
        resume=args.resume,
    )

    # Initialize generator
    generator = SQLCounterfactualGenerator(
        num_counterfactuals=args.num_counterfactuals,
        sql_timeout=args.sql_timeout,
    )

    pending = [by_id[task_id] for task_id in checkpoint.pending_ids]

    def process_sample(sample: dict) -> dict:
        return _validate_completed_sample(
            generator.generate_counterfactuals(sample),
            minimum=args.num_counterfactuals,
        )

    print(
        f"Generating SQL counterfactuals: {len(checkpoint.processed_ids)} complete, "
        f"{len(pending)} pending"
    )
    if args.num_workers > 1 and pending:
        executor = ThreadPoolExecutor(max_workers=args.num_workers)
        futures = {executor.submit(process_sample, sample): sample for sample in pending}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="counterfactual"):
                sample = futures[future]
                task_id = sample["task_id"]
                try:
                    checkpoint.record_success(future.result())
                except BaseException as exc:
                    checkpoint.record_failure(task_id, exc)
                    raise
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    else:
        for sample in tqdm(pending, desc="counterfactual"):
            task_id = sample["task_id"]
            try:
                checkpoint.record_success(process_sample(sample))
            except BaseException as exc:
                checkpoint.record_failure(task_id, exc)
                raise

    ordered = validate_stage_rows(
        checkpoint.results,
        stage="bird_counterfactual",
        required_ids=required_ids,
        require_model=True,
    )
    assert_required_model(args.model, context="completed BIRD counterfactual stage")
    checkpoint.mark_complete()
    atomic_write_json(output_path, ordered)
    print(f"generated counterfactuals for {len(ordered)}/{len(required_ids)} samples")


if __name__ == "__main__":
    main()
