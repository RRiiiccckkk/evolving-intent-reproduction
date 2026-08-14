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
from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import execute_sql, get_alternative_values, validate_result, compare_results


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
        sql_table = argument.get("sql_table", "")
        sql_value = argument.get("sql_value", "")
        sql_operator = argument.get("sql_operator", "=")

        if not sql_column or not sql_table:
            return []

        # Skip complex operators that don't support simple value swaps
        if sql_operator.upper() in ("OR", "NOT", "UNKNOWN", "IS"):
            return []

        # Get alternative values from the database
        alternatives = get_alternative_values(
            db_path, sql_table, sql_column, sql_value, limit=self.alt_value_limit
        )

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
            nl_condition = _format_nl_condition(sql_column, sql_operator, alt_str)

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
        db_path = sample.get("db_path", "")
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
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=50,
        help="Save checkpoint every N samples (default: 50)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if exists",
    )
    parser.add_argument(
        "--sql_timeout", type=int, default=30,
        help="SQL execution timeout in seconds (default: 30)",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = args.output.replace(".json", "_checkpoint.json")

    # Load input data
    print(f"Loading input data from: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    if args.num_samples is not None:
        data = data[: args.num_samples]
        print(f"Processing first {len(data)} samples")

    print(f"\nConfiguration:")
    print(f"  Counterfactuals per argument: {args.num_counterfactuals}")
    print(f"  Workers: {args.num_workers}")
    print(f"  SQL timeout: {args.sql_timeout}s")

    # Resume from checkpoint
    results: list[dict] = []
    failed = 0
    start_idx = 0
    processed_ids: set[str] = set()

    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        results = checkpoint_data.get("results", [])
        failed = checkpoint_data.get("failed", 0)
        start_idx = checkpoint_data.get("next_idx", 0)
        processed_ids = set(checkpoint_data.get("processed_ids", []))
        print(f"  Loaded {len(results)} completed results, starting from index {start_idx}")

    # Initialize generator
    generator = SQLCounterfactualGenerator(
        num_counterfactuals=args.num_counterfactuals,
        sql_timeout=args.sql_timeout,
    )

    def save_checkpoint(next_idx: int):
        cp = {
            "results": results,
            "failed": failed,
            "next_idx": next_idx,
            "processed_ids": list(processed_ids),
            "total_samples": len(data),
            "num_counterfactuals": args.num_counterfactuals,
            "dataset_type": "sql",
        }
        with open(checkpoint_path, "w") as f:
            json.dump(cp, f)

    def process_sample(idx_sample: tuple[int, dict]):
        idx, sample = idx_sample
        sample_id = sample.get("task_id", f"sample-{idx}")
        if sample_id in processed_ids:
            return None, sample_id, "skipped"
        result = generator.generate_counterfactuals(sample)
        if result is not None:
            return result, sample_id, "success"
        return None, sample_id, "failed"

    samples_to_process = [
        (start_idx + i, s) for i, s in enumerate(data[start_idx:])
    ]
    actual_idx = start_idx

    print(f"\nGenerating SQL argument counterfactuals...")

    try:
        if args.num_workers > 1:
            print(f"  Using {args.num_workers} parallel workers...")

            chunk_size = args.checkpoint_interval
            for chunk_start in range(0, len(samples_to_process), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(samples_to_process))
                chunk = samples_to_process[chunk_start:chunk_end]

                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = {
                        executor.submit(process_sample, item): item for item in chunk
                    }
                    for future in tqdm(
                        as_completed(futures),
                        desc=f"Chunk {chunk_start // chunk_size + 1}",
                        total=len(chunk),
                    ):
                        result, sample_id, status = future.result()
                        if status == "success":
                            results.append(result)
                            processed_ids.add(sample_id)
                        elif status == "failed":
                            failed += 1

                actual_idx = start_idx + chunk_end - 1
                save_checkpoint(actual_idx + 1)
                print(
                    f"\n  💾 Checkpoint saved at index {actual_idx + 1} "
                    f"({len(results)} results)"
                )
        else:
            for idx, sample in enumerate(
                tqdm(
                    data[start_idx:],
                    desc="Argument counterfactual",
                    initial=start_idx,
                    total=len(data),
                )
            ):
                actual_idx = start_idx + idx
                sample_id = sample.get("task_id", f"sample-{actual_idx}")

                if sample_id in processed_ids:
                    continue

                result = generator.generate_counterfactuals(sample)
                if result is not None:
                    results.append(result)
                    processed_ids.add(sample_id)
                else:
                    failed += 1

                if (actual_idx + 1) % args.checkpoint_interval == 0:
                    save_checkpoint(actual_idx + 1)
                    print(
                        f"\n  💾 Checkpoint saved at index {actual_idx + 1} "
                        f"({len(results)} results)"
                    )

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted! Saving checkpoint...")
        save_checkpoint(actual_idx + 1)
        print(f"  💾 Checkpoint saved to: {checkpoint_path}")
        print(f"  To resume, run with --resume flag")
        return

    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        save_checkpoint(actual_idx + 1)
        print(f"  💾 Emergency checkpoint saved to: {checkpoint_path}")
        raise

    # Save final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Successfully processed: {len(results)}/{len(data)} samples")
    print(f"  Failed: {failed}")
    print(f"  Output: {args.output}")

    # Per-argument stats
    total_perts = sum(
        s.get("counterfactual_info", {}).get("successful_counterfactuals", 0)
        for s in results
    )
    total_conds = sum(
        s.get("counterfactual_info", {}).get("total_arguments", 0)
        for s in results
    )
    print(f"  Total counterfactuals: {total_perts} across {total_conds} arguments")
    print(f"{'=' * 60}")

    # Save example for inspection
    if results:
        example_path = args.output.replace(".json", "_example.json")
        with open(example_path, "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"✓ Example saved to: {example_path}")

    # Remove checkpoint file on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Checkpoint file removed (completed successfully)")


if __name__ == "__main__":
    main()
