"""
SQL execution-based evaluator for BIRD-SQL dataset.
Evaluates model-generated SQL by executing against the database
and comparing result sets with gold SQL.

Follows the function-based evaluator pattern used in the other domain evaluators.
"""

import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

# Import db_utils directly by file path to avoid triggering the parent
# package __init__.py (which has dependencies only available when running
# from the intent_construction/intent_extraction/ directory).
_db_utils_path = (
    Path(__file__).parent.parent.parent
    / "intent_construction"
    / "intent_extraction"
    / "dataset_impl"
    / "bird_sql"
    / "db_utils.py"
)
_spec = importlib.util.spec_from_file_location("bird_sql_db_utils", _db_utils_path)
_db_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_db_utils)

SQLResult = _db_utils.SQLResult
compare_results = _db_utils.compare_results
execute_sql = _db_utils.execute_sql
validate_result = _db_utils.validate_result


# =============================================================================
# Result Data Class
# =============================================================================

@dataclasses.dataclass
class SQLEvalResult:
    """Result of SQL evaluation."""
    execution_match: bool           # Did results match?
    model_sql_valid: bool           # Did model SQL execute without error?
    model_result_empty: bool        # Was model result empty?
    gold_result_empty: bool         # Was gold result empty? (shouldn't happen)
    model_sql: str                  # Cleaned model SQL
    gold_sql: str                   # Gold SQL
    model_answer: str | None        # Stringified model result
    gold_answer: str | None         # Stringified gold result
    error: str | None = None        # Any error message


# =============================================================================
# SQL extraction from model response
# =============================================================================

# Patterns for extracting SQL from model response, tried in order
_SQL_BLOCK_PATTERN = re.compile(
    r"```(?:sql)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL | re.IGNORECASE,
)
_SELECT_PATTERN = re.compile(
    r"((?:WITH\s+.+?\)\s+)?SELECT\s+.+?)(?:;|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def extract_sql_from_response(response: str) -> str | None:
    """
    Extract a SQL query from a model's text response.

    Tries the following patterns in order:
    1. Fenced code block: ```sql ... ``` or ``` ... ```
    2. SELECT statement (possibly multi-line)
    3. Entire response as-is (if it looks like SQL)

    Post-processing: removes inline comments, normalizes whitespace.

    Args:
        response: The model's raw text response.

    Returns:
        Cleaned SQL string, or None if no SQL could be extracted.
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # 1. Try fenced code block
    match = _SQL_BLOCK_PATTERN.search(text)
    if match:
        sql = match.group(1).strip()
        if sql:
            return _clean_sql(sql)

    # 2. Try SELECT statement
    match = _SELECT_PATTERN.search(text)
    if match:
        sql = match.group(1).strip()
        if sql:
            return _clean_sql(sql)

    # 3. Last resort: if the whole response looks like SQL (starts with keyword)
    upper = text.upper().lstrip()
    sql_keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE")
    if any(upper.startswith(kw) for kw in sql_keywords):
        return _clean_sql(text)

    return None


def _clean_sql(sql: str) -> str:
    """Clean and normalize a SQL string."""
    # Remove single-line comments (-- ...)
    lines = sql.split("\n")
    cleaned_lines = []
    for line in lines:
        # Remove inline comments but preserve strings containing --
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        cleaned_lines.append(line)

    sql = "\n".join(cleaned_lines)

    # Normalize whitespace
    sql = " ".join(sql.split())

    # Remove trailing semicolons
    sql = sql.rstrip(";").strip()

    return sql


# =============================================================================
# Result stringification
# =============================================================================

def stringify_result(result: SQLResult) -> str:
    """
    Convert a SQL result to a standardized string for answer comparison.

    Handles:
    - Single scalar values: just the value (e.g., "42")
    - Single rows: comma-separated values
    - Multi-row results: one row per line, comma-separated

    Args:
        result: A SQLResult to stringify.

    Returns:
        String representation of the result, or empty string if invalid.
    """
    if not result.success or result.rows is None or not result.rows:
        return ""

    # Single scalar value
    if result.row_count == 1 and len(result.rows[0]) == 1:
        val = result.rows[0][0]
        return _format_cell(val)

    # Single row
    if result.row_count == 1:
        return ", ".join(_format_cell(v) for v in result.rows[0])

    # Multi-row
    lines = []
    for row in result.rows:
        lines.append(", ".join(_format_cell(v) for v in row))
    return "\n".join(lines)


def _format_cell(value) -> str:
    """Format a single cell value as a string."""
    if value is None:
        return "NULL"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return str(value)
    return str(value)


# =============================================================================
# Dataset routing
# =============================================================================

def is_sql_dataset(dataset_name: str) -> bool:
    """
    Check if a dataset name corresponds to the BIRD-SQL dataset.

    Used by run_experiment.py to route evaluation to the SQL evaluator.

    Args:
        dataset_name: Name of the dataset (e.g., "bird_sql", "gsm8k").

    Returns:
        True if this is a SQL-based dataset.
    """
    return bool(dataset_name) and dataset_name.lower() in ("bird_sql", "bird-sql", "birdsql")


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate_sql_response(
    model_response: str,
    gold_sql: str,
    db_path: str | Path,
) -> SQLEvalResult:
    """
    Evaluate a model-generated SQL response against gold SQL.

    Extracts SQL from the model response, executes both model and gold SQL
    against the database, and compares result sets.

    Args:
        model_response: The model's raw text response.
        gold_sql: The gold-standard SQL query.
        db_path: Path to the SQLite database to execute against.

    Returns:
        SQLEvalResult with match status, validity, and detailed results.
    """
    db_path = Path(db_path)

    # Extract SQL from model response
    model_sql = extract_sql_from_response(model_response)
    if model_sql is None:
        return SQLEvalResult(
            execution_match=False,
            model_sql_valid=False,
            model_result_empty=True,
            gold_result_empty=False,
            model_sql="",
            gold_sql=gold_sql,
            model_answer=None,
            gold_answer=None,
            error="Could not extract SQL from model response",
        )

    # Execute gold SQL
    gold_result = execute_sql(db_path, gold_sql)
    if not gold_result.success:
        return SQLEvalResult(
            execution_match=False,
            model_sql_valid=False,
            model_result_empty=True,
            gold_result_empty=True,
            model_sql=model_sql,
            gold_sql=gold_sql,
            model_answer=None,
            gold_answer=None,
            error=f"Gold SQL execution failed: {gold_result.error}",
        )

    gold_answer = stringify_result(gold_result)
    gold_empty = not validate_result(gold_result)

    # Execute model SQL
    model_result = execute_sql(db_path, model_sql)
    if not model_result.success:
        return SQLEvalResult(
            execution_match=False,
            model_sql_valid=False,
            model_result_empty=True,
            gold_result_empty=gold_empty,
            model_sql=model_sql,
            gold_sql=gold_sql,
            model_answer=None,
            gold_answer=gold_answer,
            error=f"Model SQL execution failed: {model_result.error}",
        )

    model_answer = stringify_result(model_result)
    model_empty = not validate_result(model_result)

    # Compare results (order insensitive)
    match = compare_results(model_result, gold_result, order_sensitive=False)

    return SQLEvalResult(
        execution_match=match,
        model_sql_valid=True,
        model_result_empty=model_empty,
        gold_result_empty=gold_empty,
        model_sql=model_sql,
        gold_sql=gold_sql,
        model_answer=model_answer,
        gold_answer=gold_answer,
    )
