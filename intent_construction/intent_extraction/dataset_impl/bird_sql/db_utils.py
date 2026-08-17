"""
SQLite database utilities for BIRD-SQL dataset.
Handles DB queries, schema extraction, and result comparison.

Zero LLM dependency — all operations are deterministic.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SQLResult:
    """Result of a SQL execution."""
    success: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0


@dataclass
class ColumnInfo:
    """Schema information for a database column."""
    name: str
    dtype: str
    table: str
    is_primary_key: bool = False
    is_nullable: bool = True


# =============================================================================
# Numeric type detection
# =============================================================================

_NUMERIC_TYPE_PREFIXES = ("INT", "REAL", "FLOAT", "NUMERIC", "DECIMAL", "DOUBLE", "NUMBER")


def _is_numeric_type(dtype: str) -> bool:
    """Check if a SQLite column type string is numeric."""
    upper = dtype.upper().strip()
    return any(upper.startswith(prefix) for prefix in _NUMERIC_TYPE_PREFIXES)


# =============================================================================
# Core functions
# =============================================================================

def execute_sql(db_path: str | Path, sql: str, timeout: float = 30) -> SQLResult:
    """
    Execute SQL against a SQLite database.

    Opens the database in read-only mode (via URI) when possible, handles
    timeouts and errors, and returns a structured result.

    Args:
        db_path: Path to the SQLite database file.
        sql: SQL query to execute.
        timeout: Connection and query execution timeout in seconds.

    Returns:
        SQLResult with rows, columns, and counts on success; error on failure.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return SQLResult(success=False, error=f"Database not found: {db_path}")

    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    except sqlite3.OperationalError:
        # Fallback: some systems don't support URI mode
        conn = sqlite3.connect(str(db_path), timeout=timeout)

    try:
        conn.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + timeout
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() >= deadline else 0,
            1_000,
        )
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return SQLResult(
            success=True,
            rows=rows,
            columns=columns,
            row_count=len(rows),
        )
    except sqlite3.OperationalError as e:
        return SQLResult(success=False, error=f"SQL error: {e}")
    except sqlite3.DatabaseError as e:
        return SQLResult(success=False, error=f"Database error: {e}")
    except Exception as e:
        return SQLResult(success=False, error=f"Unexpected error: {e}")
    finally:
        conn.close()


def get_schema_text(db_path: str | Path, max_tables: int | None = None) -> str:
    """
    Generate human-readable schema text for prompt inclusion.

    Produces CREATE TABLE statements and up to 3 sample rows per table.

    Args:
        db_path: Path to the SQLite database file.
        max_tables: Optional limit on number of tables to include.

    Returns:
        Multi-line string with CREATE TABLE statements and sample data.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return f"-- Database not found: {db_path}"

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
    except sqlite3.Error as e:
        return f"-- Cannot open database: {e}"

    try:
        cursor = conn.cursor()
        tables = _get_table_names(cursor)

        if max_tables is not None:
            tables = tables[:max_tables]

        parts: list[str] = []
        for table in tables:
            # Get CREATE TABLE statement
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            row = cursor.fetchone()
            create_stmt = row[0] if row and row[0] else f"-- No CREATE statement for {table}"
            parts.append(f"{create_stmt};")

            # Sample rows
            try:
                cursor.execute(f"SELECT * FROM [{table}] LIMIT 3")
                sample_rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                if sample_rows:
                    parts.append(f"-- Sample rows from {table}:")
                    parts.append(f"-- Columns: {', '.join(col_names)}")
                    for srow in sample_rows:
                        parts.append(f"--   {srow}")
            except sqlite3.Error:
                pass  # Skip sample rows if table can't be queried

            parts.append("")  # blank line between tables

        return "\n".join(parts).strip()
    finally:
        conn.close()


def get_table_columns(db_path: str | Path, table: str) -> list[ColumnInfo]:
    """
    Get column information for a table via PRAGMA table_info.

    Args:
        db_path: Path to the SQLite database file.
        table: Table name.

    Returns:
        List of ColumnInfo objects describing each column.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
    except sqlite3.Error:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info([{table}])")
        # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
        rows = cursor.fetchall()
        columns: list[ColumnInfo] = []
        for row in rows:
            columns.append(ColumnInfo(
                name=row[1],
                dtype=row[2] or "",
                table=table,
                is_primary_key=bool(row[5]),
                is_nullable=not bool(row[3]),
            ))
        return columns
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def get_numeric_columns(db_path: str | Path, table: str) -> list[str]:
    """
    Return column names with numeric types, excluding primary keys.

    Numeric types: INTEGER, REAL, FLOAT, NUMERIC, DECIMAL, DOUBLE, NUMBER.
    Primary key columns are filtered out as they are usually not useful for
    function swaps (e.g., auto-increment IDs).

    Args:
        db_path: Path to the SQLite database file.
        table: Table name.

    Returns:
        List of numeric column names.
    """
    columns = get_table_columns(db_path, table)
    return [
        col.name
        for col in columns
        if _is_numeric_type(col.dtype) and not col.is_primary_key
    ]


def get_alternative_values(
    db_path: str | Path,
    table: str,
    column: str,
    exclude_value: str,
    limit: int = 10,
) -> list:
    """
    Query for distinct values in a column, excluding the current value.

    Args:
        db_path: Path to the SQLite database file.
        table: Table name.
        column: Column name.
        exclude_value: Value to exclude from results.
        limit: Maximum number of alternatives to return.

    Returns:
        List of alternative values (native Python types from sqlite3).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
    except sqlite3.Error:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT DISTINCT [{column}] FROM [{table}] "
            f"WHERE [{column}] != ? AND [{column}] IS NOT NULL LIMIT ?",
            (exclude_value, limit),
        )
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def compare_results(
    result_a: SQLResult,
    result_b: SQLResult,
    order_sensitive: bool = False,
    column_order_sensitive: bool = False,
) -> bool:
    """
    Compare two SQL results for equality.

    By default, uses set-based comparison (order insensitive) and ignores
    column order within a row. Handles type normalization so that 42, 42.0,
    and "42" are considered equal.

    Args:
        result_a: First SQL result.
        result_b: Second SQL result.
        order_sensitive: If True, row order must match.
        column_order_sensitive: If True, the order of columns within each
            row must match. If False (default), columns within a row are
            compared as a multiset, so SELECT a, b and SELECT b, a return
            equivalent results.

    Returns:
        True if results are considered equal.
    """
    if not result_a.success or not result_b.success:
        return False
    if result_a.rows is None or result_b.rows is None:
        return result_a.rows is None and result_b.rows is None

    rows_a = [tuple(_normalize_value(v) for v in row) for row in result_a.rows]
    rows_b = [tuple(_normalize_value(v) for v in row) for row in result_b.rows]

    if not column_order_sensitive:
        # Sort cells within each row so column permutations match.
        rows_a = [tuple(sorted(row)) for row in rows_a]
        rows_b = [tuple(sorted(row)) for row in rows_b]

    if order_sensitive:
        return rows_a == rows_b

    # Set-based comparison: convert to multisets (sorted lists of tuples)
    return sorted(rows_a) == sorted(rows_b)


def _normalize_value(value) -> str | None:
    """
    Normalize a single cell value for comparison.

    Converts numbers and strings to a canonical string form so that
    42, 42.0, and "42" all compare equal.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        # Convert float to int if it's a whole number (42.0 → "42")
        if value == int(value):
            return str(int(value))
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # Try to normalize numeric strings
        stripped = value.strip()
        try:
            f = float(stripped)
            if f == int(f):
                return str(int(f))
            return str(f)
        except ValueError:
            return stripped
    return str(value)


def validate_result(result: SQLResult, reject_null: bool = True) -> bool:
    """
    Check that a SQL result is successful, non-empty, and error-free.

    Args:
        result: A SQLResult to validate.
        reject_null: If True, also reject results where all values are None/NULL.

    Returns:
        True if result is valid (success=True, has rows, no error, not all-NULL).
    """
    if not (
        result.success
        and result.error is None
        and result.rows is not None
        and result.row_count > 0
    ):
        return False

    if reject_null and result.rows:
        # Reject if every cell in every row is None
        all_null = all(
            all(v is None for v in row)
            for row in result.rows
        )
        if all_null:
            return False

    return True


def get_all_tables(db_path: str | Path) -> list[str]:
    """
    List all user tables in the database (excludes sqlite_ internal tables).

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of table names.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
    except sqlite3.Error:
        return []

    try:
        cursor = conn.cursor()
        return _get_table_names(cursor)
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _get_table_names(cursor: sqlite3.Cursor) -> list[str]:
    """Get table names from an open cursor."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def count_distinct_values(db_path: str | Path, table: str, column: str) -> int:
    """
    Count distinct non-null values for a column.

    Args:
        db_path: Path to the SQLite database file.
        table: Table name.
        column: Column name.

    Returns:
        Number of distinct non-null values, or 0 on error.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return 0

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
    except sqlite3.Error:
        return 0

    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT COUNT(DISTINCT [{column}]) FROM [{table}] WHERE [{column}] IS NOT NULL"
        )
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
