"""
Partial SQL query construction for per-turn answer computation.

Given a gold SQL query and a set of revealed argument indices,
strips unrevealed WHERE predicates and executes the partial query
to get the intermediate answer.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------------
# Minimal SQL execution (mirrored from intent_construction/intent_extraction/.../db_utils.py)
# ---------------------------------------------------------------------------

@dataclass
class SQLResult:
    success: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0


def execute_sql(db_path: str | Path, sql: str, timeout: int = 30) -> SQLResult:
    db_path = Path(db_path)
    if not db_path.exists():
        return SQLResult(success=False, error=f"Database not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return SQLResult(success=True, rows=rows, columns=columns, row_count=len(rows))
    except Exception as e:
        return SQLResult(success=False, error=str(e))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Result formatting (mirrored from intent_construction/intent_extraction/.../extractor.py)
# ---------------------------------------------------------------------------

def format_sql_result(result: SQLResult) -> str:
    if not result.rows:
        return ""

    def _cell(v: object) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)

    if len(result.rows) == 1 and len(result.rows[0]) == 1:
        return _cell(result.rows[0][0])

    lines: list[str] = []
    for row in result.rows:
        if len(row) == 1:
            lines.append(_cell(row[0]))
        else:
            lines.append(", ".join(_cell(v) for v in row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AST manipulation
# ---------------------------------------------------------------------------

def _flatten_and(node: exp.Expression) -> list[exp.Expression]:
    """Recursively flatten AND nodes into a list of individual predicates."""
    if isinstance(node, exp.And):
        return _flatten_and(node.this) + _flatten_and(node.expression)
    return [node]


def strip_unrevealed_arguments(
    gold_sql: str,
    keep_indices: list[int],
    dialect: str = "sqlite",
) -> str | None:
    """Remove WHERE predicates not in *keep_indices* from *gold_sql*.

    Args:
        gold_sql: Full gold SQL query string.
        keep_indices: 0-based indices of WHERE predicates to KEEP.
            Positionally aligned with extraction order from parse_sql().
        dialect: sqlglot dialect.

    Returns:
        Modified SQL string, or None on parse failure.
    """
    try:
        parsed = sqlglot.parse_one(gold_sql, dialect=dialect)
    except Exception:
        return None

    where = parsed.find(exp.Where)
    if where is None:
        # No WHERE clause — nothing to strip.
        return parsed.sql(dialect=dialect)

    predicates = _flatten_and(where.this)
    kept = [p for i, p in enumerate(predicates) if i in set(keep_indices)]

    if not kept:
        where.pop()
    else:
        new_where = reduce(lambda a, b: exp.And(this=a, expression=b), kept)
        where.set("this", new_where)

    return parsed.sql(dialect=dialect)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def substitute_sql_value(sql: str, original: str, counterfactual: str) -> str:
    """Replace a single argument value in SQL.

    Tries single-quoted form first (string values), then unquoted
    (numeric values).  Only the first occurrence is replaced.
    """
    quoted_orig = f"'{original}'"
    quoted_pert = f"'{counterfactual}'"
    if quoted_orig in sql:
        return sql.replace(quoted_orig, quoted_pert, 1)
    if str(original) in sql:
        return sql.replace(str(original), str(counterfactual), 1)
    return sql


def prune_unused_joins(
    sql: str,
    dialect: str = "sqlite",
) -> str:
    """Remove JOINs whose tables are not referenced in SELECT or WHERE.

    Iteratively removes JOINs until stable, handling chain dependencies
    where removing one JOIN makes another's table unreferenced.

    Returns the original string on parse failure.
    """

    def _tables_in(node: exp.Expression) -> set[str]:
        return {c.table for c in node.find_all(exp.Column) if c.table}

    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return sql

    for _ in range(10):
        select = parsed.find(exp.Select)
        needed: set[str] = _tables_in(select) if select else set()

        where = parsed.find(exp.Where)
        if where:
            needed |= _tables_in(where)

        from_clause = parsed.find(exp.From)
        if from_clause:
            ft = from_clause.find(exp.Table)
            if ft:
                needed.add(ft.alias_or_name)

        joins = list(parsed.find_all(exp.Join))
        changed = True
        while changed:
            changed = False
            for j in joins:
                jt = j.find(exp.Table)
                alias = jt.alias_or_name if jt else None
                if alias in needed:
                    on_clause = j.args.get("on")
                    if on_clause:
                        new = _tables_in(on_clause) - needed
                        if new:
                            needed |= new
                            changed = True

        removed = False
        for j in joins:
            jt = j.find(exp.Table)
            alias = jt.alias_or_name if jt else None
            if alias and alias not in needed:
                j.pop()
                removed = True

        if not removed:
            break

    return parsed.sql(dialect=dialect)


# ---------------------------------------------------------------------------
# Per-turn gold construction
# ---------------------------------------------------------------------------

def build_per_turn_gold(
    raw: dict,
    intent_trajectory: list[dict],
    selected_functions: list[dict],
    counterfactual_map: dict[int, list[dict]],
    ground_truth: str,
) -> list[dict[str, str]]:
    """Compute per-turn gold SQL and answer for a SQL sample.

    For each turn in *intent_trajectory*, constructs the gold SQL that
    reflects the current function, revealed arguments, and counterfactual values,
    then executes it against the database to get the answer.

    Args:
        raw: Raw sample dict (must contain ``gold_sql``, ``db_path``).
        intent_trajectory: Serialised ``ChangePlan.intent_trajectory``
            (list of dicts, each with ``function``, ``revealed_ids``,
            ``active_values``, ``target_answer``).
        selected_functions: Counterfactual-function dicts chosen for this sample
            (each has ``predecessor_function``, ``counterfactual_sql``).
        counterfactual_map: ``{cond_id: [counterfactual_dict, ...]}`` from
            ``select_counterfactuals``.  Each *counterfactual_dict* has
            ``counterfactual_argument``, ``original_value``,
            ``counterfactual_value``.
        ground_truth: Final-turn answer string.

    Returns:
        List of ``{"sql": ..., "answer": ...}`` (one per turn), or
        ``[]`` on failure / non-SQL sample.
    """
    if raw.get("data_source") != "bird_sql":
        return []

    gold_sql = raw.get("gold_sql", "")
    db_path = raw.get("db_path", "")
    if not gold_sql or not db_path:
        return []

    # --- build lookups ------------------------------------------------
    function_lookup: dict[str, dict] = {
        pg["predecessor_function"]: pg
        for pg in selected_functions
        if pg.get("predecessor_function")
    }

    # cid → {counterfactual_argument_text → {original_value, counterfactual_value}}
    value_lookup: dict[int, dict[str, dict[str, str]]] = {}
    for cid, counterfactuals in counterfactual_map.items():
        value_lookup[cid] = {}
        for p in counterfactuals:
            text = p.get("counterfactual_argument", "")
            if text:
                value_lookup[cid][text] = {
                    "original_value": p.get("original_value", ""),
                    "counterfactual_value": p.get("counterfactual_value", ""),
                }

    # --- per-turn computation -----------------------------------------
    n_turns = len(intent_trajectory)
    per_turn: list[dict[str, str]] = []

    for t, intent in enumerate(intent_trajectory):
        is_final = t == n_turns - 1

        # Final turn: always use source gold.
        if is_final:
            per_turn.append({"sql": gold_sql, "answer": ground_truth})
            continue

        func = intent.get("function", "")
        revealed = intent.get("revealed_ids") or []
        active = intent.get("active_values")  # dict | None

        # 1. Base SQL from active function
        pg = function_lookup.get(func)
        sql = pg["counterfactual_sql"] if pg else gold_sql

        # 2. Value substitution for counterfactual arguments
        if active:
            items = active.items() if isinstance(active, dict) else active
            for cid_key, active_text in items:
                cid = int(cid_key)
                if cid in value_lookup and active_text in value_lookup[cid]:
                    sub = value_lookup[cid][active_text]
                    sql = substitute_sql_value(
                        sql, sub["original_value"], sub["counterfactual_value"],
                    )

        # 3. Strip unrevealed arguments (positional)
        keep_indices = [int(cid) - 1 for cid in sorted(revealed)]
        partial = strip_unrevealed_arguments(sql, keep_indices)
        if partial is None:
            per_turn.append({"sql": "", "answer": ""})
            continue

        # 4. Prune orphaned JOINs
        partial = prune_unused_joins(partial)

        # 5. Execute and format
        result = execute_sql(db_path, partial)
        answer = format_sql_result(result) if result.success else ""

        per_turn.append({"sql": partial, "answer": answer})

    return per_turn


def compute_turn_answer(
    gold_sql: str,
    db_path: str,
    revealed_cond_ids: frozenset[int],
    total_arguments: int,
) -> str:
    """Compute the SQL answer for a turn's revealed state.

    Args:
        gold_sql: Full gold SQL.
        db_path: Path to SQLite database file.
        revealed_cond_ids: 1-based argument IDs revealed so far.
        total_arguments: Total number of arguments in the sample.

    Returns:
        Formatted answer string, or "" on failure.
    """
    if not gold_sql or not db_path:
        return ""

    # Convert 1-based argument IDs to 0-based predicate indices.
    keep_indices = [cid - 1 for cid in sorted(revealed_cond_ids)]

    partial_sql = strip_unrevealed_arguments(gold_sql, keep_indices)
    if partial_sql is None:
        return ""

    result = execute_sql(db_path, partial_sql)
    if not result.success:
        return ""

    return format_sql_result(result)
