"""
SQL parsing utilities for BIRD-SQL dataset.
Uses sqlglot for AST-based SQL analysis and manipulation.

Zero LLM dependency — all operations are deterministic.
"""

from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlglot import exp


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class SQLCondition:
    """A single WHERE condition extracted from SQL."""
    argument_id: int
    column: str
    operator: str          # '=', '!=', '>', '<', '>=', '<=', 'LIKE', 'IN', 'BETWEEN', 'IS'
    value: str | tuple     # single value or tuple for IN/BETWEEN
    table: str             # table this column belongs to
    raw_expression: str    # original SQL fragment


@dataclass(frozen=True)
class SQLGoal:
    """The SELECT clause goal extracted from SQL."""
    aggregate: str | None  # 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', or None
    column: str            # target column ('*' for COUNT(*))
    table: str | None      # table the column belongs to (if identifiable)
    raw_expression: str    # original SQL fragment


@dataclass(frozen=True)
class SQLHavingPredicate:
    """A single predicate inside the HAVING clause (typically aggregate-bearing).

    Examples
    --------
    ``HAVING COUNT(*) > 3``      → aggregate='COUNT', column='*',     operator='>',  value='3'
    ``HAVING AVG(salary) >= 50`` → aggregate='AVG',   column='salary', operator='>=', value='50'
    """
    predicate_id: int
    aggregate: str | None        # 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', or None for non-agg LHS
    column: str                  # '*' for COUNT(*); plain column name otherwise
    operator: str                # '=', '!=', '>', '<', '>=', '<=', 'LIKE', 'BETWEEN', 'IN'
    value: str | tuple           # single value or tuple for IN/BETWEEN
    raw_expression: str          # original SQL fragment, e.g. 'COUNT(*) > 3'


@dataclass(frozen=True)
class SQLJoin:
    """A JOIN condition."""
    join_type: str         # 'INNER', 'LEFT', 'RIGHT', 'CROSS'
    left_table: str
    right_table: str
    on_condition: str      # the ON clause as string


@dataclass
class ParsedSQL:
    """Complete parsed representation of a SQL query."""
    original_sql: str
    goal: SQLGoal
    conditions: tuple[SQLCondition, ...]
    joins: tuple[SQLJoin, ...]
    tables: tuple[str, ...]
    has_group_by: bool = False
    has_having: bool = False
    has_order_by: bool = False
    has_subquery: bool = False
    has_union: bool = False
    has_cte: bool = False
    group_by_columns: tuple[str, ...] = ()
    order_by_columns: tuple[str, ...] = ()
    having_predicates: tuple["SQLHavingPredicate", ...] = ()
    limit: int | None = None
    parseable: bool = True
    parse_error: str | None = None


# =============================================================================
# Operator mapping: sqlglot expression type → SQL operator string
# =============================================================================

_PREDICATE_TO_OPERATOR: dict[type, str] = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.Like: "LIKE",
    exp.In: "IN",
    exp.Between: "BETWEEN",
    exp.Is: "IS",
}

_OPERATOR_TO_PREDICATE: dict[str, type] = {
    "=": exp.EQ,
    "!=": exp.NEQ,
    "<>": exp.NEQ,
    ">": exp.GT,
    ">=": exp.GTE,
    "<": exp.LT,
    "<=": exp.LTE,
    "LIKE": exp.Like,
    "IN": exp.In,
    "BETWEEN": exp.Between,
    "IS": exp.Is,
}

_AGG_NAME_TO_CLASS: dict[str, type] = {
    "COUNT": exp.Count,
    "SUM": exp.Sum,
    "AVG": exp.Avg,
    "MAX": exp.Max,
    "MIN": exp.Min,
}


# =============================================================================
# Internal helpers
# =============================================================================

def _build_alias_map(parsed: exp.Expression) -> dict[str, str]:
    """Build a mapping from table alias → real table name."""
    alias_map: dict[str, str] = {}
    for table in parsed.find_all(exp.Table):
        name = table.name
        alias = table.alias
        if alias:
            alias_map[alias] = name
        # Identity mapping so lookups always work
        alias_map[name] = name
    return alias_map


def _resolve_table(column_table: str, alias_map: dict[str, str]) -> str:
    """Resolve a column's table reference through the alias map."""
    if not column_table:
        return ""
    return alias_map.get(column_table, column_table)


def _extract_value(node: exp.Expression) -> str:
    """Extract a human-readable value from a literal or expression node."""
    if isinstance(node, exp.Literal):
        return node.this
    if isinstance(node, exp.Null):
        return "NULL"
    if isinstance(node, exp.Boolean):
        return str(node.this)
    if isinstance(node, exp.Neg):
        inner = _extract_value(node.this)
        return f"-{inner}"
    return node.sql()


def _flatten_and(node: exp.Expression) -> list[exp.Expression]:
    """Recursively flatten AND nodes into a list of individual predicates."""
    if isinstance(node, exp.And):
        return _flatten_and(node.this) + _flatten_and(node.expression)
    return [node]


def _extract_from_tables(parsed: exp.Expression) -> list[str]:
    """Extract all table names from the query (FROM + JOINs), resolving aliases."""
    tables: list[str] = []
    for table in parsed.find_all(exp.Table):
        name = table.name
        if name and name not in tables:
            tables.append(name)
    return tables


# =============================================================================
# Core extraction functions
# =============================================================================

def extract_where_conditions(parsed: exp.Expression) -> list[SQLCondition]:
    """
    Walk the WHERE clause AST and extract individual conditions.

    For AND, splits into individual conditions. For OR, keeps as a single
    compound condition with operator 'OR'.

    Handles: EQ, NEQ, GT, GTE, LT, LTE, Like, In, Between, Is.
    Resolves table aliases to actual table names.

    Args:
        parsed: A sqlglot expression (full parsed query).

    Returns:
        List of SQLCondition objects, one per predicate.
    """
    where = parsed.find(exp.Where)
    if where is None:
        return []

    alias_map = _build_alias_map(parsed)

    # Flatten top-level ANDs; keep OR as compound
    predicates = _flatten_and(where.this)

    conditions: list[SQLCondition] = []
    cid = 0

    for pred in predicates:
        condition = _predicate_to_condition(pred, cid, alias_map)
        if condition is not None:
            conditions.append(condition)
            cid += 1

    return conditions


def _predicate_to_condition(
    pred: exp.Expression,
    cid: int,
    alias_map: dict[str, str],
) -> SQLCondition | None:
    """Convert a single predicate AST node into a SQLCondition."""

    # OR: keep as compound condition
    if isinstance(pred, exp.Or):
        return SQLCondition(
            argument_id=cid,
            column="",
            operator="OR",
            value=pred.sql(),
            table="",
            raw_expression=pred.sql(),
        )

    # Parenthesized expression: unwrap
    if isinstance(pred, exp.Paren):
        return _predicate_to_condition(pred.this, cid, alias_map)

    # NOT expression
    if isinstance(pred, exp.Not):
        inner = _predicate_to_condition(pred.this, cid, alias_map)
        if inner is not None:
            return SQLCondition(
                argument_id=cid,
                column=inner.column,
                operator=f"NOT {inner.operator}",
                value=inner.value,
                table=inner.table,
                raw_expression=pred.sql(),
            )
        return SQLCondition(
            argument_id=cid,
            column="",
            operator="NOT",
            value=pred.sql(),
            table="",
            raw_expression=pred.sql(),
        )

    # Binary comparison predicates: EQ, NEQ, GT, GTE, LT, LTE, Like
    for pred_type, op_str in _PREDICATE_TO_OPERATOR.items():
        if pred_type in (exp.In, exp.Between, exp.Is):
            continue  # handled separately below
        if isinstance(pred, pred_type):
            col_node = pred.this
            val_node = pred.expression
            column = col_node.name if isinstance(col_node, exp.Column) else col_node.sql()
            table = ""
            if isinstance(col_node, exp.Column):
                table = _resolve_table(col_node.table, alias_map)
            value = _extract_value(val_node)
            return SQLCondition(
                argument_id=cid,
                column=column,
                operator=op_str,
                value=value,
                table=table,
                raw_expression=pred.sql(),
            )

    # IN predicate
    if isinstance(pred, exp.In):
        col_node = pred.this
        column = col_node.name if isinstance(col_node, exp.Column) else col_node.sql()
        table = ""
        if isinstance(col_node, exp.Column):
            table = _resolve_table(col_node.table, alias_map)
        # Collect IN-list values (skip subqueries)
        values = tuple(_extract_value(v) for v in pred.expressions)
        return SQLCondition(
            argument_id=cid,
            column=column,
            operator="IN",
            value=values if values else pred.sql(),
            table=table,
            raw_expression=pred.sql(),
        )

    # BETWEEN predicate
    if isinstance(pred, exp.Between):
        col_node = pred.this
        column = col_node.name if isinstance(col_node, exp.Column) else col_node.sql()
        table = ""
        if isinstance(col_node, exp.Column):
            table = _resolve_table(col_node.table, alias_map)
        low = _extract_value(pred.args["low"])
        high = _extract_value(pred.args["high"])
        return SQLCondition(
            argument_id=cid,
            column=column,
            operator="BETWEEN",
            value=(low, high),
            table=table,
            raw_expression=pred.sql(),
        )

    # IS (NULL / NOT NULL)
    if isinstance(pred, exp.Is):
        col_node = pred.this
        column = col_node.name if isinstance(col_node, exp.Column) else col_node.sql()
        table = ""
        if isinstance(col_node, exp.Column):
            table = _resolve_table(col_node.table, alias_map)
        value = _extract_value(pred.expression)
        return SQLCondition(
            argument_id=cid,
            column=column,
            operator="IS",
            value=value,
            table=table,
            raw_expression=pred.sql(),
        )

    # Fallback: unrecognized predicate type — wrap raw SQL
    return SQLCondition(
        argument_id=cid,
        column="",
        operator="UNKNOWN",
        value=pred.sql(),
        table="",
        raw_expression=pred.sql(),
    )


def extract_select_goal(parsed: exp.Expression) -> SQLGoal:
    """
    Identify the primary SELECT goal from the query.

    If the SELECT contains aggregate functions, uses the first one as the goal.
    Otherwise, uses the first selected column.

    Args:
        parsed: A sqlglot expression (full parsed query).

    Returns:
        SQLGoal describing the primary target of the query.
    """
    alias_map = _build_alias_map(parsed)
    select = parsed.find(exp.Select)
    if select is None:
        return SQLGoal(aggregate=None, column="*", table=None, raw_expression="")

    expressions = select.expressions
    if not expressions:
        return SQLGoal(aggregate=None, column="*", table=None, raw_expression="")

    # Look for aggregate functions across all SELECT expressions
    for select_expr in expressions:
        # Unwrap aliases
        inner = select_expr.this if isinstance(select_expr, exp.Alias) else select_expr

        if isinstance(inner, exp.AggFunc):
            return _agg_to_goal(inner, alias_map, select_expr.sql())

        # Check for aggregates nested inside the expression
        agg = inner.find(exp.AggFunc) if hasattr(inner, "find") else None
        if agg is not None:
            return _agg_to_goal(agg, alias_map, select_expr.sql())

    # No aggregate — use first column/expression
    first = expressions[0]
    inner = first.this if isinstance(first, exp.Alias) else first
    if isinstance(inner, exp.Column):
        table = _resolve_table(inner.table, alias_map) or None
        return SQLGoal(
            aggregate=None,
            column=inner.name,
            table=table,
            raw_expression=first.sql(),
        )
    if isinstance(inner, exp.Star):
        return SQLGoal(aggregate=None, column="*", table=None, raw_expression="*")

    return SQLGoal(
        aggregate=None,
        column=inner.sql(),
        table=None,
        raw_expression=first.sql(),
    )


def _agg_to_goal(
    agg: exp.AggFunc,
    alias_map: dict[str, str],
    raw: str,
) -> SQLGoal:
    """Convert an aggregate function AST node into a SQLGoal."""
    agg_name = type(agg).__name__.upper()
    # Normalize sqlglot class names to standard SQL names
    agg_name_map = {"COUNT": "COUNT", "SUM": "SUM", "AVG": "AVG", "MAX": "MAX", "MIN": "MIN"}
    agg_name = agg_name_map.get(agg_name, agg_name)

    arg = agg.this
    if isinstance(arg, exp.Star):
        return SQLGoal(aggregate=agg_name, column="*", table=None, raw_expression=raw)
    if isinstance(arg, exp.Column):
        table = _resolve_table(arg.table, alias_map) or None
        return SQLGoal(aggregate=agg_name, column=arg.name, table=table, raw_expression=raw)
    # Fallback: expression inside aggregate
    return SQLGoal(
        aggregate=agg_name,
        column=arg.sql() if arg else "*",
        table=None,
        raw_expression=raw,
    )


def extract_having_predicates(parsed: exp.Expression) -> list[SQLHavingPredicate]:
    """Walk the HAVING clause AST and extract individual predicates.

    Splits top-level ANDs into separate predicates. Each predicate's LHS is
    typically an aggregate function (``COUNT(*)``, ``AVG(col)``, etc.); when it
    is, ``aggregate`` and ``column`` are populated. For non-aggregate LHS
    (rare in HAVING), ``aggregate`` is ``None`` and ``column`` carries the raw
    LHS SQL.
    """
    having = parsed.find(exp.Having)
    if having is None:
        return []

    alias_map = _build_alias_map(parsed)
    predicates = _flatten_and(having.this)

    results: list[SQLHavingPredicate] = []
    pid = 0

    pred_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)

    for pred in predicates:
        # Unwrap parens
        if isinstance(pred, exp.Paren):
            pred = pred.this

        op_str: str | None = None
        lhs: exp.Expression | None = None
        rhs: exp.Expression | None = None
        value: str | tuple = ""

        if isinstance(pred, pred_types):
            op_str = _PREDICATE_TO_OPERATOR[type(pred)]
            lhs = pred.this
            rhs = pred.expression
            value = _extract_value(rhs)
        elif isinstance(pred, exp.Between):
            op_str = "BETWEEN"
            lhs = pred.this
            value = (_extract_value(pred.args["low"]), _extract_value(pred.args["high"]))
        elif isinstance(pred, exp.In):
            op_str = "IN"
            lhs = pred.this
            value = tuple(_extract_value(v) for v in pred.expressions) or pred.sql()
        else:
            # Unrecognized — keep raw
            results.append(SQLHavingPredicate(
                predicate_id=pid,
                aggregate=None,
                column="",
                operator="UNKNOWN",
                value=pred.sql(),
                raw_expression=pred.sql(),
            ))
            pid += 1
            continue

        agg_name: str | None = None
        col_name: str = ""

        if lhs is not None:
            for cls_name, cls in _AGG_NAME_TO_CLASS.items():
                if isinstance(lhs, cls):
                    agg_name = cls_name
                    inner = lhs.this
                    if isinstance(inner, exp.Star):
                        col_name = "*"
                    elif isinstance(inner, exp.Column):
                        col_name = inner.name
                    elif inner is not None:
                        col_name = inner.sql()
                    break
            if agg_name is None:
                # Non-aggregate LHS (column / expression)
                if isinstance(lhs, exp.Column):
                    col_name = lhs.name
                else:
                    col_name = lhs.sql()

        results.append(SQLHavingPredicate(
            predicate_id=pid,
            aggregate=agg_name,
            column=col_name,
            operator=op_str,
            value=value,
            raw_expression=pred.sql(),
        ))
        pid += 1

    # alias_map currently unused but kept for future column-table resolution
    del alias_map
    return results


def extract_joins(parsed: exp.Expression) -> list[SQLJoin]:
    """
    Extract all JOIN clauses with type and ON conditions.

    Args:
        parsed: A sqlglot expression (full parsed query).

    Returns:
        List of SQLJoin objects.
    """
    alias_map = _build_alias_map(parsed)
    joins: list[SQLJoin] = []

    # Determine the "left" table (first table in FROM clause)
    from_clause = parsed.find(exp.From)
    left_table = ""
    if from_clause:
        first_table = from_clause.find(exp.Table)
        if first_table:
            left_table = first_table.name

    for join_node in parsed.find_all(exp.Join):
        # Determine join type from side/kind attributes
        side = join_node.side or ""
        kind = join_node.kind or ""
        if side:
            join_type = side.upper()  # LEFT, RIGHT
        elif kind:
            join_type = kind.upper()  # INNER, CROSS
        else:
            join_type = "INNER"  # default plain JOIN is INNER

        # Right table is the table being joined
        right_table_node = join_node.this
        right_table = right_table_node.name if isinstance(right_table_node, exp.Table) else right_table_node.sql()

        # ON condition
        on_clause = join_node.args.get("on")
        on_str = on_clause.sql() if on_clause else ""

        joins.append(SQLJoin(
            join_type=join_type,
            left_table=left_table,
            right_table=right_table,
            on_condition=on_str,
        ))

        # For chained joins, the next join's "left" is this right table
        left_table = right_table

    return joins


# =============================================================================
# Main entry point
# =============================================================================

def parse_sql(sql: str, dialect: str = "sqlite") -> ParsedSQL:
    """
    Parse SQL and return a structured representation.

    Main entry point for SQL analysis. Uses sqlglot to parse the query,
    then walks the AST to extract all components.

    Args:
        sql: The SQL query string to parse.
        dialect: SQL dialect for parsing (default: "sqlite").

    Returns:
        ParsedSQL with all extracted components.
    """
    sql = sql.strip()

    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError as e:
        return ParsedSQL(
            original_sql=sql,
            goal=SQLGoal(aggregate=None, column="*", table=None, raw_expression=""),
            conditions=(),
            joins=(),
            tables=(),
            parseable=False,
            parse_error=str(e),
        )

    # Handle UNION: the top-level node is exp.Union, not exp.Select
    is_union = isinstance(parsed, exp.Union) or parsed.find(exp.Union) is not None

    # Extract components
    goal = extract_select_goal(parsed)
    conditions = extract_where_conditions(parsed)
    join_list = extract_joins(parsed)
    tables = _extract_from_tables(parsed)

    # Detect structural features
    group_by = parsed.find(exp.Group)
    group_by_cols: tuple[str, ...] = ()
    if group_by:
        group_by_cols = tuple(e.sql() for e in group_by.expressions)

    order_by = parsed.find(exp.Order)
    order_by_cols: tuple[str, ...] = ()
    if order_by:
        order_by_cols = tuple(
            e.this.sql() if isinstance(e, exp.Ordered) else e.sql()
            for e in order_by.expressions
        )

    limit_node = parsed.find(exp.Limit)
    limit_val: int | None = None
    if limit_node and limit_node.expression:
        try:
            limit_val = int(limit_node.expression.this)
        except (ValueError, TypeError, AttributeError):
            pass

    has_subquery = len(list(parsed.find_all(exp.Subquery))) > 0
    has_cte = parsed.find(exp.CTE) is not None

    having_preds = extract_having_predicates(parsed)

    return ParsedSQL(
        original_sql=sql,
        goal=goal,
        conditions=tuple(conditions),
        joins=tuple(join_list),
        tables=tuple(tables),
        has_group_by=group_by is not None,
        has_having=parsed.find(exp.Having) is not None,
        has_order_by=order_by is not None,
        has_subquery=has_subquery,
        has_union=is_union,
        has_cte=has_cte,
        group_by_columns=group_by_cols,
        order_by_columns=order_by_cols,
        having_predicates=tuple(having_preds),
        limit=limit_val,
    )


# =============================================================================
# SQL reconstruction
# =============================================================================

def reconstruct_sql(parsed_sql: ParsedSQL) -> str:
    """
    Rebuild SQL from parsed components.

    Useful for verifying round-trip fidelity. Uses sqlglot to re-parse and
    regenerate the original SQL with consistent formatting.

    Args:
        parsed_sql: A ParsedSQL object.

    Returns:
        Reconstructed SQL string, or the original if not parseable.
    """
    if not parsed_sql.parseable:
        return parsed_sql.original_sql

    try:
        ast = sqlglot.parse_one(parsed_sql.original_sql, dialect="sqlite")
        return ast.sql(dialect="sqlite")
    except sqlglot.errors.ParseError:
        return parsed_sql.original_sql


# =============================================================================
# SQL transformation functions
# =============================================================================

def swap_aggregate(sql: str, original_agg: str, new_agg: str) -> str:
    """
    Replace an aggregate function in SQL.

    Uses sqlglot's transform to walk the AST and replace the aggregate
    function node while preserving its arguments.

    Examples:
        swap_aggregate("SELECT COUNT(*) FROM t", "COUNT", "MAX")
        → "SELECT MAX(*) FROM t"

    Args:
        sql: Original SQL string.
        original_agg: Current aggregate name (e.g., 'COUNT').
        new_agg: New aggregate name (e.g., 'MAX').

    Returns:
        Modified SQL string with the aggregate replaced.
    """
    original_upper = original_agg.upper()
    new_upper = new_agg.upper()

    original_cls = _AGG_NAME_TO_CLASS.get(original_upper)
    new_cls = _AGG_NAME_TO_CLASS.get(new_upper)
    if original_cls is None or new_cls is None:
        # Fallback: string-based replacement
        return sql.replace(original_agg, new_agg, 1)

    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return sql.replace(original_agg, new_agg, 1)

    replaced = False

    def _replace(node: exp.Expression) -> exp.Expression:
        nonlocal replaced
        if not replaced and isinstance(node, original_cls):
            replaced = True
            return new_cls(this=node.this.copy() if node.this else node.this)
        return node

    result = parsed.transform(_replace)
    return result.sql(dialect="sqlite")


def swap_select_column(sql: str, original_col: str, new_col: str) -> str:
    """
    Replace a column inside an aggregate or SELECT clause.

    Walks the AST and replaces the first matching column reference.

    Examples:
        swap_select_column("SELECT MAX(salary) FROM t", "salary", "budget")
        → "SELECT MAX(budget) FROM t"

    Args:
        sql: Original SQL string.
        original_col: Current column name.
        new_col: New column name.

    Returns:
        Modified SQL string with the column replaced.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return sql.replace(original_col, new_col, 1)

    replaced = False

    def _replace(node: exp.Expression) -> exp.Expression:
        nonlocal replaced
        if not replaced and isinstance(node, exp.Column) and node.name == original_col:
            # Only replace columns that are in the SELECT clause (inside aggregates
            # or as direct select targets), not in WHERE/JOIN clauses.
            replaced = True
            return exp.Column(this=exp.Identifier(this=new_col))
        return node

    # Only transform within SELECT expressions to avoid changing WHERE clauses
    select = parsed.find(exp.Select)
    if select:
        for i, select_expr in enumerate(select.expressions):
            if not replaced:
                select.expressions[i] = select_expr.transform(_replace)
        return parsed.sql(dialect="sqlite")

    return sql.replace(original_col, new_col, 1)


def swap_where_value(sql: str, condition: SQLCondition, new_value: str) -> str:
    """
    Replace a WHERE condition's value.

    Uses sqlglot's transform to find and replace the specific literal value
    in the WHERE clause that matches the given condition.

    Examples:
        swap_where_value(sql, cond, "Sub-Saharan Africa")
        # WHERE Region = 'MENA' → WHERE Region = 'Sub-Saharan Africa'

    Args:
        sql: Original SQL string.
        condition: The SQLCondition whose value should be replaced.
        new_value: The new value to substitute.

    Returns:
        Modified SQL string with the WHERE value replaced.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return sql

    old_value = condition.value
    replaced = False

    def _is_target_predicate(node: exp.Expression) -> bool:
        """Check if this node is the predicate matching our condition."""
        if not hasattr(node, "this"):
            return False
        col_node = node.this
        if isinstance(col_node, exp.Column) and col_node.name == condition.column:
            return True
        return False

    def _make_literal(val: str) -> exp.Literal:
        """Create a Literal node — string or number based on content."""
        try:
            # Try parsing as number
            if "." in val:
                float(val)
            else:
                int(val)
            return exp.Literal.number(val)
        except ValueError:
            return exp.Literal.string(val)

    def _replace(node: exp.Expression) -> exp.Expression:
        nonlocal replaced
        if replaced:
            return node

        # Match binary comparison predicates (=, !=, >, <, >=, <=, LIKE)
        pred_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)
        if isinstance(node, pred_types) and _is_target_predicate(node):
            val_node = node.expression
            current_val = _extract_value(val_node)
            if isinstance(old_value, str) and current_val == old_value:
                replaced = True
                new_node = node.copy()
                new_node.set("expression", _make_literal(new_value))
                return new_node

        # Match IS predicate
        if isinstance(node, exp.Is) and _is_target_predicate(node):
            current_val = _extract_value(node.expression)
            if isinstance(old_value, str) and current_val == old_value:
                replaced = True
                new_node = node.copy()
                if new_value.upper() == "NULL":
                    new_node.set("expression", exp.Null())
                else:
                    new_node.set("expression", _make_literal(new_value))
                return new_node

        # Match BETWEEN predicate. ``new_value`` is expected to be either
        # ``"low AND high"`` (preferred) or ``"low,high"`` (legacy);
        # ``condition.value`` may be a tuple ``(low, high)`` or a string
        # in either of those forms.
        if isinstance(node, exp.Between) and _is_target_predicate(node):
            new_low = new_high = None
            v = new_value.strip()
            for sep in (" AND ", " and ", ","):
                if sep in v:
                    parts = v.split(sep, 1)
                    if len(parts) == 2:
                        new_low, new_high = (p.strip().strip("'\"") for p in parts)
                        break
            if new_low is not None and new_high is not None:
                replaced = True
                new_node = node.copy()
                new_node.set("low", _make_literal(new_low))
                new_node.set("high", _make_literal(new_high))
                return new_node

        # Match IN predicate. ``new_value`` is a comma-separated list of
        # values, optionally each quoted (e.g. ``"'A', 'B'"`` or ``"1, 2"``).
        if isinstance(node, exp.In) and _is_target_predicate(node):
            raw_items = [
                item.strip().strip("'\"")
                for item in new_value.split(",")
                if item.strip()
            ]
            if raw_items:
                replaced = True
                new_node = node.copy()
                new_node.set(
                    "expressions",
                    [_make_literal(item) for item in raw_items],
                )
                return new_node

        return node

    result = parsed.transform(_replace)
    return result.sql(dialect="sqlite")


def swap_where_operator(sql: str, condition: SQLCondition, new_operator: str) -> str:
    """
    Replace a WHERE condition's operator.

    Transforms the AST predicate node type to match the new operator while
    preserving the column and value.

    Examples:
        swap_where_operator(sql, cond, "<")
        # WHERE salary > 50000 → WHERE salary < 50000

    Args:
        sql: Original SQL string.
        condition: The SQLCondition whose operator should be replaced.
        new_operator: The new SQL operator (e.g., '<', '>=', 'LIKE').

    Returns:
        Modified SQL string with the operator replaced.
    """
    new_op_upper = new_operator.upper().strip()
    new_pred_cls = _OPERATOR_TO_PREDICATE.get(new_op_upper)
    if new_pred_cls is None:
        return sql

    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return sql

    replaced = False

    def _replace(node: exp.Expression) -> exp.Expression:
        nonlocal replaced
        if replaced:
            return node

        # Find the predicate matching our condition's column and raw_expression
        pred_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.Is)
        if isinstance(node, pred_types):
            col_node = node.this
            if isinstance(col_node, exp.Column) and col_node.name == condition.column:
                val_node = node.expression
                current_val = _extract_value(val_node)
                expected_val = condition.value if isinstance(condition.value, str) else ""
                if current_val == expected_val:
                    replaced = True
                    return new_pred_cls(
                        this=col_node.copy(),
                        expression=val_node.copy(),
                    )

        return node

    result = parsed.transform(_replace)
    return result.sql(dialect="sqlite")


def swap_having_value(sql: str, predicate: SQLHavingPredicate, new_value: str) -> str:
    """Replace the RHS literal of a HAVING predicate.

    Locates the predicate inside the HAVING clause that matches
    ``predicate`` (by aggregate, column, operator, value) and replaces its
    RHS literal with ``new_value``.

    Examples
    --------
    >>> swap_having_value("SELECT t,COUNT(*) FROM e GROUP BY t HAVING COUNT(*) > 3",
    ...                   pred, "5")
    'SELECT t,COUNT(*) FROM e GROUP BY t HAVING COUNT(*) > 5'
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return sql

    having = parsed.find(exp.Having)
    if having is None:
        return sql

    target_op = predicate.operator.upper()
    target_agg = (predicate.aggregate or "").upper()
    target_col = predicate.column
    target_val = predicate.value

    pred_types_simple = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)
    replaced = False

    def _make_literal(val: str) -> exp.Literal:
        try:
            if "." in val:
                float(val)
            else:
                int(val)
            return exp.Literal.number(val)
        except ValueError:
            return exp.Literal.string(val)

    def _matches_lhs(lhs: exp.Expression) -> bool:
        # Match aggregate-bearing LHS
        for agg_name, cls in _AGG_NAME_TO_CLASS.items():
            if isinstance(lhs, cls):
                if agg_name != target_agg:
                    return False
                inner = lhs.this
                if target_col == "*":
                    return isinstance(inner, exp.Star)
                if isinstance(inner, exp.Column):
                    return inner.name == target_col
                return inner is not None and inner.sql() == target_col
        # Non-aggregate LHS
        if target_agg == "" or target_agg is None:
            if isinstance(lhs, exp.Column):
                return lhs.name == target_col
            return lhs.sql() == target_col
        return False

    def _replace(node: exp.Expression) -> exp.Expression:
        nonlocal replaced
        if replaced:
            return node

        # Only consider nodes inside the HAVING subtree
        ancestor = node.parent
        in_having = False
        cur: exp.Expression | None = node
        while cur is not None:
            if isinstance(cur, exp.Having):
                in_having = True
                break
            cur = cur.parent
        if not in_having:
            return node
        del ancestor

        if isinstance(node, pred_types_simple):
            op_str = _PREDICATE_TO_OPERATOR[type(node)]
            if op_str.upper() != target_op:
                return node
            if not _matches_lhs(node.this):
                return node
            rhs = node.expression
            if not isinstance(target_val, str) or _extract_value(rhs) != target_val:
                return node
            replaced = True
            new_node = node.copy()
            new_node.set("expression", _make_literal(new_value))
            return new_node

        if isinstance(node, exp.Between) and target_op == "BETWEEN":
            if not _matches_lhs(node.this):
                return node
            # new_value may be 'low,high'
            try:
                low_str, high_str = (s.strip() for s in new_value.split(",", 1))
            except ValueError:
                return node
            replaced = True
            new_node = node.copy()
            new_node.set("low", _make_literal(low_str))
            new_node.set("high", _make_literal(high_str))
            return new_node

        return node

    result = parsed.transform(_replace)
    return result.sql(dialect="sqlite")


def swap_limit_value(sql: str, new_limit: int) -> str:
    """Replace the LIMIT integer in a SQL query.

    If the query has no LIMIT, returns ``sql`` unchanged.

    Examples
    --------
    >>> swap_limit_value("SELECT * FROM t LIMIT 5", 10)
    'SELECT * FROM t LIMIT 10'
    """
    if new_limit is None or int(new_limit) < 0:
        return sql

    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return sql

    limit_node = parsed.find(exp.Limit)
    if limit_node is None or limit_node.expression is None:
        return sql

    limit_node.set("expression", exp.Literal.number(str(int(new_limit))))
    return parsed.sql(dialect="sqlite")
