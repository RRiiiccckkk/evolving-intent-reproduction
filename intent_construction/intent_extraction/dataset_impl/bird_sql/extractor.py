"""
BIRD-SQL Extractor implementation.
Extracts Function + Arguments from BIRD-SQL text-to-SQL samples.

Fully programmatic (zero LLM) — uses sqlglot AST parsing to decompose
SQL queries into function (SELECT aggregate) and arguments (WHERE predicates),
then converts each to natural language via templates.
"""

import re
from pathlib import Path
from typing import Any

from intent_construction.intent_extraction.core.base_extractor import BaseExtractor
from .sql_parser import parse_sql, SQLCondition, SQLGoal, ParsedSQL
from .db_utils import execute_sql, get_schema_text, SQLResult
from .reproduction import (
    REQUIRED_MODEL,
    REPO_ROOT,
    BirdReproductionError,
    assert_required_model,
    task_id_for,
)


# Common English words that may coincide with SQL argument values
# (e.g., result='Pass' vs "passed the inspection"). These are exempt
# from value-leak validation to avoid false positives.
_COMMON_WORD_VALUES = frozenset({
    'pass', 'fail', 'male', 'female', 'bottle', 'stroke',
    'restaurant', 'alcohol', 'phone', 'none', 'legal', 'rare',
    'english', 'action', 'comedy', 'drama', 'active', 'open',
    'closed', 'pending', 'complete', 'true', 'false', 'yes',
})

# Trivial words the LLM may introduce for grammar without changing meaning
_TRIVIAL_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'it', 'its',
    'do', 'does', 'did', 'has', 'have', 'had', 'be', 'been',
    'there', 'their', 'they', 'them', 'this', 'that', 'those',
    'what', 'how', 'many', 'much', 'which', 'who', 'whom',
    'of', 'in', 'at', 'to', 'for', 'by', 'on', 'with', 'from',
    'and', 'or', 'not', 'no', 'any', 'all', 'some', 'each',
    'than', 'if', 'when', 'where', 'can', 'could', 'would',
    'should', 'will', 'shall', 'may', 'might', 'must',
    'about', 'into', 'over', 'under', 'between', 'through',
    'during', 'before', 'after', 'above', 'below',
    'he', 'she', 'him', 'her', 'his', 'hers',
})


def _normalize_for_matching(text: str) -> str:
    """Strip spaces, punctuation, and lowercase for fuzzy value matching."""
    return re.sub(r'[\s\-_.,;:\'\"()/%]', '', text.lower())


def _contains_normalized_value(text: str, value: str) -> bool:
    """Match a normalized value across whole adjacent tokens.

    Concatenating adjacent tokens still catches spacing and punctuation changes
    such as ``200 MG`` versus ``200mg``. Requiring token boundaries prevents
    short values such as ``Ms.`` from matching the suffix of ``claims``.
    """
    value_norm = _normalize_for_matching(value)
    if not value_norm:
        return False
    tokens = [
        normalized
        for token in re.findall(r'\w+', text.lower())
        if (normalized := _normalize_for_matching(token))
    ]
    for start in range(len(tokens)):
        merged = ""
        for token in tokens[start:]:
            merged += token
            if merged == value_norm:
                return True
            if len(merged) >= len(value_norm):
                break
    return False


def _check_value_leak(function: str, check_values: list[str]) -> list[str]:
    """Check if any argument sql_value still appears in the function text.

    Uses normalized matching to catch spacing/casing variants like
    ``"200 MG"`` vs ``"200mg"``. Exempts common English words.
    """
    leaked = []
    for val in check_values:
        if val.lower().strip() in _COMMON_WORD_VALUES:
            continue
        if _contains_normalized_value(function, val):
            leaked.append(val)
    return leaked


def _stem_variants(token: str) -> set[str]:
    """Generate simple morphological variants of a token."""
    stems = {token}
    if token.endswith('s'):
        stems.add(token[:-1])
    if token.endswith('es'):
        stems.add(token[:-2])
    if token.endswith('ed'):
        stems.add(token[:-2])
        stems.add(token[:-1])
    if token.endswith('ing'):
        stems.add(token[:-3])
    if token.endswith('ly'):
        stems.add(token[:-2])
    if token.endswith('ies'):
        stems.add(token[:-3] + 'y')
    if token.endswith('tion'):
        stems.add(token[:-4] + 't')
        stems.add(token[:-4] + 'te')
    return stems


def _check_meaning_change(original: str, function: str) -> set[str]:
    """Detect words in the function that don't come from the original question.

    The LLM should only *remove* argument text, not *replace* it with
    different words. Returns the set of suspicious new words.
    """
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))

    orig_tokens = tokenize(original)
    function_tokens = tokenize(function)

    # Build expanded set of original tokens with stem variants
    orig_expanded = set()
    for t in orig_tokens:
        orig_expanded |= _stem_variants(t)

    suspicious = set()
    for w in function_tokens - orig_tokens - _TRIVIAL_WORDS:
        if not (_stem_variants(w) & orig_expanded):
            suspicious.add(w)

    return suspicious


class BirdSqlExtractor(BaseExtractor):
    """
    Extractor for BIRD-SQL text-to-SQL samples.

    Deterministic extraction: parses gold SQL via sqlglot, extracts
    WHERE predicates as arguments and the SELECT aggregate as the function,
    then renders each component as natural language using templates.
    No LLM calls are made at any stage.

    Output format matches the pipeline schema expected by situated_simulation:
    task_id, function, arguments (with sql_* metadata), answer, db_path, etc.
    """

    # Paths relative to this file's directory (dataset_impl/bird_sql/)
    _DEV_DB_TEMPLATE = (
        "dev_extracted/dev_20240627/dev_databases/{db_id}/{db_id}.sqlite"
    )
    _TRAIN_DB_TEMPLATE = (
        "train_extracted/train/train_databases/{db_id}/{db_id}.sqlite"
    )

    # Operator → NL template mapping for simple binary comparisons
    _OP_TEMPLATES: dict[str, str] = {
        "=": "The {col} is {val}.",
        "!=": "The {col} is not {val}.",
        ">": "The {col} is greater than {val}.",
        ">=": "The {col} is at least {val}.",
        "<": "The {col} is less than {val}.",
        "<=": "The {col} is at most {val}.",
        "IS": "The {col} is {val}.",
    }

    def __init__(
        self,
        model: str = REQUIRED_MODEL,
        num_arguments: int = 4,
        max_verification_attempts: int = 1,
        verif_model: str = REQUIRED_MODEL,
        enable_model_verification: bool = False,
        strip_model: str = REQUIRED_MODEL,
        data_root: str | Path | None = None,
    ):
        """
        Initialize the BIRD-SQL extractor.

        Model verification is disabled by default because SQL extraction
        is deterministic — there is nothing to retry.

        Args:
            strip_model: LLM model used to strip argument values from
                the function text. Defaults to a cheap nano model.
        """
        assert_required_model(model, context="BIRD extractor model")
        assert_required_model(verif_model, context="BIRD extractor verifier model")
        assert_required_model(strip_model, context="BIRD extractor strip model")
        super().__init__(
            model=model,
            num_arguments=num_arguments,
            max_verification_attempts=max_verification_attempts,
            verif_model=verif_model,
            enable_model_verification=enable_model_verification,
        )
        self._strip_model = strip_model
        self._data_root = (
            Path(data_root).resolve()
            if data_root is not None
            else (Path(__file__).parent / "data").resolve()
        )

    # ── BaseExtractor abstract method implementations ──────────

    def get_dataset_name(self) -> str:
        return "bird_sql"

    def get_prompts_dir(self) -> Path:
        return Path(__file__).parent / "prompts"

    def _load_prompts(self) -> None:
        """No-op: SQL extraction is fully programmatic."""

    def decompose(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Parse gold SQL and extract function + arguments.

        Args:
            sample: Raw BIRD-SQL sample with at least ``gold_sql`` and
                ``question`` keys.

        Returns:
            Dict with keys:
                function:       The NL question text (from the sample).
                arguments: List of dicts, each with ``sql_condition``
                            (SQLCondition) and ``nl_condition`` (str).
                parsed_sql: The full ParsedSQL object.

        Raises:
            ValueError: If the SQL cannot be parsed.
        """
        gold_sql = sample["gold_sql"]
        parsed = parse_sql(gold_sql)

        if not parsed.parseable:
            raise ValueError(
                f"Failed to parse SQL for sample "
                f"{sample.get('index', '?')}: {parsed.parse_error}"
            )

        arguments: list[dict[str, Any]] = []
        for cond in parsed.conditions:
            arguments.append({
                "sql_condition": cond,
                "nl_condition": self._condition_to_nl(cond),
            })

        return {
            "function": sample["question"],
            "arguments": arguments,
            "parsed_sql": parsed,
        }

    def to_conversational(
        self,
        sample: dict[str, Any],
        decomposed: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert decomposed output to the conversational format
        expected by ``BaseExtractor._build_extracted``."""
        return {
            "initial_query": decomposed["function"],
            "hints": [
                {"hint": c["nl_condition"]}
                for c in decomposed["arguments"]
            ],
        }

    def verify_coverage(
        self,
        sample: dict[str, Any],
        extracted: dict[str, Any],
    ) -> bool:
        """Always True — SQL extraction is deterministic from parsing."""
        return True

    def verify_solvability(
        self,
        sample: dict[str, Any],
        extracted: dict[str, Any],
    ) -> bool:
        """Verify that the gold SQL produces a non-empty result."""
        db_path = self._resolve_db_path(sample)
        if not Path(db_path).exists():
            print(f"Warning: database not found at {db_path}")
            return False
        result = execute_sql(db_path, sample["gold_sql"])
        return result.success and result.row_count > 0

    def build_output(
        self,
        sample: dict[str, Any],
        extracted: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build the final pipeline-compatible output dict.

        The output includes all fields required by ``situated_simulation.user_simulation``
        plus SQL-specific metadata (``gold_sql``, ``db_id``, ``db_path``,
        ``schema``, ``evidence``).
        """
        db_path = self._resolve_db_path(sample)
        source = sample.get("source", "train")
        original_index = sample.get("index", sample.get("id", 0))

        if not Path(db_path).exists():
            raise BirdReproductionError(f"database not found: {db_path}")

        # Schema text for prompt inclusion
        schema = get_schema_text(db_path)
        if not schema.strip() or schema.startswith("-- Database not found"):
            raise BirdReproductionError(f"could not load schema: {db_path}")

        # Execute gold SQL to obtain the reference answer
        result = execute_sql(db_path, sample["gold_sql"])
        if not result.success or not result.rows:
            raise BirdReproductionError(
                f"gold SQL failed or returned no rows for {task_id_for(source, original_index)}: "
                f"{result.error or 'empty result'}"
            )
        answer = self._format_answer(result)
        if not answer.strip():
            raise BirdReproductionError(
                f"gold SQL produced an incomplete answer for {task_id_for(source, original_index)}"
            )

        assert_required_model(self.model, context="BIRD extractor output model")

        return {
            "task_id": task_id_for(source, original_index),
            "original_id": original_index,
            "original_index": original_index,
            "task": "sql",
            "data_source": "bird_sql",
            "question": sample["question"],
            "function": extracted["function"],
            "answer": answer,
            "gold_sql": sample["gold_sql"],
            "db_id": sample["db_id"],
            "db_path": self._portable_db_path(sample),
            "schema": schema,
            "evidence": sample.get("evidence", ""),
            "fully_specified_question": sample["question"],
            "arguments": extracted["arguments"],
            "num_arguments": len(extracted["arguments"]),
            "model_name": self.model,
        }

    # ── Override extract() for deterministic flow ──────────────

    def _strip_arguments_from_function(
        self,
        question: str,
        arguments: list[dict[str, Any]],
        max_retries: int = 3,
    ) -> str:
        """Use LLM to rewrite the question with argument values removed.

        The original BIRD question embeds WHERE-clause values (e.g.,
        "complaints from **female** clients born in **2000**"). This
        creates overlap with the separately extracted arguments. We
        rewrite the question to only describe *what* to compute, letting
        arguments be the single source of truth for filter values.

        Two programmatic checks run after each LLM attempt:

        1. **Value leak check**: verifies no argument ``sql_value`` remains
           in the function (using normalized matching to catch spacing/casing
           differences like ``"200 MG"`` vs ``"200mg"``).
        2. **Meaning change check**: verifies the function doesn't introduce
           new non-trivial words that weren't in the original question
           (catches LLM replacements like ``"failed"`` → ``"passed"``).

        If either check fails, the LLM is retried with feedback.
        """
        cond_lines = "\n".join(
            f"- {c['argument']} (field: {c.get('sql_column', '')}, value: {c.get('sql_value', '')})"
            for c in arguments
        )
        prompt_path = self.get_prompts_dir() / "strip_arguments.txt"
        template = prompt_path.read_text()
        prompt = template.replace("[[QUESTION]]", question).replace(
            "[[CONDITIONS]]", cond_lines
        )

        from intent_construction.intent_extraction.core.llm_utils import generate_text

        # Collect argument values to verify against (skip very short ones
        # that are common words like "F", "M", "1")
        check_values = []
        for c in arguments:
            val = str(c.get("sql_value", ""))
            if len(val) > 2:
                check_values.append(val)

        for attempt in range(max_retries):
            result = generate_text(
                messages=[{"role": "user", "content": prompt}],
                model=self._strip_model,
                temperature=None,
            )
            assert_required_model(
                self._strip_model, context="BIRD extractor completed strip call"
            )
            if not isinstance(result, str) or not result.strip():
                raise BirdReproductionError(
                    "BIRD extractor received an empty strip response"
                )
            stripped = result.strip().strip('"').strip("'")
            # Clean up punctuation
            while stripped.endswith(".?") or stripped.endswith("??"):
                stripped = stripped[:-1]
            if stripped.endswith("."):
                stripped = stripped[:-1] + "?"
            elif stripped and not stripped.endswith("?"):
                stripped += "?"

            # Check 1: no argument values leaked into the function
            leaked = _check_value_leak(stripped, check_values)

            # Check 2: no meaning-changing words introduced
            new_words = _check_meaning_change(question, stripped)

            if not leaked and not new_words:
                return stripped

            # Build feedback for retry
            feedback_parts = []
            if leaked:
                feedback_parts.append(
                    f"It still contains these argument values that must be removed: {leaked}"
                )
            if new_words:
                feedback_parts.append(
                    f"It introduces new words not in the original question: {new_words}. "
                    f"Only REMOVE text, do not replace with different words."
                )

            prompt = (
                f"{prompt}\n\n"
                f"Your previous answer was: {stripped}\n"
                f"{' '.join(feedback_parts)}\n"
                f"Please rewrite again:"
            )

        raise BirdReproductionError(
            "BIRD extractor exhausted strip retries without a complete valid result"
        )

    def extract(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """
        Extract function + arguments from a single BIRD-SQL sample.

        Overrides ``BaseExtractor.extract()`` because SQL extraction is
        deterministic — the retry loop and LLM fallback of the base class
        are unnecessary.  Each argument carries full SQL metadata
        (column, table, operator, value) alongside its NL text.

        The function text is cleaned via LLM to remove embedded argument
        values, ensuring no overlap between function and arguments.
        """
        sample_id = sample.get("index", sample.get("id", "unknown"))

        try:
            # 1. Parse SQL and decompose into function + arguments
            decomposed = self.decompose(sample)

            # 2. Build arguments with SQL metadata + NL text
            arguments: list[dict[str, Any]] = []
            for i, cond_info in enumerate(decomposed["arguments"]):
                sql_cond: SQLCondition = cond_info["sql_condition"]
                arguments.append({
                    "argument_id": i + 1,
                    "argument": cond_info["nl_condition"],
                    "sql_column": sql_cond.column,
                    "sql_table": sql_cond.table,
                    "sql_operator": sql_cond.operator,
                    "sql_value": (
                        sql_cond.value
                        if isinstance(sql_cond.value, str)
                        else str(sql_cond.value)
                    ),
                })

            # 3. Strip argument values from function text
            source_function = self._strip_arguments_from_function(
                decomposed["function"], arguments,
            )

            extracted: dict[str, Any] = {
                "function": source_function,
                "arguments": arguments,
            }

            # 4. Solvability is mandatory for the fixed published subset.
            if not self.verify_solvability(sample, extracted):
                raise BirdReproductionError(
                    f"gold SQL returned an empty or failed result for sample {sample_id}"
                )

            # 5. Build final output
            output = self.build_output(sample, extracted)

            return output

        except Exception as e:
            print(f"Error extracting sample {sample_id}: {e}")
            raise

    # ── NL conversion helpers ──────────────────────────────────

    def _condition_to_nl(self, cond: SQLCondition) -> str:
        """
        Convert a ``SQLCondition`` to a natural-language sentence.

        Template-based conversion with no LLM dependency::

            =        → "The {col} is {val}."
            !=       → "The {col} is not {val}."
            >        → "The {col} is greater than {val}."
            >=       → "The {col} is at least {val}."
            <        → "The {col} is less than {val}."
            <=       → "The {col} is at most {val}."
            LIKE     → "The {col} contains/starts with/ends with {val}."
            IN       → "The {col} is one of {v1}, {v2}, …"
            BETWEEN  → "The {col} is between {low} and {high}."
            IS       → "The {col} is {val}."
            OR       → "Either {cleaned raw SQL}."
            NOT …   → negated form of the inner operator
            UNKNOWN  → "Argument: {raw SQL}."
        """
        col = self._format_column_name(cond.column) if cond.column else "value"
        op = cond.operator
        val = cond.value

        # Simple binary operators with direct templates
        if op in self._OP_TEMPLATES:
            return self._OP_TEMPLATES[op].format(col=col, val=val)

        if op == "LIKE":
            return self._like_to_nl(col, str(val))

        if op == "IN":
            if isinstance(val, tuple):
                items = ", ".join(str(v) for v in val)
                return f"The {col} is one of {items}."
            return f"The {col} is one of {val}."

        if op == "BETWEEN":
            if isinstance(val, tuple) and len(val) == 2:
                return f"The {col} is between {val[0]} and {val[1]}."
            return f"The {col} is between {val}."

        if op == "OR":
            cleaned = self._clean_raw_sql(str(val))
            return f"Either {cleaned}."

        if op.startswith("NOT "):
            return self._not_to_nl(col, op[4:], val)

        # UNKNOWN / fallback
        return f"Argument: {self._clean_raw_sql(cond.raw_expression)}."

    @staticmethod
    def _like_to_nl(col: str, val: str) -> str:
        """Convert a LIKE pattern to NL."""
        clean = val.strip("'\"")
        if clean.startswith("%") and clean.endswith("%"):
            return f"The {col} contains {clean.strip('%')}."
        if clean.startswith("%"):
            return f"The {col} ends with {clean.lstrip('%')}."
        if clean.endswith("%"):
            return f"The {col} starts with {clean.rstrip('%')}."
        return f"The {col} matches the pattern {clean}."

    def _not_to_nl(self, col: str, inner_op: str, val: str | tuple) -> str:
        """Convert a NOT-prefixed operator to NL."""
        if inner_op == "=":
            return f"The {col} is not {val}."
        if inner_op == "LIKE":
            clean = str(val).strip("'\"").strip("%")
            return f"The {col} does not contain {clean}."
        if inner_op == "IN":
            if isinstance(val, tuple):
                items = ", ".join(str(v) for v in val)
                return f"The {col} is not one of {items}."
            return f"The {col} is not in {val}."
        if inner_op == "BETWEEN":
            if isinstance(val, tuple) and len(val) == 2:
                return f"The {col} is not between {val[0]} and {val[1]}."
            return f"The {col} is not between {val}."
        return f"The {col} is not {inner_op} {val}."

    # ── Utility helpers ────────────────────────────────────────

    @staticmethod
    def _format_column_name(name: str) -> str:
        """Make a SQL column name slightly more readable by
        replacing underscores with spaces."""
        # First, try to extract a natural description from complex SQL expressions
        cleaned = BirdSqlExtractor._simplify_sql_column(name)
        return cleaned.replace("_", " ")

    @staticmethod
    def _simplify_sql_column(expr: str) -> str:
        """Convert complex SQL column expressions to natural language.

        Handles TIME_TO_STR/STRFTIME date extraction, SUBSTRING for
        date parts, and CAST/NULLIF arithmetic expressions.
        """
        import re

        # Pattern: date difference (month diff) — must be checked before single TIME_TO_STR
        # TIME_TO_STR(CAST(T?.col1 ...), '%m') - TIME_TO_STR(CAST(T?.col2 ...), '%m')
        m = re.match(
            r"TIME_TO_STR\(CAST\((?:\w+\.)?\"?([^\"]+?)\"?\s+AS\s+TIMESTAMP\),\s*'%m'\)"
            r"\s*-\s*"
            r"TIME_TO_STR\(CAST\((?:\w+\.)?\"?([^\"]+?)\"?\s+AS\s+TIMESTAMP\),\s*'%m'\)",
            expr, re.IGNORECASE,
        )
        if m:
            col1 = m.group(1).replace("_", " ")
            col2 = m.group(2).replace("_", " ")
            return f"month difference between {col1} and {col2}"

        # Pattern: TIME_TO_STR(CAST(T?.col AS TIMESTAMP), '%Y') → "year of col"
        m = re.match(
            r"TIME_TO_STR\(CAST\((?:\w+\.)?\"?([^\"]+?)\"?\s+AS\s+TIMESTAMP\),\s*'([^']+)'\)",
            expr, re.IGNORECASE,
        )
        if m:
            col, fmt = m.group(1), m.group(2)
            col = col.replace("_", " ")
            if fmt == "%Y":
                return f"year of {col}"
            if fmt == "%m":
                return f"month of {col}"
            if fmt == "%Y-%m":
                return f"year-month of {col}"
            return f"date of {col}"

        # Pattern: STRFTIME('%Y', T?.col) → "year of col"
        m = re.match(
            r"STRFTIME\('([^']+)',\s*(?:\w+\.)?\"?([^\"]+?)\"?\)",
            expr, re.IGNORECASE,
        )
        if m:
            fmt, col = m.group(1), m.group(2)
            col = col.replace("_", " ")
            if fmt == "%Y":
                return f"year of {col}"
            if fmt == "%m":
                return f"month of {col}"
            if fmt == "%Y-%m":
                return f"year-month of {col}"
            return f"date of {col}"

        # Pattern: SUBSTRING(T?.col, 1, 4) → "year of col" (extracting first 4 chars = year)
        #          SUBSTRING(T?.col, 1, 7) → "year-month of col"
        #          SUBSTRING(T?.col, 1, 10) → "date of col"
        m = re.match(
            r"SUBSTRING\((?:\w+\.)?\"?([^\"]+?)\"?,\s*1,\s*(\d+)\)",
            expr, re.IGNORECASE,
        )
        if m:
            col, length = m.group(1), int(m.group(2))
            col = col.replace("_", " ")
            if length == 4:
                return f"year of {col}"
            if length == 7:
                return f"year-month of {col}"
            if length == 10:
                return f"date of {col}"
            return col

        # Pattern: CAST(T?.col1 AS FLOAT) * N / NULLIF(T?.col2, 0) → "ratio of col1 to col2"
        m = re.match(
            r"CAST\((?:\w+\.)?\"?([^\"]+?)\"?\s+AS\s+\w+\)\s*\*\s*[\d.]+\s*/\s*NULLIF\((?:\w+\.)?\"?([^\"]+?)\"?,\s*0\)",
            expr, re.IGNORECASE,
        )
        if m:
            col1 = m.group(1).replace("_", " ")
            col2 = m.group(2).replace("_", " ")
            return f"percentage of {col1} over {col2}"

        # No match — return as-is (will go through _format_column_name's underscore replacement)
        return expr

    @staticmethod
    def _clean_raw_sql(raw: str) -> str:
        """Remove table aliases (T1., T2., …) and tidy whitespace."""
        cleaned = re.sub(r"\b[A-Z]\d+\.", "", raw)
        return " ".join(cleaned.split())

    def _resolve_db_path(self, sample: dict[str, Any]) -> str:
        """Resolve the SQLite database file path for a sample (absolute)."""
        db_id = sample["db_id"]
        source = sample.get("source", "train")
        template = (
            self._DEV_DB_TEMPLATE if source == "dev" else self._TRAIN_DB_TEMPLATE
        )
        # Return absolute path so downstream scripts work from any directory
        return str((self._data_root / template.format(db_id=db_id)).resolve())

    def _portable_db_path(self, sample: dict[str, Any]) -> str:
        """Return a repository-relative DB path for relocatable artifacts."""
        absolute = Path(self._resolve_db_path(sample))
        try:
            return absolute.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            # An explicit external data_root is supported, but the default run
            # cache stays repository-relative and therefore relocatable.
            return str(absolute)

    @staticmethod
    def _format_answer(result: SQLResult) -> str:
        """
        Format a ``SQLResult`` into a human-readable answer string.

        - Single scalar  → ``"42"``
        - Single row     → ``"val1, val2"``
        - Multiple rows  → one row per line
        """
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
