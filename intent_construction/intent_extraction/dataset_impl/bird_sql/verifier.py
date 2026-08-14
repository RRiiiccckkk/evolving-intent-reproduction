"""
BIRD-SQL Verifier implementation.
Verifies extracted function + arguments against the original SQL sample.

Zero LLM dependency — verification is done via SQL execution and
normalized string comparison.
"""

from pathlib import Path
from typing import Any

from intent_construction.intent_extraction.core.base_verifier import BaseVerifier
from .db_utils import execute_sql, compare_results, SQLResult


class BirdSqlVerifier(BaseVerifier):
    """
    Verifier for BIRD-SQL samples.

    Uses SQL execution against the SQLite database to verify correctness:

    - **Coverage**: always True (extraction is deterministic from AST parsing).
    - **Solvability**: gold SQL executes successfully and its result matches
      the ground-truth string.
    - **Answer evaluation**: normalized string comparison with numeric
      equivalence (``"5"`` == ``"5.0"``) and set-based multi-row comparison.
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        num_runs: int = 1,
    ):
        super().__init__(model=model, num_runs=num_runs)

    # ── BaseVerifier abstract method implementations ───────────

    def get_dataset_name(self) -> str:
        return "bird_sql"

    def verify_coverage(
        self,
        original: dict[str, Any],
        extracted: dict[str, Any],
    ) -> bool:
        """Always True — SQL extraction is deterministic from parsing."""
        return True

    def verify_solvability(
        self,
        original: dict[str, Any],
        extracted: dict[str, Any],
        ground_truth: str,
    ) -> bool:
        """
        Verify by executing gold SQL and comparing with ground truth.

        The database path is resolved from ``extracted["db_path"]`` (preferred)
        or ``original["db_path"]``.

        Args:
            original:     Raw BIRD-SQL sample (must include ``gold_sql``).
            extracted:    Extracted dict (should include ``db_path``).
            ground_truth: Expected answer string.

        Returns:
            True if gold SQL executes successfully and its formatted result
            matches *ground_truth* after normalization.
        """
        db_path = extracted.get("db_path") or original.get("db_path", "")
        gold_sql = original.get("gold_sql", "")

        if not db_path or not Path(db_path).exists():
            print(f"Verifier: database not found at {db_path}")
            return False

        if not gold_sql:
            print("Verifier: no gold_sql provided")
            return False

        result = execute_sql(db_path, gold_sql)
        if not result.success:
            print(f"Verifier: SQL execution failed — {result.error}")
            return False

        if result.row_count == 0:
            gt = ground_truth.strip().lower()
            return gt == "" or gt == "0"

        actual_answer = self._format_result(result)
        return self.evaluate_answer(actual_answer, ground_truth)

    def extract_answer(self, response: str) -> str | None:
        """
        Extract the answer from a response string.

        For BIRD-SQL the "response" is already the stringified SQL result,
        so this is essentially an identity operation with whitespace
        normalization.
        """
        if response is None:
            return None
        stripped = response.strip()
        return stripped if stripped else None

    def evaluate_answer(
        self,
        predicted: str | None,
        ground_truth: str,
    ) -> bool:
        """
        Compare predicted and ground-truth answer strings.

        Applies normalization so that ``"5"``, ``"5.0"``, and ``" 5 "``
        are all treated as equal.  For multi-row results, set-based
        (order-insensitive) comparison is used.
        """
        if predicted is None:
            return False

        pred = self._normalize(predicted)
        gold = self._normalize(ground_truth)

        # Exact match after normalization
        if pred == gold:
            return True

        # Numeric equivalence: "5" == "5.0"
        try:
            if float(pred) == float(gold):
                return True
        except (ValueError, TypeError):
            pass

        # Set-based multi-row comparison (order-insensitive)
        pred_lines = sorted(pred.splitlines())
        gold_lines = sorted(gold.splitlines())
        if pred_lines and gold_lines and pred_lines == gold_lines:
            return True

        return False

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize a value string for comparison."""
        s = value.strip()
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return str(f)
        except (ValueError, TypeError):
            return s

    @staticmethod
    def _format_result(result: SQLResult) -> str:
        """
        Format a ``SQLResult`` to a comparable answer string.

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
