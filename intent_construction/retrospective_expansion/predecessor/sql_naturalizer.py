"""
Episodic SQL → text naturalizer (Step 5 of BIRD-SQL redesign).

Given a predecessor (NEW_SQL) and the original SQL/question, asks an LLM to
produce a short follow-up question/instruction that DOES NOT re-state the
preserved filter values (WHERE/HAVING/LIMIT) — those are implicit from
the prior conversational turn.

Quality gates run after every LLM response:
  1. Length ≤ 25 words
  2. Exactly one sentence (basic punctuation count)
  3. No leak of preserved literal values (numbers, strings) using the
     ``_check_value_leak`` style fuzzy match
  4. No banned-phrase prefix (Among those, From the same group, …) so the
     surrounding simulator can prepend its own connector
  5. Mentions at least one new-function token (heuristic: at least one alpha
     token from the changed clauses' SQL appears)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import sqlglot
from sqlglot import expressions as exp

_THIS = Path(__file__).resolve().parent

from intent_construction.intent_extraction.core.llm_utils import generate_json, load_prompt, populate_prompt  # noqa: E402

PROMPT_PATH = _THIS / "prompts" / "naturalize_sql_followup.txt"

# Common English words exempted from value-leak detection (mirror of
# extractor.py's _COMMON_WORD_VALUES, kept local for module independence).
_COMMON_WORD_VALUES = frozenset({
    'pass', 'fail', 'male', 'female', 'bottle', 'stroke',
    'restaurant', 'alcohol', 'phone', 'none', 'legal', 'rare',
    'english', 'action', 'comedy', 'drama', 'active', 'open',
    'closed', 'pending', 'complete', 'true', 'false', 'yes',
})

_BANNED_PREFIXES = (
    "among those", "from the same group", "from that same group",
    "within those", "within those same", "sticking with the same",
    "keeping the same", "with the same", "out of those", "from the same set",
)


_PROMPT_TEMPLATE: str | None = None


def _load_prompt() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = load_prompt(str(PROMPT_PATH))
    return _PROMPT_TEMPLATE


def _normalize(s: str) -> str:
    return re.sub(r'[\s\-_.,;:\'\"()/%]', '', (s or "").lower())


def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or ""))


def _sentence_count(s: str) -> int:
    return len(re.findall(r"[.!?]+(?:\s|$)", (s or "").strip())) or 1


def _extract_literal_values(sql: str) -> list[str]:
    """Pull literal numbers and string values out of a SQL string."""
    try:
        node = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return []
    vals: list[str] = []
    for lit in node.find_all(exp.Literal):
        v = lit.this
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in _COMMON_WORD_VALUES:
            vals.append(s)
    return vals


def _check_value_leak(text: str, values: Iterable[str]) -> list[str]:
    text_norm = _normalize(text)
    leaked: list[str] = []
    for v in values:
        v_norm = _normalize(v)
        if not v_norm:
            continue
        if v.lower().strip() in _COMMON_WORD_VALUES:
            continue
        # Numbers: require word-boundary-ish containment to avoid '1' matching '12'
        if v.lstrip('-').replace('.', '', 1).isdigit():
            if re.search(rf"(?<!\d){re.escape(v)}(?!\d)", text):
                leaked.append(v)
        else:
            if v_norm in text_norm:
                leaked.append(v)
    return leaked


def _starts_with_banned_prefix(text: str) -> bool:
    low = text.strip().lower()
    return any(low.startswith(p) for p in _BANNED_PREFIXES)


def _changed_clauses_sql(new_sql: str, gold_sql: str, changed: list[str]) -> str:
    """Concatenate the SQL fragments of the changed clauses for function-mention check."""
    try:
        node = sqlglot.parse_one(new_sql, dialect="sqlite")
    except Exception:
        return new_sql
    if not isinstance(node, exp.Select):
        node = node.find(exp.Select) or node
    parts: list[str] = []
    arg_map = {
        "SELECT": ("expressions", lambda v: ", ".join(e.sql() for e in (v or []))),
        "GROUP_BY": ("group", lambda v: v.sql() if v else ""),
        "ORDER_BY": ("order", lambda v: v.sql() if v else ""),
        "JOIN":     ("joins", lambda v: " ".join(j.sql() for j in (v or []))),
    }
    for c in changed:
        arg, fn = arg_map.get(c, (None, None))
        if arg is None:
            continue
        try:
            parts.append(fn(node.args.get(arg)))
        except Exception:
            pass
    return " ".join(parts)


def _has_function_mention(text: str, changed_sql_text: str) -> bool:
    """Check whether at least one substantive token from changed clauses
    appears in the naturalized text. Excludes SQL keywords and trivial words.
    """
    sql_keywords = {
        "select", "from", "where", "group", "by", "having", "order", "limit",
        "join", "on", "as", "and", "or", "not", "in", "like", "between",
        "asc", "desc", "count", "sum", "avg", "max", "min", "distinct", "case",
        "when", "then", "else", "end", "is", "null",
    }
    tokens = {t.lower() for t in re.findall(r"[a-zA-Z_]{3,}", changed_sql_text)}
    candidates = {t for t in tokens if t not in sql_keywords}
    if not candidates:
        return True  # nothing meaningful to check; don't reject
    text_norm = _normalize(text)
    for c in candidates:
        # Strip table-prefix stems like "t1_firstname" → "firstname"
        c_clean = c.split("_")[-1] if "_" in c else c
        for token in (c, c_clean):
            if len(token) >= 3 and token in text_norm:
                return True
    return False


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def naturalize_followup(
    *,
    new_sql: str,
    gold_sql: str,
    original_question: str,
    changed_clauses: list[str],
    preserved_clauses: list[str],
    model: str = "gpt-5.1",
    max_attempts: int = 3,
    temperature: float = 0.7,
    reasoning_effort: str | None = None,
    step: str = "naturalize_followup",
) -> tuple[str | None, str]:
    """Produce an episodic follow-up text for ``new_sql``.

    Returns ``(text, status)``:
      - ``text`` is the validated naturalized text or ``None`` on failure.
      - ``status`` is ``"ok"`` or a failure reason such as ``"too_long"``,
        ``"too_many_sentences"``, ``"value_leak:<values>"``,
        ``"banned_prefix"``, ``"no_function_mention"``, ``"llm_fail"``,
        ``"format_fail"``.
    """
    leak_values = _extract_literal_values(gold_sql)
    changed_sql_text = _changed_clauses_sql(new_sql, gold_sql, changed_clauses)
    template = _load_prompt()
    body = populate_prompt(template, {
        "ORIGINAL_QUESTION": original_question or "",
        "ORIGINAL_SQL": gold_sql,
        "NEW_SQL": new_sql,
        "CHANGED_CLAUSES": ", ".join(changed_clauses) or "(none)",
        "PRESERVED_CLAUSES": ", ".join(preserved_clauses) or "(none)",
    })
    messages = [{"role": "user", "content": body}]

    last_status = "llm_fail"
    for attempt in range(max_attempts):
        try:
            resp = generate_json(
                messages=messages,
                model=model,
                step=f"{step}/a{attempt}",
                max_retries=2,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            print(f"  ⚠ naturalizer LLM call failed: {e}")
            last_status = "llm_fail"
            continue
        if not resp or "text" not in resp:
            last_status = "format_fail"
            continue
        text = (resp.get("text") or "").strip()
        if not text:
            last_status = "format_fail"
            continue
        if _word_count(text) > 25:
            last_status = "too_long"
            continue
        if _sentence_count(text) > 1:
            last_status = "too_many_sentences"
            continue
        if _starts_with_banned_prefix(text):
            last_status = "banned_prefix"
            continue
        leaks = _check_value_leak(text, leak_values)
        if leaks:
            last_status = f"value_leak:{','.join(leaks[:3])}"
            continue
        if not _has_function_mention(text, changed_sql_text):
            last_status = "no_function_mention"
            continue
        return text, "ok"

    return None, last_status


__all__ = [
    "naturalize_followup",
    "_check_value_leak",
    "_extract_literal_values",
    "_starts_with_banned_prefix",
    "_word_count",
    "_sentence_count",
    "_has_function_mention",
]
