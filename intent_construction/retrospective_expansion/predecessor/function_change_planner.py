"""
Function-change planner (Stage 4a of the BIRD-SQL function predecessor redesign).

Deterministically enumerates ``(change_set, preserve_set)`` plans over the
function clauses ``{SELECT, GROUP_BY, ORDER_BY, JOIN}`` for a given parsed SQL
query. Condition clauses ``{WHERE, HAVING, LIMIT}`` and ``FROM`` are always
preserved.

Each plan tells Stage 4b (the LLM executor) exactly which clauses to rewrite
and which to keep byte-identical to the gold SQL.

This module is pure (no LLM, no DB) — it operates on a ``ParsedSQL`` object.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

# Function clauses (eligible for change). Order is canonical for serialization.
GOAL_CLAUSES: tuple[str, ...] = ("SELECT", "GROUP_BY", "ORDER_BY", "JOIN")

# Condition clauses (always preserved). Plus FROM, which is also always preserved.
COND_CLAUSES: tuple[str, ...] = ("WHERE", "HAVING", "LIMIT")
ALWAYS_PRESERVED: tuple[str, ...] = COND_CLAUSES + ("FROM",)


@dataclass(frozen=True)
class FunctionChangePlan:
    """A single (change, preserve) plan for one sample.

    Attributes
    ----------
    change_set:
        Function clauses the LLM MUST rewrite in a meaningful way.
    preserve_set:
        Function clauses the LLM MUST keep byte-identical (sub-tree equality)
        to the gold. Always disjoint from ``change_set``. Note: condition
        clauses ``WHERE``/``HAVING``/``LIMIT`` and ``FROM`` are also
        preserved but are tracked separately in ``always_preserved``.
    always_preserved:
        Clauses that are always preserved regardless of plan
        (``WHERE``/``HAVING``/``LIMIT``/``FROM``).
    score:
        Heuristic score for ranking plan candidates.
    risk:
        Optional risk tag (e.g. ``"trivial"``).
    """

    change_set: tuple[str, ...]
    preserve_set: tuple[str, ...]
    always_preserved: tuple[str, ...]
    score: int
    risk: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_set": list(self.change_set),
            "preserve_set": list(self.preserve_set),
            "always_preserved": list(self.always_preserved),
            "score": self.score,
            "risk": self.risk,
        }


# -----------------------------------------------------------------------------
# Feasibility & scoring
# -----------------------------------------------------------------------------

def _present_clauses(parsed) -> set[str]:
    """Return the set of function clauses present in the gold SQL."""
    present: set[str] = {"SELECT"}  # SELECT is always present
    if getattr(parsed, "has_group_by", False):
        present.add("GROUP_BY")
    if getattr(parsed, "has_order_by", False):
        present.add("ORDER_BY")
    if getattr(parsed, "joins", ()):  # tuple of SQLJoin
        present.add("JOIN")
    return present


def _is_feasible(change_set: frozenset[str], present: set[str], parsed) -> bool:
    """Reject change-sets that cannot yield a meaningful, valid rewrite.

    v1 conservative rules:
      - ``change_set`` must be non-empty.
      - Every clause in ``change_set`` must be present in the gold (i.e. we
        only modify what's there; we don't ask the LLM to invent a new GROUP
        BY out of thin air, which is hard to validate).
      - If HAVING is present (and thus preserved), forbid removing GROUP BY
        entirely — but since we don't allow GROUP_BY removal here, the
        LLM is free to *modify* GROUP BY as long as the HAVING preds still
        bind. The prompt makes this explicit; the validator double-checks.
    """
    if not change_set:
        return False
    if not change_set.issubset(present):
        return False
    return True


def _score(
    change_set: frozenset[str],
    *,
    has_limit: bool = False,
) -> tuple[int, str | None]:
    """Heuristic meaningfulness score; higher = more interesting rewrite."""
    score = 0
    score += 3 if "SELECT" in change_set else 0
    score += 3 if "JOIN" in change_set else 0
    score += 2 if "GROUP_BY" in change_set else 0
    score += 1 if "ORDER_BY" in change_set else 0
    if len(change_set) >= 3:
        score += 2
    risk: str | None = None
    if change_set == frozenset({"ORDER_BY"}):
        if has_limit:
            # With LIMIT, ORDER BY selects which rows survive and is therefore
            # a substantive function change rather than a presentation-only one.
            score += 2
        else:
            score -= 3
            risk = "trivial"
    return score, risk


def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union else 0.0)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def enumerate_plans(parsed) -> list[FunctionChangePlan]:
    """Enumerate every feasible plan, ordered by ``(score desc, |cs| desc)``."""
    present = _present_clauses(parsed)
    candidates: list[FunctionChangePlan] = []
    for r in range(1, len(GOAL_CLAUSES) + 1):
        for combo in combinations(GOAL_CLAUSES, r):
            cs = frozenset(combo)
            if not _is_feasible(cs, present, parsed):
                continue
            score, risk = _score(
                cs,
                has_limit=getattr(parsed, "limit", None) is not None,
            )
            preserve = tuple(sorted(present - cs, key=GOAL_CLAUSES.index))
            change = tuple(sorted(cs, key=GOAL_CLAUSES.index))
            candidates.append(FunctionChangePlan(
                change_set=change,
                preserve_set=preserve,
                always_preserved=ALWAYS_PRESERVED,
                score=score,
                risk=risk,
            ))
    candidates.sort(
        key=lambda p: (
            -p.score,
            -len(p.change_set),
            tuple(GOAL_CLAUSES.index(clause) for clause in p.change_set),
        )
    )
    return candidates


def select_diverse_plans(
    parsed,
    n_plans: int = 3,
    require_multi_clause: bool = True,
) -> list[FunctionChangePlan]:
    """Pick up to ``n_plans`` feasible plans maximizing diversity.

    Greedy strategy: start with the highest-scoring plan, then iteratively
    add the plan with the largest minimum Jaccard distance to the current
    selection (ties broken by score).

    If ``require_multi_clause`` is True and the candidate pool contains any
    plan with ``|change_set| >= 3``, force at least one such plan into the
    selection.
    """
    pool = enumerate_plans(parsed)
    if not pool:
        return []
    selected: list[FunctionChangePlan] = [pool[0]]

    remaining = pool[1:]
    while remaining and len(selected) < n_plans:
        sel_sets = [frozenset(p.change_set) for p in selected]

        def diversity_key(p: FunctionChangePlan) -> tuple[float, int]:
            cs = frozenset(p.change_set)
            min_dist = min(_jaccard_distance(cs, s) for s in sel_sets)
            return (min_dist, p.score)

        remaining.sort(key=diversity_key, reverse=True)
        next_plan = remaining.pop(0)
        selected.append(next_plan)

    if require_multi_clause:
        has_multi = any(len(p.change_set) >= 3 for p in selected)
        if not has_multi:
            multi_pool = [p for p in pool if len(p.change_set) >= 3]
            if multi_pool:
                # Replace the lowest-scoring single-clause plan with the best multi
                selected.sort(key=lambda p: (len(p.change_set), p.score))
                selected[0] = multi_pool[0]

    # Restore canonical order: score desc, |cs| desc
    selected.sort(
        key=lambda p: (
            -p.score,
            -len(p.change_set),
            tuple(GOAL_CLAUSES.index(clause) for clause in p.change_set),
        )
    )
    return selected[:n_plans]


__all__ = [
    "GOAL_CLAUSES",
    "COND_CLAUSES",
    "ALWAYS_PRESERVED",
    "FunctionChangePlan",
    "enumerate_plans",
    "select_diverse_plans",
]
