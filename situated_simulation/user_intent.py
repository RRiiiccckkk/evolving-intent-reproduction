"""
User Intent data structures for EvolvingIntent.

Maps the LaTeX formalization to code:
  - Paper: I_t = (f_t, C_t, C_rev_t, y_t)  →  UserIntent
  - Paper: c_i ∈ C_t                          →  Argument
  - Paper: ΔI_t                               →  IntentTransition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass(frozen=True)
class Argument:
    """
    A single argument c_i with its counterfactual history.

    Paper: c_i ∈ C_t (an argument in the complete argument set).
    Code: corresponds to a "argument" dict in the raw JSON data.
    """
    argument_id: int                    # argument_id in raw data
    source_text: str                     # c_i — the original/correct value
    counterfactual_variants: tuple[str, ...]  # c̃_i^(j) — counterfactual values (immutable)
    is_shared: bool = False             # True if shared across function changes


@dataclass(frozen=True)
class UserIntent:
    """
    User intent state at a single turn: I_t = (f_t, C_t, C_rev_t, y_t).

    Paper Section 2, "User intent state":
      - f_t: the function (function/task) the user wants to accomplish
      - C_t: the complete set of arguments the user has in mind
      - C_rev_t ⊆ C_t: the revealed arguments (communicated to the AI)
      - y_t: the target answer, fully determined by the intent

    C_t contains the *current* values the user has in mind at this turn.
    During argument change, some values in C_t may be counterfactual (not yet
    corrected to source). The `active_values` field tracks the actual text
    value of each argument at this turn.

    Each turn of a conversation has its own UserIntent snapshot.
    """
    function: str                       # f_t — function text
    arguments: tuple[Argument, ...]     # Argument templates (with source + counterfactual info)
    revealed_ids: frozenset[int]        # C_rev_t — argument_ids revealed so far
    target_answer: str                  # y_t — ground truth for this state ("" if unknown)

    # C_t — maps argument_id → currently active text value at this turn.
    # During argument change, some values may be counterfactual (not yet corrected).
    # If empty, defaults to source_text for all arguments.
    active_values: tuple[tuple[int, str], ...] = ()

    def get_active_value(self, argument_id: int) -> str:
        """Get the currently active text for an argument (counterfactual or source)."""
        for aid, text in self.active_values:
            if aid == argument_id:
                return text
        # Fallback to source text
        for arg in self.arguments:
            if arg.argument_id == argument_id:
                return arg.source_text
        return ""

    @property
    def is_fully_specified(self) -> bool:
        """C_rev_t ⊇ C_t — all current arguments are revealed."""
        return frozenset(a.argument_id for a in self.arguments) <= self.revealed_ids

    @property
    def is_under_specified(self) -> bool:
        """C_rev_t ⊊ C_t — some arguments unrevealed."""
        return not self.is_fully_specified

    @property
    def unrevealed_arguments(self) -> tuple[Argument, ...]:
        """C_t \\ C_rev_t — arguments not yet communicated."""
        return tuple(a for a in self.arguments if a.argument_id not in self.revealed_ids)

    @property
    def revealed_arguments(self) -> tuple[Argument, ...]:
        """C_rev_t — arguments already communicated."""
        return tuple(a for a in self.arguments if a.argument_id in self.revealed_ids)

    @property
    def has_counterfactual_values(self) -> bool:
        """True if any active values differ from source text."""
        for aid, text in self.active_values:
            for arg in self.arguments:
                if arg.argument_id == aid and text != arg.source_text:
                    return True
        return False


@dataclass(frozen=True)
class IntentTransition:
    """
    A single state transition ΔI_t := I_t ⊖ I_{t-1}.

    Paper Section 2, three transition types:
      - argument_reveal: C_rev grows, y_t = y_{t-1}
      - argument_change: |C_rev| same, but value(s) change
      - function_change: f_t ≠ f_{t-1}, C_t ∩ C_{t-1} ≠ ∅
    """
    transition_type: Literal[
        "argument_reveal",
        "argument_change",
        "function_change",
    ]
    # For argument_reveal: which argument_ids are newly revealed
    revealed_ids: tuple[int, ...] | None = None
    # For argument_change: (arg_id, old_text, new_text) tuples
    changed_arguments: tuple[tuple[int, str, str], ...] | None = None
    # For function_change: old and new function text
    old_function: str | None = None
    new_function: str | None = None


# =============================================================================
# ChangePlan — output of the rule-based intent scheduler
# =============================================================================

@dataclass
class ChangePlan:
    """
    Output of the rule-based intent scheduler.

    Describes the full trajectory of user intent states across a conversation.
    This is what user_simulation._create_*_sample() methods currently compute implicitly.

    The ChangePlan separates WHAT changes happen from HOW the user
    phrases them (handled by UserSimulator).
    """
    task_id: str
    scenario: Literal[
        "fully-specified",
        "under-specified",
        "argument-revision",
        "function-switch",
        "combined",
    ]
    domain: str                             # "math", "search", "sql", "swe_bench_verified"

    # I_0, I_1, ..., I_T — one UserIntent per user turn
    intent_trajectory: list[UserIntent]

    # ΔI_1, ΔI_2, ... — transitions between consecutive intents
    # len(transitions) == len(intent_trajectory) - 1
    transitions: list[IntentTransition]

    # Final ground truth (always == intent_trajectory[-1].target_answer)
    final_label: str

    # Reference to raw sample for metadata
    raw_sample: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_turns(self) -> int:
        """Number of user turns in this plan."""
        return len(self.intent_trajectory)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage in IntentSample.metadata."""
        return {
            "task_id": self.task_id,
            "scenario": self.scenario,
            "domain": self.domain,
            "num_turns": self.num_turns,
            "final_label": self.final_label,
            "intent_trajectory": [
                {
                    "function": intent.function,
                    "num_arguments": len(intent.arguments),
                    "revealed_ids": sorted(intent.revealed_ids),
                    "is_fully_specified": intent.is_fully_specified,
                    "has_counterfactual_values": intent.has_counterfactual_values,
                    "target_answer": intent.target_answer,
                    "active_values": {
                        str(aid): text for aid, text in intent.active_values
                    } if intent.active_values else None,
                }
                for intent in self.intent_trajectory
            ],
            "transitions": [
                {
                    "type": t.transition_type,
                    "revealed_ids": list(t.revealed_ids) if t.revealed_ids else None,
                    "changed_arguments": [
                        {"arg_id": c[0], "old": c[1], "new": c[2]}
                        for c in t.changed_arguments
                    ] if t.changed_arguments else None,
                    "old_function": t.old_function,
                    "new_function": t.new_function,
                }
                for t in self.transitions
            ],
        }


# =============================================================================
# Builder helpers — convert raw JSON data to typed structures
# =============================================================================

def arguments_from_raw(
    arguments: list[dict[str, Any]],
) -> tuple[Argument, ...]:
    """
    Convert raw argument dicts from the pipeline JSON into Argument objects.

    Raw format:
      {"argument_id": 1, "argument": "...", "counterfactual_arguments": [...]}
    Where each counterfactual_arguments entry has "counterfactual_argument" text.
    """
    args = []
    for cond in arguments:
        counterfactual = tuple(
            p["counterfactual_argument"]
            for p in cond.get("counterfactual_arguments", [])
        )
        args.append(Argument(
            argument_id=cond["argument_id"],
            source_text=cond.get("argument", ""),
            counterfactual_variants=counterfactual,
            is_shared=cond.get("is_shared", False),
        ))
    return tuple(args)
