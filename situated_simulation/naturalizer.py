"""
Unified conversation naturalizer for all domains.

Rewrites rule-based turn texts into natural-sounding conversation
using an LLM. The `search` domain has a specialised LLM subclass; all
other domains fall back to the rule-based renderer. The base interface
is shared.

Input per turn:  (function, correction, reveal, prefix) + history
Output per turn: natural user message string
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Structured turn content (the naturalizer's input per turn)
# ---------------------------------------------------------------------------

@dataclass
class StructuredTurnContent:
    """Structured representation of a single turn for the naturalizer."""

    turn_idx: int
    function_text: str | None           # function (initial or changed); None if no function event
    correction_texts: list[str]     # correction texts this turn
    argument_texts: list[str]      # reveal/argument texts this turn
    prefix: str                     # rotated prefix (suggested opening)
    rule_based_text: str            # fallback from render_turns()
    turn_type: str                  # "initial" | "function_change" | "correction" | "reveal" | "mixed"
    taxonomy_type: str | None = None  # T1-T4 for IF domain function changes


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TurnNaturalizer(ABC):
    """Base class for all turn naturalizers."""

    def seed_state(self, content: "StructuredTurnContent") -> None:
        """Seed internal tracking state from Turn 0 content (no text generated).

        Called by IntentSample.reset() so that online naturalizers know the
        initial function/arguments before the first step() call.  Subclasses
        with mutable state (e.g. _prev_function) must override.
        """

    @abstractmethod
    def naturalize_turn(
        self,
        content: StructuredTurnContent,
        history: list[tuple[str, str]],
    ) -> str:
        """Convert structured turn content into a natural user message.

        Args:
            content: Structured data for this turn.
            history: Previous turns as (user_text, llm_response) pairs.
                     llm_response is "" when not available.

        Returns:
            A natural-language user message string.
        """
        ...

    def naturalize_all(
        self,
        contents: list[StructuredTurnContent],
        llm_responses: list[str] | None = None,
    ) -> list[str]:
        """Naturalize all turns sequentially, accumulating history.

        Args:
            contents: Structured content for each turn.
            llm_responses: LLM responses between turns. If None, all
                           responses are treated as empty (batch mode).
        """
        result: list[str] = []
        full_history: list[tuple[str, str]] = []
        for i, content in enumerate(contents):
            natural = self.naturalize_turn(content, history=full_history)
            result.append(natural)
            resp = ""
            if llm_responses and i < len(llm_responses):
                resp = llm_responses[i]
            full_history.append((natural, resp))
        return result


# ---------------------------------------------------------------------------
# Rule-based (pass-through) naturalizer
# ---------------------------------------------------------------------------

class RuleBasedNaturalizer(TurnNaturalizer):
    """Returns the rule-based concatenated text unchanged."""

    def naturalize_turn(self, content: StructuredTurnContent, history: list[tuple[str, str]]) -> str:
        return content.rule_based_text


# ---------------------------------------------------------------------------
# Shared LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(
    prompt: str, model: str, max_tokens: int | None, fallback: str,
    system: str | None = None,
) -> str:
    """Call the LLM and return cleaned text, falling back on error."""
    from intent_construction.intent_extraction.core.llm_utils import (
        LLMAccountingError,
        LLMIncompleteResponse,
        generate_text,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        result = generate_text(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        natural = result.strip().strip('"').strip("'")
        return natural if natural else fallback
    except (LLMAccountingError, LLMIncompleteResponse):
        raise
    except Exception:
        return fallback


def _format_history(
    history: list[tuple[str, str]],
    window: int = 4,
    truncate: int = 120,
) -> str:
    """Format the previous turn's assistant response for prompt context.

    Only includes the most recent assistant response to prevent
    argument/detail leakage from earlier user turns.

    Each entry is a (user_text, llm_response) pair.
    """
    if not history:
        return ""
    _user_msg, assistant_msg = history[-1]
    if assistant_msg:
        return f"Assistant (previous response): {assistant_msg[:truncate]}"
    return ""


def _has_assistant_response(history: list[tuple[str, str]]) -> bool:
    """Check if the most recent history entry has a non-empty assistant response."""
    return bool(history) and bool(history[-1][1])


def _get_response_aware_block(turn_idx: int, variants: list[str]) -> str:
    """Pick a response-aware prompt block, cycling through variants by turn."""
    if not variants:
        return ""
    return variants[turn_idx % len(variants)]


# ---------------------------------------------------------------------------
# Critical-value validation
# ---------------------------------------------------------------------------

def _extract_critical_values(text: str) -> list[str]:
    """Extract values that must be preserved verbatim in rephrased output.

    Captures:
    - Numbers and dollar amounts: $2, 16, 3.5
    - Quoted strings (2+ chars, single or double)
    - Capitalized multi-word entities: "Buenos Aires", "South Asia"
    """
    values: list[str] = []

    # Numbers and dollar amounts (strip trailing sentence-ending periods)
    for m in re.findall(r"\$?\d+\.?\d*", text):
        values.append(m.rstrip("."))

    # Quoted strings (single or double)
    values.extend(re.findall(r'"([^"]{2,})"', text))
    values.extend(re.findall(r"'([^']{2,})'", text))

    # Capitalized multi-word entities (e.g. "Buenos Aires", "South Asia")
    # Require at least 2 consecutive capitalized words not at sentence start
    for m in re.finditer(r"(?<=[.!?]\s|, )([A-Z][a-z]+(?: [A-Z][a-z]+)+)", text):
        values.append(m.group(1))

    return list(dict.fromkeys(values))  # deduplicate, preserve order


def _validate_response(response: str, reference_text: str) -> bool:
    """Verify that all critical values from reference_text appear in response."""
    if not response or not response.strip():
        return False
    critical = _extract_critical_values(reference_text)
    if not critical:
        return True  # nothing to validate
    return all(value in response for value in critical)


def _build_strict_prompt(
    original_prompt: str,
    reference_text: str,
    failed_response: str,
) -> str:
    """Build a stricter retry prompt when validation failed."""
    critical = _extract_critical_values(reference_text)
    missing = [v for v in critical if v not in failed_response]
    return (
        f"Your previous attempt was missing these critical values: {missing}\n\n"
        f"Rewrite the content below as a natural user message. "
        f"You MUST include ALL of these exact values: {critical}\n\n"
        f"Content to rephrase:\n{reference_text}\n\n"
        f"Write ONLY the user's message:"
    )


_MAX_VALIDATION_RETRIES: int = 2


def _call_llm_validated(
    prompt: str, model: str, max_tokens: int | None, fallback: str,
    system: str | None = None,
) -> str:
    """Call LLM with critical-value validation and retry.

    1. Call LLM with original prompt
    2. Validate critical values from fallback (rule_based_text) appear in response
    3. On failure: retry with strict prompt listing missing values
    4. On repeated failure: return fallback (rule_based_text)
    """
    for attempt in range(_MAX_VALIDATION_RETRIES):
        result = _call_llm(prompt, model, max_tokens, fallback=fallback, system=system)
        # If _call_llm already returned the fallback (LLM error), just use it
        if result == fallback:
            return fallback
        if _validate_response(result, fallback):
            return result
        # Build strict prompt for retry
        prompt = _build_strict_prompt(prompt, fallback, result)
    # All retries exhausted — return fallback
    return fallback


# ===========================================================================
# Search Naturalizer
# ===========================================================================

_SEARCH_PROMPT_FIRST = """\
You are simulating a user talking to a deep research agent. The user wants to \
find a specific entity (person, place, event, etc.) by providing clues.

Rewrite the following message to sound more natural and conversational. Keep \
ALL the information -- do not drop any clues or details. Just make it flow \
better as natural speech.

Original message:
{original}

Write the rewritten message (one short paragraph, no preamble):"""

_SEARCH_RESPONSE_AWARE_VARIANTS = [
    """
RESPONDING TO THE ASSISTANT:
The assistant just replied to your clues. Briefly acknowledge before giving more info.
- "Hmm, that's not it. Here's another clue — ..."
- "No, not that one. Also, ..."
Do NOT praise the assistant's guess. One neutral phrase, then your new clues.""",

    """
RESPONDING TO THE ASSISTANT:
The assistant guessed or responded. React neutrally, then provide more clues.
- "Not quite. Let me add ..."
- "Nope. Try considering that ..."
Do NOT say "good guess" or "close". Stay neutral. Brief reaction, then clues.""",

    """
RESPONDING TO THE ASSISTANT:
The assistant just replied. Acknowledge briefly, then continue with new information.
- "That's not what I mean. Also ..."
- "Hmm no. Here's something else — ..."
Do NOT evaluate the guess positively. One short phrase, then your clues.""",

    """
RESPONDING TO THE ASSISTANT:
React briefly to the assistant's response before adding new clues.
- "No, different one. Another detail — ..."
- "Not exactly. Also consider ..."
Avoid positive evaluation. Neutral reaction, then continue.""",
]

_SEARCH_PROMPT_FOLLOWUP = """\
You are simulating a user talking to a deep research agent. This is a \
follow-up message in an ongoing conversation.

Previous conversation:
{history}

Rewrite the following follow-up message to sound more natural and \
conversational. Keep ALL the information -- every clue, correction, and \
detail must be preserved. Just make it flow better as natural speech.
{response_aware_block}

Original message:
{original}

Write the rewritten message (no preamble):"""


class SearchNaturalizer(TurnNaturalizer):
    """Naturalizer for search (BrowseComp+) domain.

    Simple rephrasing approach -- preserves all information, only improves
    naturalness.
    """

    def __init__(self, model: str = "gpt-5.1"):
        self.model = model

    def naturalize_turn(
        self,
        content: StructuredTurnContent,
        history: list[tuple[str, str]],
    ) -> str:
        if content.turn_idx == 0:
            prompt = _SEARCH_PROMPT_FIRST.format(
                original=content.rule_based_text,
            )
        else:
            hist_str = _format_history(history)
            resp_block = _get_response_aware_block(content.turn_idx, _SEARCH_RESPONSE_AWARE_VARIANTS) if _has_assistant_response(history) else ""
            prompt = _SEARCH_PROMPT_FOLLOWUP.format(
                history=hist_str,
                original=content.rule_based_text,
                response_aware_block=resp_block,
            )
        return _call_llm_validated(prompt, self.model, max_tokens=None, fallback=content.rule_based_text)


# ===========================================================================
# Factory
# ===========================================================================

_DOMAIN_NATURALIZERS: dict[str, type[TurnNaturalizer]] = {
    "search": SearchNaturalizer,
}


def supports_llm_naturalization(domain: str, model: str | None) -> bool:
    """Whether online LLM naturalization actually runs for this domain/model.

    True only when a model is given and the domain has a dedicated LLM
    naturalizer; otherwise turns fall back to the rule-based naturalizer.
    """
    return model is not None and domain in _DOMAIN_NATURALIZERS


def create_naturalizer(
    domain: str,
    model: str | None = None,
) -> TurnNaturalizer:
    """Create a naturalizer for the given domain.

    Only ``search`` has a dedicated LLM naturalizer; all other domains
    (math, sql, swe_bench_verified, ...) fall back to the rule-based turns.

    Args:
        domain: Dataset domain (e.g. search, math, sql).
        model: LLM model name. If None, returns RuleBasedNaturalizer.

    Returns:
        A TurnNaturalizer instance.
    """
    if model is None:
        return RuleBasedNaturalizer()
    cls = _DOMAIN_NATURALIZERS.get(domain)
    if cls is None:
        return RuleBasedNaturalizer()
    return cls(model=model)
