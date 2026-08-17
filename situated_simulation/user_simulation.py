#!/usr/bin/env python3
"""
EvolvingIntent - A DataLoader-like user simulation for multi-turn LLM evaluation.

This module provides a user-simulation environment that can be directly imported and used
in Python, similar to PyTorch DataLoader.

Usage:
    from situated_simulation.user_simulation import EvolvingIntent

    # Argument-revision: 3 turns, 2 argument revisions.
    # The scenario is inferred from num_turns / num_revisions / num_switches.
    sim = EvolvingIntent(
        data_path="final_dataset/gsm8k_final.json",
        mode="eval",
        num_turns=3,
        num_revisions=2,
    )

    # Iterate over samples
    for sample in sim:
        turns = sample.turns      # [{"role": "user", "content": "..."}, ...]
        label = sample.label      # ground-truth answer
        metadata = sample.metadata

    # Or access by index
    sample = sim[0]

    # Get length
    print(f"Total samples: {len(sim)}")
"""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal
from pathlib import Path

from situated_simulation.user_intent import ChangePlan


# =============================================================================
# Constants - Domain-specific Prefixes
# =============================================================================

# Math domain prefixes for CONDITION CHANGE
MATH_ARGUMENT_PREFIXES = {
    # Correction - single item
    "correction_single": [
        "Wait, I made a mistake.",
        "Actually, that was wrong.",
        "Sorry, I need to correct that.",
        "Hold on, that wasn't right.",
        "I made an error earlier.",
        "Let me fix that.",
    ],
    # Correction - multiple items
    "correction_multiple": [
        "Wait, I made some mistakes.",
        "Actually, those were wrong.",
        "Sorry, I need to correct a few things.",
        "Hold on, some of that wasn't right.",
        "I made several errors earlier.",
        "Let me fix a few things.",
    ],
    # Exploration / Hypothetical - single item
    "exploration_single": [
        "What if we change it to:",
        "Let's try a different case.",
        "Consider instead:",
        "Suppose instead that:",
        "Let's explore another scenario.",
        "How about this case:",
    ],
    # Exploration / Hypothetical - multiple items
    "exploration_multiple": [
        "What if we change these:",
        "Let's try different cases.",
        "Consider these instead:",
        "Suppose instead:",
        "Let's explore other scenarios.",
        "How about these cases:",
    ],
    # Revision - single item
    "revision_single": [
        "Let me revise that.",
        "Actually, let's change that.",
        "I'd like to modify that argument.",
        "Let's update that.",
        "On second thought, let's change it.",
        "Let me change that to:",
    ],
    # Revision - multiple items
    "revision_multiple": [
        "Let me revise those.",
        "Actually, let's change these.",
        "I'd like to modify some arguments.",
        "Let's update a few things.",
        "On second thought, let's change some things.",
        "Let me change these:",
    ],
    # New info - used when new (non-correction) info follows corrections in the same turn
    "new_info": [
        "Also,",
        "Additionally,",
        "On top of that,",
        "Furthermore,",
        "Besides that,",
        "Moreover,",
    ],
    # Secondary correction - used when argument corrections follow a function change
    # in the same turn. Lighter tone, feels like an aside/addition.
    "secondary_correction_single": [
        "Oh, and one more thing —",
        "By the way,",
        "And I should mention,",
        "Oh, and also,",
        "While I'm at it,",
        "And just to clarify,",
    ],
    "secondary_correction_multiple": [
        "Oh, and a few more things —",
        "By the way,",
        "And I should also mention,",
        "Oh, and also,",
        "While I'm at it,",
        "And just to clarify a few things,",
    ],
    # Argument reveal - used in under-specified scenario when revealing new arguments
    "reveal_single": [
        "Oh, I forgot to mention:",
        "I should also tell you:",
        "Here's another detail:",
        "One more thing —",
        "I forgot to add:",
        "Also, here's a constraint:",
    ],
    "reveal_multiple": [
        "Oh, I forgot to mention a few things:",
        "I should also tell you some more details:",
        "Here are some more details:",
        "A few more things —",
        "I forgot to add some details:",
        "Also, here are more constraints:",
    ],
}

# Math domain prefixes for FUNCTION CHANGE
# Note: Avoid words like "different", "new", "something else", "rephrase"
# These imply completely unrelated questions, but function changes share arguments.
MATH_FUNCTION_PREFIXES = {
    # Correction style - user realizes they asked wrong question
    "correction": [
        "Wait, that's not what I meant to ask.",
        "Actually, that's not what I wanted to know.",
        "Hold on, I asked the wrong thing.",
        "Sorry, that wasn't my intended question.",
        "Let me correct my question.",
        "That's not quite right. Here's what I meant:",
    ],
    # Related style - user asks a related question (explicitly connected)
    "related": [
        "Here's a related question.",
        "Building on that, let me ask:",
        "Along those lines, I want to know:",
        "In a similar vein, what about:",
        "Relatedly, can you tell me:",
        "On a related note:",
    ],
    # Redirect style - user wants to ask something else instead
    "redirect": [
        "Actually, here's what I really want to know.",
        "Let me ask this instead.",
        "Instead, I'd like to know:",
        "What I actually want to ask is:",
        "Let me redirect my question.",
        "Here's what I'm really curious about:",
    ],
    # Angle style - exploring another angle of the same topic
    "angle": [
        "Let's look at this from another angle.",
        "Let's approach this another way.",
        "Let me ask about a related aspect.",
        "From another perspective:",
        "Looking at it another way:",
        "Let's consider another aspect:",
    ],
}

# Math domain prefixes for PREDECESSOR function change (taxonomy-specific)
# Each type reflects a natural causal chain where the previous answer feeds
# into the next question.
MATH_BACKWARD_FUNCTION_PREFIXES = {
    # T1: lookup_then_compute — user found a value/rate, now wants to compute with it
    "T1": [
        "Thanks, that helps. Now I want to figure out:",
        "Good, now using that, can you calculate:",
        "Got it. Now based on that, what is:",
        "Okay. Now with that information, tell me:",
        "Right. So now I need to work out:",
    ],
    # T2: total_then_component — user found the total, now wants to isolate a part
    "T2": [
        "Okay, now let me ask about just one part of that.",
        "Got it. But now I only need to know about:",
        "Right. Now let's break that down —",
        "Thanks. Now zooming in on one piece:",
        "Good. But specifically, what about:",
    ],
    # T3: compute_then_extend — user computed something, now wants to extend/scale it
    "T3": [
        "Great. Now let's take that a step further.",
        "Okay, now can you extend that? Specifically:",
        "Thanks. Building on that result,",
        "Good. Now using that, what happens if:",
        "Right, so now let's scale that up —",
    ],
    # T4: reframe_problem — user reframes the problem from a different angle
    "T4": [
        "Actually, let me think about this differently.",
        "Hmm, let me approach this from another angle.",
        "Wait, let me reframe my question.",
        "Actually, I want to look at it this way instead:",
        "Let me ask about this in a different way.",
    ],
}

# Search domain prefixes for CONDITION CHANGE
SEARCH_ARGUMENT_PREFIXES = {
    # Correction - single item
    "correction_single": [
        "Wait, I got a detail wrong.",
        "Actually, that detail wasn't right.",
        "Sorry, let me correct that.",
        "Hold on, I need to fix that.",
        "I gave you wrong information earlier.",
        "Let me correct that.",
    ],
    # Correction - multiple items
    "correction_multiple": [
        "Wait, I got some details wrong.",
        "Actually, some of those details weren't right.",
        "Sorry, let me correct a few things.",
        "Hold on, some of that was wrong.",
        "I gave you some wrong information earlier.",
        "Let me correct a few things.",
    ],
    # Exploration / Hypothetical - single item
    "exploration_single": [
        "What if this detail were different:",
        "Let's try a different scenario.",
        "Consider this instead:",
        "Suppose instead that:",
        "Let's explore another case.",
        "How about this scenario:",
    ],
    # Exploration / Hypothetical - multiple items
    "exploration_multiple": [
        "What if these details were different:",
        "Let's try different scenarios.",
        "Consider these instead:",
        "Suppose instead:",
        "Let's explore other cases.",
        "How about these scenarios:",
    ],
    # Revision - single item
    "revision_single": [
        "Let me revise that.",
        "Actually, let's change that detail.",
        "I'd like to update that.",
        "Let me change that.",
        "On second thought, let's change it.",
        "Let me update that to:",
    ],
    # Revision - multiple items
    "revision_multiple": [
        "Let me revise those.",
        "Actually, let's change some of those details.",
        "I'd like to update a few things.",
        "Let me change some of those.",
        "On second thought, let's change a few things.",
        "Let me update these:",
    ],
    # New info - used when new (non-correction) info follows corrections in the same turn
    "new_info": [
        "Also,",
        "Additionally,",
        "On top of that,",
        "Furthermore,",
        "Besides that,",
        "Moreover,",
    ],
    # Secondary correction - used when argument corrections follow a function change
    "secondary_correction_single": [
        "Oh, and one more thing —",
        "By the way,",
        "And I should mention,",
        "Oh, and also,",
        "While I'm at it,",
        "And just to clarify,",
    ],
    "secondary_correction_multiple": [
        "Oh, and a few more things —",
        "By the way,",
        "And I should also mention,",
        "Oh, and also,",
        "While I'm at it,",
        "And just to clarify a few things,",
    ],
    # Argument reveal - used in under-specified scenario when revealing new arguments
    "reveal_single": [
        "Oh, I forgot to mention:",
        "I should also tell you:",
        "Here's another detail:",
        "One more thing —",
        "I forgot to add:",
        "Also, here's a clue:",
    ],
    "reveal_multiple": [
        "Oh, I forgot to mention a few things:",
        "I should also tell you some more details:",
        "Here are some more details:",
        "A few more things —",
        "I forgot to add some details:",
        "Also, here are more clues:",
    ],
}

# Search domain prefixes for FUNCTION CHANGE
SEARCH_FUNCTION_PREFIXES = {
    # Shift - user's interest naturally moves
    "shift": [
        "Now I'm curious about something else.",
        "That makes me wonder:",
        "This leads me to another question.",
        "Now what I'd like to know is:",
        "Thinking about it more, I want to ask:",
        "That brings up another question:",
    ],
    # Follow-up - building on what was discussed
    "follow_up": [
        "As a follow-up:",
        "Following up on that,",
        "Based on what we've discussed,",
        "Given what you found, I now want to know:",
        "With that in mind, can you also find out:",
        "Now that we know that, I'm wondering:",
    ],
    # Related - same topic, different angle
    "related": [
        "Here's a related question.",
        "Along those lines, I want to know:",
        "In a similar vein, what about:",
        "Relatedly, can you tell me:",
        "On a related note:",
        "Building on that, let me ask:",
    ],
    # Pivot - gentle shift of focus
    "pivot": [
        "Actually, what I'm more interested in is:",
        "What I'd really like to find out is:",
        "More importantly, I want to know:",
        "Let me focus on this instead:",
        "What I'm really after is:",
        "The thing I most want to know is:",
    ],
}

# Search domain prefixes for PREDECESSOR function change (follow-up style only)
SEARCH_BACKWARD_FUNCTION_PREFIXES = [
    "As a follow-up:",
    "Following up on that,",
    "Based on what we've discussed,",
    "Given what you found, I now want to know:",
    "With that in mind, can you also find out:",
    "Now that we know that, I'm wondering:",
    "That brings up another question:",
    "This leads me to another question.",
    "Building on that, let me ask:",
    "That makes me wonder:",
]

# SQL domain prefixes for CONDITION CHANGE (WHERE-clause filter updates)
SQL_ARGUMENT_PREFIXES = {
    # Correction - single item
    "correction_single": [
        "Wait, I got the filter wrong.",
        "Actually, that filter value wasn't right.",
        "Sorry, I need to correct that filter.",
        "Hold on, that wasn't the right value.",
        "I gave you the wrong filter.",
        "Let me fix that filter.",
    ],
    # Correction - multiple items
    "correction_multiple": [
        "Wait, I got some filters wrong.",
        "Actually, some of those filter values weren't right.",
        "Sorry, I need to correct a few filters.",
        "Hold on, some of those weren't the right values.",
        "I gave you some wrong filters.",
        "Let me fix a few filters.",
    ],
    # Exploration / Hypothetical - single item
    "exploration_single": [
        "What if we filter by this instead:",
        "Let's try a different filter value.",
        "Consider filtering by this instead:",
        "Suppose we change the filter to:",
        "Let's try another filter.",
        "How about this filter value:",
    ],
    # Exploration / Hypothetical - multiple items
    "exploration_multiple": [
        "What if we use different filters:",
        "Let's try different filter values.",
        "Consider these filters instead:",
        "Suppose we change the filters to:",
        "Let's try other filters.",
        "How about these filter values:",
    ],
    # Revision - single item
    "revision_single": [
        "Let me update that filter.",
        "Actually, let's change that filter.",
        "I'd like to modify that filter value.",
        "Let's update that filter.",
        "On second thought, let's change that filter.",
        "Let me change that filter to:",
    ],
    # Revision - multiple items
    "revision_multiple": [
        "Let me update those filters.",
        "Actually, let's change some filters.",
        "I'd like to modify a few filter values.",
        "Let's update some of those filters.",
        "On second thought, let's change a few filters.",
        "Let me change these filters:",
    ],
    # New info - used when new (non-correction) info follows corrections in the same turn
    "new_info": [
        "Also,",
        "Additionally,",
        "On top of that,",
        "Furthermore,",
        "Besides that,",
        "Moreover,",
    ],
    # Secondary correction - used when argument corrections follow a function change
    # in the same turn. Lighter tone, feels like an aside/addition.
    "secondary_correction_single": [
        "Oh, and one more thing —",
        "By the way,",
        "And I should mention,",
        "Oh, and also,",
        "While I'm at it,",
        "And just to clarify,",
    ],
    "secondary_correction_multiple": [
        "Oh, and a few more things —",
        "By the way,",
        "And I should also mention,",
        "Oh, and also,",
        "While I'm at it,",
        "And just to clarify a few things,",
    ],
    # Argument reveal - used in under-specified scenario when revealing new arguments
    "reveal_single": [
        "Oh, I forgot to mention:",
        "I should also tell you:",
        "Here's another filter:",
        "One more thing —",
        "I forgot to add:",
        "Also, here's a constraint:",
    ],
    "reveal_multiple": [
        "Oh, I forgot to mention a few things:",
        "I should also tell you some more details:",
        "Here are some more filters:",
        "A few more things —",
        "I forgot to add some details:",
        "Also, here are more constraints:",
    ],
}

# SQL domain prefixes for FUNCTION CHANGE (flat list — no taxonomy categories)
# Curated via degrade-rate analysis on bird_sql_n100 t2_g1 across gpt-5.1 and
# gpt-5.4: kept the three lowest-degrade originals (Tier 1) and added three
# conservative variants with matching strength. "Related / different angle /
# different question about this data" prefixes were removed because they
# signal context continuation and caused >80% degrade rates in both models.
SQL_FUNCTION_PREFIXES = [
    # Original Tier 1 (empirically safest in t2_g1)
    "Instead, I'd like to know:",
    "Let me change what I'm looking for.",
    "Hold on, I asked for the wrong thing.",
    # Conservative additions (matched strength, natural user self-correction)
    "On second thought, I'd like to know:",
    "Let me change my question.",
    "Actually, I asked the wrong question.",
]

# ── Function-naturalized prefix support for SQL ──────────────────────────
# Maps aggregate keywords (as they appear in transition_reason) to
# natural-language noun phrases a user would say.
AGG_NL_MAP = {
    "count": "the total number",
    "average": "the average",
    "minimum": "the minimum",
    "maximum": "the maximum",
    "sum": "the total",
}

SQL_NATURALIZED_FUNCTION_TEMPLATES = [
    "How about {new} instead?",
    "What about {new} instead?",
    "Actually, can we get {new} instead?",
    "Let's do {new} instead.",
]

SQL_NATURALIZED_COLUMN_TEMPLATES = [
    "How about {new} instead?",
    "What about {new} instead?",
    "Actually, let's look at {new} instead.",
]

# SQL domain prefixes for PREDECESSOR function change (follow-up style)
SQL_BACKWARD_FUNCTION_PREFIXES = [
    "As a follow-up:",
    "Following up on that,",
    "Based on that result,",
    "Given what we found, I now want to know:",
    "With that in mind, can you also query:",
    "Now that we know that, I'm wondering:",
    "That brings up another question:",
    "Building on that, let me ask:",
    "Good, now using that data,",
    "Thanks. Now I also want to check:",
]

# SQL domain prefixes for the multi-clause "keep-arguments, change-function"
# rewrite produced by generate_predecessors_sql_llm.py. These prefixes signal to the
# model that the user's prior filter constraints (WHERE/HAVING/LIMIT) still
# apply but the question itself has changed.
SQL_FUNCTION_FOLLOWUP_PREFIXES = [
    "Among those,",
    "From that same group,",
    "Within those same results,",
    "Sticking with the same filters,",
    "Keeping the same arguments,",
    "With the same constraints,",
    "Out of those,",
    "From the same set,",
]

# SQL predecessor prefixes that explicitly anchor to the row population
# (the entity carved out by WHERE/HAVING). The "{pop}" slot is filled with the
# pluralized FROM-table name (e.g. "schools", "players", "menu items"). These
# prefixes implement the population-scoped, "keep arguments, swap function"
# pattern: they preserve the data subset without re-listing argument values
# and without leaking function-shape (SELECT/GROUP BY/ORDER BY/LIMIT).
SQL_FUNCTION_PREFIXES_POPULATION = [
    "On the same set of {pop}, instead can you tell me:",
    "Within the same set of {pop}, instead I'd like to know:",
    "From the same group of {pop}, instead — ",
    "Among those same {pop}, instead can you tell me:",
    "Looking at the same {pop}, instead — ",
    "Keeping the same {pop} in scope, instead I want to know:",
    "On those same {pop}, instead — ",
]

# Generic fallback used when the FROM-table name is opaque (acronyms, short
# cryptic codes like "frpm", "cct"). Uses pure anaphora to the row subset.
SQL_FUNCTION_PREFIXES_GENERIC = [
    "Within that same subset, instead can you tell me:",
    "From that same group, instead — ",
    "On that same data, instead I'd like to know:",
    "Among those same rows, instead — ",
    "Within that same filtered set, instead can you tell me:",
]


# Heuristic: BIRD has cryptic table names (e.g. "frpm", "cct", "satscores").
# Use the generic fallback pool when the table name is short+lowercase or all
# uppercase or made of very short tokens.
_OPAQUE_TABLE_RE = re.compile(r"^[a-z]{1,3}$")

# Generic container nouns that read as English words but don't anchor a
# meaningful entity (e.g. BIRD's "events" table holds complaints, "logs"
# holds calls). Saying "On the same set of events" leaks no info but adds
# no anchor either, so prefer the generic anaphora pool.
_GENERIC_CONTAINER_NOUNS = frozenset({
    "event", "events",
    "log", "logs",
    "record", "records",
    "entry", "entries",
    "item", "items",
    "row", "rows",
    "datum", "data",
    "info", "infos", "information",
    "detail", "details",
    "history", "histories",
    "list", "lists",
    "table", "tables",
    "thing", "things",
    "object", "objects",
    "value", "values",
    "result", "results",
})


def _split_table_name(table: str) -> list[str]:
    """Split snake_case / CamelCase / mixed into lowercase word list."""
    parts = re.findall(
        r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", table
    )
    return [p.lower() for p in parts if p]


def _pluralize(word: str) -> str:
    """Tiny rule-based pluralizer, sufficient for BIRD table names."""
    if not word:
        return word
    if word.endswith("s"):
        return word
    if word.endswith(("x", "z")) or word.endswith(("sh", "ch")):
        return word + "es"
    if (
        len(word) > 1
        and word.endswith("y")
        and word[-2] not in "aeiou"
    ):
        return word[:-1] + "ies"
    return word + "s"


def _is_opaque_table(table: str, words: list[str]) -> bool:
    """Detect cryptic table names that wouldn't read naturally as a noun."""
    if not table or not words:
        return True
    if _OPAQUE_TABLE_RE.fullmatch(table):
        return True
    if table.isupper() and len(table) <= 5:
        return True
    if all(len(w) <= 3 for w in words):
        return True
    # Very short word with no vowels (e.g. "frpm", "cct") reads as a code,
    # not as an English noun.
    last = words[-1]
    if len(last) <= 5 and not any(c in "aeiou" for c in last):
        return True
    # Generic container nouns ("events", "logs", "records", ...) don't
    # anchor a meaningful entity. The whole phrase is a generic container
    # iff every word is one — multi-word tables like
    # "competitor_events" still anchor on the modifier.
    if all(w in _GENERIC_CONTAINER_NOUNS for w in words):
        return True
    return False


def _extract_population_noun(sql: str) -> tuple[str | None, bool]:
    """Extract a human-readable plural noun for the FROM-table population.

    Returns (noun, is_opaque). When is_opaque=True, the caller should use
    the generic prefix pool. The noun is the pluralized, space-separated
    form of the first FROM-table (e.g. "schools", "menu items").
    """
    if not sql:
        return None, True
    m = re.search(
        r"\bFROM\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
        sql,
        re.IGNORECASE,
    )
    if not m:
        return None, True
    table = m.group(1)
    words = _split_table_name(table)
    if _is_opaque_table(table, words):
        return None, True
    words[-1] = _pluralize(words[-1])
    return " ".join(words), False

# Correction-after-reveal prefixes: used when a correction follows a
# reveal in the same turn, making it clear the correction targets
# earlier turns (not what was just revealed).
CORR_AFTER_REVEAL_SINGLE = [
    "Oh, and I need to fix something from before —",
    "Also, I realize I gave you wrong info earlier.",
    "By the way, I made an error before.",
    "One more thing — I need to correct what I said earlier.",
    "Additionally, I made a mistake earlier —",
    "Also, I should fix something I mentioned before:",
]
CORR_AFTER_REVEAL_MULTIPLE = [
    "Oh, and I need to fix a few things from before —",
    "Also, I realize I gave you some wrong info earlier.",
    "By the way, I made a few errors before.",
    "One more thing — I need to correct some things I said earlier.",
    "Additionally, I made some mistakes earlier —",
    "Also, let me fix a few things I mentioned before:",
]

# Reveal-after-function prefixes: used when reveals follow a function change
# in the same turn, bridging the new question with additional context.
REVEAL_AFTER_FUNCTION_SINGLE = [
    "Additionally,",
    "Also, here's some relevant info:",
    "One more thing you should know:",
    "By the way,",
    "And for context,",
    "Oh, and you should also know that",
]
REVEAL_AFTER_FUNCTION_MULTIPLE = [
    "Additionally, here are some details:",
    "Also, here's some more context:",
    "And here are a few things you should know:",
    "By the way, here are some relevant details:",
    "Oh, and for context, here are some more details:",
    "Also, you should know the following:",
]

# Supported domains
SUPPORTED_DOMAINS = ["math", "search", "sql", "swe_bench_verified"]


# ---------------------------------------------------------------------------
# SWE-bench Verified prefixes
# ---------------------------------------------------------------------------
# Tone: a developer working an issue ticket — half-remembering details,
# pivoting between exploration / a related bug / the actual fix, correcting
# themselves as they re-read code or stack traces. Avoid math/SQL flavour.

# Function-change prefixes for SWE: developer pivots between repo orientation
# (G1), a sibling real bug (G2), and the target bug (G3). The transition
# typically reads "actually, before that..." or "wait, let me back up and
# fix this other thing first."
SWE_FUNCTION_PREFIXES: dict[str, list[str]] = {
    # Predecessor was a "real bug" (a different SWE-bench instance) — pivot
    # to another bug. Replace semantics: drop the previous, do this instead.
    "from_bug": [
        "Actually, scratch that — different bug instead:",
        "Hmm, change of plan. Forget the previous one:",
        "Hold on — let me redirect. Different issue:",
        "Wait, the priority just changed. Forget that, this is what I need:",
        "OK pivot — let's do this one instead:",
        "Actually, ignore that for now. Here's what I actually need:",
        "Switching tracks — different bug:",
        "Let me back out of that one. Different issue I'd rather tackle:",
        "Different ask — let's drop the previous one:",
        "Hmm, on reflection that's lower priority. Let's do this instead:",
    ],
    # Predecessor was an exploration question (concept / dependency / repo
    # layout / convention etc.). The user has the context they wanted and
    # is now moving on to the actual coding task. "Different bug instead"
    # would be wrong here — the previous turn was not a bug.
    "from_exploration": [
        "Got it, thanks. Now the actual task —",
        "Helpful, thanks. Different request:",
        "OK that helps. Now what I really need:",
        "Cool, makes sense. Now, separate task:",
        "Thanks for the context. Moving on to my actual issue:",
        "Right, that clears it up. Now my real ask:",
        "OK good. Different thing I need help with:",
        "Got it. On to the actual coding task:",
        "Thanks. Switching gears — what I actually want fixed:",
        "Useful, thanks. Now the real request:",
    ],
}

# Backward-compat: a flat list view, used by older callers if any.
SWE_FUNCTION_PREFIXES_FLAT = (
    SWE_FUNCTION_PREFIXES["from_bug"] + SWE_FUNCTION_PREFIXES["from_exploration"]
)


# Taxonomy types used by the SWE-bench paired data to mark exploration
# (concept / repo-layout / dependency-map / API / etc.) functions. Anything
# else (incl. None) is treated as a real bug for prefix-routing purposes.
SWE_EXPLORATION_TAXONOMIES: frozenset[str] = frozenset({
    "dependency_map",
    "io_shape",
    "module_overview",
    "public_api",
    "testing_convention",
    "repo_layout",
    "coding_convention",
})


def _swe_function_prefix_category(prev_taxonomy_type: str | None) -> str:
    """Map outgoing function's taxonomy_type to a SWE function-switch prefix bucket.

    The bucket is selected by the *previous* function's nature, since the
    prefix is what the user says when *leaving* that function:
      - exploration → moving on to real task → "from_exploration" bucket
      - real bug    → pivoting to different bug → "from_bug" bucket
    """
    if prev_taxonomy_type and prev_taxonomy_type in SWE_EXPLORATION_TAXONOMIES:
        return "from_exploration"
    return "from_bug"

# Correction prefixes when the user is updating a previously-stated detail
# about the bug (constraint/scope/approach/location/trigger). Reads like a
# developer re-reading the issue or stack trace and updating their priors.
SWE_CORRECTION_PREFIXES = [
    "Sorry, correction —",
    "Wait, I had that wrong.",
    "Actually, let me correct that —",
    "Hmm, after re-reading the code,",
    "Oh — I misread the stack trace. Actually,",
    "Let me revise what I said —",
    "Update from digging into the source:",
    "Actually no, that's not quite right —",
    "Scratch that — it's actually:",
    "I was wrong about that earlier. Updated:",
]

# "New info" prefixes used when extra details follow a correction in the
# same turn. Lighter than corrections; reads like an aside.
SWE_NEW_INFO_PREFIXES = [
    "Also,",
    "And one more thing —",
    "By the way,",
    "While I'm at it,",
    "Oh, and",
    "Plus,",
    "Btw,",
    "Also, just to add:",
]

# Reveal prefixes for under-specified scenarios — developer supplies more
# context across turns ("forgot to mention…", "just remembered…").
SWE_REVEAL_PREFIXES = [
    "I should also mention:",
    "Forgot to add —",
    "One more detail:",
    "Oh, and you should know:",
    "Additional context:",
    "Also relevant:",
    "Btw, here's another piece:",
    "Just remembered —",
    "Worth noting:",
    "And here's a related detail:",
]
# ---------------------------------------------------------------------------

# Default instructions per domain (used when instruction=None)
DEFAULT_INSTRUCTIONS = {
    "math": (
        "Solve this problem step by step. "
        "Put your final answer within \\boxed{{}}.\n\n{content}"
    ),
    "sql": (
        "Write a SQL query to answer the following question. "
        "Return only the SQL query in a ```sql code block. "
        "Select only the column(s) the question asks for — do not include "
        "extra columns such as IDs, counts, or aggregates unless explicitly "
        "requested.\n\n{content}"
    ),
    "swe_bench_verified": "{content}",
}


# Default system prompts per domain (used when system_prompt=None).
# Currently SWE is the only domain whose instruction is genuinely
# system-level (multi-turn intent-tracking guidance + output format) rather
# than per-turn framing. Other domains keep their guidance in user turn 1
# for backwards compatibility with existing eval results.
DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "swe_bench_verified": (
        "You are helping fix a bug or address a feature request in an "
        "open-source Python repository. The user will describe the issue "
        "across multiple turns, and may correct or add details as the "
        "conversation progresses — pay close attention to the FINAL state "
        "of the user's intent (the latest version of every constraint, "
        "scope, approach, location, and reproduction step), not just what "
        "they said in earlier turns.\n\n"
        "When you have enough information, propose a fix as a unified diff "
        "patch in a ```diff code block, applied against the repository "
        "files referenced by the user. Include only the minimal changes "
        "required; do not modify unrelated code."
    ),
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IntentSample:
    """A single sample representing a multi-turn conversation.

    Supports two usage modes:
    1. Direct access: iterate ``turns`` (backward-compatible).
    2. Step-wise delivery: call ``reset()`` / ``step()`` / ``is_done()``
       to drive a multi-turn conversation loop, optionally with online
       naturalization (set ``naturalizer`` before ``reset()``).
    """
    task_id: str
    turns: list[dict[str, str]]           # All turns as list of {"role": "user", "content": ...}
    label: str                             # Ground truth answer
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Step-wise delivery support (populated by create_sample) ---
    structured_contents: list | None = field(default=None, repr=False)
    naturalizer: Any | None = field(default=None, repr=False)
    recap_texts: list[str | None] | None = field(default=None, repr=False)

    # --- Mutable state (managed by reset/step) ---
    _cursor: int = field(default=0, init=False, repr=False)
    _nat_history: list = field(default_factory=list, init=False, repr=False)
    _user_turn_idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        # Capitalize the first character of the first user turn for naturalness.
        # turns[0] may be a system prompt (non-SQL domains), so find the first
        # user turn rather than assuming index 0.
        for turn in self.turns:
            if turn.get("role") == "user" and turn.get("content"):
                content = turn["content"]
                turn["content"] = content[0].upper() + content[1:]
                break

    @property
    def num_turns(self) -> int:
        """Total number of turns."""
        return len(self.turns)

    # ------------------------------------------------------------------ #
    # Step-wise conversation API
    # ------------------------------------------------------------------ #

    def reset(self) -> list[dict[str, str]]:
        """Start (or restart) step-wise delivery.

        Returns the initial messages: system prompt (if any) + first user
        turn.  Resets all mutable state so the sample can be replayed.
        """
        self._cursor = 0
        self._nat_history = []
        self._user_turn_idx = 0

        # Seed naturalizer state from Turn 0 so online step() calls
        # have correct _prev_function / _prev_arguments context.
        if self.naturalizer and self.structured_contents:
            self.naturalizer.seed_state(self.structured_contents[0])

        initial: list[dict[str, str]] = []
        while self._cursor < len(self.turns):
            turn = self.turns[self._cursor]
            initial.append(dict(turn))          # shallow copy
            self._cursor += 1
            if turn["role"] == "user":
                self._user_turn_idx = 1         # first user turn delivered
                break

        return initial

    def step(self, model_response: str) -> list[dict[str, str]]:
        """Deliver the next user turn(s).

        If *naturalizer* and *structured_contents* are set, the turn is
        naturalised on-the-fly using the accumulated conversation history
        (including *model_response*).  Otherwise the pre-built turn is
        returned verbatim.

        Must only be called when :meth:`is_done` is ``False``.
        """
        if self._cursor >= len(self.turns):
            raise RuntimeError("step() called after all turns delivered")

        # Record the model response in naturalizer history
        self._nat_history.append(model_response)

        use_naturalizer = (
            self.naturalizer is not None
            and self.structured_contents is not None
            and self._user_turn_idx < len(self.structured_contents)
        )

        if use_naturalizer:
            # Build (user_text, assistant_response) pairs for the naturalizer
            user_texts = [
                t["content"] for t in self.turns[:self._cursor]
                if t["role"] == "user"
            ]
            history_pairs: list[tuple[str, str]] = []
            for i, u in enumerate(user_texts):
                a = self._nat_history[i] if i < len(self._nat_history) else ""
                history_pairs.append((u, a))

            content = self.structured_contents[self._user_turn_idx]
            natural_text = self.naturalizer.naturalize_turn(
                content, history=history_pairs,
            )
            # Append recap (deterministic, after naturalization)
            if self.recap_texts and self._user_turn_idx < len(self.recap_texts):
                recap = self.recap_texts[self._user_turn_idx]
                if recap is not None:
                    natural_text = natural_text + " " + recap
            result = [{"role": "user", "content": natural_text}]
        else:
            # Pre-built turn delivery (like former RuleBasedUser)
            result = []
            while self._cursor < len(self.turns):
                turn = self.turns[self._cursor]
                result.append(dict(turn))
                self._cursor += 1
                if turn["role"] == "user":
                    break

        self._user_turn_idx += 1

        # When naturalizing, still advance cursor to keep is_done() in sync
        if use_naturalizer:
            while self._cursor < len(self.turns):
                self._cursor += 1
                if self.turns[self._cursor - 1]["role"] == "user":
                    break

        return result

    def is_done(self) -> bool:
        """``True`` when all user turns have been delivered."""
        return self._cursor >= len(self.turns)


def extract_ground_truth(sample: dict[str, Any]) -> str:
    """Extract ground truth answer from sample."""
    answer_raw = sample.get("answer", "")
    
    # GSM8K format: #### <number>
    gsm8k_match = re.search(r"####\s*(\-?[0-9\.\,]+)", answer_raw)
    if gsm8k_match:
        return gsm8k_match.group(1).replace(",", "")
    
    # MATH format: \boxed{...}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", answer_raw)
    if boxed_match:
        return boxed_match.group(1)
    
    return str(answer_raw) if answer_raw else ""


def _is_sql_sample(raw: dict[str, Any]) -> bool:
    """Check if a raw sample is from the BIRD-SQL dataset."""
    return raw.get("data_source") == "bird_sql"


def _sql_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Extract SQL-specific metadata fields from a raw BIRD-SQL sample.
    Returns an empty dict for non-SQL samples.
    """
    if not _is_sql_sample(raw):
        return {}
    return {
        "gold_sql": raw.get("gold_sql", ""),
        "db_path": raw.get("db_path", ""),
        "db_id": raw.get("db_id", ""),
        "schema": raw.get("schema", ""),
    }


def _build_sql_system_prompt(raw: dict[str, Any], base_prompt: str | None) -> str | None:
    """
    For SQL samples, prepend schema to the system prompt.
    For non-SQL samples, return base_prompt unchanged.
    """
    if not _is_sql_sample(raw):
        return base_prompt
    schema = raw.get("schema", "")
    sql_prompt = (
        "You are a SQL assistant. Given a database schema and a natural language "
        "question, generate a SQL query to answer the question.\n\n"
        "Important notes:\n"
        "- Use SQLite syntax\n"
        "- Return only the SQL query in a ```sql code block\n"
        "- Do not include explanations\n"
    )
    if schema:
        sql_prompt += f"\nDatabase schema:\n{schema}\n"
    if base_prompt:
        sql_prompt = base_prompt + "\n\n" + sql_prompt
    return sql_prompt



def _join_prefix_content(prefix: str, content: str) -> str:
    """Join a prefix and content, lowercasing content's first letter when natural.

    Only lowercases when the content starts with a typical instruction verb
    (e.g. "Include", "Write", "Use") so that "Also, include keyword..."
    reads naturally, while "Also, She bakes..." keeps proper casing.
    """
    if not content:
        return prefix
    _INSTRUCTION_STARTERS = {
        "include", "write", "use", "ensure", "provide", "nest", "maintain",
        "end", "start", "begin", "format", "limit", "add", "place", "put",
        "repeat", "generate", "create", "make", "list", "answer", "respond",
        "do", "the", "each", "every", "no", "at", "in", "your",
    }
    if prefix and prefix.rstrip().endswith((",", ":")):
        first_word = content.split()[0] if content.split() else ""
        if first_word.lower() in _INSTRUCTION_STARTERS:
            content = content[0].lower() + content[1:]
    return f"{prefix} {content}"


def load_data(data_path: str | Path) -> list[dict[str, Any]]:
    """
    Load Stage 3 processed data.
    
    Handles:
    - List format: [sample1, sample2, ...]
    - Checkpoint format: {"results": [...]}
    - Single sample: {sample}
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    
    if isinstance(data, dict) and "function" in data and "arguments" in data:
        data = [data]
    
    if not isinstance(data, list):
        raise ValueError("Expected list of samples or single sample dict")
    
    valid_samples = [s for s in data if isinstance(s, dict) and "function" in s and "arguments" in s]
    return valid_samples


# =============================================================================
# EvolvingIntent Environment
# =============================================================================

class EvolvingIntent:
    """
    A DataLoader-like user simulation for multi-turn LLM evaluation.
    
    Supports:
    - Iteration: for sample in sim
    - Indexing: sim[0], sim[1:10]
    - Length: len(sim)
    
    Args:
        data_path: Path to Stage 3 (predecessor) output JSON file.
                   This file should contain function, arguments, counterfactual_arguments,
                   and predecessor_functions from the full pipeline.
        mode: "eval" (deterministic) or "train" (random)
        scenario: Scenario type ("argument-revision")
        num_turns: Number of conversation turns
        num_revisions: Number of revisions (arguments to revise)
        system_prompt: Optional system prompt to prepend
        instruction: Optional instruction to wrap the first turn content
        seed: Random seed for reproducibility
    
    Example:
        >>> sim = EvolvingIntent("data.json", num_turns=4, num_revisions=2)
        >>> for sample in sim:
        ...     print(sample.turns)   # All turns
        ...     print(sample.label)   # Ground truth
        
    Scenario is automatically inferred from parameters:
        - num_turns=1, no changes → fully-specified
        - num_turns>1, no changes → under-specified  
        - num_turns≥2, revisions>0 → argument-revision
        - num_turns≥2, switches>0 → function-switch
    """
    
    def __init__(
        self,
        data_path: str | Path,
        mode: Literal["eval", "train"] = "eval",
        domain: Literal["math", "search"] = "math",
        ordering: Literal["sequential", "interleaved", "mixed", "random"] = "interleaved",
        num_turns: int = 1,
        num_revisions: int = 0,
        num_switches: int = 0,
        system_prompt: str | None = None,
        instruction: str | None = None,  # Use {content} as placeholder
        seed: int = 42,
        task_ids: list[str] | set[str] | None = None,  # Filter by specific task_ids
        naturalizer_model: str | None = None,  # LLM model for naturalization; None = rule-based
        recap_method: str | None = None,  # "prompt", "dump", "ground_truth", or None
        prefix_style: str | None = None,  # "base" (default) or "function-naturalized"
        include_evidence: bool = True,  # SQL only: include BIRD evidence field in user turn 1
    ):
        # Validate domain
        if domain not in SUPPORTED_DOMAINS:
            raise NotImplementedError(
                f"Domain '{domain}' is not supported yet. "
                f"Supported domains: {SUPPORTED_DOMAINS}"
            )

        # Validate recap_method
        _VALID_RECAP_METHODS = {None, "prompt", "dump", "ground_truth"}
        if recap_method not in _VALID_RECAP_METHODS:
            raise ValueError(
                f"recap_method must be one of {_VALID_RECAP_METHODS}, "
                f"got {recap_method!r}"
            )
        
        # Auto-infer scenario from parameters
        scenario = self._infer_scenario(num_turns, num_revisions, num_switches)
        
        # Validate parameters
        self._validate_params(num_turns, num_revisions, num_switches, scenario)
        
        self.data_path = Path(data_path)
        self.mode = mode
        self.scenario = scenario
        self.domain = domain
        self.ordering = ordering
        self.num_turns = num_turns
        self.num_revisions = num_revisions
        self.num_switches = num_switches
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPTS.get(domain)
        self.instruction = instruction or DEFAULT_INSTRUCTIONS.get(domain)
        self.seed = seed
        self.task_ids_filter = set(task_ids) if task_ids else None
        self._naturalizer_model = naturalizer_model
        self._recap_method = recap_method
        # Normalize v2 alias to canonical name (same behavior, different output filename)
        _ps = prefix_style or "base"
        self._prefix_style = "function-naturalized" if _ps == "function-naturalized-v2" else _ps
        self._include_evidence = include_evidence
        
        # Prefix counters for eval mode (cycle through prefixes)
        self._eval_prefix_counter_single = 0
        self._eval_prefix_counter_multiple = 0
        self._eval_function_prefix_counter = 0  # For function change prefixes
        self._eval_new_info_prefix_counter = 0  # For new info prefixes
        self._eval_reveal_prefix_counter = 0  # For argument reveal prefixes
        
        # Load raw data
        self._raw_data = load_data(data_path)
        
        # Filter by task_ids if provided
        if self.task_ids_filter:
            self._raw_data = [
                s for s in self._raw_data 
                if s.get("task_id") in self.task_ids_filter
            ]
        
        # Build samples (lazy or eager based on mode)
        self._samples: list[IntentSample] = []
        self._build_samples()
    
    def _infer_scenario(
        self, 
        num_turns: int, 
        num_revisions: int, 
        num_switches: int
    ) -> str:
        """
        Automatically infer scenario from parameters.
        
        Rules:
        - num_turns=1, no changes → fully-specified
        - num_turns>1, no changes → under-specified
        - revisions>0, switches=0 → argument-revision
        - revisions=0, switches>0 → function-switch
        - both >0 → combined (not yet implemented)
        """
        has_revisions = num_revisions > 0
        has_switches = num_switches > 0
        
        if has_revisions and has_switches:
            return "combined"
        
        if has_revisions:
            return "argument-revision"
        elif has_switches:
            return "function-switch"
        elif num_turns == 1:
            return "fully-specified"
        else:
            return "under-specified"
    
    def _validate_params(
        self, 
        num_turns: int, 
        num_revisions: int, 
        num_switches: int,
        scenario: str
    ):
        """Validate parameter combinations."""
        if num_turns < 1:
            raise ValueError("num_turns must be at least 1")
        
        # For eval mode, strict validation
        # For train mode, these are max values, so less strict
        # (actual values will be sampled at runtime)
    
    def _build_samples(self):
        """Build all samples from raw data using the plan-first scheduler."""
        from situated_simulation.turn_scheduler import create_sample as _ts_create
        # SWE-bench: strip symptoms before scheduling, re-insert ceil-front.
        try:
            from situated_simulation.turn_scheduler_swe import create_sample_swe as _ts_create_swe
        except ImportError:  # pragma: no cover
            _ts_create_swe = None

        random.seed(self.seed)

        # Naturalizer is attached for online step()-time naturalization only.
        # Pre-build turns are always rule-based.
        nat = None
        if self._naturalizer_model:
            from situated_simulation.naturalizer import create_naturalizer
            nat = create_naturalizer(self.domain, self._naturalizer_model)

        # Dispatch all SWE samples through the SWE wrapper. Some valid tasks
        # have no symptom argument, but still need its instance metadata
        # pass-through for repository execution and official verification.
        use_swe_scheduler = (
            _ts_create_swe is not None
            and self.domain == "swe_bench_verified"
        )
        scheduler_fn = _ts_create_swe if use_swe_scheduler else _ts_create

        for raw_sample in self._raw_data:
            # SQL: precompute the row-population noun (e.g. "schools",
            # "players") so predecessor function-switch prefixes can
            # use a population-scoped framing without leaking values.
            if self.domain == "sql":
                pop, is_opaque = _extract_population_noun(
                    raw_sample.get("gold_sql", "") or ""
                )
                self._current_pop_noun = pop
                self._current_pop_opaque = is_opaque
            else:
                self._current_pop_noun = None
                self._current_pop_opaque = True

            sample = scheduler_fn(
                raw_sample,
                g=self.num_switches,
                p=self.num_revisions,
                t=self.num_turns,
                mode=self.mode,
                domain=self.domain,
                seed=self.seed,
                get_function_change_prefix=self._get_function_change_prefix,
                get_correction_prefix=self._get_correction_prefix,
                get_reveal_prefix=self._get_reveal_prefix,
                get_reveal_after_function_prefix=self._get_reveal_after_function_prefix,
                get_corr_after_reveal_prefix=self._get_corr_after_reveal_prefix,
                get_new_info_prefix=self._get_new_info_prefix,
                join_prefix_content=_join_prefix_content,
                system_prompt=self.system_prompt,
                instruction=self.instruction,
                recap_method=self._recap_method,
                function_change_includes_function_text=self._function_change_includes_function_text(),
                include_evidence=self._include_evidence,
            )
            if sample is not None:
                if nat is not None:
                    sample.naturalizer = nat
                self._samples.append(sample)
    
    def _get_function_change_prefix(self, predecessor: bool = False,
                               taxonomy_type: str | None = None,
                               function_change_new: str | None = None,
                               function_change_type: str | None = None,
                               prev_taxonomy_type: str | None = None,
                               prev_transition_phrase: str | None = None) -> str:
        """
        Get a function change prefix based on domain.

        Args:
            predecessor: If True, use follow-up style prefixes for predecessor
                      inference (causal chain). Default False uses mixed categories.
            taxonomy_type: For math/IF domains, use taxonomy-specific prefixes
                          (T1-T4) when available from predecessor inference data.
            function_change_new: For function-naturalized style, the new function's
                            aggregate/column keyword (e.g. "average", "TAT2").
            function_change_type: "aggregate_swap" or "column_swap".
            prev_taxonomy_type: Outgoing function's taxonomy_type. Currently
                used by the SWE domain to route between "from_exploration"
                and "from_bug" prefix buckets. Other domains ignore it.
            prev_transition_phrase: Outgoing function's stored transition_phrase
                (e.g. produced by the impl-precursor generator). When set,
                SWE rendering uses it verbatim instead of picking from the
                fixed prefix pools — this preserves the LLM-authored
                "before X, fix Y first" phrasing tailored to the specific
                buggy API. Other domains ignore it.

        In eval mode, cycles through all prefixes to ensure diversity
        without randomness.
        """
        # Function-naturalized style for SQL: generate rule-based prefix
        if (self._prefix_style == "function-naturalized"
                and self.domain == "sql"
                and function_change_new
                and not predecessor):
            prefix = self._build_naturalized_function_prefix(
                function_change_new, function_change_type,
            )
            if prefix is not None:
                return prefix
            # Fall through to base prefixes if building fails
        # Predecessor inference: use dedicated follow-up prefixes
        if predecessor and self.domain == "search":
            all_prefixes = SEARCH_BACKWARD_FUNCTION_PREFIXES
            if self.mode == "eval":
                idx = self._eval_function_prefix_counter % len(all_prefixes)
                self._eval_function_prefix_counter += 1
                return all_prefixes[idx]
            else:
                return random.choice(all_prefixes)
        
        # Math domain with taxonomy type: use taxonomy-specific predecessor prefixes
        if self.domain == "math" and taxonomy_type in MATH_BACKWARD_FUNCTION_PREFIXES:
            prefix_list = MATH_BACKWARD_FUNCTION_PREFIXES[taxonomy_type]
            if self.mode == "eval":
                idx = self._eval_function_prefix_counter % len(prefix_list)
                self._eval_function_prefix_counter += 1
                return prefix_list[idx]
            else:
                return random.choice(prefix_list)
        
        # SQL domain: flat list (no taxonomy categories)
        if self.domain == "sql":
            # New: episodic "keep-arguments, change-function" prefixes for
            # multi-clause LLM-driven SQL function changes.
            if function_change_type == "llm_multi_clause":
                all_prefixes = SQL_FUNCTION_FOLLOWUP_PREFIXES
            elif predecessor:
                # Predecessor inference for SQL: use population-scoped prefixes
                # that explicitly anchor to the row subset (the entity carved
                # out by WHERE/HAVING) instead of generic anaphora. This
                # signals "keep the same data slice, swap the question"
                # without leaking argument values or function shape.
                pop = getattr(self, "_current_pop_noun", None)
                opaque = getattr(self, "_current_pop_opaque", True)
                if pop and not opaque:
                    pool = SQL_FUNCTION_PREFIXES_POPULATION
                    if self.mode == "eval":
                        idx = self._eval_function_prefix_counter % len(pool)
                        self._eval_function_prefix_counter += 1
                        return pool[idx].format(pop=pop)
                    return random.choice(pool).format(pop=pop)
                # Fallback: generic anaphora ("within that same subset, ...")
                pool = SQL_FUNCTION_PREFIXES_GENERIC
                if self.mode == "eval":
                    idx = self._eval_function_prefix_counter % len(pool)
                    self._eval_function_prefix_counter += 1
                    return pool[idx]
                return random.choice(pool)
            else:
                all_prefixes = SQL_FUNCTION_PREFIXES
            if self.mode == "eval":
                idx = self._eval_function_prefix_counter % len(all_prefixes)
                self._eval_function_prefix_counter += 1
                return all_prefixes[idx]
            else:
                return random.choice(all_prefixes)
        
        if self.domain == "math":
            domain_prefixes = MATH_FUNCTION_PREFIXES
            categories = ["correction", "related", "redirect", "angle"]
        elif self.domain == "search":
            domain_prefixes = SEARCH_FUNCTION_PREFIXES
            categories = ["shift", "follow_up", "related", "pivot"]
        elif self.domain == "swe_bench_verified":
            # When the outgoing function carried an LLM-authored transition_phrase
            # (e.g. impl-precursor "before X, fix Y first"), use it verbatim.
            # Falls back to the fixed prefix pools when no phrase is supplied.
            if prev_transition_phrase:
                return prev_transition_phrase
            # Route by the outgoing (previous) function's nature: leaving an
            # exploration question uses "from_exploration" prefixes,
            # leaving a real-bug function uses "from_bug" prefixes.
            bucket_key = _swe_function_prefix_category(prev_taxonomy_type)
            bucket = SWE_FUNCTION_PREFIXES[bucket_key]
            if self.mode == "eval":
                idx = self._eval_function_prefix_counter % len(bucket)
                self._eval_function_prefix_counter += 1
                return bucket[idx]
            return random.choice(bucket)
        else:
            raise NotImplementedError(f"Domain '{self.domain}' not implemented")
        
        if self.mode == "eval":
            # Deterministic: cycle through all prefixes across all categories
            # Build flat list of all prefixes
            all_prefixes = []
            for category in categories:
                all_prefixes.extend(domain_prefixes[category])
            
            idx = self._eval_function_prefix_counter % len(all_prefixes)
            self._eval_function_prefix_counter += 1
            return all_prefixes[idx]
        else:
            # Train: randomly pick category, then random prefix
            category = random.choice(categories)
            return random.choice(domain_prefixes[category])

    # ── Function-naturalized helpers ─────────────────────────────────────

    def _build_naturalized_function_prefix(
        self, new_val: str, change_type: str | None,
    ) -> str | None:
        """Build a naturalized prefix from pre-computed new value."""
        if change_type == "column_swap":
            templates = SQL_NATURALIZED_COLUMN_TEMPLATES
            new_nl = new_val
        else:
            # aggregate_swap (default)
            templates = SQL_NATURALIZED_FUNCTION_TEMPLATES
            new_nl = AGG_NL_MAP.get(new_val.lower(), new_val)

        if self.mode == "eval":
            idx = self._eval_function_prefix_counter % len(templates)
            self._eval_function_prefix_counter += 1
            return templates[idx].format(new=new_nl)
        return random.choice(templates).format(new=new_nl)

    def _function_change_includes_function_text(self) -> bool:
        """Whether rendered function-switch turns should include the full function sentence."""
        return not (self._prefix_style == "function-naturalized" and self.domain == "sql")

    def _get_corr_after_reveal_prefix(self, num_corrections: int) -> str:
        """Correction-after-reveal prefix."""
        pool = (CORR_AFTER_REVEAL_SINGLE if num_corrections == 1
                else CORR_AFTER_REVEAL_MULTIPLE)
        if self.mode == "eval":
            idx = self._eval_prefix_counter_single % len(pool)
            self._eval_prefix_counter_single += 1
            return pool[idx]
        return random.choice(pool)

    def _get_reveal_after_function_prefix(self, num_reveals: int) -> str:
        """Bridge prefix when reveals follow a function change in the same turn."""
        pool = (REVEAL_AFTER_FUNCTION_SINGLE if num_reveals == 1
                else REVEAL_AFTER_FUNCTION_MULTIPLE)
        if self.mode == "eval":
            idx = self._eval_reveal_prefix_counter % len(pool)
            self._eval_reveal_prefix_counter += 1
            return pool[idx]
        return random.choice(pool)

    def _get_correction_prefix(self, num_corrections: int) -> str:
        """
        Get a correction prefix based on number of corrections and domain.
        
        Args:
            num_corrections: Number of items being corrected (1 = single, 2+ = multiple)
        
        Returns:
            Appropriate prefix string
        """
        # Get domain-specific prefixes
        if self.domain == "math":
            domain_prefixes = MATH_ARGUMENT_PREFIXES
        elif self.domain == "search":
            domain_prefixes = SEARCH_ARGUMENT_PREFIXES
        elif self.domain == "sql":
            domain_prefixes = SQL_ARGUMENT_PREFIXES
        elif self.domain == "swe_bench_verified":
            # SWE: flat list, ignore num_corrections grouping for now
            if self.mode == "eval":
                idx = self._eval_prefix_counter_single % len(SWE_CORRECTION_PREFIXES)
                self._eval_prefix_counter_single += 1
                return SWE_CORRECTION_PREFIXES[idx]
            return random.choice(SWE_CORRECTION_PREFIXES)
        else:
            raise NotImplementedError(f"Domain '{self.domain}' not implemented")

        # Get all prefix categories for this count
        suffix = "single" if num_corrections == 1 else "multiple"
        categories = [
            f"correction_{suffix}",
            f"exploration_{suffix}",
            f"revision_{suffix}",
        ]
        
        if self.mode == "eval":
            # Deterministic: cycle through correction prefixes
            prefix_list = domain_prefixes[f"correction_{suffix}"]
            if num_corrections == 1:
                idx = self._eval_prefix_counter_single % len(prefix_list)
                self._eval_prefix_counter_single += 1
            else:
                idx = self._eval_prefix_counter_multiple % len(prefix_list)
                self._eval_prefix_counter_multiple += 1
            return prefix_list[idx]
        else:
            # Train: randomly pick category, then random prefix from that category
            category = random.choice(categories)
            return random.choice(domain_prefixes[category])
    
    def _get_new_info_prefix(self) -> str:
        """
        Get a new-info prefix for non-correction items that follow corrections
        in the same turn. Disambiguates new info from the correction block.
        """
        if self.domain == "math":
            domain_prefixes = MATH_ARGUMENT_PREFIXES
        elif self.domain == "search":
            domain_prefixes = SEARCH_ARGUMENT_PREFIXES
        elif self.domain == "sql":
            domain_prefixes = SQL_ARGUMENT_PREFIXES
        elif self.domain == "swe_bench_verified":
            if self.mode == "eval":
                idx = self._eval_new_info_prefix_counter % len(SWE_NEW_INFO_PREFIXES)
                self._eval_new_info_prefix_counter += 1
                return SWE_NEW_INFO_PREFIXES[idx]
            return random.choice(SWE_NEW_INFO_PREFIXES)
        else:
            raise NotImplementedError(f"Domain '{self.domain}' not implemented")
        
        prefix_list = domain_prefixes["new_info"]
        
        if self.mode == "eval":
            idx = self._eval_new_info_prefix_counter % len(prefix_list)
            self._eval_new_info_prefix_counter += 1
            return prefix_list[idx]
        else:
            return random.choice(prefix_list)
    
    def _get_reveal_prefix(self, num_reveals: int) -> str:
        """
        Get a argument reveal prefix for under-specified scenario.
        Used when revealing previously withheld arguments on turns 2+.
        
        Args:
            num_reveals: Number of arguments being revealed (1 = single, 2+ = multiple)
        
        Returns:
            Appropriate reveal prefix string
        """
        if self.domain == "search":
            domain_prefixes = SEARCH_ARGUMENT_PREFIXES
        elif self.domain == "math":
            domain_prefixes = MATH_ARGUMENT_PREFIXES
        elif self.domain == "sql":
            domain_prefixes = SQL_ARGUMENT_PREFIXES
        elif self.domain == "swe_bench_verified":
            if self.mode == "eval":
                idx = self._eval_reveal_prefix_counter % len(SWE_REVEAL_PREFIXES)
                self._eval_reveal_prefix_counter += 1
                return SWE_REVEAL_PREFIXES[idx]
            return random.choice(SWE_REVEAL_PREFIXES)
        else:
            raise NotImplementedError(f"Domain '{self.domain}' not implemented")
        
        suffix = "single" if num_reveals == 1 else "multiple"
        prefix_list = domain_prefixes[f"reveal_{suffix}"]
        
        if self.mode == "eval":
            idx = self._eval_reveal_prefix_counter % len(prefix_list)
            self._eval_reveal_prefix_counter += 1
            return prefix_list[idx]
        else:
            return random.choice(prefix_list)
    
    # =========================================================================
    # DataLoader-like interface
    # =========================================================================
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._samples)
    
    def __getitem__(self, idx: int | slice) -> IntentSample | list[IntentSample]:
        """Get sample(s) by index."""
        if isinstance(idx, slice):
            return self._samples[idx]
        return self._samples[idx]
    
    def __iter__(self) -> Iterator[IntentSample]:
        """Iterate over samples."""
        return iter(self._samples)
    
    def __repr__(self) -> str:
        return (
            f"EvolvingIntent("
            f"samples={len(self)}, "
            f"mode='{self.mode}', "
            f"scenario='{self.scenario}', "
            f"num_turns={self.num_turns}, "
            f"num_revisions={self.num_revisions})"
        )
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the simulation."""
        return {
            "total_samples": len(self),
            "mode": self.mode,
            "scenario": self.scenario,
            "num_turns": self.num_turns,
            "num_revisions": self.num_revisions,
            "data_path": str(self.data_path),
        }
    
    def to_list(self) -> list[dict[str, Any]]:
        """Convert all samples to list of dicts (for JSON export)."""
        return [
            {
                "turns": s.turns,
                "label": s.label,
                "metadata": s.metadata,
            }
            for s in self._samples
        ]
    
    def save(self, output_path: str | Path, format: str = "jsonl"):
        """Save samples to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_list()
        
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(output_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(data)} samples to {output_path}")


# =============================================================================
# Convenience function
# =============================================================================

def create_simulation(
    data_path: str | Path,
    mode: str = "eval",
    domain: str = "math",
    num_turns: int = 4,
    num_revisions: int = 1,
    num_switches: int = 0,
    system_prompt: str | None = None,
    instruction: str | None = None,
    seed: int = 42,
) -> EvolvingIntent:
    """
    Convenience function to create an EvolvingIntent user simulation.

    Example:
        >>> sim = create_simulation("data.json", num_turns=4, num_revisions=2)
        >>> for sample in sim:
        ...     print(sample)
    """
    return EvolvingIntent(
        data_path=data_path,
        mode=mode,
        domain=domain,
        num_turns=num_turns,
        num_revisions=num_revisions,
        num_switches=num_switches,
        system_prompt=system_prompt,
        instruction=instruction,
        seed=seed,
    )


# =============================================================================
# CLI (optional, for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EvolvingIntent")
    parser.add_argument("--input", "-i", required=True, help="Input JSON path")
    parser.add_argument("--num_turns", type=int, default=4)
    parser.add_argument("--num_revisions", type=int, default=1)
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--show", type=int, default=1, help="Number of samples to show")
    parser.add_argument("--instruction", type=str, default=None, help="Optional instruction (use {content} placeholder)")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    
    args = parser.parse_args()
    
    sim = EvolvingIntent(
        data_path=args.input,
        mode=args.mode,
        num_turns=args.num_turns,
        num_revisions=args.num_revisions,
        system_prompt=args.system_prompt,
        instruction=args.instruction,
    )
    
    print(sim)
    print(f"\nStats: {sim.get_stats()}")
    
    print(f"\n{'='*60}")
    print(f"Showing first {args.show} sample(s):")
    print('='*60)
    
    for i, sample in enumerate(sim):
        if i >= args.show:
            break
        print(f"\n--- Sample {i} ---")
        print(f"Task ID: {sample.task_id}")
        print(f"Label: {sample.label}")
        print(f"Num turns: {sample.num_turns}")
        print(f"\nTurns:")
        for j, turn in enumerate(sample.turns):
            print(f"  [{turn['role']}]: {turn['content'][:80]}...")
