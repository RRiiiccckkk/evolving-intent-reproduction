"""
Plan-first turn scheduler for EvolvingIntent.

Instead of building a flat content_items list and then greedily assigning
items to turns (which fails for complex constraint combinations), this module
takes a "plan-first" approach:

  Step 1: Schedule events — decide which turn gets which event (function change,
          correction) purely by type/ID, before generating any text.
  Step 2: Fill arguments — distribute argument reveals across turns.
  Step 3: Order within turn — function first, then correction, then reveals.
  Step 4: Render — add prefixes and join into final turn strings.

Key property: scheduling NEVER fails. If the raw data has enough counterfactual
functions/arguments and t >= 1 + g + p, every sample produces a valid output.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TurnEvent:
    """A single event scheduled for a turn."""
    type: str               # "function_init", "function_change", "correction"
    function_idx: int | None = None     # index into selected_functions (-1 for source)
    cond_id: int | None = None      # for correction: which argument
    corr_step: int | None = None    # for multi-step correction: 0-based step index
    corr_text: str | None = None    # for correction: the corrected text


@dataclass
class ArgumentItem:
    """A argument to show in a turn."""
    cond_id: int
    text: str               # the text to show (counterfactual or source)
    is_counterfactual: bool = False


@dataclass
class TurnSlot:
    """Blueprint for a single turn."""
    turn_idx: int
    events: list[TurnEvent] = field(default_factory=list)
    arguments: list[ArgumentItem] = field(default_factory=list)
    function_text: str | None = None    # filled in step 3


# ============================================================================
# Step 0: Select functions and argument counterfactuals from raw data
# ============================================================================

def select_functions(
    raw: dict[str, Any],
    g: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Select and reverse counterfactual functions (farthest-first for inverse change).

    Returns list of length `actual_g` (may be < g if data is insufficient).
    """
    predecessor_functions = raw.get("predecessor_functions", [])
    if g == 0 or not predecessor_functions:
        return []

    if mode == "eval":
        actual_g = min(g, len(predecessor_functions))
    else:
        max_g = min(g, len(predecessor_functions))
        actual_g = random.randint(1, max_g) if max_g > 0 else 0

    # Stored nearest-first → reverse to farthest-first
    return list(reversed(predecessor_functions[:actual_g]))


def select_counterfactuals(
    raw: dict[str, Any],
    selected_functions: list[dict[str, Any]],
    p: int,
    mode: str,
) -> dict[int, list[dict]]:
    """Select argument counterfactuals. Returns {cond_id: [counterfactual_dicts]}.

    Prefers arguments shared across functions; falls back to any counterfactual_eligible
    argument.
    """
    source_arguments = sorted(
        raw.get("arguments", []), key=lambda x: x.get("argument_id", 0)
    )
    if p == 0 or not source_arguments:
        return {}

    # Identify shared argument IDs across selected functions
    shared_cond_ids: set[int] = set()
    for pg in selected_functions:
        for cond in pg.get("counterfactual_arguments", []):
            if cond.get("is_shared", False):
                shared_cond_ids.add(cond.get("argument_id"))

    cond_by_id = {c["argument_id"]: c for c in source_arguments}

    # Prefer shared counterfactual_eligible arguments, but include others if not enough
    counterfactual_eligible_shared = [
        cond_by_id[cid] for cid in sorted(shared_cond_ids)
        if cid in cond_by_id and cond_by_id[cid].get("counterfactual_arguments")
    ]
    all_counterfactual_eligible = [
        c for c in source_arguments if c.get("counterfactual_arguments")
    ]
    if not counterfactual_eligible_shared:
        counterfactual_eligible_pool = all_counterfactual_eligible
    else:
        # Always merge shared (first) + non-shared; round-robin prefers shared
        seen = {c["argument_id"] for c in counterfactual_eligible_shared}
        counterfactual_eligible_pool = list(counterfactual_eligible_shared)
        for c in all_counterfactual_eligible:
            if c["argument_id"] not in seen:
                counterfactual_eligible_pool.append(c)
                seen.add(c["argument_id"])
    if not counterfactual_eligible_pool:
        return {}

    total_avail = sum(
        len(c.get("counterfactual_arguments", [])) for c in counterfactual_eligible_pool
    )

    if mode == "eval":
        actual_p = min(p, total_avail)
        result: dict[int, list] = {c["argument_id"]: [] for c in counterfactual_eligible_pool}
        pidx = {c["argument_id"]: 0 for c in counterfactual_eligible_pool}
        rem = actual_p
        while rem > 0:
            progress = False
            for cond in counterfactual_eligible_pool:
                if rem <= 0:
                    break
                cid = cond["argument_id"]
                plist = cond.get("counterfactual_arguments", [])
                if pidx[cid] < len(plist):
                    result[cid].append(plist[pidx[cid]])
                    pidx[cid] += 1
                    rem -= 1
                    progress = True
            if not progress:
                break
        return {k: v for k, v in result.items() if v}
    else:
        max_p = min(p, total_avail)
        actual_p = random.randint(1, max_p) if max_p > 0 else 0
        if actual_p == 0:
            return {}
        available = [
            (c["argument_id"], per)
            for c in counterfactual_eligible_pool
            for per in c.get("counterfactual_arguments", [])
        ]
        random.shuffle(available)
        result = {}
        for cid, counterfactual in available[:actual_p]:
            result.setdefault(cid, []).append(counterfactual)
        return result


# ============================================================================
# Step 1: Schedule Events
# ============================================================================

def _compute_shared_cond_ids(selected_functions: list[dict[str, Any]]) -> set[int]:
    """Return the set of argument IDs marked ``is_shared`` across *selected_functions*."""
    shared: set[int] = set()
    for pg in selected_functions:
        for cond in pg.get("counterfactual_arguments", []):
            if cond.get("is_shared", False):
                shared.add(cond.get("argument_id"))
    return shared


def _build_deadline_map(
    selected_functions: list[dict[str, Any]],
    counterfactual_cond_ids: set[int],
    source_arguments: list[dict[str, Any]],
    shared_cond_ids: set[int] | None = None,
) -> dict[int, int]:
    """For each counterfactual cid, find the earliest function index (1-based) that
    needs it in source form.

    Function indices: 1..g correspond to selected_functions[1..g-1] and g+1 for
    source_function.  Actually we use 1-based indexing into the event sequence:
    function_idx=1 means the first function *after* PG0, which is selected_functions[1].
    function_idx=g means source_function.

    Wait -- let me clarify. all_functions = selected_functions + [source_function].
    all_functions[0] = PG0 (farthest, the initial function).
    all_functions[1] = PG1
    ...
    all_functions[g] = source_function

    For the event list, we iterate function_idx from 1 to g (inclusive),
    corresponding to all_functions[1] through all_functions[g].

    The "deadline" for a counterfactual cid is the first function_idx (1..g) whose
    shared_cids include this cid.
    """
    g = len(selected_functions)
    # all_functions[0..g-1] = selected_functions, all_functions[g] = source_function
    # We iterate function_idx = 1..g

    all_source_cids = {c["argument_id"] for c in source_arguments}

    deadlines: dict[int, int] = {}
    for cid in counterfactual_cond_ids:
        # Non-shared cids: corrections can go after source function
        if shared_cond_ids is not None and cid not in shared_cond_ids:
            deadlines[cid] = g + 1  # post-source phase
            continue

        found = False
        for gi in range(1, g):  # selected_functions[1..g-1]
            pg = selected_functions[gi]
            pg_shared = {
                c.get("argument_id")
                for c in pg.get("counterfactual_arguments", [])
                if c.get("is_shared", False)
            }
            if cid in pg_shared:
                deadlines[cid] = gi
                found = True
                break
        if not found:
            # Deadline is source_function (index g)
            deadlines[cid] = g
    return deadlines


def schedule_events(
    g: int,
    p: int,
    t: int,
    correction_specs: list[tuple[int, int]],
    deadlines: dict[int, int],
) -> list[TurnSlot]:
    """Create the turn blueprint: assign events to turns.

    Args:
        g: number of function changes
        p: total number of correction events (may exceed unique cids if
           a single argument has multiple counterfactual variants)
        t: total turns
        correction_specs: ordered list of (cond_id, step_index) for each
            correction event.  Steps for the same cid must be consecutive
            and ascending. E.g. [(1,0),(1,1),(2,0)] = 2 corrections on C1
            then 1 on C2.
        deadlines: {cid: function_idx} from _build_deadline_map

    Returns a list of `t` TurnSlot objects. Turn 0 has a "function_init" event.
    Turns 1..t-1 get function_change and correction events, plus reveal-only slots.
    """
    slots = [TurnSlot(turn_idx=i) for i in range(t)]

    # Turn 0 is always the initial function
    slots[0].events.append(TurnEvent(type="function_init", function_idx=0))

    if g == 0 and p == 0:
        # Fully-specified or under-specified: no events beyond turn 0
        return slots

    # ── Build interleaved event list ──────────────────────────────
    event_list: list[TurnEvent] = []

    corrections = [
        TurnEvent(type="correction", cond_id=cid, corr_step=step)
        for cid, step in correction_specs
    ]

    if g == 0:
        # Argument-change only: corrections go sequentially
        event_list.extend(corrections)

    elif p == 0:
        # Function-change only: function changes go sequentially
        for gi in range(1, g + 1):
            function_idx = gi if gi < g else -1
            event_list.append(TurnEvent(type="function_change", function_idx=function_idx))

    else:
        # Combined: interleave corrections across g function phases.
        # Deadline for a cid block = deadline of the cid (applies to last step).
        # All steps for the same cid go to the same phase to stay contiguous.
        # Post-source phase (bucket g) holds non-shared corrections.
        phase_buckets: list[list[TurnEvent]] = [[] for _ in range(g + 1)]

        corr_with_dl = [
            (deadlines.get(ev.cond_id, g), ev) for ev in corrections
        ]
        corr_with_dl.sort(key=lambda x: (x[0], x[1].cond_id, x[1].corr_step))

        for dl, ev in corr_with_dl:
            bucket_idx = min(dl - 1, g)  # deadline g+1 → bucket g (post-source)
            phase_buckets[bucket_idx].append(ev)

        # Pre-source phases + function changes
        for gi in range(g):
            for ev in phase_buckets[gi]:
                event_list.append(ev)
            function_idx = gi + 1 if gi + 1 < g else -1
            event_list.append(TurnEvent(type="function_change", function_idx=function_idx))

        # Post-source corrections (non-shared)
        for ev in phase_buckets[g]:
            event_list.append(ev)

    assert len(event_list) == g + p, (
        f"Expected {g+p} events, got {len(event_list)}"
    )

    # ── Assign events to turns ────────────────────────────────────
    # Spread events evenly across turns 1..t-1 so that empty turns
    # (for argument reveals) appear BETWEEN events, not just at the end.
    # When post-source corrections exist, spread all events across 1..t-1.
    # Otherwise, pin the final function_change to the last turn (t-1).
    num_events = g + p
    num_post_source = len(phase_buckets[g]) if (g > 0 and p > 0) else 0

    if num_post_source > 0:
        # Post-source corrections exist → spread all events across 1..t-1
        for i, ev in enumerate(event_list):
            pos = 1 + (i * (t - 2)) // max(len(event_list) - 1, 1)
            slots[pos].events.append(ev)
    elif event_list and event_list[-1].type == "function_change" and event_list[-1].function_idx == -1:
        # Pin final function_change to last turn; spread the rest in 1..t-2
        rest = event_list[:-1]
        usable = t - 2  # turns 1..t-2
        for i, ev in enumerate(rest):
            pos = 1 + (i * usable) // max(len(rest), 1)
            slots[pos].events.append(ev)
        slots[t - 1].events.append(event_list[-1])
    else:
        for i, ev in enumerate(event_list):
            pos = 1 + (i * (t - 1)) // num_events
            slots[pos].events.append(ev)

    return slots


# ============================================================================
# Step 2: Fill Arguments
# ============================================================================

def fill_arguments(
    slots: list[TurnSlot],
    selected_functions: list[dict[str, Any]],
    source_arguments: list[dict[str, Any]],
    counterfactual_cond_ids: set[int],
    counterfactual_map: dict[int, list[dict]],
    cond_by_id: dict[int, dict],
) -> None:
    """Distribute arguments across turn slots (mutates slots in place).

    Any argument can be deferred from its natural turn to an empty turn,
    as long as counterfactual arguments appear BEFORE their correction turn.
    Keep at least 1 argument in Turn 0.
    """
    g = len(selected_functions)
    t = len(slots)
    revealed_cids: set[int] = set()

    # ── Build correction deadline map ─────────────────────────────
    # deadline[cid] = turn index of FIRST correction (counterfactual cid must appear before)
    corr_deadline: dict[int, int] = {}
    for s in slots:
        for ev in s.events:
            if ev.type == "correction" and ev.cond_id is not None:
                if ev.cond_id not in corr_deadline:
                    corr_deadline[ev.cond_id] = s.turn_idx

    # ── Find the final function_change turn (source-function restoration) ───
    # Arguments must not be deferred past this point: a never-before-seen
    # argument appearing after the final function change would alter the
    # expected answer without the model ever having seen it earlier.
    final_function_turn = t  # fallback: no constraint
    for s in slots:
        for ev in s.events:
            if ev.type == "function_change" and ev.function_idx == -1:
                final_function_turn = s.turn_idx

    # ── Available empty turn indices (sorted) ─────────────────────
    available: list[int] = sorted(
        s.turn_idx for s in slots
        if not s.events and s.turn_idx > 0 and s.turn_idx < final_function_turn
    )
    # Post-source empty turns (between source-function and end)
    available_post_source: list[int] = sorted(
        s.turn_idx for s in slots
        if not s.events and s.turn_idx > final_function_turn
    )

    # ── Gather Turn 0's arguments with deadlines ─────────────────
    # Each item: (ArgumentItem, deadline)
    # deadline = correction turn for counterfactual, t for non-counterfactual
    turn0_pool: list[tuple[ArgumentItem, int]] = []

    if g > 0:
        pg0 = selected_functions[0]
        pg0_cids: set[int] = set()
        for cond in pg0.get("counterfactual_arguments", []):
            cid = cond.get("argument_id")
            if cid in pg0_cids:
                continue
            pg0_cids.add(cid)

            if cid in counterfactual_cond_ids:
                first_p = counterfactual_map[cid][0]
                item = ArgumentItem(
                    cid, first_p.get("counterfactual_argument", ""), True)
                turn0_pool.append((item, corr_deadline.get(cid, t)))
            else:
                text = (cond_by_id[cid]["argument"]
                        if cid in cond_by_id
                        else cond.get("argument", ""))
                turn0_pool.append((ArgumentItem(cid, text, False), t))

        # Plant counterfactual cids not in PG0
        for cid in sorted(counterfactual_cond_ids):
            if cid not in pg0_cids and cid in cond_by_id:
                first_p = counterfactual_map[cid][0]
                item = ArgumentItem(
                    cid, first_p.get("counterfactual_argument", ""), True)
                turn0_pool.append((item, corr_deadline.get(cid, t)))
                pg0_cids.add(cid)
    else:
        # No function change: arguments come from source_arguments
        for cond in source_arguments:
            cid = cond["argument_id"]
            if cid in counterfactual_cond_ids:
                first_p = counterfactual_map[cid][0]
                item = ArgumentItem(
                    cid, first_p.get("counterfactual_argument", ""), True)
                turn0_pool.append((item, corr_deadline.get(cid, t)))
            else:
                turn0_pool.append(
                    (ArgumentItem(cid, cond["argument"], False), t))

    # ── Defer Turn 0 arguments to empty turns ────────────────────
    # Sort by deadline ascending: tightest deadline first (hardest to defer)
    turn0_pool.sort(key=lambda x: x[1])

    keep: list[ArgumentItem] = []
    can_defer: list[tuple[ArgumentItem, int]] = []

    for item, deadline in turn0_pool:
        # Can this go in any available empty turn before its deadline?
        if any(ti < deadline for ti in available):
            can_defer.append((item, deadline))
        else:
            keep.append(item)

    # At least 1 argument must stay in Turn 0
    if not keep and can_defer:
        keep.append(can_defer.pop(0)[0])  # tightest deadline stays

    # Place kept items in Turn 0
    for item in keep:
        slots[0].arguments.append(item)
        revealed_cids.add(item.cond_id)

    # Place deferred items in earliest available empty turn before deadline
    for item, deadline in can_defer:
        placed = False
        for i, ti in enumerate(available):
            if ti < deadline:
                slots[ti].arguments.append(item)
                revealed_cids.add(item.cond_id)
                available.pop(i)
                placed = True
                break
        if not placed:
            slots[0].arguments.append(item)
            revealed_cids.add(item.cond_id)

    # ── Function-change turns ─────────────────────────────────────────
    for slot in slots:
        for ev in slot.events:
            if ev.type != "function_change":
                continue

            new_conds: list[ArgumentItem] = []
            if ev.function_idx == -1:
                for cond in source_arguments:
                    cid = cond["argument_id"]
                    if cid not in revealed_cids:
                        new_conds.append(ArgumentItem(
                            cid, cond["argument"], False))
                        revealed_cids.add(cid)
            else:
                pg = selected_functions[ev.function_idx]
                for cond in pg.get("counterfactual_arguments", []):
                    cid = cond.get("argument_id")
                    if not cond.get("is_shared", False) and cid not in revealed_cids:
                        text = (cond_by_id[cid]["argument"]
                                if cid in cond_by_id
                                else cond.get("argument", ""))
                        new_conds.append(ArgumentItem(cid, text, False))
                        revealed_cids.add(cid)

            # Defer some to remaining empty turns
            if available and len(new_conds) > 1:
                defer_n = min(len(new_conds) - 1, len(available))
                for ci in new_conds[-defer_n:]:
                    ti = available.pop(0)
                    slots[ti].arguments.append(ci)
                new_conds = new_conds[:-defer_n]

            slot.arguments.extend(new_conds)

    # ── Distribute remaining unrevealed arguments ────────────────
    remaining: list[ArgumentItem] = []
    for cond in source_arguments:
        cid = cond["argument_id"]
        if cid not in revealed_cids:
            remaining.append(ArgumentItem(cid, cond["argument"], False))
            revealed_cids.add(cid)

    if remaining:
        all_avail = available + available_post_source
        if all_avail:
            for ci in remaining:
                if all_avail:
                    slots[all_avail.pop(0)].arguments.append(ci)
                else:
                    slots[0].arguments.append(ci)
        else:
            target = [s for s in slots[1:] if (s.events or s.arguments)
                      and s.turn_idx <= final_function_turn]
            if not target:
                target = [slots[0]]
            for i, ci in enumerate(remaining):
                target[i % len(target)].arguments.append(ci)


# ============================================================================
# Step 3: Fill Text Content (function text, correction text)
# ============================================================================

def fill_texts(
    slots: list[TurnSlot],
    selected_functions: list[dict[str, Any]],
    source_function: str,
    counterfactual_map: dict[int, list[dict]],
    cond_by_id: dict[int, dict],
) -> None:
    """Fill function_text and correction corr_text in each slot (mutates in place).

    For multi-step corrections on the same cid with N variants [v1, v2, ..., vN]:
      correction chain = [v2, v3, ..., vN, source]
      step 0 → chain[0] = v2   (still wrong)
      step 1 → chain[1] = v3   (still wrong)
      ...
      step N-1 → chain[N-1] = source  (final correction)

    For single-variant (N=1): chain = [source], step 0 → source.
    """
    for slot in slots:
        for ev in slot.events:
            if ev.type == "function_init":
                if selected_functions:
                    slot.function_text = selected_functions[0].get("predecessor_function", "")
                else:
                    slot.function_text = source_function
            elif ev.type == "function_change":
                if ev.function_idx == -1:
                    slot.function_text = source_function
                else:
                    slot.function_text = selected_functions[ev.function_idx].get(
                        "predecessor_function", ""
                    )
            elif ev.type == "correction":
                cid = ev.cond_id
                step = ev.corr_step or 0
                # Correction chain: intermediate counterfactual values + source
                counterfactuals = counterfactual_map.get(cid, [])
                chain = [p.get("counterfactual_argument", "") for p in counterfactuals[1:]]
                chain.append(cond_by_id[cid]["argument"])
                # Use step to index into chain
                ev.corr_text = chain[min(step, len(chain) - 1)]


# ============================================================================
# Step 4: Render Turns → list[str]
# ============================================================================

def render_turns(
    slots: list[TurnSlot],
    selected_functions: list[dict[str, Any]],
    is_predecessor: bool,
    get_function_change_prefix,
    get_correction_prefix,
    get_reveal_prefix,
    get_reveal_after_function_prefix,
    get_corr_after_reveal_prefix,
    get_new_info_prefix,
    join_prefix_content,
    function_change_includes_function_text: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Render each turn slot into a final text string.

    Within each turn the order is:
      1. Function (initial or changed) — always first
      2. Correction — after function
      3. Arguments/reveals — after everything else

    Prefix rules:
      - Turn 0 function: no prefix
      - Function change: get_function_change_prefix(...)
      - Correction: get_correction_prefix(1)
      - Reveal after function in same turn: get_reveal_after_function_prefix(n)
      - Reveal after correction in same turn: get_new_info_prefix()
      - Reveal (standalone turn): get_reveal_prefix(n)

    Returns:
      (turn_texts, turn_meta) — turn_meta has per-turn dicts with keys:
        prefix (str), turn_type (str), taxonomy_type (str|None)
    """
    turn_texts = []
    turn_meta: list[dict[str, Any]] = []

    # Helper: extract aggregate/column keyword from a predecessor_function's
    # transition_reason.  Returns (keyword, change_type) or (None, None).
    _re_agg = re.compile(r"Changed aggregate from (\w+) to (\w+)", re.IGNORECASE)
    _re_col = re.compile(r"Changed target column from (.+) to (.+)", re.IGNORECASE)

    def _extract_function_keyword(pg: dict[str, Any]) -> tuple[str | None, str | None]:
        """Return (keyword_this_function_uses, change_type) from a predecessor_function dict."""
        # New SQL multi-clause LLM rewrite: route via dedicated change_type so
        # downstream prefix routing can pick the "follow-up" prefix list.
        if pg.get("transition_type") == "llm_multi_clause":
            return None, "llm_multi_clause"
        reason = pg.get("transition_reason", "")
        m = _re_agg.search(reason)
        if m:
            return m.group(2).lower(), "aggregate_swap"  # "to" part
        m = _re_col.search(reason)
        if m:
            return m.group(2).strip(), "column_swap"
        return None, None

    def _extract_source_keyword(pg: dict[str, Any]) -> tuple[str | None, str | None]:
        """Return (source_function_keyword, change_type) from a predecessor_function dict."""
        if pg.get("transition_type") == "llm_multi_clause":
            return None, "llm_multi_clause"
        reason = pg.get("transition_reason", "")
        m = _re_agg.search(reason)
        if m:
            return m.group(1).lower(), "aggregate_swap"  # "from" part = source
        m = _re_col.search(reason)
        if m:
            return m.group(1).strip(), "column_swap"
        return None, None

    # Track the *outgoing* (previous-active) function's taxonomy_type as we
    # walk slots in order, so prefix selectors (currently only the SWE
    # domain uses it) can route based on what the user was just talking
    # about, not what they're transitioning to. Initialised to selected_functions[0]
    # because that's the function active at slot 0 (function_init).
    prev_taxonomy_type: str | None = (
        selected_functions[0].get("taxonomy_type") if selected_functions else None
    )
    # Track the outgoing function's transition_phrase too. SWE impl-precursor
    # entries store an LLM-authored "before X, fix Y first" phrase that we
    # plumb into get_function_change_prefix verbatim instead of routing through
    # the fixed prefix pools. Initialised to selected_functions[0]'s phrase
    # because that's the function active at slot 0.
    prev_transition_phrase: str | None = (
        selected_functions[0].get("transition_phrase") if selected_functions else None
    )

    for slot in slots:
        parts = []
        has_function = False
        has_correction = False
        num_reveals = len(slot.arguments)
        # Track the primary prefix used for this turn
        primary_prefix = ""
        turn_type = "reveal"  # default
        taxonomy_type = None

        # 1. Function text
        if slot.function_text is not None:
            has_function = True
            function_events = [e for e in slot.events
                           if e.type in ("function_init", "function_change")]
            if function_events and function_events[0].type == "function_init":
                # First turn: no prefix
                turn_type = "initial"
                parts.append(slot.function_text)
            else:
                # Function change: add prefix
                turn_type = "function_change"
                ge = function_events[0] if function_events else None

                # Determine current (incoming) function's keyword
                if ge and ge.function_idx is not None and ge.function_idx >= 0:
                    pg = selected_functions[ge.function_idx]
                    taxonomy_type = pg.get("taxonomy_type")
                    cur_keyword, cur_change_type = _extract_function_keyword(pg)
                elif ge and ge.function_idx == -1 and selected_functions:
                    # Source function restoration
                    taxonomy_type = selected_functions[0].get("taxonomy_type")
                    cur_keyword, cur_change_type = _extract_source_keyword(selected_functions[0])
                else:
                    taxonomy_type = None
                    cur_keyword, cur_change_type = None, None

                pfx = get_function_change_prefix(
                    predecessor=is_predecessor,
                    taxonomy_type=taxonomy_type,
                    function_change_new=cur_keyword,
                    function_change_type=cur_change_type,
                    prev_taxonomy_type=prev_taxonomy_type,
                    prev_transition_phrase=prev_transition_phrase,
                )
                primary_prefix = pfx
                if function_change_includes_function_text:
                    parts.append(f"{pfx} {slot.function_text}")
                else:
                    parts.append(pfx)

                # After this function change, the new "outgoing" for the next
                # transition is the function we just transitioned TO.
                prev_taxonomy_type = taxonomy_type
                # Also update the carried transition_phrase from the new
                # outgoing function. Falls back to None when the new function has
                # no phrase (e.g. source restoration or non-impl-precursor).
                if ge and ge.function_idx is not None and ge.function_idx >= 0:
                    prev_transition_phrase = (
                        selected_functions[ge.function_idx].get("transition_phrase")
                    )
                else:
                    prev_transition_phrase = None

        # 2. Correction text
        corr_events = [e for e in slot.events if e.type == "correction"]
        if corr_events:
            has_correction = True
            if not has_function:
                turn_type = "correction"
            else:
                turn_type = "mixed"
            for ce in corr_events:
                pfx = get_correction_prefix(num_corrections=1)
                if not primary_prefix:
                    primary_prefix = pfx
                parts.append(join_prefix_content(pfx, ce.corr_text or ""))

        # 3. Arguments/reveals
        if slot.arguments:
            cond_texts = [ci.text for ci in slot.arguments]
            joined_conds = " ".join(cond_texts)

            if has_function and has_correction:
                # Reveals come after correction which came after function
                pfx = get_new_info_prefix()
                parts.append(join_prefix_content(pfx, joined_conds))
            elif has_function:
                if slot.turn_idx == 0:
                    # Turn 0: arguments are part of the initial message
                    parts.append(joined_conds)
                else:
                    # Reveals in same turn as function change
                    pfx = get_reveal_after_function_prefix(num_reveals=num_reveals)
                    parts.append(join_prefix_content(pfx, joined_conds))
            elif has_correction:
                # Reveals in same turn as correction
                pfx = get_new_info_prefix()
                parts.append(join_prefix_content(pfx, joined_conds))
            else:
                # Pure reveal turn
                pfx = get_reveal_prefix(num_reveals=num_reveals)
                if not primary_prefix:
                    primary_prefix = pfx
                parts.append(join_prefix_content(pfx, joined_conds))

        turn_texts.append(" ".join(parts))
        turn_meta.append({
            "prefix": primary_prefix,
            "turn_type": turn_type,
            "taxonomy_type": taxonomy_type,
        })

    return turn_texts, turn_meta


# ============================================================================
# Change Plan Generation
# ============================================================================

def _compute_sql_turn_answer(
    raw: dict[str, Any],
    revealed_cond_ids: frozenset[int],
    total_arguments: int,
) -> str:
    """Compute SQL answer for a partial set of revealed arguments."""
    try:
        try:
            from .sql_partial import compute_turn_answer
        except ImportError:
            from situated_simulation.sql_partial import compute_turn_answer
        return compute_turn_answer(
            gold_sql=raw.get("gold_sql", ""),
            db_path=raw.get("db_path", ""),
            revealed_cond_ids=revealed_cond_ids,
            total_arguments=total_arguments,
        )
    except Exception:
        return ""


def _compute_per_turn_gold(
    raw: dict[str, Any],
    change_plan: Any,
    selected_functions: list[dict[str, Any]],
    counterfactual_map: dict[int, list[dict]],
    ground_truth: str,
) -> list[dict[str, str]]:
    """Compute per-turn gold SQL+answer for SQL samples.

    Returns ``[]`` for non-SQL samples or on failure.
    """
    if raw.get("data_source") != "bird_sql":
        return []
    try:
        try:
            from .sql_partial import build_per_turn_gold
        except ImportError:
            from situated_simulation.sql_partial import build_per_turn_gold

        # Serialise UserIntent objects to dicts that build_per_turn_gold expects.
        trajectory = []
        for intent in change_plan.intent_trajectory:
            if isinstance(intent, dict):
                trajectory.append(intent)
            else:
                trajectory.append({
                    "function": intent.function,
                    "revealed_ids": sorted(intent.revealed_ids),
                    "active_values": {
                        str(aid): text for aid, text in intent.active_values
                    } if intent.active_values else None,
                })
        return build_per_turn_gold(
            raw, trajectory, selected_functions, counterfactual_map, ground_truth,
        )
    except Exception:
        return []


def build_change_plan(
    slots: list[TurnSlot],
    raw: dict[str, Any],
    selected_functions: list[dict[str, Any]],
    source_function: str,
    source_arguments: list[dict[str, Any]],
    counterfactual_cond_ids: set[int],
    counterfactual_map: dict[int, list[dict]],
    ground_truth: str,
    domain: str,
    scenario: str,
) -> "ChangePlan":
    """Build a ChangePlan directly from the scheduled turn slots."""
    try:
        from .user_intent import (
            UserIntent, IntentTransition, ChangePlan,
            arguments_from_raw,
        )
    except ImportError:
        from situated_simulation.user_intent import (
            UserIntent, IntentTransition, ChangePlan,
            arguments_from_raw,
        )

    task_id = raw.get("task_id", "")
    args = arguments_from_raw(source_arguments)

    trajectory: list[UserIntent] = []
    transitions: list[IntentTransition] = []

    # Track state across turns
    current_fn = source_function
    revealed_ids: set[int] = set()
    active_vals: dict[int, str] = {}  # cid -> currently active text

    # Initialize counterfactual values
    cond_by_id = {c["argument_id"]: c for c in source_arguments}
    for cid in counterfactual_cond_ids:
        first_p = counterfactual_map[cid][0]
        active_vals[cid] = first_p.get("counterfactual_argument", "")

    g = len(selected_functions)

    for slot in slots:
        prev_fn = current_fn
        prev_revealed = set(revealed_ids)
        prev_active = dict(active_vals)

        # Process events
        for ev in slot.events:
            if ev.type == "function_init":
                if g > 0:
                    current_fn = selected_functions[0].get("predecessor_function", "")
                else:
                    current_fn = source_function
            elif ev.type == "function_change":
                if ev.function_idx == -1:
                    current_fn = source_function
                else:
                    current_fn = selected_functions[ev.function_idx].get(
                        "predecessor_function", ""
                    )
            elif ev.type == "correction":
                cid = ev.cond_id
                # Use the actual correction text (may be intermediate for
                # multi-step chains, only the final step is source)
                if ev.corr_text is not None:
                    active_vals[cid] = ev.corr_text
                else:
                    active_vals[cid] = cond_by_id[cid]["argument"]

        # Process reveals
        for ci in slot.arguments:
            revealed_ids.add(ci.cond_id)

        # Always use source arguments as the argument set — they represent
        # the full universe of arguments.  active_values tracks which
        # values are currently counterfactual.
        turn_args = args

        # Build active_values tuple (only non-source values)
        av_tuple = tuple(
            (cid, text) for cid, text in sorted(active_vals.items())
            if text != cond_by_id.get(cid, {}).get("argument", "")
        )

        # Target answer
        is_final = (slot.turn_idx == len(slots) - 1)
        if is_final:
            target_answer = ground_truth
        elif scenario == "under-specified" and raw.get("data_source") == "bird_sql":
            target_answer = _compute_sql_turn_answer(
                raw, frozenset(revealed_ids), len(source_arguments),
            )
        elif scenario == "under-specified":
            target_answer = ground_truth  # y_t constant for IF/Math
        else:
            target_answer = ""

        intent = UserIntent(
            function=current_fn,
            arguments=turn_args,
            revealed_ids=frozenset(revealed_ids),
            target_answer=target_answer,
            active_values=av_tuple,
        )
        trajectory.append(intent)

        # Build transition (skip turn 0)
        if slot.turn_idx > 0:
            fn_changed = current_fn != prev_fn
            newly_revealed = revealed_ids - prev_revealed
            val_changes = []
            for cid in sorted(active_vals):
                old_val = prev_active.get(cid)
                new_val = active_vals[cid]
                if old_val is not None and old_val != new_val:
                    val_changes.append((cid, old_val, new_val))

            if fn_changed:
                trans = IntentTransition(
                    transition_type="function_change",
                    old_function=prev_fn,
                    new_function=current_fn,
                    changed_arguments=tuple(val_changes) if val_changes else None,
                    revealed_ids=tuple(sorted(newly_revealed)) if newly_revealed else None,
                )
            elif val_changes:
                trans = IntentTransition(
                    transition_type="argument_change",
                    changed_arguments=tuple(val_changes),
                    revealed_ids=tuple(sorted(newly_revealed)) if newly_revealed else None,
                )
            else:
                trans = IntentTransition(
                    transition_type="argument_reveal",
                    revealed_ids=tuple(sorted(newly_revealed)) if newly_revealed else None,
                )
            transitions.append(trans)

    return ChangePlan(
        task_id=task_id,
        scenario=scenario,
        domain=domain,
        intent_trajectory=trajectory,
        transitions=transitions,
        final_label=ground_truth,
        raw_sample=raw,
    )


# ============================================================================
# Structured content extraction for naturalizer
# ============================================================================

def _build_structured_contents(
    slots: list[TurnSlot],
    turn_texts: list[str],
    turn_meta: list[dict[str, Any]],
) -> list:
    """Build StructuredTurnContent list from slots and render_turns output.

    Imported lazily to avoid circular imports.
    """
    try:
        from .naturalizer import StructuredTurnContent
    except ImportError:
        from situated_simulation.naturalizer import StructuredTurnContent

    contents = []
    for slot, text, meta in zip(slots, turn_texts, turn_meta):
        corr_texts = [
            e.corr_text or "" for e in slot.events if e.type == "correction"
        ]
        cond_texts = [ci.text for ci in slot.arguments]

        contents.append(StructuredTurnContent(
            turn_idx=slot.turn_idx,
            function_text=slot.function_text,
            correction_texts=corr_texts,
            argument_texts=cond_texts,
            prefix=meta["prefix"],
            rule_based_text=text,
            turn_type=meta["turn_type"],
            taxonomy_type=meta.get("taxonomy_type"),
        ))
    return contents


# ============================================================================
# Recap computation
# ============================================================================

_RECAP_PROMPT_COT = (
    "Before answering, first list all the arguments from our conversation "
    "that apply to this question, then provide your answer."
)


def _compute_recap_texts(
    recap_method: str | None,
    turn_texts: list[str],
    change_plan: "ChangePlan",
    selected_functions: list[dict[str, Any]],
    source_arguments: list[dict[str, Any]],
    source_function: str,
) -> list[str | None]:
    """Compute per-turn recap text for each recap method.

    Returns a list aligned with *turn_texts* (one entry per user turn).
    Turn 0 always returns ``None``; single-turn conversations return all
    ``None``.
    """
    num_turns = len(turn_texts)
    if recap_method is None or num_turns <= 1:
        return [None] * num_turns

    recaps: list[str | None] = [None]  # turn 0 — nothing to recap

    if recap_method == "prompt":
        for _ in range(1, num_turns):
            recaps.append(_RECAP_PROMPT_COT)

    elif recap_method == "dump":
        for t in range(1, num_turns):
            dumped = " ".join(turn_texts[:t])
            recaps.append(
                'Before answering, here is what I have previously told you: '
                f'"{dumped}". '
                'Use the arguments that are relevant to my current request.'
            )

    elif recap_method == "ground_truth":
        trajectory = change_plan.intent_trajectory

        # Map function text → function index (-1 for source function)
        function_to_idx: dict[str, int] = {source_function: -1}
        for i, pg in enumerate(selected_functions):
            function_to_idx[pg["predecessor_function"]] = i

        # Relevant argument IDs per function index
        source_cids = {c["argument_id"] for c in source_arguments}
        function_relevant: dict[int, set[int]] = {-1: source_cids}
        for i, pg in enumerate(selected_functions):
            function_relevant[i] = {
                c["argument_id"]
                for c in pg.get("counterfactual_arguments", [])
            }

        # Comprehensive argument text map (source + function-specific)
        cond_text_map: dict[int, str] = {
            c["argument_id"]: c["argument"] for c in source_arguments
        }
        for pg in selected_functions:
            for c in pg.get("counterfactual_arguments", []):
                if c["argument_id"] not in cond_text_map:
                    cond_text_map[c["argument_id"]] = c["argument"]

        for t in range(1, num_turns):
            intent = trajectory[t]
            function_idx = function_to_idx.get(intent.function, -1)
            relevant_cids = function_relevant.get(function_idx, source_cids)

            # Only arguments revealed in PREVIOUS turns (not the current
            # turn — those are already stated in the same message).
            prev_revealed = trajectory[t - 1].revealed_ids
            recap_cids = set(relevant_cids & prev_revealed)

            # Also exclude arguments CORRECTED this turn — the correction
            # text already appears in the current message.
            if t - 1 < len(change_plan.transitions):
                trans = change_plan.transitions[t - 1]
                if trans.changed_arguments:
                    for cid, _old, _new in trans.changed_arguments:
                        recap_cids.discard(cid)

            # Build argument texts with active values at turn t
            # (reflects any corrections that happened this turn)
            active_val_map = dict(intent.active_values)
            argument_texts: list[str] = []
            for cid in sorted(recap_cids):
                text = active_val_map.get(cid, cond_text_map.get(cid, ""))
                if text:
                    argument_texts.append(text)

            # Always recap the function so the model doesn't lose track,
            # UNLESS the function was changed THIS turn (the change text
            # already states it in the message).
            function_changed_this_turn = intent.function != trajectory[t - 1].function
            include_function = not function_changed_this_turn

            if include_function and argument_texts:
                joined = " ".join(argument_texts)
                recap = (
                    f"Before answering, recall that the function is: "
                    f"{intent.function}. "
                    f"The following arguments apply: {joined}"
                )
            elif include_function:
                recap = (
                    f"Before answering, recall that the function is: "
                    f"{intent.function}."
                )
            elif argument_texts:
                joined = " ".join(argument_texts)
                recap = (
                    f"Before answering, recall that the following "
                    f"arguments apply: {joined}"
                )
            else:
                recap = None

            recaps.append(recap)
    else:
        raise ValueError(f"Unknown recap_method: {recap_method!r}")

    return recaps


# ============================================================================
# Main Entry Point
# ============================================================================

def create_sample(
    raw: dict[str, Any],
    g: int,
    p: int,
    t: int,
    mode: str,
    domain: str,
    seed: int | None = None,
    # Prefix functions (callables from EvolvingIntent)
    get_function_change_prefix=None,
    get_correction_prefix=None,
    get_reveal_prefix=None,
    get_reveal_after_function_prefix=None,
    get_corr_after_reveal_prefix=None,
    get_new_info_prefix=None,
    join_prefix_content=None,
    # Optional
    system_prompt: str | None = None,
    instruction: str | None = None,
    recap_method: str | None = None,
    function_change_includes_function_text: bool = True,
    include_evidence: bool = True,
    *,
    post_fill_hook=None,
) -> "IntentSample | None":
    """Unified sample creation for all scenarios.

    Returns None only if the raw data lacks sufficient counterfactual functions or
    arguments. Never fails due to placement constraints.

    ``post_fill_hook`` (optional): callable invoked after ``fill_arguments``
    and Step 2c (text-fix for redistributed counterfactual arguments) but BEFORE
    ``fill_texts``. Signature::

        post_fill_hook(slots, raw, selected_functions, source_arguments,
                       counterfactual_cond_ids, counterfactual_map, cond_by_id)

    The hook may mutate ``slots`` (typically by inserting domain-specific
    ``ArgumentItem``s into ``slot.arguments``) before text rendering.
    Default ``None`` keeps behavior byte-identical to the historical
    pipeline. Used by SWE-bench scheduling to inject symptom arguments
    in function-first order.
    """
    try:
        from .user_simulation import (
            IntentSample, extract_ground_truth, _is_sql_sample, _sql_metadata,
            _build_sql_system_prompt, _join_prefix_content as default_join,
        )
    except ImportError:
        from situated_simulation.user_simulation import (
            IntentSample, extract_ground_truth, _is_sql_sample, _sql_metadata,
            _build_sql_system_prompt, _join_prefix_content as default_join,
        )

    if join_prefix_content is None:
        join_prefix_content = default_join

    source_function = raw.get("function", "")
    source_arguments = sorted(
        raw.get("arguments", []), key=lambda x: x.get("argument_id", 0)
    )
    task_id = raw.get("task_id", "")
    ground_truth = extract_ground_truth(raw)

    if not source_function or not source_arguments:
        return None

    # --- Step 0: Select functions and argument counterfactuals ---
    selected_functions = select_functions(raw, g, mode)
    actual_g = len(selected_functions)

    counterfactual_map = select_counterfactuals(raw, selected_functions, p, mode)
    actual_p = sum(len(v) for v in counterfactual_map.values())  # total correction events
    counterfactual_cond_ids = set(counterfactual_map.keys())

    # Build ordered correction specs: (cid, step) for each correction event.
    # Spread across unique cids first (round-robin), then multi-step as fallback.
    correction_specs: list[tuple[int, int]] = []
    for cid in sorted(counterfactual_map.keys()):
        for step in range(len(counterfactual_map[cid])):
            correction_specs.append((cid, step))

    # Validate: if we requested g>0 but got none, skip
    if g > 0 and actual_g == 0:
        return None
    if p > 0 and actual_p == 0:
        return None

    # Adjust t: must be >= min_turns, and cap to avoid empty turns
    min_turns = 1 + actual_g + actual_p
    if t < min_turns:
        return None

    # Cap t so we don't create turns with nothing to show.
    # Rather than trying to predict deferable counts upfront (which is fragile
    # due to overlapping argument IDs across functions), we use a generous cap here
    # and trim empty turns after fill_arguments runs.
    # Conservative upper bound: total unique arguments across all sources.
    all_cond_ids = {c["argument_id"] for c in source_arguments}
    for gobj in selected_functions:
        for cond in gobj.get("counterfactual_arguments", []):
            if not cond.get("is_shared", False):
                all_cond_ids.add(cond.get("argument_id"))
    deferable = max(0, len(all_cond_ids) - 1)
    max_useful_turns = min(t, min_turns + deferable)
    actual_t = max(min_turns, max_useful_turns)

    cond_by_id = {c["argument_id"]: c for c in source_arguments}

    # --- Step 1: Schedule events ---
    shared_cond_ids = _compute_shared_cond_ids(selected_functions)
    deadlines = _build_deadline_map(
        selected_functions, counterfactual_cond_ids, source_arguments,
        shared_cond_ids=shared_cond_ids,
    )
    slots = schedule_events(actual_g, actual_p, actual_t, correction_specs, deadlines)

    # --- Step 2: Fill arguments ---
    fill_arguments(
        slots, selected_functions, source_arguments,
        counterfactual_cond_ids, counterfactual_map, cond_by_id,
    )

    # --- Step 2b: Remove empty slots (no events AND no arguments) ---
    # With events spread evenly, empty slots may appear anywhere.
    # Keep Turn 0 always; remove truly empty inner/trailing slots.
    #
    # Before trimming, redistribute: if a slot is empty but a neighbor has
    # multiple arguments, steal one so the slot survives.  This ensures
    # the requested turn count is honoured whenever possible.
    #
    # Redundancy guard: a counterfactual argument stolen to a turn after ANY of
    # its corrections is redundant — the correction already communicated the
    # same value.  Only steal non-redundant arguments.
    _corr_turn_indices: dict[int, list[int]] = {}
    for s in slots:
        for ev in s.events:
            if ev.type == "correction" and ev.cond_id is not None:
                _corr_turn_indices.setdefault(ev.cond_id, []).append(s.turn_idx)

    def _is_redundant_at(ci: ArgumentItem, dest_turn: int) -> bool:
        if not ci.is_counterfactual:
            return False
        ct = _corr_turn_indices.get(ci.cond_id, [])
        return any(t < dest_turn for t in ct)

    for s in slots:
        if s.turn_idx == 0 or s.events or s.arguments:
            continue
        # Find the nearest donor that stays non-empty after moving an argument.
        # An event-bearing slot can donate its sole argument because the event
        # still gives that turn content. Turn 0 must retain an argument so the
        # initial request is not reduced to a bare function.
        best_donor = None
        best_dist = float("inf")
        best_ci_idx = -1
        for d in slots:
            can_donate = len(d.arguments) > 1 or (
                d.turn_idx != 0 and bool(d.events) and len(d.arguments) == 1
            )
            if d is s or not can_donate:
                continue
            for idx in range(len(d.arguments) - 1, -1, -1):
                if not _is_redundant_at(d.arguments[idx], s.turn_idx):
                    dist = abs(d.turn_idx - s.turn_idx)
                    if dist < best_dist:
                        best_dist = dist
                        best_donor = d
                        best_ci_idx = idx
                    break
        if best_donor is not None:
            s.arguments.append(best_donor.arguments.pop(best_ci_idx))

    slots = [s for s in slots
             if s.turn_idx == 0 or s.events or s.arguments]
    # Re-number turn indices after removal
    for i, slot in enumerate(slots):
        slot.turn_idx = i
    actual_t = len(slots)

    # --- Step 2c: Fix stale text for redistributed counterfactual arguments ---
    # Step 2b may have moved a counterfactual ArgumentItem past its correction
    # chain.  Compute what the active value would be at the reveal turn and
    # update .text accordingly.
    if counterfactual_cond_ids:
        # Build per-cid ordered list of correction turn indices
        corr_turns: dict[int, list[int]] = {cid: [] for cid in counterfactual_cond_ids}
        for s in slots:
            for ev in s.events:
                if ev.type == "correction" and ev.cond_id in corr_turns:
                    corr_turns[ev.cond_id].append(s.turn_idx)

        for s in slots:
            for ci in s.arguments:
                if not ci.is_counterfactual or ci.cond_id not in counterfactual_cond_ids:
                    continue
                # How many corrections for this cid precede this turn?
                preceding = sum(
                    1 for ct in corr_turns[ci.cond_id] if ct < s.turn_idx
                )
                if preceding == 0:
                    continue  # text is still the initial counterfactual value
                # Build the same correction chain as fill_texts
                counterfactuals = counterfactual_map.get(ci.cond_id, [])
                chain = [p.get("counterfactual_argument", "") for p in counterfactuals[1:]]
                chain.append(cond_by_id[ci.cond_id]["argument"])
                ci.text = chain[min(preceding - 1, len(chain) - 1)]
                if ci.text == cond_by_id[ci.cond_id]["argument"]:
                    ci.is_counterfactual = False

    # --- Step 2d: Domain-specific post-fill hook ---
    # Hook receives the (still-mutable) slot layout after Step 2c and may
    # inject domain-specific ArgumentItems before text rendering. Pure
    # insertion only; removal would re-introduce the stale-text issues we
    # fixed in earlier rounds. Default: no-op (byte-identical for non-SWE).
    if post_fill_hook is not None:
        post_fill_hook(
            slots,
            raw,
            selected_functions,
            source_arguments,
            counterfactual_cond_ids,
            counterfactual_map,
            cond_by_id,
        )

    # --- Step 3: Fill text content ---
    fill_texts(slots, selected_functions, source_function, counterfactual_map, cond_by_id)

    # --- Step 4: Render turns ---
    is_predecessor = any(
        pg.get("is_predecessor", False) for pg in selected_functions
    )

    turn_texts, turn_meta = render_turns(
        slots,
        selected_functions,
        is_predecessor,
        get_function_change_prefix=get_function_change_prefix,
        get_correction_prefix=get_correction_prefix,
        get_reveal_prefix=get_reveal_prefix,
        get_reveal_after_function_prefix=get_reveal_after_function_prefix,
        get_corr_after_reveal_prefix=get_corr_after_reveal_prefix,
        get_new_info_prefix=get_new_info_prefix,
        join_prefix_content=join_prefix_content,
        function_change_includes_function_text=function_change_includes_function_text,
    )

    # --- Step 4b: Build structured contents (always, for step() support) ---
    structured = _build_structured_contents(slots, turn_texts, turn_meta)

    # --- Infer scenario ---
    if actual_g == 0 and actual_p == 0:
        scenario = "fully-specified" if t == 1 else "under-specified"
    elif actual_g > 0 and actual_p == 0:
        scenario = "function-switch"
    elif actual_g == 0 and actual_p > 0:
        scenario = "argument-revision"
    else:
        scenario = "combined"

    # --- Build change plan ---
    change_plan = build_change_plan(
        slots, raw, selected_functions, source_function, source_arguments,
        counterfactual_cond_ids, counterfactual_map, ground_truth, domain, scenario,
    )

    # --- Compute recap texts ---
    recap_texts = _compute_recap_texts(
        recap_method, turn_texts, change_plan,
        selected_functions, source_arguments, source_function,
    )

    # --- Build turns list ---
    turns = []
    # System prompt (non-SQL domains only; SQL uses user turn 1)
    if system_prompt and not _is_sql_sample(raw):
        turns.append({"role": "system", "content": system_prompt})

    for i, text in enumerate(turn_texts):
        content = text
        # Append recap to turn content (before instruction/schema wrapping)
        if recap_texts[i] is not None:
            content = content + " " + recap_texts[i]
        if i == 0:
            # SQL: prepend schema (and evidence if available) to user turn 1, BIRD-style
            if _is_sql_sample(raw):
                schema = raw.get("schema", "")
                evidence = raw.get("evidence", "") if include_evidence else ""
                prefix_parts = []
                if schema:
                    prefix_parts.append(f"Database schema:\n{schema}")
                if evidence and evidence.strip():
                    prefix_parts.append(f"Evidence:\n{evidence.strip()}")
                if prefix_parts:
                    content = "\n\n".join(prefix_parts) + f"\n\n{content}"
            if instruction:
                content = instruction.format(content=content)
        turns.append({"role": "user", "content": content})

    # --- Per-turn gold for SQL evaluation ---
    per_turn_gold = _compute_per_turn_gold(
        raw, change_plan, selected_functions, counterfactual_map, ground_truth,
    )

    # --- Metadata ---
    metadata = {
        "task_id": task_id,
        "scenario": scenario,
        "mode": mode,
        "num_turns": actual_t,
        "requested_num_turns": t,
        "num_arguments": len(source_arguments),
        "num_counterfactual_arguments": actual_p,
        "num_predecessor_functions": actual_g,
        "function": source_function,
        "source_function": source_function,
        "answer": ground_truth,
        "change_plan": change_plan.to_dict(),
        "recap_method": recap_method,
        "instruction_id_list": raw.get("instruction_id_list"),
        "kwargs": raw.get("kwargs"),
        "data_source": raw.get("data_source"),
        "per_turn_gold": per_turn_gold,
        **_sql_metadata(raw),
    }

    # Backfill target_answer in serialized change_plan for SQL turns
    if per_turn_gold:
        cp_dict = metadata["change_plan"]
        for i, gold_entry in enumerate(per_turn_gold):
            traj = cp_dict.get("intent_trajectory", [])
            if i < len(traj) and gold_entry.get("answer") and not traj[i].get("target_answer"):
                traj[i]["target_answer"] = gold_entry["answer"]

    return IntentSample(
        task_id=task_id,
        turns=turns,
        label=ground_truth,
        metadata=metadata,
        structured_contents=structured,
        recap_texts=recap_texts,
    )
