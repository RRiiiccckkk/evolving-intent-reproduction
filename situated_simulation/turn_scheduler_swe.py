"""SWE-bench Verified-specific scheduling overlay.

Strategy
--------
The generic ``turn_scheduler.create_sample`` is reused unchanged. SWE-specific
behavior is added in two surgical steps:

  1. Strip ``symptom``-category arguments from the raw input before the
     scheduler runs. Symptoms are not counterfactual_eligible and we want to control
     where they appear; removing them up front prevents the scheduler from
     deferring them across turns and keeps the turn-budget calculation
     focused on the real counterfactual_eligible / fixed user-supplied arguments.

  2. Inject the stripped symptoms BACK into ``slot.arguments`` via a
     ``post_fill_hook`` invoked after ``fill_arguments`` (and Step 2c)
     but BEFORE ``fill_texts`` runs. Symptoms are inserted at index 0 of
     each receiving slot's ``arguments`` list, so when ``fill_texts`` and
     ``render_turns`` then build the user content, each turn reads:

         function_text  →  symptom(s)  →  other arguments

     This is the natural developer-conversation order: state the function,
     describe what's wrong, then add the rest of the context.

  3. ``structured_contents`` is built downstream from the same slot layout,
     so the naturalizer / step()-time delivery sees symptoms automatically.
     There is no post-render text mutation.

Per-phase distribution: each function phase's symptoms are placed only within
that phase's turns (slots). In a function-switch scenario with predecessors G1,
G2 and target G3, G1's symptoms occupy G1's turn(s), etc. — never crossing
phases. Phase membership is determined by walking ``slot.events`` for
``function_change`` events (no reliance on post-render metadata).

Failure modes that are guarded with ``NotImplementedError`` rather than
silently producing wrong data:

  * ``recap_method='dump'`` (recap is computed before this overlay would
    add symptoms; the resulting recap would not match the rendered text).
  * Two predecessor functions share the same ``predecessor_function`` text (would
    merge their phases).
  * The source target's function text equals a predecessor's ``predecessor_function``
    text (would merge target and predecessor phases).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from .turn_scheduler import ArgumentItem, create_sample
except ImportError:  # pragma: no cover - allows running as a script
    from situated_simulation.turn_scheduler import ArgumentItem, create_sample


# ---------------------------------------------------------------------------
# Ceil-front math
# ---------------------------------------------------------------------------

def distribute_ceil_front(n_items: int, n_buckets: int) -> list[int]:
    """Return per-bucket counts using ceil-front uniform distribution.

    Examples:
      >>> distribute_ceil_front(3, 2)
      [2, 1]
      >>> distribute_ceil_front(7, 3)
      [3, 2, 2]
      >>> distribute_ceil_front(2, 4)
      [1, 1, 0, 0]
    """
    if n_buckets <= 0:
        return []
    base = n_items // n_buckets
    extras = n_items % n_buckets
    return [base + (1 if i < extras else 0) for i in range(n_buckets)]


# ---------------------------------------------------------------------------
# Step 1: strip symptoms from raw
# ---------------------------------------------------------------------------

def _strip_symptoms(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Return ``(stripped_raw, target_symptoms, predecessor_symptoms_by_function_text)``.

    The returned ``stripped_raw`` is a deep copy of ``raw`` with all
    ``category == "symptom"`` entries removed from ``raw["arguments"]`` and
    from each ``raw["predecessor_functions"][i]["counterfactual_arguments"]``.

    ``predecessor_symptoms_by_function_text`` is keyed by the predecessor's
    ``predecessor_function`` so that the injection hook can match against the
    function-text-based phase map. Two predecessors with the same function text
    would merge phases ambiguously — refuse early instead of silently
    losing data.
    """
    stripped = deepcopy(raw)
    _normalize_function_punctuation(stripped)

    target_symptoms = [
        c for c in (stripped.get("arguments") or [])
        if c.get("category") == "symptom"
    ]
    stripped["arguments"] = [
        c for c in (stripped.get("arguments") or [])
        if c.get("category") != "symptom"
    ]

    predecessor_symptoms: dict[str, list[dict[str, Any]]] = {}
    for pg in stripped.get("predecessor_functions") or []:
        function_text = pg.get("predecessor_function") or ""
        syms = [
            c for c in (pg.get("counterfactual_arguments") or [])
            if c.get("category") == "symptom"
        ]
        if syms:
            if function_text in predecessor_symptoms:
                raise NotImplementedError(
                    "Two predecessor functions share the same function text "
                    f"({function_text!r}). The SWE symptom overlay groups symptoms "
                    "by function text; duplicate predecessor texts would force "
                    "their symptoms into the same phase. Predecessor texts "
                    "must be unique."
                )
            predecessor_symptoms[function_text] = syms
        pg["counterfactual_arguments"] = [
            c for c in (pg.get("counterfactual_arguments") or [])
            if c.get("category") != "symptom"
        ]

    return stripped, target_symptoms, predecessor_symptoms


def _normalize_function_punctuation(stripped: dict[str, Any]) -> None:
    """Ensure function texts end with sentence-terminal punctuation.

    SWE functions are imperatives ("Fix X", "Make Y", "Ensure Z") that the
    extractor often leaves without a trailing period, which produces
    run-on sentences when ``render_turns`` joins the function with the next
    argument using a single space (e.g.,
    ``"...overwritten on save() In the attached example code..."``).

    Mutates ``stripped`` in place: appends "." to ``stripped["function"]`` and
    to each ``stripped["predecessor_functions"][i]["predecessor_function"]`` whenever
    the text does not already end in ``.``, ``!``, or ``?``. Idempotent.
    The Stage 1 extraction prompt has been updated to make this guarantee
    upstream as well; this function is the runtime safety net.
    """
    def _fix(text: str) -> str:
        if not isinstance(text, str):
            return text
        stripped_text = text.rstrip()
        if not stripped_text:
            return text
        if stripped_text[-1] in ".!?":
            return text
        # Preserve the original trailing whitespace, just insert the
        # terminal "." right before it.
        trailing = text[len(stripped_text):]
        return stripped_text + "." + trailing

    if "function" in stripped:
        stripped["function"] = _fix(stripped.get("function") or "")
    for pg in stripped.get("predecessor_functions") or []:
        if "predecessor_function" in pg:
            pg["predecessor_function"] = _fix(pg.get("predecessor_function") or "")


def _repair_phase_leaks(
    slots,
    selected_functions: list[dict[str, Any]],
    source_function_text: str,
    cond_by_id: dict[int, dict] | None = None,
    source_cid_set: set[int] | None = None,
) -> None:
    """Move every argument out of slots whose active function does not own it.

    Owner rule (matches paper notation, generalised across domains):
      - **Source function** owns ``cid`` iff ``cid`` belongs to the source
        argument set (i.e. is one of the source ``arguments[*].argument_id``).
        For most domains every cid is in that set (PG counterfactuals are
        alternative *values* of the same cid). For SWE-pairing samples the
        source function ``G_target`` and the predecessor function ``G_pred`` are
        independent real bugs with disjoint cid sets, so cids that exist
        only in ``PG_pred.counterfactual_arguments`` are NOT owned by source.
      - **Counterfactual function** ``PG_k`` owns ``cid`` iff ``cid`` appears in
        ``PG_k["counterfactual_arguments"]``.

    When a cid lives in slot whose active function does not own it, we move it
    to the *first* slot whose active function does own it. Search order:
      1. Forward to the source slot (if source owns the cid).
      2. Backward/forward to a PG slot whose counterfactual_arguments include
         the cid (for cids that are exclusively owned by a PG, e.g.
         G_pred-only cids in SWE pairing).

    Mutates ``slots`` in place. ``cond_by_id`` is used to reset stale
    counterfactual text on items moved past the source turn.

    This runs only on the SWE path; the generic scheduler is unchanged
    for math/IF/search/SQL. The behavior is *backward-compatible* for
    non-SWE: when every cid is in ``source_cid_set``, the new ownership
    test reduces to the prior "source owns everything" rule.
    """
    g = len(selected_functions)
    if g == 0:
        return  # no PG phases → nothing to repair

    # Build the per-PG owner sets. For PG_k (1-indexed by paper notation,
    # 0-indexed here in selected_functions), owner_sets[k] = cids in
    # PG_k.counterfactual_arguments.
    owner_sets: list[set[int]] = []
    for pg in selected_functions:
        owner_sets.append({
            c.get("argument_id")
            for c in pg.get("counterfactual_arguments", [])
            if c.get("argument_id") is not None
        })

    # Default source_cid_set to "all cids ever placed", which preserves the
    # old "source owns everything" behavior whenever the caller doesn't
    # supply a tighter set. SWE callers DO supply it.
    if source_cid_set is None:
        source_cid_set = set()
        for slot in slots:
            for ci in slot.arguments:
                source_cid_set.add(ci.cond_id)
        for owners in owner_sets:
            source_cid_set |= owners

    # Compute active fn per slot AFTER processing that slot's events
    # (matches build_change_plan and renderer convention).
    active: list[int] = []
    cur = 0
    source_slot_idx = -1
    pg_slot_idx: dict[int, int] = {}  # PG_k function_idx → first slot active under it
    for i, slot in enumerate(slots):
        for ev in slot.events:
            if ev.type == "function_init":
                cur = 0
                # function_init activates PG_0 starting at this slot.
                pg_slot_idx.setdefault(0, i)
            elif ev.type == "function_change":
                cur = ev.function_idx if ev.function_idx is not None else -1
                if ev.function_idx == -1:
                    source_slot_idx = i
                elif ev.function_idx >= 0:
                    pg_slot_idx.setdefault(ev.function_idx, i)
        # If no function_init/function_change has fired yet (slot 0 with no events),
        # the implicit active is PG_0 anyway — record it.
        if cur >= 0:
            pg_slot_idx.setdefault(cur, i)
        active.append(cur)

    if source_slot_idx < 0:
        return  # no source restoration → no safe destination

    def _owns(active_idx: int, cid: int) -> bool:
        """Does the active function at ``active_idx`` own ``cid``?"""
        if active_idx == -1:
            # Source owns cid iff cid is in the source argument set. For
            # non-SWE samples this is "every cid"; for SWE pairing this
            # excludes G_pred-only cids.
            return cid in source_cid_set
        if not (0 <= active_idx < len(owner_sets)):
            return True
        return cid in owner_sets[active_idx]

    def _find_owner_slot(cid: int) -> int:
        """Return slot index whose active function owns ``cid``, or -1.

        Prefers the source slot; otherwise picks the earliest PG slot that
        owns the cid (= the PG that introduced it).
        """
        if cid in source_cid_set:
            return source_slot_idx
        for k, owners in enumerate(owner_sets):
            if cid in owners and k in pg_slot_idx:
                return pg_slot_idx[k]
        return -1

    for slot_idx, slot in enumerate(slots):
        kept: list = []
        moved_to_source: list = []
        moved_other: list[tuple[int, Any]] = []  # (target_slot_idx, item)
        for ci in slot.arguments:
            if _owns(active[slot_idx], ci.cond_id):
                kept.append(ci)
                continue
            target = _find_owner_slot(ci.cond_id)
            if target == -1 or target == slot_idx:
                # No safe destination — leave it where it was; the model
                # already saw it, repairing further would lose data.
                kept.append(ci)
                continue
            if target == source_slot_idx:
                moved_to_source.append(ci)
            else:
                moved_other.append((target, ci))

        if not (moved_to_source or moved_other):
            continue

        # Stale-text reset on items being moved past the source turn —
        # by the time source renders, every prior correction has fired.
        if cond_by_id is not None:
            for ci in moved_to_source:
                if ci.is_counterfactual and ci.cond_id in cond_by_id:
                    ci.text = cond_by_id[ci.cond_id].get("argument", ci.text)
                    ci.is_counterfactual = False

        slot.arguments = kept
        if moved_to_source:
            slots[source_slot_idx].arguments.extend(moved_to_source)
        for target_idx, ci in moved_other:
            slots[target_idx].arguments.append(ci)


# ---------------------------------------------------------------------------
# Step 1b: phase-aware redistribute (post-leak-repair)
# ---------------------------------------------------------------------------

def _redistribute_within_phase(
    slots,
    selected_functions: list[dict[str, Any]],
    source_function_text: str,
    source_cid_set: set[int] | None = None,
) -> None:
    """Fill phase-internal empty slots from same-phase donors.

    Runs after ``_repair_phase_leaks`` has corrected any cross-phase leaks
    introduced by the generic Step 2b redistribute pass. The scenario:

      * scheduler reserved t turns for the user-requested t.
      * after fill_arguments, a phase has fewer cids than slots, leaving
        empty slots inside the phase.
      * generic Step 2b redistribute filled them by stealing cross-phase.
      * leak repair moved the stolen cids back to their owner phase.
      * net result: empty same-phase slot resurfaces.

    This pass redistributes within-phase donors only: for each empty slot,
    find the nearest slot in the SAME phase with len(arguments) >= 2 and
    move ONE non-redundant argument over. Cross-phase steals stay
    forbidden (otherwise the leak repair would just move it back).

    Idempotent and safe even when no empty slots exist.
    """
    if not slots:
        return

    # Build {turn_idx: phase_index} where phase_index counts function phases
    # 0..g for selected_functions[0..g-1] and g for source. Walk events to track
    # the active phase.
    g = len(selected_functions)
    if selected_functions:
        cur_phase = 0
    else:
        cur_phase = g
    phase_per_turn: dict[int, int] = {}
    for s in slots:
        for ev in s.events:
            if ev.type != "function_change":
                continue
            if ev.function_idx == -1:
                cur_phase = g
            elif isinstance(ev.function_idx, int) and 0 <= ev.function_idx < g:
                cur_phase = ev.function_idx
        phase_per_turn[s.turn_idx] = cur_phase

    by_phase: dict[int, list] = {}
    for s in slots:
        by_phase.setdefault(phase_per_turn[s.turn_idx], []).append(s)

    # Precompute correction turn indices for redundancy check.
    corr_turns: dict[int, list[int]] = {}
    for s in slots:
        for ev in s.events:
            if ev.type == "correction" and ev.cond_id is not None:
                corr_turns.setdefault(ev.cond_id, []).append(s.turn_idx)

    def _is_redundant_at(ci, dest_turn: int) -> bool:
        if not getattr(ci, "is_counterfactual", False):
            return False
        ct = corr_turns.get(getattr(ci, "cond_id", None), [])
        return any(t < dest_turn for t in ct)

    for ph_slots in by_phase.values():
        empties = [s for s in ph_slots
                   if not s.events and not s.arguments and s.turn_idx > 0]
        if not empties:
            continue
        # Iterate empty slots, each picking a donor with len(arguments)
        # >= 2 OR len(arguments) >= 1 if no >=2 donor remains. The
        # "max(argument_count) - 1" heuristic ensures we can balance
        # an even spread (e.g. 3 cids across 3 slots: each ends at 1)
        # without leaving any empty.
        for empty in empties:
            best_donor = None
            best_idx = -1
            best_dist = float("inf")
            best_donor_load = 0
            for d in ph_slots:
                if d is empty or len(d.arguments) == 0:
                    continue
                # Find a non-redundant cid we can pop. Last-first.
                for idx in range(len(d.arguments) - 1, -1, -1):
                    if not _is_redundant_at(d.arguments[idx], empty.turn_idx):
                        # Prefer richer donors (more cids) so we balance,
                        # then nearer slot, then pickable cid.
                        load = len(d.arguments)
                        # Skip donor that would itself become empty unless
                        # it's the only option (we sort donors by load
                        # descending so this is rare).
                        dist = abs(d.turn_idx - empty.turn_idx)
                        # Comparator: maximize load, minimize dist.
                        # We use (-load, dist) so smaller tuple is better.
                        key = (-load, dist)
                        cur_key = (-best_donor_load, best_dist)
                        if best_donor is None or key < cur_key:
                            best_donor = d
                            best_idx = idx
                            best_dist = dist
                            best_donor_load = load
                        break
            if best_donor is not None and best_donor_load >= 1:
                # Don't drain the last cid from a donor when the donor
                # would itself end up empty AND it's not the slot 0 anchor
                # (which the trim pass keeps unconditionally). If the only
                # donors have load==1 we still pop, accepting that another
                # phase slot becomes empty; that's the math limit and is
                # better than leaving the current slot empty.
                empty.arguments.append(
                    best_donor.arguments.pop(best_idx)
                )


# ---------------------------------------------------------------------------
# Step 1c: align moved argument text with its actual reveal turn
# ---------------------------------------------------------------------------

def _sync_counterfactual_argument_values(
    slots,
    counterfactual_map: dict[int, list[dict]],
    cond_by_id: dict[int, dict],
) -> None:
    """Render moved counterfactual arguments at their value on that turn."""
    correction_turns: dict[int, list[int]] = {}
    for slot in slots:
        for event in slot.events:
            if event.type == "correction" and event.cond_id is not None:
                correction_turns.setdefault(event.cond_id, []).append(slot.turn_idx)

    for slot in slots:
        for item in slot.arguments:
            variants = counterfactual_map.get(item.cond_id)
            source = cond_by_id.get(item.cond_id)
            if not variants or source is None:
                continue

            preceding = sum(
                turn_idx < slot.turn_idx
                for turn_idx in correction_turns.get(item.cond_id, [])
            )
            if preceding == 0:
                value = variants[0].get("counterfactual_argument", "")
            else:
                chain = [
                    variant.get("counterfactual_argument", "")
                    for variant in variants[1:]
                ]
                chain.append(source.get("argument", ""))
                value = chain[min(preceding - 1, len(chain) - 1)]

            item.text = value
            item.is_counterfactual = value != source.get("argument", "")


# ---------------------------------------------------------------------------
# Step 2: phase mapping at slot level
# ---------------------------------------------------------------------------

def _slot_phase_map(
    slots,
    selected_functions: list[dict[str, Any]],
    source_function_text: str,
) -> dict[str, list[int]]:
    """Walk ``slots`` and return ``{function_text: [turn_idx, ...]}`` per phase.

    Phase identifier is the active function's text:
      - Initial phase = ``selected_functions[0]['predecessor_function']`` if any
        predecessors are selected, else ``source_function_text``.
      - Each ``function_change`` event with ``function_idx == -1`` transitions to
        the source target phase.
      - Each ``function_change`` event with a non-negative integer ``function_idx``
        transitions to ``selected_functions[function_idx]['predecessor_function']``.

    A slot belongs to whichever phase is active AT THAT SLOT (i.e., after
    applying any function_change events on the same slot, since the function change
    is rendered before the slot's user content).
    """
    out: dict[str, list[int]] = {}
    if selected_functions:
        current = selected_functions[0].get("predecessor_function", "") or ""
    else:
        current = source_function_text

    for slot in slots:
        for ev in slot.events:
            if ev.type != "function_change":
                continue
            if ev.function_idx == -1:
                current = source_function_text
            elif (
                isinstance(ev.function_idx, int)
                and 0 <= ev.function_idx < len(selected_functions)
            ):
                current = selected_functions[ev.function_idx].get("predecessor_function", "") or ""
        out.setdefault(current, []).append(slot.turn_idx)
    return out


# ---------------------------------------------------------------------------
# Step 3: build the post-fill hook (closure capturing symptoms)
# ---------------------------------------------------------------------------

def _make_inject_hook(
    target_symptoms: list[dict[str, Any]],
    predecessor_symptoms: dict[str, list[dict[str, Any]]],
    source_function_text: str,
):
    """Return a post-fill hook that injects symptoms into slot.arguments.

    Pure insertion: nothing is removed from ``slot.arguments``. Symptoms
    were stripped from ``raw`` before scheduling, so the scheduler never
    placed them anywhere — the slots arrive without symptoms and we add
    them. No empty-turn or stale-text concerns.
    """
    sympts_by_phase: dict[str, list[dict[str, Any]]] = dict(predecessor_symptoms)
    if target_symptoms:
        if source_function_text in sympts_by_phase:
            raise NotImplementedError(
                "The source (target) function text "
                f"({source_function_text!r}) collides with a predecessor's function "
                "text. The SWE symptom overlay groups symptoms by function text; "
                "this collision would merge target and predecessor symptoms "
                "into a single phase. The target's function text must be distinct "
                "from every predecessor's function text."
            )
        sympts_by_phase[source_function_text] = target_symptoms

    def hook(
        slots,
        raw: dict[str, Any],
        selected_functions: list[dict[str, Any]],
        source_arguments: list[dict[str, Any]],
        counterfactual_cond_ids: set[int],
        counterfactual_map: dict[int, list[dict]],
        cond_by_id: dict[int, dict],
    ) -> None:
        # Repair cross-phase leaks introduced by the generic
        # ``fill_arguments``: any argument placed in a slot whose active
        # function does not own it is moved to a slot that does. This is
        # SWE-only — the generic path is unchanged for other domains.
        #
        # We pass ``source_cid_set`` derived from the *source arguments list*
        # so the ownership rule correctly excludes G_pred-only cids from
        # source ownership in SWE pairing samples (where G_target and G_pred
        # are independent real bugs with disjoint cid sets).
        source_cid_set = {
            c.get("argument_id") for c in source_arguments
            if c.get("argument_id") is not None
        }
        _repair_phase_leaks(
            slots, selected_functions, source_function_text,
            cond_by_id=cond_by_id,
            source_cid_set=source_cid_set,
        )

        # After leak repair, the generic Step 2b redistribute (in
        # turn_scheduler.create_sample) may have left phase-internal empty
        # slots. Specifically: when an exploration function owns 0 arguments
        # and the user requested t > min_turns, the leftover reveal slot
        # ends up in the source phase (or impl phase) but the generic
        # redistribute picked donors from the wrong phase, the leak repair
        # moved them back, and the empty slot resurfaced. Run a SWE-only,
        # phase-aware redistribute pass to fill those empties from same-
        # phase donors so the scheduler honours the requested t whenever
        # mathematically possible. Cross-phase steals are still forbidden.
        _redistribute_within_phase(
            slots, selected_functions, source_function_text,
            source_cid_set=source_cid_set,
        )

        # Symptom injection: if no symptoms exist, skip the inject loop but
        # still run the canonical-order sort below.
        if sympts_by_phase:
            phase_map = _slot_phase_map(slots, selected_functions, source_function_text)

            # Index slots by turn_idx for O(1) lookup.
            slot_by_turn = {s.turn_idx: s for s in slots}

            for phase_text, syms in sympts_by_phase.items():
                if not syms:
                    continue
                slot_indices = phase_map.get(phase_text, [])
                if not slot_indices:
                    # Phase didn't appear in the rendered slots (typically
                    # when the corresponding predecessor wasn't selected by
                    # ``select_functions``). Skip silently — its symptoms have
                    # no phase to belong to.
                    continue
                counts = distribute_ceil_front(len(syms), len(slot_indices))
                cursor = 0
                for slot_idx, count in zip(slot_indices, counts):
                    if count <= 0:
                        continue
                    chunk = syms[cursor:cursor + count]
                    cursor += count
                    target_slot = slot_by_turn.get(slot_idx)
                    if target_slot is None:
                        continue
                    for sym in reversed(chunk):
                        cid = sym.get("argument_id")
                        text = sym.get("argument", "")
                        if cid is None:
                            continue
                        target_slot.arguments.insert(
                            0, ArgumentItem(cid, text, False)
                        )

        # Symptom injection can make a previously single-argument donor
        # splittable. Run the phase-aware pass again so a reserved reveal
        # slot receives real, not duplicated, information before rendering.
        # This matters for the paper's 7-turn Evolve setting: each of its two
        # reveal boundaries must carry a newly disclosed argument.
        _redistribute_within_phase(
            slots, selected_functions, source_function_text,
            source_cid_set=source_cid_set,
        )

        # The generic stale-text pass ran before this SWE hook. Repairs above
        # can move an argument across its correction boundary, so recompute
        # the value visible at the argument's final reveal turn.
        _sync_counterfactual_argument_values(
            slots, counterfactual_map, cond_by_id,
        )

        # Final pass: deterministically order within-slot arguments for
        # SWE. The scheduler's planting + leak-repair leaves slot.arguments
        # in an order that depends on ``t`` (because deferring path differs
        # with the number of empty slots). For SWE that produces unstable
        # rendering across t — same cids appear in different sentence order
        # when only ``t`` changes. We canonicalise the order by SWE
        # argument-category hierarchy (symptom > trigger > location >
        # approach > scope > constraint > unknown) with cid as tiebreaker.
        # This is purely cosmetic from a semantics standpoint but gives the
        # model (and the human reader) consistent rendering.
        cat_priority = {
            "symptom": 0,
            "trigger": 1,
            "location": 2,
            "approach": 3,
            "scope": 4,
            "constraint": 5,
        }
        cid_to_cat: dict[int, str] = {}
        for c in raw.get("arguments", []) or []:
            if (cid := c.get("argument_id")) is not None:
                cid_to_cat[cid] = (c.get("category") or "")
        for pg in raw.get("predecessor_functions", []) or []:
            for c in pg.get("counterfactual_arguments", []) or []:
                if (cid := c.get("argument_id")) is not None:
                    cid_to_cat.setdefault(cid, c.get("category") or "")

        def _sort_key(ci):
            cat = cid_to_cat.get(ci.cond_id, "")
            return (cat_priority.get(cat, 99), ci.cond_id)

        for slot in slots:
            slot.arguments.sort(key=_sort_key)

    return hook


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def create_sample_swe(raw: dict[str, Any], *args, **kwargs):
    """SWE-bench-Verified version of ``turn_scheduler.create_sample``.

    Differences from the generic entry point:

      1. Symptom-category arguments are stripped from ``raw`` before
         scheduling.
      2. The generic scheduler runs unchanged on the symptom-free input.
      3. A ``post_fill_hook`` injects the stripped symptoms back into the
         right slots (per-phase, ceil-front distributed) at
         ``slot.arguments[0]``, so the rendered turn order becomes
         ``function_text -> symptom(s) -> other arguments``.
    """
    if kwargs.get("recap_method") == "dump":
        raise NotImplementedError(
            "create_sample_swe does not support recap_method='dump' yet. "
            "The generic scheduler computes dump recap text from the "
            "symptom-stripped slot layout, so the recap would not include "
            "the symptoms that the rendered turns do contain."
        )

    stripped, target_syms, pred_syms = _strip_symptoms(raw)
    source_function_text = stripped.get("function", "") or ""
    hook = _make_inject_hook(target_syms, pred_syms, source_function_text)

    # Refuse to silently override a caller-provided hook.
    if "post_fill_hook" in kwargs and kwargs["post_fill_hook"] is not None:
        raise NotImplementedError(
            "create_sample_swe does not support a caller-supplied "
            "post_fill_hook (it installs its own to inject symptoms)."
        )
    kwargs["post_fill_hook"] = hook
    sample = create_sample(stripped, *args, **kwargs)

    # Pass-through SWE-specific raw fields onto sample.metadata for
    # post-hoc analysis / instance_id resolution. This is purely additive:
    # it never overwrites existing keys and is gated on the SWE wrapper, so
    # other domains' metadata layout is untouched.
    if sample is not None and isinstance(getattr(sample, "metadata", None), dict):
        for key in ("task", "original_id", "swe_bench_metadata"):
            if key in raw and key not in sample.metadata:
                sample.metadata[key] = raw[key]
        # Also surface instance_id explicitly (mirrors swebench convention)
        # without depending on raw key naming.
        if "instance_id" not in sample.metadata and raw.get("original_id"):
            sample.metadata["instance_id"] = raw["original_id"]

        # Drop empty user turns produced when the production scheduler
        # spreads more event slots across turns than there are arguments
        # to reveal. SWE samples have noticeably fewer arguments per
        # instance than other domains, so the gap-filling logic in
        # ``turn_scheduler.fill_arguments`` can leave intermediate turns
        # with no content. Empty user messages are never useful (they
        # produce noisy LLM rounds and break viewer rendering), so we
        # collapse them here.
        #
        # We also align the metadata.change_plan trajectory + transitions
        # so downstream consumers (per-turn evaluators, viewer recap, etc.)
        # see a consistent (turns, trajectory) pair after the collapse.
        _collapse_empty_user_turns(sample)

    return sample


def _collapse_empty_user_turns(sample) -> None:
    """Drop empty user turns from a IntentSample in-place; sync metadata.

    Mutates ``sample.turns`` and ``sample.metadata`` (specifically the
    ``num_turns`` field and ``change_plan.intent_trajectory`` /
    ``change_plan.transitions``). System turns are never dropped.
    Recap-related state (``recap_texts``, ``structured_contents``) is also
    realigned by user-turn index so that ``IntentSample.reset()`` / ``step()``
    drive the correct trimmed sequence.
    """
    turns = list(sample.turns or [])
    if not turns:
        return

    # Identify which user-turn indices to keep.
    keep_user_idx: list[int] = []
    user_idx = -1
    new_turns: list[dict] = []
    for t in turns:
        if t.get("role") == "user":
            user_idx += 1
            content = (t.get("content") or "").strip()
            if not content:
                continue  # drop empty user turn
            keep_user_idx.append(user_idx)
            new_turns.append(t)
        else:
            new_turns.append(t)

    if len(new_turns) == len(turns):
        return  # nothing changed

    sample.turns = new_turns

    # Realign per-user-turn lists so reset()/step() track the trimmed turns.
    if isinstance(getattr(sample, "structured_contents", None), list):
        sc = sample.structured_contents
        if len(sc) > max(keep_user_idx, default=-1):
            sample.structured_contents = [sc[i] for i in keep_user_idx if i < len(sc)]
    if isinstance(getattr(sample, "recap_texts", None), list):
        rt = sample.recap_texts
        if len(rt) > max(keep_user_idx, default=-1):
            sample.recap_texts = [rt[i] for i in keep_user_idx if i < len(rt)]

    # Update metadata.num_turns and trim trajectory entries that no longer
    # correspond to a rendered turn.
    md = sample.metadata or {}
    md["num_turns"] = len(keep_user_idx)
    cp = md.get("change_plan")
    if isinstance(cp, dict):
        traj = cp.get("intent_trajectory")
        if isinstance(traj, list) and len(traj) >= len(keep_user_idx):
            cp["intent_trajectory"] = [traj[i] for i in keep_user_idx if i < len(traj)]
        # transitions describe boundaries: between turn k and turn k+1.
        # Keep only transitions whose "post" turn (i.e., index k+1 in the
        # original numbering) is preserved. transitions[k] sits between
        # original turn k and turn k+1.
        tr = cp.get("transitions")
        if isinstance(tr, list):
            kept_set = set(keep_user_idx)
            cp["transitions"] = [
                tr[k] for k in range(len(tr))
                if (k + 1) in kept_set
            ]
    sample.metadata = md


__all__ = [
    "create_sample_swe",
    "distribute_ceil_front",
]
