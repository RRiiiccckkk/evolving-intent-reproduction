from situated_simulation.turn_scheduler import schedule_events


def _events_in_order(slots):
    return [event for slot in slots for event in slot.events]


def test_function_switch_schedule_starts_at_first_goal_and_ends_at_source_goal():
    slots = schedule_events(
        g=2,
        p=0,
        t=5,
        correction_specs=[],
        deadlines={},
    )

    assert len(slots) == 5
    assert [slot.turn_idx for slot in slots] == list(range(5))

    events = _events_in_order(slots)
    assert [(event.type, event.function_idx) for event in events] == [
        ("function_init", 0),
        ("function_change", 1),
        ("function_change", -1),
    ]
    assert slots[-1].events[0].function_idx == -1


def test_combined_schedule_preserves_correction_deadlines_and_event_counts():
    slots = schedule_events(
        g=2,
        p=2,
        t=5,
        correction_specs=[(10, 0), (20, 0)],
        deadlines={10: 1, 20: 3},
    )

    events = _events_in_order(slots)
    changes = [event for event in events if event.type == "function_change"]
    corrections = [event for event in events if event.type == "correction"]

    assert len(changes) == 2
    assert len(corrections) == 2
    assert [(event.cond_id, event.corr_step) for event in corrections] == [
        (10, 0),
        (20, 0),
    ]

    shared_correction = next(
        i for i, event in enumerate(events)
        if event.type == "correction" and event.cond_id == 10
    )
    first_change = next(
        i for i, event in enumerate(events)
        if event.type == "function_change" and event.function_idx == 1
    )
    source_change = next(
        i for i, event in enumerate(events)
        if event.type == "function_change" and event.function_idx == -1
    )
    post_source_correction = next(
        i for i, event in enumerate(events)
        if event.type == "correction" and event.cond_id == 20
    )

    assert shared_correction < first_change
    assert source_change < post_source_correction


def test_argument_revision_keeps_multistep_corrections_in_order():
    correction_specs = [(3, 0), (3, 1), (7, 0)]
    slots = schedule_events(
        g=0,
        p=len(correction_specs),
        t=5,
        correction_specs=correction_specs,
        deadlines={},
    )

    events = _events_in_order(slots)
    assert events[0].type == "function_init"
    assert [
        (event.cond_id, event.corr_step)
        for event in events
        if event.type == "correction"
    ] == correction_specs
