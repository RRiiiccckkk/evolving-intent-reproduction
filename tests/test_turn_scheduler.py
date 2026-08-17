from situated_simulation.turn_scheduler import create_sample, schedule_events
from situated_simulation.sql_partial import execute_sql, substitute_sql_value
from situated_simulation.turn_scheduler_swe import create_sample_swe
from situated_simulation.user_simulation import _join_prefix_content


def _events_in_order(slots):
    return [event for slot in slots for event in slot.events]


def test_sql_value_substitution_escapes_apostrophes():
    assert substitute_sql_value(
        "SELECT 1 WHERE title = 'Original'",
        "Original",
        "Someone Else's Happiness",
    ) == "SELECT 1 WHERE title = 'Someone Else''s Happiness'"
    assert substitute_sql_value(
        "SELECT 1 WHERE brand = 'Barq''s'",
        "Barq's",
        "A&W",
    ) == "SELECT 1 WHERE brand = 'A&W'"


def test_partial_sql_execution_enforces_query_timeout(tmp_path):
    database = tmp_path / "timeout.sqlite"
    database.touch()
    result = execute_sql(
        database,
        """
        WITH RECURSIVE counter(value) AS (
            SELECT 1
            UNION ALL
            SELECT value + 1 FROM counter WHERE value < 100000000
        )
        SELECT SUM(value) FROM counter
        """,
        timeout=0.001,
    )

    assert result.success is False
    assert "interrupted" in (result.error or "").lower()


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


def test_swe_evolve_uses_two_real_reveals_across_exactly_seven_turns():
    raw = {
        "task_id": "swe-seven-turn-regression",
        "function": "Fix target behavior",
        "answer": "",
        "arguments": [
            {
                "argument_id": 1,
                "argument": "The current behavior fails.",
                "category": "symptom",
                "counterfactual_eligible": False,
                "counterfactual_arguments": [],
            },
            {
                "argument_id": 2,
                "argument": "Use the correct implementation.",
                "category": "approach",
                "counterfactual_eligible": True,
                "counterfactual_arguments": [
                    {"counterfactual_argument": "Use the old implementation."}
                ],
            },
            {
                "argument_id": 3,
                "argument": "Preserve compatibility.",
                "category": "constraint",
                "counterfactual_eligible": True,
                "counterfactual_arguments": [
                    {"counterfactual_argument": "Drop compatibility."}
                ],
            },
        ],
        "predecessor_functions": [
            {
                "predecessor_function": "Plan a related feature",
                "counterfactual_arguments": [
                    {
                        "argument_id": 2001 + index,
                        "argument": f"Feature requirement {index + 1}.",
                        "category": "constraint",
                        "is_shared": False,
                    }
                    for index in range(4)
                ],
                "is_predecessor": True,
                "taxonomy_type": "parallel_subsystem",
                "transition_type": "impl_precursor",
                "transition_phrase": "Switch to the prerequisite fix.",
            },
            {
                "predecessor_function": "Explain the package",
                "counterfactual_arguments": [],
                "is_predecessor": False,
                "taxonomy_type": "dependency_map",
                "transition_type": "exploration",
            },
        ],
    }

    sample = create_sample_swe(
        raw,
        g=2,
        p=2,
        t=7,
        mode="eval",
        domain="swe_bench_verified",
        get_function_change_prefix=lambda **_: "Switch:",
        get_correction_prefix=lambda **_: "Correction:",
        get_reveal_prefix=lambda **_: "More:",
        get_reveal_after_function_prefix=lambda **_: "Details:",
        get_corr_after_reveal_prefix=lambda **_: "Correction:",
        get_new_info_prefix=lambda **_: "Also:",
        join_prefix_content=_join_prefix_content,
    )

    user_turns = [turn for turn in sample.turns if turn["role"] == "user"]
    transitions = sample.metadata["change_plan"]["transitions"]

    assert len(user_turns) == 7
    assert all(turn["content"].strip() for turn in user_turns)
    assert [transition["type"] for transition in transitions] == [
        "function_change",
        "function_change",
        "argument_reveal",
        "argument_change",
        "argument_reveal",
        "argument_change",
    ]
    assert transitions[2]["revealed_ids"] == [2]
    assert transitions[4]["revealed_ids"] == [3]
    assert "use the old implementation." in user_turns[3]["content"]
    assert "use the correct implementation." in user_turns[4]["content"]
    assert user_turns[3]["content"] != user_turns[4]["content"]


def test_sql_combined_schedule_keeps_all_seven_requested_turns():
    raw = {
        "task_id": "bird-seven-turn-regression",
        "data_source": "bird_sql",
        "function": "How many matching records are there?",
        "answer": "1",
        "arguments": [
            {
                "argument_id": 1,
                "argument": "The status is active.",
                "counterfactual_arguments": [
                    {"counterfactual_argument": "The status is inactive."}
                ],
            },
            {
                "argument_id": 2,
                "argument": "The region is west.",
                "counterfactual_arguments": [
                    {"counterfactual_argument": "The region is east."}
                ],
            },
            {
                "argument_id": 3,
                "argument": "The year is 2025.",
                "counterfactual_arguments": [],
            },
        ],
        "predecessor_functions": [
            {
                "predecessor_function": "Which records match?",
                "counterfactual_arguments": [],
                "is_predecessor": True,
            },
            {
                "predecessor_function": "Which identifiers match?",
                "counterfactual_arguments": [],
                "is_predecessor": True,
            },
        ],
    }

    sample = create_sample(
        raw,
        g=2,
        p=2,
        t=7,
        mode="eval",
        domain="sql",
        get_function_change_prefix=lambda **_: "Switch:",
        get_correction_prefix=lambda **_: "Correction:",
        get_reveal_prefix=lambda **_: "More:",
        get_reveal_after_function_prefix=lambda **_: "Details:",
        get_corr_after_reveal_prefix=lambda **_: "Correction:",
        get_new_info_prefix=lambda **_: "Also:",
        join_prefix_content=_join_prefix_content,
    )

    user_turns = [turn for turn in sample.turns if turn["role"] == "user"]
    transitions = sample.metadata["change_plan"]["transitions"]

    assert len(user_turns) == 7
    assert all(turn["content"].strip() for turn in user_turns)
    assert [transition["type"] for transition in transitions] == [
        "function_change",
        "function_change",
        "argument_reveal",
        "argument_change",
        "argument_reveal",
        "argument_change",
    ]
    assert transitions[2]["revealed_ids"] == [3]
    assert transitions[4]["revealed_ids"] == [2]
    assert "the region is east." in user_turns[5]["content"].lower()
    assert "the region is west." in user_turns[6]["content"].lower()
