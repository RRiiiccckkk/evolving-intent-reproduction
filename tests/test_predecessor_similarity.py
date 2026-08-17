from pathlib import Path

from intent_construction.retrospective_expansion.predecessor import generate_predecessors


def test_gsm8k_similarity_prompt_uses_goal_placeholders(monkeypatch):
    prompts_dir = (
        Path(generate_predecessors.__file__).parent
        / "prompts"
    )
    generator = generate_predecessors.PredecessorGenerator.__new__(
        generate_predecessors.PredecessorGenerator
    )
    generator.similarity_prompt_template = (
        prompts_dir / "similarity_check_gsm8k.txt"
    ).read_text()
    generator.judge_model = "offline-test-model"
    generator.temperature = 1.0
    generator.reasoning_effort = None

    captured = {}

    def fake_generate_json(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        captured["kwargs"] = kwargs
        return {"similar": False}

    monkeypatch.setattr(generate_predecessors, "generate_json", fake_generate_json)

    goal_a = "How many eggs remain?"
    goal_b = "How much money do the eggs earn?"
    assert generator._llm_similarity_check(goal_a, goal_b) is False
    assert f"Question A: {goal_a}" in captured["prompt"]
    assert f"Question B: {goal_b}" in captured["prompt"]
    assert "{goal_a}" not in captured["prompt"]
    assert "{goal_b}" not in captured["prompt"]
    assert captured["kwargs"]["model"] == "offline-test-model"
    assert captured["kwargs"]["max_retries"] == 1
    assert "max_tokens" not in captured["kwargs"]


def test_failed_cross_turn_verification_stops_before_independence(monkeypatch):
    generator = generate_predecessors.PredecessorGenerator.__new__(
        generate_predecessors.PredecessorGenerator
    )
    generator.num_predecessors = 3
    generator.max_verify_attempts = 2
    generator._verifier = object()
    generator._extract_answer_keywords = lambda _answer: set()

    generated = []

    def fake_generate_chain(**_kwargs):
        chain = [{"full_arguments": []} for _ in range(3)]
        generated.append(chain)
        return chain

    generator._generate_chain = fake_generate_chain
    generator._verify_chain = lambda **_kwargs: {
        "passed": False,
        "details": "cross-turn relevance failed",
    }
    generator._verify_functional_independence = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("independence must not run after cross-turn failure")
    )

    result = generator.generate_predecessors(
        {
            "task_id": "canary",
            "function": "find an entity",
            "arguments": [{"argument_id": 1, "argument": "constraint"}],
            "answer": "entity",
        }
    )

    assert result is None
    assert len(generated) == 2
