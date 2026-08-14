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

    captured = {}

    def fake_generate_text(messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        captured["kwargs"] = kwargs
        return "DIFFERENT"

    monkeypatch.setattr(generate_predecessors, "generate_text", fake_generate_text)

    goal_a = "How many eggs remain?"
    goal_b = "How much money do the eggs earn?"
    assert generator._llm_similarity_check(goal_a, goal_b) is False
    assert f"Question A: {goal_a}" in captured["prompt"]
    assert f"Question B: {goal_b}" in captured["prompt"]
    assert "{goal_a}" not in captured["prompt"]
    assert "{goal_b}" not in captured["prompt"]
    assert captured["kwargs"]["model"] == "offline-test-model"
