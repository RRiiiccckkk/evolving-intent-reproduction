from __future__ import annotations

import json
import subprocess
from pathlib import Path

from reproduction.audit_safe_export import audit_index


SAFE_TEST_KEY = "sk-" + "safe-test-key-" + "1234567890abcdef"


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def _private_source(path: Path) -> tuple[str, str]:
    query = "Which exact private benchmark entity satisfies this deliberately unique clue set?"
    document = (
        "PRIVATE GOLD DOCUMENT "
        + "This evidence must never enter the repository index. " * 20
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "query": query,
                "answer": "private answer",
                "gold_docs": [{"text": document}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return query, document


def _repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    private = tmp_path / "private.jsonl"
    query, document = _private_source(private)
    return repo, private, query, document


def _audit(repo: Path, private: Path, key: str = SAFE_TEST_KEY):
    return audit_index(repo_root=repo, private_source=private, api_key=key)


def test_safe_export_audit_accepts_aggregate_only_index(tmp_path: Path) -> None:
    repo, private, _, _ = _repo(tmp_path)
    (repo / "report.json").write_text('{"accuracy": 0.5}\n', encoding="utf-8")
    _git(repo, "add", "report.json")

    report = _audit(repo, private)

    assert report["status"] == "pass"
    assert report["private_rows"] == 1
    assert report["violations"] == []


def test_safe_export_audit_rejects_plaintext_query(tmp_path: Path) -> None:
    repo, private, query, _ = _repo(tmp_path)
    (repo / "raw.json").write_text(json.dumps({"query": query}), encoding="utf-8")
    _git(repo, "add", "raw.json")

    report = _audit(repo, private)

    assert report["status"] == "fail"
    assert {item["category"] for item in report["violations"]} == {
        "plaintext_query"
    }


def test_safe_export_audit_rejects_gold_document_and_key(tmp_path: Path) -> None:
    repo, private, _, document = _repo(tmp_path)
    key = "sk-" + "private-key-" + "1234567890abcdef"
    (repo / "leak.txt").write_text(document[:256] + "\n" + key, encoding="utf-8")
    _git(repo, "add", "leak.txt")

    report = _audit(repo, private, key)
    categories = {item["category"] for item in report["violations"]}

    assert report["status"] == "fail"
    assert "gold_document" in categories
    assert "credential_exact" in categories
    assert "credential_pattern" in categories


def test_safe_export_audit_rejects_private_result_path(tmp_path: Path) -> None:
    repo, private, _, _ = _repo(tmp_path)
    raw = repo / "evaluation/experiments/fully_specified/browsecomp_plus_n100/raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", str(raw.relative_to(repo)))

    report = _audit(repo, private)

    assert report["status"] == "fail"
    assert any(
        item["category"] == "raw_browsecomp_result_path"
        for item in report["violations"]
    )


def test_repository_ignores_raw_browsecomp_results() -> None:
    root = Path(__file__).resolve().parents[1]
    candidates = (
        "evaluation/experiments/fully_specified/browsecomp_plus_n100/raw.json",
        "evaluation/experiments/combined_independent/browsecomp_plus_n100/raw.json",
        "evaluation/logs/browsecomp/run.log",
    )
    for candidate in candidates:
        result = subprocess.run(
            ["git", "check-ignore", "-q", candidate],
            cwd=root,
            check=False,
        )
        assert result.returncode == 0, candidate
