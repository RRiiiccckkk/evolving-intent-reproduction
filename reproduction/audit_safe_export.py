#!/usr/bin/env python3
"""Fail closed when a Git index contains private BrowseComp+ material."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from reproduction.run_with_cc_switch import DEFAULT_CC_SWITCH_DB, load_key


GENERIC_SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}"),
)
FORBIDDEN_PATH_PREFIXES = (
    "reproduction/runs/browsecomp-plan-a-n100/",
    "evaluation/logs/browsecomp/",
    "final_dataset/browsecomp",
)


class ExportAuditError(RuntimeError):
    """Raised when the staged export cannot be audited safely."""


def _run_git(repo_root: Path, arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise ExportAuditError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def read_index(repo_root: Path) -> dict[str, bytes]:
    """Read the exact Git index, not ignored or unstaged worktree files."""
    raw_paths = _run_git(repo_root, ["ls-files", "-z", "--cached"])
    paths = [
        value.decode("utf-8", errors="surrogateescape")
        for value in raw_paths.split(b"\0")
        if value
    ]
    return {
        path: _run_git(repo_root, ["show", f":{path}"])
        for path in paths
    }


def _private_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExportAuditError(f"private BrowseComp+ source is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExportAuditError(
                    f"private source has invalid JSON on line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ExportAuditError(
                    f"private source line {line_number} is not an object"
                )
            rows.append(row)
    if not rows:
        raise ExportAuditError("private BrowseComp+ source is empty")
    return rows


def _encoded_forms(value: str) -> set[bytes]:
    forms = {value.encode("utf-8")}
    escaped = json.dumps(value, ensure_ascii=False)[1:-1].encode("utf-8")
    forms.add(escaped)
    return {
        form
        for form in forms
        if len(form) >= 32 and b"\n" not in form and b"\r" not in form
    }


def _document_fragments(value: str, width: int = 256) -> set[bytes]:
    if len(value) <= width:
        return _encoded_forms(value)
    limit = len(value) - width
    offsets = {0, limit // 4, limit // 2, (3 * limit) // 4, limit}
    fragments: set[bytes] = set()
    for offset in offsets:
        fragments.update(_encoded_forms(value[offset : offset + width]))
    return fragments


def private_markers(path: Path) -> tuple[set[bytes], set[bytes], int]:
    queries: set[bytes] = set()
    documents: set[bytes] = set()
    rows = _private_rows(path)
    for row in rows:
        query = row.get("query")
        if isinstance(query, str):
            queries.update(_encoded_forms(query))
        gold_docs = row.get("gold_docs")
        if not isinstance(gold_docs, list):
            continue
        for document in gold_docs:
            text = document.get("text") if isinstance(document, dict) else document
            if isinstance(text, str):
                documents.update(_document_fragments(text))
    if not queries:
        raise ExportAuditError("private source contains no auditable queries")
    if not documents:
        raise ExportAuditError("private source contains no auditable gold documents")
    return queries, documents, len(rows)


def _git_grep_markers(repo_root: Path, markers: set[bytes]) -> set[str]:
    """Use Git's fixed-string matcher against the exact staged index."""
    if not markers:
        raise ExportAuditError("cannot scan an empty private marker set")
    result = subprocess.run(
        ["git", "grep", "--cached", "-I", "-z", "-l", "-F", "-f", "-", "--"],
        cwd=repo_root,
        input=b"\n".join(sorted(markers)) + b"\n",
        check=False,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise ExportAuditError(f"git grep private-material scan failed: {detail}")
    return {
        value.decode("utf-8", errors="surrogateescape")
        for value in result.stdout.split(b"\0")
        if value
    }


def _path_violation(path: str) -> str | None:
    if path.startswith(FORBIDDEN_PATH_PREFIXES):
        return "private_runtime_path"
    if "/browsecomp_plus_n100/" in path and not path.endswith(".compact.json"):
        return "raw_browsecomp_result_path"
    return None


def audit_index(
    *,
    repo_root: Path,
    private_source: Path,
    api_key: str,
) -> dict[str, Any]:
    indexed = read_index(repo_root.resolve())
    queries, documents, row_count = private_markers(private_source.resolve())
    query_paths = _git_grep_markers(repo_root.resolve(), queries)
    document_paths = _git_grep_markers(repo_root.resolve(), documents)
    key = api_key.encode("utf-8")
    if len(key) < 16:
        raise ExportAuditError("API key is too short for an exact-material audit")

    violations: list[dict[str, str]] = []
    for path, content in indexed.items():
        path_category = _path_violation(path)
        if path_category is not None:
            violations.append({"category": path_category, "path": path})
        if key in content:
            violations.append({"category": "credential_exact", "path": path})
        if any(pattern.search(content) for pattern in GENERIC_SECRET_PATTERNS):
            violations.append({"category": "credential_pattern", "path": path})
        if path in query_paths:
            violations.append({"category": "plaintext_query", "path": path})
        if path in document_paths:
            violations.append({"category": "gold_document", "path": path})

    return {
        "status": "pass" if not violations else "fail",
        "tracked_files": len(indexed),
        "tracked_bytes": sum(len(content) for content in indexed.values()),
        "private_rows": row_count,
        "query_markers": len(queries),
        "gold_document_markers": len(documents),
        "violations": violations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--private-source",
        type=Path,
        default=(
            Path(__file__).parents[1]
            / "reproduction/runs/browsecomp-plan-a-n100/private/"
            "raw_data_with_gold_docs.jsonl"
        ),
    )
    parser.add_argument(
        "--cc-switch-db",
        type=Path,
        default=DEFAULT_CC_SWITCH_DB,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = audit_index(
            repo_root=args.repo_root,
            private_source=args.private_source,
            api_key=load_key(args.cc_switch_db),
        )
    except (ExportAuditError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
