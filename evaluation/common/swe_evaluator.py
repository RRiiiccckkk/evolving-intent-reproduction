"""
SWE-bench Verified evaluator + patch extractor.

Public surface:
    extract_patch(response: str) -> str           # always returns a string ("" if none)
    swe_system_prompt() -> str                    # single-turn system prompt
    SWEEvalResult                                 # per-sample result dataclass
    evaluate_swe_response(...)                    # high-level scoring entry

Design:
- The extractor is a single-pass priority cascade. Each rule operates on the
  full response and returns either a patch string or None. The first non-None
  wins. Adding a new rule means appending to the list, not nesting if/else.
- The evaluator is a thin adapter: extract patch -> hand to swe_harness ->
  pack the result into SWEEvalResult. No Docker / harness logic lives here.
- This module is import-cheap (does NOT import swebench at module load) so
  it can be safely registered alongside math/IF/SQL evaluators.

Patch extraction priority:
    1. Fenced ```diff / ```patch blocks (last one wins; LLMs sometimes
       narrate then fence the final patch).
    2. Fenced ``` ... ``` blocks whose content starts with `diff --git`.
    3. Unfenced contiguous patch block starting at `diff --git`.
    4. Unfenced minimal diff starting at `--- a/` (no `diff --git` header).

A response with no extractable patch is treated as a clean failure
(SWEEvalResult.correct = False, harness_error = "no_patch_extracted") —
the harness is never invoked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

from .swe_harness import HarnessResult, SWEHarness, get_default_harness

logger = logging.getLogger(__name__)


# =============================================================================
# Patch extraction
# =============================================================================


_FENCED_DIFF_RE = re.compile(
    r"```(?:diff|patch)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_FENCED_ANY_RE = re.compile(
    r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```",
    re.DOTALL,
)

_UNFENCED_DIFF_GIT_RE = re.compile(
    r"(?:^|\n)(diff --git .*?)(?=\n```|\n\n[A-Z]|\Z)",
    re.DOTALL,
)

_UNFENCED_MINIMAL_RE = re.compile(
    r"(?:^|\n)(--- a/.+?\n\+\+\+ b/.+?\n@@.+?)(?=\n```|\n\n[A-Z]|\Z)",
    re.DOTALL,
)


def _normalize(patch: str) -> str:
    """Light normalization for round-tripping through git apply."""
    if not patch:
        return ""
    patch = patch.strip()
    # Ensure trailing newline (git apply is strict about this).
    if not patch.endswith("\n"):
        patch += "\n"
    return patch


def _looks_like_unified_diff(text: str) -> bool:
    """Heuristic check: must contain at least one hunk and a file header."""
    if not text:
        return False
    has_header = bool(re.search(r"^(?:diff --git |--- a/)", text, re.MULTILINE))
    has_hunk = "@@" in text
    return has_header and has_hunk


# Each rule returns the *raw* patch text (caller normalizes), or None.
def _rule_fenced_diff(response: str) -> str | None:
    matches = _FENCED_DIFF_RE.findall(response)
    if not matches:
        return None
    # Last fenced block wins (model often discusses then commits).
    candidate = matches[-1]
    return candidate if _looks_like_unified_diff(candidate) else None


def _rule_fenced_any(response: str) -> str | None:
    for candidate in reversed(_FENCED_ANY_RE.findall(response)):
        stripped = candidate.lstrip()
        if stripped.startswith("diff --git") or stripped.startswith("--- a/"):
            if _looks_like_unified_diff(candidate):
                return candidate
    return None


def _rule_unfenced_diff_git(response: str) -> str | None:
    matches = _UNFENCED_DIFF_GIT_RE.findall(response)
    if not matches:
        return None
    candidate = matches[-1]
    return candidate if _looks_like_unified_diff(candidate) else None


def _rule_unfenced_minimal(response: str) -> str | None:
    matches = _UNFENCED_MINIMAL_RE.findall(response)
    if not matches:
        return None
    candidate = matches[-1]
    return candidate if _looks_like_unified_diff(candidate) else None


_EXTRACTION_RULES: tuple[Callable[[str], str | None], ...] = (
    _rule_fenced_diff,
    _rule_fenced_any,
    _rule_unfenced_diff_git,
    _rule_unfenced_minimal,
)


def extract_patch(response: str | None) -> str:
    """Return a unified-diff string, or '' if none found.

    Never raises. Idempotent: extract_patch(extract_patch(x)) == extract_patch(x)
    when x already is a clean patch.
    """
    if not response:
        return ""

    # Idempotence shortcut.
    if response.lstrip().startswith(("diff --git ", "--- a/")) and _looks_like_unified_diff(response):
        return _normalize(response)

    for rule in _EXTRACTION_RULES:
        candidate = rule(response)
        if candidate:
            return _normalize(candidate)

    return ""


# =============================================================================
# Prompt
# =============================================================================


_SYSTEM_PROMPT = (
    "You are an expert software engineer working on the SWE-bench Verified benchmark. "
    "The user will describe a bug or change request from a real GitHub issue, possibly "
    "across multiple turns and possibly with corrections to earlier statements. Your job "
    "is to produce a patch that resolves the *final* version of the request.\n"
    "\n"
    "Output requirements:\n"
    "- Reply with a single unified-diff patch in a fenced ```diff code block.\n"
    "- The patch must be applicable with `git apply` against the project's repository at "
    "its base commit.\n"
    "- Use standard `diff --git a/<path> b/<path>` headers and `@@` hunk markers.\n"
    "- Modify only the production source files needed to resolve the issue. Do NOT modify "
    "test files; the test suite will be run separately.\n"
    "- Keep the patch minimal: change only the lines needed for the fix.\n"
    "- If you cannot determine a fix, still emit the closest reasonable patch you can.\n"
    "\n"
    "Do not include explanations outside the diff block. The diff is the only thing that "
    "will be evaluated."
)


def swe_system_prompt() -> str:
    return _SYSTEM_PROMPT


# =============================================================================
# Result + scoring
# =============================================================================


@dataclass
class SWEEvalResult:
    """Per-sample evaluation result for SWE."""

    instance_id: str
    correct: bool
    patch: str
    patch_extracted: bool
    patch_apply_ok: bool | None
    ftp_pass: list[str] = field(default_factory=list)
    ftp_fail: list[str] = field(default_factory=list)
    ptp_pass: list[str] = field(default_factory=list)
    ptp_fail: list[str] = field(default_factory=list)
    harness_error: str | None = None
    duration_s: float = 0.0
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_harness(cls, hr: HarnessResult, patch: str) -> SWEEvalResult:
        return cls(
            instance_id=hr.instance_id,
            correct=hr.resolved,
            patch=patch,
            patch_extracted=hr.patch_extracted,
            patch_apply_ok=hr.patch_apply_ok,
            ftp_pass=hr.ftp_pass,
            ftp_fail=hr.ftp_fail,
            ptp_pass=hr.ptp_pass,
            ptp_fail=hr.ptp_fail,
            harness_error=hr.harness_error,
            duration_s=hr.duration_s,
            from_cache=hr.from_cache,
        )


def evaluate_swe_response(
    response: str | None,
    instance_id: str,
    model_name: str = "evolvingintent",
    harness: SWEHarness | None = None,
) -> SWEEvalResult:
    """Extract patch from `response`, score against `instance_id` via harness."""
    patch = extract_patch(response)
    if not patch:
        return SWEEvalResult(
            instance_id=instance_id,
            correct=False,
            patch="",
            patch_extracted=False,
            patch_apply_ok=False,
            harness_error="no_patch_extracted",
        )
    h = harness if harness is not None else get_default_harness()
    hr = h.verify_patch(instance_id=instance_id, patch=patch, model_name=model_name)
    return SWEEvalResult.from_harness(hr, patch=patch)


# =============================================================================
# Domain detection
# =============================================================================


def is_swe_dataset(metadata: dict[str, Any] | None) -> bool:
    """Match the same heuristic used elsewhere (run_experiment evaluate_sample)."""
    if not metadata:
        return False
    if metadata.get("task") in ("swe_bench", "swe_bench_verified"):
        return True
    ds = metadata.get("data_source") or ""
    return "swe" in ds.lower()


def instance_id_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Pull the SWE-bench instance_id (e.g. 'django__django-12273') out of sample metadata.

    Our simulator stores it as `original_id` in the sample row; some downstream
    paths might also surface it as `instance_id` directly. Tolerate both.
    """
    if not metadata:
        return None
    for key in ("original_id", "instance_id"):
        v = metadata.get(key)
        if isinstance(v, str) and v:
            return v
    return None
