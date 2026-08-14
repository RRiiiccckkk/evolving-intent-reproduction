"""
SWE-bench Verified Extractor implementation.
Extracts Function + Arguments from GitHub issue problem statements.

Key design decisions:
- Function: moderate specificity (function/module name + bug type)
- Arguments: extracted ONLY from problem_statement (no patch usage)
- Arguments are categorized (symptom, trigger, location, approach, scope, constraint)
- Counterfactual-eligibility is determined by category (approach/location/scope/constraint → counterfactual-eligible)
- Gold patch is used ONLY for post-extraction verification (alignment check)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from intent_construction.intent_extraction.core.base_extractor import BaseExtractor
from intent_construction.intent_extraction.core.llm_utils import generate_json, generate_text, load_prompt, populate_prompt


# Categories that are counterfactual-eligible for downstream Argument Change
COUNTERFACTUAL_ELIGIBLE_CATEGORIES = {"location", "approach", "scope", "constraint"}


def clean_problem_statement(text: str) -> str:
    """Strip GitHub issue boilerplate from problem statement."""
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove common boilerplate patterns
    boilerplate_patterns = [
        r'Please be sure to check out our contributing guidelines.*?(?=\n\n|\Z)',
        r'Please be sure to check out our code of conduct.*?(?=\n\n|\Z)',
        r'This comments are hidden when you submit the issue.*?(?=\n\n|\Z)',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_affected_files(patch: str) -> list[str]:
    """Extract affected file paths from a unified diff patch."""
    files = re.findall(r'^diff --git a/(.*?) b/', patch, re.MULTILINE)
    return sorted(set(files))


class SWEBenchVerifiedExtractor(BaseExtractor):
    """Extractor for SWE-bench Verified GitHub issues."""

    def __init__(
        self,
        model: str = "gpt-5.1",
        num_arguments: int = 4,
        max_verification_attempts: int = 5,
        verif_model: str = "gpt-5.1",
        enable_model_verification: bool = True
    ):
        super().__init__(
            model=model,
            num_arguments=num_arguments,
            max_verification_attempts=max_verification_attempts,
            verif_model=verif_model,
            enable_model_verification=enable_model_verification
        )

    def get_dataset_name(self) -> str:
        return "swe_bench_verified"

    def get_prompts_dir(self) -> Path:
        return Path(__file__).parent / "prompts"

    def _load_prompts(self) -> None:
        """Load SWE-bench-specific prompt templates."""
        prompts_dir = self.get_prompts_dir()
        self.prompt_decompose = load_prompt(prompts_dir / "segmentation.txt")
        self.prompt_conversational = load_prompt(prompts_dir / "conversational.txt")
        self.prompt_verification = load_prompt(prompts_dir / "verification.txt")
        self.prompt_patch_alignment = load_prompt(prompts_dir / "patch_alignment.txt")

    def decompose(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Decompose the problem_statement into function and arguments.
        Uses ONLY the problem_statement (no patch).
        """
        problem_statement = clean_problem_statement(sample["question"])

        prompt = populate_prompt(
            self.prompt_decompose,
            {"PROBLEM_STATEMENT": problem_statement}
        )

        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.model,
            step="extraction-decompose"
        )

        return {
            "function": result.get("function", ""),
            "arguments": result.get("arguments", [])
        }

    def to_conversational(
        self,
        sample: Dict[str, Any],
        decomposed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 2: Transform function + arguments into conversational format."""
        problem_statement = clean_problem_statement(sample["question"])

        prompt = populate_prompt(
            self.prompt_conversational,
            {
                "PROBLEM_STATEMENT": problem_statement,
                "GOAL": decomposed["function"],
                "CONDITIONS": json.dumps(decomposed["arguments"])
            }
        )

        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.model,
            step="extraction-conversational"
        )

        # Preserve category metadata from decompose step
        hints = result.get("hints", [])
        arguments = decomposed.get("arguments", [])
        for i, hint in enumerate(hints):
            if i < len(arguments):
                hint["category"] = arguments[i].get("category", "symptom")

        # Preserve the clean decompose function (not the conversational rephrasing)
        result["_decompose_function"] = decomposed["function"]

        return result

    def _build_extracted(self, conversational: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override base to preserve category metadata and the clean decompose function.
        The base _build_extracted drops category info from hints.
        """
        # Use the clean decompose function, not the conversational initial_query
        function = conversational.get("_decompose_function", conversational["initial_query"])
        arguments = []
        for i, hint in enumerate(conversational.get("hints", [])):
            arguments.append({
                "argument_id": i + 1,
                "argument": hint["hint"],
                "category": hint.get("category", "symptom"),
            })
        return {"function": function, "arguments": arguments}

    def verify_coverage(
        self,
        sample: Dict[str, Any],
        extracted: Dict[str, Any]
    ) -> bool:
        """Step 3: Verify extraction contains all essential issue information."""
        problem_statement = clean_problem_statement(sample["question"])

        arguments = [{"argument_id": 0, "argument": extracted["function"]}]
        for cond in extracted.get("arguments", []):
            arguments.append({
                "argument_id": cond["argument_id"],
                "argument": cond["argument"]
            })

        prompt = populate_prompt(
            self.prompt_verification,
            {
                "PROBLEM_STATEMENT": problem_statement,
                "CONDITIONS": json.dumps(arguments)
            }
        )

        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.model,
            step="extraction-verification"
        )

        return result.get("coverage", "incomplete") == "complete"

    def verify_solvability(
        self,
        sample: Dict[str, Any],
        extracted: Dict[str, Any]
    ) -> bool:
        """
        Step 4: Verify extraction is aligned with the gold patch.

        Unlike math domains (where we solve and compare answers), SWE-bench
        verification checks that the extracted function + arguments are consistent
        with the actual fix. The patch is used ONLY here, not during extraction.
        """
        patch = sample.get("patch", "")
        if not patch:
            # No patch available — skip alignment check
            return True

        arguments_str = json.dumps(extracted.get("arguments", []), indent=2)

        prompt = populate_prompt(
            self.prompt_patch_alignment,
            {
                "GOAL": extracted["function"],
                "CONDITIONS": arguments_str,
                "PATCH": patch[:3000]  # Truncate very large patches
            }
        )

        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.verif_model,
            step="extraction-patch-alignment"
        )

        aligned = result.get("aligned", False)
        if not aligned:
            reasoning = result.get("reasoning", "unknown")
            sample_id = sample.get("id", "unknown")
            print(f"  [Patch Alignment] ✗ Misaligned for {sample_id}: {reasoning[:100]}")

        return aligned

    def build_output(
        self,
        sample: Dict[str, Any],
        extracted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the final output format with SWE-bench metadata."""
        sample_id = sample.get("id", "unknown")
        instance_id = sample.get("instance_id", sample_id)
        split = sample.get("split", "test")

        # Enrich arguments with counterfactual-eligibility based on category
        arguments = []
        for cond in extracted.get("arguments", []):
            category = cond.get("category", "symptom")
            arguments.append({
                "argument_id": cond["argument_id"],
                "argument": cond["argument"],
                "category": category,
                "counterfactual_eligible": category in COUNTERFACTUAL_ELIGIBLE_CATEGORIES,
            })

        # Reconstruct fully_specified_question from function + arguments
        parts = [extracted["function"]]
        for cond in arguments:
            parts.append(cond["argument"])
        fully_specified = " ".join(parts)

        result = {
            "task_id": f"extracted-swe_bench_verified-{split}-{instance_id}",
            "original_id": instance_id,
            "task": "swe_bench",
            "question": sample.get("question", ""),
            "answer": "",  # SWE-bench uses test-suite verification, not answer matching
            "fully_specified_question": fully_specified,
            "function": extracted["function"],
            "arguments": arguments,
            "num_arguments": len(arguments),
            "model_name": self.model,
            # SWE-bench specific metadata
            "swe_bench_metadata": {
                "repo": sample.get("repo", ""),
                "base_commit": sample.get("base_commit", ""),
                "version": sample.get("version", ""),
                "difficulty": sample.get("difficulty", ""),
                "affected_files": parse_affected_files(sample.get("patch", "")),
                "patch_summary": "",  # Populated by verifier if needed
                "FAIL_TO_PASS": json.loads(sample.get("FAIL_TO_PASS", "[]"))
                    if isinstance(sample.get("FAIL_TO_PASS"), str)
                    else sample.get("FAIL_TO_PASS", []),
                "PASS_TO_PASS": json.loads(sample.get("PASS_TO_PASS", "[]"))
                    if isinstance(sample.get("PASS_TO_PASS"), str)
                    else sample.get("PASS_TO_PASS", []),
            },
        }

        return result
