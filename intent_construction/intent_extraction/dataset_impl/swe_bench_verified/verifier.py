"""
SWE-bench Verified Verifier implementation.

Key differences from math/IF verifiers:
- No answer-matching solvability check (SWE-bench answers are code patches)
- Coverage verification: same LLM-based check as other domains
- Patch alignment: verifies extracted function+arguments are consistent with actual fix
- No extract_answer/evaluate_answer (not applicable for code patches)
"""

import json
from typing import Dict, Any, Optional

from intent_construction.intent_extraction.core.base_verifier import BaseVerifier
from intent_construction.intent_extraction.core.llm_utils import generate_json, load_prompt, populate_prompt


class SWEBenchVerifiedVerifier(BaseVerifier):
    """Verifier for SWE-bench Verified dataset."""

    def __init__(self, model: str = "gpt-5.1", num_runs: int = 1):
        super().__init__(model=model, num_runs=num_runs)

    def get_dataset_name(self) -> str:
        return "swe_bench_verified"

    def verify_coverage(
        self,
        original: Dict[str, Any],
        extracted: Dict[str, Any]
    ) -> bool:
        """Verify that extracted function + arguments cover all essential issue info."""
        from pathlib import Path

        prompts_dir = Path(__file__).parent / "prompts"
        prompt_template = load_prompt(prompts_dir / "verification.txt")

        problem_statement = original.get("question", "")

        arguments = [{"argument_id": 0, "argument": extracted["function"]}]
        for cond in extracted.get("arguments", []):
            arguments.append({
                "argument_id": cond["argument_id"],
                "argument": cond["argument"]
            })

        prompt = populate_prompt(
            prompt_template,
            {
                "PROBLEM_STATEMENT": problem_statement,
                "CONDITIONS": json.dumps(arguments)
            }
        )

        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.model,
            step="verification-coverage"
        )

        return result.get("coverage", "incomplete") == "complete"

    def verify_solvability(
        self,
        original: Dict[str, Any],
        extracted: Dict[str, Any],
        ground_truth: str = ""
    ) -> bool:
        """
        Verify extraction is aligned with the gold patch.

        For SWE-bench, 'solvability' means the extraction is consistent with
        the actual fix. We don't attempt to solve the problem — we verify that
        the function and arguments accurately describe what was fixed.
        """
        from pathlib import Path

        patch = original.get("patch", "")
        if not patch:
            return True

        prompts_dir = Path(__file__).parent / "prompts"
        prompt_template = load_prompt(prompts_dir / "patch_alignment.txt")

        arguments_str = json.dumps(extracted.get("arguments", []), indent=2)

        prompt = populate_prompt(
            prompt_template,
            {
                "GOAL": extracted.get("function", ""),
                "CONDITIONS": arguments_str,
                "PATCH": patch[:3000]
            }
        )

        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.model,
            step="verification-patch-alignment"
        )

        return result.get("aligned", False)

    def extract_answer(self, response: str) -> Optional[str]:
        """Not applicable for SWE-bench (answers are code patches, not text)."""
        return None

    def evaluate_answer(self, predicted: Optional[str], ground_truth: str) -> bool:
        """Not applicable for SWE-bench (evaluation is via test suite)."""
        return False
