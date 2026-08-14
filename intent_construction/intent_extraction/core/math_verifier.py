"""
General-purpose math verifier.

Used for answer-equivalence checking across math datasets (e.g. GSM8K) and by
the predecessor-generation functional-independence test. Relies on the `math-verify`
library for robust mathematical equivalence, with a string-normalization
fallback.
"""

import re
from typing import Dict, List, Any, Optional

from intent_construction.intent_extraction.core.base_verifier import BaseVerifier
from intent_construction.intent_extraction.core.llm_utils import generate_text

# Import math-verify for robust mathematical equivalence checking
try:
    from math_verify import verify as math_verify_check, parse as math_verify_parse
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math-verify not installed. Using fallback string comparison.")


class MathVerifier(BaseVerifier):
    """General math verifier (boxed / LaTeX / numeric answers)."""

    def __init__(
        self,
        model: str = "gpt-5.1",
        num_runs: int = 1
    ):
        super().__init__(model=model, num_runs=num_runs)

        self.system_prompt = """You are a mathematics expert solving competition-level problems. 
Solve the problem step by step, showing your work clearly.
At the end, provide your final answer in the format:
Answer: <your_answer>

Where <your_answer> is the numerical or mathematical answer to the problem."""

    def get_dataset_name(self) -> str:
        return "math"

    def verify_coverage(
        self, 
        original: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        return True

    def verify_solvability(
        self, 
        original: Dict[str, Any], 
        extracted: Dict[str, Any],
        ground_truth: str
    ) -> bool:
        # Build concat prompt from function + arguments
        texts = [extracted.get("function", "")]
        for cond in sorted(extracted.get("arguments", []), key=lambda x: x.get("argument_id", 0)):
            texts.append(cond.get("argument", ""))
        concat_prompt = " ".join(texts)

        correct_count = 0

        for _ in range(self.num_runs):
            response = generate_text(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": concat_prompt}],
                model=self.model,
                temperature=0.7,
                max_tokens=2000
            )

            answer = self.extract_answer(response)
            if self.evaluate_answer(answer, ground_truth):
                correct_count += 1

        return correct_count >= (self.num_runs // 2 + 1)

    def extract_answer(self, response: str) -> Optional[str]:
        if response is None:
            return None

        patterns = [
            r"Answer:\s*\$?\\?boxed\{([^}]+)\}",
            r"Answer:\s*\$([^$]+)\$",
            r"Answer:\s*([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        lines = response.strip().split('\n')
        if lines:
            return lines[-1].strip()

        return None

    def evaluate_answer(self, predicted: Optional[str], ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        Uses math-verify for robust mathematical equivalence checking.
        Falls back to string normalization if math-verify fails.
        """
        if predicted is None:
            return False

        # Try math-verify first (robust mathematical equivalence)
        if MATH_VERIFY_AVAILABLE:
            try:
                pred_parsed = math_verify_parse(predicted)
                gt_parsed = math_verify_parse(ground_truth)
                if math_verify_check(pred_parsed, gt_parsed):
                    return True
            except Exception:
                # math-verify failed, fall through to string comparison
                pass

        # Fallback: string normalization comparison
        pred_norm = self._normalize_answer(predicted)
        gt_norm = self._normalize_answer(ground_truth)

        return pred_norm == gt_norm

    def _normalize_answer(self, answer: str) -> str:
        if answer is None:
            return ""

        ans = answer.strip().lower()
        ans = re.sub(r'\\boxed\{([^}]+)\}', r'\1', ans)
        ans = re.sub(r'\$([^$]+)\$', r'\1', ans)
        ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', ans)
        ans = re.sub(r'[,\s\\$]', '', ans)

        return ans
