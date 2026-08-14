"""
GSM8k Verifier implementation.
"""

import re
from typing import Dict, List, Any, Optional

from intent_construction.intent_extraction.core.base_verifier import BaseVerifier
from intent_construction.intent_extraction.core.llm_utils import generate_text


class GSM8kVerifier(BaseVerifier):
    """Verifier for GSM8k math word problems."""
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        num_runs: int = 1
    ):
        super().__init__(model=model, num_runs=num_runs)
        
        self.system_prompt = """You are helping the user solve a math problem. If you propose a numerical solution, you should provide it at the end of your response in the following format:
```answer
<numerical_answer>
```
Where <numerical_answer> is the numerical answer to the problem."""
    
    def get_dataset_name(self) -> str:
        return "gsm8k"
    
    def verify_coverage(
        self, 
        original: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        """
        Verify that extracted function + arguments contain all essential information.
        For GSM8k, this is handled during extraction (Step 3).
        """
        # GSM8k uses LLM-based verification in the extractor
        # This method is for standalone verification if needed
        return True
    
    def verify_solvability(
        self, 
        original: Dict[str, Any], 
        extracted: Dict[str, Any],
        ground_truth: str
    ) -> bool:
        """Verify that model can solve using extracted function + arguments."""
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
                max_tokens=1000
            )
            
            answer = self.extract_answer(response)
            if self.evaluate_answer(answer, ground_truth):
                correct_count += 1
        
        # Pass if majority of runs are correct
        return correct_count >= (self.num_runs // 2 + 1)
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from model response."""
        try:
            # Try ```answer format first
            answer = re.findall(r"```answer\s*(.*?)\s*```", response, re.DOTALL)[-1]
            return answer.strip()
        except:
            # Fallback: try #### format
            try:
                if "####" in response:
                    answer = response.split("####")[-1].strip()
                    return answer.split()[0] if answer.split() else answer
            except:
                pass
            return None
    
    def evaluate_answer(self, predicted: Optional[str], ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        if predicted is None:
            return False
        
        # Extract ground truth
        gold = self._extract_ground_truth(ground_truth).lower()
        
        # Normalize both answers
        regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]
        
        try:
            extracted_answer = predicted.strip()
            extracted_answer = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", extracted_answer)[-1]
            extracted_answer = extracted_answer[0] if extracted_answer[0] else extracted_answer[1]
            extracted_answer = extracted_answer.lower()
        except:
            extracted_answer = predicted.strip().lower()
        
        for regex in regexes_to_ignore:
            extracted_answer = re.sub(regex, "", extracted_answer)
            gold = re.sub(regex, "", gold)
        
        return extracted_answer == gold
    
    def _extract_ground_truth(self, ground_truth: str) -> str:
        """Extract ground truth answer from GSM8k format."""
        if "####" in ground_truth:
            return ground_truth.split("####")[1].strip()
        return ground_truth.strip()
