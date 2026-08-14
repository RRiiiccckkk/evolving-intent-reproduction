"""
GSM8k Extractor implementation.
Extracts Function + Arguments from GSM8k math word problems.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from intent_construction.intent_extraction.core.base_extractor import BaseExtractor
from intent_construction.intent_extraction.core.llm_utils import generate_json, generate_text, load_prompt, populate_prompt


class GSM8kExtractor(BaseExtractor):
    """Extractor for GSM8k math word problems."""
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        num_arguments: int = 4,
        max_verification_attempts: int = 5,
        verif_model: str = "gpt-5.1",
        enable_model_verification: bool = True
    ):
        # Cache for original question responses (to avoid redundant API calls)
        self.original_response_cache = {}
        
        # System prompt for model performance verification
        self.system_prompt = """You are helping the user solve a math problem. If you propose a numerical solution, you should provide it at the end of your response in the following format:
```answer
<numerical_answer>
```
Where <numerical_answer> is the numerical answer to the problem."""
        
        super().__init__(
            model=model,
            num_arguments=num_arguments,
            max_verification_attempts=max_verification_attempts,
            verif_model=verif_model,
            enable_model_verification=enable_model_verification
        )
    
    def get_dataset_name(self) -> str:
        return "gsm8k"
    
    def get_prompts_dir(self) -> Path:
        return Path(__file__).parent / "prompts"
    
    def _load_prompts(self) -> None:
        """Load GSM8k-specific prompt templates."""
        prompts_dir = self.get_prompts_dir()
        self.prompt_decompose = load_prompt(prompts_dir / "segmentation.txt")
        self.prompt_conversational = load_prompt(prompts_dir / "conversational.txt")
        self.prompt_verification = load_prompt(prompts_dir / "verification.txt")
    
    def decompose(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Decompose the question into function and arguments."""
        question = sample["question"]
        
        prompt = populate_prompt(
            self.prompt_decompose,
            {"QUESTION": question}
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
        question = sample["question"]
        
        prompt = populate_prompt(
            self.prompt_conversational,
            {
                "QUESTION": question,
                "GOAL": decomposed["function"],
                "CONDITIONS": json.dumps(decomposed["arguments"])
            }
        )
        
        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.model,
            step="extraction-conversational"
        )
        
        return result
    
    def verify_coverage(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        """Step 3: Verify that extraction contains all essential information."""
        question = sample["question"]
        
        # Convert to argument format for verification prompt
        arguments = [{"argument_id": 0, "argument": extracted["function"]}]
        for cond in extracted.get("arguments", []):
            arguments.append({
                "argument_id": cond["argument_id"],
                "argument": cond["argument"]
            })
        
        prompt = populate_prompt(
            self.prompt_verification,
            {
                "QUERY": question,
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
        Step 4: Verify that model can solve using extracted function + arguments.
        
        Logic:
        1. Test concat (function + arguments concatenated) first
        2. If concat is correct → PASS
        3. If concat is wrong:
           - Test original question (cached)
           - If concat answer == original answer → PASS (info preserved)
           - Otherwise → FAIL (info lost)
        """
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        sample_id = sample.get("id", "unknown")
        
        try:
            # Build concat prompt from function + arguments
            texts = [extracted["function"]]
            for cond in sorted(extracted.get("arguments", []), key=lambda x: x["argument_id"]):
                texts.append(cond["argument"])
            concat_prompt = " ".join(texts)
            
            # Test concat
            concat_response = generate_text(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": concat_prompt}],
                model=self.verif_model,
                temperature=0.7,
                max_tokens=1000
            )
            
            concat_answer = self._extract_answer(concat_response)
            concat_correct = self._evaluate_answer(concat_answer, ground_truth)
            
            # If concat is correct, we're done!
            if concat_correct:
                print(f"  [Step 4] ✓ Concat correct for {sample_id}")
                return True
            
            # Concat is wrong, need to check original question
            if sample_id not in self.original_response_cache:
                original_response = generate_text(
                    [{"role": "system", "content": self.system_prompt},
                     {"role": "user", "content": question}],
                    model=self.verif_model,
                    temperature=0.7,
                    max_tokens=1000
                )
                self.original_response_cache[sample_id] = original_response
            else:
                original_response = self.original_response_cache[sample_id]
            
            original_answer = self._extract_answer(original_response)
            
            # Compare answers
            if concat_answer == original_answer:
                print(f"  [Step 4] ✓ Concat matches original (both wrong) for {sample_id}")
                return True
            else:
                gt = self._extract_ground_truth(ground_truth)
                print(f"  [Step 4] ✗ Concat differs from original for {sample_id}")
                print(f"    Original: {original_answer}, Concat: {concat_answer}, GT: {gt}")
                return False
                
        except Exception as e:
            print(f"  [Step 4] Error in model verification for {sample_id}: {e}")
            return False
    
    def build_output(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the final output format."""
        sample_id = sample.get("id", "unknown")
        split = sample.get("split", "eval")
        
        result = {
            "task_id": f"extracted-gsm8k-{split}-{sample_id}",
            "original_id": sample_id,
            "task": "math",
            "question": sample["question"],
            "answer": self._extract_ground_truth(sample.get("answer", "")),
            "fully_specified_question": sample["question"],
            # New format
            "function": extracted["function"],
            "arguments": extracted.get("arguments", []),
            "num_arguments": len(extracted.get("arguments", [])),
            "model_name": self.model,
        }
        
        # Clear cache for this sample
        self.original_response_cache = {}
        
        return result
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from model response."""
        try:
            # Try ```answer format first (MTCO style)
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
    
    def _extract_ground_truth(self, ground_truth: str) -> str:
        """Extract ground truth answer from GSM8k format."""
        if "####" in ground_truth:
            return ground_truth.split("####")[1].strip()
        return ground_truth.strip()
    
    def _evaluate_answer(self, predicted: Optional[str], ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        if predicted is None:
            return False
        
        # Extract ground truth
        gold = self._extract_ground_truth(ground_truth).lower()
        
        # Normalize both answers (remove $, commas, etc.)
        regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]
        
        try:
            # Extract numerical answer
            extracted_answer = predicted.strip()
            extracted_answer = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", extracted_answer)[-1]
            extracted_answer = extracted_answer[0] if extracted_answer[0] else extracted_answer[1]
            extracted_answer = extracted_answer.lower()
        except:
            extracted_answer = predicted.strip().lower()
        
        # Apply regexes
        for regex in regexes_to_ignore:
            extracted_answer = re.sub(regex, "", extracted_answer)
            gold = re.sub(regex, "", gold)
        
        return extracted_answer == gold
    
    def verify_with_llm_judge(
        self,
        sample: Dict[str, Any],
        extracted: Dict[str, Any]
    ) -> bool:
        """
        2nd pass verification using LLM-as-Judge.
        Checks if extracted function + arguments contain equivalent information to original.
        """
        sample_id = sample.get("id", "unknown")
        question = sample["question"]
        
        # Build reconstructed problem from function + arguments
        texts = [extracted["function"]]
        for cond in sorted(extracted.get("arguments", []), key=lambda x: x["argument_id"]):
            texts.append(cond["argument"])
        reconstructed = " ".join(texts)
        
        try:
            # Load LLM judge prompt
            judge_prompt = load_prompt(self.get_prompts_dir() / "llm_judge.txt")
            filled_prompt = populate_prompt(
                judge_prompt,
                {
                    "ORIGINAL": question,
                    "RECONSTRUCTED": reconstructed,
                }
            )
            
            result = generate_json(
                [{"role": "user", "content": filled_prompt}],
                model=self.verif_model,
                step="llm-judge"
            )
            
            if result and result.get("equivalent", False):
                print(f"  [LLM-Judge] ✓ Information equivalent for {sample_id}")
                if result.get("reasoning"):
                    print(f"    Reason: {result['reasoning'][:100]}...")
                return True
            else:
                print(f"  [LLM-Judge] ✗ Information NOT equivalent for {sample_id}")
                if result and result.get("missing_info"):
                    print(f"    Missing: {result['missing_info']}")
                return False
                
        except Exception as e:
            print(f"  [LLM-Judge] Error for {sample_id}: {e}")
            return False
