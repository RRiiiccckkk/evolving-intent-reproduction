"""
BrowseComp-Plus Extractor implementation.
For complex research queries requiring multi-constraint information retrieval.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from intent_construction.intent_extraction.core.base_extractor import BaseExtractor
from intent_construction.intent_extraction.core.llm_utils import generate_json, generate_text, load_prompt, populate_prompt


class BrowseCompPlusExtractor(BaseExtractor):
    """Extractor for BrowseComp-Plus research queries."""
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        num_arguments: int = 6,
        max_verification_attempts: int = 5,
        verif_model: str = "gpt-5.1",
        enable_model_verification: bool = True
    ):
        self.original_response_cache = {}
        
        self.system_prompt = """You are an expert research assistant specializing in finding specific information based on multiple constraints.
Given a research query with arguments/constraints, analyze the evidence documents provided and identify the answer.
At the end, provide your final answer in the format:
Answer: <your_answer>

Where <your_answer> is the specific name, fact, or piece of information requested."""
        
        super().__init__(
            model=model,
            num_arguments=num_arguments,
            max_verification_attempts=max_verification_attempts,
            verif_model=verif_model,
            enable_model_verification=enable_model_verification
        )
        
        # Reasoning models (gpt-5*) need reasoning_effort instead of temperature
        self._is_reasoning = "gpt-5" in model
        self._is_verif_reasoning = "gpt-5" in verif_model
    
    def get_dataset_name(self) -> str:
        return "browsecomp_plus"
    
    def get_prompts_dir(self) -> Path:
        return Path(__file__).parent / "prompts"
    
    def _load_prompts(self) -> None:
        prompts_dir = self.get_prompts_dir()
        self.prompt_decompose = load_prompt(prompts_dir / "segmentation.txt")
        self.prompt_conversational = load_prompt(prompts_dir / "conversational.txt")
        self.prompt_verification = load_prompt(prompts_dir / "verification.txt")
    
    def decompose(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Decompose the query into function and arguments."""
        question = sample["question"]
        
        prompt = populate_prompt(
            self.prompt_decompose,
            {"QUESTION": question}
        )
        
        result = generate_json(
            [{"role": "user", "content": prompt}],
            model=self.model,
            step="extraction-decompose",
            reasoning_effort="low" if self._is_reasoning else None
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
            step="extraction-conversational",
            reasoning_effort="low" if self._is_reasoning else None
        )
        
        return result
    
    def verify_coverage(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        question = sample["question"]
        
        # Build arguments list for verification
        arguments = [{"argument_id": 0, "argument": extracted.get("function", "")}]
        for cond in extracted.get("arguments", []):
            arguments.append({
                "argument_id": cond.get("argument_id", 0),
                "argument": cond.get("argument", "")
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
            step="extraction-verification",
            reasoning_effort="low" if self._is_reasoning else None
        )
        
        return result.get("coverage", "incomplete") == "complete"
    
    def verify_solvability(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        """Verify solvability using evidence documents as context."""
        question = sample["question"]
        ground_truth = sample.get("answer", "")
        sample_id = sample.get("id", "unknown")
        
        try:
            # Build concat prompt from function + arguments
            texts = [extracted.get("function", "")]
            for cond in sorted(extracted.get("arguments", []), key=lambda x: x.get("argument_id", 0)):
                texts.append(cond.get("argument", ""))
            concat_prompt = " ".join(texts)
            
            # Build evidence context from gold/evidence docs
            evidence_context = self._build_evidence_context(sample)
            
            if evidence_context:
                full_prompt = f"Based on the following evidence documents, answer the research query.\n\n{evidence_context}\n\nQuery: {concat_prompt}"
            else:
                full_prompt = concat_prompt
            
            # Reasoning models (gpt-5*) don't support temperature
            text_kwargs = {"model": self.verif_model, "max_tokens": 2000}
            if "gpt-5" in self.verif_model:
                text_kwargs["reasoning_effort"] = "low"
            else:
                text_kwargs["temperature"] = 0.7
            
            concat_response = generate_text(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": full_prompt}],
                **text_kwargs
            )
            
            concat_answer = self._extract_answer(concat_response)
            concat_correct = self._evaluate_answer(concat_answer, ground_truth)
            
            if concat_correct:
                print(f"  [Step 4] ✓ Concat correct for {sample_id}")
                return True
            
            # Try with original question + evidence
            if sample_id not in self.original_response_cache:
                if evidence_context:
                    orig_prompt = f"Based on the following evidence documents, answer the research query.\n\n{evidence_context}\n\nQuery: {question}"
                else:
                    orig_prompt = question
                    
                original_response = generate_text(
                    [{"role": "system", "content": self.system_prompt},
                     {"role": "user", "content": orig_prompt}],
                    **text_kwargs
                )
                self.original_response_cache[sample_id] = original_response
            else:
                original_response = self.original_response_cache[sample_id]
            
            original_answer = self._extract_answer(original_response)
            
            if self._answers_equivalent(concat_answer, original_answer):
                print(f"  [Step 4] ✓ Concat matches original (info preserved) for {sample_id}")
                return True
            else:
                print(f"  [Step 4] ✗ Concat differs from original for {sample_id}")
                print(f"    Original: {original_answer}, Concat: {concat_answer}, GT: {ground_truth}")
                return False
                
        except Exception as e:
            print(f"  [Step 4] Error in model verification for {sample_id}: {e}")
            return False
    
    def _build_evidence_context(self, sample: Dict[str, Any]) -> str:
        """Build evidence context from gold and evidence documents."""
        docs = sample.get("gold_docs", []) or sample.get("evidence_docs", [])
        if not docs:
            return ""
        
        # Use up to 3 gold docs to keep context manageable
        context_parts = []
        for i, doc in enumerate(docs[:3]):
            text = doc.get("text", "")
            # Truncate very long documents
            if len(text) > 3000:
                text = text[:3000] + "..."
            url = doc.get("url", "")
            context_parts.append(f"[Document {i+1}] (Source: {url})\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def build_output(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> Dict[str, Any]:
        sample_id = sample.get("id", "unknown")
        split = sample.get("split", "test")
        
        result = {
            "task_id": f"extracted-browsecomp_plus-{split}-{sample_id}",
            "original_id": sample_id,
            "task": "search",
            "question": sample["question"],
            "answer": sample.get("answer", ""),
            "fully_specified_question": sample["question"],
            "function": extracted.get("function", ""),
            "arguments": extracted.get("arguments", []),
            "num_arguments": len(extracted.get("arguments", [])),
            "model_name": self.model,
            "data_source": "browsecomp_plus",
            "query_id": sample.get("query_id", ""),
        }
        
        self.original_response_cache = {}
        return result
    
    def _extract_answer(self, response: str) -> Optional[str]:
        if response is None:
            return None
            
        patterns = [
            r"Answer:\s*(.+?)(?:\n|$)",
            r"The answer is[:\s]+(.+?)(?:\.|$)",
            r"(?:^|\n)([^\n]+)$",  # last line fallback
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        lines = response.strip().split('\n')
        if lines:
            return lines[-1].strip()
        
        return None
    
    def _evaluate_answer(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        
        pred_norm = self._normalize_answer(predicted)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Exact match
        if pred_norm == gt_norm:
            return True
        
        # Check if ground truth is contained in prediction (for short factual answers)
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return True
        
        return False
    
    def _answers_equivalent(self, ans1: Optional[str], ans2: Optional[str]) -> bool:
        if ans1 is None or ans2 is None:
            return ans1 == ans2
        
        norm1 = self._normalize_answer(ans1)
        norm2 = self._normalize_answer(ans2)
        
        return norm1 == norm2 or norm1 in norm2 or norm2 in norm1
    
    def _normalize_answer(self, answer: str) -> str:
        if answer is None:
            return ""
        
        ans = answer.strip().lower()
        # Remove common punctuation and extra whitespace
        ans = re.sub(r'[.,;:!?"\'\(\)]', '', ans)
        ans = re.sub(r'\s+', ' ', ans)
        ans = ans.strip()
        
        return ans
    
    def verify_with_llm_judge(
        self,
        sample: Dict[str, Any],
        extracted: Dict[str, Any]
    ) -> bool:
        sample_id = sample.get("id", "unknown")
        question = sample["question"]
        
        # Build reconstructed text from function + arguments
        texts = [extracted.get("function", "")]
        for cond in sorted(extracted.get("arguments", []), key=lambda x: x.get("argument_id", 0)):
            texts.append(cond.get("argument", ""))
        reconstructed = " ".join(texts)
        
        try:
            judge_prompt = load_prompt(self.get_prompts_dir() / "llm_judge.txt")
            filled_prompt = populate_prompt(
                judge_prompt,
                {
                    "ORIGINAL": question,
                    "RECONSTRUCTED": reconstructed
                }
            )
            
            result = generate_json(
                [{"role": "user", "content": filled_prompt}],
                model=self.verif_model,
                step="llm-judge",
                reasoning_effort="low" if self._is_verif_reasoning else None
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
