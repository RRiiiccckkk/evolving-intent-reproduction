"""
BrowseComp-Plus Verifier implementation.
"""

import re
from typing import Dict, List, Any, Optional

from intent_construction.intent_extraction.core.base_verifier import BaseVerifier
from intent_construction.intent_extraction.core.llm_utils import generate_text


class BrowseCompPlusVerifier(BaseVerifier):
    """Verifier for BrowseComp-Plus research queries."""
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        num_runs: int = 1
    ):
        super().__init__(model=model, num_runs=num_runs)
        
        self.system_prompt = """You are an expert research assistant specializing in finding specific information based on multiple constraints.
Given a research query with arguments/constraints, analyze the evidence documents provided and identify the answer.
At the end, provide your final answer in the format:
Answer: <your_answer>

Where <your_answer> is the specific name, fact, or piece of information requested."""
    
    def get_dataset_name(self) -> str:
        return "browsecomp_plus"
    
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
        
        # Build evidence context
        evidence_context = self._build_evidence_context(original)
        if evidence_context:
            full_prompt = f"Based on the following evidence documents, answer the research query.\n\n{evidence_context}\n\nQuery: {concat_prompt}"
        else:
            full_prompt = concat_prompt
        
        correct_count = 0
        
        for _ in range(self.num_runs):
            # Reasoning models (gpt-5*) don't support temperature
            text_kwargs = {"model": self.model, "max_tokens": 2000}
            if "gpt-5" in self.model:
                text_kwargs["reasoning_effort"] = "low"
            else:
                text_kwargs["temperature"] = 0.7
            
            response = generate_text(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": full_prompt}],
                **text_kwargs
            )
            
            answer = self.extract_answer(response)
            if self.evaluate_answer(answer, ground_truth):
                correct_count += 1
        
        return correct_count >= (self.num_runs // 2 + 1)
    
    def _build_evidence_context(self, sample: Dict[str, Any]) -> str:
        """Build evidence context from gold and evidence documents."""
        docs = sample.get("gold_docs", []) or sample.get("evidence_docs", [])
        if not docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(docs[:3]):
            text = doc.get("text", "")
            if len(text) > 3000:
                text = text[:3000] + "..."
            url = doc.get("url", "")
            context_parts.append(f"[Document {i+1}] (Source: {url})\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def extract_answer(self, response: str) -> Optional[str]:
        if response is None:
            return None
            
        patterns = [
            r"Answer:\s*(.+?)(?:\n|$)",
            r"The answer is[:\s]+(.+?)(?:\.|$)",
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
        if predicted is None:
            return False
        
        pred_norm = self._normalize_answer(predicted)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Exact match
        if pred_norm == gt_norm:
            return True
        
        # Containment check (for short factual answers)
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return True
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        if answer is None:
            return ""
        
        ans = answer.strip().lower()
        ans = re.sub(r'[.,;:!?"\'\(\)]', '', ans)
        ans = re.sub(r'\s+', ' ', ans)
        ans = ans.strip()
        
        return ans
