"""
Base Extractor class for dataset-specific extraction implementations.
Extracts Function + Arguments from benchmark problems.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

from intent_construction.intent_extraction.core.llm_utils import (
    LLMAccountingError,
    LLMIncompleteResponse,
)


class BaseExtractor(ABC):
    """Abstract base class for dataset-specific extractors."""
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        num_arguments: int = 4,
        max_verification_attempts: int = 5,
        verif_model: str = "gpt-5.1",
        enable_model_verification: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            model: LLM model identifier for extraction
            num_arguments: Expected number of arguments to extract
            max_verification_attempts: Max retries for verification
            verif_model: Model to use for performance verification
            enable_model_verification: Whether to run model performance verification
        """
        self.model = model
        self.num_arguments = num_arguments
        self.max_verification_attempts = max_verification_attempts
        self.verif_model = verif_model
        self.enable_model_verification = enable_model_verification
        
        # Load prompts
        self._load_prompts()
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the dataset name (e.g., 'gsm8k', 'math', 'search')."""
        pass
    
    @abstractmethod
    def get_prompts_dir(self) -> Path:
        """Return the path to dataset-specific prompts directory."""
        pass
    
    @abstractmethod
    def _load_prompts(self) -> None:
        """Load dataset-specific prompt templates."""
        pass
    
    @abstractmethod
    def decompose(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Decompose the sample into function and arguments.
        
        Args:
            sample: Input sample dict
            
        Returns:
            Dict with 'function' and 'arguments' keys
        """
        pass
    
    @abstractmethod
    def to_conversational(
        self, 
        sample: Dict[str, Any], 
        decomposed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Step 2: Transform function + arguments into conversational format.
        
        Args:
            sample: Original sample dict
            decomposed: Dict with 'function' and 'arguments' from Step 1
            
        Returns:
            Dict with 'initial_query' and 'hints'
        """
        pass
    
    @abstractmethod
    def verify_coverage(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        """
        Step 3: Verify that extraction contains all essential information.
        
        Args:
            sample: Original sample dict
            extracted: Dict with function and arguments
            
        Returns:
            True if verification passed, False otherwise
        """
        pass
    
    @abstractmethod
    def verify_solvability(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        """
        Step 4: Verify that model can solve using extracted function + arguments.
        
        Args:
            sample: Original sample dict
            extracted: Dict with function and arguments
            
        Returns:
            True if verification passed, False otherwise
        """
        pass
    
    @abstractmethod
    def build_output(
        self, 
        sample: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the final output format.
        
        Args:
            sample: Original sample dict
            extracted: Dict with function and arguments
            
        Returns:
            Final output dict
        """
        pass
    
    def extract(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main entry point: Extract function + arguments from a single sample.
        
        Args:
            sample: Input sample dict
            
        Returns:
            Extracted sample or None if failed
        """
        sample_id = sample.get("id", "unknown")

        # Store best attempt for fallback
        best_extracted = None
        coverage_failed_extractions = []  # Store extractions that failed coverage
        verification_infra_failures = 0   # solvability calls that errored (timeout/transport)
        definitive_verification_failures = 0  # solvability answered "wrong" (no error)

        for attempt in range(1, self.max_verification_attempts + 1):
            try:
                # Step 1: Decompose into function + arguments
                decomposed = self.decompose(sample)
                
                # Validate argument count is within reasonable range
                num_arguments = len(decomposed.get("arguments", []))
                max_expected = self.num_arguments + 2
                if num_arguments < 1:
                    print(f"Warning: Too few arguments ({num_arguments}), expected at least 1")
                elif num_arguments > max_expected:
                    print(f"Warning: Too many arguments ({num_arguments}), expected at most {max_expected}")
                
                # Step 2: Conversational transformation
                conversational = self.to_conversational(sample, decomposed)
                
                # Build extracted format (function + arguments)
                extracted = self._build_extracted(conversational)
                
                # Step 3: Coverage verification
                if not self.verify_coverage(sample, extracted):
                    print(f"Coverage verification failed for {sample_id}, retrying... (attempt {attempt})")
                    coverage_failed_extractions.append(extracted)
                    continue
                
                # Save as best attempt (passed coverage)
                best_extracted = extracted
                
                # Step 4: Solvability verification (optional)
                if self.enable_model_verification:
                    self._verification_infra_error = False
                    if not self.verify_solvability(sample, extracted):
                        if getattr(self, "_verification_infra_error", False):
                            verification_infra_failures += 1
                        else:
                            definitive_verification_failures += 1
                        print(f"Model verification failed for {sample_id}, retrying... (attempt {attempt})")
                        continue
                
                # All verifications passed
                return self.build_output(sample, extracted)
                
            except (LLMAccountingError, LLMIncompleteResponse):
                raise
            except Exception as e:
                print(f"Error processing {sample_id} (attempt {attempt}): {e}")
                continue
        
        # 2nd pass: Try LLM-as-Judge if we have a best attempt
        if best_extracted is not None and hasattr(self, 'verify_with_llm_judge'):
            print(f"  [2nd Pass] Trying LLM-as-Judge for {sample_id}...")
            if self.verify_with_llm_judge(sample, best_extracted):
                return self.build_output(sample, best_extracted)
        
        # 3rd pass: For coverage-failed extractions, try solvability verification
        # If the model can solve it, the extraction is good enough!
        if coverage_failed_extractions and self.enable_model_verification:
            print(f"  [3rd Pass] Coverage failed but trying solvability for {sample_id}...")
            for extracted in coverage_failed_extractions:
                try:
                    if self.verify_solvability(sample, extracted):
                        print(f"  ✅ Solvability passed! Accepting extraction for {sample_id}")
                        return self.build_output(sample, extracted)
                except (LLMAccountingError, LLMIncompleteResponse):
                    raise
                except Exception as e:
                    print(f"  Solvability check failed: {e}")
                    continue
        
        # 4th pass: the published-ID protocol cannot drop a sample whose only
        # failure mode is that the solvability verifier itself kept timing out
        # (degenerate provider generation on one prompt). Coverage passed and
        # no verification ever returned a definitive wrong answer, so accept
        # the best extraction with an explicit audit flag.
        if (
            best_extracted is not None
            and definitive_verification_failures == 0
            and verification_infra_failures > 0
        ):
            print(
                f"  [4th Pass] Solvability verifier unavailable (infra timeout) for "
                f"{sample_id}; accepting coverage-passed extraction with audit flag"
            )
            output = self.build_output(sample, best_extracted)
            if isinstance(output, dict):
                output["solvability_verification"] = "skipped_infra_timeout"
            return output

        print(f"Failed after {self.max_verification_attempts} attempts + fallbacks. Skipping {sample_id}")
        return None

    def _build_extracted(self, conversational: Dict[str, Any]) -> Dict[str, Any]:
        """Convert conversational format to function + arguments."""
        function = conversational["initial_query"]
        arguments = []
        for i, hint in enumerate(conversational.get("hints", [])):
            arguments.append({
                "argument_id": i + 1,
                "argument": hint["hint"]
            })
        return {"function": function, "arguments": arguments}
    
    # Legacy compatibility: alias for extract()
    def shard_question(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Legacy method name. Use extract() instead."""
        return self.extract(sample)
