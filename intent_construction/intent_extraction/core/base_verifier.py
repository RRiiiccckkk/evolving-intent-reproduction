"""
Base Verifier class for dataset-specific verification implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseVerifier(ABC):
    """Abstract base class for dataset-specific verifiers."""
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        num_runs: int = 1
    ):
        """
        Initialize the verifier.
        
        Args:
            model: LLM model identifier for verification
            num_runs: Number of verification runs (for reliability)
        """
        self.model = model
        self.num_runs = num_runs
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the dataset name."""
        pass
    
    @abstractmethod
    def verify_coverage(
        self, 
        original: Dict[str, Any], 
        extracted: Dict[str, Any]
    ) -> bool:
        """
        Verify that extracted function + arguments contain all essential information.
        
        Args:
            original: Original sample dict
            extracted: Dict with 'function' and 'arguments'
            
        Returns:
            True if all information is preserved
        """
        pass
    
    @abstractmethod
    def verify_solvability(
        self, 
        original: Dict[str, Any], 
        extracted: Dict[str, Any],
        ground_truth: str
    ) -> bool:
        """
        Verify that model can solve using extracted function + arguments.
        
        Args:
            original: Original sample dict
            extracted: Dict with 'function' and 'arguments'
            ground_truth: Expected answer
            
        Returns:
            True if model can solve correctly
        """
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the answer from model response.
        Dataset-specific answer extraction logic.
        
        Args:
            response: Model response text
            
        Returns:
            Extracted answer or None
        """
        pass
    
    @abstractmethod
    def evaluate_answer(self, predicted: Optional[str], ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        Dataset-specific evaluation logic.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            True if answers match
        """
        pass
