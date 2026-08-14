"""
Dataset registry for function + argument extraction.
Maps dataset names to their extractor and verifier classes.
"""

from typing import Dict, Type, Tuple, Any
from intent_construction.intent_extraction.core.base_extractor import BaseExtractor
from intent_construction.intent_extraction.core.base_verifier import BaseVerifier


# Registry of dataset implementations
# Format: dataset_name -> (ExtractorClass, VerifierClass)
_REGISTRY: Dict[str, Tuple[Type[BaseExtractor], Type[BaseVerifier]]] = {}


def register_dataset(
    name: str, 
    extractor_cls: Type[BaseExtractor], 
    verifier_cls: Type[BaseVerifier]
) -> None:
    """
    Register a dataset implementation.
    
    Args:
        name: Dataset name (e.g., 'gsm8k', 'math')
        extractor_cls: Extractor class for this dataset
        verifier_cls: Verifier class for this dataset
    """
    _REGISTRY[name] = (extractor_cls, verifier_cls)


def get_extractor(name: str, **kwargs) -> BaseExtractor:
    """
    Get an extractor instance for the specified dataset.
    
    Args:
        name: Dataset name
        **kwargs: Arguments to pass to the extractor constructor
        
    Returns:
        Extractor instance
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available datasets: {list(_REGISTRY.keys())}"
        )
    extractor_cls, _ = _REGISTRY[name]
    return extractor_cls(**kwargs)


def get_verifier(name: str, **kwargs) -> BaseVerifier:
    """
    Get a verifier instance for the specified dataset.
    
    Args:
        name: Dataset name
        **kwargs: Arguments to pass to the verifier constructor
        
    Returns:
        Verifier instance
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available datasets: {list(_REGISTRY.keys())}"
        )
    _, verifier_cls = _REGISTRY[name]
    return verifier_cls(**kwargs)


def list_datasets() -> list:
    """Return list of available dataset names."""
    return list(_REGISTRY.keys())


# Auto-register available datasets
def _auto_register():
    """Auto-register all available dataset implementations."""
    # GSM8k
    from intent_construction.intent_extraction.dataset_impl.gsm8k import GSM8kExtractor, GSM8kVerifier
    register_dataset("gsm8k", GSM8kExtractor, GSM8kVerifier)
    
    # BrowseComp-Plus
    from intent_construction.intent_extraction.dataset_impl.browsecomp_plus import BrowseCompPlusExtractor, BrowseCompPlusVerifier
    register_dataset("browsecomp_plus", BrowseCompPlusExtractor, BrowseCompPlusVerifier)
    
    # SWE-bench Verified
    from intent_construction.intent_extraction.dataset_impl.swe_bench_verified import SWEBenchVerifiedExtractor, SWEBenchVerifiedVerifier
    register_dataset("swe_bench_verified", SWEBenchVerifiedExtractor, SWEBenchVerifiedVerifier)
    
    # BIRD-SQL (optional — graceful fallback if not yet implemented)
    try:
        from intent_construction.intent_extraction.dataset_impl.bird_sql import BirdSqlExtractor, BirdSqlVerifier
        register_dataset("bird_sql", BirdSqlExtractor, BirdSqlVerifier)
    except ImportError:
        pass


_auto_register()
