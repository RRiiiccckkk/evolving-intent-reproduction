"""
BrowseComp-Plus dataset module for function + argument extraction.
830 complex research queries from BrowseComp-Plus benchmark (Tevatron).
"""

from .extractor import BrowseCompPlusExtractor
from .verifier import BrowseCompPlusVerifier

__all__ = ["BrowseCompPlusExtractor", "BrowseCompPlusVerifier"]
