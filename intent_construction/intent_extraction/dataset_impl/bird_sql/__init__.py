__all__ = ["BirdSqlExtractor", "BirdSqlVerifier"]


def __getattr__(name):
    """Keep download/integrity helpers free of SQL/LLM import side effects."""
    if name == "BirdSqlExtractor":
        from .extractor import BirdSqlExtractor

        return BirdSqlExtractor
    if name == "BirdSqlVerifier":
        from .verifier import BirdSqlVerifier

        return BirdSqlVerifier
    raise AttributeError(name)
