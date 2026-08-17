"""BM25-based retriever for BrowseComp functional independence checking.

Lightweight alternative to FaissRetriever that requires no GPU or embedding
model. Used in the function predecessor pipeline to verify that fabricated
arguments don't change the answer to the original function.
"""

import json
import re
from typing import Optional

DEFAULT_CORPUS_DATASET = "Tevatron/browsecomp-plus-corpus"
DEFAULT_CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _load_corpus(dataset: str, revision: str):
    from datasets import load_dataset

    return load_dataset(dataset, split="train", revision=revision)


def _build_index(tokenized_corpus: list[list[str]]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi(tokenized_corpus)


class BM25Retriever:
    """BM25-based sparse retriever over the BrowseComp-Plus corpus.

    Thread-safe for concurrent reads after initialization (BM25Okapi is
    read-only once built). No GPU required.
    """

    def __init__(
        self,
        corpus_dataset: str = DEFAULT_CORPUS_DATASET,
        corpus_revision: str = DEFAULT_CORPUS_REVISION,
        snippet_max_words: int = 500,
        k: int = 5,
    ):
        self.k = k
        self.snippet_max_words = snippet_max_words

        # Load corpus from HuggingFace
        print(f"  [BM25Retriever] Loading corpus: {corpus_dataset}")
        ds = _load_corpus(corpus_dataset, corpus_revision)
        self.docids: list[str] = []
        self.texts: list[str] = []
        for row in ds:
            self.docids.append(row["docid"])
            self.texts.append(row["text"])
        print(f"  [BM25Retriever] Corpus loaded: {len(self.docids)} documents")

        # Build BM25 index
        print("  [BM25Retriever] Building BM25 index...")
        tokenized_corpus = [_tokenize(text) for text in self.texts]
        self.bm25 = _build_index(tokenized_corpus)
        print("  [BM25Retriever] BM25 index ready")

    def search(self, query: str, k: int | None = None) -> str:
        """Search and return JSON string of results.

        Returns the same format as FaissRetriever.search() for compatibility:
        ``[{"docid": ..., "score": ..., "snippet": ...}, ...]``
        """
        top_k = k or self.k
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            idx = int(idx)
            docid = self.docids[idx]
            text = self.texts[idx]

            # Truncate snippet by word count
            words = text.split()
            if len(words) > self.snippet_max_words:
                text = " ".join(words[: self.snippet_max_words])

            results.append({
                "docid": docid,
                "score": float(scores[idx]),
                "snippet": text,
            })

        return json.dumps(results, indent=2)
