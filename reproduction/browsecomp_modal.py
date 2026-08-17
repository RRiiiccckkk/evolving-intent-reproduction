"""Modal deployment for the fixed BrowseComp+ Qwen3-Embedding-8B retriever.

Deploy with ``modal deploy -m reproduction.browsecomp_modal`` and pass the
reported endpoint URL as ``RETRIEVER_URL`` to ``evaluation/scripts/run_browsecomp.sh``.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import modal
except ImportError:  # Allows local tooling to inspect the module without Modal.
    modal = None


APP_NAME = "evolving-intent-browsecomp-qwen3-8b"
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
SNIPPET_TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CORPUS_NAME = "Tevatron/browsecomp-plus-corpus"
CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
INDEX_REPOSITORY = "Tevatron/browsecomp-plus-indexes"
INDEX_REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
RETRIEVER_REVISION = (
    "browsecomp-retriever-"
    "e10361ce3ec95089dd79d3058cb33b73cb3b1da043b239c3525a6c7a7abd5ece"
)
INDEX_ROOT = f"/cache/indexes/{INDEX_REVISION}"
INDEX_PATTERN = f"{INDEX_ROOT}/qwen3-embedding-8b/corpus.shard*.pkl"


def health_payload() -> dict[str, str]:
    """Return deployment identity without loading the GPU retriever."""
    return {
        "status": "ok",
        "retriever_revision": RETRIEVER_REVISION,
    }


def ready_payload() -> dict[str, str]:
    return {
        "status": "ready",
        "retriever_revision": RETRIEVER_REVISION,
        "model_revision": MODEL_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "index_revision": INDEX_REVISION,
    }


if modal is not None:
    app = modal.App(APP_NAME)
    health_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "fastapi>=0.116,<1"
    )
    cache = modal.Volume.from_name("browsecomp-plus-qwen3-cache", create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "datasets>=4,<5",
            "faiss-cpu>=1.10,<2",
            "huggingface-hub>=0.34,<1",
            "numpy>=2,<3",
            "openai>=2,<3",
            "torch>=2.7,<3",
            "transformers>=4.53,<5",
            "tqdm>=4.67,<5",
            "fastapi>=0.116,<1",
        )
        .add_local_python_source(
            "evaluation",
            "intent_construction",
            "situated_simulation",
        )
    )

    @app.function(image=health_image, timeout=60)
    @modal.fastapi_endpoint(method="GET")
    def health() -> dict[str, str]:
        return health_payload()

    @app.cls(
        image=image,
        gpu="L40S",
        volumes={"/cache": cache},
        timeout=60 * 60,
        scaledown_window=5 * 60,
    )
    @modal.concurrent(max_inputs=8)
    class BrowseCompRetriever:
        @modal.enter()
        def load(self) -> None:
            from huggingface_hub import snapshot_download

            os.environ["HF_HOME"] = "/cache/huggingface"
            snapshot_download(
                repo_id=INDEX_REPOSITORY,
                repo_type="dataset",
                revision=INDEX_REVISION,
                allow_patterns=["qwen3-embedding-8b/*"],
                local_dir=INDEX_ROOT,
            )
            from evaluation.runners.run_browsecomp_experiment import FaissRetriever

            self.retriever = FaissRetriever(
                index_pattern=INDEX_PATTERN,
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                corpus_dataset=CORPUS_NAME,
                corpus_revision=CORPUS_REVISION,
                snippet_tokenizer_revision=SNIPPET_TOKENIZER_REVISION,
                snippet_max_tokens=512,
                k=5,
                device="cuda:0",
            )
            cache.commit()

        @modal.fastapi_endpoint(method="POST")
        def search(self, payload: dict[str, Any]) -> dict[str, Any]:
            import json

            query = payload.get("query")
            k = payload.get("k", 5)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            if k != 5:
                raise ValueError("Plan A fixes search_k at 5")
            return {
                "results": json.loads(self.retriever.search(query)),
                "retriever_revision": RETRIEVER_REVISION,
            }

        @modal.fastapi_endpoint(method="GET")
        def ready(self) -> dict[str, str]:
            return ready_payload()

else:
    app = None


def main() -> None:
    if modal is None:
        raise SystemExit(
            "Modal is not installed. Install reproduction/requirements-browsecomp.txt first."
        )
    print("Deploy with: modal deploy -m reproduction.browsecomp_modal")


if __name__ == "__main__":
    main()
