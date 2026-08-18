"""Serve the pinned BrowseComp+ retriever on localhost of an RTX 5090 node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from reproduction.browsecomp_modal import (
    CORPUS_NAME,
    CORPUS_REVISION,
    INDEX_REPOSITORY,
    INDEX_REVISION,
    MODEL_NAME,
    MODEL_REVISION,
    RETRIEVER_REVISION,
    SNIPPET_TOKENIZER_REVISION,
)


def ready_payload() -> dict[str, str]:
    return {
        "status": "ready",
        "runtime": "rtx-5090",
        "retriever_revision": RETRIEVER_REVISION,
        "model_revision": MODEL_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "index_revision": INDEX_REVISION,
    }


def validate_search_payload(payload: Any) -> tuple[str, int]:
    if not isinstance(payload, dict):
        raise ValueError("search payload must be an object")
    query = payload.get("query")
    k = payload.get("k", 5)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k != 5:
        raise ValueError("the formal run fixes search_k at 5")
    return query, k


def prepare_index(cache_root: Path) -> str:
    from huggingface_hub import snapshot_download

    index_root = cache_root / "indexes" / INDEX_REVISION
    snapshot_download(
        repo_id=INDEX_REPOSITORY,
        repo_type="dataset",
        revision=INDEX_REVISION,
        allow_patterns=["qwen3-embedding-8b/*"],
        local_dir=index_root,
    )
    index_pattern = str(index_root / "qwen3-embedding-8b" / "corpus.shard*.pkl")
    manifest = {
        **ready_payload(),
        "model": MODEL_NAME,
        "corpus": CORPUS_NAME,
        "snippet_tokenizer_revision": SNIPPET_TOKENIZER_REVISION,
        "index_pattern": index_pattern,
    }
    manifest_path = cache_root / "retriever_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return index_pattern


def create_app(cache_root: Path, device: str):
    from fastapi import FastAPI
    from evaluation.runners.run_browsecomp_experiment import FaissRetriever

    if not device.startswith("cuda:"):
        raise ValueError("the formal 5090 retriever requires a CUDA device")
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    index_pattern = prepare_index(cache_root)
    retriever = FaissRetriever(
        index_pattern=index_pattern,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        corpus_dataset=CORPUS_NAME,
        corpus_revision=CORPUS_REVISION,
        snippet_tokenizer_revision=SNIPPET_TOKENIZER_REVISION,
        snippet_max_tokens=512,
        k=5,
        device=device,
    )

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/ready")
    def ready() -> dict[str, str]:
        return ready_payload()

    @app.post("/search")
    def search(payload: dict[str, Any]) -> dict[str, Any]:
        query, _ = validate_search_payload(payload)
        return {
            "results": json.loads(retriever.search(query)),
            "retriever_revision": RETRIEVER_REVISION,
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("retriever must bind to localhost")
    import uvicorn

    uvicorn.run(
        create_app(args.cache_root, args.device),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
