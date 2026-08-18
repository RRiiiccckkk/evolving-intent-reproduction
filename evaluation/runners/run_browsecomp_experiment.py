#!/usr/bin/env python3
"""
BrowseComp-Plus evaluation with agentic search + EvolvingIntent.

Follows the BrowseComp-Plus methodology (https://github.com/texttron/BrowseComp-Plus):
- LLM agent uses function calling to search a FAISS-indexed corpus
- Retriever: Qwen3-Embedding-8B
- Agent iterates: think → search → read → ... → answer
- Multi-turn: EvolvingIntent provides function/argument changes across turns

Usage:
    python run_browsecomp_experiment.py \
        --data_path ../final_dataset/browsecomp_plus_n100.json \
        --models gpt-5.1 \
        --dataset_name browsecomp_plus_n100 \
        --num_turns 1 \
        --num_workers 10
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import tempfile
import time
import logging
import unicodedata
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from urllib import request as urllib_request
from urllib import error as urllib_error
from tqdm import tqdm

import threading
import numpy as np

from intent_construction.intent_extraction.core.llm_utils import (
    LLMAccountingError,
    LLMIncompleteResponse,
    _accounted_api_call,
    _chat_final_content,
    _is_responses_api_model,
    _responses_final_content,
    _supports_responses_api,
    clean_model_name,
    get_client,
    resolve_model_name,
)
from situated_simulation.user_simulation import EvolvingIntent

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
BROWSECOMP_PLUS_DIR = Path(__file__).parent.parent.parent / "BrowseComp-Plus"

# Default paths
DEFAULT_INDEX_PATTERN = str(BROWSECOMP_PLUS_DIR / "indexes" / "qwen3-embedding-8b" / "corpus.shard*.pkl")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
DEFAULT_CORPUS_DATASET = "Tevatron/browsecomp-plus-corpus"
DEFAULT_MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DEFAULT_SNIPPET_TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
DEFAULT_INDEX_REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
PLAN_A_RETRIEVER_REVISION = (
    "browsecomp-retriever-"
    "e10361ce3ec95089dd79d3058cb33b73cb3b1da043b239c3525a6c7a7abd5ece"
)
PLAN_A_MODEL = "kimi-k2.6"
PLAN_A_REASONING_EFFORT = "medium"
PLAN_A_SEARCH_CAP = 50
PLAN_A_SAMPLE_COUNT = 100
PLAN_A_CHECKPOINT_SCHEMA_VERSION = 2
PLAN_A_SCENARIOS = {(1, 0, 0), (7, 2, 2)}
PLAN_A_TASK_IDS_FILE = (
    Path(__file__).resolve().parents[2]
    / "intent_construction"
    / "eval_indices"
    / "browsecomp_plus_task_ids.json"
)

# BrowseComp+ search agent instruction (from BrowseComp-Plus repo)
SEARCH_AGENT_INSTRUCTION = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {content}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

# Search tool definition for function calling
SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Perform a search on a knowledge source. Returns top-5 hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                }
            },
            "required": ["query"],
        },
    }
}


def atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace a JSON artifact without exposing partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _task_checkpoint_path(checkpoint_dir: Path, task_id: str) -> Path:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._") or "task"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return checkpoint_dir / f"{readable[:96]}-{digest}.json"


def _canonical_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint input is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _checkpoint_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _checkpoint_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(
        f"checkpoint input contains unsupported value {type(value).__qualname__}"
    )


def _file_sha256_or_marker(path: str | Path | None) -> str:
    if path is None:
        return "none"
    target = Path(path)
    if not target.is_file():
        return f"missing:{target}"
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_input_sha256(sample: Any) -> str:
    return _canonical_sha256(
        _checkpoint_json_value({
            "task_id": getattr(sample, "task_id", None),
            "label": getattr(sample, "label", None),
            "metadata": getattr(sample, "metadata", None),
            "turns": getattr(sample, "turns", None),
            "structured_contents": getattr(sample, "structured_contents", None),
            "recap_texts": getattr(sample, "recap_texts", None),
        })
    )


def _source_bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            raise ValueError(f"checkpoint source file is missing: {path}")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _retriever_checkpoint_policy(retriever: Any) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "class": f"{type(retriever).__module__}.{type(retriever).__qualname__}",
    }
    for attribute in (
        "k",
        "snippet_max_tokens",
        "device",
        "revision",
        "model_name",
        "model_revision",
        "corpus_dataset",
        "corpus_revision",
        "snippet_tokenizer_revision",
    ):
        value = getattr(retriever, attribute, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            policy[attribute] = value
    url = getattr(retriever, "url", None)
    if isinstance(url, str):
        policy["url_sha256"] = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return policy


def _evaluation_checkpoint_envelope(
    *,
    task_id: str,
    input_sha256: str,
    policy: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PLAN_A_CHECKPOINT_SCHEMA_VERSION,
        "task_id": task_id,
        "input_sha256": input_sha256,
        "policy": policy,
        "result": result,
    }


def _load_result_mapping(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"result artifact must be a JSON object: {path}")
    results: dict[str, dict[str, Any]] = {}
    for task_id, result in payload.items():
        if not isinstance(task_id, str) or not isinstance(result, dict):
            raise ValueError(f"invalid task result in {path}")
        if result.get("task_id") not in (None, task_id):
            raise ValueError(f"task id mismatch in {path}: {task_id}")
        results[task_id] = result
    return results


def _result_is_complete(result: Any, expected_responses: int | None = None) -> bool:
    if not isinstance(result, dict) or result.get("success") is not True:
        return False
    prediction = result.get("prediction")
    if not isinstance(prediction, str) or not prediction.strip():
        return False
    responses = result.get("responses")
    if not isinstance(responses, list) or not responses:
        return False
    if expected_responses is not None and len(responses) != expected_responses:
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("response"), str)
        and bool(item["response"].strip())
        for item in responses
    )


def _load_task_checkpoints(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not checkpoint_dir.exists():
        return results
    for path in sorted(checkpoint_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict) or not isinstance(result.get("task_id"), str):
            raise ValueError(f"invalid task checkpoint: {path}")
        results[result["task_id"]] = result
    return results


def _load_bound_task_checkpoints(
    checkpoint_dir: Path,
    *,
    input_hashes: dict[str, str],
    policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not checkpoint_dir.exists():
        return results
    for path in sorted(checkpoint_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            envelope = json.load(handle)
        if not isinstance(envelope, dict):
            raise ValueError(f"invalid bound task checkpoint: {path}")
        task_id = envelope.get("task_id")
        if not isinstance(task_id, str) or task_id not in input_hashes:
            raise ValueError(f"checkpoint has an unexpected task ID: {path}")
        expected = _evaluation_checkpoint_envelope(
            task_id=task_id,
            input_sha256=input_hashes[task_id],
            policy=policy,
            result={},
        )
        for key in ("schema_version", "task_id", "input_sha256", "policy"):
            if envelope.get(key) != expected[key]:
                raise ValueError(
                    f"checkpoint {key} does not match current input/policy: {path}"
                )
        result = envelope.get("result")
        if not isinstance(result, dict) or result.get("task_id") != task_id:
            raise ValueError(f"checkpoint has an invalid task result: {path}")
        results[task_id] = result
    return results


def _usage_ledger_marker(required: bool = False) -> tuple[Path, int] | None:
    raw = os.environ.get("LLM_USAGE_LEDGER_PATH", "").strip()
    if not raw:
        if required:
            raise RuntimeError("Plan A requires LLM_USAGE_LEDGER_PATH")
        return None
    path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    return path, path.stat().st_size if path.exists() else 0


def _assert_usage_models_since(
    marker: tuple[Path, int] | None,
    expected_model: str,
) -> None:
    if marker is None:
        return
    path, offset = marker
    if not path.exists():
        return
    unexpected: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        handle.seek(offset)
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("event") != "usage":
                continue
            requested_model = entry.get("requested_model")
            resolved_model = entry.get("resolved_model")
            if requested_model != expected_model:
                unexpected.add(f"requested={requested_model!r}")
            if resolved_model != expected_model:
                unexpected.add(f"resolved={resolved_model!r}")
    if unexpected:
        raise RuntimeError(
            f"model lock violated after execution; expected {expected_model!r}, "
            f"observed {sorted(unexpected)!r}"
        )


def _assert_model_lock(
    models: list[str],
    naturalizer_model: str | None,
    judge_model: str,
    locked_model: str,
) -> None:
    if models != [locked_model]:
        raise ValueError(f"Plan A requires exactly one model: {locked_model}")
    if naturalizer_model != locked_model:
        raise ValueError(f"naturalizer model must be {locked_model}")
    if judge_model != locked_model:
        raise ValueError(f"judge model must be {locked_model}")


def _read_task_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    task_ids = payload.get("task_ids", payload) if isinstance(payload, dict) else payload
    if not isinstance(task_ids, list) or not all(
        isinstance(task_id, str) and task_id for task_id in task_ids
    ):
        raise ValueError(f"task ID file must contain non-empty strings: {path}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"task ID file contains duplicates: {path}")
    return task_ids


def _lock_model_environment(locked_model: str) -> None:
    existing = os.environ.get("LLM_LOCKED_MODEL", "").strip()
    if existing and existing != locked_model:
        raise ValueError(
            f"LLM_LOCKED_MODEL is {existing!r}, expected {locked_model!r}"
        )
    os.environ["LLM_LOCKED_MODEL"] = locked_model


# =============================================================================
# FAISS Retriever (adapted from BrowseComp-Plus)
# =============================================================================

class FaissRetriever:
    """FAISS-based dense retriever using Qwen3-Embedding-8B."""

    def __init__(
        self,
        index_pattern: str = DEFAULT_INDEX_PATTERN,
        model_name: str = DEFAULT_MODEL_NAME,
        corpus_dataset: str = DEFAULT_CORPUS_DATASET,
        model_revision: str = DEFAULT_MODEL_REVISION,
        corpus_revision: str = DEFAULT_CORPUS_REVISION,
        snippet_tokenizer_revision: str = DEFAULT_SNIPPET_TOKENIZER_REVISION,
        snippet_max_tokens: int = 512,
        k: int = 5,
        device: str = "cuda:0",
    ):
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.device = device
        self.model_name = model_name
        self.model_revision = model_revision
        self.corpus_dataset = corpus_dataset
        self.corpus_revision = corpus_revision
        self.snippet_tokenizer_revision = snippet_tokenizer_revision
        self.revision = PLAN_A_RETRIEVER_REVISION

        # Load FAISS index
        self._load_index(index_pattern)
        # Load embedding model
        self._load_model(model_name, model_revision)
        # Load corpus
        self._load_corpus(corpus_dataset, corpus_revision)
        # Load snippet tokenizer
        from transformers import AutoTokenizer
        self.snippet_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            revision=snippet_tokenizer_revision,
        )
        # Thread lock for GPU FAISS + embedding model
        self._lock = threading.Lock()

    def _load_index(self, pattern: str):
        """Load FAISS index from sharded pickle files."""
        import faiss

        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No index files found: {pattern}")

        print(f"  Loading {len(files)} index shards...")

        # Load all shards into a single numpy array
        all_reps = []
        self.lookup = []
        for fpath in files:
            with open(fpath, "rb") as f:
                reps, lookup = pickle.load(f)
            all_reps.append(np.array(reps))
            self.lookup.extend(lookup)

        all_reps = np.vstack(all_reps).astype(np.float32)
        dim = all_reps.shape[1]

        # Build FAISS index (inner product for normalized embeddings)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_reps)

        # GPU acceleration
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index, co)

        print(f"  FAISS index loaded: {len(self.lookup)} vectors, dim={dim}")

    def _load_model(self, model_name: str, model_revision: str):
        """Load Qwen3-Embedding model directly (no tevatron dependency)."""
        import torch
        from transformers import AutoTokenizer, AutoModel

        print(f"  Loading embedding model: {model_name}")

        self.embed_model = AutoModel.from_pretrained(
            model_name,
            revision=model_revision,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.embed_model = self.embed_model.to(self.device)
        self.embed_model.eval()
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            padding_side="left",
        )
        print("  Embedding model loaded")

    def _load_corpus(self, dataset_name: str, corpus_revision: str):
        """Load corpus from HuggingFace dataset."""
        from datasets import load_dataset
        print(f"  Loading corpus: {dataset_name}")
        ds = load_dataset(dataset_name, split="train", revision=corpus_revision)
        self.docid_to_text = {row["docid"]: row["text"] for row in ds}
        print(f"  Corpus loaded: {len(self.docid_to_text)} documents")

    def search(self, query: str) -> str:
        """Search and return JSON string of results (matches BrowseComp-Plus format)."""
        torch = self._torch
        task_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

        input_text = task_prefix + query
        batch_dict = self.tokenizer(
            [input_text],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )

        # Lock protects GPU embedding model + FAISS GPU index (not thread-safe)
        with self._lock:
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.amp.autocast(self.device.split(":")[0]):
                with torch.no_grad():
                    outputs = self.embed_model(**batch_dict)
                    last_hidden = outputs.last_hidden_state
                    attention_mask = batch_dict["attention_mask"]
                    seq_lens = attention_mask.sum(dim=1) - 1
                    q_reps = last_hidden[torch.arange(last_hidden.size(0)), seq_lens]
                    q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=-1)
                    q_reps = q_reps.cpu().float().numpy()

            scores, indices = self.index.search(q_reps, self.k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            docid = self.lookup[index]
            text = self.docid_to_text.get(docid, "Text not found")

            # Truncate snippet
            tokens = self.snippet_tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > self.snippet_max_tokens:
                text = self.snippet_tokenizer.decode(
                    tokens[:self.snippet_max_tokens], skip_special_tokens=True
                )

            results.append({
                "docid": docid,
                "score": float(score),
                "snippet": text,
            })

        return json.dumps(results, indent=2)


class RemoteRetriever:
    """HTTP adapter for the pinned Qwen3-Embedding-8B retriever."""

    def __init__(
        self,
        url: str,
        k: int = 5,
        timeout_seconds: float = 1800.0,
        max_attempts: int = 3,
        bearer_token: str | None = None,
        revision: str | None = None,
    ):
        if not url or not url.startswith(("https://", "http://")):
            raise ValueError("retriever URL must be an absolute HTTP(S) URL")
        self.url = url
        self.k = k
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.bearer_token = bearer_token
        self.revision = revision

    def search(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be a non-empty string")
        body = json.dumps({"query": query, "k": self.k}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        req = urllib_request.Request(self.url, data=body, headers=headers, method="POST")
        if self.max_attempts < 1:
            raise ValueError("retriever max_attempts must be positive")
        for attempt in range(self.max_attempts):
            try:
                with urllib_request.urlopen(req, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib_error.HTTPError as exc:
                if exc.code not in {408, 429} and exc.code < 500:
                    raise
                if attempt + 1 >= self.max_attempts:
                    raise
            except (urllib_error.URLError, TimeoutError):
                if attempt + 1 >= self.max_attempts:
                    raise
            time.sleep(min(2 ** attempt, 30))
        if isinstance(payload, dict):
            observed_revision = payload.get("retriever_revision")
            if self.revision is not None and observed_revision != self.revision:
                raise RuntimeError(
                    "remote retriever revision mismatch: "
                    f"expected {self.revision!r}, got {observed_revision!r}"
                )
            payload = payload.get("results")
        if not isinstance(payload, list):
            raise RuntimeError("remote retriever returned an invalid response")
        for item in payload:
            if not isinstance(item, dict) or not isinstance(item.get("docid"), str):
                raise RuntimeError("remote retriever returned an invalid result item")
        return json.dumps(payload, indent=2, ensure_ascii=False)


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer_browsecomp(response: str) -> Optional[str]:
    """Extract answer from BrowseComp+ format response."""
    if not response:
        return None

    # Look for "Exact Answer: ..." pattern
    match = re.search(r"Exact Answer:\s*(.+?)(?:\n|Confidence:|$)", response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Clean up common artifacts
        answer = answer.rstrip(".")
        return answer

    # Fallback: look for answer after last line
    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("Explanation:") and not line.startswith("Confidence:"):
            return line

    return None


def extract_confidence(response: str) -> Optional[float]:
    """Extract confidence score from response."""
    if not response:
        return None
    match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)\s*%?", response)
    if match:
        return float(match.group(1))
    return None


# =============================================================================
# Agentic Search Loop (function calling)
# =============================================================================

# Synthetic user nudge used by --force_final_answer when the agentic loop
# exhausts its search budget without producing a final textual answer.
FORCE_FINAL_ANSWER_NUDGE = (
    "You have exhausted your search budget. Based on the evidence gathered so far, "
    "provide your final answer NOW in the required format. Do not call any more tools.\n\n"
    "Explanation: {your explanation, citing docids in [brackets]}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100%}"
)


def _is_kimi_k2_model(model: str) -> bool:
    return clean_model_name(model).lower().rsplit("/", 1)[-1].startswith("kimi-k2.")


def _chat_choice(response: Any):
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMIncompleteResponse("API returned no chat completion choice")
    choice = choices[0]
    if getattr(choice, "finish_reason", None) in {"length", "max_tokens"}:
        raise LLMIncompleteResponse("chat completion was truncated")
    return choice


def _force_final_chat_answer(
    client,
    deployment_name: str,
    requested_model: str,
    conversation: list[Any],
    reasoning_effort: str | None,
) -> str:
    payload: dict[str, Any] = {
        "model": deployment_name,
        "messages": [
            *conversation,
            {"role": "user", "content": FORCE_FINAL_ANSWER_NUDGE},
        ],
    }
    if not _is_kimi_k2_model(deployment_name):
        payload["tool_choice"] = "none"
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    response = _accounted_api_call(
        client.chat.completions.create,
        payload,
        requested_model=requested_model,
        resolved_model=deployment_name,
        api="chat.completions",
    )
    return _chat_final_content(response)


def run_agentic_search(
    messages: list[dict],
    model: str,
    retriever: FaissRetriever,
    max_iterations: int = 51,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    max_retries: int = 100,
    reasoning_effort: str | None = None,
    max_tool_calls: int | None = None,
    force_final_answer: bool = False,
) -> dict[str, Any]:
    """
    Run agentic search loop using OpenAI function calling.

    The agent iteratively calls the search tool until it produces a final answer.
    Returns the final response and metadata about tool usage.

    Args:
        max_iterations: Max LLM round-trips (API calls). Each round-trip may
            produce multiple parallel tool calls.
        max_tool_calls: Hard cap on total tool calls executed. When reached,
            the loop stops after the current iteration. None means no cap.
    """
    deployment_name = resolve_model_name(model)
    # Use Responses API for any closed-source reasoning model supported there
    # (gpt-5.x, codex/pro). Grok reasoning models stay on chat completions.
    use_responses = _supports_responses_api(model)
    client = get_client(use_responses_api=use_responses)

    all_tool_calls = []
    retrieved_docids = set()
    tool_call_count = 0

    if use_responses:
        return _run_agentic_search_responses(
            client, deployment_name, model, messages, retriever,
            max_iterations, max_tokens, max_retries,
            all_tool_calls, retrieved_docids,
            max_tool_calls=max_tool_calls,
            reasoning_effort=reasoning_effort,
            force_final_answer=force_final_answer,
        )

    # Working copy of messages
    conversation = list(messages)
    initial_len = len(conversation)

    for iteration in range(max_iterations):
        # Call LLM with tool definitions
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": deployment_name,
                    "messages": conversation,
                    "tools": [SEARCH_TOOL_DEF],
                }
                if _is_kimi_k2_model(deployment_name):
                    kwargs["parallel_tool_calls"] = False
                else:
                    kwargs["tool_choice"] = "auto"

                if temperature is not None and reasoning_effort is None:
                    kwargs["temperature"] = temperature

                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort

                response = _accounted_api_call(
                    client.chat.completions.create,
                    kwargs,
                    requested_model=model,
                    resolved_model=deployment_name,
                    api="chat.completions",
                )
                break
            except (LLMAccountingError, LLMIncompleteResponse):
                raise
            except Exception as e:
                error_str = str(e)
                if attempt < max_retries - 1:
                    match = re.search(r"[Tt]ry again in (\d+) seconds", error_str)
                    delay = int(match.group(1)) + 1 if match else min(2 ** attempt, 60)
                    time.sleep(delay)
                else:
                    raise

        choice = _chat_choice(response)
        message = choice.message

        # Check if the model wants to call tools
        if message.tool_calls:
            # Add assistant message with tool calls
            conversation.append(message.model_dump())

            for tool_call in message.tool_calls:
                if tool_call.function.name == "search":
                    if max_tool_calls is not None and tool_call_count >= max_tool_calls:
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": "per-turn search budget exhausted"}),
                        })
                        continue
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        # GPT 5.2 sometimes returns empty/malformed arguments
                        args = {}
                    query = args.get("query", "")
                    result = retriever.search(query)

                    # Track retrieved docids
                    try:
                        docs = json.loads(result)
                        for doc in docs:
                            retrieved_docids.add(doc.get("docid", ""))
                    except:
                        pass

                    tool_call_count += 1
                    all_tool_calls.append({
                        "query": query,
                        "iteration": iteration,
                    })

                    # Add tool response
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                else:
                    # Unknown tool
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": f"Unknown tool: {tool_call.function.name}"}),
                    })
        else:
            # No tool calls - agent has produced final answer
            final_content = message.content or ""
            if not final_content.strip():
                raise LLMIncompleteResponse("agent returned no final answer content")
            conversation.append({"role": "assistant", "content": final_content})
            return {
                "response": final_content,
                "new_messages": conversation[initial_len:],
                "tool_call_count": tool_call_count,
                "tool_calls": all_tool_calls,
                "retrieved_docids": list(retrieved_docids),
                "iterations": iteration + 1,
                "finish_reason": choice.finish_reason,
            }

        # The cap is checked for every requested tool call above, so parallel
        # calls cannot overshoot it. Once exhausted, force a text-only answer.
        if max_tool_calls is not None and tool_call_count >= max_tool_calls:
            if not force_final_answer:
                raise LLMIncompleteResponse(
                    "per-turn search budget exhausted before a final answer"
                )
            final_content = _force_final_chat_answer(
                client,
                deployment_name,
                model,
                conversation,
                reasoning_effort,
            )
            conversation.append({"role": "assistant", "content": final_content})
            return {
                "response": final_content,
                "new_messages": conversation[initial_len:],
                "tool_call_count": tool_call_count,
                "tool_calls": all_tool_calls,
                "retrieved_docids": list(retrieved_docids),
                "iterations": iteration + 1,
                "finish_reason": "max_tool_calls_forced_final",
            }

    # Max iterations reached
    if force_final_answer:
        for attempt in range(max_retries):
            try:
                forced_content = _force_final_chat_answer(
                    client,
                    deployment_name,
                    model,
                    conversation,
                    reasoning_effort,
                )
                break
            except (LLMAccountingError, LLMIncompleteResponse):
                raise
            except Exception as e:
                error_str = str(e)
                if attempt < max_retries - 1:
                    m = re.search(r"[Tt]ry again in (\d+) seconds", error_str)
                    delay = int(m.group(1)) + 1 if m else min(2 ** attempt, 60)
                    time.sleep(delay)
                else:
                    raise
        conversation.append({"role": "assistant", "content": forced_content})
        return {
            "response": forced_content,
            "new_messages": conversation[initial_len:],
            "tool_call_count": tool_call_count,
            "tool_calls": all_tool_calls,
            "retrieved_docids": list(retrieved_docids),
            "iterations": max_iterations,
            "finish_reason": "max_iterations_forced_final",
        }

    raise LLMIncompleteResponse("agent exhausted round trips without a final answer")


# Responses API tool definition (flat format, no nested "function" key)
SEARCH_TOOL_DEF_RESPONSES = {
    "type": "function",
    "name": SEARCH_TOOL_DEF["function"]["name"],
    "description": SEARCH_TOOL_DEF["function"]["description"],
    "parameters": SEARCH_TOOL_DEF["function"]["parameters"],
}


def _force_final_responses_answer(
    client,
    deployment_name: str,
    requested_model: str,
    conversation: list[dict],
    reasoning_effort: str | None,
) -> str:
    payload: dict[str, Any] = {
        "model": deployment_name,
        "input": [
            *conversation,
            {"role": "user", "content": FORCE_FINAL_ANSWER_NUDGE},
        ],
        "tool_choice": "none",
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    response = _accounted_api_call(
        client.responses.create,
        payload,
        requested_model=requested_model,
        resolved_model=deployment_name,
        api="responses",
    )
    return _responses_final_content(response)


def _run_agentic_search_responses(
    client,
    deployment_name: str,
    requested_model: str,
    messages: list[dict],
    retriever: FaissRetriever,
    max_iterations: int,
    max_tokens: int | None,
    max_retries: int,
    all_tool_calls: list,
    retrieved_docids: set,
    max_tool_calls: int | None = None,
    reasoning_effort: str | None = None,
    force_final_answer: bool = False,
) -> dict[str, Any]:
    """Agentic search loop using the Responses API for codex/pro models."""
    tool_call_count = 0
    # Build initial input from messages, preserving tool-related items from prior turns
    conversation: list[dict] = []
    for m in messages:
        if "type" in m:
            # Responses API format item (function_call, function_call_output)
            conversation.append(m)
        else:
            conversation.append({"role": m["role"], "content": m.get("content", "")})
    initial_len = len(conversation)

    for iteration in range(max_iterations):
        for attempt in range(max_retries):
            try:
                api_params: dict[str, Any] = {
                    "model": deployment_name,
                    "input": conversation,
                    "tools": [SEARCH_TOOL_DEF_RESPONSES],
                    "tool_choice": "auto",
                }
                if reasoning_effort:
                    api_params["reasoning"] = {"effort": reasoning_effort}
                resp = _accounted_api_call(
                    client.responses.create,
                    api_params,
                    requested_model=requested_model,
                    resolved_model=deployment_name,
                    api="responses",
                )
                break
            except (LLMAccountingError, LLMIncompleteResponse):
                raise
            except Exception as e:
                error_str = str(e)
                if attempt < max_retries - 1:
                    match = re.search(r"[Tt]ry again in (\d+) seconds", error_str)
                    delay = int(match.group(1)) + 1 if match else min(2 ** attempt, 60)
                    time.sleep(delay)
                else:
                    raise

        # Process output items
        has_tool_call = False
        final_text = ""
        for item in resp.output:
            if item.type == "function_call":
                has_tool_call = True
                if item.name == "search":
                    if max_tool_calls is not None and tool_call_count >= max_tool_calls:
                        conversation.append({
                            "type": "function_call",
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        })
                        conversation.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"error": "per-turn search budget exhausted"}),
                        })
                        continue
                    try:
                        args = json.loads(item.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    query = args.get("query", "")
                    result = retriever.search(query)

                    try:
                        docs = json.loads(result)
                        for doc in docs:
                            retrieved_docids.add(doc.get("docid", ""))
                    except:
                        pass

                    tool_call_count += 1
                    all_tool_calls.append({
                        "query": query,
                        "iteration": iteration,
                    })

                    # Append the function call and its output to conversation
                    conversation.append({
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    })
                    conversation.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result,
                    })
                else:
                    conversation.append({
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    })
                    conversation.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"error": f"Unknown tool: {item.name}"}),
                    })
            elif item.type == "message":
                for content in item.content:
                    if hasattr(content, "text"):
                        final_text += content.text

        if not has_tool_call:
            if not final_text.strip():
                raise LLMIncompleteResponse("agent returned no final answer content")
            conversation.append({"role": "assistant", "content": final_text})
            return {
                "response": final_text,
                "new_messages": conversation[initial_len:],
                "tool_call_count": tool_call_count,
                "tool_calls": all_tool_calls,
                "retrieved_docids": list(retrieved_docids),
                "iterations": iteration + 1,
                "finish_reason": "stop",
            }

        # Per-call enforcement above prevents a parallel batch from overshooting.
        if max_tool_calls is not None and tool_call_count >= max_tool_calls:
            if not force_final_answer:
                raise LLMIncompleteResponse(
                    "per-turn search budget exhausted before a final answer"
                )
            final_text = _force_final_responses_answer(
                client,
                deployment_name,
                requested_model,
                conversation,
                reasoning_effort,
            )
            conversation.append({"role": "assistant", "content": final_text})
            return {
                "response": final_text,
                "new_messages": conversation[initial_len:],
                "tool_call_count": tool_call_count,
                "tool_calls": all_tool_calls,
                "retrieved_docids": list(retrieved_docids),
                "iterations": iteration + 1,
                "finish_reason": "max_tool_calls_forced_final",
            }

    # Max iterations — ensure conversation ends with clean assistant message
    if force_final_answer:
        for attempt in range(max_retries):
            try:
                forced_text = _force_final_responses_answer(
                    client,
                    deployment_name,
                    requested_model,
                    conversation,
                    reasoning_effort,
                )
                break
            except (LLMAccountingError, LLMIncompleteResponse):
                raise
            except Exception as e:
                error_str = str(e)
                if attempt < max_retries - 1:
                    m = re.search(r"[Tt]ry again in (\d+) seconds", error_str)
                    delay = int(m.group(1)) + 1 if m else min(2 ** attempt, 60)
                    time.sleep(delay)
                else:
                    raise
        conversation.append({"role": "assistant", "content": forced_text})
        return {
            "response": forced_text,
            "new_messages": conversation[initial_len:],
            "tool_call_count": tool_call_count,
            "tool_calls": all_tool_calls,
            "retrieved_docids": list(retrieved_docids),
            "iterations": max_iterations,
            "finish_reason": "max_iterations_forced_final",
        }

    raise LLMIncompleteResponse("agent exhausted round trips without a final answer")


# =============================================================================
# Multi-turn Conversation Runner
# =============================================================================

def run_multi_turn_browsecomp(
    sample,  # IntentSample
    model: str,
    retriever: FaissRetriever,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_search_iterations: int = 51,
    reasoning_effort: str | None = None,
    max_tool_calls: int | None = None,
    no_retrieval_history: bool = False,
    force_final_answer: bool = False,
) -> dict[str, Any]:
    """
    Run multi-turn BrowseComp+ evaluation.

    For each user turn, runs the agentic search loop.
    """
    messages = []
    all_responses = []
    total_tool_calls = 0
    all_retrieved_docids = set()

    # Initialize: get first turn(s) from sample
    initial_turns = sample.reset()
    for turn in initial_turns:
        messages.append(turn)

    while True:
        # Check if last message is a user turn requiring response
        last_user = next(
            (t for t in reversed(messages) if t.get("role") == "user"), None
        )
        if last_user is None:
            break

        try:
            result = run_agentic_search(
                messages=messages,
                model=model,
                retriever=retriever,
                max_iterations=max_search_iterations,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_tool_calls=max_tool_calls,
                force_final_answer=force_final_answer,
            )

            all_responses.append(result)
            total_tool_calls += result["tool_call_count"]
            all_retrieved_docids.update(result["retrieved_docids"])

            # Add full conversation (tool calls + results + final response)
            if no_retrieval_history:
                # Only keep the final assistant response, drop tool calls and results
                messages.append({"role": "assistant", "content": result["response"]})
            else:
                messages.extend(result["new_messages"])

        except (LLMAccountingError, LLMIncompleteResponse):
            raise
        except Exception as e:
            user_messages = [m["content"] for m in messages if m.get("role") == "user"]
            return {
                "success": False,
                "error": str(e),
                "responses": all_responses,
                "prediction": None,
                "total_tool_calls": total_tool_calls,
                "retrieved_docids": list(all_retrieved_docids),
                "user_messages": user_messages,
            }

        # Check if sample has more turns
        if sample.is_done():
            break

        # Get next user turn(s) from sample
        next_turns = sample.step(result["response"])
        for turn in next_turns:
            messages.append(turn)

    # Extract answer from final response
    final_response = all_responses[-1]["response"] if all_responses else None
    prediction = extract_answer_browsecomp(final_response) if final_response else None
    confidence = extract_confidence(final_response) if final_response else None

    # Collect user messages for display (especially important for LLM user)
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]

    return {
        "success": True,
        "error": None,
        "responses": all_responses,
        "prediction": prediction,
        "confidence": confidence,
        "total_tool_calls": total_tool_calls,
        "retrieved_docids": list(all_retrieved_docids),
        "user_messages": user_messages,
    }


# =============================================================================
# Answer Checking (exact match for BrowseComp+)
# =============================================================================

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def em_score(label: str, pred: str) -> bool:
    """
    Smart exact-match scoring with normalization.
    Handles diacritics, punctuation, name order, articles, etc.
    Adapted from PPP-Agent (sunnweiwei/PPP-Agent).
    """
    ign = {'a', 'an', 'the', 'of', 'on', 'in', 'and', '&', 'for', 'to', 'by', 'with'}
    deacc = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    def norm(s: str) -> str:
        s = deacc(s).lower()
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
        s = re.sub(r'[\u2018\u2019\u201A\u201B\u2032"""\'`*#]+', '', s)
        s = re.sub(r'[:–—\-_/.,;!()?]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    strip = lambda s: re.sub(r'\s+', '', norm(s))
    toks = lambda s: [t for t in norm(s).split() if t not in ign and not re.fullmatch(r'\d{4}', t)]
    if strip(label) == strip(pred):
        return True
    lt, pt = toks(label), toks(pred)
    if not lt or not pt:
        return False
    if Counter(lt) == Counter(pt):
        return True
    if len(lt) >= 2 and len(pt) >= 2 and lt[-1] == pt[-1]:
        f1, f2 = lt[0], pt[0]
        if f1 == f2 or (min(len(f1), len(f2)) >= 4 and (f1.startswith(f2) or f2.startswith(f1))):
            return True
    head = lambda s: strip(re.split(r'[:–—-]', norm(s), maxsplit=1)[0])
    if head(label) == head(pred):
        return True
    # Token-subset (year-preserving): all label tokens in pred, at most 1 extra
    # Handles "65" vs "65 albums", "115" vs "115 students"
    lt_full = [t for t in norm(label).split() if t not in ign]
    pt_full = [t for t in norm(pred).split() if t not in ign]
    if lt_full and pt_full and set(lt_full).issubset(set(pt_full)) and len(pt_full) <= len(lt_full) + 1:
        return True
    # Fuzzy match: edit distance ≤ 1 on stripped forms (min length 5)
    # Handles minor typos like "Smollett" vs "Smollet"
    sl, sp = strip(label), strip(pred)
    if len(sl) >= 5 and len(sp) >= 5 and _edit_distance(sl, sp) <= 1:
        return True
    return False


# LLM-as-Judge grading template (adapted from PPP-Agent)
GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct based on the [correct_answer].

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must follow this format:

extracted_final_answer: The final exact answer extracted from the [response]. Put 'None' if there is no exact answer.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer]. Focus only on meaningful differences. Do not solve the problem or argue for a different answer.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer], contains all essential information, or is equivalent despite minor differences (name order, Inc/Ltd, diacritics, punctuation, case, quotation marks, articles like "a"/"the", standard name shortenings, interchangeable punctuation such as ":" / "-" / "–", presence/absence of subtitles, minor spacing). Answer 'no' only if the answer is factually incorrect, missing essential information, or contradicts the [correct_answer].
""".strip()


def parse_judge_response(judge_response: str) -> dict:
    """Parse the LLM judge response to extract the verdict."""
    result = {"correct": None, "extracted_answer": None, "reasoning": None, "parse_error": False}

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract correct (yes/no)
    for pattern in [
        r"\*\*correct:\*\*\s*(yes|no)",
        r"\*\*correct\*\*:\s*(yes|no)",
        r"correct:\s*(yes|no)",
    ]:
        match = re.search(pattern, judge_response, re.IGNORECASE)
        if match:
            result["correct"] = match.group(1).lower() == "yes"
            break

    # Extract extracted_final_answer
    for pattern in [
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ]:
        match = re.search(pattern, judge_response, re.IGNORECASE | re.DOTALL)
        if match:
            result["extracted_answer"] = match.group(1).strip()
            break

    # Extract reasoning
    for pattern in [
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct|\ncorrect:|$)",
        r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct|\ncorrect:|$)",
        r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
    ]:
        match = re.search(pattern, judge_response, re.IGNORECASE | re.DOTALL)
        if match:
            result["reasoning"] = match.group(1).strip()
            break

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def llm_judge(
    question: str,
    predicted: str,
    ground_truth: str,
    judge_model: str = "gpt-5.1",
    max_retries: int = 3,
) -> dict:
    """
    Use LLM as judge to evaluate answer correctness.

    Returns dict with 'correct' (bool), 'reasoning', 'extracted_answer', 'judge_raw'.
    """
    deployment_name = resolve_model_name(judge_model)
    use_responses = _is_responses_api_model(judge_model)
    client = get_client(use_responses_api=use_responses)

    prompt = GRADER_TEMPLATE.format(
        question=question,
        response=predicted,
        correct_answer=ground_truth,
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            if use_responses:
                resp = _accounted_api_call(
                    client.responses.create,
                    {"model": deployment_name, "input": messages},
                    requested_model=judge_model,
                    resolved_model=deployment_name,
                    api="responses",
                )
                judge_text = _responses_final_content(resp)
            else:
                response = _accounted_api_call(
                    client.chat.completions.create,
                    {
                        "model": deployment_name,
                        "messages": messages,
                        "temperature": 0.0,
                    },
                    requested_model=judge_model,
                    resolved_model=deployment_name,
                    api="chat.completions",
                )
                judge_text = _chat_final_content(response)
            parsed = parse_judge_response(judge_text)

            if not parsed["parse_error"]:
                return {
                    "correct": parsed["correct"],
                    "reasoning": parsed["reasoning"],
                    "extracted_answer": parsed["extracted_answer"],
                    "judge_raw": judge_text,
                }
        except (LLMAccountingError, LLMIncompleteResponse):
            raise
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                match = re.search(r"[Tt]ry again in (\d+) seconds", error_str)
                delay = int(match.group(1)) + 1 if match else min(2 ** attempt, 60)
                time.sleep(delay)
            else:
                logger.warning(f"LLM judge failed after {max_retries} attempts: {e}")

    # Fallback: if all retries fail, return uncertain
    return {"correct": False, "reasoning": "Judge failed", "extracted_answer": None, "judge_raw": None}


def check_answer_browsecomp(
    predicted: Optional[str],
    ground_truth: str,
    question: str = "",
    judge_model: str = "gpt-5.1",
) -> dict:
    """
    2-tier answer checking: EM score first, then LLM judge if EM fails.

    Returns dict with:
        - correct (bool): whether the answer is correct
        - method (str): 'em' or 'llm_judge' or 'no_prediction'
        - judge_result (dict|None): LLM judge details if used
    """
    if predicted is None or predicted.strip() == "":
        return {"correct": False, "method": "no_prediction", "judge_result": None}

    # Tier 1: EM check
    if em_score(ground_truth, predicted):
        return {"correct": True, "method": "em", "judge_result": None}

    # Tier 2: LLM judge
    judge_result = llm_judge(
        question=question,
        predicted=predicted,
        ground_truth=ground_truth,
        judge_model=judge_model,
    )
    return {
        "correct": judge_result["correct"],
        "method": "llm_judge",
        "judge_result": judge_result,
    }


# =============================================================================
# Single Sample Evaluation
# =============================================================================

def evaluate_sample(
    sample,  # IntentSample
    model: str,
    retriever: Any,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_search_iterations: int = 51,
    judge_model: str = "gpt-5.1",
    reasoning_effort: str | None = None,
    max_tool_calls: int | None = None,
    no_retrieval_history: bool = False,
    force_final_answer: bool = False,
) -> dict[str, Any]:
    """Evaluate a single BrowseComp+ sample."""

    result = run_multi_turn_browsecomp(
        sample=sample,
        model=model,
        retriever=retriever,
        temperature=temperature,
        max_tokens=max_tokens,
        max_search_iterations=max_search_iterations,
        reasoning_effort=reasoning_effort,
        max_tool_calls=max_tool_calls,
        no_retrieval_history=no_retrieval_history,
        force_final_answer=force_final_answer,
    )

    # Extract the question from the first user turn for the judge
    question = ""
    for turn in sample.turns:
        if turn.get("role") == "user":
            question = turn["content"]
            break

    grading = check_answer_browsecomp(
        result["prediction"], sample.label, question=question, judge_model=judge_model,
    )

    return {
        "task_id": sample.task_id,
        "prediction": result["prediction"],
        "confidence": result.get("confidence"),
        "correct": grading["correct"],
        "grading_method": grading["method"],
        "judge_result": grading.get("judge_result"),
        "ground_truth": sample.label,
        "success": result["success"],
        "error": result["error"],
        "total_tool_calls": result["total_tool_calls"],
        "retrieved_docids": result["retrieved_docids"],
        "responses": result["responses"],
        "user_messages": result.get("user_messages", []),
        "metadata": sample.metadata,
    }


# =============================================================================
# Experiment Runner
# =============================================================================

def infer_scenario(num_turns, num_revisions, num_switches):
    if num_revisions > 0 and num_switches > 0:
        return "combined"
    if num_revisions > 0:
        return "argument_revision"
    elif num_switches > 0:
        return "function_switch"
    elif num_turns == 1:
        return "fully_specified"
    else:
        return "under_specified"


def run_experiment(
    data_path: str,
    dataset_name: str,
    model: str,
    retriever: Any,
    num_turns: int = 1,
    num_revisions: int = 0,
    num_switches: int = 0,
    ordering: str = "interleaved",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    num_samples: Optional[int] = None,
    num_workers: int = 1,
    task_ids_file: Optional[str] = None,
    max_search_iterations: int = 51,
    naturalizer_model: str | None = None,
    judge_model: str = "gpt-5.1",
    reasoning_effort: str | None = None,
    max_tool_calls: int | None = 50,
    recap_method: str | None = None,
    no_retrieval_history: bool = False,
    force_final_answer: bool = False,
    expected_samples: int | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run a single BrowseComp+ experiment configuration."""

    scenario = infer_scenario(num_turns, num_revisions, num_switches)

    # Output filename (strip quota prefix for clean naming)
    save_name = clean_model_name(model)
    if scenario == "fully_specified":
        output_filename = f"{save_name}.json"
    elif scenario == "under_specified":
        output_filename = f"{save_name}_t{num_turns}.json"
    elif scenario == "argument_revision":
        output_filename = f"{save_name}_t{num_turns}_p{num_revisions}_{ordering}.json"
    elif scenario == "function_switch":
        output_filename = f"{save_name}_t{num_turns}_g{num_switches}.json"
    elif scenario == "combined":
        output_filename = f"{save_name}_t{num_turns}_g{num_switches}_p{num_revisions}.json"
    else:
        output_filename = f"{save_name}.json"

    # Append _naturalized suffix when using online naturalizer
    if naturalizer_model:
        output_filename = output_filename.replace(".json", "_naturalized.json")

    # Append recap method suffix
    if recap_method:
        output_filename = output_filename.replace(".json", f"_recap-{recap_method}.json")

    # Append reasoning effort suffix (e.g., _reasoning-low, _reasoning-medium)
    if reasoning_effort is not None:
        output_filename = output_filename.replace(".json", f"_reasoning-{reasoning_effort}.json")

    # Append no-tool-history suffix when prior-turn tool calls / retrieved docs
    # are dropped between user turns (--no_retrieval_history).
    if no_retrieval_history:
        output_filename = output_filename.replace(".json", "_no-tool-history.json")

    # Append force-final suffix when an extra LLM call coaxes a final answer
    # after the agentic loop exhausts its search budget (--force_final_answer).
    if force_final_answer:
        output_filename = output_filename.replace(".json", "_force-final.json")

    # Use "combined_independent" folder to separate from legacy coupled
    # results, matching run_experiment.py.
    dir_scenario = "combined_independent" if scenario == "combined" else scenario
    exp_dir = EXPERIMENTS_DIR / dir_scenario / dataset_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / output_filename

    # Load task_ids filter
    task_ids = None
    if task_ids_file:
        with open(task_ids_file, 'r') as f:
            task_ids_data = json.load(f)
            task_ids = task_ids_data.get('task_ids', task_ids_data) if isinstance(task_ids_data, dict) else task_ids_data
        if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
            raise ValueError("task_ids_file must contain a list of string task IDs")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_ids_file contains duplicate task IDs")

    # Load simulator with search-agent instruction
    sim = EvolvingIntent(
        data_path=data_path,
        mode="eval",
        domain="search",
        num_turns=num_turns,
        num_revisions=num_revisions,
        num_switches=num_switches,
        ordering=ordering,
        instruction=SEARCH_AGENT_INSTRUCTION,
        task_ids=task_ids,
        naturalizer_model=naturalizer_model,
        recap_method=recap_method,
    )

    samples = list(sim)
    if num_samples:
        samples = samples[:num_samples]
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError(
            f"expected exactly {expected_samples} selected samples, found {len(samples)}"
        )

    print(f"\n{'='*60}")
    print(f"BrowseComp+ Evaluation")
    print(f"{'='*60}")
    print(f"Scenario: {scenario}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"Samples: {len(samples)}")
    print(f"Turns: {num_turns}")
    if num_revisions > 0:
        print(f"Revisions: {num_revisions}")
    if num_switches > 0:
        print(f"Switches: {num_switches}")
    print(f"Temperature: {temperature}")
    print(f"Max search iterations: {max_search_iterations} (per turn)")
    if max_tool_calls is not None:
        print(f"Max tool calls: {max_tool_calls} (per turn)")
    else:
        print(f"Max tool calls: unlimited")
    if recap_method:
        print(f"Recap method: {recap_method}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    output_path = Path(output_path)
    checkpoint_dir = Path(f"{output_path}.checkpoints")
    write_lock = threading.Lock()
    strict_bound_resume = (
        model == PLAN_A_MODEL
        and os.environ.get("LLM_LOCKED_MODEL", "").strip() == PLAN_A_MODEL
    )
    if strict_bound_resume and reasoning_effort != PLAN_A_REASONING_EFFORT:
        raise ValueError(
            f"Plan A checkpoints require reasoning_effort={PLAN_A_REASONING_EFFORT}"
        )
    resolved_model = resolve_model_name(model)
    if strict_bound_resume and resolved_model != PLAN_A_MODEL:
        raise ValueError(
            f"Plan A requires resolved model {PLAN_A_MODEL!r}, got {resolved_model!r}"
        )
    if strict_bound_resume:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[2]
    checkpoint_input_hashes = {
        sample.task_id: _sample_input_sha256(sample) for sample in samples
    }
    checkpoint_policy = {
        "workflow": "browsecomp-plan-a-evaluation",
        "schema_version": PLAN_A_CHECKPOINT_SCHEMA_VERSION,
        "requested_model": model,
        "resolved_model": resolved_model,
        "reasoning_effort": reasoning_effort,
        "scenario": {
            "name": scenario,
            "num_turns": num_turns,
            "num_revisions": num_revisions,
            "num_switches": num_switches,
            "ordering": ordering,
        },
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_search_iterations": max_search_iterations,
        "max_tool_calls": max_tool_calls,
        "naturalizer_model": naturalizer_model,
        "judge_model": judge_model,
        "recap_method": recap_method,
        "no_retrieval_history": no_retrieval_history,
        "force_final_answer": force_final_answer,
        "dataset_sha256": _file_sha256_or_marker(data_path),
        "task_ids_sha256": _file_sha256_or_marker(task_ids_file),
        "source_bundle_sha256": _source_bundle_sha256(
            [
                Path(__file__).resolve(),
                repository_root / "situated_simulation" / "user_simulation.py",
                repository_root / "situated_simulation" / "naturalizer.py",
                repository_root
                / "intent_construction"
                / "intent_extraction"
                / "core"
                / "llm_utils.py",
            ]
        ),
        "instruction_sha256": hashlib.sha256(
            SEARCH_AGENT_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "retriever": _retriever_checkpoint_policy(retriever),
        "selected_task_ids": [sample.task_id for sample in samples],
    }

    # Recover both the aggregate and per-task checkpoints. A failed or empty
    # task result remains visible but is treated as pending on the next run.
    if strict_bound_resume:
        if output_path.exists() and not checkpoint_dir.exists():
            raise ValueError(
                "Plan A found an unbound legacy aggregate without checkpoints; "
                "use a new output directory"
            )
        results = _load_bound_task_checkpoints(
            checkpoint_dir,
            input_hashes=checkpoint_input_hashes,
            policy=checkpoint_policy,
        )
    else:
        results = _load_result_mapping(output_path)
        results.update(_load_task_checkpoints(checkpoint_dir))
    selected_ids = {sample.task_id for sample in samples}
    unexpected_ids = set(results) - selected_ids
    if unexpected_ids:
        raise ValueError(
            f"existing result artifact contains {len(unexpected_ids)} unexpected task IDs"
        )
    completed_ids = {
        task_id
        for task_id, result in results.items()
        if _result_is_complete(result, expected_responses=num_turns)
    }
    pending_samples = [sample for sample in samples if sample.task_id not in completed_ids]
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} complete, {len(pending_samples)} pending")

    def _save_incremental() -> None:
        with write_lock:
            atomic_write_json(output_path, results)

    def _save_task_result(result: dict[str, Any]) -> None:
        task_id = result["task_id"]
        with write_lock:
            checkpoint_payload: dict[str, Any] = result
            if strict_bound_resume:
                checkpoint_payload = _evaluation_checkpoint_envelope(
                    task_id=task_id,
                    input_sha256=checkpoint_input_hashes[task_id],
                    policy=checkpoint_policy,
                    result=result,
                )
            atomic_write_json(
                _task_checkpoint_path(checkpoint_dir, task_id),
                checkpoint_payload,
            )
            results[task_id] = result
            atomic_write_json(output_path, results)

    def _failure_result(sample, error: BaseException) -> dict[str, Any]:
        return {
            "task_id": sample.task_id,
            "prediction": None,
            "correct": False,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "responses": [],
            "total_tool_calls": 0,
            "retrieved_docids": [],
            "user_messages": [],
            "metadata": getattr(sample, "metadata", {}),
        }

    _save_incremental()

    stop_event = threading.Event()

    def _evaluate_and_checkpoint(sample) -> dict[str, Any]:
        if stop_event.is_set():
            raise CancelledError("evaluation cancelled after another task failed")
        try:
            result = evaluate_sample(
                sample, model, retriever, temperature, max_tokens,
                max_search_iterations, judge_model,
                reasoning_effort, max_tool_calls, no_retrieval_history,
                force_final_answer,
            )
        except Exception as error:
            _save_task_result(_failure_result(sample, error))
            if fail_fast:
                stop_event.set()
            raise
        _save_task_result(result)
        if fail_fast and not _result_is_complete(
            result, expected_responses=num_turns
        ):
            stop_event.set()
            raise LLMIncompleteResponse(
                f"task {result['task_id']} did not produce a complete response"
            )
        return result

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_evaluate_and_checkpoint, sample): sample
                for sample in pending_samples
            }

            for future in tqdm(as_completed(futures), total=len(pending_samples), desc=f"{model}"):
                try:
                    future.result()
                except CancelledError:
                    continue
                except Exception as e:
                    sample = futures[future]
                    if fail_fast:
                        stop_event.set()
                        for pending in futures:
                            pending.cancel()
                        raise
                    import traceback; traceback.print_exc()
                    print(f"\n❌ Error: {sample.task_id}: {e}")
    else:
        for sample in tqdm(pending_samples, desc=f"{model}"):
            try:
                _evaluate_and_checkpoint(sample)
            except Exception as e:
                if fail_fast:
                    raise
                import traceback; traceback.print_exc()
                print(f"\n❌ Error: {sample.task_id}: {e}")

    _save_incremental()

    # Summary
    total = len(results)
    correct = sum(1 for r in results.values() if r.get("correct", False))
    failed = sum(1 for r in results.values() if not r.get("success", True))
    accuracy = correct / total if total > 0 else 0.0
    avg_tool_calls = (
        sum(r.get("total_tool_calls", 0) for r in results.values()) / total
        if total > 0 else 0.0
    )

    # Grading method breakdown
    em_correct = sum(1 for r in results.values() if r.get("grading_method") == "em")
    llm_judged = sum(1 for r in results.values() if r.get("grading_method") == "llm_judge")
    llm_correct = sum(1 for r in results.values()
                      if r.get("grading_method") == "llm_judge" and r.get("correct", False))

    summary = {
        "scenario": scenario,
        "dataset": dataset_name,
        "model": model,
        "num_turns": num_turns,
        "num_revisions": num_revisions,
        "num_switches": num_switches,
        "total_samples": total,
        "correct": correct,
        "failed": failed,
        "accuracy": accuracy,
        "avg_tool_calls": avg_tool_calls,
        "em_correct": em_correct,
        "llm_judged": llm_judged,
        "llm_correct": llm_correct,
        "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n✓ Results: {output_path}")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"  Grading: {em_correct} EM correct, {llm_judged} LLM-judged ({llm_correct} correct)")
    print(f"  Avg tool calls: {avg_tool_calls:.1f}")
    if failed > 0:
        print(f"  ⚠ Failed: {failed}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BrowseComp+ evaluation with agentic search")

    parser.add_argument("--data_path", required=True, help="Path to predecessor BrowseComp+ JSON")
    parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate")
    parser.add_argument("--dataset_name", required=True, help="Dataset name for output folder")

    # Scenario parameters
    parser.add_argument("--num_turns", type=int, default=1)
    parser.add_argument("--num_revisions", type=int, default=0)
    parser.add_argument("--num_switches", type=int, default=0)
    parser.add_argument("--ordering", default="interleaved")

    # Execution parameters
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Deprecated compatibility option; Plan A sends no output-token limit",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--task_ids_file", type=str, default=None)

    # Retriever parameters
    parser.add_argument("--index_pattern", default=DEFAULT_INDEX_PATTERN)
    parser.add_argument("--embedding_model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--corpus_dataset", default=DEFAULT_CORPUS_DATASET)
    parser.add_argument("--retriever_device", default="cuda:0")
    parser.add_argument(
        "--retriever_url", default=None,
        help="Pinned retriever HTTP endpoint; skips in-process model/index loading",
    )
    parser.add_argument(
        "--retriever_revision",
        default=None,
        help="Expected immutable retriever deployment revision",
    )
    parser.add_argument(
        "--retriever_bearer_token", default=os.environ.get("BROWSECOMP_RETRIEVER_TOKEN"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--snippet_max_tokens", type=int, default=512)
    parser.add_argument("--search_k", type=int, default=5)
    parser.add_argument("--max_search_iterations", type=int, default=51,
                        help="Max LLM round-trips per turn for agentic search")
    parser.add_argument("--max_tool_calls", type=int, default=PLAN_A_SEARCH_CAP,
                        help="Hard cap on tool calls per turn (Plan A: 50)")
    parser.add_argument("--run_grid", action="store_true",
                        help="Run all 16 grid configs (g=0..3 × p=0..3) with shared retriever")
    parser.add_argument("--run_plan", action="store_true",
                        help="Run planned 24 configs (log-scale sweep) with shared retriever")
    parser.add_argument("--run_plan_lite", action="store_true",
                        help="Run the 6-config lite plan (1 baseline + 1 under-spec + "
                             "1 function-switch + 1 cond-change + 2 combined) with shared retriever")
    parser.add_argument("--naturalizer_model", type=str, default=None,
                        help="Model for online naturalizer (default: None = use pre-built turns)")
    parser.add_argument("--judge_model", type=str, default=PLAN_A_MODEL,
                        help="Model for answer judging after deterministic matching")
    parser.add_argument("--locked_model", type=str, default=None,
                        help="Require agent, naturalizer, and judge to use this one model")
    parser.add_argument("--expected_samples", type=int, default=None,
                        help="Fail unless the selected evaluation set has this exact size")
    parser.add_argument("--fail_fast", action="store_true",
                        help="Abort after an empty, incomplete, or failed task response")
    parser.add_argument("--require_usage_ledger", action="store_true",
                        help="Require LLM_USAGE_LEDGER_PATH and audit appended model IDs")
    parser.add_argument("--usage_only_accounting", action="store_true",
                        help="Reject LLM_COST_HARD_CAP_USD; record usage without a spend gate")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort for GPT-5 models (default: None = model default)")
    parser.add_argument("--recap_method", type=str, default=None,
                        choices=["prompt", "dump", "ground_truth"],
                        help="Recap method: prompt (CoT nudge), dump (paste prior turns), ground_truth (exact state)")
    parser.add_argument("--no_retrieval_history", action="store_true",
                        help="Do not carry over retrieved documents across turns (only keep final assistant response)")
    parser.add_argument("--force_final_answer", action="store_true",
                        help="When max_search_iterations is exhausted without a final textual "
                             "answer, make one extra text-only LLM call with a final-answer "
                             "instruction and tool_choice='none' to coax the model "
                             "into producing the "
                             "required 'Exact Answer: ...' format. Saves results with a "
                             "_force-final suffix to avoid clobbering baseline runs.")

    args = parser.parse_args()

    # Validate models
    for model in args.models:
        if not model:
            print(f"❌ Unsupported model: {model!r}")
            print("   Set a model id served by your OpenAI / Azure OpenAI account "
                  "(e.g. gpt-5.1).")
            return
    if args.max_tokens is not None:
        raise ValueError("Plan A does not allow client-supplied output-token limits")
    if args.locked_model:
        _assert_model_lock(
            args.models,
            args.naturalizer_model,
            args.judge_model,
            args.locked_model,
        )
        _lock_model_environment(args.locked_model)
    if args.locked_model == PLAN_A_MODEL:
        if args.reasoning_effort != "medium":
            raise ValueError("Plan A requires --reasoning_effort medium")
        configured_effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
        if configured_effort and configured_effort != "medium":
            raise ValueError("Plan A requires LLM_REASONING_EFFORT=medium")
        os.environ["LLM_REASONING_EFFORT"] = "medium"
        if args.max_tool_calls != PLAN_A_SEARCH_CAP:
            raise ValueError("Plan A requires exactly 50 search calls per turn")
        if args.max_search_iterations != PLAN_A_SEARCH_CAP + 1:
            raise ValueError("Plan A requires exactly 51 model round trips per turn")
        if args.expected_samples != PLAN_A_SAMPLE_COUNT:
            raise ValueError("Plan A requires the fixed 100-task evaluation set")
        if args.num_samples != PLAN_A_SAMPLE_COUNT:
            raise ValueError("Plan A requires --num_samples 100")
        if not args.force_final_answer or not args.fail_fast:
            raise ValueError("Plan A requires --force_final_answer and --fail_fast")
        if not args.require_usage_ledger or not args.usage_only_accounting:
            raise ValueError("Plan A requires usage-only ledger accounting")
        if not args.retriever_url:
            raise ValueError("Plan A requires the pinned retriever endpoint")
        if args.retriever_revision != PLAN_A_RETRIEVER_REVISION:
            raise ValueError(
                "Plan A requires the pinned retriever revision "
                f"{PLAN_A_RETRIEVER_REVISION}"
            )
        if args.search_k != 5:
            raise ValueError("Plan A fixes retriever search_k at 5")
        if args.run_grid or args.run_plan or args.run_plan_lite:
            raise ValueError("Plan A runs only the paper single and evolve scenarios")
        scenario_key = (args.num_turns, args.num_revisions, args.num_switches)
        if scenario_key not in PLAN_A_SCENARIOS:
            raise ValueError(
                "Plan A scenarios are single (t=1,p=0,g=0) and "
                "evolve (t=7,p=2,g=2)"
            )
        if not args.task_ids_file:
            raise ValueError("Plan A requires the fixed published task ID file")
        supplied_ids = _read_task_ids(Path(args.task_ids_file))
        fixed_ids = _read_task_ids(PLAN_A_TASK_IDS_FILE)
        if supplied_ids != fixed_ids or len(supplied_ids) != PLAN_A_SAMPLE_COUNT:
            raise ValueError("Plan A task IDs do not match the published fixed 100 IDs")
    if args.usage_only_accounting and os.environ.get("LLM_COST_HARD_CAP_USD", "").strip():
        raise ValueError("usage-only accounting forbids LLM_COST_HARD_CAP_USD")
    if args.locked_model and os.environ.get("LLM_DISABLE_OUTPUT_LIMITS", "").lower() not in {
        "1", "true", "yes", "on",
    }:
        raise ValueError("Plan A requires LLM_DISABLE_OUTPUT_LIMITS=1")
    ledger_marker = _usage_ledger_marker(required=args.require_usage_ledger)
    if args.require_usage_ledger:
        os.environ["LLM_REQUIRE_USAGE_ACCOUNTING"] = "1"

    # Initialize retriever (shared across all experiments)
    if args.retriever_url:
        print("Initializing pinned retriever client...")
        retriever = RemoteRetriever(
            args.retriever_url,
            k=args.search_k,
            bearer_token=args.retriever_bearer_token,
            revision=args.retriever_revision,
        )
    else:
        print("Initializing local FAISS retriever...")
        retriever = FaissRetriever(
            index_pattern=args.index_pattern,
            model_name=args.embedding_model,
            corpus_dataset=args.corpus_dataset,
            snippet_max_tokens=args.snippet_max_tokens,
            k=args.search_k,
            device=args.retriever_device,
        )
    print("✓ Retriever ready")

    # Run experiments
    all_summaries = []

    if args.run_grid:
        # Grid mode: run all 16 g×p configs for each model
        GRID_CONFIGS = [
            (0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 4),
            (1, 0, 2), (1, 1, 3), (1, 2, 4), (1, 3, 5),
            (2, 0, 3), (2, 1, 4), (2, 2, 5), (2, 3, 6),
            (3, 0, 4), (3, 1, 5), (3, 2, 6), (3, 3, 7),
        ]
        for model in args.models:
            print(f"\n{'='*60}")
            print(f"Grid evaluation for: {model}")
            print(f"{'='*60}")
            for i, (g, p, t) in enumerate(GRID_CONFIGS):
                config_name = f"t{t}_g{g}_p{p}"
                print(f"\n[{i+1}/16] {config_name} (turns={t}, functions={g}, conds={p})")
                try:
                    summary = run_experiment(
                        data_path=args.data_path,
                        dataset_name=args.dataset_name,
                        model=model,
                        retriever=retriever,
                        num_turns=t,
                        num_revisions=p,
                        num_switches=g,
                        ordering=args.ordering,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        num_samples=args.num_samples,
                        num_workers=args.num_workers,
                        task_ids_file=args.task_ids_file,
                        max_search_iterations=args.max_search_iterations,
                        naturalizer_model=args.naturalizer_model,
                        judge_model=args.judge_model,
                        reasoning_effort=args.reasoning_effort,
                        max_tool_calls=args.max_tool_calls,
                        recap_method=args.recap_method,
                        no_retrieval_history=args.no_retrieval_history,
                        force_final_answer=args.force_final_answer,
                        expected_samples=args.expected_samples,
                        fail_fast=args.fail_fast,
                    )
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"\n❌ FAILED: {config_name}: {e}")
                    if args.locked_model:
                        _assert_usage_models_since(ledger_marker, args.locked_model)
                    if args.fail_fast:
                        raise
                    import traceback
                    traceback.print_exc()
    elif args.run_plan:
        # Planned experiment configs (19 configs, minimal turns):
        # (g, p, t) — fully_specified, under_specified, argument_revision,
        # function_switch, combined. All use t=1+g+p (minimal) except under_specified.
        PLAN_CONFIGS = [
            # fully_specified
            (0, 0, 1),
            # under_specified: t=2,4,8
            (0, 0, 2), (0, 0, 4), (0, 0, 8),
            # argument_revision: minimal turns t=1+p
            (0, 1, 2), (0, 2, 3), (0, 3, 4),
            # function_switch: minimal turns t=1+g
            (1, 0, 2), (2, 0, 3), (3, 0, 4),
            # combined: g=1..3 x p=1..3, t=1+g+p
            (1, 1, 3), (1, 2, 4), (1, 3, 5),
            (2, 1, 4), (2, 2, 5), (2, 3, 6),
            (3, 1, 5), (3, 2, 6), (3, 3, 7),
        ]
        # Skip fully-specified (single-turn) when using LLM user simulator,
        # but only if it's not the only config (we always need the baseline).
        # NOTE: We no longer skip (0,0,1) — the fully_specified baseline is
        # needed for the grid display and doesn't invoke the user simulator
        # beyond the initial turn anyway.
        total = len(PLAN_CONFIGS)
        for model in args.models:
            print(f"\n{'='*60}")
            print(f"Plan evaluation for: {model} ({total} configs)")
            print(f"{'='*60}")
            for i, (g, p, t) in enumerate(PLAN_CONFIGS):
                config_name = f"t{t}_g{g}_p{p}"
                print(f"\n[{i+1}/{total}] {config_name} (turns={t}, functions={g}, conds={p})")
                try:
                    summary = run_experiment(
                        data_path=args.data_path,
                        dataset_name=args.dataset_name,
                        model=model,
                        retriever=retriever,
                        num_turns=t,
                        num_revisions=p,
                        num_switches=g,
                        ordering=args.ordering,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        num_samples=args.num_samples,
                        num_workers=args.num_workers,
                        task_ids_file=args.task_ids_file,
                        max_search_iterations=args.max_search_iterations,
                        naturalizer_model=args.naturalizer_model,
                        judge_model=args.judge_model,
                        reasoning_effort=args.reasoning_effort,
                        max_tool_calls=args.max_tool_calls,
                        recap_method=args.recap_method,
                        no_retrieval_history=args.no_retrieval_history,
                        force_final_answer=args.force_final_answer,
                        expected_samples=args.expected_samples,
                        fail_fast=args.fail_fast,
                    )
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"\n❌ FAILED: {config_name}: {e}")
                    if args.locked_model:
                        _assert_usage_models_since(ledger_marker, args.locked_model)
                    if args.fail_fast:
                        raise
                    import traceback
                    traceback.print_exc()
    elif args.run_plan_lite:
        # Lite plan: 6 configs covering all 5 scenarios with one combined-with-reveal
        # (g, p, t) tuples — chosen to give one representative config per scenario,
        # using the canonical n100 dataset (g=3, p>=6 per sample, no drops).
        PLAN_CONFIGS = [
            (0, 0, 1),  # 1. fully-specified
            (0, 0, 3),  # 2. under-specified
            (2, 0, 3),  # 3. function-switch
            (0, 2, 3),  # 4. argument-revision
            (2, 2, 5),  # 5. combined, no reveal (t = 1 + g + p)
            (2, 2, 7),  # 6. combined, with 2 extra reveal-only turns
        ]
        total = len(PLAN_CONFIGS)
        for model in args.models:
            print(f"\n{'='*60}")
            print(f"Plan-lite evaluation for: {model} ({total} configs)")
            print(f"{'='*60}")
            for i, (g, p, t) in enumerate(PLAN_CONFIGS):
                config_name = f"t{t}_g{g}_p{p}"
                print(f"\n[{i+1}/{total}] {config_name} (turns={t}, functions={g}, conds={p})")
                try:
                    summary = run_experiment(
                        data_path=args.data_path,
                        dataset_name=args.dataset_name,
                        model=model,
                        retriever=retriever,
                        num_turns=t,
                        num_revisions=p,
                        num_switches=g,
                        ordering=args.ordering,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        num_samples=args.num_samples,
                        num_workers=args.num_workers,
                        task_ids_file=args.task_ids_file,
                        max_search_iterations=args.max_search_iterations,
                        naturalizer_model=args.naturalizer_model,
                        judge_model=args.judge_model,
                        reasoning_effort=args.reasoning_effort,
                        max_tool_calls=args.max_tool_calls,
                        recap_method=args.recap_method,
                        no_retrieval_history=args.no_retrieval_history,
                        force_final_answer=args.force_final_answer,
                        expected_samples=args.expected_samples,
                        fail_fast=args.fail_fast,
                    )
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"\n❌ FAILED: {config_name}: {e}")
                    if args.locked_model:
                        _assert_usage_models_since(ledger_marker, args.locked_model)
                    if args.fail_fast:
                        raise
                    import traceback
                    traceback.print_exc()
    else:
        for model in args.models:
            try:
                summary = run_experiment(
                    data_path=args.data_path,
                    dataset_name=args.dataset_name,
                    model=model,
                    retriever=retriever,
                    num_turns=args.num_turns,
                    num_revisions=args.num_revisions,
                    num_switches=args.num_switches,
                    ordering=args.ordering,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    num_samples=args.num_samples,
                    num_workers=args.num_workers,
                    task_ids_file=args.task_ids_file,
                    max_search_iterations=args.max_search_iterations,
                    naturalizer_model=args.naturalizer_model,
                    judge_model=args.judge_model,
                    reasoning_effort=args.reasoning_effort,
                    max_tool_calls=args.max_tool_calls,
                    recap_method=args.recap_method,
                    no_retrieval_history=args.no_retrieval_history,
                    force_final_answer=args.force_final_answer,
                    expected_samples=args.expected_samples,
                    fail_fast=args.fail_fast,
                )
                all_summaries.append(summary)
            except Exception as e:
                print(f"\n❌ Error with {model}: {e}")
                if args.locked_model:
                    _assert_usage_models_since(ledger_marker, args.locked_model)
                if args.fail_fast:
                    raise
                import traceback
                traceback.print_exc()

    if args.locked_model:
        _assert_usage_models_since(ledger_marker, args.locked_model)

    if all_summaries:
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"\n{'Model':<20} {'Accuracy':<15} {'Avg Tools':<12} {'Samples':<10}")
        print("-"*60)
        for s in all_summaries:
            print(f"{s['model']:<20} {s['accuracy']:.2%} ({s['correct']}/{s['total_samples']}) "
                  f"{s['avg_tool_calls']:<12.1f}")


if __name__ == "__main__":
    main()
