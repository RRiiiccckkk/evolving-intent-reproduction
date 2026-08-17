"""
LLM utility functions for Function & Argument extraction.

Calls go through OpenAI, Azure OpenAI, or an OpenAI-compatible API, selected
from environment variables:

- **OpenAI**:        ``OPENAI_API_KEY``
- **Azure OpenAI**:  ``AZURE_OPENAI_API_KEY`` + ``AZURE_OPENAI_ENDPOINT``
- **Compatible**:    ``LLM_API_KEY`` + ``LLM_BASE_URL``

When more than one is configured, set
``LLM_BACKEND=openai|azure|compatible`` to disambiguate. Compatible providers
can optionally map logical model names with ``LLM_MODEL_MAP`` (a JSON object).

Set ``LLM_USAGE_LEDGER_PATH`` to append per-response token usage as JSONL.
For a fail-closed cost limit, also set ``LLM_COST_HARD_CAP_USD`` and
``LLM_PRICE_MAP`` (JSON; prices are USD per one million tokens).
See :func:`get_client` and :func:`resolve_model_name`.
"""
import os
import json
import math
import random
import threading
import time
import uuid
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Mapping, Optional
from openai import AzureOpenAI, OpenAI


class LLMAccountingError(RuntimeError):
    """Raised when configured usage accounting cannot be completed safely."""


class LLMBudgetExceeded(LLMAccountingError):
    """Raised before or after a call that would exceed the configured cap."""


class LLMIncompleteResponse(RuntimeError):
    """Raised when a provider returns reasoning but no complete final answer."""


_ACCOUNTING_LOCK = threading.RLock()
_CLIENT_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=1800.0,
    write=600.0,
    pool=600.0,
)
_CLIENT_MAX_RETRIES = 0


# =============================================================================
# Retry/back-off policy
# =============================================================================
#
# Single source of truth for how long to wait before retrying a failed
# OpenAI / Azure OpenAI API call. Implements the best-practice retry policy
# (provider rate-limit advisory):
#
#   1. Honor the ``Retry-After`` HTTP header when the server provides it.
#   2. Fall back to parsing the error body (older deployments / SDKs
#      surface the cool-off time in the message rather than the header).
#   3. Final fallback is exponential backoff (``2**attempt`` capped).
#   4. Add uniform 0-50% jitter so N concurrent workers don't all wake
#      up at the same instant and re-create the same thundering-herd
#      storm the server just throttled.
#
# Every retry call site in this module routes through
# ``_compute_retry_wait`` so the policy stays consistent across
# generate_text / generate_json / generate_multi_turn /
# generate_with_tools / generate_with_tools_responses.

def _compute_retry_wait(e: Exception, attempt: int = 0, max_cap: int = 120) -> float:
    """Compute jittered wait time for a retry, honoring ``Retry-After``.

    Args:
        e: The exception raised by the API call. If it has a
           ``response`` attribute (httpx-style), we try to read the
           ``Retry-After`` header from it.
        attempt: 0-based retry attempt index, used for exponential
           backoff when no server hint is available.
        max_cap: Upper bound (seconds) for the exponential-backoff
           fallback. The server's own ``Retry-After`` hint is honored
           even if it exceeds ``max_cap`` (it knows its quota window
           better than we do).

    Returns:
        Wait time in seconds, including jitter and a 1s safety margin.
    """
    base_wait: Optional[float] = None

    # 1. Retry-After header from the HTTP response (RFC 7231 — seconds).
    response = getattr(e, "response", None)
    if response is not None:
        try:
            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                base_wait = float(retry_after)
        except (AttributeError, ValueError, TypeError):
            pass

    # 2. Body regex (some providers only put the hint here, not in headers).
    if base_wait is None:
        import re as _re
        error_str = str(e)
        m = (
            _re.search(r"[Tt]ry again in (\d+) seconds", error_str)
            or _re.search(r"retry after (\d+) seconds", error_str)
        )
        if m:
            base_wait = float(m.group(1))

    # 3. Exponential backoff fallback (capped).
    if base_wait is None:
        base_wait = float(min(2 ** attempt, max_cap))

    # Jitter: uniform 0-50% on top to break thundering-herd retry storms.
    jitter = random.uniform(0, base_wait * 0.5)

    # +1s margin so we clear the server's rate-limit window cleanly.
    return base_wait + jitter + 1.0


# =============================================================================
# Backend configuration (OpenAI / Azure OpenAI / OpenAI-compatible)
# =============================================================================
#
# Three backends are supported, selected from environment variables:
#   - OpenAI:        OPENAI_API_KEY
#   - Azure OpenAI:  AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
#   - Compatible:    LLM_API_KEY + LLM_BASE_URL
#
# When more than one is configured, set LLM_BACKEND=openai|azure|compatible to
# disambiguate. The legacy automatic order remains Azure, then OpenAI, then the
# compatible provider. Azure API versions and model/deployment mappings are
# configurable (see below).

AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_RESPONSES_API_VERSION = os.environ.get(
    "AZURE_OPENAI_RESPONSES_API_VERSION", "2025-03-01-preview"
)

# Models that only support the Responses API (not chat completions)
RESPONSES_API_MODELS = {"codex", "pro"}

# Model identifier substrings that require max_completion_tokens instead of
# max_tokens (reasoning models whose token budget covers thinking + output)
REASONING_MODEL_TAGS = {"gpt-5", "o3", "kimi"}


def clean_model_name(model: str) -> str:
    """Return the canonical model id used for API calls and output filenames.

    This is a normalization hook: it currently returns the model id
    unchanged, but provides a single place to apply provider-neutral name
    normalization if needed.
    """
    return model


# =============================================================================
# Model capability detectors
# =============================================================================
# These operate on the canonical (cleaned) model id returned by
# ``clean_model_name`` so detection is independent of any name normalization.


def _is_responses_api_model(model: str) -> bool:
    """Check if a model requires the Responses API instead of Chat Completions."""
    m = clean_model_name(model)
    return any(tag in m for tag in RESPONSES_API_MODELS)


def _needs_max_completion_tokens(model: str) -> bool:
    """Check if a model needs max_completion_tokens instead of max_tokens."""
    return any(tag in clean_model_name(model) for tag in REASONING_MODEL_TAGS)


def _is_closed_reasoning_model(model: str) -> bool:
    """Closed-source models with hidden chain-of-thought.

    These models do not return their reasoning trace in plain text. To preserve
    reasoning continuity across turns we must use the Responses API with
    ``include=["reasoning.encrypted_content"]`` and feed the encrypted reasoning
    items back into the next call. White-box models keep their CoT in the
    assistant message text and do not need this treatment.
    """
    m = clean_model_name(model)
    # codex / pro require Responses API and have hidden reasoning
    if any(tag in m for tag in RESPONSES_API_MODELS):
        return True
    # o1 / o3 reasoning families
    if m.startswith(("o1", "o3")):
        return True
    # gpt-5.x reasoning variants. The "-chat" siblings (e.g. gpt-5.1-chat) are
    # non-reasoning chat-only models and do NOT need the Responses API.
    if m.startswith("gpt-5") and "-chat" not in m:
        return True
    # Explicit reasoning variants (e.g. grok-4-1-fast-reasoning). Exclude
    # ``non-reasoning`` siblings which contain ``-reasoning`` as a substring.
    if "-reasoning" in m and "non-reasoning" not in m:
        return True
    return False


def _supports_responses_api(model: str) -> bool:
    """Whether this model can be served via the Responses API.

    Some closed-reasoning models reject Responses API calls (e.g. Grok
    reasoning variants). For those we fall back to chat completions, which
    means the hidden reasoning trace is dropped between turns.
    """
    if not _is_closed_reasoning_model(model):
        return False
    m = clean_model_name(model)
    if "grok" in m.lower():
        return False
    return True


def _requires_responses_api_for_tools(model: str) -> bool:
    """Whether tool calling for this model must use the Responses API.

    As of 2026-05, gpt-5.5 rejects ``tools=`` + ``reasoning_effort=`` on chat
    completions (400: "Function tools with reasoning_effort are not supported
    ... Please use /v1/responses instead."). Earlier gpt-5.x accept
    tools+reasoning on chat completions and stay there.
    """
    if not _supports_responses_api(model):
        return False
    return clean_model_name(model).startswith("gpt-5.5")


def _normalize_chat_payload(
    payload: Mapping[str, Any], resolved_model: str
) -> Dict[str, Any]:
    """Apply narrow provider/model parameter compatibility rules."""
    normalized = dict(payload)
    if normalized.get("reasoning_effort") is None:
        normalized.pop("reasoning_effort", None)

    provider_model = clean_model_name(resolved_model).lower().rsplit("/", 1)[-1]
    is_compatible_kimi_k2 = (
        _select_backend() == "compatible" and provider_model.startswith("kimi-k2.")
    )
    if is_compatible_kimi_k2:
        # Kimi K2 compatible endpoints only accept their default temperature
        # (currently 1), and reject explicit tool_choice values. Omitting both
        # fields is the portable representation. Search stays sequential so a
        # parallel batch cannot overshoot the per-turn budget.
        normalized.pop("temperature", None)
        normalized.pop("tool_choice", None)
        if normalized.get("tools"):
            normalized["parallel_tool_calls"] = False
        else:
            normalized.pop("parallel_tool_calls", None)
        if "max_tokens" in normalized:
            normalized.setdefault(
                "max_completion_tokens", normalized.pop("max_tokens")
            )
    return normalized


# =============================================================================
# Client construction
# =============================================================================


def _select_backend() -> str:
    """Return the active backend: ``openai``, ``azure``, or ``compatible``."""
    backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if backend in ("generic", "openai-compatible", "openai_compatible"):
        backend = "compatible"
    if backend in ("openai", "azure"):
        return backend
    if backend == "compatible":
        return backend
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"):
        return "compatible"
    raise RuntimeError(
        "No LLM credentials found. Set OPENAI_API_KEY; "
        "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT; or "
        "LLM_API_KEY + LLM_BASE_URL. When more than one backend is set, use "
        "LLM_BACKEND=openai|azure|compatible to disambiguate."
    )


def _azure_deployment_map() -> Dict[str, str]:
    """Optional model-id -> Azure-deployment-name map from
    ``AZURE_OPENAI_DEPLOYMENT_MAP`` (JSON object). Empty if unset/invalid."""
    raw = os.environ.get("AZURE_OPENAI_DEPLOYMENT_MAP")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _compatible_model_map() -> Dict[str, str]:
    """Return the logical-model -> provider-model map from ``LLM_MODEL_MAP``.

    Unlike the legacy Azure map, a malformed compatible-provider map raises a
    configuration error. Silently ignoring a new provider's model map can send
    paid traffic to the wrong model.
    """
    raw = os.environ.get("LLM_MODEL_MAP")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAccountingError("LLM_MODEL_MAP must be a JSON object") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in parsed.items()
    ):
        raise LLMAccountingError(
            "LLM_MODEL_MAP must map string logical names to string model ids"
        )
    return parsed


def resolve_model_name(model: str) -> str:
    """Resolve the model/deployment id to send to the API.

    On Azure, applies the optional ``AZURE_OPENAI_DEPLOYMENT_MAP`` so users
    whose Azure deployment names differ from the canonical model id can map
    them explicitly.
    """
    name = clean_model_name(model)
    locked_model = os.environ.get("LLM_LOCKED_MODEL", "").strip()
    if locked_model and name != locked_model:
        raise LLMAccountingError(
            f"model lock requires {locked_model!r}; refused requested model {name!r}"
        )
    backend = _select_backend()
    if backend == "azure":
        resolved = _azure_deployment_map().get(name, name)
    elif backend == "compatible":
        resolved = _compatible_model_map().get(name, name)
    else:
        resolved = name
    if locked_model and resolved != locked_model:
        raise LLMAccountingError(
            f"model lock requires provider model {locked_model!r}; "
            f"refused resolved model {resolved!r}"
        )
    return resolved


def get_client(use_responses_api: bool = False):
    """Create an OpenAI, Azure, or compatible client from credentials.

    Args:
        use_responses_api: Select the Responses API version (Azure only).

    Returns:
        An ``openai.OpenAI`` or ``openai.AzureOpenAI`` client. Compatible
        providers use the standard OpenAI client with a custom ``base_url``.
    """
    backend = _select_backend()
    if backend == "azure":
        if not os.environ.get("AZURE_OPENAI_API_KEY") or not os.environ.get(
            "AZURE_OPENAI_ENDPOINT"
        ):
            raise RuntimeError(
                "LLM_BACKEND=azure requires AZURE_OPENAI_API_KEY and "
                "AZURE_OPENAI_ENDPOINT"
            )
        api_version = (
            AZURE_RESPONSES_API_VERSION if use_responses_api else AZURE_API_VERSION
        )
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=api_version,
            timeout=_CLIENT_TIMEOUT,
            max_retries=_CLIENT_MAX_RETRIES,
        )
    if backend == "compatible":
        if not os.environ.get("LLM_API_KEY") or not os.environ.get("LLM_BASE_URL"):
            raise RuntimeError(
                "LLM_BACKEND=compatible requires LLM_API_KEY and LLM_BASE_URL"
            )
        return OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            timeout=_CLIENT_TIMEOUT,
            max_retries=_CLIENT_MAX_RETRIES,
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("LLM_BACKEND=openai requires OPENAI_API_KEY")
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=_CLIENT_TIMEOUT,
        max_retries=_CLIENT_MAX_RETRIES,
    )


# =============================================================================
# Usage ledger and fail-closed cost cap
# =============================================================================


def _parse_nonnegative_float(raw: Any, setting: str) -> float:
    """Parse a finite, non-negative configuration value."""
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise LLMAccountingError(f"{setting} must be a non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise LLMAccountingError(f"{setting} must be a finite, non-negative number")
    return value


def _hard_cap_usd() -> Optional[float]:
    raw = os.environ.get("LLM_COST_HARD_CAP_USD")
    if raw is None or not raw.strip():
        return None
    return _parse_nonnegative_float(raw, "LLM_COST_HARD_CAP_USD")


def _ledger_path() -> Optional[Path]:
    raw = os.environ.get("LLM_USAGE_LEDGER_PATH", "").strip()
    if not raw:
        return None
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def _usage_accounting_required() -> bool:
    return os.environ.get("LLM_REQUIRE_USAGE_ACCOUNTING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _price_map() -> Dict[str, Any]:
    """Read ``LLM_PRICE_MAP``.

    Each model maps to USD-per-million-token prices. The short keys are
    ``input``, ``output``, ``cached_input``, and ``reasoning``. The latter two
    are optional and fall back to input/output respectively. Explicit keys
    such as ``input_usd_per_1m`` and ``output_usd_per_1m`` are also accepted.
    A ``*`` or ``default`` entry may provide fallback prices.
    """
    raw = os.environ.get("LLM_PRICE_MAP", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAccountingError("LLM_PRICE_MAP must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise LLMAccountingError("LLM_PRICE_MAP must be a JSON object")
    return parsed


_PRICE_ALIASES = {
    "input": (
        "input",
        "prompt",
        "input_per_million",
        "input_per_1m_usd",
        "prompt_per_million",
        "prompt_per_1m_usd",
        "input_usd_per_1m",
        "input_usd_per_million",
        "prompt_usd_per_1m",
    ),
    "output": (
        "output",
        "completion",
        "output_per_million",
        "output_per_1m_usd",
        "completion_per_million",
        "completion_per_1m_usd",
        "output_usd_per_1m",
        "output_usd_per_million",
        "completion_usd_per_1m",
    ),
    "cached_input": (
        "cached_input",
        "cached",
        "cache_read",
        "cached_input_per_million",
        "cached_input_per_1m_usd",
        "cached_input_usd_per_1m",
        "cached_input_usd_per_million",
        "cache_read_usd_per_1m",
    ),
    "reasoning": (
        "reasoning",
        "reasoning_output",
        "reasoning_per_million",
        "reasoning_per_1m_usd",
        "reasoning_usd_per_1m",
        "reasoning_usd_per_million",
    ),
}


def _model_prices(requested_model: str, resolved_model: str) -> Optional[Dict[str, float]]:
    prices_by_model = _price_map()
    raw_entry: Any = None
    matched_name: Optional[str] = None
    for name in (requested_model, resolved_model, "*", "default"):
        if name in prices_by_model:
            raw_entry = prices_by_model[name]
            matched_name = name
            break
    if raw_entry is None:
        return None
    if not isinstance(raw_entry, Mapping):
        raise LLMAccountingError(
            f"LLM_PRICE_MAP[{matched_name!r}] must be a JSON object"
        )

    normalized: Dict[str, float] = {}
    for canonical, aliases in _PRICE_ALIASES.items():
        for alias in aliases:
            if alias in raw_entry:
                normalized[canonical] = _parse_nonnegative_float(
                    raw_entry[alias], f"LLM_PRICE_MAP[{matched_name!r}].{alias}"
                )
                break

    if "input" not in normalized or "output" not in normalized:
        raise LLMAccountingError(
            f"LLM_PRICE_MAP[{matched_name!r}] requires input and output prices"
        )
    normalized.setdefault("cached_input", normalized["input"])
    normalized.setdefault("reasoning", normalized["output"])
    return normalized


def _field(obj: Any, *names: str) -> Any:
    """Get the first non-None field from a dict or SDK model."""
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, Mapping):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _chat_final_content(response: Any) -> str:
    """Return complete Chat Completions content, never hidden reasoning."""
    choices = _field(response, "choices")
    if not choices:
        raise LLMIncompleteResponse("API returned no chat completion choice")
    choice = choices[0]
    finish_reason = _field(choice, "finish_reason")
    if finish_reason in {"length", "max_tokens"}:
        raise LLMIncompleteResponse(
            f"API response was truncated before a final answer ({finish_reason})"
        )
    content = _field(_field(choice, "message"), "content")
    if not isinstance(content, str) or not content.strip():
        raise LLMIncompleteResponse("API returned no final answer content")
    return content


def _responses_final_content(response: Any) -> str:
    """Return completed Responses API text or raise an explicit failure."""
    status = _field(response, "status")
    if status == "incomplete" or _field(response, "incomplete_details") is not None:
        raise LLMIncompleteResponse("Responses API returned an incomplete response")
    content = _field(response, "output_text")
    if not isinstance(content, str) or not content.strip():
        raise LLMIncompleteResponse("Responses API returned no final answer content")
    return content


def _token_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMAccountingError(f"API returned an invalid token count: {value!r}") from exc
    if parsed < 0:
        raise LLMAccountingError(f"API returned a negative token count: {parsed}")
    return parsed


def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
    """Normalize Chat Completions and Responses API token usage."""
    usage = _field(response, "usage")
    if usage is None:
        return None

    input_raw = _field(usage, "input_tokens", "prompt_tokens")
    output_raw = _field(usage, "output_tokens", "completion_tokens")
    if input_raw is None or output_raw is None:
        return None

    input_details = _field(usage, "input_tokens_details", "prompt_tokens_details")
    output_details = _field(
        usage, "output_tokens_details", "completion_tokens_details"
    )
    cached_raw = _field(
        input_details,
        "cached_tokens",
        "cache_read_tokens",
        "prompt_cache_hit_tokens",
    )
    if cached_raw is None:
        cached_raw = _field(
            usage,
            "cached_tokens",
            "cache_read_tokens",
            "prompt_cache_hit_tokens",
        )
    reasoning_raw = _field(output_details, "reasoning_tokens")
    if reasoning_raw is None:
        reasoning_raw = _field(usage, "reasoning_tokens")

    input_tokens = _token_count(input_raw)
    output_tokens = _token_count(output_raw)
    cached_tokens = _token_count(cached_raw)
    reasoning_tokens = _token_count(reasoning_raw)
    if cached_tokens > input_tokens:
        raise LLMAccountingError("API cached token count exceeds input token count")
    if reasoning_tokens > output_tokens:
        raise LLMAccountingError("API reasoning token count exceeds output token count")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
    }


def _usage_cost_usd(usage: Dict[str, int], prices: Dict[str, float]) -> float:
    uncached_input = usage["input_tokens"] - usage["cached_tokens"]
    visible_output = usage["output_tokens"] - usage["reasoning_tokens"]
    cost = (
        uncached_input * prices["input"]
        + usage["cached_tokens"] * prices["cached_input"]
        + visible_output * prices["output"]
        + usage["reasoning_tokens"] * prices["reasoning"]
    ) / 1_000_000
    return round(cost, 12)


def _estimated_payload_tokens(payload: Mapping[str, Any]) -> int:
    """Return a conservative byte-level upper estimate for request tokens."""
    try:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise LLMAccountingError("Unable to estimate request size for budget cap") from exc
    # Byte-level BPE cannot use more tokens than encoded bytes. The fixed
    # margin covers provider-side chat wrappers and special tokens.
    return len(serialized.encode("utf-8")) + 512


def _parse_positive_token_limit(raw: Any, setting: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise LLMAccountingError(f"{setting} must be a positive integer") from exc
    if value <= 0:
        raise LLMAccountingError(f"{setting} must be a positive integer")
    return value


def _configured_default_output_tokens() -> Optional[int]:
    raw = os.environ.get("LLM_DEFAULT_MAX_OUTPUT_TOKENS")
    if raw is None or not raw.strip():
        return None
    return _parse_positive_token_limit(raw, "LLM_DEFAULT_MAX_OUTPUT_TOKENS")


def _output_limits_disabled() -> bool:
    return os.environ.get("LLM_DISABLE_OUTPUT_LIMITS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_output_reservation() -> int:
    raw = os.environ.get("LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS", "4096")
    return _parse_positive_token_limit(
        raw, "LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS"
    )


def _estimated_call_cost_usd(
    payload: Mapping[str, Any],
    max_output_tokens: Optional[int],
    prices: Dict[str, float],
) -> float:
    if max_output_tokens is None:
        output_tokens = _default_output_reservation()
    else:
        try:
            output_tokens = int(max_output_tokens)
        except (TypeError, ValueError) as exc:
            raise LLMAccountingError("Maximum output tokens must be an integer") from exc
        if output_tokens < 0:
            raise LLMAccountingError("Maximum output tokens cannot be negative")
    input_rate = max(prices["input"], prices["cached_input"])
    output_rate = max(prices["output"], prices["reasoning"])
    return (
        _estimated_payload_tokens(payload) * input_rate
        + output_tokens * output_rate
    ) / 1_000_000


def _read_ledger_state(handle: Any, strict: bool) -> tuple[float, Dict[str, float]]:
    """Return paid spend and unresolved reservations from a JSONL ledger."""
    handle.seek(0)
    spent = 0.0
    reservations: Dict[str, float] = {}
    for line_number, raw_line in enumerate(handle, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise LLMAccountingError(
                    f"Usage ledger contains invalid JSON on line {line_number}"
                ) from exc
            continue
        if not isinstance(entry, Mapping):
            if strict:
                raise LLMAccountingError(
                    f"Usage ledger line {line_number} must be a JSON object"
                )
            continue

        event = entry.get("event", "usage")
        reservation_id = entry.get("reservation_id")
        if event == "reservation":
            raw_reserved = entry.get("estimated_cost_usd")
            if not isinstance(reservation_id, str) or not reservation_id:
                if strict:
                    raise LLMAccountingError(
                        f"Usage ledger line {line_number} has no reservation_id"
                    )
                continue
            if raw_reserved is None:
                if strict:
                    raise LLMAccountingError(
                        f"Usage ledger line {line_number} has no estimated_cost_usd"
                    )
                continue
            if strict and reservation_id in reservations:
                raise LLMAccountingError(
                    f"Usage ledger line {line_number} duplicates an active reservation"
                )
            reservations[reservation_id] = _parse_nonnegative_float(
                raw_reserved,
                f"usage ledger line {line_number} estimated_cost_usd",
            )
            continue

        if event == "release":
            if isinstance(reservation_id, str):
                reservations.pop(reservation_id, None)
            elif strict:
                raise LLMAccountingError(
                    f"Usage ledger line {line_number} has no reservation_id"
                )
            continue

        if event != "usage":
            continue

        raw_cost = entry.get("cost_usd")
        if raw_cost is None:
            if strict:
                raise LLMAccountingError(
                    f"Usage ledger line {line_number} has no cost_usd; "
                    "cannot enforce the hard cap"
                )
            if isinstance(reservation_id, str):
                reservations.pop(reservation_id, None)
            continue
        spent += _parse_nonnegative_float(
            raw_cost, f"usage ledger line {line_number} cost_usd"
        )
        if isinstance(reservation_id, str):
            reservations.pop(reservation_id, None)
    return spent, reservations


def _read_ledger_spend(handle: Any, strict: bool) -> float:
    """Backward-compatible helper returning only settled spend."""
    return _read_ledger_state(handle, strict)[0]


def _append_ledger_event(handle: Any, entry: Mapping[str, Any]) -> None:
    handle.seek(0, os.SEEK_END)
    handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _with_locked_ledger(path: Path, operation):
    """Run ``operation(handle)`` while holding an advisory file lock."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise LLMAccountingError(f"Cannot open usage ledger {path}: {exc}") from exc
    lock_module = None
    locked = False
    try:
        try:
            import fcntl as lock_module

            lock_module.flock(handle.fileno(), lock_module.LOCK_EX)
            locked = True
        except ImportError:
            pass
        try:
            return operation(handle)
        except LLMAccountingError:
            raise
        except OSError as exc:
            raise LLMAccountingError(
                f"Unable to update usage ledger {path}: {exc}"
            ) from exc
    except OSError as exc:
        raise LLMAccountingError(f"Unable to lock usage ledger {path}: {exc}") from exc
    finally:
        try:
            if locked and lock_module is not None:
                lock_module.flock(handle.fileno(), lock_module.LOCK_UN)
        finally:
            handle.close()


def _reserve_budget(
    requested_model: str,
    resolved_model: str,
    payload: Mapping[str, Any],
    max_output_tokens: Optional[int],
    api: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    cap = _hard_cap_usd()
    if cap is None:
        return None
    path = _ledger_path()
    if path is None:
        raise LLMAccountingError(
            "LLM_COST_HARD_CAP_USD requires LLM_USAGE_LEDGER_PATH"
        )
    prices = _model_prices(requested_model, resolved_model)
    if prices is None:
        raise LLMAccountingError(
            f"No LLM_PRICE_MAP entry for {requested_model!r} or "
            f"resolved model {resolved_model!r}"
        )
    estimated_cost = _estimated_call_cost_usd(payload, max_output_tokens, prices)
    reservation_id = uuid.uuid4().hex

    def reserve(handle: Any) -> None:
        spent, active_reservations = _read_ledger_state(handle, strict=True)
        active = sum(active_reservations.values())
        projected = spent + active + estimated_cost
        if projected > cap + 1e-12:
            raise LLMBudgetExceeded(
                f"LLM cost cap would be exceeded before the call: "
                f"spent=${spent:.6f}, reserved=${active:.6f}, "
                f"call_estimate=${estimated_cost:.6f}, cap=${cap:.6f}"
            )
        _append_ledger_event(
            handle,
            {
                "event": "reservation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reservation_id": reservation_id,
                "estimated_cost_usd": round(estimated_cost, 12),
                "requested_model": requested_model,
                "resolved_model": resolved_model,
                "api": api,
            },
        )

    with _ACCOUNTING_LOCK:
        _with_locked_ledger(path, reserve)
    return {
        "id": reservation_id,
        "path": path,
        "cap_usd": cap,
        "prices": prices,
        "estimated_cost_usd": estimated_cost,
    }


def _release_reservation(
    reservation: Optional[Dict[str, Any]], reason: str = "api_error"
) -> None:
    """Persistently release a reservation after an unbilled API failure."""
    if reservation is None:
        return

    def release(handle: Any) -> None:
        _, active_reservations = _read_ledger_state(handle, strict=True)
        if reservation["id"] not in active_reservations:
            return
        _append_ledger_event(
            handle,
            {
                "event": "release",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reservation_id": reservation["id"],
                "reason": reason,
            },
        )

    with _ACCOUNTING_LOCK:
        _with_locked_ledger(reservation["path"], release)


def _record_response_usage(
    response: Any,
    requested_model: str,
    resolved_model: str,
    api: str,
    reservation: Optional[Dict[str, Any]] = None,
) -> None:
    path = reservation["path"] if reservation is not None else _ledger_path()
    cap = reservation["cap_usd"] if reservation is not None else _hard_cap_usd()
    usage = _extract_usage(response)
    if usage is None:
        if cap is not None or _usage_accounting_required():
            raise LLMAccountingError(
                "API response has no usable token accounting; required usage "
                "recording failed closed"
            )
        return
    if path is None:
        return

    prices = (
        reservation["prices"]
        if reservation is not None
        else _model_prices(requested_model, resolved_model)
    )
    if prices is None:
        raise LLMAccountingError(
            f"No LLM_PRICE_MAP entry for {requested_model!r} or "
            f"resolved model {resolved_model!r}"
        )
    cost = _usage_cost_usd(usage, prices) if prices is not None else None
    backend = _select_backend()

    def append_entry(handle: Any) -> tuple[float, float]:
        spent, active_reservations = _read_ledger_state(
            handle, strict=cap is not None
        )
        reservation_id = reservation["id"] if reservation is not None else None
        if reservation_id is not None and reservation_id not in active_reservations:
            raise LLMAccountingError(
                f"Usage reservation {reservation_id} is missing from {path}; "
                "hard cap enforcement failed closed"
            )
        if reservation_id is not None:
            active_reservations.pop(reservation_id)
        cumulative = spent + (cost or 0.0)
        entry = {
            "event": "usage",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "api": api,
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            **usage,
            "cost_usd": cost,
            "cumulative_cost_usd": round(cumulative, 12) if cost is not None else None,
        }
        if reservation_id is not None:
            entry["reservation_id"] = reservation_id
        _append_ledger_event(handle, entry)
        committed = cumulative + sum(active_reservations.values())
        return cumulative, committed

    cumulative, committed = _with_locked_ledger(path, append_entry)
    if cap is not None and committed > cap + 1e-12:
        raise LLMBudgetExceeded(
            f"LLM cost cap exceeded after the call: "
            f"spent=${cumulative:.6f}, committed=${committed:.6f}, "
            f"cap=${cap:.6f}. "
            f"Usage was recorded in {path}."
        )


def _accounted_api_call(
    create,
    payload: Dict[str, Any],
    *,
    requested_model: str,
    resolved_model: str,
    api: str,
    max_output_tokens: Optional[int] = None,
):
    """Call an SDK endpoint with budget reservation and usage accounting."""
    # Plan-specific output limits are independent of accounting policy. A
    # hard cap still supplies its conservative fallback for unbounded calls.
    payload = dict(payload)
    if _usage_accounting_required() and _ledger_path() is None:
        raise LLMAccountingError(
            "LLM_REQUIRE_USAGE_ACCOUNTING requires LLM_USAGE_LEDGER_PATH"
        )
    locked_effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
    if locked_effort:
        if locked_effort not in {"low", "medium", "high"}:
            raise LLMAccountingError(
                "LLM_REASONING_EFFORT must be low, medium, or high"
            )
        payload.pop("temperature", None)
        if api == "responses":
            payload["reasoning"] = {"effort": locked_effort}
        else:
            payload["reasoning_effort"] = locked_effort
    if api == "chat.completions":
        payload = _normalize_chat_payload(payload, resolved_model)
    output_limits_disabled = _output_limits_disabled()
    if output_limits_disabled:
        for field in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            payload.pop(field, None)
        max_output_tokens = None
    if _ledger_path() is not None and _model_prices(
        requested_model, resolved_model
    ) is None:
        raise LLMAccountingError(
            f"No LLM_PRICE_MAP entry for {requested_model!r} or "
            f"resolved model {resolved_model!r}"
        )
    if max_output_tokens is None:
        for field in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            if field in payload:
                try:
                    max_output_tokens = int(payload[field])
                except (TypeError, ValueError) as exc:
                    raise LLMAccountingError(
                        f"{field} must be an integer"
                    ) from exc
                break
    if max_output_tokens is None and not output_limits_disabled:
        default_output_tokens = _configured_default_output_tokens()
        if default_output_tokens is not None or _hard_cap_usd() is not None:
            max_output_tokens = (
                default_output_tokens
                if default_output_tokens is not None
                else _default_output_reservation()
            )
            if api == "responses":
                payload["max_output_tokens"] = max_output_tokens
            elif _needs_max_completion_tokens(
                requested_model
            ) or _needs_max_completion_tokens(resolved_model):
                payload["max_completion_tokens"] = max_output_tokens
            else:
                payload["max_tokens"] = max_output_tokens

    reservation = _reserve_budget(
        requested_model, resolved_model, payload, max_output_tokens, api=api
    )
    try:
        response = create(**payload)
    except BaseException:
        _release_reservation(reservation, reason="api_error")
        raise

    with _ACCOUNTING_LOCK:
        _record_response_usage(
            response=response,
            requested_model=requested_model,
            resolved_model=resolved_model,
            api=api,
            reservation=reservation,
        )
    return response


def generate_json(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4o",
    step: str = "generation",
    max_retries: int = 3,
    temperature: float = 0.7,
    rate_limit_retries: int = 10,
    reasoning_effort: str = None
) -> Dict[str, Any]:
    """
    Generate JSON response from LLM.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model identifier (e.g., 'gpt-4o', 'gpt-5.1')
        step: Step name for logging purposes
        max_retries: Maximum number of retries on failure
        temperature: Sampling temperature
        rate_limit_retries: Maximum retries for rate limit (429) errors
        reasoning_effort: Thinking effort level for reasoning models ('low', 'medium', 'high')
        
    Returns:
        Parsed JSON response from the model
    """
    deployment_name = resolve_model_name(model)
    use_responses = _is_responses_api_model(model)
    client = get_client(use_responses_api=use_responses)
    
    rate_limit_count = 0
    attempt = 0

    while attempt < max_retries:
        try:
            print(f"[{step}] Calling model: {model} (deployment: {deployment_name}, attempt {attempt + 1}/{max_retries})")
            
            if use_responses:
                # Use Responses API for codex/pro models
                # Pass structured messages to preserve multi-turn context
                input_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                ]
                api_params = {
                    "model": deployment_name,
                    "input": input_messages,
                    "text": {"format": {"type": "json_object"}},
                }
                if reasoning_effort:
                    api_params["reasoning"] = {"effort": reasoning_effort}
                
                response = _accounted_api_call(
                    client.responses.create,
                    api_params,
                    requested_model=model,
                    resolved_model=deployment_name,
                    api="responses",
                )
                response_content = _responses_final_content(response)
            else:
                # Use Chat Completions API
                api_params = {
                    "model": deployment_name,
                    "messages": messages,
                    "response_format": {"type": "json_object"}
                }
                
                if reasoning_effort:
                    api_params["reasoning_effort"] = reasoning_effort
                else:
                    api_params["temperature"] = temperature
                
                response = _accounted_api_call(
                    client.chat.completions.create,
                    api_params,
                    requested_model=model,
                    resolved_model=deployment_name,
                    api="chat.completions",
                )
                response_content = _chat_final_content(response)
            result = _parse_json_response(response_content)
            
            print(f"[{step}] Success!")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error on attempt {attempt + 1}: {e}")
            print(f"Response content: {response_content[:200]}...")
            attempt += 1
            if attempt >= max_retries:
                raise
            time.sleep(1)
            
        except (LLMAccountingError, LLMIncompleteResponse):
            raise

        except Exception as e:
            error_str = str(e)
            # Handle rate limit errors separately (don't count as failure)
            if "429" in error_str or "rate" in error_str.lower() or "Token limit" in error_str:
                rate_limit_count += 1
                wait_time = _compute_retry_wait(e, attempt=rate_limit_count, max_cap=30)
                print(f"⏳ Rate limit hit ({rate_limit_count}/{rate_limit_retries}), waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                
                if rate_limit_count >= rate_limit_retries:
                    print(f"❌ Rate limit retry exhausted after {rate_limit_count} attempts")
                    raise
                
                # Don't increment attempt counter for rate limits
                continue
            
            print(f"Error on attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt >= max_retries:
                raise
            time.sleep(1)
    
    raise RuntimeError(f"Failed to generate valid JSON after {max_retries} attempts")


def _parse_json_response(content: str) -> Any:
    """Parse one JSON value, optionally wrapped in a Markdown JSON fence."""
    normalized = content.strip()
    if normalized.startswith("```"):
        if len(normalized) <= 6 or not normalized.endswith("```"):
            raise json.JSONDecodeError("unterminated JSON code fence", normalized, 0)
        fenced = normalized[3:-3]
        lowered = fenced.lower()
        if lowered.startswith("!json"):
            fenced = fenced[5:]
            if fenced and not fenced[0].isspace():
                raise json.JSONDecodeError("invalid JSON code fence", normalized, 0)
        elif lowered.startswith("json"):
            fenced = fenced[4:]
            if fenced and not fenced[0].isspace():
                raise json.JSONDecodeError("invalid JSON code fence", normalized, 0)
        elif fenced and fenced[0] not in "\r\n{[":
            raise json.JSONDecodeError("invalid JSON code fence", normalized, 0)
        normalized = fenced.strip()
    return json.loads(normalized)


def load_prompt(prompt_file: str) -> str:
    """Load prompt template from file."""
    with open(prompt_file, "r") as f:
        return f.read()


def populate_prompt(template: str, replacements: Dict[str, str]) -> str:
    """
    Populate prompt template with actual values.
    
    Args:
        template: Prompt template string
        replacements: Dict of placeholder -> value mappings
        
    Returns:
        Populated prompt string
    """
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(f"[[{placeholder}]]", value)
    return result


def generate_text(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    timeout: Optional[float] = None
) -> str:
    """
    Generate text response from LLM.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model identifier (e.g., 'gpt-4o', 'gpt-5.1')
        max_retries: Maximum number of retries on failure
        temperature: Sampling temperature (None to use model default)
        max_tokens: Maximum tokens to generate (None for model default)
        reasoning_effort: Reasoning effort level ('low', 'medium', 'high') for GPT-5 models
        timeout: Optional per-call read timeout in seconds (None keeps the
            shared client default). Bounds verification calls whose prompts
            can trigger degenerate, effectively unbounded generations.

    Returns:
        Text response from the model
    """
    deployment_name = resolve_model_name(model)
    use_responses = _is_responses_api_model(model)
    client = get_client(use_responses_api=use_responses)
    if timeout is not None:
        client = client.with_options(
            timeout=httpx.Timeout(
                connect=30.0,
                read=float(timeout),
                write=600.0,
                pool=600.0,
            )
        )
    
    for attempt in range(max_retries):
        try:
            if use_responses:
                # Use Responses API for codex/pro models
                # Pass structured messages to preserve multi-turn context
                input_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                ]
                api_params = {
                    "model": deployment_name,
                    "input": input_messages,
                }
                if reasoning_effort is not None:
                    api_params["reasoning"] = {"effort": reasoning_effort}
                if max_tokens is not None:
                    api_params["max_output_tokens"] = max_tokens
                
                response = _accounted_api_call(
                    client.responses.create,
                    api_params,
                    requested_model=model,
                    resolved_model=deployment_name,
                    api="responses",
                    max_output_tokens=max_tokens,
                )
                return _responses_final_content(response)
            else:
                # Use Chat Completions API
                kwargs = {
                    "model": deployment_name,
                    "messages": messages,
                }
                
                if temperature is not None and reasoning_effort is None:
                    kwargs["temperature"] = temperature
                
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort
                
                if max_tokens is not None:
                    if _needs_max_completion_tokens(model):
                        kwargs["max_completion_tokens"] = max_tokens
                    else:
                        kwargs["max_tokens"] = max_tokens
                    
                response = _accounted_api_call(
                    client.chat.completions.create,
                    kwargs,
                    requested_model=model,
                    resolved_model=deployment_name,
                    api="chat.completions",
                    max_output_tokens=max_tokens,
                )
                return _chat_final_content(response)
            
        except (LLMAccountingError, LLMIncompleteResponse):
            raise

        except Exception as e:
            error_str = str(e)
            print(f"Error on attempt {attempt + 1}: {e}")
            
            if attempt == max_retries - 1:
                raise
            
            wait_time = _compute_retry_wait(e, attempt=attempt)
            print(f"  ⏳ Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
    
    raise RuntimeError(f"Failed to generate text after {max_retries} attempts")


# =============================================================================
# Multi-turn helper with reasoning-trace continuity
# =============================================================================
#
# ``generate_text`` returns only the assistant's plaintext, which is sufficient
# for single-turn use. For *multi-turn* use against closed-source reasoning
# models (gpt-5.x, o3, grok *-reasoning, codex/pro), the model's hidden
# chain-of-thought is not returned and is therefore lost between turns. This
# means the model re-derives its understanding of dynamic intent from scratch
# each turn instead of continuing prior reasoning.
#
# ``generate_multi_turn`` solves this for closed reasoning models by going
# through the Responses API with ``include=["reasoning.encrypted_content"]``
# and returning the full list of output items (reasoning + message). Callers
# extend their conversation list with these items so the next call carries the
# encrypted reasoning blob forward. ``store=False`` is used so no state is
# retained server-side; continuity is purely client-side.
#
# White-box / non-reasoning models keep their CoT in plaintext assistant
# content, so the function falls back to chat completions and returns a single
# ``{"role":"assistant","content":text}`` item — the same shape callers used
# manually with ``generate_text``.


def generate_multi_turn(
    messages: List[Dict[str, Any]],
    model: str,
    max_retries: int = 3,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """Run one multi-turn generation step, preserving reasoning across turns.

    Args:
        messages: Conversation so far. May contain plain chat-style dicts
            ``{"role": ..., "content": ...}`` and Responses-API output items
            (``{"type": "reasoning", ...}``, ``{"type": "message", ...}``).
            Both shapes are accepted by the Responses API.
        model: Model identifier (e.g. ``gpt-5.1``).
        temperature: Ignored for closed reasoning models (they only accept
            ``reasoning_effort``).
        max_tokens: Maximum output tokens.
        reasoning_effort: ``low`` / ``medium`` / ``high`` for reasoning models.

    Returns:
        Tuple of (assistant_text, output_items). The caller must append
        ``output_items`` to ``messages`` before the next call to preserve
        reasoning continuity.
    """
    # ── Non-reasoning models (gpt-4o, *-chat, ...): plaintext CoT lives in
    # assistant.content, so chat.completions is fine. ───────────────────
    if not _is_closed_reasoning_model(model):
        text = generate_text(
            messages=messages,
            model=model,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return text, [{"role": "assistant", "content": text}]

    # ── Closed reasoning path: Responses API with encrypted reasoning ───
    # Some closed-reasoning models (Grok) are not supported on the Responses
    # endpoint. Route them through chat completions; reasoning trace will be
    # lost between turns but the conversation remains functional.
    if not _supports_responses_api(model):
        text = generate_text(
            messages=messages,
            model=model,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return text, [{"role": "assistant", "content": text}]

    deployment_name = resolve_model_name(model)
    client = get_client(use_responses_api=True)

    api_params: dict[str, Any] = {
        "model": deployment_name,
        "input": list(messages),
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if reasoning_effort is not None:
        api_params["reasoning"] = {"effort": reasoning_effort}
    if max_tokens is not None:
        api_params["max_output_tokens"] = max_tokens

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = _accounted_api_call(
                client.responses.create,
                api_params,
                requested_model=model,
                resolved_model=deployment_name,
                api="responses",
                max_output_tokens=max_tokens,
            )
            text = _responses_final_content(response)
            output_items = [
                item.model_dump(exclude_none=True) for item in response.output
            ]
            return text, output_items
        except (LLMAccountingError, LLMIncompleteResponse):
            raise

        except Exception as e:
            last_error = e
            error_str = str(e)
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                break
            wait_time = _compute_retry_wait(e, attempt=attempt)
            print(f"  ⏳ Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)

    assert last_error is not None
    raise last_error


# =============================================================================
# Native function-calling helper (chat-completions only)
# =============================================================================
#
# Sibling of ``generate_text`` that exposes the OpenAI ``tools=`` /
# ``tool_choice=`` parameters and returns the *full* ChatCompletion
# response object (not just the assistant text). Required by the SWE
# mini-swe-agent integration, which uses native bash-tool calling on
# GPT-5.x models.
#
# Scope:
# - chat-completions only. Responses-API
#   deployments (codex / pro) raise loudly. Adding tool calling for
#   those is out-of-scope for v1.
# - Same retry / quota-rotation / temperature / reasoning_effort /
#   max_completion_tokens semantics as ``generate_text``. We deliberately
#   mirror that function rather than refactor it, because every other
#   domain still depends on the existing helper unchanged.
# =============================================================================


def generate_with_tools(
    messages: List[Dict[str, Any]],
    model: str = "gpt-5.1",
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool = True,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 3,
):
    """Native function-calling counterpart of ``generate_text``.

    Returns the full ``openai.types.chat.ChatCompletion`` object so the
    caller can inspect ``response.choices[0].message.tool_calls``,
    ``.model_dump()``, etc. (mini-swe-agent's ``LitellmModel.query`` does
    exactly that). For non-tool-calling use cases keep using
    ``generate_text``.

    Parameter parity with ``generate_text`` is preserved deliberately:
      - temperature/reasoning_effort mutual exclusion
      - ``max_completion_tokens`` for reasoning models, ``max_tokens`` else
      - same retry / rate-limit-parsing loop

    Out-of-scope (raises explicitly):
      - Responses-API-only deployments (codex, pro)

    Args:
        messages: standard chat-completions message list. Pass through
            assistant ``tool_calls`` and tool-message ``tool_call_id`` as-is;
            do NOT flatten to ``{role, content}``, or Azure will reject
            the next round-trip with 400 ("messages with role 'tool' must
            be a response to a preceding message with 'tool_calls'").
        tools: function-tool list (e.g. ``[BASH_TOOL]`` from mini-swe-agent).
        tool_choice: ``"auto" | "required" | "none" | {"type":"function",...}``.
            Defaults to ``"required"`` to match mini's swebench.yaml prompt
            ("every response must include exactly one bash tool call").
        parallel_tool_calls: whether the model may emit multiple tool calls
            per turn. Mini's native ``swebench.yaml`` sets this True.
    """
    if _is_responses_api_model(model):
        raise NotImplementedError(
            f"generate_with_tools does not support Responses-API-only "
            f"deployments ({model!r}). codex / pro models require a different "
            "tool-call surface that has not been implemented in this helper."
        )

    deployment_name = resolve_model_name(model)
    client = get_client(use_responses_api=False)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": deployment_name,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                # tool_choice is only meaningful when tools is supplied.
                # Forward only when explicitly set; otherwise let the API
                # default ("auto") apply, matching mini-swe-agent v2 stock.
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
                # parallel_tool_calls likewise.
                kwargs["parallel_tool_calls"] = parallel_tool_calls

            # temperature / reasoning_effort: mirror generate_text exactly.
            if temperature is not None and reasoning_effort is None:
                kwargs["temperature"] = temperature
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort

            if max_tokens is not None:
                if _needs_max_completion_tokens(model):
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

            response = _accounted_api_call(
                client.chat.completions.create,
                kwargs,
                requested_model=model,
                resolved_model=deployment_name,
                api="chat.completions",
                max_output_tokens=max_tokens,
            )
            if not response.choices:
                raise RuntimeError("API returned empty choices")
            return response

        except LLMAccountingError:
            raise

        except Exception as e:
            last_error = e
            error_str = str(e)
            print(f"Error on attempt {attempt + 1}: {e}")

            if attempt == max_retries - 1:
                raise

            wait_time = _compute_retry_wait(e, attempt=attempt)
            print(f"  ⏳ Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)

    assert last_error is not None
    raise last_error


# =============================================================================
# Native function-calling helper (Responses API)
# =============================================================================
#
# Sibling of ``generate_with_tools`` for deployments where the provider requires
# the Responses API rather than chat completions for tool calling. Notably
# ``gpt-5.5`` rejects ``tools= + reasoning_effort=`` on chat completions
# with a 400 ("Function tools with reasoning_effort are not supported for
# gpt-5.5 in /v1/chat/completions. Please use /v1/responses
# instead.").
#
# Format expectations:
# - ``messages`` is the *Responses* input shape: a list of items, each
#   either a role-message dict (``{"role":"system|user|assistant",
#   "content":...}``) or a Responses-format item dict (``{"type":
#   "function_call", "call_id":..., "name":..., "arguments":...}``,
#   ``{"type":"function_call_output", "call_id":..., "output":...}``,
#   etc.).
# - ``tools`` is the *Responses* tool-def shape: flat
#   ``{"type":"function", "name":..., "description":..., "parameters":...}``
#   (NO nested ``function:`` key).
# - Returns the raw ``Response`` object — the caller (mini-swe-agent's
#   ``LitellmResponseModel`` subclass) inspects ``response.output``,
#   ``response.model_dump()``, etc.
# =============================================================================


def generate_with_tools_responses(
    messages: List[Dict[str, Any]],
    model: str = "gpt-5.5",
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    parallel_tool_calls: Optional[bool] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 3,
):
    """Responses-API counterpart of ``generate_with_tools``.

    Use this for closed-reasoning deployments that the provider requires on
    ``/v1/responses`` for tool calling (gpt-5.x). Returns the raw
    ``Response`` object; do NOT wrap it in a synthetic ChatCompletion.

    Out-of-scope (raises explicitly):
      - OSS models
      - models that ``_supports_responses_api`` rejects (Grok reasoning
        non-closed-reasoning chat models which should stay on
        ``generate_with_tools``)

    Reasoning continuity: encrypted reasoning items ARE preserved
    between calls. We request ``include=["reasoning.encrypted_content"]``
    and ``store=False`` so the API echoes back encrypted reasoning blobs
    that mini's ``_prepare_messages_for_api`` flattens onto the next
    turn's input. ``function_call`` / ``function_call_output`` items
    likewise round-trip. (Mirrors the math/IF/search-domain
    ``generate_multi_turn`` pattern.)
    """
    if not _supports_responses_api(model):
        raise NotImplementedError(
            f"generate_with_tools_responses requires a Responses-API-capable "
            f"deployment; {model!r} is not. Use generate_with_tools instead."
        )

    deployment_name = resolve_model_name(model)
    client = get_client(use_responses_api=True)

    # Errors on which we should fail fast (schema bugs, not transient).
    # Detected by string match because the underlying client raises
    # ``openai.BadRequestError`` whose message embeds the JSON body.
    fail_fast_markers = (
        "invalid_request_error",
        "BadRequestError",
        "400 ",
        "unsupported_parameter",
        "Function tools with reasoning_effort are not supported",
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            api_params: Dict[str, Any] = {
                "model": deployment_name,
                "input": messages,
                "store": False,
                "include": ["reasoning.encrypted_content"],
            }
            if tools:
                api_params["tools"] = tools
                if tool_choice is not None:
                    api_params["tool_choice"] = tool_choice
                if parallel_tool_calls is not None:
                    api_params["parallel_tool_calls"] = parallel_tool_calls
            if max_tokens is not None:
                api_params["max_output_tokens"] = max_tokens
            if reasoning_effort is not None:
                api_params["reasoning"] = {"effort": reasoning_effort}

            response = _accounted_api_call(
                client.responses.create,
                api_params,
                requested_model=model,
                resolved_model=deployment_name,
                api="responses",
                max_output_tokens=max_tokens,
            )
            return response

        except LLMAccountingError:
            raise

        except Exception as e:
            last_error = e
            error_str = str(e)
            print(f"Error on attempt {attempt + 1}: {e}")

            # Fail-fast on schema errors. Retrying a 400 is wasteful and
            # masks real bugs in input translation.
            if any(m in error_str for m in fail_fast_markers):
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = _compute_retry_wait(e, attempt=attempt)
            print(f"  ⏳ Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)

    assert last_error is not None
    raise last_error
