"""
LLM utility functions for Function & Argument extraction.

Calls go through the OpenAI or Azure OpenAI APIs, selected from environment
variables:

- **OpenAI**:        ``OPENAI_API_KEY``
- **Azure OpenAI**:  ``AZURE_OPENAI_API_KEY`` + ``AZURE_OPENAI_ENDPOINT``

When both are configured, set ``LLM_BACKEND=openai|azure`` to disambiguate.
See :func:`get_client` and :func:`resolve_model_name`.
"""
import os
import json
import random
import time
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI, OpenAI


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
# Backend configuration (OpenAI / Azure OpenAI, key-based)
# =============================================================================
#
# Two backends are supported, selected from environment variables:
#   - OpenAI:        OPENAI_API_KEY
#   - Azure OpenAI:  AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
#
# When both are configured, set LLM_BACKEND=openai|azure to disambiguate;
# otherwise Azure takes precedence. Azure API versions and deployment-name
# mapping are configurable (see below).

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


# =============================================================================
# Client construction
# =============================================================================


def _select_backend() -> str:
    """Return the active backend: ``"openai"`` or ``"azure"``."""
    backend = os.environ.get("LLM_BACKEND", "").strip().lower()
    if backend in ("openai", "azure"):
        return backend
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No LLM credentials found. Set OPENAI_API_KEY, or "
        "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT. When both are set, use "
        "LLM_BACKEND=openai|azure to disambiguate."
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


def resolve_model_name(model: str) -> str:
    """Resolve the model/deployment id to send to the API.

    On Azure, applies the optional ``AZURE_OPENAI_DEPLOYMENT_MAP`` so users
    whose Azure deployment names differ from the canonical model id can map
    them explicitly.
    """
    name = clean_model_name(model)
    if _select_backend() == "azure":
        return _azure_deployment_map().get(name, name)
    return name


def get_client(use_responses_api: bool = False):
    """Create an OpenAI or Azure OpenAI client from environment credentials.

    Args:
        use_responses_api: Select the Responses API version (Azure only).

    Returns:
        An ``openai.OpenAI`` or ``openai.AzureOpenAI`` client.
    """
    if _select_backend() == "azure":
        api_version = (
            AZURE_RESPONSES_API_VERSION if use_responses_api else AZURE_API_VERSION
        )
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=api_version,
        )
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
    
    for attempt in range(max_retries):
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
                
                response = client.responses.create(**api_params)
                response_content = response.output_text
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
                
                response = client.chat.completions.create(**api_params)
                response_content = response.choices[0].message.content
                if response_content is None:
                    response_content = getattr(response.choices[0].message, "reasoning_content", None)
            result = json.loads(response_content)
            
            print(f"[{step}] Success!")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error on attempt {attempt + 1}: {e}")
            print(f"Response content: {response_content[:200]}...")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
            
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
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    
    raise RuntimeError(f"Failed to generate valid JSON after {max_retries} attempts")


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
    reasoning_effort: Optional[str] = None
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
        
    Returns:
        Text response from the model
    """
    deployment_name = resolve_model_name(model)
    use_responses = _is_responses_api_model(model)
    client = get_client(use_responses_api=use_responses)
    
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
                
                response = client.responses.create(**api_params)
                return response.output_text
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
                    
                response = client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise RuntimeError("API returned empty choices")
                content = response.choices[0].message.content
                # Reasoning models may put output in reasoning_content when
                # content is None (e.g., token budget exhausted during thinking)
                if content is None:
                    content = getattr(response.choices[0].message, "reasoning_content", None)
                return content
            
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
            response = client.responses.create(**api_params)
            text = response.output_text
            output_items = [
                item.model_dump(exclude_none=True) for item in response.output
            ]
            if not text and not any(
                it.get("type") == "message" for it in output_items
            ):
                # Reasoning emitted but no message item (e.g. token budget
                # exhausted during thinking). Surface as retryable error.
                raise RuntimeError(
                    "Responses API returned no message item "
                    "(reasoning may have exhausted token budget)"
                )
            return text, output_items
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

            response = client.chat.completions.create(**kwargs)
            if not response.choices:
                raise RuntimeError("API returned empty choices")
            return response

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

            response = client.responses.create(**api_params)
            return response

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
