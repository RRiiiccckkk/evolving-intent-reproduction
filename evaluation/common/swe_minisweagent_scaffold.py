"""
Mini-SWE-agent v2 scaffold for EvolvingIntent.

Entry point: ``run_evolvingintent_with_mini_agent(user_turns, instance, model)``.

How multi-turn user injection works
-----------------------------------
mini-swe-agent's ``DefaultAgent`` runs ``while True: step()`` until the model
emits the magic string ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` (which becomes
a ``Submitted`` exception) or another ``InterruptAgentFlow`` subclass fires.

EvolvingIntent hands us a *script* of user turns: ``[turn_0, turn_1, ..., turn_N]``.
turn_0 is the initial task; turn_1..N are intent updates / corrections that the
user makes while the model works.

Strategy: drive ``agent.step()`` ourselves. Whenever the agent tries to submit
(raises ``Submitted``) but we still have undelivered user turns, we *intercept*
the submission, append the next user turn as a follow-up message, and resume.
Only when the agent submits AFTER all user turns are delivered do we accept
the patch as final.

This is the mini-agent-recommended pattern: the ``InteractiveAgent`` upstream
uses the same ``InterruptAgentFlow`` machinery to let a human interrupt a
running agent (cf. ``minisweagent/agents/interactive.py``). Here, the
"interrupting human" is our scripted sequence.

Patch extraction: after a final ``Submitted`` is accepted, the unified diff
lives in ``e.messages[-1]["extra"]["submission"]``. We hand that to the
existing :func:`evaluation.common.swe_harness.SWEHarness.verify_patch` for
official-harness scoring.

Constraints respected
---------------------
- We use a custom **text-based model** (``LLMTextbasedModel``) that
  routes LLM calls through our existing ``llm_utils.generate_text`` (Azure /
  shared client's auth chain). mini-swe-agent's stock ``LitellmModel`` cannot reach
  the shared client without provider configuration, so a thin adapter is required.
- The model uses **text-based action parsing** (regex on triple-backtick
  ```bash``` blocks), so we don't need provider-side function calling. This
  matches mini's ``swebench_backticks.yaml`` config style.
- Each instance gets its own mini-swe-agent docker container (image
  ``swebench/sweb.eval.x86_64.<id>:latest``). After ``Submitted``, the patch
  is sent to OUR ``swe_harness`` which spins up a *separate* swebench docker
  for grading. The image is shared between both runtimes.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

import minisweagent
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import get_environment
from minisweagent.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
    UserInterruption,
)
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_text import (
    format_observation_messages,
    parse_regex_actions,
)
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

# Bring our existing shared LLM client into scope so the model adapter can
# delegate to it (preserves Azure AD auth, retry policy, model-name mapping).
from intent_construction.intent_extraction.core.llm_utils import (  # noqa: E402
    generate_text as _llm_generate_text,
    generate_with_tools as _llm_generate_with_tools,
    generate_with_tools_responses as _llm_generate_with_tools_responses,
)
from evaluation.swe_bench.state import (  # noqa: E402
    TOOL_CALL_LIMIT_PER_TURN,
    HardeningError,
    ToolCallCounter,
    assert_no_output_limit_fields,
    assert_kimi_tool_policy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Model: text-based action parsing
# =============================================================================


_DEFAULT_ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
_DEFAULT_FORMAT_ERROR = (
    "Please always provide EXACTLY ONE action in a `mswea_bash_command` "
    "fenced block, found {{actions|length}} actions."
)


class LLMTextbasedModelConfig(LitellmModelConfig):
    """Config for the text-based model. Inherits all litellm fields then overrides."""

    action_regex: str = _DEFAULT_ACTION_REGEX
    format_error_template: str = _DEFAULT_FORMAT_ERROR


class LLMTextbasedModel(LitellmModel):
    """Drop-in mini-swe-agent Model that delegates to llm_utils.generate_text.

    Overrides:
      - ``__init__``: install our config class.
      - ``_query``: bypass litellm entirely; call ``generate_text`` and wrap the
        result in a litellm-compatible response shape so the rest of the
        ``LitellmModel.query`` pipeline (which reads ``response.choices[0].message``,
        ``response.model_dump``, etc.) keeps working.
      - ``_calculate_cost``: the API doesn't expose per-call cost; return 0.
      - ``_parse_actions``: regex on the assistant content (text-based mode).
      - ``format_observation_messages``: emit observations as plain user
        messages (not tool result messages, since we're not using tool calling).
    """

    def __init__(self, **kwargs):
        super().__init__(config_class=LLMTextbasedModelConfig, **kwargs)

    # ---- core query path ---------------------------------------------------

    def _query(self, messages: list[dict[str, str]], **kwargs):
        # Strip any litellm-specific kwargs / multimodal content we cannot honor.
        plain_messages = [
            {"role": m.get("role", "user"), "content": _flatten_content(m.get("content"))}
            for m in messages
        ]
        # Map mini's model_kwargs → generate_text's parameters.
        gt_kwargs: dict[str, Any] = {"model": self.config.model_name}
        merged = {**(self.config.model_kwargs or {}), **kwargs}
        if "temperature" in merged:
            gt_kwargs["temperature"] = merged["temperature"]
        if "max_tokens" in merged:
            gt_kwargs["max_tokens"] = merged["max_tokens"]
        if "reasoning_effort" in merged:
            gt_kwargs["reasoning_effort"] = merged["reasoning_effort"]

        text = _llm_generate_text(messages=plain_messages, **gt_kwargs)
        text = text or ""

        # Fake a litellm-shaped response object so the rest of the LitellmModel
        # pipeline (query() in the parent) keeps working unchanged.
        return _make_fake_litellm_response(content=text, model_name=self.config.model_name)

    def _calculate_cost(self, response) -> dict[str, float]:  # type: ignore[override]
        # the API doesn't surface per-call cost; report 0 to keep agent.cost
        # numerically usable without crashing on missing cost metadata.
        return {"cost": 0.0}

    # ---- output parsing ----------------------------------------------------

    def _parse_actions(self, response) -> list[dict]:
        content = response.choices[0].message.content or ""
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        # Text-based mode: observations come back as user-role messages, not
        # tool-result messages.
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )


def _flatten_content(content: Any) -> str:
    """Reduce mini's possibly-multimodal content list back to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                # litellm style: {"type": "text", "text": "..."}
                if "text" in chunk:
                    parts.append(str(chunk["text"]))
                elif "content" in chunk:
                    parts.append(str(chunk["content"]))
            elif isinstance(chunk, str):
                parts.append(chunk)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _make_fake_litellm_response(*, content: str, model_name: str):
    """Build a minimal object that quacks like ``litellm.completion(...)`` return."""
    msg = SimpleNamespace(
        content=content,
        role="assistant",
        # Used by LitellmModel.query: message.model_dump() → dict
        model_dump=lambda: {"role": "assistant", "content": content},
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop", index=0)
    response = SimpleNamespace(
        choices=[choice],
        model=model_name,
        # Used by LitellmModel.query: response.model_dump()
        model_dump=lambda: {
            "model": model_name,
            "choices": [{"message": {"role": "assistant", "content": content}}],
        },
    )
    return response


# =============================================================================
# Custom Model: native function calling
# =============================================================================


class LLMToolModel(LitellmModel):
    """Native function-calling sibling of ``LLMTextbasedModel``.

    Routes the LLM call through ``llm_utils.generate_with_tools`` (shared
    Azure auth + chat-completions + native ``tools=`` parameter) and
    inherits the rest of the pipeline from mini-swe-agent's
    ``LitellmModel``: ``_parse_actions`` (reads
    ``response.choices[0].message.tool_calls``),
    ``format_observation_messages`` (emits ``role:"tool"`` messages
    with ``tool_call_id``), ``_prepare_messages_for_api`` (strips
    mini's ``extra`` keys; preserves OpenAI-native fields), etc.

    Critical: ``_query`` does NOT flatten the message list. The audited
    text-based variant flattens to ``{role, content}`` because that
    shape is sufficient for completions-only mode; doing the same here
    drops ``tool_calls`` / ``tool_call_id`` and triggers Azure 400 on
    turn ≥ 2 (audit A item #5; audit B "LARGEST under-acknowledged risk").
    """

    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, Any]], **kwargs):
        merged = {**(self.config.model_kwargs or {}), **kwargs}
        tool_choice = merged.get("tool_choice")
        if "kimi" in self.config.model_name.lower() and tool_choice is not None:
            raise HardeningError(
                "Kimi thinking mode forbids sending an explicit tool_choice"
            )
        # Pass parent-prepared messages straight through. The parent's
        # ``_prepare_messages_for_api`` (litellm_model.py:75-78) has
        # already stripped mini-internal ``extra`` keys; what remains is
        # the OpenAI-native shape (role / content / tool_calls / tool_call_id).
        #
        # ``tool_choice`` is forwarded ONLY when explicitly configured
        # (in model_kwargs or via per-call kwargs). When unset we let the
        # API default ("auto") apply — same as mini-swe-agent v2's stock
        # ``swebench.yaml`` which does not pin tool_choice.
        return _llm_generate_with_tools(
            messages=messages,
            model=self.config.model_name,
            tools=[BASH_TOOL],
            tool_choice=tool_choice,  # None means the helper omits the field.
            parallel_tool_calls=merged.get("parallel_tool_calls", True),
            temperature=merged.get("temperature"),
            max_tokens=merged.get("max_tokens"),
            reasoning_effort=merged.get("reasoning_effort"),
        )

    def _calculate_cost(self, response) -> dict[str, float]:  # type: ignore[override]
        # the API doesn't surface per-call cost; report 0 to keep agent.cost
        # numerically usable without crashing on missing cost metadata.
        # Same as LLMTextbasedModel.
        return {"cost": 0.0}


# =============================================================================
# Custom Model: Responses API native function calling
# =============================================================================
# Required for closed-reasoning deployments where chat completions rejects
# ``tools= + reasoning_effort=`` on /v1/chat/completions (e.g. gpt-5.5).
# We subclass mini's ``LitellmResponseModel`` to inherit its native
# Responses-API plumbing (input flattening of prior ``response`` objects,
# ``parse_toolcall_actions_response``, ``BASH_TOOL_RESPONSE_API``,
# ``function_call_output`` observation formatter) and only override
# ``_query`` to route through our shared LLM client.

from minisweagent.models.litellm_response_model import (  # noqa: E402
    LitellmResponseModel,
    LitellmResponseModelConfig,
)


class LLMResponseToolModel(LitellmResponseModel):
    """Responses-API sibling of ``LLMToolModel`` for gpt-5.x.

    Inherits message-shape handling and tool parsing from
    ``LitellmResponseModel``; only ``_query`` differs (shared client
    instead of ``litellm.responses``).

    Reasoning continuity: encrypted reasoning IS preserved between
    iterations. ``generate_with_tools_responses`` requests
    ``include=["reasoning.encrypted_content"]`` with ``store=False`` so
    the API echoes back encrypted reasoning blobs on every step.
    Mini's ``_prepare_messages_for_api`` flattens response output items
    on the next turn — including reasoning items with their
    ``encrypted_content`` — and ``function_call`` /
    ``function_call_output`` items continue to round-trip as before.
    """

    # Output-only fields the API returns on response items but rejects on
    # subsequent input. Discovered empirically (gpt-5.5 returned 400
    # ``Unknown parameter: 'input[2].status'`` when mini's stock
    # ``_prepare_messages_for_api`` kept the field). We strip these before
    # echoing prior output items back in. Add to this set if the API surfaces
    # other output-only fields in future deployments.
    _OUTPUT_ONLY_FIELDS = frozenset({
        "status",
        # Per-item id (``fc_…``, ``msg_…``, ``rs_…``). the API accepts ``id``
        # back on Responses input today, but it conflicts with our
        # synthesised filler items (``function_call_output``) that don't
        # have one — keep it out for consistency.
        "id",
    })

    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmResponseModelConfig, **kwargs)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        # Inherit mini's flattening (response objects → output items),
        # then strip output-only fields that the API rejects on input.
        prepared = super()._prepare_messages_for_api(messages)
        cleaned = []
        for msg in prepared:
            cleaned.append({
                k: v for k, v in msg.items()
                if k not in self._OUTPUT_ONLY_FIELDS
            })
        return cleaned

    def _query(self, messages: list[dict[str, Any]], **kwargs):
        merged = {**(self.config.model_kwargs or {}), **kwargs}
        # Use mini's Responses-format BASH tool def (flat, no nested
        # "function" key).
        from minisweagent.models.utils.actions_toolcall_response import (
            BASH_TOOL_RESPONSE_API,
        )
        return _llm_generate_with_tools_responses(
            messages=messages,
            model=self.config.model_name,
            tools=[BASH_TOOL_RESPONSE_API],
            tool_choice=merged.get("tool_choice"),
            parallel_tool_calls=merged.get("parallel_tool_calls"),
            max_tokens=merged.get("max_tokens"),
            reasoning_effort=merged.get("reasoning_effort"),
        )

    def _calculate_cost(self, response) -> dict[str, float]:  # type: ignore[override]
        return {"cost": 0.0}


_MINI_PKG_DIR = Path(minisweagent.__file__).resolve().parent


def _load_default_swebench_config() -> dict:
    """Load mini-swe-agent's text-based SWE-bench config (regex-on-backticks).

    We use ``swebench_backticks.yaml`` so the model is prompted to emit
    ```` ```mswea_bash_command ```` blocks rather than tool-call JSON. This
    matches our regex parser and works without provider-side function calling.
    """
    cfg_dir = _MINI_PKG_DIR / "config" / "benchmarks"
    for name in ("swebench_backticks.yaml", "swebench.yaml"):
        p = cfg_dir / name
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No swebench config found in {cfg_dir}")


def _load_default_swebench_config_native() -> dict:
    """Load mini-swe-agent's native tool-calling SWE-bench config.

    Pinned to ``swebench.yaml`` (NOT the backticks variant). This is the
    config whose ``system_template`` instructs the model to call the
    ``bash`` tool on every turn, and whose ``model.model_kwargs`` set
    ``parallel_tool_calls: true``. We then override the model section
    to point at our LLM adapter.
    """
    cfg_dir = _MINI_PKG_DIR / "config" / "benchmarks"
    p = cfg_dir / "swebench.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"swebench.yaml not found at {p}. mini-swe-agent install "
            "is missing the native config; cannot run with use_tool_calling=True."
        )
    with open(p) as f:
        return yaml.safe_load(f)


# =============================================================================
# Intent-update prompt wrapper
# =============================================================================


_INTENT_UPDATE_PREFIX = (
    "[USER UPDATE — supersedes any earlier conflicting information] "
)


def _wrap_intent_update(turn_text: str, *, on_submit: bool) -> str:
    """Wrap a follow-up user turn so the model treats it as authoritative."""
    text = turn_text.strip()
    if on_submit:
        return (
            "Hold on — before you finalize, the user has new information.\n\n"
            + _INTENT_UPDATE_PREFIX + text
            + "\n\nPlease re-evaluate your work in light of this update and "
            "continue. Do not submit yet unless this update has been fully "
            "incorporated."
        )
    return _INTENT_UPDATE_PREFIX + text


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class MiniAgentResult:
    """Result of running mini-swe-agent on a EvolvingIntent conversation."""

    instance_id: str
    submission: str  # unified diff (empty if agent never submitted cleanly)
    exit_status: str  # e.g. "Submitted", "LimitsExceeded", "Error"
    n_steps: int
    n_user_turns_delivered: int
    cost: float
    error: str | None = None
    final_messages: list[dict] = field(default_factory=list)
    scaffold_mode: str = "textbased"  # "native" | "textbased"
    # Per-turn step accounting (Path B in the per-turn cap design).
    # ``step_limit_per_turn`` is None when running in legacy single-cap
    # mode (Path A); ``per_turn_steps`` is empty in that case.
    step_limit_per_turn: int | None = None
    per_turn_steps: list[int] = field(default_factory=list)
    tool_call_limit_per_turn: int | None = None
    per_turn_tool_calls: list[int] = field(default_factory=list)


# =============================================================================
# Main entry point
# =============================================================================


class ToolCallLimitExceeded(LimitsExceeded):
    """Raised before an environment call would exceed the per-turn cap."""


class EnvironmentExecutionError(RuntimeError):
    """Raised when SWE-ReX cannot execute another command reliably."""


def _environment_failure(output: Any) -> str | None:
    """Return a concise SWE-ReX transport/runtime failure, if present."""
    if not isinstance(output, dict) or output.get("returncode") != -1:
        return None
    extra = output.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    exception_type = str(extra.get("exception_type") or "EnvironmentError")
    detail = str(
        extra.get("exception")
        or output.get("exception_info")
        or output.get("output")
        or "remote execution failed"
    ).strip()
    return f"{exception_type}: {detail}"


class _ToolBudgetEnvironment:
    """Count actual calls and stop on unrecoverable SWE-ReX failures."""

    def __init__(self, environment: Any, counter: ToolCallCounter) -> None:
        self.environment = environment
        self.counter = counter

    def execute(self, action: dict, *args: Any, **kwargs: Any) -> Any:
        try:
            self.counter.consume()
        except HardeningError as exc:
            raise ToolCallLimitExceeded(
                {
                    "role": "exit",
                    "content": "ToolCallLimitExceeded",
                    "extra": {
                        "exit_status": "ToolCallLimitExceeded",
                        "submission": "",
                        "tool_call_limit_per_turn": self.counter.limit,
                    },
                }
            ) from exc
        output = self.environment.execute(action, *args, **kwargs)
        failure = _environment_failure(output)
        if failure is not None:
            raise EnvironmentExecutionError(failure)
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        values = self.environment.get_template_vars(**kwargs)
        return {
            **values,
            "tool_calls_this_turn": self.counter.current,
            "tool_call_limit_per_turn": self.counter.limit,
        }

    def serialize(self) -> dict[str, Any]:
        payload = self.environment.serialize()
        payload.setdefault("info", {}).setdefault("config", {})["tool_budget"] = {
            "limit_per_turn": self.counter.limit,
            "completed_turns": list(self.counter.completed_turns),
            "current_turn": self.counter.current,
        }
        return payload

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)


def _ensure_pending_tool_call_resolved(agent, observation_text: str) -> None:
    """Native-mode protocol fix for intercepted ``Submitted``.

    When mini-swe-agent's ``Submitted`` exception fires from inside
    ``execute_actions``, the assistant message containing the BASH
    tool call has already been appended to ``agent.messages`` (cf.
    ``DefaultAgent.step``: ``execute_actions(query())`` — query
    appended the assistant message before execute_actions raised).
    Mini's normal flow then catches ``Submitted`` and appends an
    ``role:"exit"`` message which terminates the loop.

    EvolvingIntent intercepts that exit, discarding the exit message and
    appending the next user-intent-update turn instead. **In native
    tool-calling mode** that leaves the assistant's ``tool_calls``
    unmatched: the very next API call would have ``role:"user"``
    immediately after ``role:"assistant"`` with ``tool_calls=[...]``,
    which Azure/OpenAI rejects with 400 (every assistant tool_call must
    be followed by ``role:"tool"`` results with matching ``tool_call_id``).

    This helper synthesises the missing tool-result messages BEFORE the
    user intent update is appended, so the conversation history stays
    legal. ``observation_text`` is the bash output that triggered the
    submission magic string; we hand it back as the tool result content
    (truncated for safety).

    For text-based mode this is a no-op (the last assistant message
    has no ``tool_calls`` field).

    For Responses-API mode the last message has shape
    ``{object: "response", output: [{type: "function_call", call_id, ...}, ...]}``;
    we walk ``output`` for ``function_call`` items lacking a matching
    ``function_call_output`` and synthesize fillers in
    Responses-item shape.
    """
    if not agent.messages:
        return
    last = agent.messages[-1]

    # ---- Responses-API path (LitellmResponseModel produced this msg) ----
    # Mini's LitellmResponseModel.query stores response.model_dump() as a
    # message; that dict has ``object: "response"`` and ``output: [...]``.
    if last.get("object") == "response":
        output_items = last.get("output") or []
        # Find function_call items not yet followed by a matching
        # function_call_output. Since this assistant ``response`` is the
        # tail of agent.messages, NO ``function_call_output`` items can
        # follow it yet — every function_call here is unmatched.
        obs_truncated = (observation_text or "")[:2000]
        fillers: list[dict[str, Any]] = []
        for item in output_items:
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type != "function_call":
                continue
            call_id = (
                item.get("call_id") if isinstance(item, dict) else None
            ) or (item.get("id") if isinstance(item, dict) else None)
            if not call_id:
                continue
            fillers.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": obs_truncated,
            })
        if fillers:
            agent.add_messages(*fillers)
        return

    # ---- Chat-Completions path (existing) ----
    if last.get("role") != "assistant":
        return
    pending_calls = last.get("tool_calls") or []
    if not pending_calls:
        return  # text-based mode, nothing to do

    # Truncate to keep the prompt small; the model already saw the
    # observation in the previous step's run.
    obs_truncated = (observation_text or "")[:2000]

    fillers = []
    for tc in pending_calls:
        # Tool calls are ChatCompletionMessageFunctionToolCall objects when
        # they pass through model_dump in mini's pipeline; after the dump
        # they're plain dicts.
        if isinstance(tc, dict):
            tc_id = tc.get("id")
        else:
            tc_id = getattr(tc, "id", None)
        if not tc_id:
            continue
        fillers.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": obs_truncated,
        })
    if fillers:
        agent.add_messages(*fillers)


def run_evolvingintent_with_mini_agent(
    user_turns: list[str],
    instance: dict,
    *,
    model_name: str,
    cost_limit: float = 3.0,
    step_limit: int = 0,  # 0 = unlimited (mini-swe-agent default)
    step_limit_per_turn: int | None = None,
    tool_call_limit_per_turn: int | None = None,
    output_path: Path | str | None = None,
    overrides: dict | None = None,
    use_tool_calling: bool = False,
) -> MiniAgentResult:
    """Run a multi-turn user intent-change conversation through mini-swe-agent v2.

    Parameters
    ----------
    user_turns
        Pre-rendered user turn texts from EvolvingIntent. Index 0 = initial task,
        1..N = intent updates / corrections / additions.
    instance
        SWE-bench Verified instance dict (must include at least
        ``instance_id``).
    model_name
        Model identifier (e.g. ``"gpt-5.1"``).
    cost_limit, step_limit
        Mini-agent termination guards. ``step_limit=0`` disables the per-step
        guard.
    step_limit_per_turn
        Optional per-turn step cap. When ``None`` (default), behavior is
        identical to today: ``step_limit`` is the global cap on the entire
        trajectory. When set, mini's ``agent.config.step_limit`` is set to
        ``step_limit_per_turn`` and ``agent.n_calls`` is reset to 0 at
        every turn boundary, giving each user-intent turn its own fresh
        budget. ``step_limit_per_turn=0`` means unlimited per turn
        (mirrors mini's existing ``step_limit=0`` semantics). When this
        kwarg is set, the legacy ``step_limit`` argument is ignored
        (warning printed).
    tool_call_limit_per_turn
        Optional cap on actual environment ``execute`` calls in each user turn.
        The hardened reproduction accepts exactly 200. Native parallel tool
        calls are disabled whenever this cap is active.
    output_path
        Optional path for mini-agent's trajectory dump.
    overrides
        Optional dict deeply merged over the default config.
    use_tool_calling
        If True, route LLM calls through native OpenAI function calling
        (``LLMToolModel`` + ``swebench.yaml``). If False (default),
        use the regex-on-fenced-blocks path (``LLMTextbasedModel`` +
        ``swebench_backticks.yaml``). Default False until 5-sample
        native sanity has been validated end-to-end.

    Returns
    -------
    MiniAgentResult with the final patch and run metadata.
    """
    if not user_turns:
        raise ValueError("user_turns must contain at least one entry")

    instance_id = instance.get("instance_id") or instance.get("original_id")
    if not instance_id:
        raise ValueError("instance dict must contain 'instance_id' or 'original_id'")

    # ----- build config -----
    if use_tool_calling:
        config = _load_default_swebench_config_native()
    else:
        config = _load_default_swebench_config()
    config.setdefault("model", {})
    config["model"]["model_name"] = model_name
    # Strip any tool-related fields from the inherited template.
    config["model"].pop("model_class", None)
    config.setdefault("agent", {})
    config["agent"]["cost_limit"] = cost_limit
    # Per-turn cap mode: agent's own step_limit becomes the per-turn cap;
    # we reset agent.n_calls at every turn boundary in the outer loop so
    # mini's existing LimitsExceeded mechanism enforces it per turn.
    # Legacy mode: agent's step_limit is the global cap, exactly as before.
    per_turn_mode = step_limit_per_turn is not None
    if per_turn_mode:
        if step_limit and step_limit > 0:
            print(
                "[swe_scaffold] both --step_limit and --step_limit_per_turn "
                "set; per-turn wins. Legacy --step_limit will be ignored."
            )
        config["agent"]["step_limit"] = int(step_limit_per_turn)
    else:
        config["agent"]["step_limit"] = step_limit
    if output_path is not None:
        config["agent"]["output_path"] = str(output_path)
    config.setdefault("environment", {})
    config["environment"]["image"] = get_swebench_docker_image_name(instance)
    config["environment"].setdefault("environment_class", "docker")

    if overrides:
        _deep_merge(config, overrides)

    tool_counter: ToolCallCounter | None = None
    if tool_call_limit_per_turn is not None:
        tool_counter = ToolCallCounter(int(tool_call_limit_per_turn))
        config.setdefault("model", {}).setdefault("model_kwargs", {})[
            "parallel_tool_calls"
        ] = False
        if "kimi" in model_name.lower():
            assert_kimi_tool_policy(config["model"])
    assert_no_output_limit_fields(config, location="mini-swe-agent config")

    # ----- build env / model / agent -----
    base_env = get_environment(config["environment"])
    env = (
        _ToolBudgetEnvironment(base_env, tool_counter)
        if tool_counter is not None
        else base_env
    )
    if use_tool_calling:
        # Route to Responses-API model only when the deployment requires
        # it (currently gpt-5.5; see llm_utils._requires_responses_api_for_tools).
        # All other native-tool-calling models (gpt-5.1, kimi, grok,
        # deepseek) stay on the Chat-Completions path so existing
        # validated baselines remain code-path-stable.
        from intent_construction.intent_extraction.core.llm_utils import _requires_responses_api_for_tools
        if _requires_responses_api_for_tools(model_name):
            model = LLMResponseToolModel(**config.get("model", {}))
        else:
            model = LLMToolModel(**config.get("model", {}))
    else:
        model = LLMTextbasedModel(**config.get("model", {}))
    agent = DefaultAgent(model=model, env=env, **config.get("agent", {}))

    # ----- seed messages: system + user_turns[0] (rendered through instance template) -----
    agent.extra_template_vars |= {"task": user_turns[0]}
    agent.messages = []
    agent.add_messages(
        agent.model.format_message(
            role="system",
            content=agent._render_template(agent.config.system_template),
        ),
        agent.model.format_message(
            role="user",
            content=agent._render_template(agent.config.instance_template),
        ),
    )

    # ----- drive step() loop with multi-turn user injection -----
    delivered = 1  # turn 0 is in the seeded instance template
    n_steps = 0
    submission = ""
    exit_status = ""
    error: str | None = None

    # Per-turn step accounting (only meaningful in per-turn mode).
    # ``turn_start_n_calls`` tracks ``agent.n_calls`` at the start of the
    # current turn so we can compute that turn's step count via delta —
    # this counts steps even when a turn ends via exception (Submitted /
    # LimitsExceeded), which is more accurate than incrementing on the
    # success path of agent.step().
    per_turn_steps: list[int] = []
    turn_start_n_calls = 0
    per_turn_tool_calls: list[int] = []

    def _finalize_current_turn() -> None:
        """Record current turn's step count; reset n_calls if per-turn mode.

        Safe to call multiple times (idempotent: turn_start_n_calls is
        re-anchored after each call).
        """
        nonlocal turn_start_n_calls
        if not per_turn_mode and tool_counter is None:
            return
        if per_turn_mode:
            steps_this_turn = max(0, agent.n_calls - turn_start_n_calls)
            per_turn_steps.append(steps_this_turn)
            agent.n_calls = 0
            turn_start_n_calls = 0
        if tool_counter is not None:
            per_turn_tool_calls.append(tool_counter.finish_turn())

    try:
        while True:
            try:
                agent.step()
                n_steps += 1
                agent.n_consecutive_format_errors = 0
            except EnvironmentExecutionError as e:
                exit_status = "EnvironmentExecutionError"
                error = f"EnvironmentExecutionError: {e}"
                _finalize_current_turn()
                break
            except FormatError as e:
                # Match DefaultAgent.run: malformed responses are retried only
                # up to mini-swe-agent's configured consecutive-error limit.
                agent.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                agent.n_consecutive_format_errors += 1
                limit = agent.config.max_consecutive_format_errors
                if 0 < limit <= agent.n_consecutive_format_errors:
                    agent.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {
                                "exit_status": "RepeatedFormatError",
                                "submission": "",
                            },
                        },
                    )
                    exit_status = "RepeatedFormatError"
                    error = (
                        "RepeatedFormatError: model returned no valid tool call "
                        f"{agent.n_consecutive_format_errors} consecutive times"
                    )
                    _finalize_current_turn()
                    break
                agent.add_messages(*e.messages)
            except Submitted as e:
                if delivered < len(user_turns):
                    # Native-mode protocol fix: ensure the assistant's
                    # tool_calls (the BASH submit call that triggered
                    # this exception) is paired with role:"tool"
                    # observation messages BEFORE we append the
                    # user intent update. No-op in text-based mode.
                    submit_obs = ""
                    if e.messages:
                        # mini's Submitted carries the magic-string
                        # observation in messages[0]["extra"]["submission"]
                        # or the exit message content.
                        meta = e.messages[-1].get("extra") or {}
                        submit_obs = meta.get("submission") or ""
                    if use_tool_calling:
                        _ensure_pending_tool_call_resolved(agent, submit_obs)

                    # Finalize this turn (record step count) and rearm
                    # the per-turn budget BEFORE injecting next user
                    # intent turn.
                    _finalize_current_turn()

                    next_turn = user_turns[delivered]
                    interrupt = agent.model.format_message(
                        role="user",
                        content=_wrap_intent_update(next_turn, on_submit=True),
                    )
                    agent.add_messages(interrupt)
                    delivered += 1
                else:
                    agent.add_messages(*e.messages)
                    extra = e.messages[-1].get("extra", {}) if e.messages else {}
                    submission = extra.get("submission", "") or ""
                    exit_status = extra.get("exit_status", "Submitted")
                    _finalize_current_turn()
                    break
            except LimitsExceeded as e:
                # mini raises LimitsExceeded for BOTH step and cost cap
                # hits. Distinguish here so we never silently advance
                # past a cost-cap breach (cost is global, not per-turn).
                cost_capped = (
                    0 < float(agent.config.cost_limit or 0)
                    <= float(getattr(agent, "cost", 0.0) or 0.0)
                )
                if (not cost_capped) and per_turn_mode and delivered < len(user_turns):
                    # Per-turn step exhaustion mid-conversation: synthesize
                    # tool-result fillers (no-op when no pending tool_calls
                    # / function_call), advance to next intent turn,
                    # rearm per-turn budget. Crucially, do NOT call
                    # agent.add_messages(*e.messages) — that would inject
                    # a role:"exit" message and confuse downstream code.
                    if use_tool_calling:
                        _ensure_pending_tool_call_resolved(agent, "")

                    _finalize_current_turn()

                    next_turn = user_turns[delivered]
                    interrupt = agent.model.format_message(
                        role="user",
                        content=_wrap_intent_update(next_turn, on_submit=True),
                    )
                    agent.add_messages(interrupt)
                    delivered += 1
                else:
                    # Terminal: cost cap hit, or final-turn exhaustion,
                    # or legacy single-cap mode. Append exit message,
                    # autosubmit, break.
                    agent.add_messages(*e.messages)
                    if cost_capped:
                        exit_status = "CostLimitExceeded"
                    elif isinstance(e, ToolCallLimitExceeded):
                        exit_status = "ToolCallLimitExceeded"
                    else:
                        exit_status = "LimitsExceeded"
                    submission = _autosubmit_via_git_diff(base_env)
                    _finalize_current_turn()
                    break
            except InterruptAgentFlow as e:
                agent.add_messages(*e.messages)
            finally:
                agent.save(agent.config.output_path)
    except Exception as e:  # pragma: no cover (defensive)
        import traceback
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        exit_status = "Error"
        try:
            submission = _autosubmit_via_git_diff(base_env)
        except Exception:
            submission = ""
        _finalize_current_turn()

    if not submission and exit_status not in (
        "Submitted",
        "RepeatedFormatError",
        "EnvironmentExecutionError",
    ):
        try:
            submission = _autosubmit_via_git_diff(base_env)
        except Exception:
            pass

    cost = float(getattr(agent, "cost", 0.0) or 0.0)

    # ----- close container -----
    _close_environment(base_env)

    # In per-turn mode, the global n_steps may undercount (it skips
    # exception paths). Use the per-turn list as ground truth, falling
    # back to n_steps when that mode is off.
    if per_turn_mode:
        n_steps_total = sum(per_turn_steps)
    else:
        n_steps_total = n_steps

    return MiniAgentResult(
        instance_id=instance_id,
        submission=submission,
        exit_status=exit_status or "Unknown",
        n_steps=n_steps_total,
        n_user_turns_delivered=delivered,
        cost=cost,
        error=error,
        final_messages=list(agent.messages),
        scaffold_mode="native" if use_tool_calling else "textbased",
        step_limit_per_turn=step_limit_per_turn if per_turn_mode else None,
        per_turn_steps=per_turn_steps,
        tool_call_limit_per_turn=(
            TOOL_CALL_LIMIT_PER_TURN if tool_counter is not None else None
        ),
        per_turn_tool_calls=per_turn_tool_calls,
    )


# =============================================================================
# Helpers
# =============================================================================


def _deep_merge(dst: dict, src: dict) -> None:
    """Recursive in-place merge of ``src`` into ``dst``."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _autosubmit_via_git_diff(env: Any) -> str:
    """Last-resort patch extraction: ``git add -A && git diff --cached`` inside the env.

    DockerEnvironment.execute expects an action dict ``{"command": str}``.
    """
    action = {"command": "cd /testbed && git add -A && git diff --cached"}
    try:
        out = env.execute(action, timeout=60)
    except TypeError:
        out = env.execute(action)
    if isinstance(out, dict):
        if out.get("returncode") not in (0, None):
            return ""
        return (out.get("output") or "").strip()
    if isinstance(out, str):
        return out.strip()
    return ""


def _close_environment(env: Any) -> None:
    """Close Docker environments and stop SWE-ReX environments."""
    for method_name in ("close", "stop"):
        method = getattr(env, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            return
