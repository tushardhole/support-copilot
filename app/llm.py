"""
app/llm.py — LLM-agnostic client built on the OpenAI SDK.

Any OpenAI-compatible provider works: set OPENAI_BASE_URL + OPENAI_API_KEY in .env.
Providers tested: OpenAI, Ollama, vLLM, Azure OpenAI, Together AI.

Public API
----------
client = LLMClient()          # uses singleton settings
reply, usage = client.chat(messages)
obj, usage   = client.chat_structured(messages, MyPydanticModel)
reply        = client.chat_stream(messages)  # generator of str chunks

Key design choices explained inline (interview-ready comments).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, settings as _default_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Cost table ────────────────────────────────────────────────────────────────
# (input_usd_per_million, output_usd_per_million)
# These are approximate and change often — treat as a rough gauge, not billing.
_COST_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o":            (2.50,  10.00),
    "gpt-4o-mini":       (0.15,   0.60),
    "gpt-4-turbo":      (10.00,  30.00),
    "gpt-3.5-turbo":     (0.50,   1.50),
    "o1":               (15.00,  60.00),
    "o1-mini":           (3.00,  12.00),
    "o3-mini":           (1.10,   4.40),
}

# Errors that are safe to retry (transient) vs ones we should surface immediately.
_RETRYABLE = (RateLimitError, APIConnectionError)


# ── Usage / cost record ───────────────────────────────────────────────────────

class UsageRecord(BaseModel):
    """Token and cost accounting for one LLM call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def __str__(self) -> str:
        return (
            f"{self.model} | "
            f"in={self.prompt_tokens} out={self.completion_tokens} "
            f"total={self.total_tokens} | "
            f"~${self.estimated_cost_usd:.5f}"
        )


def _build_usage(model: str, prompt_tokens: int, completion_tokens: int) -> UsageRecord:
    """Compute cost from the cost table; fall back to $0 for unknown models."""
    # Partial-match: sort by key length descending so "gpt-4o-mini" is tried
    # before "gpt-4o" (which is a prefix of "gpt-4o-mini").
    matched_key = next(
        (k for k in sorted(_COST_PER_M, key=len, reverse=True) if model.startswith(k)),
        None,
    )
    if matched_key:
        in_cost, out_cost = _COST_PER_M[matched_key]
        cost = (prompt_tokens * in_cost + completion_tokens * out_cost) / 1_000_000
    else:
        cost = 0.0
        logger.debug("No cost entry for model '%s'; cost set to 0.", model)

    return UsageRecord(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=cost,
    )


# ── Retry decorator ───────────────────────────────────────────────────────────
# Why tenacity?  Built-in retry logic; declarative; composable.
# Why exponential back-off with jitter?  Avoids thundering-herd when many
# parallel calls all hit a rate-limit at the same instant.
def _make_retry():
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin, vendor-neutral wrapper over openai.OpenAI.

    Why "thin"?
      Heavy frameworks (LangChain) abstract so much that debugging becomes hard
      and upgrading the SDK breaks unpredictably.  A thin wrapper gives full
      control while remaining vendor-neutral via base_url.

    Structured output strategy (in order of preference):
      1. json_schema strict mode  — guarantees schema conformance; gpt-4o+.
      2. JSON mode + validate     — any model; may need a repair loop.
      3. Repair loop              — append error + schema, ask model to fix.
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        cfg = cfg or _default_settings
        # openai.OpenAI accepts any base_url → works with Ollama, vLLM, etc.
        self._client = OpenAI(
            base_url=cfg.openai_base_url,
            api_key=cfg.openai_api_key or "ollama",  # Ollama ignores the key
        )
        self._model = cfg.llm_model
        self._max_tokens = cfg.llm_max_tokens
        self._temperature = cfg.llm_temperature

    # ── chat ──────────────────────────────────────────────────────────────────

    @_make_retry()
    def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, UsageRecord]:
        """
        Send a list of messages and return (reply_text, usage).

        Parameters override instance-level defaults so callers can tune
        per-call without re-constructing the client.
        """
        response = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        usage = _build_usage(
            response.model,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )
        logger.debug("chat() %s", usage)
        return text, usage

    # ── chat_stream ───────────────────────────────────────────────────────────

    def chat_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Stream response tokens as a generator of strings.

        Usage stats are not available mid-stream; log a warning if needed.
        Typical use: Streamlit st.write_stream() or FastAPI SSE endpoint.
        """
        stream = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    # ── chat_structured ───────────────────────────────────────────────────────

    @_make_retry()
    def chat_structured(
        self,
        messages: list[ChatCompletionMessageParam],
        schema: type[T],
        *,
        model: str | None = None,
        strict: bool = True,
        max_repair_attempts: int = 1,
        **kwargs: Any,
    ) -> tuple[T, UsageRecord]:
        """
        Return a validated Pydantic model instance from the LLM.

        Strategy
        --------
        1. Try json_schema strict mode (gpt-4o and newer OpenAI-compat models).
           This guarantees schema conformance at the API level.
        2. If that fails (older models / providers), fall back to json_object
           mode, parse manually, and run a repair loop on ValidationError.

        Interview note
        --------------
        "How do you guarantee structured output?"
          → Schema-level enforcement (strict mode) is the gold standard.
            For models that don't support it: json_object mode + Pydantic
            validation + a repair loop that appends the error to messages and
            asks the model to fix it.  Cap repairs at 1-2 to avoid loops.
        """
        used_model = model or self._model
        json_schema = schema.model_json_schema()

        # ── Attempt 1: strict json_schema mode ────────────────────────────────
        try:
            response = self._client.chat.completions.create(
                model=used_model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema,
                        "strict": strict,
                    },
                },
                max_tokens=self._max_tokens,
                **kwargs,
            )
            raw = response.choices[0].message.content or ""
            obj = schema.model_validate_json(raw)
            usage = _build_usage(
                response.model,
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
            )
            logger.debug("chat_structured() strict mode OK — %s", usage)
            return obj, usage

        except (APIStatusError, ValidationError) as first_err:
            logger.warning(
                "Strict json_schema mode failed (%s). Falling back to json_object.",
                first_err,
            )

        # ── Attempt 2: json_object mode + manual parse ────────────────────────
        fallback_messages = list(messages) + [
            {
                "role": "system",
                "content": (
                    f"You MUST respond with ONLY valid JSON that matches this schema "
                    f"exactly (no markdown, no explanation):\n{json.dumps(json_schema, indent=2)}"
                ),
            }
        ]

        for attempt in range(max_repair_attempts + 1):
            response = self._client.chat.completions.create(
                model=used_model,
                messages=fallback_messages,
                response_format={"type": "json_object"},
                max_tokens=self._max_tokens,
                **kwargs,
            )
            raw = response.choices[0].message.content or ""
            usage = _build_usage(
                response.model,
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
            )

            try:
                obj = schema.model_validate_json(raw)
                logger.debug("chat_structured() json_object attempt %d OK — %s", attempt, usage)
                return obj, usage
            except ValidationError as ve:
                if attempt < max_repair_attempts:
                    # Repair loop: tell the model exactly what was wrong.
                    logger.warning("Repair attempt %d: %s", attempt + 1, ve)
                    fallback_messages.append({"role": "assistant", "content": raw})
                    fallback_messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response failed schema validation:\n{ve}\n\n"
                            f"Fix it and return ONLY valid JSON."
                        ),
                    })
                else:
                    raise RuntimeError(
                        f"chat_structured() failed after {max_repair_attempts} repair "
                        f"attempt(s). Last error: {ve}\nRaw response: {raw}"
                    ) from ve

        raise RuntimeError("Unreachable")  # mypy satisfaction


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import `llm` wherever you need a ready-to-use client.
# Re-construct LLMClient(cfg=...) if you need a different config (e.g. in tests).
llm = LLMClient()
