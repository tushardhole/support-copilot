"""
tests/test_llm.py — Unit tests for app/llm.py.

All tests mock the OpenAI client so no API key is required.
Run: uv run pytest tests/test_llm.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from app.llm import LLMClient, UsageRecord, _build_usage


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_response(content: str, model: str = "gpt-4o-mini", prompt_tokens: int = 10, completion_tokens: int = 20):
    """Build a minimal mock that looks like openai.types.chat.ChatCompletion."""
    response = MagicMock()
    response.model = model
    response.choices[0].message.content = content
    response.choices[0].delta.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


# ── UsageRecord / cost ────────────────────────────────────────────────────────

def test_build_usage_known_model():
    usage = _build_usage("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)
    assert abs(usage.estimated_cost_usd - 0.15) < 0.001


def test_build_usage_unknown_model():
    usage = _build_usage("some-unknown-llm", prompt_tokens=100, completion_tokens=100)
    assert usage.estimated_cost_usd == 0.0


def test_build_usage_partial_model_name():
    # "gpt-4o-mini-2024-07-18" should match "gpt-4o-mini" prefix
    usage = _build_usage("gpt-4o-mini-2024-07-18", prompt_tokens=0, completion_tokens=1_000_000)
    assert abs(usage.estimated_cost_usd - 0.60) < 0.001


def test_usage_record_str():
    u = UsageRecord(
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.00005,
    )
    s = str(u)
    assert "gpt-4o-mini" in s
    assert "in=100" in s
    assert "out=50" in s


# ── LLMClient.chat ────────────────────────────────────────────────────────────

def test_chat_returns_text_and_usage():
    mock_response = _make_mock_response("Hello! How can I help?")
    with patch("app.llm.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        text, usage = client.chat([{"role": "user", "content": "Hi"}])

    assert text == "Hello! How can I help?"
    assert isinstance(usage, UsageRecord)
    assert usage.total_tokens == 30


def test_chat_respects_override_params():
    mock_response = _make_mock_response("ok", model="gpt-4o")
    with patch("app.llm.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        client.chat(
            [{"role": "user", "content": "test"}],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=512,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]

    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 512


# ── LLMClient.chat_structured ─────────────────────────────────────────────────

class SupportCategory(BaseModel):
    category: str
    confidence: float
    reason: str


def test_chat_structured_strict_mode_success():
    payload = '{"category": "billing", "confidence": 0.95, "reason": "refund request"}'
    mock_response = _make_mock_response(payload)
    with patch("app.llm.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient()
        obj, usage = client.chat_structured(
            [{"role": "user", "content": "I want a refund"}],
            SupportCategory,
        )

    assert obj.category == "billing"
    assert obj.confidence == 0.95
    assert isinstance(usage, UsageRecord)


def test_chat_structured_repair_loop():
    """Strict mode raises APIStatusError; fallback json_object mode succeeds after 1 repair."""
    from openai import APIStatusError

    bad_json = '{"wrong_field": 42}'
    good_json = '{"category": "technical", "confidence": 0.8, "reason": "bug report"}'

    strict_error = APIStatusError(
        message="strict json_schema not supported",
        response=MagicMock(status_code=400, headers={}),
        body={"error": {"message": "unsupported"}},
    )

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise strict_error
        if call_count == 2:
            return _make_mock_response(bad_json)
        return _make_mock_response(good_json)

    with patch("app.llm.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.side_effect = side_effect

        client = LLMClient()
        obj, usage = client.chat_structured(
            [{"role": "user", "content": "The app crashes"}],
            SupportCategory,
            strict=True,
            max_repair_attempts=1,
        )

    assert obj.category == "technical"
    assert call_count == 3  # strict attempt + bad json + repaired good json


# ── LLMClient.chat_stream ─────────────────────────────────────────────────────

def test_chat_stream_yields_chunks():
    chunks = ["Hello", " world", "!"]

    def _make_chunk(content):
        c = MagicMock()
        c.choices[0].delta.content = content
        return c

    mock_stream = iter([_make_chunk(c) for c in chunks])

    with patch("app.llm.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_stream

        client = LLMClient()
        result = list(client.chat_stream([{"role": "user", "content": "Hi"}]))

    assert result == chunks
