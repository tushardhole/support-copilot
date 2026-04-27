"""
tests/test_prompts.py — Unit tests for the prompt template loader.

No LLM calls; tests only template rendering correctness.
Run: uv run pytest tests/test_prompts.py -v
"""

from __future__ import annotations

import pytest

from app.prompts.loader import list_templates, load


def test_load_support_agent_renders():
    prompt = load("support_agent", version=1, company="TestCorp")
    assert "TestCorp" in prompt
    assert "customer support agent" in prompt


def test_load_support_agent_default_company():
    prompt = load("support_agent", version=1)
    assert "Acme" in prompt


def test_load_support_agent_with_context():
    prompt = load("support_agent", version=1, context="The refund policy is 30 days.")
    assert "refund policy" in prompt
    assert "Relevant knowledge base context" in prompt


def test_load_support_agent_without_context_no_kb_section():
    prompt = load("support_agent", version=1)
    assert "Relevant knowledge base context" not in prompt


def test_load_support_agent_with_tools():
    prompt = load("support_agent", version=1, tools=["kb_search", "lookup_order"])
    assert "kb_search" in prompt
    assert "lookup_order" in prompt


def test_load_triage():
    prompt = load("triage", version=1, message="I can't log in to my account")
    assert "I can't log in" in prompt
    assert "account" in prompt


def test_load_missing_template_raises():
    with pytest.raises(Exception):
        load("nonexistent_template", version=99)


def test_load_missing_variable_raises():
    """StrictUndefined means missing variables raise instead of silently rendering empty."""
    with pytest.raises(Exception):
        # triage_v1.j2 requires {{ message }}, so omitting it should fail
        load("triage", version=1)


def test_list_templates_returns_j2_files():
    templates = list_templates()
    assert any(t.endswith(".j2") for t in templates)
    assert "support_agent_v1.j2" in templates
    assert "triage_v1.j2" in templates
