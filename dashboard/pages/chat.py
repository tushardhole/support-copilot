"""
dashboard/pages/chat.py — Live Support Copilot chat UI.

Module 1: basic streaming chat with the LLM client.
Future modules add RAG context (M2), tool calls (M4), multi-agent graph (M5).
"""

from __future__ import annotations

import streamlit as st

from app.config import settings
from app.llm import LLMClient
from app.prompts.loader import load


def _get_client() -> LLMClient:
    """Cache the client in session state so we don't reconstruct per rerun."""
    if "llm_client" not in st.session_state:
        st.session_state["llm_client"] = LLMClient()
    return st.session_state["llm_client"]


def render() -> None:
    st.title("💬 Support Copilot Chat")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Chat settings")
        model_override = st.text_input("Model", value=settings.llm_model)
        temperature = st.slider("Temperature", 0.0, 1.0, float(settings.llm_temperature), 0.05)
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

    # ── Config check ──────────────────────────────────────────────────────────
    if not settings.openai_api_key:
        st.warning(
            "No `OPENAI_API_KEY` found in `.env`. "
            "Add your API key to enable live chat.",
            icon="⚠️",
        )

    # ── Chat history ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input ─────────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask the support copilot…"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build messages for the LLM
        system_prompt = load("support_agent", version=1, company="Acme")
        api_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"]
        ]

        # Stream response
        client = _get_client()
        with st.chat_message("assistant"):
            response_text = st.write_stream(
                client.chat_stream(
                    api_messages,
                    model=model_override or None,
                    temperature=temperature,
                )
            )

        st.session_state["messages"].append({"role": "assistant", "content": response_text})

        # ── Usage pill (shown under each reply) ───────────────────────────────
        # We can't get streaming usage stats easily; show a note instead.
        st.caption(f"Model: `{model_override or settings.llm_model}` · M2+ adds RAG context")

    # ── Capability roadmap (collapsed) ────────────────────────────────────────
    with st.expander("🗺️ Roadmap — what gets added per module"):
        st.markdown(
            """
| Module | Feature added to Chat |
|--------|----------------------|
| M1 ✅ | Streaming LLM chat + versioned system prompt |
| M2 | RAG: answers cite KB chunks |
| M4 | Tool calling: order lookup, ticket creation |
| M5 | Multi-agent: triage → specialist → reviewer |
| M6 | Guardrails: PII redaction, jailbreak shield |
| M8 | Per-user memory + personalization |
"""
        )
