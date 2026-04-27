"""
dashboard/pages/chat.py — Live Support Copilot chat UI.

Stub page — grows as modules are completed:
  Module 1: basic LLM chat
  Module 2: RAG-augmented answers
  Module 4: tool-calling agent
  Module 5: multi-agent graph
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("💬 Support Copilot Chat")
    st.info(
        "🚧 **Chat unlocks in Module 1.**\n\n"
        "Complete **Module 1 – LLM-agnostic Client** to enable live chat here.",
        icon="🔒",
    )

    st.markdown("### What this page will do")
    st.markdown(
        """
| Module | Feature added |
|--------|--------------|
| M1 | Basic LLM chat (streaming) |
| M2 | RAG-augmented answers with citations |
| M4 | Tool-calling (order lookup, ticket creation) |
| M5 | Multi-agent graph with live step streaming |
| M6 | Guardrails: PII redaction + jailbreak shield |
| M8 | Per-user memory + personalization |
"""
    )
