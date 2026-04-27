"""
dashboard/pages/traces.py — Agent run viewer + Langfuse trace links.

Stub page — grows with Module 5 (streaming traces) and Module 7 (Langfuse).
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("🔍 Agent Traces")
    st.info(
        "🚧 **Traces unlock in Module 5.**\n\n"
        "Complete **Module 5 – Multi-agent LangGraph** to stream live traces here.",
        icon="🔒",
    )

    st.markdown("### What traces will show")
    st.markdown(
        """
| Module | Trace feature |
|--------|--------------|
| M5 | LangGraph node execution stream (in-app viewer) |
| M7 | Langfuse integration: cost, latency, token counts per span |
| M7 | Clickable Langfuse deep-links per run |
"""
    )
