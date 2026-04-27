"""
dashboard/pages/evals.py — Ragas + LLM-judge evaluation scores.

Stub page — grows with Module 3 (Ragas) and Module 7 (LLM-as-judge + pytest).
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("📊 Evaluation Scores")
    st.info(
        "🚧 **Evals unlock in Module 3.**\n\n"
        "Complete **Module 3 – RAG v2** to see Ragas scores here.",
        icon="🔒",
    )

    st.markdown("### Planned eval tiers")
    st.markdown(
        """
| Tier | Tool | Metrics | Module |
|------|------|---------|--------|
| Component – RAG | Ragas | Faithfulness, Answer Relevancy, Context Precision/Recall | M3 |
| Unit – deterministic | pytest | Pass/fail on fixed inputs | M7 |
| End-to-end – LLM judge | Custom rubric | Helpfulness, Safety, Groundedness | M7 |
"""
    )
