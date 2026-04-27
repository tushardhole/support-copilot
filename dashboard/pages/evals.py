"""
dashboard/pages/evals.py — Ragas + LLM-judge evaluation scores.

Reads evals_results/ragas_results.json (produced by `python -m app.evals.ragas_eval`).
Module 3: Ragas RAG metrics.
Module 7: LLM-as-judge + pytest regression suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parents[2] / "evals_results" / "ragas_results.json"

METRIC_DESCRIPTIONS = {
    "faithfulness": "Is the answer supported by retrieved context? (hallucination check)",
    "answer_relevancy": "Is the answer relevant to the question?",
    "context_precision": "Are retrieved chunks actually needed to answer?",
    "context_recall": "Does context cover the ground-truth answer?",
}


def _load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def render() -> None:
    st.title("📊 Evaluation Scores")

    results = _load_results()

    if results is None:
        st.info(
            "No eval results found. Run the Ragas evaluation to populate this page:\n\n"
            "```bash\nuv run python -m app.evals.ragas_eval\n```",
            icon="📋",
        )
        st.markdown("### Planned eval tiers")
        st.markdown(
            """
| Tier | Tool | Metrics | Module |
|------|------|---------|--------|
| Component – RAG | Ragas | Faithfulness, Answer Relevancy, Context Precision/Recall | M3 ✅ |
| Unit – deterministic | pytest | Pass/fail on fixed inputs | M7 |
| End-to-end – LLM judge | Custom rubric | Helpfulness, Safety, Groundedness | M7 |
"""
        )
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    scores = results.get("scores", {})
    st.subheader("Ragas Scores")
    st.caption(
        f"Retrieval mode: **{results.get('retrieval_mode', '?')}** · "
        f"Top-K: **{results.get('top_k', '?')}** · "
        f"Questions: **{results.get('num_questions', '?')}**"
    )

    cols = st.columns(4)
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col, key in zip(cols, metric_keys):
        val = scores.get(key, 0.0)
        color = "normal" if val >= 0.7 else "inverse"
        col.metric(label=key.replace("_", " ").title(), value=f"{val:.3f}")

    # ── Radar chart ───────────────────────────────────────────────────────────
    vals = [scores.get(k, 0.0) for k in metric_keys]
    labels = [k.replace("_", " ").title() for k in metric_keys]

    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="RAG v2 (hybrid)",
        line_color="royalblue",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metric explanations ───────────────────────────────────────────────────
    with st.expander("What do these metrics mean?"):
        for key, desc in METRIC_DESCRIPTIONS.items():
            score = scores.get(key, 0.0)
            st.markdown(f"**{key.replace('_', ' ').title()}** ({score:.3f}): {desc}")

    # ── Per-question breakdown ────────────────────────────────────────────────
    st.divider()
    st.subheader("Per-question breakdown")
    per_q = results.get("per_question", [])
    for i, item in enumerate(per_q, 1):
        with st.expander(f"Q{i}: {item['question']}"):
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown(f"**Ground truth:** {item['ground_truth']}")
            st.markdown("**Retrieved contexts:**")
            for j, ctx in enumerate(item.get("contexts", []), 1):
                st.markdown(f"*[{j}]* {ctx[:300]}{'…' if len(ctx) > 300 else ''}")

    # ── Roadmap for M7 ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("🗺️ Module 7 additions"):
        st.markdown(
            """
| Feature | Description |
|---------|-------------|
| Unit evals | `pytest` deterministic tests on fixed inputs |
| LLM-as-judge | Rubric-based scoring: helpfulness, safety, groundedness |
| Langfuse traces | Cost + latency per span; links from eval run → trace |
| Score history | Trend chart across multiple eval runs |
"""
        )
