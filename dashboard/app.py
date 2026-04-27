"""
dashboard/app.py — Streamlit Tracker Dashboard (multi-page shell).

Pages (stubs grow with each module):
  Journey  — curriculum progress (this module)
  Chat     — live support-copilot (Module 1+)
  Evals    — Ragas + LLM-judge scores (Module 3+)
  Traces   — agent run viewer (Module 5+)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Copilot — Tracker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = {
    "📚 Journey": "journey",
    "💬 Chat": "chat",
    "📊 Evals": "evals",
    "🔍 Traces": "traces",
}

with st.sidebar:
    st.title("🤖 Support Copilot")
    st.caption("AI Engineering Capstone")
    st.divider()
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Built module-by-module →\nsay **'do Module N'** to continue.")

active = PAGES[page]

# ── Route pages ───────────────────────────────────────────────────────────────
if active == "journey":
    from dashboard.pages.journey import render
    render()
elif active == "chat":
    from dashboard.pages.chat import render
    render()
elif active == "evals":
    from dashboard.pages.evals import render
    render()
elif active == "traces":
    from dashboard.pages.traces import render
    render()
