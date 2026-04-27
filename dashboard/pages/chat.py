"""
dashboard/pages/chat.py — Live Support Copilot chat UI.

Module 1: basic streaming chat.
Module 2: RAG toggle — retrieves KB context and cites sources.
Future modules add tool calls (M4), multi-agent graph (M5), guardrails (M6).
"""

from __future__ import annotations

import streamlit as st

from app.config import settings
from app.llm import LLMClient
from app.prompts.loader import load


def _get_client() -> LLMClient:
    if "llm_client" not in st.session_state:
        st.session_state["llm_client"] = LLMClient()
    return st.session_state["llm_client"]


def _rag_retrieve(query: str, top_k: int) -> tuple[str, list]:
    """Lazy import so the page loads even before the KB is ingested."""
    try:
        from app.rag.retrieve import retrieve_and_format
        return retrieve_and_format(query, top_k=top_k)
    except Exception as exc:
        st.warning(f"RAG retrieval error: {exc}", icon="⚠️")
        return "", []


def render() -> None:
    st.title("💬 Support Copilot Chat")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Chat settings")
        model_override = st.text_input("Model", value=settings.llm_model)
        temperature = st.slider("Temperature", 0.0, 1.0, float(settings.llm_temperature), 0.05)
        st.divider()
        use_rag = st.toggle("🔍 Use RAG (KB lookup)", value=False)
        if use_rag:
            top_k = st.slider("Top-K chunks", 1, 10, 5)
        else:
            top_k = 5
        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

    # ── Config / KB warnings ──────────────────────────────────────────────────
    if not settings.openai_api_key:
        st.warning("No `OPENAI_API_KEY` in `.env` — live chat disabled.", icon="⚠️")

    if use_rag:
        try:
            from app.rag.store import vector_store
            count = vector_store.count()
            if count == 0:
                st.info(
                    "KB is empty. Run `uv run python -m app.rag.ingest` to index `data/kb/`.",
                    icon="📚",
                )
            else:
                st.caption(f"📚 KB: {count} chunks indexed")
        except Exception:
            pass

    # ── Chat history ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**[{src['rank']}]** `{src['source']}` — "
                            f"score={src['score']:.2f}, chunk #{src['chunk_index']}"
                        )

    # ── Input ─────────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask the support copilot…"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Build messages ────────────────────────────────────────────────────
        retrieved_chunks = []
        if use_rag:
            with st.spinner("Searching KB…"):
                context, retrieved_chunks = _rag_retrieve(user_input, top_k)

            if context:
                system_prompt = load("rag_answer", version=1, company="Acme",
                                     context=context, question=user_input)
                api_messages = [{"role": "system", "content": system_prompt}]
            else:
                system_prompt = load("support_agent", version=1, company="Acme")
                api_messages = [{"role": "system", "content": system_prompt}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"]
                ]
        else:
            system_prompt = load("support_agent", version=1, company="Acme")
            api_messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"]
            ]

        # ── Stream response ───────────────────────────────────────────────────
        client = _get_client()
        with st.chat_message("assistant"):
            response_text = st.write_stream(
                client.chat_stream(api_messages, model=model_override or None, temperature=temperature)
            )
            if retrieved_chunks:
                with st.expander("📎 Sources"):
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        from pathlib import Path
                        st.markdown(
                            f"**[{i}]** `{Path(chunk.source).name}` — "
                            f"score={chunk.score:.2f}, chunk #{chunk.chunk_index}"
                        )

        sources = [
            {"rank": i + 1, "source": chunk.source,
             "score": chunk.score, "chunk_index": chunk.chunk_index}
            for i, chunk in enumerate(retrieved_chunks)
        ]
        st.session_state["messages"].append(
            {"role": "assistant", "content": response_text, "sources": sources}
        )
        mode_label = f"RAG (top-{top_k})" if use_rag else "direct"
        st.caption(f"Model: `{model_override or settings.llm_model}` · mode: {mode_label}")

    # ── Roadmap ───────────────────────────────────────────────────────────────
    with st.expander("🗺️ Roadmap — features per module"):
        st.markdown(
            """
| Module | Feature |
|--------|---------|
| M1 ✅ | Streaming LLM chat + versioned system prompt |
| M2 ✅ | RAG: KB lookup → cited answers |
| M4 | Tool calling: order lookup, ticket creation |
| M5 | Multi-agent: triage → specialist → reviewer |
| M6 | Guardrails: PII redaction, jailbreak shield |
| M8 | Per-user memory + personalization |
"""
        )
