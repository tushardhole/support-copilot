# Support Copilot

> A Customer Support AI built module-by-module across a 10-module AI Engineering capstone.
> Each module introduces one production concept — read, build, run, reflect.

**Stack:** OpenAI SDK (vendor-neutral) · ChromaDB · LangGraph · Ragas · Presidio · FastAPI · Streamlit · SQLite · uv

---

## Quick Start

```bash
# 1. Clone
git clone git@github.com:tushardhole/support-copilot.git
cd support-copilot

# 2. Copy and fill in your secrets
cp .env.example .env

# 3. Create virtual environment and install all deps (< 30s)
uv sync

# 4. Launch the Tracker Dashboard
uv run streamlit run dashboard/app.py
```

Open http://localhost:8501 — the **Journey** page shows all 10 modules with progress, learning objectives, concepts, and interview questions. Status and notes persist to `dashboard/tracker.yaml`.

---

## Repo Layout

```
support-copilot/
  app/
    config.py            # pydantic-settings: model, base_url, api_key, embed model
    llm.py               # LLM-agnostic client + retries + structured outputs  [M1]
    prompts/             # versioned prompt templates                           [M1]
    rag/                 # ingest, chunk, embed, retrieve, rerank               [M2-M3]
    tools/               # function-calling tools (kb_search, orders, tickets)  [M4]
    agents/              # single agent [M4], LangGraph multi-agent [M5]
    guardrails/          # PII, jailbreak, output validators                    [M6]
    evals/               # Ragas + LLM-as-judge datasets & runners              [M3, M7]
    api/                 # FastAPI SSE streaming endpoint                        [M9]
  data/
    kb/                  # markdown / PDFs (sample support knowledge base)
    db.sqlite            # tickets, users, orders                               [M4+]
  dashboard/
    app.py               # Streamlit: Journey | Chat | Evals | Traces
    tracker.yaml         # curriculum state (all 10 modules, status, notes)
    pages/
      journey.py         # module progress cards with save-to-yaml              [M0]
      chat.py            # live copilot chat                                    [M1+]
      evals.py           # Ragas + LLM-judge scores over time                  [M3+]
      traces.py          # agent run viewer + Langfuse links                   [M5+]
  tests/
  evals_results/
  pyproject.toml
  .env.example
  README.md
```

---

## Modules

| # | Title | Status |
|---|-------|:------:|
| 0 | Scaffold + Tracker Dashboard | ✅ |
| 1 | LLM-agnostic Client + Structured Outputs | ✅ |
| 2 | RAG v1 — Naive (chunk → embed → retrieve → answer) | ✅ |
| 3 | RAG v2 — Production (BM25+dense, rerank, HyDE, Ragas evals) | ✅ |
| 4 | Tools + Single Agent (ReAct, tool registry, SQLite) | ⬜ |
| 5 | Multi-agent with LangGraph (Triage → specialists → Reviewer) | ⬜ |
| 6 | Guardrails + Safety (Presidio PII, jailbreak, output grounding) | ⬜ |
| 7 | Evals + Observability (3-tier evals, Langfuse tracing) | ⬜ |
| 8 | Memory + Personalization (buffer, summary, per-user store) | ⬜ |
| 9 | Deployment (FastAPI SSE, Docker, docker-compose) | ⬜ |
| 10 | System Design + Interview Prep (25 high-signal Qs) | ⬜ |

---

## Key Design Decisions

### Vendor-neutral LLM client
`app/config.py` exposes `openai_base_url` + `openai_api_key`. The client (`app/llm.py`, M1) constructs `openai.OpenAI(base_url=..., api_key=...)`. Switching from OpenAI → Ollama → vLLM = two env-var changes, zero code changes.

### 12-factor config
All secrets and runtime settings live in `.env` (never committed). `pydantic-settings` reads them with full type validation and sane defaults. See `app/config.py`.

### Dependency management with uv
`uv sync` installs from `pyproject.toml` into an isolated `.venv` reproducibly across machines. All module dependencies are declared upfront — `uv add <pkg>` as each module is implemented.

### Dashboard as living surface
The Streamlit dashboard grows with every module. It starts as a curriculum tracker (M0) and becomes the live interface for chat, evals, and traces by M5+.
