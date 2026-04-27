"""
app/prompts/loader.py — Versioned prompt template loader using Jinja2.

Why versioned templates?
  Prompts are code. They have bugs, regressions, and improvements. Versioning
  them (v1, v2, …) lets you A/B test, roll back, and audit exactly what the
  model received in any historical trace.

Convention
----------
  Templates live in app/prompts/ as <name>_v<N>.j2
  Caller pins a version: load("support_agent", version=1)
  All variables are passed as keyword args → render(context=..., tone=...)

Jinja2 chosen over string.Template because:
  - Conditional blocks, loops — useful for few-shot examples
  - Auto-escaping off (we want raw text, not HTML-escaped)
  - Whitespace control with {%- ... -%}
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_PROMPTS_DIR = Path(__file__).parent

# StrictUndefined raises immediately if a variable is missing in the template.
# This catches prompt bugs at render time, not silently at inference time.
_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def load(name: str, *, version: int = 1, **variables: object) -> str:
    """
    Load and render a prompt template.

    Parameters
    ----------
    name:      Template base name (e.g. "support_agent")
    version:   Version number (e.g. 1 → looks for support_agent_v1.j2)
    **variables: Jinja2 template variables to interpolate

    Returns
    -------
    Rendered prompt string, ready to use as a message content.

    Example
    -------
    >>> system_prompt = load("support_agent", version=1, company="Acme")
    """
    filename = f"{name}_v{version}.j2"
    template = _env.get_template(filename)
    return template.render(**variables)


def list_templates() -> list[str]:
    """Return all available template filenames (useful for debugging)."""
    return sorted(p.name for p in _PROMPTS_DIR.glob("*.j2"))
