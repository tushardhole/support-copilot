"""
dashboard/pages/journey.py — Curriculum progress tracker.

Reads dashboard/tracker.yaml and renders:
  - Overall progress bar
  - Per-module expandable cards with objectives, concepts, interview Qs
  - Status toggles that write back to tracker.yaml
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

TRACKER_PATH = Path(__file__).resolve().parent.parent / "tracker.yaml"

STATUS_EMOJI = {
    "pending": "⬜",
    "in_progress": "🔄",
    "done": "✅",
}

STATUS_COLOR = {
    "pending": "#6b7280",
    "in_progress": "#f59e0b",
    "done": "#10b981",
}

STATUS_LABEL = {
    "pending": "Pending",
    "in_progress": "In Progress",
    "done": "Done ✅",
}


def _load() -> dict:
    with open(TRACKER_PATH) as f:
        return yaml.safe_load(f)


def _save(data: dict) -> None:
    with open(TRACKER_PATH, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def render() -> None:
    st.title("📚 Learning Journey")
    st.caption("Track your progress through the AI Engineering capstone — one module at a time.")

    data = _load()
    modules = data.get("curriculum", [])

    # ── Overall progress ──────────────────────────────────────────────────────
    total = len(modules)
    done_count = sum(1 for m in modules if m["status"] == "done")
    in_progress_count = sum(1 for m in modules if m["status"] == "in_progress")
    progress_ratio = done_count / total if total else 0.0

    col_prog, col_stats = st.columns([3, 1])
    with col_prog:
        st.subheader(f"Overall Progress — {done_count}/{total} modules complete")
        st.progress(progress_ratio)
    with col_stats:
        st.metric("Done", done_count)
        st.metric("In Progress", in_progress_count)
        st.metric("Remaining", total - done_count - in_progress_count)

    st.divider()

    # ── Module cards ──────────────────────────────────────────────────────────
    for idx, module in enumerate(modules):
        status = module.get("status", "pending")
        emoji = STATUS_EMOJI[status]
        color = STATUS_COLOR[status]

        header = f"{emoji} {module['title']}"

        # Auto-expand the first in_progress module
        expanded = status == "in_progress"

        with st.expander(header, expanded=expanded):
            # Status badge
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 10px;"
                f"border-radius:12px;font-size:0.8rem;font-weight:600'>"
                f"{STATUS_LABEL[status]}</span>",
                unsafe_allow_html=True,
            )
            st.write("")

            # Three-column layout: objectives | concepts | interview Qs
            col_obj, col_con, col_iq = st.columns(3)

            with col_obj:
                st.markdown("**🎯 Objectives**")
                for obj in module.get("objectives", []):
                    st.markdown(f"- {obj}")

            with col_con:
                st.markdown("**🧠 Concepts**")
                for con in module.get("concepts", []):
                    st.markdown(f"- {con}")

            with col_iq:
                st.markdown("**💡 Interview Questions**")
                for iq in module.get("interview_qs", []):
                    st.markdown(f"- *{iq}*")

            # Code files
            files = module.get("code_files", [])
            if files:
                st.markdown("**📁 Key Files**")
                st.code("  ".join(files), language=None)

            # Notes textarea
            notes_key = f"notes_{module['id']}"
            current_notes = module.get("notes", "") or ""
            notes = st.text_area(
                "Notes / reflections",
                value=current_notes,
                key=notes_key,
                height=80,
                placeholder="Add your notes here…",
            )

            # Status selector + save
            col_status, col_save = st.columns([2, 1])
            with col_status:
                status_options = list(STATUS_LABEL.keys())
                new_status = st.selectbox(
                    "Status",
                    status_options,
                    index=status_options.index(status),
                    key=f"status_{module['id']}",
                    format_func=lambda s: STATUS_LABEL[s],
                )
            with col_save:
                st.write("")  # vertical spacer
                st.write("")
                if st.button("💾 Save", key=f"save_{module['id']}"):
                    # Re-load, patch, save to avoid overwriting parallel edits
                    fresh = _load()
                    for m in fresh["curriculum"]:
                        if m["id"] == module["id"]:
                            m["status"] = new_status
                            m["notes"] = notes
                            break
                    _save(fresh)
                    st.success("Saved!")
                    st.rerun()
