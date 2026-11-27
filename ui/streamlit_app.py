"""Streamlit UI. T10 (topic and outline review), T15 (report) and T25 (analysis, chat) by Aasritha.

Run:  uv run streamlit run ui/streamlit_app.py

Screens are chosen by st.session_state.page:  topic -> outline -> writing   (T15 adds report, T25 analysis and chat)
"""

import html
import sys
import uuid
from pathlib import Path

import httpx
import streamlit as st

# Streamlit puts only this file's folder on the import path; add the project root so `app` can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_client  # noqa: E402
import styles  # noqa: E402

from app.config import settings  # noqa: E402

st.set_page_config(page_title="ResearchPilot", page_icon="🧭", layout="centered")
styles.apply()


# ---------- state helpers ----------


def init_state() -> None:
    st.session_state.setdefault("page", "topic")
    st.session_state.setdefault("topic", "")
    st.session_state.setdefault("run_id", None)
    st.session_state.setdefault("sections", [])  # outline being edited; each has a private "key" for widget keys


def reset() -> None:
    for name in ("page", "topic", "run_id", "sections"):
        st.session_state.pop(name, None)
    # widget keys of the old outline would otherwise leak into the next one
    for name in [k for k in st.session_state if str(k).startswith(("title-", "subs-", "topic_input"))]:
        st.session_state.pop(name)


def to_editable(sections: list[dict]) -> list[dict]:
    """Add a stable key per section so widgets keep their values when another section is removed."""
    return [{**s, "key": uuid.uuid4().hex[:8]} for s in sections]


def to_outline(sections: list[dict]) -> list[dict]:
    """The outline in API shape: no UI keys, empty subtopic lines dropped, blank-titled sections removed."""
    outline = []
    for s in sections:
        title = s["title"].strip()
        if not title:
            continue
        subtopics = [line.strip() for line in s["subtopics"] if line.strip()]
        outline.append({"id": s["id"], "title": title, "subtopics": subtopics})
    return outline


# ---------- API status ----------


@st.cache_data(ttl=30, show_spinner=False)
def api_provider() -> str | None:
    """The API's LLM provider, or None when the API is down.

    Streamlit re-runs this script on every click. Unchecked, a down API made every click wait 3 to 5 seconds on
    Windows (a refused connection is retried, and `localhost` tries IPv6 and then IPv4). So the answer is kept for
    30 seconds and the check gives up after 1 second.
    """
    try:
        response = httpx.get(f"{settings.api_base_url}/health", timeout=1)
        response.raise_for_status()
        return response.json()["llm_provider"]
    except (httpx.HTTPError, ValueError, KeyError):
        return None


# ---------- screens ----------


def sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand">🧭 ResearchPilot</div>', unsafe_allow_html=True)
        st.caption("Your AI research analyst")
        # Mock mode never calls the API, so it does not check it either
        if api_client.USE_MOCK:
            st.info("Mock mode (UI_USE_MOCK=1): fake data, no API needed")
        elif provider := api_provider():
            st.success(f"API running ({provider})")
        else:
            st.warning("API not reachable; start it: uv run uvicorn app.api.main:app --reload")
        if st.session_state.page != "topic":
            st.button("Start a new topic", on_click=reset)


EXAMPLE_TOPICS = [
    "State of electric two-wheelers in India",
    "Generative AI in healthcare",
    "Quick commerce in India",
    "Future of renewable energy storage",
]


def use_example(text: str) -> None:
    st.session_state["topic_input"] = text


def topic_screen() -> None:
    styles.hero("ResearchPilot", "Give it a topic. Get a cited report you can trust, steer and chat with.")
    topic = st.text_input(
        "What do you want to research?", key="topic_input", placeholder="e.g. State of electric two-wheelers in India"
    )
    st.caption("Or try an example:")
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLE_TOPICS):
        columns[index % 2].button(
            example, key=f"example-{index}", on_click=use_example, args=(example,), use_container_width=True
        )
    st.write("")
    if st.button("Start research", type="primary", disabled=not topic.strip()):
        with st.spinner("Planning the outline..."):
            try:
                result = api_client.start_research(topic.strip())
            except api_client.ApiError as exc:
                st.error(str(exc))
                return
        st.session_state.topic = topic.strip()
        st.session_state.run_id = result["run_id"]
        st.session_state.sections = to_editable(result["sections"])
        st.session_state.page = "outline"
        st.rerun()


def outline_screen() -> None:
    styles.page_header("Review the outline", "Edit, add or remove sections, then approve to start writing.")
    st.markdown(f'<span class="topic-pill">{html.escape(st.session_state.topic)}</span>', unsafe_allow_html=True)
    st.write("")

    sections = st.session_state.sections
    for number, section in enumerate(sections, start=1):
        key = section["key"]
        with st.container(border=True):
            st.markdown(
                f'<div class="sec-head"><span class="badge">{number}</span>Section {number}</div>',
                unsafe_allow_html=True,
            )
            title_col, remove_col = st.columns([6, 1], vertical_alignment="bottom")
            section["title"] = title_col.text_input(f"Section {number} title", section["title"], key=f"title-{key}")
            if remove_col.button("Remove", key=f"remove-{key}"):
                st.session_state.sections = [s for s in sections if s["key"] != key]
                st.rerun()
            text = st.text_area(
                "Subtopics (one per line)", "\n".join(section["subtopics"]), key=f"subs-{key}", height=100
            )
            section["subtopics"] = text.splitlines()

    if st.button("Add section"):
        sections.append({"id": None, "title": "", "subtopics": [], "key": uuid.uuid4().hex[:8]})
        st.rerun()

    outline = to_outline(sections)
    if not outline:
        st.warning("The outline needs at least one section with a title.")
    if st.button("Approve outline", type="primary", disabled=not outline):
        with st.spinner("Sending the approved outline..."):
            try:
                api_client.approve_outline(st.session_state.run_id, outline)
            except api_client.ApiError as exc:
                st.error(str(exc))
                return
        st.session_state.sections = to_editable(outline)
        st.session_state.page = "writing"
        st.rerun()


def writing_screen() -> None:
    styles.page_header("Writing your report", "Outline approved. The agent is now working through each section.")
    st.success("Outline approved")
    for number, section in enumerate(st.session_state.sections, start=1):
        st.markdown(
            '<div class="work-item"><span class="spin"></span>'
            f'<span class="badge">{number}</span>{html.escape(section["title"])}</div>',
            unsafe_allow_html=True,
        )
    st.info("The finished report appears here once the report view is added (T15).")


init_state()
sidebar()
styles.step_tracker(st.session_state.page)
{"topic": topic_screen, "outline": outline_screen, "writing": writing_screen}[st.session_state.page]()
