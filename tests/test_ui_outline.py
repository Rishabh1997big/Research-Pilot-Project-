"""T10: the topic and outline screens, driven with Streamlit's AppTest and the mock API (no network, no keys)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def mock_api(monkeypatch):
    monkeypatch.setenv("UI_USE_MOCK", "1")


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def start(topic: str = "Quick commerce in India") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.text_input[0].set_value(topic).run()
    button(at, "Start research").click().run()
    return at


def test_start_button_needs_a_topic():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert button(at, "Start research").disabled


def test_topic_leads_to_outline_with_planned_sections():
    at = start()
    assert not at.exception
    assert at.session_state["page"] == "outline"
    assert at.session_state["topic"] == "Quick commerce in India"
    assert len([t for t in at.text_input if t.label.startswith("Section")]) == 4


def test_remove_section_keeps_the_others_values():
    at = start()
    first_key = at.session_state["sections"][0]["key"]
    button(at, "Remove").click().run()
    titles = [t.value for t in at.text_input if t.label.startswith("Section")]
    assert titles == ["Key players", "Government policy", "Challenges and outlook"]
    assert first_key not in {s["key"] for s in at.session_state["sections"]}


def test_added_section_with_blank_title_is_not_approved():
    at = start()
    button(at, "Add section").click().run()
    assert len([t for t in at.text_input if t.label.startswith("Section")]) == 5
    button(at, "Approve outline").click().run()
    assert at.session_state["page"] == "writing"
    assert len(at.session_state["sections"]) == 4  # the blank one was dropped


def test_edits_are_kept_when_approving():
    at = start()
    at.text_input(key=f"title-{at.session_state['sections'][0]['key']}").set_value("Market size")
    at.text_area(key=f"subs-{at.session_state['sections'][0]['key']}").set_value("sales\n\n growth ")
    button(at, "Approve outline").click().run()
    first = at.session_state["sections"][0]
    assert first["title"] == "Market size"
    assert first["subtopics"] == ["sales", "growth"]


def test_new_topic_button_resets():
    at = start()
    button(at, "Start a new topic").click().run()
    assert at.session_state["page"] == "topic"
    assert os.environ["UI_USE_MOCK"] == "1"
