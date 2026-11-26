import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agents.research.nodes import planner as planner_module
from app.agents.research.nodes.planner import (
    MAX_QUERIES,
    MAX_SECTIONS,
    Outline,
    PlannerError,
    SectionPlan,
    clean_outline,
    outline_to_state,
    plan_outline,
    planner,
)
from app.agents.research.state import ResearchState, initial_state, new_section, ordered_sections


def section(title: str, subtopics=("a", "b"), queries=("q1", "q2")) -> SectionPlan:
    return SectionPlan(title=title, subtopics=list(subtopics), queries=list(queries))


def outline(*titles: str) -> Outline:
    return Outline(sections=[section(t) for t in titles])


def fake_llm(*replies: Outline):
    """A structured LLM that returns the given outlines in turn and records the messages it was sent."""
    calls: list = []
    remaining = list(replies)

    def reply(messages):
        calls.append(messages)
        return remaining.pop(0)

    return RunnableLambda(reply), calls


# ---- cleaning ----


def test_clean_outline_drops_blank_repeated_and_reference_sections():
    raw = Outline(
        sections=[
            section("  Market   size "),
            section(""),
            section("market size"),
            section("References"),
            section("Key players"),
        ]
    )

    cleaned = clean_outline(raw, "EVs")

    assert [s.title for s in cleaned.sections] == ["Market size", "Key players"]


def test_clean_outline_caps_sections_queries_and_subtopics():
    raw = Outline(sections=[section(f"S{n}", subtopics="abcdef", queries="12345") for n in range(10)])

    cleaned = clean_outline(raw, "EVs")

    assert len(cleaned.sections) == MAX_SECTIONS
    assert all(len(s.queries) == MAX_QUERIES for s in cleaned.sections)
    assert all(len(s.subtopics) == 4 for s in cleaned.sections)


def test_clean_outline_removes_repeated_queries_and_adds_one_when_missing():
    raw = Outline(sections=[section("Policy", queries=["EV policy", " ev  POLICY ", ""]), section("Risks", queries=[])])

    cleaned = clean_outline(raw, "EVs in India")

    assert cleaned.sections[0].queries == ["EV policy"]
    assert cleaned.sections[1].queries == ["EVs in India Risks"]


def test_clean_outline_turns_typographic_hyphens_and_spaces_into_plain_ones():
    raw = Outline(sections=[section("Two‑wheeler sales", queries=["electric two‑wheeler sales New Delhi"])])

    cleaned = clean_outline(raw, "EVs")

    assert cleaned.sections[0].title == "Two-wheeler sales"
    assert cleaned.sections[0].queries == ["electric two-wheeler sales New Delhi"]


def test_years_are_removed_from_queries_unless_the_topic_names_one():
    raw = Outline(sections=[section("Sales", queries=["EV sales India 2024", "EV price trend 2022-2025 India"])])

    assert clean_outline(raw, "EVs in India").sections[0].queries == ["EV sales India", "EV price trend India"]
    assert clean_outline(raw, "EV sales in 2024").sections[0].queries[0] == "EV sales India 2024"


# ---- planning ----


def test_plan_outline_sends_the_topic_and_the_rules():
    llm, calls = fake_llm(outline("A", "B", "C"))

    result = plan_outline("  EVs in India ", llm=llm)

    assert [s.title for s in result.sections] == ["A", "B", "C"]
    system, human = calls[0]
    assert "References" in system.content and "Today's date is" in system.content
    assert human.content == "Topic: EVs in India"


def test_too_few_sections_asks_again_once():
    llm, calls = fake_llm(outline("A", "References"), outline("A", "B", "C"))

    result = plan_outline("EVs", llm=llm)

    assert len(calls) == 2
    assert len(result.sections) == 3


def test_too_few_sections_twice_gives_a_clear_error():
    llm, _ = fake_llm(outline("A"), outline("A", "B"))

    with pytest.raises(PlannerError, match="fewer than 3"):
        plan_outline("EVs", llm=llm)


def test_empty_topic_is_rejected_without_an_llm_call():
    llm, calls = fake_llm()

    with pytest.raises(PlannerError, match="empty"):
        plan_outline("   ", llm=llm)
    assert calls == []


# ---- state update ----


def test_outline_becomes_planned_sections_in_order():
    update = outline_to_state(outline("Market", "Policy", "Outlook"), initial_state("EVs"))

    assert update["section_order"] == ["sec-1", "sec-2", "sec-3"]
    assert update["sections_to_run"] == ["sec-1", "sec-2", "sec-3"]
    assert update["outline_approved"] is False
    first = update["sections"]["sec-1"]
    assert (first["title"], first["status"], first["queries"]) == ("Market", "planned", ["q1", "q2"])


def test_re_planning_removes_old_sections_and_never_reuses_their_ids():
    state = initial_state("EVs")
    state["sections"] = {"sec-1": new_section("sec-1", "Old A"), "sec-2": new_section("sec-2", "Old B")}
    state["section_order"] = ["sec-1", "sec-2"]

    update = outline_to_state(outline("New A", "New B", "New C"), state)

    assert update["sections"]["sec-1"] is None and update["sections"]["sec-2"] is None
    assert update["section_order"] == ["sec-3", "sec-4", "sec-5"]


def test_planner_node_in_a_real_graph_with_checkpointer(monkeypatch):
    llm, _ = fake_llm(outline("Market", "Policy", "Outlook"))
    monkeypatch.setattr(planner_module, "get_structured_llm", lambda schema: llm)

    builder = StateGraph(ResearchState)
    builder.add_node("plan", planner)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", END)
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "t"}}
        graph.invoke(initial_state("EVs in India"), config)
        saved = graph.get_state(config).values

    assert [s["title"] for s in ordered_sections(saved)] == ["Market", "Policy", "Outlook"]
    assert saved["outline_approved"] is False
    assert saved["metrics"]["llm_calls"] == 0  # the fake is not a chat model; real calls are counted by track_usage
