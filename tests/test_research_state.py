from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.research.state import (
    ResearchState,
    add_metrics,
    initial_state,
    merge_by_id,
    new_section,
    ordered_sections,
)
from app.llm.usage import track_usage


def test_merge_by_id_replaces_adds_and_removes():
    left = {"sec-1": "old", "sec-2": "keep", "sec-3": "drop"}

    merged = merge_by_id(left, {"sec-1": "new", "sec-3": None, "sec-4": "added"})

    assert merged == {"sec-1": "new", "sec-2": "keep", "sec-4": "added"}
    assert left["sec-1"] == "old"  # the reducer does not change its input


def test_add_metrics_sums_partial_counts():
    total = add_metrics({"llm_calls": 2, "search_calls": 1}, {"llm_calls": 1, "pages_fetched": 4})

    assert total["llm_calls"] == 3
    assert total["search_calls"] == 1
    assert total["pages_fetched"] == 4
    assert total["output_tokens"] == 0


def test_llm_usage_counts_fit_the_metrics_reducer():
    with track_usage() as usage:
        GenericFakeChatModel(
            messages=iter(
                [AIMessage(content="ok", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})]
            )
        ).invoke("hi")

    total = add_metrics({"llm_calls": 1, "search_calls": 2}, usage.as_metrics())

    assert total["llm_calls"] == 2
    assert total["search_calls"] == 2
    assert total["input_tokens"] == 10


def test_ordered_sections_follows_outline_order_and_skips_removed_ids():
    state = initial_state("topic")
    state["sections"] = {"sec-1": new_section("sec-1", "A"), "sec-2": new_section("sec-2", "B")}
    state["section_order"] = ["sec-2", "sec-gone", "sec-1"]

    assert [s["title"] for s in ordered_sections(state)] == ["B", "A"]


def _graph_with_parallel_section_writers():
    """Plan two sections, then update each in its own parallel branch, like the real research graph will."""

    def plan(state: ResearchState):
        return {
            "sections": {"sec-1": new_section("sec-1", "Market"), "sec-2": new_section("sec-2", "Policy")},
            "section_order": ["sec-1", "sec-2"],
            "sources": {"S1": {"id": "S1", "url": "https://example.org/a", "title": "A", "domain": "example.org"}},
        }

    def fan_out(state: ResearchState):
        return [Send("write", {"section": state["sections"][sid]}) for sid in state["section_order"]]

    def write(payload: dict):
        section = {**payload["section"], "draft": f"Draft of {payload['section']['title']} [S1]", "status": "drafted"}
        return {"sections": {section["id"]: section}, "metrics": {"llm_calls": 1}}

    builder = StateGraph(ResearchState)
    builder.add_node("plan", plan)
    builder.add_node("write", write)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out, ["write"])
    builder.add_edge("write", END)
    return builder


def test_parallel_section_updates_merge_and_survive_a_sqlite_checkpoint():
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = _graph_with_parallel_section_writers().compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "test-thread"}}

        graph.invoke(initial_state("EV two-wheelers in India"), config)
        saved = graph.get_state(config).values

    assert [s["status"] for s in ordered_sections(saved)] == ["drafted", "drafted"]
    assert saved["sections"]["sec-2"]["draft"] == "Draft of Policy [S1]"
    assert saved["sources"]["S1"]["domain"] == "example.org"
    assert saved["metrics"]["llm_calls"] == 2
