"""T11: outline approval. The graph pauses after planning and resumes with the approved or edited outline."""

import copy

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from app.agents.research import graph as graph_module
from app.agents.research.checkpoint import open_checkpointer
from app.agents.research.nodes import planner as planner_module
from app.agents.research.nodes import sources
from app.agents.research.nodes.approval import (
    MAX_SUBTOPICS,
    OutlineError,
    apply_approval,
    approve_outline,
    proposal,
)
from app.agents.research.nodes.planner import MAX_QUERIES, MAX_SECTIONS, Outline, SectionPlan
from app.agents.research.state import initial_state, new_section, ordered_sections
from app.tools.search import SearchBatch

TOPIC = "Electric scooters in India"

PLAN = Outline(
    sections=[
        SectionPlan(
            title="Sales and growth", subtopics=["scooter sales", "growth rate"], queries=["sales q1", "sales q2"]
        ),
        SectionPlan(title="Subsidy and policy", subtopics=["subsidy"], queries=["policy q1"]),
        SectionPlan(title="Outlook", subtopics=["forecast"], queries=["outlook q1"]),
    ]
)


def planned_state():
    """The state after planning: sec-1, sec-2, sec-3."""
    state = initial_state(TOPIC)
    state["sections"] = {
        f"sec-{n}": new_section(f"sec-{n}", p.title, p.subtopics, p.queries) for n, p in enumerate(PLAN.sections, 1)
    }
    state["section_order"] = ["sec-1", "sec-2", "sec-3"]
    state["sections_to_run"] = ["sec-1", "sec-2", "sec-3"]
    return state


def keep(sid, title, *subtopics):
    return {"id": sid, "title": title, "subtopics": list(subtopics)}


# ---- what the user is asked to review ----


def test_proposal_lists_the_outline_in_report_order():
    state = planned_state()
    state["section_order"] = ["sec-3", "sec-1", "sec-2"]

    shown = proposal(state)

    assert shown["topic"] == TOPIC
    assert [s["id"] for s in shown["sections"]] == ["sec-3", "sec-1", "sec-2"]
    assert shown["sections"][1] == {
        "id": "sec-1",
        "title": "Sales and growth",
        "subtopics": ["scooter sales", "growth rate"],
    }


# ---- applying the approved outline ----


@pytest.mark.parametrize("resume", [None, {}, {"sections": None}])
def test_approving_without_changes_keeps_the_outline(resume):
    state = planned_state()

    update = apply_approval(state, resume)

    assert update["outline_approved"] is True
    assert update["section_order"] == ["sec-1", "sec-2", "sec-3"]
    assert update["sections_to_run"] == ["sec-1", "sec-2", "sec-3"]
    assert update["sections"]["sec-1"] == state["sections"]["sec-1"]  # queries and all


def test_edited_subtopics_keep_the_title_and_add_a_query_for_the_new_subtopic():
    update = apply_approval(
        planned_state(), {"sections": [keep("sec-2", "Subsidy and policy", "subsidy", "import duties")]}
    )

    section = update["sections"]["sec-2"]
    assert section["subtopics"] == ["subsidy", "import duties"]
    assert section["queries"] == ["policy q1", f"{TOPIC} Subsidy and policy import duties"]


def test_a_new_subtopic_is_never_the_query_that_gets_cut():
    state = planned_state()
    state["sections"]["sec-1"]["queries"] = ["a", "b", "c"]  # already at the cap

    update = apply_approval(state, {"sections": [keep("sec-1", "Sales and growth", "scooter sales", "exports")]})

    queries = update["sections"]["sec-1"]["queries"]
    assert len(queries) == MAX_QUERIES
    assert queries[-1] == f"{TOPIC} Sales and growth exports"
    assert queries[:2] == ["a", "b"]


def test_a_renamed_section_gets_fresh_queries_and_keeps_its_id():
    update = apply_approval(planned_state(), {"sections": [keep("sec-3", "Future outlook", "forecast")]})

    section = update["sections"]["sec-3"]
    assert (section["id"], section["title"]) == ("sec-3", "Future outlook")
    assert section["queries"] == [f"{TOPIC} Future outlook forecast"]


def test_removed_sections_are_deleted_and_the_list_order_is_the_report_order():
    update = apply_approval(
        planned_state(),
        {"sections": [keep("sec-3", "Outlook", "forecast"), keep("sec-1", "Sales and growth", "scooter sales")]},
    )

    assert update["section_order"] == ["sec-3", "sec-1"]
    assert update["sections"]["sec-2"] is None  # merge_by_id removes it
    assert update["sections_to_run"] == ["sec-3", "sec-1"]


def test_new_sections_get_new_ids_and_queries_from_title_and_subtopics():
    update = apply_approval(
        planned_state(),
        {
            "sections": [
                keep("sec-1", "Sales and growth", "scooter sales", "growth rate"),
                {"id": None, "title": "  Battery   costs ", "subtopics": ["cell prices", "Cell prices", " "]},
                {"id": None, "title": "Charging", "subtopics": []},
            ]
        },
    )

    assert update["section_order"] == ["sec-1", "sec-4", "sec-5"]
    battery = update["sections"]["sec-4"]
    assert battery["title"] == "Battery costs" and battery["subtopics"] == ["cell prices"]
    assert battery["queries"] == [f"{TOPIC} Battery costs cell prices"]
    assert battery["status"] == "planned"
    assert update["sections"]["sec-5"]["queries"] == [f"{TOPIC} Charging"]  # no subtopics: search the title
    assert update["sections"]["sec-2"] is None and update["sections"]["sec-3"] is None


def test_sections_without_a_title_are_dropped():
    update = apply_approval(
        planned_state(),
        {"sections": [keep("sec-1", "Sales and growth"), {"id": None, "title": "  ", "subtopics": ["x"]}]},
    )

    assert update["section_order"] == ["sec-1"]


@pytest.mark.parametrize(
    ("resume", "message"),
    [
        ({"sections": []}, "at least one section"),
        ({"sections": [{"id": None, "title": " "}]}, "at least one section"),
        ({"sections": [keep("sec-9", "Ghost")]}, "Unknown section ID 'sec-9'"),
        ({"sections": [keep("sec-1", "A"), keep("sec-1", "B")]}, "appears twice"),
        ({"sections": [keep("sec-1", "Same"), keep("sec-2", "same ")]}, "Two sections are both called"),
        ({"sections": [{"id": None, "title": f"S{n}"} for n in range(MAX_SECTIONS + 1)]}, "most allowed"),
        ({"sections": "nope"}, "not valid .sections"),
        ({"sections": [{"id": "sec-1"}]}, "not valid .sections > 0 > title"),
        ({"sections": [{"id": None, "title": "T" * 121}]}, "not valid .sections > 0 > title"),
        ({"sections": [{"id": None, "title": "T", "subtopics": ["s"] * 11}]}, "subtopics"),
        ({"sections": [{"id": None, "title": "T", "subtopics": ["s" * 201]}]}, "subtopics"),
        ({"sections": [keep("sec-1", "References")]}, "cannot be a section"),
        ({"sections": [{"id": None, "title": " sources "}]}, "cannot be a section"),
        ([], "not valid"),
        ("", "not valid"),
        (0, "not valid"),
        (["sec-1"], "not valid"),
    ],
)
def test_invalid_outlines_are_rejected_with_a_message_for_the_user(resume, message):
    with pytest.raises(OutlineError, match=message):
        apply_approval(planned_state(), resume)


# ---- the graph: pause, resume, survive a restart ----


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(planner_module, "get_structured_llm", lambda schema: RunnableLambda(lambda _: PLAN))
    searched: list[list[str]] = []

    def search(queries, max_results):
        searched.append(list(queries))
        return SearchBatch(results=[], search_calls=1)

    monkeypatch.setattr(sources, "search_queries", search)
    monkeypatch.setattr(sources, "read_pages", lambda urls: {})
    return searched


CONFIG = {"configurable": {"thread_id": "run-1"}}


def test_graph_stops_after_planning_and_shows_the_outline(fake_pipeline):
    with open_checkpointer(":memory:") as saver:
        graph = graph_module.build_research_graph(saver)

        graph.invoke(initial_state(TOPIC), CONFIG)
        snapshot = graph.get_state(CONFIG)

    assert snapshot.next == ("approve_outline",)
    shown = snapshot.tasks[0].interrupts[0].value
    assert [s["title"] for s in shown["sections"]] == ["Sales and growth", "Subsidy and policy", "Outlook"]
    assert fake_pipeline == []  # nothing was searched before approval
    assert snapshot.values["outline_approved"] is False


def test_resume_with_an_edited_outline_researches_only_the_approved_sections(fake_pipeline):
    with open_checkpointer(":memory:") as saver:
        graph = graph_module.build_research_graph(saver)
        graph.invoke(initial_state(TOPIC), CONFIG)

        graph.invoke(
            Command(
                resume={
                    "sections": [
                        keep("sec-1", "Sales and growth", "scooter sales"),
                        {"id": None, "title": "Battery costs", "subtopics": ["cell prices"]},
                    ]
                }
            ),
            CONFIG,
        )
        final = graph.get_state(CONFIG).values

    assert final["outline_approved"] is True
    assert [s["title"] for s in ordered_sections(final)] == ["Sales and growth", "Battery costs"]
    assert set(final["sections"]) == {"sec-1", "sec-4"}  # sec-2 and sec-3 are gone from the state
    assert sorted(map(tuple, fake_pipeline)) == sorted(
        [("sales q1", "sales q2"), (f"{TOPIC} Battery costs cell prices",)]
    )


def test_invalid_resume_asks_again_with_the_reason_and_a_corrected_outline_then_works(fake_pipeline):
    with open_checkpointer(":memory:") as saver:
        graph = graph_module.build_research_graph(saver)
        graph.invoke(initial_state(TOPIC), CONFIG)

        graph.invoke(Command(resume={"sections": []}), CONFIG)  # no error is raised; the run just asks again

        snapshot = graph.get_state(CONFIG)
        assert snapshot.next == ("approve_outline",)
        asked = snapshot.tasks[0].interrupts[-1].value
        assert "at least one section" in asked["error"]
        assert [s["id"] for s in asked["sections"]] == ["sec-1", "sec-2", "sec-3"]  # the outline is shown again

        graph.invoke(Command(resume={"sections": [keep("sec-9", "Ghost")]}), CONFIG)  # wrong again
        assert "Unknown section ID" in graph.get_state(CONFIG).tasks[0].interrupts[-1].value["error"]

        graph.invoke(Command(resume={"sections": None}), CONFIG)  # a good answer finally

        final = graph.get_state(CONFIG)
        assert final.next == () and final.values["outline_approved"] is True
        assert len(ordered_sections(final.values)) == 3


def test_a_paused_run_resumes_after_the_process_restarts(fake_pipeline, tmp_path):
    db = tmp_path / "data" / "checkpoints.sqlite"  # the folder does not exist yet
    with open_checkpointer(db) as saver:
        graph_module.build_research_graph(saver).invoke(initial_state(TOPIC), CONFIG)

    # a new saver on the same file, as a restarted API would open
    with open_checkpointer(db) as saver:
        graph = graph_module.build_research_graph(saver)
        assert graph.get_state(CONFIG).next == ("approve_outline",)
        assert [s["title"] for s in proposal(graph.get_state(CONFIG).values)["sections"]][0] == "Sales and growth"

        graph.invoke(Command(resume={"sections": None}), CONFIG)

        assert graph.get_state(CONFIG).values["outline_approved"] is True


def test_approval_node_is_wired_to_the_function():
    assert graph_module.approve_outline is approve_outline


def test_an_empty_resume_value_does_not_resume_the_run(fake_pipeline):
    """LangGraph ignores Command(resume={}), so approving as proposed must send {"sections": None} (see approval.py)."""
    with open_checkpointer(":memory:") as saver:
        graph = graph_module.build_research_graph(saver)
        graph.invoke(initial_state(TOPIC), CONFIG)

        graph.invoke(Command(resume={}), CONFIG)

        assert graph.get_state(CONFIG).next == ("approve_outline",)  # still waiting


# ---- edge cases ----


def test_falsy_garbage_never_approves_by_accident():
    """Only None means "as proposed". An empty list or string used to slip through as an approval."""
    for garbage in ([], "", 0, False):
        with pytest.raises(OutlineError):
            apply_approval(planned_state(), garbage)
    assert apply_approval(planned_state(), None)["outline_approved"] is True


def test_a_blank_id_means_a_new_section():
    update = apply_approval(
        planned_state(),
        {
            "sections": [
                keep("sec-1", "Sales and growth"),
                {"id": "", "title": "Batteries"},
                {"id": "  ", "title": "Grid"},
            ]
        },
    )

    assert update["section_order"] == ["sec-1", "sec-4", "sec-5"]


def test_extra_fields_from_the_ui_are_ignored():
    update = apply_approval(
        planned_state(),
        {
            "sections": [
                {**keep("sec-1", "Sales and growth", "scooter sales", "growth rate"), "key": "ab12", "queries": ["x"]}
            ]
        },
    )

    assert update["sections"]["sec-1"]["queries"] == ["sales q1", "sales q2"]  # the planner's, not the UI's


def test_limits_are_inclusive():
    full = [{"id": None, "title": f"S{n}", "subtopics": ["s"] * MAX_SUBTOPICS} for n in range(MAX_SECTIONS)]
    full[0]["title"] = "T" * 120

    update = apply_approval(planned_state(), {"sections": full})

    assert len(update["section_order"]) == MAX_SECTIONS
    assert len(update["sections"]["sec-4"]["subtopics"]) == 1  # ten identical subtopics collapse to one


def test_whitespace_only_changes_keep_the_planner_queries():
    state = planned_state()
    state["sections"]["sec-1"]["title"] = "Sales  and growth"  # as stored, with a double space

    update = apply_approval(state, {"sections": [keep("sec-1", " Sales and growth ", "Scooter Sales", "growth rate")]})

    assert update["sections"]["sec-1"]["queries"] == ["sales q1", "sales q2"]
    assert update["sections"]["sec-1"]["subtopics"] == ["Scooter Sales", "growth rate"]  # tidied, text kept


def test_removing_a_subtopic_keeps_the_queries():
    update = apply_approval(planned_state(), {"sections": [keep("sec-1", "Sales and growth", "scooter sales")]})

    assert update["sections"]["sec-1"]["subtopics"] == ["scooter sales"]
    assert update["sections"]["sec-1"]["queries"] == ["sales q1", "sales q2"]


def test_a_removed_sections_id_is_not_handed_to_a_new_section():
    update = apply_approval(
        planned_state(),
        {"sections": [keep("sec-1", "Sales and growth"), {"id": None, "title": "Batteries"}]},  # sec-2, sec-3 removed
    )

    assert update["section_order"] == ["sec-1", "sec-4"]


def test_the_state_is_never_changed_and_the_update_shares_no_lists_with_it():
    state = planned_state()
    before = copy.deepcopy(state)

    update = apply_approval(state, {"sections": [keep("sec-1", "Sales and growth", "scooter sales", "exports")]})

    assert state == before
    new = update["sections"]["sec-1"]
    assert new["queries"] is not state["sections"]["sec-1"]["queries"]
    assert new["passages"] is not state["sections"]["sec-1"]["passages"]
    update["section_order"].append("x")
    update["sections_to_run"].append("y")
    assert update["sections_to_run"] == ["sec-1", "y"]


def test_the_outline_shown_to_the_user_includes_the_section_limit_and_ignores_ghost_ids():
    state = planned_state()
    state["section_order"].append("sec-99")  # in the order but not in the sections

    shown = proposal(state)

    assert shown["max_sections"] == MAX_SECTIONS
    assert [s["id"] for s in shown["sections"]] == ["sec-1", "sec-2", "sec-3"]


def test_error_messages_name_the_field_and_do_not_leak_the_input():
    with pytest.raises(OutlineError) as caught:
        apply_approval(planned_state(), {"sections": [{"id": None, "title": "secret " * 40}]})

    assert "title" in str(caught.value) and "secret" not in str(caught.value)


def test_a_restart_in_the_middle_of_a_correction_round_loses_nothing(fake_pipeline, tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    with open_checkpointer(db) as saver:
        graph = graph_module.build_research_graph(saver)
        graph.invoke(initial_state(TOPIC), CONFIG)
        graph.invoke(Command(resume={"sections": []}), CONFIG)  # a bad answer; the run asks again

    with open_checkpointer(db) as saver:  # the API restarted
        graph = graph_module.build_research_graph(saver)
        asked = graph.get_state(CONFIG).tasks[0].interrupts[-1].value
        assert "at least one section" in asked["error"]

        graph.invoke(Command(resume={"sections": [keep("sec-2", "Subsidy and policy", "subsidy")]}), CONFIG)

        final = graph.get_state(CONFIG)
        assert final.next == () and final.values["section_order"] == ["sec-2"]


def test_two_runs_on_one_saver_do_not_mix(fake_pipeline):
    one, two = {"configurable": {"thread_id": "run-1"}}, {"configurable": {"thread_id": "run-2"}}
    with open_checkpointer(":memory:") as saver:
        graph = graph_module.build_research_graph(saver)
        graph.invoke(initial_state("Topic one"), one)
        graph.invoke(initial_state("Topic two"), two)

        graph.invoke(Command(resume={"sections": [keep("sec-1", "Only this", "x")]}), one)

        assert graph.get_state(one).values["section_order"] == ["sec-1"]
        assert graph.get_state(two).next == ("approve_outline",)  # run two is still waiting, untouched
        assert len(graph.get_state(two).values["section_order"]) == 3
        assert graph.get_state(two).values["topic"] == "Topic two"
