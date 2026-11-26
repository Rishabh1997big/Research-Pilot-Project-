"""T11: nasty inputs for outline approval. A second line of defence beside test_approval.py.

Whatever the user (or a buggy UI) sends, apply_approval must either return a valid update or raise OutlineError with a
message. It must never raise anything else, never change the state, and never produce a state the later nodes cannot
use.
"""

import copy
import random
import string

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from app.agents.research import graph as graph_module
from app.agents.research.checkpoint import open_checkpointer
from app.agents.research.nodes import planner as planner_module
from app.agents.research.nodes import sources
from app.agents.research.nodes.approval import OutlineError, apply_approval
from app.agents.research.nodes.planner import MAX_QUERIES, MAX_SECTIONS, Outline, SectionPlan
from app.agents.research.state import initial_state, new_section
from app.tools.search import SearchBatch

TOPIC = "Electric scooters in India"


def planned_state(n: int = 3):
    state = initial_state(TOPIC)
    for i in range(1, n + 1):
        state["sections"][f"sec-{i}"] = new_section(
            f"sec-{i}", f"Title {i}", [f"sub {i}a", f"sub {i}b"], [f"q{i}a", f"q{i}b"]
        )
    state["section_order"] = [f"sec-{i}" for i in range(1, n + 1)]
    state["sections_to_run"] = list(state["section_order"])
    return state


def sec(sid, title, *subs):
    return {"id": sid, "title": title, "subtopics": list(subs)}


# ---- strange but valid text ----


@pytest.mark.parametrize(
    "title",
    [
        "Sales & growth (2024–25): what's next?",
        "बाज़ार का आकार",  # Hindi
        "电动两轮车市场",  # Chinese
        "EVs 🛵⚡ in India",
        "a",  # one character
        "<script>alert(1)</script>",
        "Robert'); DROP TABLE sections;--",
        "Line one\nLine two\ttabbed",
        "  \u00a0 non-breaking\u00a0space  ",
    ],
)
def test_unusual_text_is_accepted_and_tidied(title):
    update = apply_approval(planned_state(), {"sections": [sec(None, title, "x")]})

    created = update["sections"]["sec-4"]
    assert created["title"] == " ".join(title.split())
    assert created["queries"] and all(q.strip() for q in created["queries"])
    assert "  " not in created["title"] and "\n" not in created["title"]


def test_punctuation_only_title_is_kept_but_still_gets_a_query():
    update = apply_approval(planned_state(), {"sections": [sec(None, "???", "x")]})

    assert update["sections"]["sec-4"]["queries"] == [f"{TOPIC} ??? x"]


# ---- wrong types and shapes ----


@pytest.mark.parametrize(
    "resume",
    [
        {"sections": [None]},
        {"sections": [123]},
        {"sections": ["sec-1"]},
        {"sections": [[]]},
        {"sections": {"sec-1": "x"}},
        {"sections": [{"id": 5, "title": "x"}]},
        {"sections": [{"id": ["sec-1"], "title": "x"}]},
        {"sections": [{"id": None, "title": None}]},
        {"sections": [{"id": None, "title": 7}]},
        {"sections": [{"id": None, "title": "x", "subtopics": "not a list"}]},
        {"sections": [{"id": None, "title": "x", "subtopics": [1, 2]}]},
        {"sections": [{"id": None, "title": "x", "subtopics": None}]},
        {"sections": [{"id": None, "title": {"a": 1}}]},
        {"sections": 5},
        {"sections": True},
        True,
        3.14,
        object(),
        b"bytes",
        [{"id": None, "title": "x"}],  # a bare list instead of {"sections": [...]}
        float("nan"),
    ],
)
def test_wrong_types_only_ever_give_an_outline_error(resume):
    state = planned_state()
    before = copy.deepcopy(state)

    with pytest.raises(OutlineError) as caught:
        apply_approval(state, resume)

    assert str(caught.value)  # there is a message to show
    assert state == before


def test_a_json_string_is_not_an_outline():
    with pytest.raises(OutlineError):
        apply_approval(planned_state(), '{"sections": []}')


# ---- ids ----


@pytest.mark.parametrize("bad_id", ["SEC-1", "sec-01", "sec 1", "S1", "sec-", "../etc/passwd", "sec-1; drop"])
def test_ids_must_match_existing_sections_exactly(bad_id):
    with pytest.raises(OutlineError, match="Unknown section ID"):
        apply_approval(planned_state(), {"sections": [sec(bad_id, "x")]})


def test_ids_are_trimmed_so_a_padded_id_still_matches():
    update = apply_approval(planned_state(), {"sections": [sec(" sec-2 ", "Title 2", "sub 2a", "sub 2b")]})

    assert update["section_order"] == ["sec-2"]


def test_new_ids_continue_after_the_highest_even_with_gaps():
    state = planned_state(3)
    del state["sections"]["sec-2"]  # a gap, as after an earlier removal
    state["section_order"].remove("sec-2")

    update = apply_approval(state, {"sections": [sec("sec-1", "Title 1"), sec(None, "A"), sec(None, "B")]})

    assert update["section_order"] == ["sec-1", "sec-4", "sec-5"]


def test_orphan_sections_not_in_the_order_are_cleaned_up():
    state = planned_state(2)
    state["sections"]["sec-9"] = new_section("sec-9", "Orphan")  # in sections, but not in section_order

    update = apply_approval(state, {"sections": None})

    assert update["sections"]["sec-9"] is None
    assert update["section_order"] == ["sec-1", "sec-2"]


# ---- reordering and re-approval ----


def test_reversing_the_order_changes_only_the_order():
    state = planned_state(3)
    reverse = [sec(f"sec-{i}", f"Title {i}", f"sub {i}a", f"sub {i}b") for i in (3, 2, 1)]

    update = apply_approval(state, {"sections": reverse})

    assert update["section_order"] == ["sec-3", "sec-2", "sec-1"]
    for sid in ("sec-1", "sec-2", "sec-3"):
        assert update["sections"][sid] == state["sections"][sid]


def test_a_section_that_already_has_research_keeps_it():
    state = planned_state(2)
    state["sections"]["sec-1"].update(
        passages=[{"source_id": "S1", "text": "x"}], source_ids=["S1"], status="researching", revision=2
    )

    update = apply_approval(state, {"sections": [sec("sec-1", "Title 1", "sub 1a", "sub 1b", "new")]})

    kept = update["sections"]["sec-1"]
    assert kept["passages"] == [{"source_id": "S1", "text": "x"}] and kept["source_ids"] == ["S1"]
    assert (kept["status"], kept["revision"]) == ("researching", 2)


def test_approving_twice_gives_the_same_result():
    state = planned_state(3)
    first = apply_approval(state, {"sections": None})
    second = apply_approval(
        {**state, "sections": {k: v for k, v in first["sections"].items() if v}}, {"sections": None}
    )

    assert first == second


def test_a_single_section_outline_is_enough():
    update = apply_approval(planned_state(), {"sections": [sec("sec-3", "Title 3")]})

    assert update["section_order"] == ["sec-3"] and update["sections"]["sec-1"] is None


def test_an_empty_state_with_no_sections_cannot_be_approved_as_proposed():
    with pytest.raises(OutlineError, match="at least one section"):
        apply_approval(initial_state(TOPIC), None)


# ---- queries ----


def test_a_section_with_no_subtopics_searches_its_title():
    update = apply_approval(planned_state(), {"sections": [sec(None, "Charging")]})

    assert update["sections"]["sec-4"]["queries"] == [f"{TOPIC} Charging"]


def test_adding_many_subtopics_never_exceeds_the_query_cap():
    subs = [f"extra {n}" for n in range(8)]
    update = apply_approval(planned_state(), {"sections": [sec("sec-1", "Title 1", "sub 1a", "sub 1b", *subs)]})

    queries = update["sections"]["sec-1"]["queries"]
    assert len(queries) == MAX_QUERIES and len(set(q.lower() for q in queries)) == MAX_QUERIES
    assert queries[-1] == f"{TOPIC} Title 1 extra 2"  # new ones first; the planner's older queries make way


def test_subtopic_text_differing_only_in_case_is_not_an_addition():
    update = apply_approval(planned_state(), {"sections": [sec("sec-1", "Title 1", "SUB 1A", "Sub 1b")]})

    assert update["sections"]["sec-1"]["queries"] == ["q1a", "q1b"]


# ---- fuzz: random outlines, invariants always hold ----

ALPHABET = string.ascii_letters + string.digits + " \t\n-_&!?🛵é"


def random_text(rng, limit):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, limit)))


def random_entry(rng, ids):
    pick = rng.random()
    entry_id = rng.choice([*ids, None, None, "", "sec-99", "nope"]) if pick < 0.9 else rng.choice([5, [], {}])
    return {
        "id": entry_id,
        "title": random_text(rng, 130) if rng.random() < 0.95 else rng.choice([None, 1, "References", "  "]),
        "subtopics": [random_text(rng, 210) for _ in range(rng.randint(0, 12))],
    }


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_invariants(seed):
    rng = random.Random(seed)
    state = planned_state(rng.randint(1, 5))
    before = copy.deepcopy(state)
    ids = list(state["sections"])
    resume = {"sections": [random_entry(rng, ids) for _ in range(rng.randint(0, 9))]}

    try:
        update = apply_approval(state, resume)
    except OutlineError as exc:
        assert str(exc)
        assert state == before
        return

    assert state == before  # never changed
    order = update["section_order"]
    assert 1 <= len(order) <= MAX_SECTIONS
    assert len(set(order)) == len(order)  # no section twice
    assert update["sections_to_run"] == order and update["outline_approved"] is True
    kept = {sid: s for sid, s in update["sections"].items() if s is not None}
    assert set(kept) == set(order)  # exactly the ordered sections are left in the state
    assert len({s["title"].lower() for s in kept.values()}) == len(kept)  # titles are unique
    for sid, section in kept.items():
        assert section["id"] == sid and section["title"] and section["title"] == " ".join(section["title"].split())
        assert 1 <= len(section["queries"]) <= MAX_QUERIES
        assert all(q.strip() for q in section["queries"]) and len(set(section["queries"])) == len(section["queries"])
        assert len({s.lower() for s in section["subtopics"]}) == len(section["subtopics"])
    new_ids = set(order) - set(ids)
    assert all(int(n.split("-")[1]) > len(ids) for n in new_ids)  # new IDs are never an id that was in use
    for sid in set(ids) - set(order):
        assert update["sections"][sid] is None  # a removed section is removed, not left behind


# ---- the checkpointer helper ----


def test_checkpointer_accepts_a_string_path_and_reuses_an_existing_file(tmp_path):
    db = str(tmp_path / "a" / "b" / "c.sqlite")
    for _ in range(2):  # the second open meets an existing file
        with open_checkpointer(db) as saver:
            assert saver.get_tuple({"configurable": {"thread_id": "none"}}) is None


def test_default_checkpointer_path_comes_from_settings(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "checkpoint_db", tmp_path / "deep" / "cp.sqlite")

    with open_checkpointer():
        pass

    assert (tmp_path / "deep" / "cp.sqlite").exists()


def test_building_the_graph_without_a_checkpointer_is_refused():
    with pytest.raises(TypeError):
        graph_module.build_research_graph()  # type: ignore[call-arg]


# ---- the whole graph with hostile answers in a row ----


def test_many_bad_answers_in_a_row_then_a_good_one():
    plan = Outline(sections=[SectionPlan(title=f"S{n}", subtopics=["a"], queries=["q"]) for n in range(3)])
    import pytest as _p

    mp = _p.MonkeyPatch()
    mp.setattr(planner_module, "get_structured_llm", lambda schema: RunnableLambda(lambda _: plan))
    mp.setattr(sources, "search_queries", lambda queries, max_results: SearchBatch(results=[], search_calls=1))
    mp.setattr(sources, "read_pages", lambda urls: {})
    config = {"configurable": {"thread_id": "hostile"}}
    bad = [
        {"sections": []},
        {"sections": [sec("zzz", "x")]},
        {"sections": "x"},
        {"sections": [sec(None, "References")]},
    ] * 3
    try:
        with open_checkpointer(":memory:") as saver:
            graph = graph_module.build_research_graph(saver)
            graph.invoke(initial_state(TOPIC), config)
            for answer in bad:
                graph.invoke(Command(resume=answer), config)
                snapshot = graph.get_state(config)
                assert snapshot.next == ("approve_outline",)
                assert snapshot.tasks[0].interrupts[-1].value["error"]
                assert len(snapshot.tasks[0].interrupts[-1].value["sections"]) == 3  # still the original outline

            graph.invoke(Command(resume={"sections": [sec("sec-2", "S1", "a")]}), config)

            final = graph.get_state(config)
            assert final.next == () and final.values["section_order"] == ["sec-2"]
    finally:
        mp.undo()
