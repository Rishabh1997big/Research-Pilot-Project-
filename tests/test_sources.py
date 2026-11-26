import random

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.agents.research import graph as graph_module
from app.agents.research.nodes import planner as planner_module
from app.agents.research.nodes import sources
from app.agents.research.nodes.planner import Outline, SectionPlan
from app.agents.research.nodes.sources import (
    MAX_PASSAGES_PER_SECTION,
    Registry,
    best_passages,
    choose_candidates,
    read_node,
    search_node,
    split_passages,
)
from app.agents.research.state import cited_source_ids, initial_state, new_section, ordered_sections
from app.tools.search import SearchBatch


def result(url: str, query: str = "q1", title: str = "") -> dict:
    return {"url": url, "title": title or f"Title of {url}", "snippet": "", "query": query, "provider": "fake"}


def page(url: str, text: str, title: str = "") -> dict:
    return {"url": url, "title": title, "text": text}


SALES = "Electric scooter sales in India rose to 1.2 million units, with strong growth in every state. " * 3
POLICY = (
    "The government subsidy for electric scooters was reduced, and the new policy targets local battery making. " * 3
)
OTHER = "The weather in the capital was mild, and the cricket season began with a large crowd at the stadium. " * 3


FILLER = "market city buyer price model range battery dealer report survey data trend region brand factory".split()


def own_text(url: str, text: str) -> str:
    """`text` with words of its own in every line (picked by the URL), like a real article: the registry must not
    treat two fake pages as copies of each other."""
    rng = random.Random(url)
    return "\n".join(f"{line} {' '.join(rng.choices(FILLER, k=12))}." for line in text.splitlines())


def section(section_id: str, title: str, queries=("q1",), subtopics=()) -> dict:
    return new_section(section_id, title, list(subtopics), list(queries))


# ---- choosing candidate pages ----


def test_candidates_take_each_querys_results_in_turn():
    results = [result(f"https://a{n}.org", "q1") for n in range(3)] + [
        result(f"https://b{n}.org", "q2") for n in range(3)
    ]

    chosen = [hit["url"] for hit in choose_candidates(results)]

    assert chosen[:4] == ["https://a0.org", "https://b0.org", "https://a1.org", "https://b1.org"]


def test_candidates_limit_each_site_and_skip_pdfs():
    results = [
        result("https://big.org/1"),
        result("https://www.big.org/2"),
        result("https://big.org/3"),
        result("https://c.org/report.pdf"),
        result("https://d.org/x"),
    ]

    assert [hit["url"] for hit in choose_candidates(results)] == [
        "https://big.org/1",
        "https://www.big.org/2",
        "https://d.org/x",
    ]


def test_candidates_skip_social_media_and_the_search_providers_own_pages():
    results = [
        result("https://www.linkedin.com/posts/x"),
        result("https://m.facebook.com/videos/y"),
        result("https://tavily.com/knowledge/company/z"),
        result("https://news.org/a"),
    ]

    assert [hit["url"] for hit in choose_candidates(results)] == ["https://news.org/a"]


def test_candidates_are_capped():
    results = [result(f"https://s{n}.org") for n in range(20)]

    assert len(choose_candidates(results)) == sources.CANDIDATES_PER_SECTION


# ---- passages ----


def test_split_passages_joins_paragraphs_up_to_the_limit():
    text = "\n".join(["A" * 400, "B" * 400, "C" * 400])

    passages = split_passages(text, max_chars=1000)

    assert [len(p) for p in passages] == [801, 400]


def test_split_passages_splits_long_paragraphs_at_sentence_ends():
    text = " ".join(["This is one sentence about electric scooters."] * 60)

    passages = split_passages(text, max_chars=500)

    assert all(len(p) <= 500 for p in passages)
    assert all(p.endswith(".") for p in passages)


def test_best_passages_picks_the_relevant_ones():
    # Each block is long enough (about 800 characters) to become a passage of its own.
    other, sales, policy = (" ".join([block.strip()] * 3) for block in (OTHER, SALES, POLICY))

    best = best_passages("\n".join([other, sales, policy]), ["scooter", "sales", "growth"], limit=2)

    assert best[0] == sales
    assert other not in best


def test_best_passages_never_picks_text_without_query_terms():
    assert best_passages(OTHER, ["scooter", "subsidy"], limit=3) == []


# ---- registry ----


def test_registry_gives_new_pages_the_next_ids_and_keeps_existing_ones():
    existing = {"S1": {"id": "S1", "url": "https://old.org/a", "title": "Old", "domain": "old.org"}}
    registry = Registry(existing)

    first = registry.id_for({"url": "https://new.org/x", "title": "New"}, page("https://new.org/x", ""))
    again = registry.id_for({"url": "http://www.new.org/x/", "title": ""}, page("", ""))  # same page, other spelling
    old = registry.id_for({"url": "https://old.org/a?utm_source=t", "title": ""}, page("", ""))

    assert (first, again, old) == ("S2", "S2", "S1")
    assert registry.new_sources == {"S2": {"id": "S2", "url": "https://new.org/x", "title": "New", "domain": "new.org"}}


def test_registry_gives_a_copy_of_an_article_on_another_site_the_same_id():
    registry = Registry({})
    article = SALES * 4

    original = registry.id_for({"url": "https://finance.yahoo.com/a", "title": "A"}, page("", article))
    copy = registry.id_for({"url": "https://uk.finance.yahoo.com/a", "title": "A"}, page("", article))
    different = registry.id_for({"url": "https://b.org/x", "title": "B"}, page("", POLICY * 4))

    assert (original, copy, different) == ("S1", "S1", "S2")
    assert list(registry.new_sources) == ["S1", "S2"]


def test_registry_matches_copies_that_differ_slightly():
    registry = Registry({})
    article = own_text("https://finance.yahoo.com/a", "\n".join([SALES] * 4))
    us = "Contact the distributor with any inquiries.\n" + article
    uk = "Contact the distributor with any enquiries.\n" + article

    assert registry.id_for({"url": "https://finance.yahoo.com/a", "title": ""}, page("", us)) == "S1"
    assert registry.id_for({"url": "https://uk.finance.yahoo.com/a", "title": ""}, page("", uk)) == "S1"


def test_paid_press_releases_are_not_used(monkeypatch):
    state = two_section_state()
    state["search_results"] = {
        "sec-1": [{"url": "https://ads.org/pr", "title": "PR"}, {"url": "https://news.org/a", "title": "News"}]
    }
    texts = {"https://ads.org/pr": "This is a paid press release.\n" + SALES * 3, "https://news.org/a": SALES * 3}
    monkeypatch.setattr(sources, "read_pages", lambda urls: {u: page(u, texts[u]) for u in urls})

    update = read_node(state)

    assert [s["url"] for s in update["sources"].values()] == ["https://news.org/a"]


def test_read_node_does_not_list_a_copied_article_twice_in_a_section(monkeypatch):
    state = two_section_state()
    state["search_results"] = {
        "sec-1": [
            {"url": "https://finance.yahoo.com/a", "title": "A"},
            {"url": "https://uk.finance.yahoo.com/a", "title": "A"},
        ]
    }
    text = "\n".join([SALES * 3, POLICY * 3])
    monkeypatch.setattr(sources, "read_pages", lambda urls: {u: page(u, text) for u in urls})

    update = read_node(state)

    assert update["sections"]["sec-1"]["source_ids"] == ["S1"]
    assert list(update["sources"]) == ["S1"]


# ---- nodes ----


def two_section_state() -> dict:
    state = initial_state("Electric scooters in India")
    state["sections"] = {
        "sec-1": section("sec-1", "Sales and growth", subtopics=["scooter sales growth"]),
        "sec-2": section("sec-2", "Subsidy and policy", subtopics=["government subsidy policy"]),
    }
    state["section_order"] = ["sec-1", "sec-2"]
    state["sections_to_run"] = ["sec-1", "sec-2"]
    return state


def test_search_node_searches_each_section_and_keeps_candidates(monkeypatch):
    found = {
        "sec-1": SearchBatch(results=[result("https://a.org/sales")], search_calls=1),
        "sec-2": SearchBatch(results=[result("https://b.org/policy")], search_calls=2),
    }
    state = two_section_state()
    by_queries = {tuple(state["sections"][sid]["queries"]): sid for sid in found}
    state["sections"]["sec-2"]["queries"] = ["q2"]
    by_queries = {("q1",): "sec-1", ("q2",): "sec-2"}
    monkeypatch.setattr(sources, "search_queries", lambda queries, max_results: found[by_queries[tuple(queries)]])

    update = search_node(state)

    assert update["search_results"] == {
        "sec-1": [{"url": "https://a.org/sales", "title": "Title of https://a.org/sales"}],
        "sec-2": [{"url": "https://b.org/policy", "title": "Title of https://b.org/policy"}],
    }
    assert update["metrics"] == {"search_calls": 3}
    assert {s["status"] for s in update["sections"].values()} == {"researching"}


def test_read_node_registers_read_pages_and_attaches_passages_with_their_ids(monkeypatch):
    state = two_section_state()
    state["search_results"] = {
        "sec-1": [{"url": "https://blocked.org/x", "title": "Blocked"}, {"url": "https://a.org/sales", "title": "A"}],
        "sec-2": [{"url": "https://b.org/policy", "title": "B"}, {"url": "https://www.a.org/sales/", "title": "A"}],
    }
    pages = {
        "https://blocked.org/x": None,
        "https://a.org/sales": page("https://a.org/sales", f"{SALES}\n{POLICY}"),
        "https://b.org/policy": page("https://b.org/policy", POLICY),
        "https://www.a.org/sales/": page("https://www.a.org/sales/", f"{SALES}\n{POLICY}"),
    }
    monkeypatch.setattr(sources, "read_pages", lambda urls: {url: pages[url] for url in urls})

    update = read_node(state)

    # Only pages that were read are registered; the same page found by both sections gets one ID.
    assert sorted(update["sources"]) == ["S1", "S2"]
    assert update["sources"]["S1"]["url"] == "https://a.org/sales"
    assert update["sections"]["sec-1"]["source_ids"] == ["S1"]
    assert update["sections"]["sec-2"]["source_ids"] == ["S2", "S1"]
    # Every passage carries the ID of the page it came from, and each section picks its own passages.
    sec1 = update["sections"]["sec-1"]["passages"]
    assert {p["source_id"] for p in sec1} == {"S1"}
    assert "sales" in sec1[0]["text"].lower()
    assert update["search_results"] == {}
    assert update["metrics"] == {"pages_fetched": 3}


def test_read_node_caps_passages_per_section(monkeypatch):
    state = two_section_state()
    urls = [f"https://s{n}.org" for n in range(6)]
    state["search_results"] = {"sec-1": [{"url": u, "title": u} for u in urls]}
    long_text = "\n".join([SALES] * 10)
    monkeypatch.setattr(sources, "read_pages", lambda u: {url: page(url, own_text(url, long_text)) for url in u})

    passages = read_node(state)["sections"]["sec-1"]["passages"]

    assert len(passages) == MAX_PASSAGES_PER_SECTION
    assert {p["source_id"] for p in passages} == {f"S{n}" for n in range(1, 7)}  # every source got a turn


def test_read_node_keeps_a_section_with_no_readable_pages(monkeypatch):
    state = two_section_state()
    state["search_results"] = {"sec-1": [{"url": "https://blocked.org", "title": "B"}]}
    monkeypatch.setattr(sources, "read_pages", lambda urls: dict.fromkeys(urls))

    update = read_node(state)

    assert update["sections"]["sec-1"]["passages"] == []
    assert update["sources"] == {}


# ---- the graph so far: plan -> search -> read ----


def test_graph_plans_waits_for_approval_then_searches_and_reads(monkeypatch):
    plan = Outline(
        sections=[
            SectionPlan(title="Sales and growth", subtopics=["scooter sales"], queries=["scooter sales India"]),
            SectionPlan(title="Subsidy and policy", subtopics=["subsidy"], queries=["scooter subsidy India"]),
            SectionPlan(title="Outlook", subtopics=["growth outlook"], queries=["scooter growth outlook India"]),
        ]
    )
    monkeypatch.setattr(planner_module, "get_structured_llm", lambda schema: RunnableLambda(lambda _: plan))
    monkeypatch.setattr(
        sources,
        "search_queries",
        lambda queries, max_results: SearchBatch(
            results=[result(f"https://site{n}.org/{queries[0].replace(' ', '-')}", queries[0]) for n in range(2)],
            search_calls=1,
        ),
    )
    monkeypatch.setattr(
        sources, "read_pages", lambda urls: {u: page(u, own_text(u, f"{SALES}\n{POLICY}")) for u in urls}
    )

    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = graph_module.build_research_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "t"}}
        graph.invoke(initial_state("Electric scooters in India"), config)  # stops at approve_outline (T11)
        graph.invoke(Command(resume={"sections": None}), config)  # approve the outline as planned
        saved = graph.get_state(config).values

    sections = ordered_sections(saved)
    assert [s["title"] for s in sections] == ["Sales and growth", "Subsidy and policy", "Outlook"]
    assert all(s["status"] == "researching" and s["passages"] for s in sections)
    assert sorted(saved["sources"]) == ["S1", "S2", "S3", "S4", "S5", "S6"]
    every_id = {p["source_id"] for s in sections for p in s["passages"]}
    assert every_id <= set(saved["sources"])  # no passage points at an unregistered source
    assert saved["metrics"]["search_calls"] == 3 and saved["metrics"]["pages_fetched"] == 6
    assert saved["search_results"] == {}
    assert cited_source_ids("[S1][S6]") == ["S1", "S6"]
