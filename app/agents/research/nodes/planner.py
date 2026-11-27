"""T06 (Rishabh): planner node. Turns the topic into an outline: sections, each with subtopics and search queries.

    from app.agents.research.nodes.planner import planner, plan_outline

    builder.add_node("plan", planner)          # in the research graph
    plan_outline("EV two-wheelers in India")    # just the outline, as Pydantic objects

The LLM returns an `Outline` (structured output). The schema has no min/max limits, because Groq's strict JSON mode
can reject them; the limits are applied in code instead (clean_outline): empty and repeated sections are dropped, a
"References" section is dropped (the compiler builds the reference list), and sections and queries are capped. The
query cap matters: every query is one web search, and Tavily credits are limited.
"""

import logging
import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.agents.research.state import SECTION_ID_PREFIX, ResearchState, Section, new_section, next_id
from app.llm.provider import get_structured_llm
from app.llm.usage import track_usage

logger = logging.getLogger(__name__)

MIN_SECTIONS = 3
MAX_SECTIONS = 6
MAX_SUBTOPICS = 4
MAX_TITLE_CHARS = 120  # also the limit the outline approval accepts (nodes/approval.py)
MAX_SUBTOPIC_CHARS = 200
MAX_QUERIES = 3  # per section; at most MAX_SECTIONS * MAX_QUERIES web searches per report
ATTEMPTS = 2  # ask again once if the outline has too few usable sections

# A year or year range in a query ("2024", "2022-2025"). The model's years come from its training data and are often
# out of date, so they are removed from queries unless the topic itself names a year.
YEAR = re.compile(r"\b(?:19|20)\d{2}(?:\s*[-‐-―]\s*(?:19|20)\d{2})?\b")

NOT_RESEARCHABLE = re.compile(r"^(references?|sources|bibliography|citations|further reading)$", re.IGNORECASE)

# gpt-oss models write typographic hyphens and spaces ("two‑wheeler", "New Delhi"); in search queries these
# can miss pages that use plain characters, so they are turned into a plain "-" and " ".
PLAIN_CHARACTERS = str.maketrans({**dict.fromkeys("‐‑‒–—−", "-"), **dict.fromkeys("   ", " ")})


class SectionPlan(BaseModel):
    title: str = Field(description="Short section heading, e.g. 'Market size and growth'")
    subtopics: list[str] = Field(description="2 to 4 specific points this section must cover")
    queries: list[str] = Field(
        description="2 or 3 web search queries that find sources for this section. Each is specific and stands on "
        "its own (includes the topic's key terms), with no quotes or site: operators."
    )


class Outline(BaseModel):
    sections: list[SectionPlan] = Field(description=f"{MIN_SECTIONS} to {MAX_SECTIONS} sections in reading order")


class PlannerError(RuntimeError):
    """The LLM did not produce a usable outline."""


SYSTEM_PROMPT = """You plan research reports. Given a topic, write the outline of a factual report on it.

Rules:
- {min_sections} to {max_sections} sections in a logical reading order, covering the topic's most important aspects
  (for example: current state and key numbers, main players, policy or regulation, challenges and risks, outlook).
- Every section must be answerable from web sources, because every claim in it will be cited. No "References" or
  "Sources" section; the reference list is built automatically.
- Sections must not overlap: each fact belongs in exactly one section.
- For each section give 2 to 4 subtopics and 2 or 3 web search queries. A query must work on its own in a search
  engine, so repeat the topic's key terms in it.
- Your own knowledge may be out of date, and the search finds the current facts. So in subtopics and queries:
  - do not put years or date ranges; search engines already favour recent pages,
  - do not name companies, products, people or government schemes unless the topic names them; use general terms
    instead, such as "leading electric scooter makers India" or "government subsidy electric scooters India".

Today's date is {today}."""


def planner(state: ResearchState) -> dict:
    """Graph node: plan the outline for state["topic"] and put its sections into the state."""
    with track_usage() as usage:
        outline = plan_outline(state["topic"])
    return {**outline_to_state(outline, state), "metrics": usage.as_metrics()}


def plan_outline(topic: str, llm: Runnable | None = None) -> Outline:
    """The cleaned outline for `topic`. Raises PlannerError if the LLM gives too few usable sections twice."""
    topic = topic.strip()
    if not topic:
        raise PlannerError("The topic is empty")
    llm = llm or get_structured_llm(Outline)
    messages = [
        SystemMessage(
            SYSTEM_PROMPT.format(min_sections=MIN_SECTIONS, max_sections=MAX_SECTIONS, today=date.today().isoformat())
        ),
        HumanMessage(f"Topic: {topic}"),
    ]
    for attempt in range(1, ATTEMPTS + 1):
        outline = clean_outline(llm.invoke(messages), topic)
        if len(outline.sections) >= MIN_SECTIONS:
            return outline
        logger.warning("Outline for %r has %d usable sections (attempt %d)", topic, len(outline.sections), attempt)
    raise PlannerError(f"The LLM gave fewer than {MIN_SECTIONS} usable sections for {topic!r}, {ATTEMPTS} times")


def clean_outline(outline: Outline, topic: str) -> Outline:
    """Strip blanks, drop repeated and non-researchable sections, cap the counts and the length of titles and
    subtopics (the outline approval rejects longer ones), and make sure every section has at least one search
    query."""
    sections: list[SectionPlan] = []
    seen_titles: set[str] = set()
    for plan in outline.sections:
        title = _shorten(_tidy(plan.title), MAX_TITLE_CHARS)
        if not title or NOT_RESEARCHABLE.match(title) or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        subtopics = [_shorten(sub, MAX_SUBTOPIC_CHARS) for sub in _unique(plan.subtopics)[:MAX_SUBTOPICS]]
        raw_queries = plan.queries if YEAR.search(topic) else [YEAR.sub(" ", q) for q in plan.queries]
        queries = _unique(raw_queries)[:MAX_QUERIES] or [f"{topic} {title}"]
        sections.append(SectionPlan(title=title, subtopics=subtopics, queries=queries))
        if len(sections) == MAX_SECTIONS:
            break
    return Outline(sections=sections)


def outline_to_state(outline: Outline, state: ResearchState) -> dict:
    """State update for a new outline: new sections in order, old sections removed, outline waiting for approval.

    New IDs continue after the highest ID ever used in this run, so re-planning never reuses an old section's ID.
    """
    old_ids = [*state.get("sections", {}), *state.get("section_order", [])]
    sections: dict[str, Section | None] = dict.fromkeys(old_ids)  # None removes a section (merge_by_id)
    order: list[str] = []
    for plan in outline.sections:
        section_id = next_id(SECTION_ID_PREFIX, [*old_ids, *order])
        sections[section_id] = new_section(section_id, plan.title, plan.subtopics, plan.queries)
        order.append(section_id)
    return {"sections": sections, "section_order": order, "sections_to_run": order, "outline_approved": False}


def _shorten(text: str, limit: int) -> str:
    """`text` cut to at most `limit` characters, at a word boundary when there is one."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if text[limit] != " " and " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip(" ,;:-–—/")


def _unique(items: list[str]) -> list[str]:
    """Stripped, non-empty items, without repeats (ignoring case), in their first order."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        cleaned = _tidy(item)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique.append(cleaned)
    return unique


def _tidy(text: str) -> str:
    """Plain hyphens and spaces, single spaces, no spaces at the ends."""
    return " ".join(text.translate(PLAIN_CHARACTERS).split())
