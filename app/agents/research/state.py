"""T03 (Rishabh): LangGraph state for the research agent.

Design rules (see docs/architecture.md, "State design"):

1. Each section carries its own queries, passages, sources, draft, citations and status. A change request can then
   re-run one section without touching the others.
2. Sources live in one registry keyed by a short ID ("S1", "S2"...). Passages and citations point at these IDs from
   the moment a page is read, so a citation is never added after the fact.
3. `sections` and `sources` are dicts keyed by ID, merged by a reducer. Parallel nodes (one per section) can each
   return only the section they changed, and setting a key to None removes it (a section deleted in outline review).
4. `section_order` is the outline order. The report contains exactly the sections listed there, in that order.
5. Everything in the state is plain dicts, lists and strings (TypedDict), so the SQLite checkpointer can save and
   restore it without registering custom classes. Pydantic models are used only for LLM structured output and are
   converted to these dicts before they enter the state.
6. Source IDs must be assigned in one sequential step (T07), never inside parallel per-section nodes, or two
   sections could hand out the same ID.
"""

from typing import Annotated, Literal, Required, TypedDict, TypeVar

T = TypeVar("T")

SectionStatus = Literal["planned", "researching", "drafted", "verified", "needs_revision"]
FeedbackKind = Literal["add_subtopic", "expand_section", "replace_sources", "other"]


class Source(TypedDict):
    id: str  # "S1", "S2"... written in drafts as [S1]
    url: str
    title: str
    domain: str  # used by the source diversity metric


class Passage(TypedDict):
    source_id: str
    text: str


class Citation(TypedDict):
    claim: str
    source_ids: list[str]
    supported: bool | None  # None until the verifier has checked the claim
    note: str  # verifier's reason when a claim is not supported


class Section(TypedDict):
    id: str  # "sec-1", "sec-2"...
    title: str
    subtopics: list[str]
    queries: list[str]
    passages: list[Passage]
    source_ids: list[str]
    draft: str
    citations: list[Citation]
    status: SectionStatus
    revision: int  # how many times this section has been rewritten after feedback


class Feedback(TypedDict):
    request: str  # the user's words, e.g. "Make section 3 more detailed"
    kind: FeedbackKind
    section_ids: list[str]  # sections the request applies to


class RunMetrics(TypedDict):
    llm_calls: int
    search_calls: int
    pages_fetched: int
    input_tokens: int
    output_tokens: int


def merge_by_id(left: dict[str, T] | None, right: dict[str, T | None] | None) -> dict[str, T]:
    """Reducer for ID-keyed dicts: entries in `right` replace entries in `left`; a None value removes the entry."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def add_metrics(left: RunMetrics | None, right: dict[str, int] | None) -> RunMetrics:
    """Reducer for metrics: nodes return only the counts they add, e.g. {"llm_calls": 1}."""
    total = empty_metrics()
    for part in (left or {}, right or {}):
        for name, count in part.items():
            total[name] = total.get(name, 0) + count  # type: ignore[literal-required]
    return total


class ResearchState(TypedDict, total=False):
    topic: Required[str]
    section_order: list[str]
    sections: Annotated[dict[str, Section], merge_by_id]
    sources: Annotated[dict[str, Source], merge_by_id]
    outline_approved: bool
    feedback: list[Feedback]
    sections_to_run: list[str]  # sections the next search, write and verify pass works on
    report: str  # final Markdown with numbered citations and a reference list
    metrics: Annotated[RunMetrics, add_metrics]


def empty_metrics() -> RunMetrics:
    return {"llm_calls": 0, "search_calls": 0, "pages_fetched": 0, "input_tokens": 0, "output_tokens": 0}


def new_section(
    section_id: str, title: str, subtopics: list[str] | None = None, queries: list[str] | None = None
) -> Section:
    return {
        "id": section_id,
        "title": title,
        "subtopics": list(subtopics or []),
        "queries": list(queries or []),
        "passages": [],
        "source_ids": [],
        "draft": "",
        "citations": [],
        "status": "planned",
        "revision": 0,
    }


def initial_state(topic: str) -> ResearchState:
    """Input for a new research run."""
    return {
        "topic": topic.strip(),
        "section_order": [],
        "sections": {},
        "sources": {},
        "outline_approved": False,
        "feedback": [],
        "sections_to_run": [],
        "report": "",
        "metrics": empty_metrics(),
    }


def ordered_sections(state: ResearchState) -> list[Section]:
    """Sections in outline order, skipping IDs that are no longer in `sections`."""
    sections = state.get("sections", {})
    return [sections[sid] for sid in state.get("section_order", []) if sid in sections]
