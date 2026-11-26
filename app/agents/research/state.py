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
7. New IDs come from next_section_id() and next_source_id(), and [S#] citation marks are read with
   cited_source_ids(), so the writer, compiler, verifier and evaluation all agree on the formats.
"""

import re
from collections.abc import Iterable
from typing import Annotated, Literal, Required, TypedDict, TypeVar

T = TypeVar("T")

SECTION_ID_PREFIX = "sec-"  # sec-1, sec-2...
SOURCE_ID_PREFIX = "S"  # S1, S2...
# How the writer cites sources in a draft: [S1], or several in one mark: [S1, S3]
CITATION_MARK = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")

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
    # Sections the request applies to. Empty for "add_subtopic" means the topic gets a new section of its own
    # ("Add a subtopic on government policy"). For "replace_sources", the weak sources of a section are the ones
    # that no supported citation uses (see the verifier results in Section.citations).
    section_ids: list[str]


class SearchHit(TypedDict):
    """A page the search found for a section, waiting to be read. It gets a source ID only once it has been read."""

    url: str
    title: str


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
    """Reducer for metrics: nodes return only the counts they add, e.g. {"llm_calls": 1}.

    An unknown name raises, so a typo such as "llm_call" fails the run instead of being silently dropped from the
    totals.
    """
    total = empty_metrics()
    for part in (left or {}, right or {}):
        for name, count in part.items():
            if name not in total:
                raise ValueError(f"Unknown metric {name!r}; RunMetrics has: {', '.join(total)}")
            total[name] += count  # type: ignore[literal-required]
    return total


class ResearchState(TypedDict, total=False):
    topic: Required[str]
    section_order: list[str]
    sections: Annotated[dict[str, Section], merge_by_id]
    sources: Annotated[dict[str, Source], merge_by_id]
    outline_approved: bool
    feedback: list[Feedback]
    sections_to_run: list[str]  # sections the next search, write and verify pass works on
    # Search results per section, from the search step to the read step; the read step empties it again.
    search_results: dict[str, list[SearchHit]]
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
        "search_results": {},
        "report": "",
        "metrics": empty_metrics(),
    }


def ordered_sections(state: ResearchState) -> list[Section]:
    """Sections in outline order, skipping IDs that are no longer in `sections`."""
    sections = state.get("sections", {})
    return [sections[sid] for sid in state.get("section_order", []) if sid in sections]


def next_id(prefix: str, existing: Iterable[str]) -> str:
    """One more than the highest ID in use: next_id("S", ["S1", "S3"]) == "S4". Gaps are never filled."""
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
    numbers = [int(match.group(1)) for key in existing if (match := pattern.fullmatch(key))]
    return f"{prefix}{max(numbers, default=0) + 1}"


def next_section_id(state: ResearchState) -> str:
    """ID for a new section (outline edit, or a change request that adds a section)."""
    return next_id(SECTION_ID_PREFIX, [*state.get("sections", {}), *state.get("section_order", [])])


def next_source_id(state: ResearchState) -> str:
    """ID for a new source. Sources are never removed, so a source ID is never reused within a run.

    Only call this from the one sequential registry step (rule 6). To register several sources in that step, use
    next_id(SOURCE_ID_PREFIX, ...) on the IDs you have handed out so far.
    """
    return next_id(SOURCE_ID_PREFIX, state.get("sources", {}))


def cited_source_ids(text: str) -> list[str]:
    """Source IDs cited in a draft, in order of first use. Accepts [S1], [S1][S2] and [S1, S2]."""
    found: list[str] = []
    for group in CITATION_MARK.findall(text):
        for source_id in re.split(r"\s*,\s*", group):
            if source_id not in found:
                found.append(source_id)
    return found
