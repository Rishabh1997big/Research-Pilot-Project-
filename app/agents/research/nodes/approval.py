"""T11: outline approval. The graph pauses after planning, the user reviews the outline, and the graph resumes with the
approved or edited outline.

    pause:   the node calls interrupt() with the proposed outline; the run stops and is saved by the checkpointer
    resume:  graph.invoke(Command(resume={"sections": [...]}), config) continues from the saved state

What the resume value looks like (the same shape the UI shows and sends back):

    {"sections": [{"id": "sec-1", "title": "...", "subtopics": ["...", ...]},   # kept or edited section
                  {"id": None,    "title": "...", "subtopics": [...]}]}        # new section; the ID is assigned here

    - A section left out of the list is removed. The list order is the new report order.
    - `Command(resume={"sections": None})` approves the outline exactly as proposed.

LangGraph only treats a resume as a resume when its value is non-empty: `Command(resume={})` is silently ignored (the
run stays paused) and `Command(resume=None)` crashes. That is why "approve as proposed" is {"sections": None}, not {}.
The API (T24) must always send a non-empty dict.

What approving does to the state: edited sections keep everything the planner gave them except what the user changed;
a new or retitled section gets new search queries made from its title and subtopics (no LLM call, so approval is
instant and free); `section_order` and `sections_to_run` follow the approved list; `outline_approved` becomes True.

If the resume value is not valid (no sections, an unknown ID, a repeated title...), the node does not fail. It pauses
again with the same outline plus an "error" message for the user, and waits for a corrected outline. (Raising instead
would not work: LangGraph stores the bad resume value and replays it on every later resume.)
"""

from collections import Counter
from typing import Annotated, Any

from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

from app.agents.research.nodes.planner import MAX_QUERIES, MAX_SECTIONS, NOT_RESEARCHABLE
from app.agents.research.state import SECTION_ID_PREFIX, ResearchState, Section, new_section, next_id

MAX_TITLE_CHARS = 120
MAX_SUBTOPICS = 10
MAX_SUBTOPIC_CHARS = 200

Title = Annotated[str, StringConstraints(max_length=MAX_TITLE_CHARS)]
Subtopic = Annotated[str, StringConstraints(max_length=MAX_SUBTOPIC_CHARS)]


class OutlineError(ValueError):
    """The approved outline cannot be used. The message is written to be shown to the user."""


class ApprovedSection(BaseModel):
    model_config = ConfigDict(extra="ignore")  # a UI may send more fields (queries, key...); they are not used

    id: str | None = None  # None (or blank) for a section the user added
    title: Title
    subtopics: list[Subtopic] = Field(default_factory=list, max_length=MAX_SUBTOPICS)

    @field_validator("id")
    @classmethod
    def _blank_id_is_new(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class ApprovedOutline(BaseModel):
    sections: list[ApprovedSection] | None = None  # None means: approve the outline as proposed


def approve_outline(state: ResearchState) -> dict:
    """Graph node: show the outline to the user and wait. The resume value decides the final outline; an unusable one
    is sent back to the user with an error message until a usable one arrives."""
    shown = proposal(state)
    while True:
        resume_value = interrupt(shown)  # on a replay this returns the earlier answers in order, then waits
        try:
            return apply_approval(state, resume_value)
        except OutlineError as exc:
            shown = {**proposal(state), "error": str(exc)}  # ask again, with the reason


def proposal(state: ResearchState) -> dict:
    """What the user is asked to review: the topic, the outline in report order, and the section limit.
    This is the interrupt payload."""
    sections = state.get("sections", {})
    return {
        "topic": state["topic"],
        "max_sections": MAX_SECTIONS,
        "sections": [
            {"id": sid, "title": sections[sid]["title"], "subtopics": list(sections[sid]["subtopics"])}
            for sid in state.get("section_order", [])
            if sid in sections
        ],
    }


def apply_approval(state: ResearchState, resume_value: Any) -> dict:
    """State update for the approved outline. Raises OutlineError if it cannot be used.

    Never changes `state`: the update it returns holds its own copies of every list.
    """
    current = state.get("sections", {})
    entries = _entries(state, resume_value)
    _check(entries, current)

    topic = state["topic"]
    used_ids = [*current, *state.get("section_order", [])]
    sections: dict[str, Section | None] = dict.fromkeys(current)  # None removes a section (merge_by_id)
    order: list[str] = []
    for entry in entries:
        title = _tidy(entry.title)
        subtopics = _unique(entry.subtopics)
        if entry.id is None:
            section_id = next_id(SECTION_ID_PREFIX, [*used_ids, *order])
            queries = _queries(topic, title, subtopics, keep=[])
            sections[section_id] = new_section(section_id, title, subtopics, queries)
        else:
            section_id = entry.id
            old = current[section_id]
            sections[section_id] = {
                **old,
                "title": title,
                "subtopics": subtopics,
                "queries": _section_queries(topic, old, title, subtopics),
                "passages": list(old["passages"]),
                "source_ids": list(old["source_ids"]),
                "citations": list(old["citations"]),
            }
        order.append(section_id)

    return {
        "sections": sections,
        "section_order": order,
        "sections_to_run": list(order),
        "outline_approved": True,
    }


def _entries(state: ResearchState, resume_value: Any) -> list[ApprovedSection]:
    """The approved sections, in report order. Only None (or sections: None) means "as proposed"; any other value
    that is not a valid outline, even an empty list or string, is an error and never approves by accident."""
    try:
        approved = ApprovedOutline.model_validate({} if resume_value is None else resume_value)
    except ValidationError as exc:
        error = exc.errors()[0]
        where = " > ".join(str(part) for part in error["loc"])
        raise OutlineError(f"The approved outline is not valid ({where}: {error['msg']}).") from exc

    if approved.sections is not None:
        return [s for s in approved.sections if s.title.strip()]  # a section with no title is dropped

    current = state.get("sections", {})
    return [
        ApprovedSection(id=sid, title=current[sid]["title"], subtopics=current[sid]["subtopics"])
        for sid in state.get("section_order", [])
        if sid in current
    ]


def _check(entries: list[ApprovedSection], current: dict[str, Section]) -> None:
    if not entries:
        raise OutlineError("The outline needs at least one section with a title.")
    if len(entries) > MAX_SECTIONS:
        raise OutlineError(
            f"The outline has {len(entries)} sections; the most allowed is {MAX_SECTIONS}, "
            "because every section costs several web searches."
        )
    ids = [e.id for e in entries if e.id is not None]
    unknown = [i for i in ids if i not in current]
    if unknown:
        raise OutlineError(f"Unknown section ID {unknown[0]!r}. Add new sections with id null.")
    if len(set(ids)) != len(ids):
        raise OutlineError("A section appears twice in the outline.")
    titles = [_tidy(e.title) for e in entries]
    for title in titles:
        if NOT_RESEARCHABLE.match(title):
            raise OutlineError(
                f"{title!r} cannot be a section: it has nothing to search for. The reference list is built "
                "automatically."
            )
    counts = Counter(title.lower() for title in titles)
    for title in titles:
        if counts[title.lower()] > 1:
            raise OutlineError(f"Two sections are both called {title!r}. Give each section its own title.")


def _section_queries(topic: str, old: Section, title: str, subtopics: list[str]) -> list[str]:
    """Queries for a section that was already in the outline. A new heading means new searches; the same heading
    keeps its searches and gets one more for each subtopic the user added."""
    if title != _tidy(old["title"]):
        return _queries(topic, title, subtopics, keep=[])
    known = {s.lower() for s in old["subtopics"]}
    added = [s for s in subtopics if s.lower() not in known]
    return _queries(topic, title, added, keep=old["queries"]) if added else list(old["queries"])


def _queries(topic: str, title: str, subtopics: list[str], keep: list[str]) -> list[str]:
    """Search queries for a new or changed section: one per subtopic (title only if there are none), plus as many of
    the queries in `keep` as fit under the cap. New queries come first, so a subtopic the user just added is never
    the one that gets cut."""
    seen = {q.lower() for q in keep}
    extra: list[str] = []
    for subtopic in subtopics or [""]:
        query = _tidy(f"{topic} {title} {subtopic}")
        if query.lower() not in seen:
            seen.add(query.lower())
            extra.append(query)
    extra = extra[:MAX_QUERIES]
    return [*keep[: MAX_QUERIES - len(extra)], *extra]


def _tidy(text: str) -> str:
    """Single spaces, none at the ends."""
    return " ".join(text.split())


def _unique(items: list[str]) -> list[str]:
    """Tidied, non-empty items without repeats (ignoring case), in their first order."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        cleaned = _tidy(item)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique.append(cleaned)
    return unique
