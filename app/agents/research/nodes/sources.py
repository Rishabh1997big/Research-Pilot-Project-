"""T07 (Rishabh): the search and read steps of the research graph, and the source registry.

    search node: for every section in sections_to_run, run its queries and choose the candidate pages to read.
    read node:   read every candidate page once (in parallel), pick the passages relevant to each section, then give
                 each page that was read a stable source ID (S1, S2...) and attach the passages with that ID.

Why it works this way:
- Only pages that were read get a source ID. Every registered source therefore exists and was read, and the writer
  can cite only text it was given (the brief's "no citation, no claim" and "do not fabricate sources").
- IDs are handed out in one sequential step, in outline order, so the same run always numbers sources the same way
  and two sections never hand out the same ID (state.py, rule 6). Pages are fetched in parallel threads before that.
- A page found by two sections is read once and gets one ID (matched with url_key from the search tool); each
  section still picks its own passages from it. A page registered earlier in the run keeps its ID.
- Passages are picked with BM25 keyword scoring against the section's title, subtopics and queries: no model
  download and the same result every time.
"""

import logging
import math
import re
from collections import Counter

from langchain_core.runnables.config import ContextThreadPoolExecutor

from app.agents.research.state import (
    SOURCE_ID_PREFIX,
    Passage,
    ResearchState,
    SearchHit,
    Section,
    Source,
    next_id,
)
from app.tools.reader import Page, read_pages
from app.tools.search import SearchResult, domain_of, search_queries, url_key

logger = logging.getLogger(__name__)

RESULTS_PER_QUERY = 5
CANDIDATES_PER_SECTION = 8  # pages read per section; some will fail, so more than MAX_SOURCES_PER_SECTION
MAX_PER_DOMAIN = 2  # per section, so the report does not lean on one site (source diversity metric)
MAX_SOURCES_PER_SECTION = 6
PASSAGES_PER_SOURCE = 3
MAX_PASSAGES_PER_SECTION = 12  # keeps the writer's prompt to roughly 3,000 tokens of sources
PASSAGE_CHARS = 1000
MIN_PASSAGE_CHARS = 150  # shorter chunks are usually menus, captions or link lists
SAME_TEXT_OVERLAP = 0.8  # two pages whose texts overlap this much are copies of one article

# Sites that are not citable evidence for a research report: social media posts and forums, and the search
# provider's own pages (in testing Tavily returned its company-database page for an unrelated UK firm named
# "Q Commerce"). Subdomains count too.
SKIPPED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "tiktok.com",
    "pinterest.com",
    "reddit.com",
    "quora.com",
    "tavily.com",
}

# Paid press releases and sponsored posts are advertising, not evidence; seen in testing as syndicated
# "This is a paid press release" market reports. Checked at the start of the page.
PAID_CONTENT = re.compile(r"paid press release|sponsored (content|post|article)|advertorial", re.IGNORECASE)

WORD = re.compile(r"[a-z0-9]+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = set(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what when where "
    "which who why will with about into over than their there these those vs".split()
)


# ---- graph nodes ----


def search_node(state: ResearchState) -> dict:
    """Search every section in sections_to_run (sections in parallel) and keep its candidate pages."""
    sections = state.get("sections", {})
    section_ids = [sid for sid in state.get("sections_to_run", []) if sid in sections]
    if not section_ids:
        return {"search_results": {}}

    def search_section(section_id: str):
        return search_queries(sections[section_id]["queries"], max_results=RESULTS_PER_QUERY)

    with ContextThreadPoolExecutor(max_workers=len(section_ids)) as pool:
        batches = dict(zip(section_ids, pool.map(search_section, section_ids), strict=True))

    for section_id, batch in batches.items():
        if batch.failed_queries:
            logger.warning("%s: no results for %s", section_id, batch.failed_queries)
    return {
        "search_results": {sid: choose_candidates(batch.results) for sid, batch in batches.items()},
        "sections": {sid: {**sections[sid], "status": "researching"} for sid in section_ids},
        "metrics": {"search_calls": sum(batch.search_calls for batch in batches.values())},
    }


def read_node(state: ResearchState) -> dict:
    """Read the candidate pages, register the ones that could be read, and attach passages with their source IDs."""
    hits_by_section = state.get("search_results", {})
    sections = state.get("sections", {})
    pages_by_url = read_pages([hit["url"] for hits in hits_by_section.values() for hit in hits])

    registry = Registry(state.get("sources", {}))
    updated: dict[str, Section] = {}
    # Outline order, so source IDs are numbered the same way every time.
    for section_id in [sid for sid in state.get("section_order", []) if sid in hits_by_section and sid in sections]:
        section = sections[section_id]
        picked = pick_sources(section, hits_by_section[section_id], pages_by_url)
        passages: list[Passage] = []
        source_ids: list[str] = []
        for hit, page, texts in picked:
            source_id = registry.id_for(hit, page)
            if source_id in source_ids:  # a copy of a page this section already uses
                continue
            source_ids.append(source_id)
            passages.extend({"source_id": source_id, "text": text} for text in texts)
        if not picked:
            logger.warning("%s (%s): none of its pages could be read", section_id, section["title"])
        updated[section_id] = {**section, "passages": _cap_passages(passages), "source_ids": source_ids}

    return {
        "sources": registry.new_sources,
        "sections": updated,
        "search_results": {},
        "metrics": {"pages_fetched": sum(page is not None for page in pages_by_url.values())},
    }


# ---- source registry ----


class Registry:
    """Hands out source IDs: an already registered page keeps its ID, a new page gets the next free one.

    A page is matched by its URL (url_key), and within one read step also by its text: copies of one article on two
    sites get one ID, so a report does not count the same article as two sources. Copies are not always identical
    (in testing, finance.yahoo.com and uk.finance.yahoo.com differed only in "inquiries" / "enquiries"), so texts
    count as the same when SAME_TEXT_OVERLAP of their 5-word sequences match.
    """

    def __init__(self, sources: dict[str, Source]) -> None:
        self._id_by_key = {url_key(source["url"]): source_id for source_id, source in sources.items()}
        self._shingles: list[tuple[frozenset[int], str]] = []
        self._used_ids = list(sources)
        self.new_sources: dict[str, Source] = {}

    def id_for(self, hit: SearchHit, page: Page) -> str:
        key = url_key(hit["url"])
        if key in self._id_by_key:
            return self._id_by_key[key]

        shingles = _shingles(page["text"])
        for other, other_id in self._shingles:
            if shingles and len(shingles & other) / len(shingles | other) >= SAME_TEXT_OVERLAP:
                self._id_by_key[key] = other_id  # the same article at another URL
                return other_id

        source_id = next_id(SOURCE_ID_PREFIX, self._used_ids)
        self._used_ids.append(source_id)
        self._id_by_key[key] = source_id
        self._shingles.append((shingles, source_id))
        self.new_sources[source_id] = {
            "id": source_id,
            "url": hit["url"],
            "title": hit["title"] or page["title"] or domain_of(hit["url"]),
            "domain": domain_of(hit["url"]),
        }
        return source_id


# ---- choosing pages and passages ----


def choose_candidates(results: list[SearchResult]) -> list[SearchHit]:
    """Up to CANDIDATES_PER_SECTION pages, taking the queries' results in turn (best of each query first), at most
    MAX_PER_DOMAIN per site, no PDFs (the reader handles HTML only) and nothing from SKIPPED_DOMAINS."""
    by_query: dict[str, list[SearchResult]] = {}
    for result in results:
        by_query.setdefault(result["query"], []).append(result)

    chosen: list[SearchHit] = []
    per_domain: Counter[str] = Counter()
    for rank in range(max((len(r) for r in by_query.values()), default=0)):
        for query_results in by_query.values():
            if rank >= len(query_results) or len(chosen) == CANDIDATES_PER_SECTION:
                continue
            result = query_results[rank]
            domain = domain_of(result["url"])
            if (
                result["url"].lower().split("?")[0].endswith(".pdf")
                or _is_skipped_site(domain)
                or per_domain[domain] >= MAX_PER_DOMAIN
            ):
                continue
            per_domain[domain] += 1
            chosen.append({"url": result["url"], "title": result["title"]})
    return chosen


def pick_sources(
    section: Section, hits: list[SearchHit], pages_by_url: dict[str, Page | None]
) -> list[tuple[SearchHit, Page, list[str]]]:
    """The first MAX_SOURCES_PER_SECTION pages that were read and have passages relevant to the section."""
    query_terms = _terms(" ".join([section["title"], *section["subtopics"], *section["queries"]]))
    picked = []
    for hit in hits:
        page = pages_by_url.get(hit["url"])
        if page is None or _is_paid_content(page["text"]):
            continue
        texts = best_passages(page["text"], query_terms, PASSAGES_PER_SOURCE)
        if texts:
            picked.append((hit, page, texts))
        if len(picked) == MAX_SOURCES_PER_SECTION:
            break
    return picked


def split_passages(text: str, max_chars: int = PASSAGE_CHARS) -> list[str]:
    """Consecutive paragraphs joined into passages of at most max_chars; long paragraphs are split at sentence ends."""
    pieces: list[str] = []
    for paragraph in (p.strip() for p in text.splitlines()):
        if len(paragraph) <= max_chars:
            pieces.extend([paragraph] if paragraph else [])
        else:
            pieces.extend(_split_long(paragraph, max_chars))

    passages: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            passages.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        passages.append(current)
    return passages


def best_passages(text: str, query_terms: list[str], limit: int) -> list[str]:
    """The `limit` passages of a page that best match the query terms (BM25), best first. Passages without any query
    term, and very short ones, are never picked."""
    passages = [p for p in split_passages(text) if len(p) >= MIN_PASSAGE_CHARS]
    if not passages or not query_terms:
        return []
    term_counts = [Counter(_terms(p)) for p in passages]
    lengths = [sum(counts.values()) or 1 for counts in term_counts]
    average_length = sum(lengths) / len(lengths)
    k1, b = 1.5, 0.75

    def idf(term: str) -> float:
        containing = sum(1 for counts in term_counts if term in counts)
        return math.log(1 + (len(passages) - containing + 0.5) / (containing + 0.5))

    weights = {term: idf(term) for term in set(query_terms)}
    scores = []
    for counts, length in zip(term_counts, lengths, strict=True):
        score = sum(
            weight * counts[term] * (k1 + 1) / (counts[term] + k1 * (1 - b + b * length / average_length))
            for term, weight in weights.items()
            if counts[term]
        )
        scores.append(score)
    ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    return [passages[i] for i in ranked[:limit] if scores[i] > 0]


def _cap_passages(passages: list[Passage]) -> list[Passage]:
    """At most MAX_PASSAGES_PER_SECTION, taking each source's best passage first, then each one's second best..."""
    by_source: dict[str, list[Passage]] = {}
    for passage in passages:
        by_source.setdefault(passage["source_id"], []).append(passage)
    capped: list[Passage] = []
    for rank in range(PASSAGES_PER_SOURCE):
        for source_passages in by_source.values():
            if rank < len(source_passages) and len(capped) < MAX_PASSAGES_PER_SECTION:
                capped.append(source_passages[rank])
    return capped


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(paragraph):
        while len(sentence) > max_chars:  # a "sentence" with no full stop, e.g. a long list
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def _shingles(text: str) -> frozenset[int]:
    """Hashes of the 5-word sequences in the first 3,000 characters; empty for texts too short to compare."""
    words = WORD.findall(text[:3000].lower())
    if len(words) < 30:
        return frozenset()
    return frozenset(hash(tuple(words[i : i + 5])) for i in range(len(words) - 4))


def _is_skipped_site(domain: str) -> bool:
    return any(domain == skipped or domain.endswith(f".{skipped}") for skipped in SKIPPED_DOMAINS)


def _is_paid_content(text: str) -> bool:
    return bool(PAID_CONTENT.search(text[:400]))


def _terms(text: str) -> list[str]:
    return [word for word in WORD.findall(text.lower()) if word not in STOPWORDS and len(word) > 1]
