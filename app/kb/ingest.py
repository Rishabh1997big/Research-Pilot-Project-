"""T09 (Deepanshu): chunk the report by section, embed the chunks with BGE-small and store them in Chroma.

    from app.kb.ingest import ingest_report, search_report

    ingest_report(state, report_id="run-42")                          # every section of the report
    ingest_report(state, report_id="run-42", section_ids=["sec-3"])    # after a change request: only sec-3
    search_report("run-42", "Which companies lead the market?", k=4)   # nearest chunks, for checks and T14

Input is the research state (topic, section_order, sections with their drafts, sources), not the compiled Markdown:
the state already says which section a paragraph belongs to and which sources it cites, so nothing is parsed back
out of the Markdown. Until the writer (T12) exists, tests use tests/fixtures/sample_report.json.

How a report is stored:
- One Chroma collection per report ("report-<report_id>"), with cosine distance.
- A chunk is one or more whole paragraphs of one section's draft, at most CHUNK_CHARS long. A chunk never mixes
  two sections, so every answer can name its section. A paragraph longer than that is split at sentence ends.
- The embedded text starts with the section title, so a chunk like "Blinkit leads with 46%..." is still found by
  "Who leads quick commerce?".
- Metadata per chunk: section_id, section_title, section_position (1, 2... in outline order), chunk_index, and the
  source_ids and urls the chunk cites ([S#] marks in its text). A chunk without citations has no source keys,
  because Chroma rejects empty lists.
- Re-ingesting a section first deletes its old chunks, and a full ingest also deletes the chunks of sections that
  are no longer in the report. So a change request re-embeds only the sections it changed.

Embeddings run locally with fastembed (ONNX, no API key). The model is downloaded once into data/models. Chroma's
anonymous usage telemetry is switched off.
"""

import logging
import re
import threading
from collections.abc import Iterable
from typing import TypedDict

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.agents.research.state import ResearchState, cited_source_ids, ordered_sections
from app.config import Settings, settings

logger = logging.getLogger(__name__)

CHUNK_CHARS = 900  # about 200 tokens: small enough to retrieve precisely, big enough to hold a whole claim
# BAAI's recommended instruction for short retrieval queries to bge-*-v1.5 models; passages get no prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Chunk(TypedDict):
    id: str
    text: str  # what is embedded and returned: "<section title>\n<paragraphs>"
    metadata: dict


class SearchHit(TypedDict):
    text: str
    metadata: dict
    distance: float  # cosine distance: 0 is identical, higher is less similar


# ---- chunking ----


def chunk_report(state: ResearchState, section_ids: Iterable[str] | None = None) -> list[Chunk]:
    """Chunks of the given sections (all sections in the outline when None), in outline order."""
    wanted = set(section_ids) if section_ids is not None else None
    sources = state.get("sources", {})
    chunks: list[Chunk] = []
    for position, section in enumerate(ordered_sections(state), start=1):
        if wanted is not None and section["id"] not in wanted:
            continue
        if not section["draft"].strip():
            logger.warning("%s (%s) has no draft yet; nothing to ingest", section["id"], section["title"])
            continue
        for index, text in enumerate(split_draft(section["draft"])):
            metadata: dict = {
                "section_id": section["id"],
                "section_title": section["title"],
                "section_position": position,
                "chunk_index": index,
            }
            cited = [sid for sid in cited_source_ids(text) if sid in sources]
            if cited:
                metadata["source_ids"] = cited
                metadata["urls"] = [sources[sid]["url"] for sid in cited]
            chunks.append(
                {"id": f"{section['id']}-{index}", "text": f"{section['title']}\n{text}", "metadata": metadata}
            )
    return chunks


def split_draft(draft: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Whole paragraphs (blank-line separated) joined up to max_chars; a longer paragraph is split at sentence ends."""
    pieces: list[str] = []
    for paragraph in (" ".join(p.split()) for p in re.split(r"\n\s*\n", draft)):
        if not paragraph:
            continue
        pieces.extend([paragraph] if len(paragraph) <= max_chars else _split_sentences(paragraph, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_sentences(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(paragraph):
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


# ---- storing and searching ----


def ingest_report(
    state: ResearchState,
    report_id: str,
    section_ids: Iterable[str] | None = None,
    client: chromadb.api.ClientAPI | None = None,
    embedder: "Embedder | None" = None,
) -> int:
    """Embed and store the report's chunks (only `section_ids` when given). Returns the number of chunks stored."""
    collection = _collection(report_id, client)
    sections = set(section_ids) if section_ids is not None else {s["id"] for s in ordered_sections(state)}
    if section_ids is None:
        # Sections removed from the report since the last ingest
        existing = {m["section_id"] for m in collection.get(include=["metadatas"])["metadatas"] or []}
        sections |= existing - {s["id"] for s in ordered_sections(state)}
    for section_id in sections:
        collection.delete(where={"section_id": section_id})

    chunks = chunk_report(state, section_ids)
    if chunks:
        embeddings = (embedder or default_embedder()).embed_passages([c["text"] for c in chunks])
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=embeddings,
        )
    logger.info("Ingested %d chunks into %s", len(chunks), collection.name)
    return len(chunks)


def search_report(
    report_id: str,
    question: str,
    k: int = 4,
    client: chromadb.api.ClientAPI | None = None,
    embedder: "Embedder | None" = None,
) -> list[SearchHit]:
    """The k chunks nearest to the question, nearest first."""
    collection = _collection(report_id, client)
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[(embedder or default_embedder()).embed_query(question)],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": text, "metadata": metadata, "distance": distance}
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]


def collection_name(report_id: str) -> str:
    """A valid Chroma collection name for the report: letters, digits, ".", "_" and "-", at most 63 characters."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", report_id).strip("-._")
    return f"report-{cleaned or 'unnamed'}"[:63].rstrip("-._")


def chroma_client(cfg: Settings | None = None) -> chromadb.api.ClientAPI:
    cfg = cfg or settings
    return chromadb.PersistentClient(path=str(cfg.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False))


def _collection(report_id: str, client: chromadb.api.ClientAPI | None):
    return (client or chroma_client()).get_or_create_collection(
        collection_name(report_id), metadata={"hnsw:space": "cosine"}
    )


# ---- embeddings ----


class Embedder:
    """BGE-small through fastembed. Loaded once, on first use."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, question: str) -> list[float]:
        return next(iter(self._model.query_embed(QUERY_INSTRUCTION + question))).tolist()


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def default_embedder() -> Embedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = Embedder(settings.embedding_model, str(settings.embedding_cache_dir))
        return _embedder
