import copy
import hashlib
import json
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.kb.ingest import CHUNK_CHARS, chunk_report, collection_name, ingest_report, search_report, split_draft

SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "sample_report.json").read_text(encoding="utf-8"))


class FakeEmbedder:
    """Bag-of-words vectors: texts that share words are close. No model download."""

    DIM = 64

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for word in text.lower().split():
            word = word.strip(".,:;[]()?")
            if len(word) > 3:
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        return vector if any(vector) else [1.0] + [0.0] * (self.DIM - 1)

    def embed_passages(self, texts):
        self.passages = getattr(self, "passages", 0) + len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, question):
        return self._vector(question)


@pytest.fixture
def client():
    return chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True))


@pytest.fixture
def report_id(request):
    return f"test-{request.node.name}"  # the ephemeral client is shared in a process; one collection per test


def sample() -> dict:
    return copy.deepcopy(SAMPLE)


# ---- chunking ----


def test_chunks_follow_sections_and_carry_their_metadata():
    chunks = chunk_report(sample())

    assert [c["metadata"]["section_id"] for c in chunks][0] == "sec-1"
    first = chunks[0]
    assert first["id"] == "sec-1-0"
    assert first["text"].startswith("Market size and growth\n")  # the section title is part of the embedded text
    assert first["metadata"] == {
        "section_id": "sec-1",
        "section_title": "Market size and growth",
        "section_position": 1,
        "chunk_index": 0,
        "source_ids": ["S1", "S2"],  # sec-1's two short paragraphs fit in one chunk
        "urls": ["https://example.org/sample/qcom-orders", "https://example.org/sample/dark-stores"],
    }


def test_a_chunk_lists_only_the_sources_it_cites():
    report = sample()
    cited = " ".join(f"Gig workers in the sample city {n} receive welfare payments [S4]." for n in range(14))
    report["sections"]["sec-3"]["draft"] = f"{cited}\n\nThe sample report found no national rule on delivery times."

    chunks = {c["id"]: c for c in chunk_report(report, ["sec-3"])}

    assert chunks["sec-3-0"]["metadata"]["source_ids"] == ["S4"]
    assert "source_ids" not in chunks["sec-3-1"]["metadata"]  # no citation: no source keys


def test_no_chunk_crosses_a_section_or_exceeds_the_size_limit():
    chunks = chunk_report(sample())

    for chunk in chunks:
        body = chunk["text"].split("\n", 1)[1]
        assert len(body) <= CHUNK_CHARS
        assert chunk["metadata"]["section_title"] == chunk["text"].split("\n", 1)[0]
    assert len([c for c in chunks if c["metadata"]["section_id"] == "sec-2"]) >= 2  # the long paragraph was split


def test_split_draft_keeps_whole_paragraphs_together_when_they_fit():
    draft = "First paragraph.\n\nSecond paragraph.\n\n\nThird   paragraph\nwith a line break."

    assert split_draft(draft) == ["First paragraph.\n\nSecond paragraph.\n\nThird paragraph with a line break."]


def test_split_draft_splits_a_long_paragraph_at_sentence_ends():
    paragraph = " ".join(f"Sentence number {n} about dark stores." for n in range(60))

    pieces = split_draft(paragraph, max_chars=300)

    assert all(len(p) <= 300 and p.endswith(".") for p in pieces)
    assert " ".join(pieces) == paragraph


def test_sections_without_a_draft_are_skipped():
    report = sample()
    report["sections"]["sec-3"]["draft"] = ""

    assert {c["metadata"]["section_id"] for c in chunk_report(report)} == {"sec-1", "sec-2"}


def test_only_the_requested_sections_are_chunked():
    assert {c["metadata"]["section_id"] for c in chunk_report(sample(), ["sec-2"])} == {"sec-2"}


def test_collection_names_are_valid_for_chroma():
    assert collection_name("run 42 / Quick commerce!") == "report-run-42-Quick-commerce"
    assert len(collection_name("x" * 200)) <= 63
    assert collection_name("???") == "report-unnamed"


# ---- storing and searching ----


def test_ingest_stores_every_chunk_and_search_finds_the_right_section(client, report_id):
    embedder = FakeEmbedder()

    stored = ingest_report(sample(), report_id, client=client, embedder=embedder)
    hits = search_report(report_id, "What is a dark store warehouse?", k=2, client=client, embedder=embedder)

    assert stored == len(chunk_report(sample()))
    assert hits[0]["metadata"]["section_id"] == "sec-2"
    assert "dark stores" in hits[0]["text"]
    assert hits[0]["distance"] <= hits[1]["distance"]


def test_re_ingesting_one_section_replaces_only_that_section(client, report_id):
    embedder = FakeEmbedder()
    ingest_report(sample(), report_id, client=client, embedder=embedder)
    report = sample()
    report["sections"]["sec-3"]["draft"] = "A new national code for gig workers was proposed in the sample [S4]."

    stored = ingest_report(report, report_id, section_ids=["sec-3"], client=client, embedder=embedder)

    collection = client.get_collection(collection_name(report_id))
    sec3 = collection.get(where={"section_id": "sec-3"})["documents"]
    assert stored == 1 and len(sec3) == 1 and "new national code" in sec3[0]
    assert collection.get(where={"section_id": "sec-1"})["ids"] == ["sec-1-0"]  # untouched
    assert embedder.passages == len(chunk_report(sample())) + 1  # only the changed section was embedded again


def test_a_full_ingest_removes_sections_that_left_the_report(client, report_id):
    embedder = FakeEmbedder()
    ingest_report(sample(), report_id, client=client, embedder=embedder)
    report = sample()
    report["section_order"] = ["sec-1", "sec-2"]
    del report["sections"]["sec-3"]

    ingest_report(report, report_id, client=client, embedder=embedder)

    collection = client.get_collection(collection_name(report_id))
    assert collection.get(where={"section_id": "sec-3"})["ids"] == []


def test_sources_can_be_filtered(client, report_id):
    ingest_report(sample(), report_id, client=client, embedder=FakeEmbedder())

    ids = client.get_collection(collection_name(report_id)).get(where={"source_ids": {"$contains": "S4"}})["ids"]

    assert ids == ["sec-3-0"]


def test_search_on_an_empty_report_returns_nothing(client):
    assert search_report("never-ingested", "anything", client=client, embedder=FakeEmbedder()) == []
