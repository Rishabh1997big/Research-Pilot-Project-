"""The knowledge base with the real BGE-small model and a real Chroma store (in a temp folder).
Skipped by default; run with:  uv run pytest -m live   (the first run downloads the model, about 70 MB)"""

import json
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.kb.ingest import ingest_report, search_report

pytestmark = pytest.mark.live

SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "sample_report.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("chroma")
    client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
    ingest_report(SAMPLE, "live-sample", client=client)
    return client


@pytest.mark.parametrize(
    ("question", "section_id"),
    [
        ("How fast is quick commerce growing?", "sec-1"),
        ("What is a dark store?", "sec-2"),
        ("Which platform has the biggest share?", "sec-2"),
        ("Do delivery partners get any welfare benefits?", "sec-3"),
    ],
)
def test_questions_find_the_right_section(client, question, section_id):
    hits = search_report("live-sample", question, k=3, client=client)

    assert hits[0]["metadata"]["section_id"] == section_id


def test_an_unrelated_question_is_further_away_than_a_related_one(client):
    related = search_report("live-sample", "What is a dark store?", k=1, client=client)[0]["distance"]
    unrelated = search_report("live-sample", "What is the capital of France?", k=1, client=client)[0]["distance"]

    assert unrelated > related
