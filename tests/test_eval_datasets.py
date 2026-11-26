import json

import pytest

from eval.datasets import DatasetError, load_qa_set, load_topics, pending_answers

# ---- the real files ----


def test_real_topics_are_valid():
    topics = load_topics()

    assert len(topics) == 15
    assert all(len(t.expected_aspects) >= 5 for t in topics)


def test_real_qa_set_is_valid_and_balanced():
    qa = load_qa_set()

    assert len(qa) == 45
    assert sum(q.answerable for q in qa) == 30
    assert sum(not q.answerable for q in qa) == 15


def test_expected_answers_are_still_pending_until_reports_are_frozen():
    qa = load_qa_set()

    assert len(pending_answers(qa)) == 30
    with pytest.raises(DatasetError, match="expected_answer still missing"):
        load_qa_set(require_answers=True)


# ---- the checks catch bad data ----

ASPECTS = ["market size", "key players", "government policy"]
TOPICS = [{"id": f"t{n:02d}", "topic": f"Topic number {n}", "expected_aspects": ASPECTS} for n in range(1, 11)]


def answerable(qid: str, topic_id: str = "t01", **extra) -> dict:
    return {
        "id": qid,
        "topic_id": topic_id,
        "question": f"Question {qid}?",
        "answerable": True,
        "aspect": ASPECTS[0],
    } | extra


def unanswerable(qid: str, topic_id: str = "t01", **extra) -> dict:
    return {
        "id": qid,
        "topic_id": topic_id,
        "question": f"Question {qid}?",
        "answerable": False,
        "reason": "Off topic",
    } | extra


def full_qa_set() -> list[dict]:
    """A valid Q&A set: one answerable and one unanswerable question per topic."""
    rows = []
    for topic in TOPICS:
        rows += [answerable(f"q{len(rows) + 1:02d}", topic["id"]), unanswerable(f"q{len(rows) + 2:02d}", topic["id"])]
    return rows


def write(tmp_path, name: str, rows) -> object:
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def load(tmp_path, qa_rows, topic_rows=TOPICS, **kwargs):
    topics = load_topics(write(tmp_path, "topics.json", topic_rows))
    return load_qa_set(write(tmp_path, "qa.json", qa_rows), topics=topics, **kwargs)


def test_a_valid_synthetic_set_loads(tmp_path):
    assert len(load(tmp_path, full_qa_set())) == 20


@pytest.mark.parametrize(
    ("topic_rows", "message"),
    [
        (TOPICS[:9], "at least 10"),
        (TOPICS + [TOPICS[0]], "duplicate topic ID"),
        ([{**TOPICS[0], "expected_aspects": ["one", "two"]}, *TOPICS[1:]], "t01"),
        ([{**TOPICS[0], "expected_aspects": ["a", "b", "A"]}, *TOPICS[1:]], "duplicate aspect"),
        ([{**TOPICS[0], "extra": 1}, *TOPICS[1:]], "t01"),
    ],
)
def test_bad_topics_are_rejected(tmp_path, topic_rows, message):
    with pytest.raises(DatasetError, match=message):
        load_topics(write(tmp_path, "topics.json", topic_rows))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows + [answerable("q01", "t02")], "duplicate question ID"),
        (lambda rows: rows + [answerable("q99", "t99")], "unknown topic t99"),
        (lambda rows: rows + [answerable("q99", aspect="not an aspect")], "not an aspect of t01"),
        (lambda rows: rows + [answerable("q99", aspect=None)], "needs the aspect"),
        (lambda rows: rows + [unanswerable("q99", reason=None)], "needs a reason"),
        (lambda rows: rows + [unanswerable("q99", expected_answer="Yes")], "no aspect or expected_answer"),
        (lambda rows: [r for r in rows if not (r["topic_id"] == "t03" and not r["answerable"])], "t03 needs both"),
        (lambda rows: rows + [{**rows[0], "id": "q99"}], "duplicate question"),
    ],
)
def test_bad_questions_are_rejected(tmp_path, change, message):
    with pytest.raises(DatasetError, match=message):
        load(tmp_path, change(full_qa_set()))


def test_require_answers_passes_once_every_answer_is_written(tmp_path):
    rows = [{**r, "expected_answer": "An answer"} if r["answerable"] else r for r in full_qa_set()]

    assert len(load(tmp_path, rows, require_answers=True)) == 20


def test_unreadable_file_gives_a_clear_error(tmp_path):
    path = tmp_path / "topics.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DatasetError, match="cannot read topics.json"):
        load_topics(path)
