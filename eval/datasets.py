"""T05 (Aasritha): load and check the test topics and the chatbot Q&A set.

    from eval.datasets import load_topics, load_qa_set

    topics = load_topics()                          # 15 Topic objects
    qa = load_qa_set(require_answers=True)          # for evaluation runs: fails while expected answers are missing

Every load checks the files, so a typo in the JSON fails loudly instead of skewing the scores:
unique IDs and questions, every question points at an existing topic, every answerable question checks one of its
topic's expected aspects, every unanswerable question says why, and every topic has both kinds of question.

Expected answers for answerable questions are written after the reports are generated and frozen (see README.md),
because the chatbot answers from the report. Until then they are null; pending_answers() lists them.
"""

import json
from collections import Counter
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

M = TypeVar("M", bound=BaseModel)

EVAL_DIR = Path(__file__).resolve().parent
TOPICS_FILE = EVAL_DIR / "topics.json"
QA_FILE = EVAL_DIR / "qa_set.json"

MIN_TOPICS = 10  # the brief asks for 10 to 15 test topics
MIN_ASPECTS = 3


class DatasetError(ValueError):
    """The topics or Q&A file is invalid."""


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^t\d{2,3}$")
    topic: str = Field(min_length=5)
    expected_aspects: list[str] = Field(min_length=MIN_ASPECTS)  # used by the coverage metric


class QAItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^q\d{2,3}$")
    topic_id: str
    question: str = Field(min_length=5)
    answerable: bool
    aspect: str | None = None  # answerable: the expected aspect of the topic this question checks
    expected_answer: str | None = None  # answerable: written from the frozen report
    reason: str | None = None  # unanswerable: why the report cannot contain the answer

    @model_validator(mode="after")
    def _fields_match_answerable(self) -> "QAItem":
        if self.answerable:
            if not self.aspect:
                raise ValueError("an answerable question needs the aspect it checks")
            if self.reason:
                raise ValueError("an answerable question has no reason; that field is for unanswerable ones")
        else:
            if not self.reason:
                raise ValueError("an unanswerable question needs a reason")
            if self.aspect or self.expected_answer:
                raise ValueError("an unanswerable question has no aspect or expected_answer")
        return self


def load_topics(path: Path = TOPICS_FILE) -> list[Topic]:
    topics = [_parse(Topic, row, path) for row in _read(path)]
    if len(topics) < MIN_TOPICS:
        raise DatasetError(f"{path.name}: {len(topics)} topics, the brief asks for at least {MIN_TOPICS}")
    _no_duplicates(path, "topic ID", [t.id for t in topics])
    _no_duplicates(path, "topic", [t.topic.lower() for t in topics])
    for topic in topics:
        _no_duplicates(path, f"aspect in {topic.id}", [a.lower() for a in topic.expected_aspects])
    return topics


def load_qa_set(path: Path = QA_FILE, topics: list[Topic] | None = None, require_answers: bool = False) -> list[QAItem]:
    """The Q&A set, checked against the topics. require_answers=True for evaluation runs."""
    topics_by_id = {t.id: t for t in (topics if topics is not None else load_topics())}
    items = [_parse(QAItem, row, path) for row in _read(path)]
    _no_duplicates(path, "question ID", [q.id for q in items])
    _no_duplicates(path, "question", [q.question.lower() for q in items])

    for item in items:
        topic = topics_by_id.get(item.topic_id)
        if topic is None:
            raise DatasetError(f"{path.name}: {item.id} points at unknown topic {item.topic_id}")
        if item.answerable and item.aspect not in topic.expected_aspects:
            raise DatasetError(f"{path.name}: {item.id} checks {item.aspect!r}, which is not an aspect of {topic.id}")

    for topic_id in topics_by_id:
        kinds = {q.answerable for q in items if q.topic_id == topic_id}
        if kinds != {True, False}:
            raise DatasetError(f"{path.name}: {topic_id} needs both answerable and unanswerable questions")

    if require_answers and (pending := pending_answers(items)):
        raise DatasetError(f"{path.name}: expected_answer still missing for {', '.join(pending)}")
    return items


def pending_answers(items: list[QAItem]) -> list[str]:
    """IDs of answerable questions that do not have an expected answer yet."""
    return [q.id for q in items if q.answerable and not (q.expected_answer or "").strip()]


def _read(path: Path) -> list[dict]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(rows, list):
        raise DatasetError(f"{path.name} must contain a JSON list")
    return rows


def _parse(model: type[M], row: object, path: Path) -> M:
    try:
        return model.model_validate(row)
    except ValidationError as exc:
        row_id = row.get("id", "?") if isinstance(row, dict) else "?"
        raise DatasetError(f"{path.name}: {row_id}: {exc.errors()[0]['msg']}") from exc


def _no_duplicates(path: Path, what: str, values: list[str]) -> None:
    if repeated := sorted(v for v, n in Counter(values).items() if n > 1):
        raise DatasetError(f"{path.name}: duplicate {what}: {', '.join(repeated)}")
