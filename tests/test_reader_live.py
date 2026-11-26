"""The reader on a real, stable page. Skipped by default; run with:  uv run pytest -m live

Wikipedia is tested only when READER_CONTACT is set: it answers 403 to readers without a contact.
"""

import pytest

from app.tools.reader import read_page

pytestmark = pytest.mark.live


def test_reads_a_real_page():
    page = read_page("https://docs.python.org/3/tutorial/introduction.html")

    assert page is not None
    assert len(page["text"]) > 2000
    assert "python" in page["text"].lower()


def test_reads_wikipedia_when_a_contact_is_set():
    from app.config import settings

    if not settings.reader_contact.strip():
        pytest.skip("READER_CONTACT is empty in .env; Wikipedia blocks readers without a contact")
    page = read_page(
        "https://en.wikipedia.org/wiki/Electric_vehicle", cfg=settings.model_copy(update={"page_cache": False})
    )

    assert page is not None and len(page["text"]) > 5000
