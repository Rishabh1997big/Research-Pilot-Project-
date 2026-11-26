"""The reader on a real, stable page. Skipped by default; run with:  uv run pytest -m live

(Not Wikipedia: it answers 403 to every client whose User-Agent has no contact details; see app/tools/reader.py.)
"""

import pytest

from app.tools.reader import read_page

pytestmark = pytest.mark.live


def test_reads_a_real_page():
    page = read_page("https://docs.python.org/3/tutorial/introduction.html")

    assert page is not None
    assert len(page["text"]) > 2000
    assert "python" in page["text"].lower()
