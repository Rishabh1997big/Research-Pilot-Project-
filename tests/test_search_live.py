"""Real web searches. Skipped by default; run with:  uv run pytest -m live"""

import pytest

from app.config import settings
from app.tools.search import search_queries, url_key

pytestmark = pytest.mark.live

NO_CACHE = settings.model_copy(update={"search_cache": False})


def test_real_search_returns_unique_urls():
    batch = search_queries(["electric two-wheeler sales India", "electric scooter market share India"], cfg=NO_CACHE)

    assert batch.results, f"no results; failed queries: {batch.failed_queries}"
    assert all(r["url"].startswith("http") for r in batch.results)
    assert len({url_key(r["url"]) for r in batch.results}) == len(batch.results)


def test_tavily_alone():
    if not settings.tavily_api_key.strip():
        pytest.skip("TAVILY_API_KEY is empty in .env")
    batch = search_queries(
        ["FAME subsidy electric scooters India"], cfg=NO_CACHE.model_copy(update={"search_provider": "tavily"})
    )

    assert batch.results and batch.results[0]["provider"] == "tavily"
