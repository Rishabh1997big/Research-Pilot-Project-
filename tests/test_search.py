import logging

import httpx
import pytest

from app.config import Settings
from app.tools import search
from app.tools.search import SearchConfigError, domain_of, provider_order, search_queries, url_key


@pytest.fixture
def cfg(tmp_path) -> Settings:
    # tavily_api_key is set explicitly: app.config loads the real .env into the environment, and no test may spend
    # real Tavily credits.
    return Settings(_env_file=None, search_provider="ddgs", tavily_api_key="", search_cache_dir=tmp_path / "cache")


@pytest.fixture
def with_tavily(cfg) -> Settings:
    return cfg.model_copy(update={"tavily_api_key": "tvly-test"})


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(search, "BACKOFF_SECONDS", 0)
    monkeypatch.setattr(search, "MIN_INTERVAL_SECONDS", 0)


def fake_provider(monkeypatch, provider: str, pages: dict[str, list[str] | Exception]) -> list[str]:
    """Replace a provider with canned results per query (an Exception fails that query). Returns the calls made."""
    calls: list[str] = []

    def fetch(query: str, max_results: int, cfg: Settings):
        calls.append(query)
        outcome = pages[query]
        if isinstance(outcome, Exception):
            raise outcome
        return [
            {"url": url, "title": f"T {url}", "snippet": "", "query": query, "provider": provider}
            for url in outcome[:max_results]
        ]

    monkeypatch.setitem(search._PROVIDERS, provider, fetch)
    return calls


# ---- duplicates, retries, cache ----


def test_duplicate_urls_are_removed_across_queries(monkeypatch, cfg):
    fake_provider(
        monkeypatch,
        "ddgs",
        {
            "q1": ["https://a.org/report", "https://b.org/x"],
            "q2": ["http://www.a.org/report/?utm_source=ddg#top", "https://c.org/y"],
        },
    )

    batch = search_queries(["q1", "q2"], cfg=cfg)

    assert [r["url"] for r in batch.results] == ["https://a.org/report", "https://b.org/x", "https://c.org/y"]
    assert batch.results[2]["query"] == "q2"
    assert batch.search_calls == 2


def test_blank_and_repeated_queries_are_searched_once(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"q1": ["https://a.org"]})

    search_queries(["q1", " q1 ", "", "  "], cfg=cfg)

    assert calls == ["q1"]


def test_without_fallback_a_failing_query_is_retried_then_skipped(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"bad": RuntimeError("202 Ratelimit"), "good": ["https://a.org"]})

    batch = search_queries(["bad", "good"], cfg=cfg)

    assert calls.count("bad") == search.MAX_RETRIES + 1
    assert batch.failed_queries == ["bad"]
    assert [r["url"] for r in batch.results] == ["https://a.org"]
    assert batch.search_calls == 2


def test_a_query_that_fails_once_succeeds_on_retry(monkeypatch, cfg):
    outcomes = [RuntimeError("timeout"), [{"url": "https://a.org", "title": "", "snippet": "", "query": "q"}]]

    def flaky(query, max_results, cfg):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setitem(search._PROVIDERS, "ddgs", flaky)

    batch = search_queries(["q"], cfg=cfg)

    assert batch.failed_queries == []
    assert len(batch.results) == 1


def test_results_are_cached_and_reused(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"q1": ["https://a.org"]})

    first = search_queries(["q1"], cfg=cfg)
    second = search_queries(["q1"], cfg=cfg)

    assert calls == ["q1"]
    assert second.results == first.results
    assert second.search_calls == 0


def test_cache_can_be_switched_off(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"q1": ["https://a.org"]})
    no_cache = cfg.model_copy(update={"search_cache": False})

    search_queries(["q1"], cfg=no_cache)
    search_queries(["q1"], cfg=no_cache)

    assert calls == ["q1", "q1"]
    assert not cfg.search_cache_dir.exists()


def test_failed_and_empty_searches_are_not_cached(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"empty": [], "bad": RuntimeError("down")})

    search_queries(["empty", "bad"], cfg=cfg)
    search_queries(["empty", "bad"], cfg=cfg)

    assert calls.count("empty") == 2


# ---- provider choice and Tavily fallback ----


def test_provider_order():
    base = Settings(_env_file=None, tavily_api_key="")

    assert provider_order(base) == ["ddgs"]
    assert provider_order(base.model_copy(update={"tavily_api_key": "k"})) == ["ddgs", "tavily"]
    assert provider_order(base.model_copy(update={"tavily_api_key": "k", "search_provider": "tavily"})) == [
        "tavily",
        "ddgs",
    ]


def test_tavily_without_a_key_gives_a_clear_error(cfg):
    with pytest.raises(SearchConfigError, match="TAVILY_API_KEY"):
        search_queries(["q"], cfg=cfg.model_copy(update={"search_provider": "tavily"}))


def test_tavily_takes_over_after_one_failed_ddgs_attempt(monkeypatch, with_tavily, caplog):
    ddgs_calls = fake_provider(monkeypatch, "ddgs", {"q": RuntimeError("202 Ratelimit")})
    tavily_calls = fake_provider(monkeypatch, "tavily", {"q": ["https://a.org"]})

    with caplog.at_level(logging.WARNING, logger="app.tools.search"):
        batch = search_queries(["q"], cfg=with_tavily)

    assert ddgs_calls == ["q"]  # one attempt only, no slow retries when a fallback exists
    assert tavily_calls == ["q"]
    assert batch.results[0]["provider"] == "tavily"
    assert batch.search_calls == 1
    assert "using tavily" in caplog.text


def test_empty_ddgs_answer_also_moves_to_tavily(monkeypatch, with_tavily):
    fake_provider(monkeypatch, "ddgs", {"q": []})
    fake_provider(monkeypatch, "tavily", {"q": ["https://a.org"]})

    assert len(search_queries(["q"], cfg=with_tavily).results) == 1


def test_query_is_skipped_only_when_every_provider_fails(monkeypatch, with_tavily):
    fake_provider(monkeypatch, "ddgs", {"q": RuntimeError("blocked")})
    tavily_calls = fake_provider(monkeypatch, "tavily", {"q": RuntimeError("401")})

    batch = search_queries(["q"], cfg=with_tavily)

    assert batch.failed_queries == ["q"]
    assert len(tavily_calls) == search.MAX_RETRIES + 1  # the last provider gets the retries


def test_tavily_request_and_response_mapping(monkeypatch, with_tavily):
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, json=json)
        payload = {"results": [{"url": "https://a.org/x", "title": "A", "content": "Snippet", "score": 0.9}]}
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(search.httpx, "post", fake_post)

    results = search._tavily_search("ev sales", 3, with_tavily)

    assert sent["headers"]["Authorization"] == "Bearer tvly-test"
    assert sent["json"] == {"query": "ev sales", "max_results": 3, "search_depth": "basic"}
    assert results == [
        {"url": "https://a.org/x", "title": "A", "snippet": "Snippet", "query": "ev sales", "provider": "tavily"}
    ]


# ---- URL keys ----


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://www.a.org/x", "http://a.org/x/"),
        ("https://a.org/x?utm_source=t&utm_medium=m", "https://a.org/x"),
        ("https://a.org/x#section-2", "https://a.org/x"),
        ("https://A.org/x?fbclid=123&id=7", "https://a.org/x?id=7"),
        ("https://a.org:443/x", "https://a.org/x"),
    ],
)
def test_url_key_matches_the_same_page(a, b):
    assert url_key(a) == url_key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://a.org/x?id=1", "https://a.org/x?id=2"),
        ("https://a.org/X", "https://a.org/x"),  # paths are case sensitive
        ("https://news.a.org/x", "https://a.org/x"),
    ],
)
def test_url_key_keeps_different_pages_apart(a, b):
    assert url_key(a) != url_key(b)


def test_domain_of():
    assert domain_of("https://www.Autocarindia.com/bike-news/x") == "autocarindia.com"
