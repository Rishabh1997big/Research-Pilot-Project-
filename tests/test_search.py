import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

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
    base = Settings(_env_file=None, search_provider="ddgs", tavily_api_key="")

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


# ---- T13: more URL keys ----


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://a.org/x?ref=home", "https://a.org/x"),
        ("https://a.org/x?UTM_Source=t&gclid=9&msclkid=8&ref_src=tw", "https://a.org/x"),
        ("  https://a.org/x  ", "https://a.org/x"),
        ("http://a.org:80/x", "https://a.org/x"),
    ],
)
def test_url_key_ignores_tracking_parameters_spaces_and_default_ports(a, b):
    assert url_key(a) == url_key(b)


def test_url_key_keeps_a_non_default_port():
    assert url_key("https://a.org:8080/x") != url_key("https://a.org/x")


@pytest.mark.parametrize(
    ("url", "domain"),
    [("https://sub.A.org:8080/x", "sub.a.org"), ("not a url", ""), ("", "")],
)
def test_domain_of_edge_cases(url, domain):
    assert domain_of(url) == domain


# ---- T13: provider mapping ----


def test_ddgs_results_are_mapped_and_results_without_a_link_dropped(monkeypatch, cfg):
    sent = {}

    class FakeDDGS:
        def __init__(self, timeout):
            sent["timeout"] = timeout

        def text(self, query, max_results, backend):
            sent.update(query=query, max_results=max_results, backend=backend)
            return [
                {"href": "https://a.org/x", "title": "A", "body": "Snippet"},
                {"title": "no link"},
                {"href": "https://b.org/y"},
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    results = search._ddgs_search("ev sales", 4, cfg)

    assert sent == {"timeout": search.TIMEOUT_SECONDS, "query": "ev sales", "max_results": 4, "backend": "auto"}
    assert results == [
        {"url": "https://a.org/x", "title": "A", "snippet": "Snippet", "query": "ev sales", "provider": "ddgs"},
        {"url": "https://b.org/y", "title": "", "snippet": "", "query": "ev sales", "provider": "ddgs"},
    ]


def test_tavily_missing_fields_and_results_without_a_url(monkeypatch, with_tavily):
    def fake_post(url, headers, json, timeout):
        payload = {
            "results": [
                {"url": "", "title": "no url"},
                {"title": "missing url"},
                {"url": "https://a.org/x", "title": None, "content": None},
            ]
        }
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(search.httpx, "post", fake_post)

    assert search._tavily_search("q", 5, with_tavily) == [
        {"url": "https://a.org/x", "title": "", "snippet": "", "query": "q", "provider": "tavily"}
    ]


def test_tavily_answer_without_results_is_empty(monkeypatch, with_tavily):
    monkeypatch.setattr(
        search.httpx, "post", lambda url, **kwargs: httpx.Response(200, json={}, request=httpx.Request("POST", url))
    )

    assert search._tavily_search("q", 5, with_tavily) == []


def test_tavily_http_error_is_raised_for_the_fallback_to_handle(monkeypatch, with_tavily):
    monkeypatch.setattr(
        search.httpx, "post", lambda url, **kwargs: httpx.Response(401, request=httpx.Request("POST", url))
    )

    with pytest.raises(httpx.HTTPStatusError):
        search._tavily_search("q", 5, with_tavily)


# ---- T13: Tavily as the primary provider ----


def test_ddgs_takes_over_when_tavily_is_primary_and_fails(monkeypatch, with_tavily):
    tavily_calls = fake_provider(monkeypatch, "tavily", {"q": RuntimeError("432 credits used up")})
    ddgs_calls = fake_provider(monkeypatch, "ddgs", {"q": ["https://a.org"]})

    batch = search_queries(["q"], cfg=with_tavily.model_copy(update={"search_provider": "tavily"}))

    assert tavily_calls == ["q"]
    assert ddgs_calls == ["q"]
    assert batch.results[0]["provider"] == "ddgs"


def test_max_results_reaches_the_provider(monkeypatch, cfg):
    fake_provider(monkeypatch, "ddgs", {"q": [f"https://a.org/{n}" for n in range(10)]})

    assert len(search_queries(["q"], max_results=3, cfg=cfg).results) == 3


def test_a_skipped_query_is_logged_with_provider_attempts_and_reason(monkeypatch, cfg, caplog):
    fake_provider(monkeypatch, "ddgs", {"bad": RuntimeError("202 Ratelimit")})

    with caplog.at_level(logging.WARNING, logger="app.tools.search"):
        search_queries(["bad"], cfg=cfg)

    assert "ddgs failed 3 times on 'bad' (RuntimeError: 202 Ratelimit)" in caplog.text


# ---- T13: cache keys ----


def test_cache_ignores_the_case_of_the_query(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"EV Sales": ["https://a.org"], "ev sales": ["https://b.org"]})

    search_queries(["EV Sales"], cfg=cfg)
    batch = search_queries(["ev sales"], cfg=cfg)

    assert calls == ["EV Sales"]
    assert batch.search_calls == 0


def test_cache_is_kept_apart_per_max_results(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"q": ["https://a.org", "https://b.org"]})

    search_queries(["q"], max_results=1, cfg=cfg)
    batch = search_queries(["q"], max_results=2, cfg=cfg)

    assert calls == ["q", "q"]
    assert len(batch.results) == 2


def test_a_damaged_cache_file_is_ignored(monkeypatch, cfg):
    calls = fake_provider(monkeypatch, "ddgs", {"q": ["https://a.org"]})
    search_queries(["q"], cfg=cfg)
    next(cfg.search_cache_dir.iterdir()).write_text("{not json", encoding="utf-8")

    batch = search_queries(["q"], cfg=cfg)

    assert calls == ["q", "q"]
    assert [r["url"] for r in batch.results] == ["https://a.org"]


def test_a_failed_cache_write_does_not_lose_the_results(monkeypatch, cfg, caplog):
    fake_provider(monkeypatch, "ddgs", {"q": ["https://a.org"]})
    cfg.search_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.search_cache_dir.write_text("a file where the cache folder should be", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="app.tools.search"):
        batch = search_queries(["q"], cfg=cfg)

    assert [r["url"] for r in batch.results] == ["https://a.org"]
    assert "Could not cache search results" in caplog.text


def test_two_threads_caching_the_same_query_use_their_own_temporary_files(monkeypatch, cfg, caplog):
    """Two sections that search the same query at the same moment both write its cache file."""
    both_writing = threading.Barrier(2, timeout=5)
    temporary_files: list[str] = []
    write_text = Path.write_text

    def write_text_together(self, *args, **kwargs):
        temporary_files.append(self.name)
        both_writing.wait()  # neither thread moves its file into place before the other has written
        return write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text_together)
    results = [{"url": "https://a.org", "title": "", "snippet": "", "query": "q", "provider": "ddgs"}]

    with caplog.at_level(logging.WARNING, logger="app.tools.search"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(search._cache_put, "q", 5, results, cfg) for _ in range(2)]
            for future in futures:
                future.result(timeout=10)  # re-raises any error from a worker thread

    assert len(set(temporary_files)) == 2
    assert "Could not cache" not in caplog.text
    assert [p.suffix for p in cfg.search_cache_dir.iterdir()] == [".json"]  # no temporary file left behind
    assert search._cache_get("q", 5, cfg) == results


# ---- T13: spacing, backoff and one ddgs request at a time ----


class FakeClock:
    """Stands in for the time module: sleeping moves the clock forward instead of waiting."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(search, "time", fake)
    monkeypatch.setattr(search, "_last_request_at", 0.0)
    return fake


def test_ddgs_requests_are_spaced_out(monkeypatch, cfg, clock):
    monkeypatch.setattr(search, "MIN_INTERVAL_SECONDS", 1.0)
    fake_provider(monkeypatch, "ddgs", {"q1": ["https://a.org"], "q2": ["https://b.org"], "q3": ["https://c.org"]})

    search_queries(["q1", "q2", "q3"], cfg=cfg)

    assert clock.sleeps == [1.0, 1.0]  # none before the first request


def test_tavily_requests_are_not_spaced_out(monkeypatch, with_tavily, clock):
    monkeypatch.setattr(search, "MIN_INTERVAL_SECONDS", 1.0)
    fake_provider(monkeypatch, "tavily", {"q1": ["https://a.org"], "q2": ["https://b.org"]})

    search_queries(["q1", "q2"], cfg=with_tavily.model_copy(update={"search_provider": "tavily"}))

    assert clock.sleeps == []


def test_retries_back_off_exponentially(monkeypatch, cfg, clock):
    monkeypatch.setattr(search, "BACKOFF_SECONDS", 2.0)
    monkeypatch.setattr(search, "MAX_RETRIES", 3)  # a fourth attempt tells doubling apart from adding
    fake_provider(monkeypatch, "ddgs", {"bad": RuntimeError("timeout")})

    search_queries(["bad"], cfg=cfg)

    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_parallel_sections_send_ddgs_requests_one_at_a_time(monkeypatch, cfg):
    held_lock: list[bool] = []

    def fetch(query, max_results, cfg):
        # Every ddgs request must run while it holds the module's request lock, so two never overlap.
        held_lock.append(search._request_lock.locked())
        return [{"url": f"https://a.org/{query}", "title": "", "snippet": "", "query": query, "provider": "ddgs"}]

    monkeypatch.setitem(search._PROVIDERS, "ddgs", fetch)
    queries = [f"q{n}" for n in range(4)]

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [pool.submit(search_queries, [query], cfg=cfg) for query in queries]
        batches = [future.result(timeout=10) for future in futures]  # re-raises any error from a worker thread

    assert held_lock == [True] * len(queries)
    assert [batch.results[0]["url"] for batch in batches] == [f"https://a.org/{query}" for query in queries]
