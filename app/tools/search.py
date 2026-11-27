"""T04 (Sabrina): web search for each section's queries, with duplicate URLs removed.

    from app.tools.search import search_queries

    batch = search_queries(section["queries"], max_results=5)
    batch.results         # SearchResult dicts in the order found, one per URL
    batch.search_calls    # queries sent to a search provider (cache hits are free), for RunMetrics "search_calls"
    batch.failed_queries  # queries every provider failed on; results from the other queries are kept

Providers: SEARCH_PROVIDER picks the primary one, ddgs (free, no key) or tavily (needs TAVILY_API_KEY). When
TAVILY_API_KEY is set, the other provider is the fallback, like the LLM providers.

Free search engines push back on automated traffic: in testing DuckDuckGo answered 202 ("slow down"), other engines
answered 403 or timed out, and one ddgs query took 10 to 50 seconds or failed outright. Tavily answered in about
3 seconds. So this module:

- uses the ddgs "auto" backend, which tries several engines; DuckDuckGo alone returned no results at all,
- with a fallback configured, gives the primary one attempt and then moves on; without one, retries with backoff,
- sends ddgs requests one at a time with a gap between them, also when parallel section branches search at once,
- skips a query that every provider failed on, instead of failing the whole section,
- caches results in data/search_cache (SEARCH_CACHE in .env), so re-runs and the v1/v2 evaluation reuse the same
  results. That keeps the comparison fair and saves rate limits and Tavily credits. Delete the folder for fresh
  results.

Duplicates are found with url_key(), which ignores "www.", http vs https, a trailing slash, #fragments and tracking
parameters. The source registry (T07) should use url_key() too, so a page found by two sections gets one source ID.
"""

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from langsmith import traceable

from app.config import Settings, settings

logger = logging.getLogger(__name__)

SearchProvider = Literal["ddgs", "tavily"]

MAX_RETRIES = 2  # without a fallback: up to 3 attempts per query
BACKOFF_SECONDS = 2.0  # wait 2 s, then 4 s, between attempts
MIN_INTERVAL_SECONDS = 1.0  # gap between two ddgs requests
TIMEOUT_SECONDS = 10
TAVILY_URL = "https://api.tavily.com/search"

TRACKING_PARAMS = {"fbclid", "gclid", "msclkid", "ref", "ref_src"}  # plus every utm_* parameter


class SearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    query: str  # the query that found it
    provider: str  # "ddgs" or "tavily"


@dataclass
class SearchBatch:
    results: list[SearchResult] = field(default_factory=list)
    search_calls: int = 0
    failed_queries: list[str] = field(default_factory=list)


class SearchError(RuntimeError):
    """A query failed on every attempt."""


class SearchConfigError(RuntimeError):
    """The chosen search provider is missing its API key in .env."""


@traceable(run_type="tool", name="web_search")
def search_queries(queries: Iterable[str], max_results: int = 5, cfg: Settings = settings) -> SearchBatch:
    """Search every query (blank and repeated queries once) and return the results with duplicate URLs removed."""
    providers = provider_order(cfg)
    batch = SearchBatch()
    seen: set[str] = set()
    for query in dict.fromkeys(q.strip() for q in queries if q.strip()):
        results = _cache_get(query, max_results, cfg)
        if results is None:
            batch.search_calls += 1
            try:
                results = _fetch(query, max_results, providers, cfg)
            except SearchError as exc:
                logger.warning("Search skipped: %s", exc)
                batch.failed_queries.append(query)
                continue
            _cache_put(query, max_results, results, cfg)

        for result in results:
            key = url_key(result["url"])
            if key not in seen:
                seen.add(key)
                batch.results.append(result)
    return batch


def provider_order(cfg: Settings = settings) -> list[SearchProvider]:
    """The primary provider, then the fallback when it is configured."""
    has_tavily_key = bool(cfg.tavily_api_key.strip())
    if cfg.search_provider == "tavily":
        if not has_tavily_key:
            raise SearchConfigError("SEARCH_PROVIDER is tavily, but TAVILY_API_KEY is empty in .env")
        return ["tavily", "ddgs"]
    return ["ddgs", "tavily"] if has_tavily_key else ["ddgs"]


def url_key(url: str) -> str:
    """The same key for URLs that point at the same page, e.g. http://www.a.org/x/?utm_source=t and https://a.org/x."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    params = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not name.lower().startswith("utm_") and name.lower() not in TRACKING_PARAMS
    ]
    query = f"?{urlencode(params)}" if params else ""
    return f"{host}{port}{parts.path.rstrip('/')}{query}"


def domain_of(url: str) -> str:
    """Domain without "www.", e.g. for the source diversity metric."""
    return (urlsplit(url.strip()).hostname or "").lower().removeprefix("www.")


# ---- providers ----


def _ddgs_search(query: str, max_results: int, cfg: Settings) -> list[SearchResult]:
    from ddgs import DDGS

    raw = DDGS(timeout=TIMEOUT_SECONDS).text(query, max_results=max_results, backend="auto")
    return [
        {
            "url": i["href"],
            "title": i.get("title", ""),
            "snippet": i.get("body", ""),
            "query": query,
            "provider": "ddgs",
        }
        for i in raw
        if i.get("href")
    ]


def _tavily_search(query: str, max_results: int, cfg: Settings) -> list[SearchResult]:
    response = httpx.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {cfg.tavily_api_key.strip()}"},
        json={"query": query, "max_results": max_results, "search_depth": "basic"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return [
        {
            "url": item["url"],
            "title": item.get("title") or "",
            "snippet": item.get("content") or "",
            "query": query,
            "provider": "tavily",
        }
        for item in response.json().get("results", [])
        if item.get("url")
    ]


_PROVIDERS: dict[str, Callable[[str, int, Settings], list[SearchResult]]] = {
    "ddgs": _ddgs_search,
    "tavily": _tavily_search,
}


# ---- fallback, retries and spacing ----


def _fetch(query: str, max_results: int, providers: list[SearchProvider], cfg: Settings) -> list[SearchResult]:
    """Results from the first provider that finds any. An empty answer also moves on to the next provider."""
    errors: list[str] = []
    for position, provider in enumerate(providers):
        is_last = position == len(providers) - 1
        try:
            results = _fetch_with_retry(provider, query, max_results, cfg, attempts=MAX_RETRIES + 1 if is_last else 1)
        except SearchError as exc:
            errors.append(str(exc))
            if not is_last:
                logger.warning("%s; using %s", exc, providers[position + 1])
            continue
        if results or is_last:
            return results
    raise SearchError("; ".join(errors))


def _fetch_with_retry(
    provider: SearchProvider, query: str, max_results: int, cfg: Settings, attempts: int
) -> list[SearchResult]:
    fetch = _PROVIDERS[provider]
    error: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
        try:
            with _one_request_at_a_time() if provider == "ddgs" else nullcontext():
                return fetch(query, max_results, cfg)
        except Exception as exc:  # rate limits, timeouts, "no results", HTTP errors
            error = exc
    reason = str(error).replace("\n", " ")[:200]
    tries = f"{attempts} times" if attempts > 1 else "once"
    raise SearchError(f"{provider} failed {tries} on {query!r} ({type(error).__name__}: {reason})") from error


_request_lock = threading.Lock()
_last_request_at = 0.0


@contextmanager
def _one_request_at_a_time() -> Iterator[None]:
    global _last_request_at
    with _request_lock:
        wait = _last_request_at + MIN_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            yield
        finally:
            _last_request_at = time.monotonic()


# ---- cache ----


def _cache_file(query: str, max_results: int, cfg: Settings) -> Path:
    key = f"{max_results}|{query.lower()}"
    return cfg.search_cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"


def _cache_get(query: str, max_results: int, cfg: Settings) -> list[SearchResult] | None:
    if not cfg.search_cache:
        return None
    try:
        return json.loads(_cache_file(query, max_results, cfg).read_text(encoding="utf-8"))["results"]
    except (OSError, ValueError, KeyError):
        return None


def _cache_put(query: str, max_results: int, results: list[SearchResult], cfg: Settings) -> None:
    if not cfg.search_cache or not results:
        return
    path = _cache_file(query, max_results, cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")  # one per thread: sections search in parallel
        tmp.write_text(json.dumps({"query": query, "results": results}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Could not cache search results for %r: %s", query, exc)
