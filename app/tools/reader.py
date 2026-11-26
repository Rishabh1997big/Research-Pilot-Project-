"""T08 (Sabrina): fetch web pages and return their clean main text.

A basic version, written with T07 so the research graph can read real pages; Sabrina owns T08 and its tests (T13).

    from app.tools.reader import read_page, read_pages

    page = read_page("https://example.org/article")   # Page(url, title, text) or None
    pages = read_pages(urls)                           # {url: Page or None}, several pages at the same time

A page is skipped (None) when it cannot be fetched (timeouts, 403 blocks, closed connections), is not HTML (PDFs,
images), is larger than MAX_BYTES, or has less than MIN_CHARS of main text. In testing, short texts came from
paywalls and pages that hide their numbers (e.g. "close to million units"), so they are not worth citing.

Known gap: Wikipedia answers 403 to every client whose User-Agent has no contact details (its robot policy), so
Wikipedia pages are skipped for now. Fixing it needs a User-Agent with a contact address for the project.
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

import httpx

# urllib3 warns about SOCKS proxies on import of trafilatura's dependencies; we never use one.
warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
import trafilatura  # noqa: E402

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15
MAX_BYTES = 5_000_000
MIN_CHARS = 500
MAX_WORKERS = 8
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class Page(TypedDict):
    url: str  # the URL that was asked for, which is the one to cite
    title: str
    text: str


def read_page(url: str, client: httpx.Client | None = None) -> Page | None:
    """The page's main text, or None when it cannot be used (see the module docstring)."""
    if client is None:
        with _new_client() as own_client:
            return read_page(url, own_client)

    body = _fetch_html(url, client)
    if body is None:
        return None
    try:
        document = trafilatura.bare_extraction(body, url=url, include_tables=True, with_metadata=True)
    except Exception as exc:  # trafilatura can fail on broken markup
        logger.info("Skipped %s: extraction failed (%s)", url, exc)
        return None
    text = (getattr(document, "text", None) or "").strip()
    if len(text) < MIN_CHARS:
        logger.info("Skipped %s: only %d characters of text", url, len(text))
        return None
    return {"url": url, "title": (getattr(document, "title", None) or "").strip(), "text": text}


def read_pages(urls: list[str], client: httpx.Client | None = None) -> dict[str, Page | None]:
    """read_page() for every URL (each once), MAX_WORKERS at a time."""
    unique = list(dict.fromkeys(urls))
    if not unique:
        return {}
    if client is None:
        with _new_client() as own_client:
            return read_pages(unique, own_client)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(unique))) as pool:
        return dict(zip(unique, pool.map(lambda url: read_page(url, client), unique), strict=True))


def _new_client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=TIMEOUT_SECONDS, follow_redirects=True)


def _fetch_html(url: str, client: httpx.Client) -> bytes | None:
    try:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                logger.info("Skipped %s: HTTP %d", url, response.status_code)
                return None
            content_type = response.headers.get("content-type", "").lower()
            if "html" not in content_type:
                logger.info("Skipped %s: not HTML (%s)", url, content_type or "no content type")
                return None
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    logger.info("Skipped %s: larger than %d bytes", url, MAX_BYTES)
                    return None
            return bytes(body)
    except httpx.HTTPError as exc:
        logger.info("Skipped %s: %s", url, type(exc).__name__)
        return None
