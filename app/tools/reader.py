"""T08 (Sabrina): fetch web pages and PDFs and return their clean main text.

    from app.tools.reader import read_page, read_pages

    page = read_page("https://example.org/article")   # Page(url, title, text) or None
    pages = read_pages(urls)                           # {url: Page or None}, several pages at the same time

HTML goes through trafilatura (main text, no menus or ads). PDFs, often the best sources (research papers, government
reports), go through pypdf: the first MAX_PDF_PAGES pages, with lines joined into paragraphs and words split by a
hyphen at a line end put back together (see tidy_pdf_text for how "electri-/city" and "two-/wheeler" are told apart).

A page is skipped (None) when it cannot be fetched (timeouts, 403 blocks, closed connections), is neither HTML nor
PDF, is too large, or has less than MIN_CHARS of text. In testing, short texts came from paywalls and pages that hide
their numbers (e.g. "close to million units"), so they are not worth citing. A site that blocks us (403, 429) is
skipped, never worked around.

Who we are: with READER_CONTACT set in .env, every request says so in its User-Agent ("ResearchPilot/0.1 (research
assistant; +contact)"). Wikipedia, for one, answers 403 to clients that do not give contact details (its robot
policy). Without READER_CONTACT, the reader uses a common browser User-Agent and Wikipedia pages are skipped.

Results are cached in data/page_cache (PAGE_CACHE in .env), so a re-run, and above all the v1 and v2 evaluation
runs, read exactly the same text; a live page can change between two runs. Pages that can never be used (neither HTML
nor PDF, 404, blocked with 403, too short) are cached as skipped, so they are not fetched again. Temporary failures
(timeouts, closed connections, 429 rate limits, 5xx server errors) are not cached and are tried again next time.
Delete the folder to read pages afresh.

Measured on 153 real search results before PDF support: 73% read in 24 to 33 seconds (8 at a time). Skipped: 22 PDFs,
11 blocked (403), 2 not found, 2 paywalled, 4 connection problems or timeouts, 1 rate limited. Run again with the
cache, only the 4 temporary failures went to the network.
"""

import hashlib
import io
import json
import logging
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import httpx
from pypdf import PdfReader

from app.config import Settings, settings

# urllib3 warns about SOCKS proxies on import of trafilatura's dependencies; we never use one.
warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
import trafilatura  # noqa: E402

logging.getLogger("pypdf").setLevel(logging.ERROR)  # pypdf warns about every slightly broken PDF
logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15  # to read the page
CONNECT_TIMEOUT_SECONDS = 5  # to reach the server; one that cannot connect in 5 s rarely does in 15
MAX_HTML_BYTES = 5_000_000
MAX_PDF_BYTES = 20_000_000  # reports are larger; a 5 MB research paper turned up in testing
MAX_PDF_PAGES = 40  # the summary and main findings come first
MIN_CHARS = 500
MAX_WORKERS = 8
READER_VERSION = 2  # raise when the reader learns to read pages it used to skip (2: PDFs)
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
# Titles that are only a file name, e.g. "Q-com - Ranjekar & Roy.docx" (seen in testing)
FILE_NAME_TITLE = re.compile(r"(\.(docx?|pdf|pptx?|xlsx?)\s*$)|(^microsoft (word|powerpoint) -)", re.IGNORECASE)
LINE_END_HYPHEN = re.compile(r"(\w+)-\n(\w+)")  # "electri-\ncity", "two-\nwheeler"
WORD_CHARS = re.compile(r"\w+")


class Page(TypedDict):
    url: str  # the URL that was asked for, which is the one to cite
    title: str
    text: str


def read_page(url: str, client: httpx.Client | None = None, cfg: Settings | None = None) -> Page | None:
    """The page's main text, or None when it cannot be used (see the module docstring)."""
    cfg = cfg or settings
    cached = _cache_get(url, cfg)
    if cached is not None:
        return cached or None  # an empty dict marks a page that can never be used
    if client is None:
        with new_client(cfg) as own_client:
            return read_page(url, own_client, cfg)

    try:
        page = _read(url, client)
    except Skip as skip:
        logger.info("Skipped %s: %s", url, skip)
        if skip.permanent:
            _cache_put(url, {}, cfg, skipped=str(skip))
        return None
    _cache_put(url, page, cfg)
    return page


def read_pages(
    urls: list[str], client: httpx.Client | None = None, cfg: Settings | None = None
) -> dict[str, Page | None]:
    """read_page() for every URL (each once), MAX_WORKERS at a time."""
    cfg = cfg or settings
    unique = list(dict.fromkeys(urls))
    if not unique:
        return {}
    if client is None:
        with new_client(cfg) as own_client:
            return read_pages(unique, own_client, cfg)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(unique))) as pool:
        return dict(zip(unique, pool.map(lambda url: read_page(url, client, cfg), unique), strict=True))


def user_agent(cfg: Settings) -> str:
    """Our own name and contact when READER_CONTACT is set, otherwise a common browser's."""
    contact = cfg.reader_contact.strip()
    return f"ResearchPilot/0.1 (research assistant; +{contact})" if contact else BROWSER_USER_AGENT


def new_client(cfg: Settings | None = None) -> httpx.Client:
    headers = {
        "User-Agent": user_agent(cfg or settings),
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    timeout = httpx.Timeout(TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    return httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)


class Skip(Exception):
    """Why a page cannot be used. Permanent reasons are cached; temporary ones (timeouts, rate limits, server
    errors) are not, so the page is tried again next time."""

    def __init__(self, reason: str, permanent: bool) -> None:
        super().__init__(reason)
        self.permanent = permanent


PERMANENT_HTTP_STATUS = {400, 401, 402, 403, 404, 410, 451}  # 429 and 5xx are usually temporary


def _read(url: str, client: httpx.Client) -> Page:
    body = _fetch(url, client)
    title, text = _pdf_text(body) if body.startswith(b"%PDF") else _html_text(body, url)
    text = text.strip()
    if len(text) < MIN_CHARS:
        raise Skip(f"only {len(text)} characters of text", permanent=True)
    if FILE_NAME_TITLE.search(title):
        title = ""
    return {"url": url, "title": title.strip(), "text": text}


def _html_text(body: bytes, url: str) -> tuple[str, str]:
    try:
        document = trafilatura.bare_extraction(body, url=url, include_tables=True, with_metadata=True)
    except Exception as exc:  # trafilatura can fail on broken markup
        raise Skip(f"extraction failed ({exc})", permanent=True) from exc
    return getattr(document, "title", None) or "", getattr(document, "text", None) or ""


def _pdf_text(body: bytes) -> tuple[str, str]:
    try:
        pdf = PdfReader(io.BytesIO(body))
        if pdf.is_encrypted and not pdf.decrypt(""):
            raise Skip("PDF is password protected", permanent=True)
        pages = [page.extract_text() or "" for page in pdf.pages[:MAX_PDF_PAGES]]
        title = (pdf.metadata.title if pdf.metadata else None) or ""
    except Skip:
        raise
    except Exception as exc:  # pypdf raises many kinds of errors on broken files
        raise Skip(f"PDF extraction failed ({type(exc).__name__})", permanent=True) from exc
    return title, tidy_pdf_text("\n\n".join(pages))


def tidy_pdf_text(text: str) -> str:
    """PDF text as one paragraph per line: single line breaks (pypdf breaks every printed line, sometimes after every
    word) become spaces, and blank lines separate paragraphs.

    A hyphen at a line end is either a word split to fit the line ("electri-/city") or a real hyphen that happened to
    fall there ("two-/wheeler", "well-/known"). Without a dictionary the two look the same, so the document decides:
    the parts are joined only when the joined word appears elsewhere in the same text ("electricity"). Otherwise the
    hyphen is kept, because a kept hyphen loses nothing that was printed, while a wrong join ("twowheeler") breaks
    the word for passage matching.
    """
    words = set(WORD_CHARS.findall(text.lower()))

    def rejoin(match: re.Match[str]) -> str:
        left, right = match.groups()
        return left + right if (left + right).lower() in words else f"{left}-{right}"

    text = LINE_END_HYPHEN.sub(rejoin, text)
    paragraphs = (" ".join(block.split()) for block in re.split(r"\n\s*\n", text))
    return "\n".join(p for p in paragraphs if p)


def _fetch(url: str, client: httpx.Client) -> bytes:
    """The body of an HTML page or a PDF."""
    try:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise Skip(f"HTTP {response.status_code}", response.status_code in PERMANENT_HTTP_STATUS)
            content_type = response.headers.get("content-type", "").lower()
            is_pdf = "pdf" in content_type or (
                "octet-stream" in content_type and url.lower().split("?")[0].endswith(".pdf")
            )
            if not is_pdf and "html" not in content_type:
                raise Skip(f"neither HTML nor PDF ({content_type or 'no content type'})", permanent=True)
            limit = MAX_PDF_BYTES if is_pdf else MAX_HTML_BYTES
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > limit:
                    raise Skip(f"larger than {limit} bytes", permanent=True)
            return bytes(body)
    except httpx.HTTPError as exc:  # timeouts, closed connections: try again next time
        raise Skip(type(exc).__name__, permanent=False) from exc


# ---- cache ----


def _cache_file(url: str, cfg: Settings) -> Path:
    return cfg.page_cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:32]}.json"


def _skip_context(cfg: Settings) -> str:
    """What a cached skip depends on. A skip made by an older reader (e.g. before PDF support) or with another
    User-Agent (e.g. before READER_CONTACT was set, when Wikipedia answered 403) is not trusted: the page is tried
    again. Pages that were read stay valid either way."""
    return f"reader {READER_VERSION} | {user_agent(cfg)}"


def _cache_get(url: str, cfg: Settings) -> Page | dict | None:
    """The cached page, {} for a page known to be unusable, or None when the page is not in the cache."""
    if not cfg.page_cache:
        return None
    try:
        entry = json.loads(_cache_file(url, cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("url") != url or not isinstance(entry.get("page"), dict):
        return None
    if not entry["page"] and entry.get("skipped_by") != _skip_context(cfg):
        return None  # skipped under other conditions: try again
    return entry["page"]


def _cache_put(url: str, page: Page | dict, cfg: Settings, skipped: str = "") -> None:
    if not cfg.page_cache:
        return
    entry = {"url": url, "page": page}
    if skipped:
        entry |= {"skipped": skipped, "skipped_by": _skip_context(cfg)}
    path = _cache_file(url, cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")  # one per thread: several threads may cache at once
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Could not cache page %s: %s", url, exc)
