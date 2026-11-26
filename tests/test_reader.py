import io
import json
import logging

import httpx
import pytest
from pypdf import PdfReader, PdfWriter

from app.config import Settings
from app.tools import reader
from app.tools.reader import read_page, read_pages


@pytest.fixture(autouse=True)
def cache_in_tmp(tmp_path, monkeypatch) -> Settings:
    """Every test caches pages in its own temporary folder, never in the real data/page_cache."""
    cfg = Settings(_env_file=None, page_cache=True, page_cache_dir=tmp_path / "pages")
    monkeypatch.setattr(reader, "settings", cfg)
    return cfg


# Paragraphs must differ: trafilatura drops repeated paragraphs as boilerplate.
ARTICLE = "".join(
    f"<p>Paragraph {n}: electric two-wheeler sales in India grew strongly, led by established makers and new "
    f"start-ups, with {n * 7} percent growth in region {n}.</p>"
    for n in range(2)
)
GOOD_HTML = (
    "<html><head><title>EV sales report</title></head>"
    f"<body><article><h1>EV sales</h1>{ARTICLE}"
    + "".join(
        f"<p>More detail {n}: charging networks expanded in {n + 3} more cities this quarter.</p>" for n in range(8)
    )
    + "</article></body></html>"
)


def client_for(routes: dict[str, httpx.Response | Exception]) -> httpx.Client:
    """An httpx client that answers from `routes` instead of the network."""

    def handle(request: httpx.Request) -> httpx.Response:
        outcome = routes[str(request.url)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.Client(transport=httpx.MockTransport(handle))


def html(body: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(status, headers={"content-type": content_type}, content=body.encode())


def test_reads_the_main_text_and_title():
    page = read_page("https://a.org/ev", client_for({"https://a.org/ev": html(GOOD_HTML)}))

    assert page is not None
    assert page["url"] == "https://a.org/ev"
    assert "electric two-wheeler sales in India grew strongly" in page["text"]
    assert "charging networks expanded" in page["text"]
    assert page["title"] == "EV sales"  # trafilatura prefers the page's main heading over <title>


@pytest.mark.parametrize(
    "response",
    [
        html(GOOD_HTML, status=403),
        html(GOOD_HTML, content_type="image/png"),
        html("<html><body><p>Subscribe to read this article.</p></body></html>"),
        httpx.ConnectError("connection closed by the remote host"),
        httpx.ReadTimeout("timed out"),
    ],
    ids=["blocked", "image", "paywall-short-text", "connection-closed", "timeout"],
)
def test_unusable_pages_are_skipped(response):
    assert read_page("https://a.org/x", client_for({"https://a.org/x": response})) is None


def test_pages_larger_than_the_limit_are_skipped(monkeypatch):
    monkeypatch.setattr(reader, "MAX_HTML_BYTES", 1000)

    assert read_page("https://a.org/big", client_for({"https://a.org/big": html(GOOD_HTML)})) is None


def test_read_pages_reads_each_url_once_and_keeps_failures_as_none():
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return html(GOOD_HTML) if "good" in str(request.url) else html("", status=404)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    pages = read_pages(["https://a.org/good", "https://a.org/bad", "https://a.org/good"], client)

    assert sorted(calls) == ["https://a.org/bad", "https://a.org/good"]
    assert pages["https://a.org/good"] is not None
    assert pages["https://a.org/bad"] is None


# ---- cache ----


def counting_client(response: httpx.Response) -> tuple[httpx.Client, list[str]]:
    calls: list[str] = []

    def handle(request):
        calls.append(str(request.url))
        return response

    return httpx.Client(transport=httpx.MockTransport(handle)), calls


def test_a_page_that_was_read_comes_from_the_cache_the_next_time():
    client, calls = counting_client(html(GOOD_HTML))

    first = read_page("https://a.org/ev", client)
    second = read_page("https://a.org/ev", client)

    assert calls == ["https://a.org/ev"]
    assert second == first


def test_failed_reads_are_not_cached():
    client, calls = counting_client(html(GOOD_HTML, status=429))

    read_page("https://a.org/busy", client)
    read_page("https://a.org/busy", client)

    assert len(calls) == 2


def test_cache_can_be_switched_off(cache_in_tmp):
    client, calls = counting_client(html(GOOD_HTML))
    no_cache = cache_in_tmp.model_copy(update={"page_cache": False})

    read_page("https://a.org/ev", client, no_cache)
    read_page("https://a.org/ev", client, no_cache)

    assert len(calls) == 2
    assert not cache_in_tmp.page_cache_dir.exists()


def test_a_damaged_cache_file_is_ignored(cache_in_tmp):
    client, calls = counting_client(html(GOOD_HTML))
    read_page("https://a.org/ev", client)
    next(cache_in_tmp.page_cache_dir.iterdir()).write_text("{not json", encoding="utf-8")

    page = read_page("https://a.org/ev", client)

    assert page is not None
    assert len(calls) == 2


@pytest.mark.parametrize(
    "response",
    [html(GOOD_HTML, status=404), html(GOOD_HTML, status=403), html(GOOD_HTML, content_type="image/png")],
    ids=["not-found", "blocked", "image"],
)
def test_pages_that_can_never_be_used_are_not_fetched_again(response):
    client, calls = counting_client(response)

    assert read_page("https://a.org/x", client) is None
    assert read_page("https://a.org/x", client) is None
    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_temporary_failures_are_tried_again(status):
    client, calls = counting_client(html(GOOD_HTML, status=status))

    read_page("https://a.org/x", client)
    read_page("https://a.org/x", client)

    assert len(calls) == 2


def test_timeouts_are_tried_again():
    calls = []

    def handle(request):
        calls.append(1)
        raise httpx.ReadTimeout("timed out")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    read_page("https://a.org/slow", client)
    read_page("https://a.org/slow", client)

    assert len(calls) == 2


# ---- PDFs ----


def make_pdf(lines: list[str], title: str = "") -> bytes:
    """A small, valid one-page PDF with `lines` of text (Helvetica) and an optional title in its metadata."""

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")

    stream = "BT /F1 10 Tf 40 780 Td 13 TL " + " ".join(f"({escape(line)}) '" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Title ({escape(title)}) >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


PDF_LINES = [
    "Rise of Quick Commerce in India: Business Models and Infrastructure.",
    "Quick commerce orders grew strongly as dark stores spread across large cities,",
    "and the leading platforms raised fresh funding to expand into smaller towns.",
    "Demand for groceries and daily essentials drove most orders, while electri-",
    "city and logistics costs remained a large share of operating expenses.",
] + [f"Finding {n}: delivery times stayed under fifteen minutes in {n + 4} major metro areas." for n in range(6)]


def pdf_response(body: bytes, content_type: str = "application/pdf") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": content_type}, content=body)


def test_reads_a_pdf_and_joins_its_lines_into_paragraphs():
    body = make_pdf(PDF_LINES, title="Rise of Quick Commerce in India")

    page = read_page("https://iima.ac.in/q-com.pdf", client_for({"https://iima.ac.in/q-com.pdf": pdf_response(body)}))

    assert page is not None
    assert page["title"] == "Rise of Quick Commerce in India"
    assert "dark stores spread across large cities, and the leading platforms" in page["text"]  # lines joined
    assert "electricity and logistics costs" in page["text"]  # word split by a hyphen put back together


def test_a_pdf_sent_as_a_generic_download_is_read_too():
    url = "https://gov.in/report.pdf"
    body = make_pdf(PDF_LINES)

    page = read_page(url, client_for({url: pdf_response(body, "application/octet-stream")}))

    assert page is not None and "Quick commerce orders" in page["text"]


def test_a_title_that_is_only_a_file_name_is_dropped():
    url = "https://iima.ac.in/q.pdf"
    body = make_pdf(PDF_LINES, title="Q-com - Ranjekar & Roy.docx")

    assert read_page(url, client_for({url: pdf_response(body)}))["title"] == ""


def test_broken_and_oversized_pdfs_are_skipped(monkeypatch):
    broken = client_for({"https://a.org/broken.pdf": pdf_response(b"%PDF-1.4 not really a pdf")})
    assert read_page("https://a.org/broken.pdf", broken) is None

    monkeypatch.setattr(reader, "MAX_PDF_BYTES", 500)
    big = client_for({"https://a.org/big.pdf": pdf_response(make_pdf(PDF_LINES))})
    assert read_page("https://a.org/big.pdf", big) is None


def test_tidy_pdf_text():
    raw = "Rise\nof\nQuick\nCommerce\n\nelectri-\ncity  costs\n\n\n"

    assert reader.tidy_pdf_text(raw) == "Rise of Quick Commerce\nelectricity costs"


# ---- who we say we are ----


def test_user_agent_names_the_project_and_contact_when_set():
    cfg = Settings(_env_file=None, reader_contact="team@example.org")

    assert reader.user_agent(cfg) == "ResearchPilot/0.1 (research assistant; +team@example.org)"
    with reader.new_client(cfg) as client:
        assert client.headers["User-Agent"] == reader.user_agent(cfg)


def test_user_agent_falls_back_to_a_browser_when_no_contact_is_set():
    assert reader.user_agent(Settings(_env_file=None, reader_contact="  ")) == reader.BROWSER_USER_AGENT


def test_a_cached_skip_is_retried_once_the_user_agent_changes(cache_in_tmp):
    client, calls = counting_client(html(GOOD_HTML, status=403))
    read_page("https://wiki.org/x", client)  # blocked without a contact: cached as skipped
    read_page("https://wiki.org/x", client)
    assert len(calls) == 1

    with_contact = cache_in_tmp.model_copy(update={"reader_contact": "team@example.org"})
    read_page("https://wiki.org/x", client, with_contact)

    assert len(calls) == 2


def test_a_cached_skip_from_an_older_reader_is_retried(monkeypatch):
    client, calls = counting_client(html(GOOD_HTML, content_type="image/png"))
    read_page("https://a.org/x", client)

    monkeypatch.setattr(reader, "READER_VERSION", reader.READER_VERSION + 1)
    read_page("https://a.org/x", client)

    assert len(calls) == 2


def test_a_page_that_was_read_stays_cached_when_the_user_agent_changes(cache_in_tmp):
    client, calls = counting_client(html(GOOD_HTML))
    read_page("https://a.org/ev", client)

    read_page("https://a.org/ev", client, cache_in_tmp.model_copy(update={"reader_contact": "team@example.org"}))

    assert len(calls) == 1


# ---- T13: the reader's own HTTP client ----


def test_new_client_settings():
    with reader.new_client(Settings(_env_file=None, reader_contact="")) as client:
        assert client.follow_redirects
        assert client.timeout.read == reader.TIMEOUT_SECONDS
        assert client.timeout.connect == reader.CONNECT_TIMEOUT_SECONDS
        assert "application/pdf" in client.headers["Accept"]
        assert "text/html" in client.headers["Accept"]


def test_read_page_without_a_client_opens_and_closes_its_own(monkeypatch):
    own = client_for({"https://a.org/ev": html(GOOD_HTML)})
    monkeypatch.setattr(reader, "new_client", lambda cfg=None: own)

    assert read_page("https://a.org/ev") is not None
    assert own.is_closed


def test_read_pages_without_a_client_shares_one_client(monkeypatch):
    made = []

    def new_client(cfg=None):
        made.append(1)
        return client_for({"https://a.org/1": html(GOOD_HTML), "https://a.org/2": html(GOOD_HTML)})

    monkeypatch.setattr(reader, "new_client", new_client)

    pages = read_pages(["https://a.org/1", "https://a.org/2"])

    assert len(made) == 1
    assert all(pages.values())


def test_read_pages_with_no_urls_opens_no_client(monkeypatch):
    def no_client(cfg=None):
        raise AssertionError("no client is needed for an empty list")

    monkeypatch.setattr(reader, "new_client", no_client)

    assert read_pages([]) == {}


class RecordingPool:
    """Stands in for ThreadPoolExecutor: records the pool size it was given and runs the work in this thread."""

    sizes: list[int] = []

    def __init__(self, max_workers: int) -> None:
        RecordingPool.sizes.append(max_workers)

    def __enter__(self) -> "RecordingPool":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def map(self, fn, items):
        return [fn(item) for item in items]


@pytest.mark.parametrize(("url_count", "pool_size"), [(6, 2), (1, 1)])
def test_read_pages_reads_at_most_max_workers_pages_at_once(monkeypatch, url_count, pool_size):
    monkeypatch.setattr(reader, "MAX_WORKERS", 2)
    monkeypatch.setattr(reader, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(RecordingPool, "sizes", [])
    urls = [f"https://a.org/{n}" for n in range(url_count)]

    pages = read_pages(urls, client_for({url: html(GOOD_HTML) for url in urls}))

    assert RecordingPool.sizes == [pool_size]  # never more threads than MAX_WORKERS, nor than pages to read
    assert list(pages) == urls and all(pages.values())


def test_a_redirected_page_is_cited_by_the_url_that_was_asked_for():
    def handle(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://a.org/new"})
        return html(GOOD_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True)

    page = read_page("https://a.org/old", client)

    assert page is not None and page["url"] == "https://a.org/old"


# ---- T13: content types ----


def test_a_response_without_a_content_type_is_skipped_for_good():
    client, calls = counting_client(httpx.Response(200, content=GOOD_HTML.encode()))

    assert read_page("https://a.org/x", client) is None
    assert read_page("https://a.org/x", client) is None
    assert len(calls) == 1


def test_a_generic_download_that_is_not_a_pdf_url_is_skipped():
    url = "https://a.org/file.bin"

    assert read_page(url, client_for({url: pdf_response(make_pdf(PDF_LINES), "application/octet-stream")})) is None


def test_a_pdf_url_with_a_query_string_is_read_as_a_pdf():
    url = "https://gov.in/report.PDF?download=1"

    page = read_page(url, client_for({url: pdf_response(make_pdf(PDF_LINES), "application/octet-stream")}))

    assert page is not None and "Quick commerce orders" in page["text"]


def test_a_page_whose_extraction_crashes_is_skipped_for_good(monkeypatch):
    def crash(*args, **kwargs):
        raise ValueError("broken markup")

    monkeypatch.setattr(reader.trafilatura, "bare_extraction", crash)
    client, calls = counting_client(html(GOOD_HTML))

    assert read_page("https://a.org/x", client) is None
    assert read_page("https://a.org/x", client) is None
    assert len(calls) == 1


def test_an_html_title_that_is_only_a_file_name_is_dropped(monkeypatch):
    monkeypatch.setattr(
        reader.trafilatura,
        "bare_extraction",
        lambda *args, **kwargs: type("Doc", (), {"title": "Microsoft Word - final report", "text": "x" * 600})(),
    )

    page = read_page("https://a.org/x", client_for({"https://a.org/x": html(GOOD_HTML)}))

    assert page is not None and page["title"] == ""


# ---- T13: more PDFs ----


def encrypted(body: bytes, user_password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(body)))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_a_password_protected_pdf_is_skipped_for_good():
    client, calls = counting_client(pdf_response(encrypted(make_pdf(PDF_LINES), "secret")))

    assert read_page("https://a.org/locked.pdf", client) is None
    assert read_page("https://a.org/locked.pdf", client) is None
    assert len(calls) == 1


def test_an_encrypted_pdf_without_a_password_is_read():
    url = "https://a.org/open.pdf"

    page = read_page(url, client_for({url: pdf_response(encrypted(make_pdf(PDF_LINES), ""))}))

    assert page is not None and "Quick commerce orders" in page["text"]


def test_only_the_first_max_pdf_pages_are_read(monkeypatch):
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(make_pdf(PDF_LINES))))
    writer.append(PdfReader(io.BytesIO(make_pdf(["Appendix: the last page has no findings worth citing."] * 12))))
    body = io.BytesIO()
    writer.write(body)
    monkeypatch.setattr(reader, "MAX_PDF_PAGES", 1)

    page = read_page("https://a.org/two.pdf", client_for({"https://a.org/two.pdf": pdf_response(body.getvalue())}))

    assert page is not None
    assert "Quick commerce orders" in page["text"]
    assert "Appendix" not in page["text"]


@pytest.mark.parametrize(
    ("raw", "tidy"),
    [
        ("well-\nknown", "wellknown"),
        ("pre- \npost", "pre- post"),  # a hyphen followed by a space is a dash, not a split word
        ("", ""),
        ("  \n\n  \n", ""),
    ],
)
def test_tidy_pdf_text_edge_cases(raw, tidy):
    assert reader.tidy_pdf_text(raw) == tidy


# ---- T13: cache files ----


def test_a_cache_entry_for_another_url_is_ignored(cache_in_tmp):
    client, calls = counting_client(html(GOOD_HTML))
    path = reader._cache_file("https://a.org/ev", cache_in_tmp)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"url": "https://other.org/", "page": {}}), encoding="utf-8")

    page = read_page("https://a.org/ev", client)

    assert page is not None
    assert len(calls) == 1


def test_a_skip_is_cached_with_its_reason(cache_in_tmp):
    client, _ = counting_client(html(GOOD_HTML, status=404))
    read_page("https://a.org/gone", client)

    entry = json.loads(reader._cache_file("https://a.org/gone", cache_in_tmp).read_text(encoding="utf-8"))

    assert entry["page"] == {}
    assert entry["skipped"] == "HTTP 404"
    assert entry["skipped_by"] == reader._skip_context(cache_in_tmp)


def test_a_failed_cache_write_does_not_lose_the_page(cache_in_tmp, caplog):
    cache_in_tmp.page_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_in_tmp.page_cache_dir.write_text("a file where the cache folder should be", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="app.tools.reader"):
        page = read_page("https://a.org/ev", client_for({"https://a.org/ev": html(GOOD_HTML)}))

    assert page is not None
    assert "Could not cache page" in caplog.text


def test_no_temporary_files_are_left_in_the_cache(cache_in_tmp):
    read_pages(
        [f"https://a.org/{n}" for n in range(4)],
        client_for({f"https://a.org/{n}": html(GOOD_HTML) for n in range(4)}),
    )

    assert sorted(p.suffix for p in cache_in_tmp.page_cache_dir.iterdir()) == [".json"] * 4
