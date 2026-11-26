import httpx
import pytest

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
