import httpx
import pytest

from app.tools import reader
from app.tools.reader import read_page, read_pages

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
        html(GOOD_HTML, content_type="application/pdf"),
        html("<html><body><p>Subscribe to read this article.</p></body></html>"),
        httpx.ConnectError("connection closed by the remote host"),
        httpx.ReadTimeout("timed out"),
    ],
    ids=["blocked", "pdf", "paywall-short-text", "connection-closed", "timeout"],
)
def test_unusable_pages_are_skipped(response):
    assert read_page("https://a.org/x", client_for({"https://a.org/x": response})) is None


def test_pages_larger_than_the_limit_are_skipped(monkeypatch):
    monkeypatch.setattr(reader, "MAX_BYTES", 1000)

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
