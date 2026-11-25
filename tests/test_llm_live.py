"""Real calls to Groq and Gemini with the keys in .env. Skipped by default; run with:  uv run pytest -m live

Each provider is tested alone, with no fallback, because a fallback hides a provider that is broken: Groq structured
output failed on every short prompt with tool calling, and the offline tests could not catch it.
"""

import pytest
from pydantic import BaseModel

from app.config import settings
from app.llm.provider import PROVIDERS, get_llm, get_structured_llm, is_configured
from app.llm.usage import track_usage

pytestmark = pytest.mark.live


class Capital(BaseModel):
    country: str
    capital: str


def only(provider: str):
    """Settings that use `provider` with no fallback."""
    if not is_configured(provider):
        pytest.skip(f"{provider.upper()}_API_KEY or {provider.upper()}_MODEL is empty in .env")
    other = next(p for p in PROVIDERS if p != provider)
    return settings.model_copy(update={"llm_provider": provider, f"{other}_api_key": ""})


@pytest.mark.parametrize("provider", PROVIDERS)
def test_plain_reply(provider):
    reply = get_llm(cfg=only(provider)).invoke("Reply with just the word: pong")

    assert "pong" in reply.text.lower()


@pytest.mark.parametrize("provider", PROVIDERS)
def test_structured_reply_to_a_short_prompt(provider):
    with track_usage() as usage:
        answer = get_structured_llm(Capital, cfg=only(provider)).invoke("Capital of India?")

    assert "delhi" in answer.capital.lower()
    assert usage.llm_calls == 1
    assert usage.input_tokens > 0 and usage.output_tokens > 0
