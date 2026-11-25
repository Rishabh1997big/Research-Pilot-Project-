import logging

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import Settings
from app.llm.provider import LLMConfigError, get_chat_model, get_llm, get_structured_llm, with_fallback
from app.observability import run_config


def make_settings(**values) -> Settings:
    base = {
        "llm_provider": "groq",
        "groq_api_key": "",
        "groq_model": "",
        "gemini_api_key": "",
        "gemini_model": "",
    }
    return Settings(_env_file=None, **{**base, **values})


GROQ = {"groq_api_key": "test-groq-key", "groq_model": "test-groq-model"}
GEMINI = {"gemini_api_key": "test-gemini-key", "gemini_model": "test-gemini-model"}


def test_builds_the_chosen_provider():
    assert isinstance(get_chat_model("groq", cfg=make_settings(**GROQ)), ChatGroq)
    assert isinstance(get_chat_model("gemini", cfg=make_settings(**GEMINI)), ChatGoogleGenerativeAI)


def test_one_provider_means_no_fallback():
    llm = get_llm(cfg=make_settings(**GROQ))

    assert isinstance(llm, ChatGroq)


def test_other_provider_becomes_fallback_when_configured():
    llm = get_llm(cfg=make_settings(**GROQ, **GEMINI))

    assert isinstance(llm, RunnableWithFallbacks)
    assert isinstance(llm.runnable.bound, ChatGroq)
    assert isinstance(llm.fallbacks[0], ChatGoogleGenerativeAI)


def test_gemini_can_be_primary():
    llm = get_llm(cfg=make_settings(llm_provider="gemini", **GROQ, **GEMINI))

    assert isinstance(llm.runnable.bound, ChatGoogleGenerativeAI)
    assert isinstance(llm.fallbacks[0], ChatGroq)


def test_missing_key_for_primary_provider_gives_a_clear_error():
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        get_llm(cfg=make_settings(**GEMINI))


class Answer(BaseModel):
    text: str


def test_structured_llm_keeps_the_fallback():
    llm = get_structured_llm(Answer, cfg=make_settings(**GROQ, **GEMINI))

    assert isinstance(llm, RunnableWithFallbacks)
    assert len(llm.fallbacks) == 1


def test_groq_structured_output_uses_json_schema():
    # Tool calling (LangChain's default) made gpt-oss models reply in prose instead of calling the tool.
    llm = get_structured_llm(Answer, cfg=make_settings(**GROQ))

    assert llm.first.kwargs["response_format"]["type"] == "json_schema"


class BrokenChatModel(BaseChatModel):
    """Fails every call, like a provider that is rate limited or down."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("rate limit reached")

    @property
    def _llm_type(self) -> str:
        return "broken"


def test_fallback_answers_and_logs_why_the_primary_failed(caplog):
    llm = with_fallback(BrokenChatModel(), GenericFakeChatModel(messages=iter(["pong"])), "groq", "gemini")

    with caplog.at_level(logging.WARNING, logger="app.llm.provider"):
        reply = llm.invoke("ping")

    assert reply.text == "pong"
    assert "groq call failed (RuntimeError: rate limit reached); using gemini" in caplog.text


def test_run_config_tags_module_version_and_topic():
    config = run_config("research", topic="EV two-wheelers", thread_id="t-1", cfg=make_settings(run_version="v2"))

    assert config["tags"] == ["research", "v2"]
    assert config["metadata"] == {"module": "research", "version": "v2", "topic": "EV two-wheelers"}
    assert config["configurable"] == {"thread_id": "t-1"}


def test_extra_metadata_cannot_overwrite_version():
    config = run_config("chatbot", cfg=make_settings(run_version="v2"), version="v1", user="u1")

    assert config["metadata"] == {"user": "u1", "module": "chatbot", "version": "v2"}
