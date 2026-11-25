"""T02 (Rishabh): LLM provider layer.

Every agent gets its model from here, so switching providers is a .env change, not a code change:

    from app.llm.provider import get_llm, get_structured_llm

    get_llm().invoke("Hello").text                  # plain chat model
    get_structured_llm(Outline).invoke(messages)    # returns an Outline (Pydantic) object

Read replies with `.text`, not `.content`: Gemini returns `.content` as a list of blocks, `.text` is always a string.

LLM_PROVIDER picks the primary provider. If the other provider also has a key and a model in .env, it becomes the
fallback: each provider first retries on its own (rate limits, timeouts), and only if it still fails does the call
move to the other provider. A warning is logged when that happens, so a broken provider does not go unnoticed.
With only one provider configured there is no fallback.

To count LLM calls and tokens for RunMetrics, wrap the calls in `track_usage()` from app.llm.usage.
"""

import logging
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import Settings, settings

logger = logging.getLogger(__name__)

Provider = Literal["groq", "gemini"]
PROVIDERS: tuple[Provider, ...] = ("groq", "gemini")

MAX_RETRIES = 3
TIMEOUT_SECONDS = 60


class LLMConfigError(RuntimeError):
    """Raised when the provider that is needed has no API key or model in .env."""


def is_configured(provider: Provider, cfg: Settings = settings) -> bool:
    key, model = _key_and_model(provider, cfg)
    return bool(key and model)


def get_chat_model(provider: Provider, temperature: float = 0.0, cfg: Settings = settings) -> BaseChatModel:
    """One provider's chat model, with its own retry and timeout. No fallback."""
    key, model = _key_and_model(provider, cfg)
    if not key or not model:
        prefix = provider.upper()
        raise LLMConfigError(f"{prefix}_API_KEY and {prefix}_MODEL must both be set in .env to use {provider}")

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model, api_key=key, temperature=temperature, max_retries=MAX_RETRIES, timeout=TIMEOUT_SECONDS
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model, api_key=key, temperature=temperature, max_retries=MAX_RETRIES, timeout=TIMEOUT_SECONDS
    )


def get_llm(temperature: float = 0.0, cfg: Settings = settings) -> Runnable:
    """The primary provider's model, with the other provider as fallback when it is configured."""
    primary, fallback = _primary_and_fallback(cfg)
    model = get_chat_model(primary, temperature, cfg)
    if fallback is None:
        return model
    return with_fallback(model, get_chat_model(fallback, temperature, cfg), primary, fallback)


def get_structured_llm(schema: type[BaseModel], temperature: float = 0.0, cfg: Settings = settings) -> Runnable:
    """Like get_llm(), but every call returns an instance of `schema`.

    Structured output is attached to each provider before the fallback is added, because a fallback wrapper
    cannot add structured output afterwards.
    """
    primary, fallback = _primary_and_fallback(cfg)
    structured = _structured(get_chat_model(primary, temperature, cfg), primary, schema)
    if fallback is None:
        return structured
    return with_fallback(
        structured, _structured(get_chat_model(fallback, temperature, cfg), fallback, schema), primary, fallback
    )


def with_fallback(primary_runnable: Runnable, fallback_runnable: Runnable, primary: str, fallback: str) -> Runnable:
    """`primary_runnable` with `fallback_runnable` behind it, logging a warning whenever the fallback is needed."""
    notice = _FallbackNotice(primary, fallback)
    return primary_runnable.with_config(callbacks=[notice]).with_fallbacks([fallback_runnable])


class _FallbackNotice(BaseCallbackHandler):
    """Logs why the primary provider failed (after its own retries), just before the fallback takes over."""

    def __init__(self, primary: str, fallback: str) -> None:
        self.primary = primary
        self.fallback = fallback

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        reason = str(error).replace("\n", " ")[:200]
        logger.warning("%s call failed (%s: %s); using %s", self.primary, type(error).__name__, reason, self.fallback)


def _structured(model: BaseChatModel, provider: Provider, schema: type[BaseModel]) -> Runnable:
    if provider == "groq":
        # Constrained decoding, so the reply always matches the schema. LangChain's default (tool calling) failed on
        # every short prompt with gpt-oss models in testing: the model answered in prose instead of calling the tool.
        return model.with_structured_output(schema, method="json_schema", strict=True)
    return model.with_structured_output(schema)


def _primary_and_fallback(cfg: Settings) -> tuple[Provider, Provider | None]:
    primary: Provider = cfg.llm_provider
    if not is_configured(primary, cfg):
        raise LLMConfigError(
            f"LLM_PROVIDER is {primary}, but {primary.upper()}_API_KEY or {primary.upper()}_MODEL is empty in .env"
        )
    other = next(p for p in PROVIDERS if p != primary)
    return primary, (other if is_configured(other, cfg) else None)


def _key_and_model(provider: Provider, cfg: Settings) -> tuple[str, str]:
    if provider == "groq":
        return cfg.groq_api_key.strip(), cfg.groq_model.strip()
    return cfg.gemini_api_key.strip(), cfg.gemini_model.strip()
