"""T02 (Rishabh): count LLM calls and tokens, for the `metrics` key of the research state (RunMetrics).

Every chat model call made inside the `with` block is counted, including calls inside structured output and calls
that went to the fallback provider. Token counts come from the provider's own usage report.

    from app.llm.usage import track_usage

    def planner(state):
        with track_usage() as usage:
            outline = get_structured_llm(Outline).invoke(messages)
        return {"sections": ..., "metrics": usage.as_metrics()}

Each block counts only its own calls: a block nested inside another one does not add to the outer block. LangGraph
runs each node in its own context, so parallel nodes never mix their counts.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook


class UsageCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = _usage_of(response)
        with self._lock:
            self.llm_calls += 1
            if usage:
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

    def as_metrics(self) -> dict[str, int]:
        """Counts with the same names as RunMetrics, ready to return from a node as {"metrics": ...}."""
        return {"llm_calls": self.llm_calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


# One context variable, registered once: LangChain then adds the active counter to every run started in this context.
_active_counter: ContextVar[UsageCounter | None] = ContextVar("researchpilot_usage_counter", default=None)
register_configure_hook(_active_counter, inheritable=True)


@contextmanager
def track_usage() -> Iterator[UsageCounter]:
    counter = UsageCounter()
    token = _active_counter.set(counter)
    try:
        yield counter
    finally:
        _active_counter.reset(token)


def _usage_of(response: LLMResult) -> dict[str, Any] | None:
    try:
        generation = response.generations[0][0]
    except IndexError:
        return None
    if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
        return generation.message.usage_metadata  # type: ignore[return-value]
    return None
