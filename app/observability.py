"""T02 (Rishabh): LangSmith tracing helpers.

Tracing itself is switched on by LANGSMITH_TRACING, LANGSMITH_API_KEY and LANGSMITH_PROJECT in .env (app.config
loads them into the environment). This module adds the same tags and metadata to every run, so traces can be
filtered in LangSmith by module, system version and topic.

    from app.observability import run_config

    graph.invoke(state, config=run_config("research", topic=topic, thread_id=thread_id))
"""

from typing import Literal

from langchain_core.runnables import RunnableConfig

from app.config import Settings, settings

Module = Literal["research", "analysis", "chatbot", "eval"]


def tracing_enabled(cfg: Settings = settings) -> bool:
    return cfg.langsmith_tracing and bool(cfg.langsmith_api_key.strip())


def run_config(
    module: Module,
    topic: str | None = None,
    thread_id: str | None = None,
    cfg: Settings = settings,
    **extra_metadata: str,
) -> RunnableConfig:
    """RunnableConfig with tags (module, version) and metadata for one run.

    thread_id is the LangGraph checkpoint thread; pass it for graphs that pause and resume (human review).
    """
    # module and version come last so extra metadata cannot overwrite the keys traces are filtered by
    metadata: dict[str, str] = {**extra_metadata, "module": module, "version": cfg.run_version}
    if topic:
        metadata["topic"] = topic

    config: RunnableConfig = {
        "run_name": f"{module}:{topic}" if topic else module,
        "tags": [module, cfg.run_version],
        "metadata": metadata,
    }
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    return config
