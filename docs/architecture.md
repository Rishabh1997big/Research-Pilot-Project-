# ResearchPilot architecture

Owner: Rishabh (T30)

## Overview
## Research agent graph
## State design
## Human-in-the-loop and checkpointing
## Analysis agent (and why it is a separate graph)
## Knowledge base and chatbot
## Key decisions and trade-offs

- **Swappable LLM providers** (`app/llm/provider.py`). Groq or Gemini is chosen by `LLM_PROVIDER`. When both have
  keys, the other one is a fallback: each provider retries on its own first (rate limits, timeouts), then the call
  moves to the other. Free tiers have rate limits, so this keeps long research runs going. Each fallback is logged
  with the reason, because a silent fallback hides a broken provider.
- **Structured output on Groq uses `json_schema`, not tool calling.** With gpt-oss models, LangChain's default (tool
  calling) failed on 9 of 9 short prompts: the model answered in prose instead of calling the tool, and every call
  quietly went to the Gemini fallback. `json_schema` (constrained decoding) succeeded on 9 of 9. Live tests
  (`uv run pytest -m live`) test each provider alone, without the fallback, to catch this kind of failure.
- **Token and call counts per node** (`app/llm/usage.py`). Structured output returns only the parsed object, so the
  token usage is lost. `track_usage()` collects it from LangChain callbacks instead and returns it in the
  `RunMetrics` shape.
- **Traces tagged by version** (`app/observability.py`). Every run carries module, `RUN_VERSION` and topic tags, so
  v1 and v2 runs can be compared side by side in LangSmith.
