# ResearchPilot architecture

Owner: Rishabh (T30)

## Overview
## Research agent graph
## State design

Code: `app/agents/research/state.py`.

| Key | Type | Reducer | Purpose |
|---|---|---|---|
| `topic` | str | replace | The user's topic |
| `section_order` | list of section IDs | replace | Outline order; the report has exactly these sections |
| `sections` | dict: section ID to `Section` | merge by ID, None removes | Each section with its own queries, passages, sources, draft, citations, status |
| `sources` | dict: source ID to `Source` | merge by ID | Source registry (`S1`, `S2`...): URL, title, domain |
| `outline_approved` | bool | replace | Set when the user approves the outline |
| `feedback` | list of `Feedback` | replace | Change requests with their kind and target sections |
| `sections_to_run` | list of section IDs | replace | Sections the next search, write and verify pass works on |
| `report` | str | replace | Final Markdown with numbered citations and references |
| `metrics` | `RunMetrics` | sum | LLM calls, search calls, pages fetched, tokens |

A `Section` holds `id`, `title`, `subtopics`, `queries`, `passages` (each tied to a source ID), `source_ids`,
`draft` (with inline `[S#]` citations), `citations` (claim, source IDs, verifier result), `status`
(planned, researching, drafted, verified, needs_revision) and `revision`.

Why it looks like this:

- **Section-level editing.** Everything a section needs is inside the section, so a change request re-runs only the
  sections in `sections_to_run` and leaves the rest as they are.
- **Citations from the start.** A passage is stored with its source ID when the page is read. The writer only sees
  passages with IDs, so every claim it writes can point at a source, and the verifier can check that source.
- **Parallel sections.** Sections are researched in parallel branches. The merge-by-ID reducer lets each branch
  return only its own section, and the metrics reducer adds up the counts from all branches.
- **Checkpoint-safe.** The state is plain dicts and lists, so the SQLite checkpointer saves and restores it for the
  outline approval pause without any custom serialisation. Pydantic is used only for LLM structured output.
- **Source IDs are assigned in one sequential step**, never in parallel branches, so two sections cannot hand out
  the same ID.
- **One place for ID and citation formats.** `next_section_id()` and `next_source_id()` hand out the next ID after
  the highest one in use. Sources are never removed, so a source ID always means the same page. `cited_source_ids()`
  reads `[S1]`, `[S1][S2]` and `[S1, S2]` marks, and the writer, compiler, verifier and evaluation all use it.
- **Metric typos fail loudly.** The metrics reducer rejects unknown names, so `{"llm_call": 1}` raises instead of
  going missing from the totals.

How change requests map onto the state:

| Request | `Feedback` | What re-runs |
|---|---|---|
| "Make section 3 more detailed" | `expand_section`, `section_ids=["sec-3"]` | Search, write and verify for sec-3 |
| "Add a subtopic on government policy" | `add_subtopic`, `section_ids=[]` | A new section (`next_section_id()`) is added to `section_order` and researched |
| "Add battery costs to section 2" | `add_subtopic`, `section_ids=["sec-2"]` | sec-2 gets the subtopic and is re-run |
| "Replace the weak sources in section 2" | `replace_sources`, `section_ids=["sec-2"]` | Sources that no supported citation uses are dropped from sec-2, then it is re-searched |

In every case only the listed sections are put in `sections_to_run`, and `revision` goes up by one for each.

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
