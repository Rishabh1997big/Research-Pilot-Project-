# ResearchPilot

A multi-agent research assistant. Give it a topic and it plans an outline, searches the web, reads the sources and
writes a report where every claim is cited. You approve the outline before writing starts and can ask for changes to
single sections afterwards. It also gives a sentiment analysis with recommendations, lets you chat with the report,
and comes with an evaluation setup that scores all of this.

Built with LangGraph, LangChain, FastAPI, Streamlit, Chroma and LangSmith.

## 1. Install the tools

Install [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and all dependencies for you):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Set up the project

```powershell
git clone <repo-url>
cd researchpilot
uv sync
copy .env.example .env
```

Then fill in `.env`:

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | <https://console.groq.com/keys> |
| `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> |
| `LANGSMITH_API_KEY` | <https://smith.langchain.com> (Settings, API Keys) |
| `TAVILY_API_KEY` | Optional, only if `SEARCH_PROVIDER=tavily` |

`LLM_PROVIDER` and `SEARCH_PROVIDER` switch providers without code changes. Embeddings run locally and need no key.

## 3. Run

```powershell
uv run uvicorn app.api.main:app --reload      # API at http://localhost:8000 (docs at /docs)
uv run streamlit run ui/streamlit_app.py      # UI at http://localhost:8501
uv run pytest                                 # tests
uv run ruff check .                           # lint
```

## 4. Project structure

| Folder | What lives there | Tasks |
|---|---|---|
| `app/config.py` | Settings from `.env` (`from app.config import settings`) | T01 |
| `app/llm/` | Swappable LLM provider layer | T02 |
| `app/agents/research/` | Research graph: state, planner, sources, writer, compiler, verifier, change requests | T03, T06, T07, T11, T12, T16, T17, T21 |
| `app/agents/analysis/` | Sentiment and recommendations | T18, T23 |
| `app/tools/` | Web search and page reader | T04, T08 |
| `app/kb/` | Chunking, embeddings, Chroma, RAG chatbot | T09, T14, T19, T20 |
| `app/api/` | FastAPI routes | T24 |
| `ui/` | Streamlit screens | T10, T15, T25 |
| `eval/` | Test topics, Q&A set, metrics, evaluation runs | T05, T22, T26, T29 |
| `docs/` | Architecture doc and evaluation report | T29, T30 |
| `tests/` | Tests | all |
| `data/` | Local Chroma DB and checkpoints (not committed) | |

Each placeholder file names its task ID and owner. The full task list is in the project tracker sheet.

## 5. Git workflow

- Never push to `main` directly. Every change goes through a pull request.
- Branch name: task ID plus a short name, for example `feature/T04-search-tool`.
- Commit messages start with the task ID, for example `T04: add ddgs search tool`.
- Assign the reviewer listed for that task in the tracker sheet. After the merge, update the task's status in the sheet.
- Run `uv run pytest` and `uv run ruff check .` before opening a pull request.
- Never commit `.env` or API keys.
