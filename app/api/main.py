"""FastAPI app. T24 (Deepanshu) adds the research, analysis, ingest and chat routes here.

Run:  uv run uvicorn app.api.main:app --reload
"""

from fastapi import FastAPI

from app.config import settings

app = FastAPI(title="ResearchPilot API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm_provider": settings.llm_provider}
