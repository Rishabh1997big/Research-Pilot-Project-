"""The only place the UI talks to the backend.

While the API routes (T24, Deepanshu) do not exist, `UI_USE_MOCK=1` (the default) returns fake data in the same shape
the real API will use, so the screens can be built and tried now. Set `UI_USE_MOCK=0` to call the real API.

The endpoint paths and JSON shapes below are PROPOSED; agree them with Deepanshu (T24) and Rishabh (T11) and change
them here, in one place.

Outline shape (matches Section in app/agents/research/state.py):
    {"id": "sec-1" | None, "title": "...", "subtopics": ["...", ...]}
A new section added in the UI has id None; the backend assigns the ID.
"""

import os
import time

import httpx

from app.config import settings

USE_MOCK = os.getenv("UI_USE_MOCK", "1") != "0"
TIMEOUT = 300  # planning and research take a while


class ApiError(Exception):
    """The API could not be reached or returned an error. The message is safe to show to the user."""


def _post(path: str, payload: dict) -> dict:
    try:
        response = httpx.post(f"{settings.api_base_url}{path}", json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise ApiError(f"API call to {path} failed: {exc}") from exc


def start_research(topic: str) -> dict:
    """Run the planner for a topic. Returns {"run_id": str, "sections": [outline sections]}.

    The research graph then waits for outline approval (T11).
    """
    if USE_MOCK:
        time.sleep(1)
        return {
            "run_id": "mock-run-1",
            "sections": [
                {"id": "sec-1", "title": "Market overview", "subtopics": ["sales volume", "growth rate"]},
                {"id": "sec-2", "title": "Key players", "subtopics": ["market share", "new entrants"]},
                {"id": "sec-3", "title": "Government policy", "subtopics": ["subsidies", "regulation"]},
                {"id": "sec-4", "title": "Challenges and outlook", "subtopics": ["risks", "forecast"]},
            ],
        }
    return _post("/research", {"topic": topic})


def approve_outline(run_id: str, sections: list[dict]) -> dict:
    """Send the approved (possibly edited) outline; the paused graph resumes. Returns {"status": "writing"}."""
    if USE_MOCK:
        time.sleep(1)
        return {"status": "writing"}
    return _post(f"/research/{run_id}/approve", {"sections": sections})
