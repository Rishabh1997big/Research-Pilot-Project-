"""The planner with the real LLMs, each provider alone. Skipped by default; run with:  uv run pytest -m live"""

import pytest

from app.agents.research.nodes.planner import MAX_SECTIONS, MIN_SECTIONS, YEAR, Outline, plan_outline
from app.config import settings
from app.llm.provider import PROVIDERS, get_structured_llm, is_configured

pytestmark = pytest.mark.live


@pytest.mark.parametrize("provider", PROVIDERS)
def test_real_outline(provider):
    if not is_configured(provider):
        pytest.skip(f"{provider} is not configured in .env")
    other = next(p for p in PROVIDERS if p != provider)
    cfg = settings.model_copy(update={"llm_provider": provider, f"{other}_api_key": ""})

    outline = plan_outline("Quick commerce in India", llm=get_structured_llm(Outline, cfg=cfg))

    assert MIN_SECTIONS <= len(outline.sections) <= MAX_SECTIONS
    for section in outline.sections:
        assert section.title and section.subtopics and section.queries
        assert not any(YEAR.search(q) for q in section.queries)
