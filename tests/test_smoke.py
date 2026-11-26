from fastapi.testclient import TestClient

from app.api.main import app
from app.config import ROOT_DIR, Settings, settings


def test_health_returns_ok():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_printed_settings_never_show_api_keys():
    keys = {"groq_api_key": "secret-groq", "gemini_api_key": "secret-gemini", "tavily_api_key": "secret-tavily"}
    cfg = Settings(_env_file=None, langsmith_api_key="secret-langsmith", **keys)

    assert "secret" not in repr(cfg)
    assert "secret" not in str(cfg)
    assert cfg.groq_api_key == "secret-groq"  # still readable in code


def test_storage_paths_are_inside_the_project():
    assert settings.chroma_dir.is_relative_to(ROOT_DIR)
    assert settings.checkpoint_db.is_relative_to(ROOT_DIR)
