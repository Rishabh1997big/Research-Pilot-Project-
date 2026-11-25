from fastapi.testclient import TestClient

from app.api.main import app
from app.config import ROOT_DIR, settings


def test_health_returns_ok():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_storage_paths_are_inside_the_project():
    assert settings.chroma_dir.is_relative_to(ROOT_DIR)
    assert settings.checkpoint_db.is_relative_to(ROOT_DIR)
