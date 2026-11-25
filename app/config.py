"""App settings, read once from the environment and the project's .env file.

Use it anywhere with:  from app.config import settings
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

# Put .env values into os.environ too, so libraries that read env vars directly (LangSmith) see them.
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["groq", "gemini", "ollama"] = "groq"
    groq_api_key: str = ""
    groq_model: str = ""
    gemini_api_key: str = ""
    gemini_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""

    # Search
    search_provider: Literal["ddgs", "tavily"] = "ddgs"
    tavily_api_key: str = ""

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "researchpilot"

    # Storage
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: Path = Path("data/chroma")
    checkpoint_db: Path = Path("data/checkpoints.sqlite")

    # App
    api_base_url: str = "http://localhost:8000"

    @field_validator("chroma_dir", "checkpoint_db")
    @classmethod
    def _relative_to_project_root(cls, value: Path) -> Path:
        # Relative paths in .env are taken from the project root, not from wherever the command was run.
        return value if value.is_absolute() else ROOT_DIR / value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
