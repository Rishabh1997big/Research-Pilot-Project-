"""App settings, read once from the environment and the project's .env file.

Use it anywhere with:  from app.config import settings
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

# Put .env values into os.environ too, so libraries that read env vars directly (LangSmith) see them.
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # API keys use Field(repr=False), so printing the settings (logs, test failures, tracebacks) never shows them.

    # LLM
    llm_provider: Literal["groq", "gemini"] = "groq"
    groq_api_key: str = Field("", repr=False)
    groq_model: str = ""
    gemini_api_key: str = Field("", repr=False)
    gemini_model: str = ""

    # Search
    search_provider: Literal["ddgs", "tavily"] = "ddgs"
    tavily_api_key: str = Field("", repr=False)
    # Keep search results on disk, so re-runs and v1/v2 evaluation reuse them instead of hitting rate limits again
    search_cache: bool = True
    search_cache_dir: Path = Path("data/search_cache")
    # Keep the text of pages that were read, so v1 and v2 evaluation runs read exactly the same page text
    page_cache: bool = True
    page_cache_dir: Path = Path("data/page_cache")
    # A contact (email or project URL) the reader sends in its User-Agent, so sites know who is reading their pages.
    # Some sites (Wikipedia) block readers that give none. Empty: the reader uses a common browser User-Agent.
    reader_contact: str = Field("", repr=False)

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = Field("", repr=False)
    langsmith_project: str = "researchpilot"
    # Tag on every trace, so runs of different system versions (e.g. v1 without verifier, v2 with) can be compared
    run_version: str = "v1"

    # Storage
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where the embedding model is downloaded to once (about 70 MB), instead of a temp folder that gets cleaned
    embedding_cache_dir: Path = Path("data/models")
    chroma_dir: Path = Path("data/chroma")
    checkpoint_db: Path = Path("data/checkpoints.sqlite")

    # App
    api_base_url: str = "http://localhost:8000"

    @field_validator("chroma_dir", "checkpoint_db", "search_cache_dir", "page_cache_dir", "embedding_cache_dir")
    @classmethod
    def _relative_to_project_root(cls, value: Path) -> Path:
        # Relative paths in .env are taken from the project root, not from wherever the command was run.
        return value if value.is_absolute() else ROOT_DIR / value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
