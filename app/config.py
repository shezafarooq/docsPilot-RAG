from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docspilot"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: Path = Path("data")
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_base_url: str = ""
    tracing_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
