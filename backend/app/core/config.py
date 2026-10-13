from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://phishing:phishing@localhost:5432/phishing_detection"
    ingestion_directory: Path = Path("./data/inbox")
    max_eml_size_bytes: int = 10 * 1024 * 1024
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    enable_polling: bool = False
    poll_interval_seconds: int = 30

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
