from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://phishing:phishing@localhost:5432/phishing_detection"
    ingestion_directory: Path = Path("./data/inbox")
    max_eml_size_bytes: int = 10 * 1024 * 1024
    email_inference_directory: Path = Path("./Tan Files")
    email_inference_module: str = "Email_Inference.py"
    email_model_path: Optional[Path] = None
    enable_email_nlp: bool = False
    url_inference_directory: Path = Path("./Tan Files")
    url_inference_module: str = "URL_Inference"
    url_model_path: Optional[Path] = None
    enable_url_risk: bool = False
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    enable_polling: bool = False
    poll_interval_seconds: int = 30
    artifact_directory: Path = Path("./data/artifacts")
    browser_timeout_ms: int = 15000
    browser_max_redirects: int = 10
    browser_viewport_width: int = 1440
    browser_viewport_height: int = 1000

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
