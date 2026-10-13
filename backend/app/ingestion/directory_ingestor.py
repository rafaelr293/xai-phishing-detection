from pathlib import Path

from app.core.config import Settings
from app.ingestion.eml_parser import parse_eml


class DirectoryIngestor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def discover(self) -> list[Path]:
        directory = self.settings.ingestion_directory
        directory.mkdir(parents=True, exist_ok=True)
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".eml")

    def parse(self, path: Path):
        return parse_eml(path, self.settings.max_eml_size_bytes)
