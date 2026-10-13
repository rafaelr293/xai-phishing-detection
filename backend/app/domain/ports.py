from collections.abc import Callable, Iterable
from typing import Protocol

from app.domain.schemas import DetectorResultData, ParsedEmail


class Detector(Protocol):
    name: str

    def analyze(self, email: ParsedEmail) -> DetectorResultData:
        """Analyze one parsed email without mutating the source message."""


DetectorFactory = Callable[[], Detector]


class EmailIngestor(Protocol):
    def discover(self) -> Iterable[str]:
        """Return candidate source paths without parsing or executing them."""
