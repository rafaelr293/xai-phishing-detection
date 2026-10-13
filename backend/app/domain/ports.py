from collections.abc import Callable, Iterable
from typing import Protocol

from app.domain.schemas import AnalysisContext, DetectorResultData


class Detector(Protocol):
    name: str

    def analyze(self, context: AnalysisContext) -> DetectorResultData:
        """Analyze one email and its associated URL/artifact context."""


DetectorFactory = Callable[[], Detector]


class EmailIngestor(Protocol):
    def discover(self) -> Iterable[str]:
        """Return candidate source paths without parsing or executing them."""
