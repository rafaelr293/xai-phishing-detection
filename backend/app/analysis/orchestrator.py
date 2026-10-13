from collections.abc import Iterable

from app.domain.ports import Detector
from app.domain.schemas import DetectorResultData, ParsedEmail


class AnalysisOrchestrator:
    def __init__(self, detectors: Iterable[Detector] = ()):
        self.detectors = list(detectors)

    def analyze(self, email: ParsedEmail) -> list[DetectorResultData]:
        results: list[DetectorResultData] = []
        for detector in self.detectors:
            try:
                result = detector.analyze(email)
                results.append(result)
            except Exception as exc:  # Detector failures must be visible to callers.
                results.append(
                    DetectorResultData(
                        detector=detector.name,
                        status="FAILED",
                        error_message=str(exc),
                    )
                )
        return results
