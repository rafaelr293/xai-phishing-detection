from __future__ import annotations

from collections.abc import Iterable

from app.domain.ports import Detector
from app.domain.schemas import AnalysisContext, DetectorResultData, ParsedEmail


class AnalysisOrchestrator:
    def __init__(self, detectors: Iterable[Detector] = ()):
        self.detectors = list(detectors)

    def analyze(self, input_data: ParsedEmail | AnalysisContext) -> list[DetectorResultData]:
        context = input_data if isinstance(input_data, AnalysisContext) else AnalysisContext.from_email(input_data)
        results: list[DetectorResultData] = []
        for detector in self.detectors:
            try:
                result = detector.analyze(context)
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
