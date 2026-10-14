from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.domain.schemas import DetectorResultData

DEFAULT_WEIGHTS = {
    "email_nlp": 0.25,
    "url_risk": 0.30,
    "attachment_analysis": 0.20,
    "visual_analysis": 0.25,
}


def url_analysis_status(status: str, url_records: list) -> str:
    selected_records = [record for record in url_records if record.selected_for_traversal]
    if not selected_records or status == "FAILED":
        return status
    all_complete = all(
        any(
            traversal.status == "COMPLETED"
            and (traversal.visual_analysis or {}).get("status") == "COMPLETED"
            for traversal in record.traversals
        )
        for record in selected_records
    )
    if status in {"COMPLETED", "PARTIAL"}:
        return "COMPLETED" if all_complete else "PARTIAL"
    return status


class RiskAssessment(BaseModel):
    final_score: Optional[int] = Field(default=None, ge=0, le=100)
    risk_level: Optional[str] = None
    status: str
    coverage: dict = Field(default_factory=dict)
    components: list[dict] = Field(default_factory=list)
    indicators: list[dict] = Field(default_factory=list)


class RiskAssessmentEngine:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    @staticmethod
    def risk_level(score: int | None) -> str | None:
        if score is None:
            return None
        if score >= 75:
            return "critical"
        if score >= 50:
            return "high"
        if score >= 25:
            return "medium"
        return "low"

    def assess(
        self,
        results: list[DetectorResultData],
        expected_detectors: list[str] | None = None,
    ) -> RiskAssessment:
        expected = expected_detectors or [result.detector for result in results]
        by_detector = {result.detector: result for result in results}
        available = [
            result for result in results
            if result.status == "COMPLETED" and result.risk_score is not None
        ]
        failed = [result.detector for result in results if result.status in {"FAILED", "ERROR"}]
        missing = [name for name in expected if name not in by_detector]
        denominator = sum(self.weights.get(result.detector, 1.0) for result in available)
        final_score = None
        components = []
        if denominator:
            final_score = round(sum(
                result.risk_score * self.weights.get(result.detector, 1.0)
                for result in available
            ) / denominator)
            for result in available:
                weight = self.weights.get(result.detector, 1.0)
                components.append({
                    "detector": result.detector,
                    "score": result.risk_score,
                    "weight": weight,
                    "normalized_weight": round(weight / denominator, 4),
                })

        indicators = []
        for result in results:
            indicators.extend({"detector": result.detector, **indicator.model_dump()} for indicator in result.indicators)
        status = "COMPLETED" if not failed and not missing and available else "PARTIAL" if available else "FAILED"
        return RiskAssessment(
            final_score=final_score,
            risk_level=self.risk_level(final_score),
            status=status,
            coverage={
                "expected_detectors": expected,
                "available_detectors": [result.detector for result in available],
                "failed_detectors": failed,
                "missing_detectors": missing,
            },
            components=components,
            indicators=indicators,
        )
