from __future__ import annotations

from datetime import datetime, timezone
from math import isclose

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session, joinedload

from app.analysis.explainability import ExplainabilityEngine
from app.analysis.risk_engine import DEFAULT_WEIGHTS, RiskAssessmentEngine
from app.analysis.risk_engine import url_analysis_status
from app.db.base import get_db
from app.db.models import Email, EmailAnalysis, ProcessingStatus, RiskSettings
from app.domain.schemas import DetectorResultData

router = APIRouter(prefix="/api/settings", tags=["settings"])


class RiskWeightsUpdate(BaseModel):
    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"Unknown detector weights: {', '.join(sorted(unknown))}")
        complete = {**DEFAULT_WEIGHTS, **weights}
        if any(value < 0 or value > 1 for value in complete.values()):
            raise ValueError("Weights must be between 0 and 1")
        if not isclose(sum(complete.values()), 1.0, abs_tol=0.0001):
            raise ValueError("Detector weights must total 100%")
        return complete


def _get_or_create_settings(db: Session) -> RiskSettings:
    settings = db.get(RiskSettings, 1)
    if settings is None:
        settings = RiskSettings(id=1, weights=DEFAULT_WEIGHTS.copy())
        db.add(settings)
        db.flush()
    return settings


@router.get("/risk-weights")
def get_risk_weights(db: Session = Depends(get_db)) -> dict:
    settings = _get_or_create_settings(db)
    db.commit()
    return {"weights": {**DEFAULT_WEIGHTS, **settings.weights}, "updated_at": settings.updated_at}


@router.put("/risk-weights")
def update_risk_weights(payload: RiskWeightsUpdate, db: Session = Depends(get_db)) -> dict:
    settings = _get_or_create_settings(db)
    settings.weights = payload.weights
    engine = RiskAssessmentEngine(payload.weights)
    explainability = ExplainabilityEngine()
    updated_analyses = 0
    analyses = db.query(EmailAnalysis).options(
        joinedload(EmailAnalysis.detector_results),
        joinedload(EmailAnalysis.email).joinedload(Email.url_records),
    ).all()
    for analysis in analyses:
        results = [
            DetectorResultData(
                detector=result.detector,
                status=result.status.value,
                risk_score=result.risk_score,
                confidence=result.confidence,
                indicators=result.indicators,
                evidence=result.evidence,
                error_message=result.error_message,
            )
            for result in analysis.detector_results
        ]
        assessment = engine.assess(results, [result.detector for result in results])
        explanations = explainability.explain(assessment)
        assessment.status = url_analysis_status(assessment.status, analysis.email.url_records)
        analysis.status = ProcessingStatus(assessment.status)
        analysis.risk_score = assessment.final_score
        analysis.risk_level = assessment.risk_level
        analysis.aggregation_details = assessment.model_dump()
        analysis.technical_explanation = explanations["technical"]
        analysis.user_explanation = explanations["user"]
        updated_analyses += 1
    db.commit()
    return {
        "weights": payload.weights,
        "updated_at": datetime.now(timezone.utc),
        "updated_analyses": updated_analyses,
    }