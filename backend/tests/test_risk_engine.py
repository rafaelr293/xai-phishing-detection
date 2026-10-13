from app.analysis.explainability import ExplainabilityEngine
from app.analysis.risk_engine import RiskAssessmentEngine
from app.domain.schemas import DetectorResultData, Indicator


def test_weighted_score_uses_available_detector_weights() -> None:
    engine = RiskAssessmentEngine({"email_nlp": 0.25, "url_risk": 0.75})
    assessment = engine.assess(
        [
            DetectorResultData(detector="email_nlp", status="COMPLETED", risk_score=20),
            DetectorResultData(detector="url_risk", status="COMPLETED", risk_score=80),
        ],
        ["email_nlp", "url_risk"],
    )

    assert assessment.final_score == 65
    assert assessment.risk_level == "high"
    assert assessment.status == "COMPLETED"
    assert assessment.coverage["missing_detectors"] == []


def test_missing_and_failed_detectors_are_reported_without_blocking_score() -> None:
    engine = RiskAssessmentEngine({"email_nlp": 0.5, "url_risk": 0.5, "visual_analysis": 0.5})
    assessment = engine.assess(
        [
            DetectorResultData(detector="email_nlp", status="COMPLETED", risk_score=60),
            DetectorResultData(detector="url_risk", status="FAILED"),
        ],
        ["email_nlp", "url_risk", "visual_analysis"],
    )

    assert assessment.final_score == 60
    assert assessment.status == "PARTIAL"
    assert assessment.coverage["failed_detectors"] == ["url_risk"]
    assert assessment.coverage["missing_detectors"] == ["visual_analysis"]


def test_explanation_uses_detector_evidence_and_mentions_partial_coverage() -> None:
    assessment = RiskAssessmentEngine().assess(
        [DetectorResultData(
            detector="url_risk",
            status="COMPLETED",
            risk_score=90,
            indicators=[Indicator(
                type="domain_age",
                description="The domain was registered recently",
                severity="high",
            )],
        )],
        ["url_risk", "visual_analysis"],
    )
    explanation = ExplainabilityEngine().explain(assessment)

    assert "The domain was registered recently" in explanation["technical"]
    assert "Analysis coverage was partial" in explanation["technical"]
    assert "registered recently" in explanation["user"]