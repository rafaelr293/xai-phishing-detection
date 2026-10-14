from app.analysis.email_nlp import EmailNLPDetector
from app.domain.schemas import AnalysisContext, ParsedEmail


class FakePredictor:
    def predict(self, subject: str, body: str) -> dict:
        assert subject == "Urgent account notice"
        assert body == "Verify your account immediately"
        return {"label": "phishing", "probability": 0.82, "threshold": 0.5}


def test_email_nlp_adapter_normalizes_predictor_output(tmp_path) -> None:
    email = ParsedEmail(
        content_hash="a" * 64,
        source_path="sample.eml",
        subject="Urgent account notice",
        plain_text="Verify your account immediately",
    )
    detector = EmailNLPDetector(tmp_path / "Email_Inference.py", tmp_path / "model.joblib", FakePredictor())

    result = detector.analyze(AnalysisContext(email=email))

    assert result.detector == "email_nlp"
    assert result.risk_score == 64
    assert result.confidence is None
    assert result.evidence["model_probability"] == 0.82
    assert result.indicators[0].type == "nlp_phishing_prediction"


def test_email_nlp_probability_below_threshold_has_zero_risk(tmp_path) -> None:
    class BelowThresholdPredictor:
        def predict(self, subject: str, body: str) -> dict:
            return {"label": "legitimate", "probability": 0.49, "threshold": 0.5}

    email = ParsedEmail(content_hash="b" * 64, source_path="sample.eml", plain_text="Normal message")
    detector = EmailNLPDetector(tmp_path / "Email_Inference.py", tmp_path / "model.joblib", BelowThresholdPredictor())

    result = detector.analyze(AnalysisContext(email=email))

    assert result.risk_score == 0


def test_email_nlp_threshold_and_full_probability_map_to_one_and_one_hundred() -> None:
    assert EmailNLPDetector._risk_score(0.5, 0.5) == 1
    assert EmailNLPDetector._risk_score(1.0, 0.5) == 100