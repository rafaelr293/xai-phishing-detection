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
    assert result.risk_score == 82
    assert result.confidence is None
    assert result.evidence["model_probability"] == 0.82
    assert result.indicators[0].type == "nlp_phishing_prediction"