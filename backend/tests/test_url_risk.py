import pandas as pd

from app.analysis.url_risk import URLRiskDetector
from app.domain.schemas import AnalysisContext, ParsedEmail


class FakeURLModule:
    def scan_urls(self, urls, pipeline):
        return pd.DataFrame([
            {
                "url": urls[0],
                "canonical": "example.test/login",
                "label": "phishing",
                "phishing_probability": 0.91,
                "threshold": 0.52,
                "suspicious_words_host": "login",
                "suspicious_words_path": "verify",
            }
        ])


def test_url_risk_adapter_normalizes_inference_output(tmp_path) -> None:
    detector = URLRiskDetector(tmp_path / "URL_Inference", tmp_path / "url_pipeline.joblib", pipeline=object())
    detector._module = FakeURLModule()
    context = AnalysisContext(
        email=ParsedEmail(content_hash="a" * 64, source_path="sample.eml", urls=["https://example.test/login"])
    )

    result = detector.analyze(context)

    assert result.detector == "url_risk"
    assert result.risk_score == 81
    assert result.confidence is None
    assert result.indicators[0].type == "suspicious_url"
    assert result.evidence["url_results"][0]["threshold"] == 0.52


def test_url_risk_uses_highest_probability_with_threshold_mapping(tmp_path) -> None:
    class MultipleURLModule:
        def scan_urls(self, urls, pipeline):
            return pd.DataFrame([
                {"url": urls[0], "label": "legitimate", "phishing_probability": 0.51, "threshold": 0.5},
                {"url": urls[1], "label": "phishing", "phishing_probability": 0.76, "threshold": 0.5},
            ])

    detector = URLRiskDetector(tmp_path / "URL_Inference", tmp_path / "url_pipeline.joblib", pipeline=object())
    detector._module = MultipleURLModule()
    context = AnalysisContext(
        email=ParsedEmail(
            content_hash="b" * 64,
            source_path="sample.eml",
            urls=["https://one.test", "https://two.test"],
        )
    )

    result = detector.analyze(context)

    assert result.risk_score == 52


def test_url_risk_below_threshold_has_zero_risk() -> None:
    assert URLRiskDetector._risk_score(0.49, 0.5) == 0
    assert URLRiskDetector._risk_score(0.5, 0.5) == 1
    assert URLRiskDetector._risk_score(1.0, 0.5) == 100