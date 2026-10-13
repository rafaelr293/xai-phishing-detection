from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

from app.domain.schemas import AnalysisContext, DetectorResultData, Indicator


class EmailNLPDetector:
    """Adapter for Tan's PhishingPredictor inference artifact."""

    name = "email_nlp"

    def __init__(self, inference_module: Path, model_path: Path, predictor: Any = None):
        self.inference_module = inference_module
        self.model_path = model_path
        self._predictor = predictor

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if not self.inference_module.is_file():
            raise FileNotFoundError(f"Email inference module not found: {self.inference_module}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Email model artifact not found: {self.model_path}")

        module_directory = str(self.inference_module.parent)
        if module_directory not in sys.path:
            sys.path.insert(0, module_directory)
        spec = importlib.util.spec_from_file_location("xai_email_inference", self.inference_module)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load email inference module: {self.inference_module}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predictor_class = getattr(module, "PhishingPredictor", None)
        if not callable(predictor_class):
            raise AttributeError("Email inference module does not expose PhishingPredictor")
        self._predictor = predictor_class(str(self.model_path))
        return self._predictor

    @staticmethod
    def _severity(score: int) -> str:
        if score >= 75:
            return "critical"
        if score >= 50:
            return "high"
        if score >= 25:
            return "medium"
        return "low"

    def analyze(self, context: AnalysisContext) -> DetectorResultData:
        predictor = self._load_predictor()
        email = context.email
        body = email.plain_text or email.html_body
        prediction = predictor.predict(email.subject or "", body)
        probability = float(prediction.get("probability", prediction.get("phishing_probability", 0.0)))
        probability = max(0.0, min(1.0, probability))
        score = round(probability * 100)
        label = str(prediction.get("label", "")).lower()
        threshold: Optional[float] = prediction.get("threshold")
        indicators = []
        if label == "phishing" or prediction.get("is_phishing") is True:
            indicators.append(
                Indicator(
                    type="nlp_phishing_prediction",
                    description="Email language model classified the message as phishing-like",
                    severity=self._severity(score),
                    evidence={"probability": probability, "threshold": threshold},
                )
            )
        return DetectorResultData(
            detector=self.name,
            status="COMPLETED",
            risk_score=score,
            confidence=None,
            indicators=indicators,
            evidence={
                "label": label or None,
                "model_probability": probability,
                "decision_threshold": threshold,
                "calibrated_confidence": False,
                "model_source": str(self.model_path),
            },
        )
