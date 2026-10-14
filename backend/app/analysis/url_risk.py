from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

from app.domain.schemas import AnalysisContext, DetectorResultData, Indicator


class URLRiskDetector:
    """Adapter for Tan's URL_Inference scan_urls pipeline."""

    name = "url_risk"

    def __init__(self, inference_module: Path, model_path: Path, pipeline: Any = None):
        self.inference_module = inference_module
        self.model_path = model_path
        self._pipeline = pipeline
        self._module = None

    def _load_pipeline(self) -> tuple[Any, Any]:
        if self._pipeline is not None and self._module is not None:
            return self._module, self._pipeline
        if not self.inference_module.is_file():
            raise FileNotFoundError(f"URL inference module not found: {self.inference_module}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"URL model artifact not found: {self.model_path}")

        module_directory = self.inference_module.parent
        if str(module_directory) not in sys.path:
            sys.path.insert(0, str(module_directory))
        features_path = module_directory / "URL_Features.py"
        if features_path.is_file():
            features_spec = importlib.util.spec_from_file_location("URL_Features", features_path)
            if features_spec and features_spec.loader:
                features_module = importlib.util.module_from_spec(features_spec)
                sys.modules["URL_Features"] = features_module
                features_spec.loader.exec_module(features_module)

        loader = importlib.machinery.SourceFileLoader("xai_url_inference", str(self.inference_module))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load URL inference module: {self.inference_module}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        load_pipeline = getattr(module, "load_pipeline", None)
        scan_urls = getattr(module, "scan_urls", None)
        if not callable(load_pipeline) or not callable(scan_urls):
            raise AttributeError("URL inference module must expose load_pipeline and scan_urls")
        self._module = module
        self._pipeline = load_pipeline(str(self.model_path))
        return module, self._pipeline

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 75:
            return "critical"
        if score >= 50:
            return "high"
        if score >= 25:
            return "medium"
        return "low"

    @staticmethod
    def _risk_score(probability: float, threshold: Optional[float]) -> int:
        if threshold is None:
            return round(probability * 100)
        if probability < threshold:
            return 0
        if threshold >= 1:
            return 100
        return max(1, min(100, round(1 + ((probability - threshold) / (1 - threshold)) * 99)))

    def analyze(self, context: AnalysisContext) -> DetectorResultData:
        urls = context.email.urls
        if not urls:
            return DetectorResultData(
                detector=self.name,
                status="COMPLETED",
                risk_score=0,
                evidence={"url_count": 0, "risk_level": "low", "url_results": []},
            )

        module, pipeline = self._load_pipeline()
        rows = module.scan_urls(urls, pipeline).to_dict(orient="records")
        highest_probability_row = max(rows, key=lambda row: float(row.get("phishing_probability", 0.0)), default=None)
        max_probability = float(highest_probability_row.get("phishing_probability", 0.0)) if highest_probability_row else 0.0
        max_probability = max(0.0, min(1.0, max_probability))
        highest_threshold = highest_probability_row.get("threshold") if highest_probability_row else None
        if highest_threshold is not None:
            highest_threshold = max(0.0, min(1.0, float(highest_threshold)))
        score = self._risk_score(max_probability, highest_threshold)
        indicators = [
            Indicator(
                type="suspicious_url",
                description="URL risk model classified a destination as phishing-like",
                severity=self._risk_level(self._risk_score(
                    max(0.0, min(1.0, float(row.get("phishing_probability", 0.0)))),
                    max(0.0, min(1.0, float(row["threshold"]))) if row.get("threshold") is not None else None,
                )),
                evidence={
                    "url": row.get("url"),
                    "canonical": row.get("canonical"),
                    "probability": row.get("phishing_probability"),
                    "threshold": row.get("threshold"),
                    "suspicious_words_host": row.get("suspicious_words_host"),
                    "suspicious_words_path": row.get("suspicious_words_path"),
                },
            )
            for row in rows
            if str(row.get("label", "")).lower() == "phishing"
        ]
        return DetectorResultData(
            detector=self.name,
            status="COMPLETED",
            risk_score=score,
            confidence=None,
            indicators=indicators,
            evidence={
                "url_count": len(rows),
                "risk_level": self._risk_level(score),
                "url_results": rows,
                "calibrated_confidence": False,
            },
        )
