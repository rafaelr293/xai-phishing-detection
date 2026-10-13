from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

from app.domain.schemas import AnalysisContext, DetectorResultData, Indicator


class VisualDetector:
    """Adapter for the teammate's EfficientNet screenshot classifier."""

    name = "visual_analysis"

    def __init__(self, model_path: Path, artifact_directory: Path, model: Any = None, model_module: Any = None):
        self.model_path = model_path
        self.artifact_directory = artifact_directory
        self._model = model
        self._model_module = model_module
        self._device = None

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._model_module is not None:
            return self._model, self._device, self._model_module
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Visual model artifact not found: {self.model_path}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for visual analysis") from exc

        module_path = self.model_path.parent / "Visual_Model.py"
        if not module_path.is_file():
            raise FileNotFoundError(f"Visual model module not found: {module_path}")
        if str(module_path.parent) not in sys.path:
            sys.path.insert(0, str(module_path.parent))
        spec = importlib.util.spec_from_file_location("xai_visual_model", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load visual model module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = getattr(module, "load_checkpoint", None)
        if not callable(loader):
            raise AttributeError("Visual model module does not expose load_checkpoint")
        self._model = loader(str(self.model_path), device)
        self._model_module = module
        self._device = device
        return self._model, self._device, self._model_module

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 85:
            return "critical"
        if score >= 60:
            return "high"
        if score >= 35:
            return "medium"
        return "low"

    def _screenshot_inputs(self, context: AnalysisContext) -> list[tuple[str, Path, str]]:
        inputs = []
        for url_context in context.urls:
            for traversal in url_context.traversals:
                if traversal.status != "COMPLETED" or not traversal.screenshot_path:
                    continue
                path = Path(traversal.screenshot_path)
                if not path.is_absolute():
                    path = self.artifact_directory / path
                inputs.append((url_context.url, path.resolve(), str(traversal.id)))
        return inputs

    def analyze(self, context: AnalysisContext) -> DetectorResultData:
        inputs = self._screenshot_inputs(context)
        if not inputs:
            return DetectorResultData(
                detector=self.name,
                status="COMPLETED",
                risk_score=0,
                evidence={"screenshot_count": 0, "analysis_skipped": True, "risk_level": "low"},
            )

        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("PyTorch and Pillow are required for visual analysis") from exc

        model, device, model_module = self._load_model()
        transform = model_module.eval_transform
        results = []
        for url, image_path, traversal_id in inputs:
            if not image_path.is_file():
                results.append({"url": url, "traversal_id": traversal_id, "error": "Screenshot artifact not found"})
                continue
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
            phishing_probability = float(probabilities[1])
            results.append({
                "url": url,
                "traversal_id": traversal_id,
                "image_path": str(image_path),
                "label": "phishing" if phishing_probability >= 0.5 else "legitimate",
                "phishing_probability": phishing_probability,
                "legitimate_probability": float(probabilities[0]),
            })

        valid_results = [result for result in results if "phishing_probability" in result]
        max_probability = max((result["phishing_probability"] for result in valid_results), default=0.0)
        score = round(max_probability * 100)
        indicators = [
            Indicator(
                type="visual_phishing_prediction",
                description="Computer vision model classified a captured webpage as phishing-like",
                severity=self._risk_level(round(result["phishing_probability"] * 100)),
                evidence=result,
            )
            for result in valid_results
            if result["label"] == "phishing"
        ]
        return DetectorResultData(
            detector=self.name,
            status="COMPLETED" if valid_results else "FAILED",
            risk_score=score if valid_results else None,
            confidence=None,
            indicators=indicators,
            evidence={
                "screenshot_count": len(inputs),
                "visual_results": results,
                "risk_level": self._risk_level(score),
                "calibrated_confidence": False,
            },
            error_message=None if valid_results else "No screenshot could be analyzed",
        )
