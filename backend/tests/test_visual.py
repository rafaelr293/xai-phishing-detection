from pathlib import Path

from app.analysis.visual import VisualDetector
from app.domain.schemas import AnalysisContext, ParsedEmail


def test_visual_detector_skips_when_no_completed_screenshot_exists(tmp_path: Path) -> None:
    detector = VisualDetector(tmp_path / "model.pt", tmp_path)
    context = AnalysisContext(email=ParsedEmail(content_hash="a" * 64, source_path="sample.eml"))

    result = detector.analyze(context)

    assert result.detector == "visual_analysis"
    assert result.status == "COMPLETED"
    assert result.risk_score == 0
    assert result.evidence["analysis_skipped"] is True
