from app.analysis.orchestrator import AnalysisOrchestrator
from app.domain.schemas import DetectorResultData, ParsedEmail


class FailingDetector:
    name = "failing_detector"

    def analyze(self, email: ParsedEmail) -> DetectorResultData:
        raise RuntimeError("detector unavailable")


def test_detector_failure_is_returned_as_failed_result() -> None:
    email = ParsedEmail(content_hash="a" * 64, source_path="sample.eml")

    results = AnalysisOrchestrator([FailingDetector()]).analyze(email)

    assert results[0].detector == "failing_detector"
    assert results[0].status == "FAILED"
    assert results[0].error_message == "detector unavailable"
