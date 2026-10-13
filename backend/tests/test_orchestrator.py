from app.analysis.orchestrator import AnalysisOrchestrator
from uuid import uuid4

from app.domain.schemas import AnalysisContext, DetectorResultData, ParsedEmail


class FailingDetector:
    name = "failing_detector"

    def analyze(self, email: ParsedEmail) -> DetectorResultData:
        raise RuntimeError("detector unavailable")


class ContextDetector:
    name = "context_detector"

    def __init__(self) -> None:
        self.context = None

    def analyze(self, context: AnalysisContext) -> DetectorResultData:
        self.context = context
        return DetectorResultData(detector=self.name, status="COMPLETED")


def test_detector_failure_is_returned_as_failed_result() -> None:
    email = ParsedEmail(content_hash="a" * 64, source_path="sample.eml")

    results = AnalysisOrchestrator([FailingDetector()]).analyze(email)

    assert results[0].detector == "failing_detector"
    assert results[0].status == "FAILED"
    assert results[0].error_message == "detector unavailable"


def test_orchestrator_passes_structured_analysis_context() -> None:
    detector = ContextDetector()
    email_id = uuid4()
    email = ParsedEmail(content_hash="a" * 64, source_path="sample.eml")

    results = AnalysisOrchestrator([detector]).analyze(
        AnalysisContext(email=email, email_id=email_id)
    )

    assert results[0].status == "COMPLETED"
    assert detector.context.email_id == email_id
    assert detector.context.email.content_hash == email.content_hash
