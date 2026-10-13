from datetime import datetime
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.analysis.attachment import AttachmentAnalyzer
from app.analysis.email_nlp import EmailNLPDetector
from app.analysis.explainability import ExplainabilityEngine
from app.analysis.risk_engine import RiskAssessmentEngine
from app.analysis.url_risk import URLRiskDetector
from app.analysis.visual import VisualDetector
from app.analysis.orchestrator import AnalysisOrchestrator
from app.core.config import get_settings
from app.db.base import get_db
from app.db.models import DetectorResult, Email, EmailAnalysis, EmailURL, IngestionJob, ProcessingStatus
from app.domain.schemas import AnalysisContext, RerunResponse, ScanResponse
from app.ingestion.directory_ingestor import DirectoryIngestor

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])
settings = get_settings()
detectors = [AttachmentAnalyzer()]
if settings.enable_email_nlp:
    detectors.insert(0, EmailNLPDetector(
        inference_module=settings.email_inference_directory / settings.email_inference_module,
        model_path=settings.email_model_path or settings.email_inference_directory / "phishing_model.joblib",
    ))
if settings.enable_url_risk:
    detectors.insert(0, URLRiskDetector(
        inference_module=settings.url_inference_directory / settings.url_inference_module,
        model_path=settings.url_model_path or settings.url_inference_directory / "url_pipeline.joblib",
    ))
if settings.enable_visual_analysis:
    detectors.insert(0, VisualDetector(
        model_path=settings.visual_model_path or settings.email_inference_directory / "phish360_efficientnet_b0_best.pt",
        artifact_directory=settings.artifact_directory,
    ))
orchestrator = AnalysisOrchestrator(detectors)
risk_engine = RiskAssessmentEngine()
explainability_engine = ExplainabilityEngine()


def _run_analysis(db: Session, email: Email, parsed):
    _ensure_url_records(email, parsed.urls)
    context = AnalysisContext(
        email=parsed,
        email_id=email.id,
        urls=[
            {
                "id": url_record.id,
                "url": url_record.url,
                "selected_for_traversal": url_record.selected_for_traversal,
                "traversals": [],
            }
            for url_record in email.url_records
        ],
    )
    analysis_results = orchestrator.analyze(context)
    url_result = next((result for result in analysis_results if result.detector == "url_risk"), None)
    if url_result is not None:
        url_records = {url_record.url.strip().lower(): url_record for url_record in email.url_records}
        for url_record in email.url_records:
            url_record.selected_for_traversal = False
        for result in url_result.evidence.get("url_results", []):
            url_record = url_records.get(str(result.get("url", "")).strip().lower())
            if url_record is not None:
                url_record.selected_for_traversal = str(result.get("label", "")).lower() == "phishing"

    assessment = risk_engine.assess(analysis_results, [detector.name for detector in detectors])
    explanations = explainability_engine.explain(assessment)
    analysis = EmailAnalysis(
        status=ProcessingStatus(assessment.status),
        risk_score=assessment.final_score,
        risk_level=assessment.risk_level,
        aggregation_details=assessment.model_dump(),
        technical_explanation=explanations["technical"],
        user_explanation=explanations["user"],
    )
    for result in analysis_results:
        analysis.detector_results.append(
            DetectorResult(
                detector=result.detector,
                status=ProcessingStatus(result.status),
                risk_score=result.risk_score,
                confidence=result.confidence,
                indicators=[indicator.model_dump() for indicator in result.indicators],
                evidence=result.evidence,
                error_message=result.error_message,
            )
        )
    return analysis


def _ensure_url_records(email: Email, urls: list[str]) -> None:
    existing_urls = {record.url.strip().lower() for record in email.url_records}
    for url in urls:
        if url.strip().lower() not in existing_urls:
            email.url_records.append(EmailURL(url=url))


def _refresh_analysis(db: Session, email: Email, analysis: EmailAnalysis, parsed) -> None:
    refreshed_analysis = _run_analysis(db, email, parsed)
    analysis.status = refreshed_analysis.status
    analysis.risk_score = refreshed_analysis.risk_score
    analysis.risk_level = refreshed_analysis.risk_level
    analysis.aggregation_details = refreshed_analysis.aggregation_details
    analysis.technical_explanation = refreshed_analysis.technical_explanation
    analysis.user_explanation = refreshed_analysis.user_explanation
    current_results = {result.detector: result for result in analysis.detector_results}
    for result in refreshed_analysis.detector_results:
        detector_result = current_results.pop(result.detector, None)
        if detector_result is None:
            detector_result = DetectorResult(detector=result.detector)
            analysis.detector_results.append(detector_result)
        detector_result.status = result.status
        detector_result.risk_score = result.risk_score
        detector_result.confidence = result.confidence
        detector_result.indicators = result.indicators
        detector_result.evidence = result.evidence
        detector_result.error_message = result.error_message
    for obsolete_result in current_results.values():
        db.delete(obsolete_result)


@router.post("/scan", response_model=ScanResponse, status_code=202)
def scan_directory(db: Session = Depends(get_db)) -> ScanResponse:
    settings = get_settings()
    ingestor = DirectoryIngestor(settings)
    paths = ingestor.discover()
    job = IngestionJob(discovered=len(paths), status=ProcessingStatus.COMPLETED)
    db.add(job)
    db.commit()
    db.refresh(job)

    imported = 0
    skipped = 0
    failed = 0
    for path in paths:
        try:
            parsed = ingestor.parse(path)
            existing = db.query(Email).filter(Email.content_hash == parsed.content_hash).first()
            if existing:
                _ensure_url_records(existing, parsed.urls)
                pending_analysis = next((analysis for analysis in existing.analyses if analysis.status == ProcessingStatus.PENDING), None)
                if pending_analysis:
                    _refresh_analysis(db, existing, pending_analysis, parsed)
                skipped += 1
            else:
                email = Email(
                    content_hash=parsed.content_hash,
                    source_path=parsed.source_path,
                    sender=parsed.sender,
                    recipients=parsed.recipients,
                    subject=parsed.subject,
                    headers=parsed.headers,
                    plain_text=parsed.plain_text,
                    html_body=parsed.html_body,
                    urls=parsed.urls,
                    attachments=[attachment.model_dump() for attachment in parsed.attachments],
                    received_at=parsed.received_at,
                    url_records=[EmailURL(url=url) for url in parsed.urls],
                )
                db.add(email)
                db.flush()
                email.analyses.append(_run_analysis(db, email, parsed))
                imported += 1
        except (OSError, ValueError):
            failed += 1

    job.imported = imported
    job.skipped = skipped
    job.failed = failed
    job.status = ProcessingStatus.PARTIAL if failed else ProcessingStatus.COMPLETED
    db.commit()
    return ScanResponse(job_id=job.id, status=job.status.value, discovered=len(paths), imported=imported, skipped=skipped, failed=failed)


@router.post("/rerun-analysis", response_model=RerunResponse)
def rerun_analysis(db: Session = Depends(get_db)) -> RerunResponse:
    emails = db.query(Email).all()
    rerun = 0
    failed = 0
    settings = get_settings()
    ingestor = DirectoryIngestor(settings)

    for email in emails:
        try:
            parsed = ingestor.parse(Path(email.source_path))
            _ensure_url_records(email, parsed.urls)
            analysis = max(email.analyses, key=lambda item: item.created_at or datetime.min)
            _refresh_analysis(db, email, analysis, parsed)
            rerun += 1
        except (OSError, ValueError):
            failed += 1

    db.commit()
    return RerunResponse(status="COMPLETED" if failed == 0 else "PARTIAL", requested=len(emails), rerun=rerun, failed=failed)


@router.get("/jobs/{job_id}")
def get_scan_job(job_id: UUID, db: Session = Depends(get_db)) -> dict:
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Scan job not found")
    return {
        "job_id": job.id,
        "status": job.status.value,
        "discovered": job.discovered,
        "imported": job.imported,
        "skipped": job.skipped,
        "failed": job.failed,
    }
