from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.analysis.attachment import AttachmentAnalyzer
from app.analysis.orchestrator import AnalysisOrchestrator
from app.core.config import get_settings
from app.db.base import get_db
from app.db.models import DetectorResult, Email, EmailAnalysis, IngestionJob, ProcessingStatus
from app.domain.schemas import ScanResponse
from app.ingestion.directory_ingestor import DirectoryIngestor

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])
orchestrator = AnalysisOrchestrator([AttachmentAnalyzer()])


def _attachment_analysis(parsed):
    analysis_results = orchestrator.analyze(parsed)
    analysis_status = ProcessingStatus.FAILED if any(result.status == "FAILED" for result in analysis_results) else ProcessingStatus.COMPLETED
    scores = [result.risk_score for result in analysis_results if result.risk_score is not None]
    risk_score = max(scores) if scores else None
    risk_level = next(
        (result.evidence.get("risk_level") for result in analysis_results if result.evidence.get("risk_level")),
        None,
    )
    analysis = EmailAnalysis(status=analysis_status, risk_score=risk_score, risk_level=risk_level)
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
                pending_analysis = next((analysis for analysis in existing.analyses if analysis.status == ProcessingStatus.PENDING), None)
                if pending_analysis:
                    refreshed_analysis = _attachment_analysis(parsed)
                    pending_analysis.status = refreshed_analysis.status
                    pending_analysis.risk_score = refreshed_analysis.risk_score
                    pending_analysis.risk_level = refreshed_analysis.risk_level
                    pending_analysis.detector_results.extend(
                        DetectorResult(
                            detector=result.detector,
                            status=result.status,
                            risk_score=result.risk_score,
                            confidence=result.confidence,
                            indicators=result.indicators,
                            evidence=result.evidence,
                            error_message=result.error_message,
                        )
                        for result in refreshed_analysis.detector_results
                    )
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
                )
                email.analyses.append(_attachment_analysis(parsed))
                db.add(email)
                db.flush()
                imported += 1
        except (OSError, ValueError):
            failed += 1

    job.imported = imported
    job.skipped = skipped
    job.failed = failed
    job.status = ProcessingStatus.PARTIAL if failed else ProcessingStatus.COMPLETED
    db.commit()
    return ScanResponse(job_id=job.id, status=job.status.value, discovered=len(paths), imported=imported, skipped=skipped, failed=failed)


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
