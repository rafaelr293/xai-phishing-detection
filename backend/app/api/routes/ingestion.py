from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.base import get_db
from app.db.models import Email, EmailAnalysis, IngestionJob, ProcessingStatus
from app.domain.schemas import ScanResponse
from app.ingestion.directory_ingestor import DirectoryIngestor

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


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
                email.analyses.append(EmailAnalysis(status=ProcessingStatus.PENDING))
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
