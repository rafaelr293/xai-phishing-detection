from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.db.base import get_db
from app.db.models import Email, EmailAnalysis

router = APIRouter(prefix="/api/emails", tags=["emails"])


def _summary(email: Email) -> dict:
    analysis = email.analyses[0] if email.analyses else None
    evidence = []
    if analysis:
        for result in analysis.detector_results:
            evidence.extend(result.indicators or [])
    return {
        "id": email.id,
        "score": analysis.risk_score if analysis else None,
        "level": analysis.risk_level if analysis else None,
        "sender": email.sender,
        "subject": email.subject,
        "time": email.received_at or email.created_at,
        "status": analysis.status.value if analysis else "PENDING",
        "evidence": evidence,
    }


@router.get("")
def list_emails(
    sender: Optional[str] = None,
    subject: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    query = db.query(Email).options(joinedload(Email.analyses).joinedload(EmailAnalysis.detector_results))
    if sender:
        query = query.filter(Email.sender.ilike(f"%{sender}%"))
    if subject:
        query = query.filter(Email.subject.ilike(f"%{subject}%"))
    emails = query.order_by(Email.created_at.desc()).offset(offset).limit(limit).all()
    return {"items": [_summary(email) for email in emails], "limit": limit, "offset": offset}


@router.get("/{email_id}")
def get_email(email_id: UUID, db: Session = Depends(get_db)) -> dict:
    email = (
        db.query(Email)
        .options(joinedload(Email.analyses).joinedload(EmailAnalysis.detector_results))
        .filter(Email.id == email_id)
        .first()
    )
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")
    return {
        **_summary(email),
        "recipients": email.recipients,
        "headers": email.headers,
        "plain_text": email.plain_text,
        "html_body": email.html_body,
        "urls": email.urls,
        "url_records": [
            {
                "id": url_record.id,
                "url": url_record.url,
                "selected_for_traversal": url_record.selected_for_traversal,
                "traversals": [
                    {
                        "id": traversal.id,
                        "status": traversal.status.value,
                        "requested_url": traversal.requested_url,
                        "final_url": traversal.final_url,
                        "http_status": traversal.http_status,
                        "page_title": traversal.page_title,
                        "redirect_chain": traversal.redirect_chain,
                        "blocked_requests": traversal.blocked_requests,
                        "screenshot_path": traversal.screenshot_path,
                        "screenshot_sha256": traversal.screenshot_sha256,
                        "error_message": traversal.error_message,
                    }
                    for traversal in url_record.traversals
                ],
            }
            for url_record in email.url_records
        ],
        "attachments": email.attachments,
        "analyses": [
            {
                "id": analysis.id,
                "status": analysis.status.value,
                "risk_score": analysis.risk_score,
                "risk_level": analysis.risk_level,
                "detectors": [
                    {
                        "detector": result.detector,
                        "status": result.status.value,
                        "risk_score": result.risk_score,
                        "confidence": result.confidence,
                        "indicators": result.indicators,
                        "evidence": result.evidence,
                        "error_message": result.error_message,
                    }
                    for result in analysis.detector_results
                ],
            }
            for analysis in email.analyses
        ],
    }
