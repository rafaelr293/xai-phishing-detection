from uuid import UUID

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.base import get_db
from app.db.models import EmailURL, ProcessingStatus, URLTraversal
from app.domain.schemas import URLTraversalResponse
from app.url_analysis.traversal import URLTraversalService

router = APIRouter(prefix="/api/emails", tags=["url-analysis"])


@router.post("/{email_id}/urls/{url_id}/traverse", response_model=URLTraversalResponse)
def traverse_selected_url(email_id: UUID, url_id: UUID, db: Session = Depends(get_db)) -> URLTraversalResponse:
    email_url = db.query(EmailURL).filter(EmailURL.id == url_id, EmailURL.email_id == email_id).first()
    if email_url is None:
        raise HTTPException(status_code=404, detail="Email URL not found")
    if not email_url.selected_for_traversal:
        raise HTTPException(status_code=409, detail="URL is not selected for traversal")

    traversal = URLTraversal(
        email_url_id=email_url.id,
        status=ProcessingStatus.RUNNING,
        requested_url=email_url.url,
    )
    db.add(traversal)
    db.commit()
    db.refresh(traversal)

    try:
        output = URLTraversalService(get_settings()).traverse(email_url.url, str(traversal.id))
        traversal.status = ProcessingStatus.COMPLETED
        traversal.final_url = output.final_url
        traversal.http_status = output.http_status
        traversal.page_title = output.page_title
        traversal.redirect_chain = output.redirect_chain
        traversal.blocked_requests = output.blocked_requests
        traversal.screenshot_path = output.screenshot_path
        traversal.screenshot_sha256 = output.screenshot_sha256
    except (ValueError, RuntimeError) as exc:
        traversal.status = ProcessingStatus.FAILED
        traversal.error_message = str(exc)

    db.commit()
    db.refresh(traversal)
    return URLTraversalResponse(
        traversal_id=traversal.id,
        status=traversal.status.value,
        requested_url=traversal.requested_url,
        final_url=traversal.final_url,
        http_status=traversal.http_status,
        page_title=traversal.page_title,
        redirect_chain=traversal.redirect_chain,
        blocked_requests=traversal.blocked_requests,
        screenshot_path=traversal.screenshot_path,
        screenshot_sha256=traversal.screenshot_sha256,
        error_message=traversal.error_message,
    )


@router.get("/traversals/{traversal_id}/screenshot")
def get_traversal_screenshot(traversal_id: UUID, db: Session = Depends(get_db)) -> FileResponse:
    traversal = db.get(URLTraversal, traversal_id)
    if traversal is None or not traversal.screenshot_path:
        raise HTTPException(status_code=404, detail="Screenshot not found")
    artifact_root = get_settings().artifact_directory.resolve()
    screenshot_path = (artifact_root / traversal.screenshot_path).resolve()
    if artifact_root not in screenshot_path.parents or not screenshot_path.is_file():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(screenshot_path, media_type="image/png", filename=screenshot_path.name)
