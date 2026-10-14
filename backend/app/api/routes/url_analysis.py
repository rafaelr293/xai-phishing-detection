from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.analysis.risk_engine import url_analysis_status
from app.core.config import get_settings
from app.db.base import SessionLocal, get_db
from app.db.models import EmailURL, ProcessingStatus, URLTraversal
from app.domain.schemas import DetectorResultData, URLTraversalResponse
from app.browser_worker.client import BrowserWorkerClient
from app.visual_worker_client import VisualWorkerClient

router = APIRouter(prefix="/api/emails", tags=["url-analysis"])


def _run_traversal_job(traversal_id: UUID) -> None:
    settings = get_settings()
    with SessionLocal() as db:
        traversal = db.get(URLTraversal, traversal_id)
        if traversal is None:
            return
        traversal.status = ProcessingStatus.RUNNING
        db.commit()
        try:
            if not settings.browser_worker_url:
                raise RuntimeError("Browser worker URL is not configured")
            output = BrowserWorkerClient(
                settings.browser_worker_url,
                settings.browser_worker_timeout_seconds,
            ).traverse(traversal.requested_url, str(traversal.id))
            traversal.status = ProcessingStatus.COMPLETED
            traversal.final_url = output.get("final_url")
            traversal.http_status = output.get("http_status")
            traversal.page_title = output.get("page_title")
            traversal.redirect_chain = output.get("redirect_chain", [])
            traversal.blocked_requests = output.get("blocked_requests", [])
            traversal.screenshot_path = output.get("screenshot_path")
            traversal.screenshot_sha256 = output.get("screenshot_sha256")
            _run_visual_analysis(db, traversal, settings)
        except Exception as exc:
            traversal.status = ProcessingStatus.FAILED
            traversal.error_message = str(exc)
            if traversal.screenshot_path:
                screenshot_file = settings.artifact_directory / traversal.screenshot_path
                try:
                    screenshot_file.unlink(missing_ok=True)
                except OSError:
                    pass
                traversal.screenshot_path = None
                traversal.screenshot_sha256 = None
            _refresh_email_status(traversal)
        traversal.completed_at = datetime.now(timezone.utc)
        db.commit()


def _run_visual_analysis(db: Session, traversal: URLTraversal, settings) -> None:
    if not settings.visual_worker_url or not traversal.screenshot_path:
        return
    result_data: DetectorResultData
    try:
        output = VisualWorkerClient(
            settings.visual_worker_url,
            settings.visual_worker_timeout_seconds,
        ).predict(
            str(traversal.id),
            traversal.screenshot_path,
            traversal.requested_url,
        )
        result_data = DetectorResultData.model_validate(output)
    except Exception as exc:
        result_data = DetectorResultData(
            detector="visual_analysis",
            status="FAILED",
            error_message=str(exc),
        )

    traversal.visual_analysis = result_data.model_dump()


def _refresh_email_status(traversal: URLTraversal) -> None:
    email = traversal.email_url.email
    if not email.analyses:
        return
    analysis = max(
        email.analyses,
        key=lambda item: item.updated_at or item.created_at or datetime.min.replace(tzinfo=timezone.utc),
    )
    coverage = analysis.aggregation_details.get("coverage", {})
    if coverage.get("failed_detectors") or coverage.get("missing_detectors"):
        return
    analysis.status = ProcessingStatus(url_analysis_status(analysis.status.value, email.url_records))


@router.post("/{email_id}/urls/{url_id}/traverse", response_model=URLTraversalResponse, status_code=202)
def traverse_selected_url(
    email_id: UUID,
    url_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> URLTraversalResponse:
    email_url = db.query(EmailURL).filter(EmailURL.id == url_id, EmailURL.email_id == email_id).first()
    if email_url is None:
        raise HTTPException(status_code=404, detail="Email URL not found")
    if not email_url.selected_for_traversal:
        raise HTTPException(status_code=409, detail="URL is not selected for traversal")

    traversal = URLTraversal(
        email_url_id=email_url.id,
        status=ProcessingStatus.PENDING,
        requested_url=email_url.url,
    )
    db.add(traversal)
    db.commit()
    db.refresh(traversal)

    background_tasks.add_task(_run_traversal_job, traversal.id)
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
        visual_analysis=traversal.visual_analysis,
        error_message=traversal.error_message,
    )


@router.get("/traversals/{traversal_id}", response_model=URLTraversalResponse)
def get_traversal(traversal_id: UUID, db: Session = Depends(get_db)) -> URLTraversalResponse:
    traversal = db.get(URLTraversal, traversal_id)
    if traversal is None:
        raise HTTPException(status_code=404, detail="Traversal not found")
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
        visual_analysis=traversal.visual_analysis,
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
