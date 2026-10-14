from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.analysis.explainability import ExplainabilityEngine
from app.analysis.risk_engine import url_analysis_status
from app.analysis.risk_engine import RiskAssessmentEngine
from app.core.config import get_settings
from app.db.base import SessionLocal, get_db
from app.db.models import DetectorResult, EmailAnalysis, EmailURL, ProcessingStatus, URLTraversal
from app.domain.schemas import DetectorResultData, URLTraversalResponse
from app.browser_worker.client import BrowserWorkerClient
from app.visual_worker_client import VisualWorkerClient

router = APIRouter(prefix="/api/emails", tags=["url-analysis"])
risk_engine = RiskAssessmentEngine()
explainability_engine = ExplainabilityEngine()


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
        _refresh_email_analysis(traversal)
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


def _refresh_email_analysis(traversal: URLTraversal) -> None:
    email = traversal.email_url.email
    if not email.analyses:
        return
    analysis = max(
        email.analyses,
        key=lambda item: item.updated_at or item.created_at or datetime.min.replace(tzinfo=timezone.utc),
    )
    visual_results = []
    visual_scores = []
    visual_indicators = []
    visual_failed = False
    selected_records = [record for record in email.url_records if record.selected_for_traversal]
    for record in selected_records:
        for record_traversal in record.traversals:
            visual = record_traversal.visual_analysis or {}
            if visual.get("status") == "COMPLETED":
                visual_scores.append(visual.get("risk_score"))
                visual_indicators.extend(visual.get("indicators", []))
                visual_results.extend(visual.get("evidence", {}).get("visual_results", []))
            elif record_traversal.status == ProcessingStatus.COMPLETED.value:
                visual_failed = True

    complete_visuals = all(
        any(
            (record_traversal.visual_analysis or {}).get("status") == "COMPLETED"
            for record_traversal in record.traversals
        )
        for record in selected_records
    )
    visual_status = "COMPLETED" if complete_visuals else "FAILED" if visual_failed and not visual_scores else "PARTIAL"
    visual_result = next(
        (result for result in analysis.detector_results if result.detector == "visual_analysis"),
        None,
    )
    if visual_result is None:
        visual_result = DetectorResult(detector="visual_analysis")
        analysis.detector_results.append(visual_result)
    visual_result.status = ProcessingStatus(visual_status)
    visual_result.risk_score = max(visual_scores, default=None)
    visual_result.confidence = None
    visual_result.indicators = visual_indicators
    visual_result.evidence = {
        "screenshot_count": len(visual_results),
        "visual_results": visual_results,
        "risk_level": risk_engine.risk_level(visual_result.risk_score),
        "calibrated_confidence": False,
    }
    visual_result.error_message = "One or more visual analyses failed" if visual_failed else None

    results = [
        DetectorResultData(
            detector=result.detector,
            status=result.status.value,
            risk_score=result.risk_score,
            confidence=result.confidence,
            indicators=result.indicators,
            evidence=result.evidence,
            error_message=result.error_message,
        )
        for result in analysis.detector_results
    ]
    assessment = risk_engine.assess(results, [result.detector for result in results])
    assessment.status = url_analysis_status(assessment.status, email.url_records)
    explanations = explainability_engine.explain(assessment)
    analysis.status = ProcessingStatus(assessment.status)
    analysis.risk_score = assessment.final_score
    analysis.risk_level = assessment.risk_level
    analysis.aggregation_details = assessment.model_dump()
    analysis.technical_explanation = explanations["technical"]
    analysis.user_explanation = explanations["user"]


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


@router.get("/traversals/{traversal_id}/heatmap")
def get_traversal_heatmap(traversal_id: UUID, db: Session = Depends(get_db)) -> FileResponse:
    traversal = db.get(URLTraversal, traversal_id)
    if traversal is None:
        raise HTTPException(status_code=404, detail="Traversal not found")
    visual_results = (traversal.visual_analysis or {}).get("evidence", {}).get("visual_results", [])
    heatmap_path_value = next((result.get("heatmap_path") for result in visual_results if result.get("heatmap_path")), None)
    if not heatmap_path_value:
        raise HTTPException(status_code=404, detail="Heatmap not found")
    artifact_root = get_settings().artifact_directory.resolve()
    heatmap_path = (artifact_root / heatmap_path_value).resolve()
    if artifact_root not in heatmap_path.parents or not heatmap_path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(heatmap_path, media_type="image/png", filename=heatmap_path.name)
