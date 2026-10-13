from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.analysis.visual import VisualDetector
from app.domain.schemas import AnalysisContext, DetectorResultData, ParsedEmail, TraversalContext, URLContext

ARTIFACT_DIRECTORY = Path(os.getenv("ARTIFACT_DIRECTORY", "/app/data/artifacts")).resolve()
MODEL_PATH = Path(os.getenv(
    "VISUAL_MODEL_PATH",
    "/app/Tan Files/phish360_efficientnet_b0_best.pt",
)).resolve()
detector = VisualDetector(MODEL_PATH, ARTIFACT_DIRECTORY)

app = FastAPI(title="XAI Visual Analysis Worker", version="0.1.0")


class VisualRequest(BaseModel):
    traversal_id: UUID
    screenshot_path: str
    url: str = ""


def _safe_screenshot_path(path: str) -> str:
    candidate = (ARTIFACT_DIRECTORY / path).resolve()
    if ARTIFACT_DIRECTORY not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Screenshot artifact not found")
    return str(candidate.relative_to(ARTIFACT_DIRECTORY))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=DetectorResultData)
def predict(request: VisualRequest) -> DetectorResultData:
    safe_path = _safe_screenshot_path(request.screenshot_path)
    context = AnalysisContext(
        email=ParsedEmail(content_hash="0" * 64, source_path="visual-worker"),
        urls=[
            URLContext(
                id=uuid4(),
                url=request.url,
                traversals=[
                    TraversalContext(
                        id=request.traversal_id,
                        email_url_id=uuid4(),
                        requested_url=request.url,
                        screenshot_path=safe_path,
                        status="COMPLETED",
                    )
                ],
            )
        ],
    )
    try:
        return detector.analyze(context)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
