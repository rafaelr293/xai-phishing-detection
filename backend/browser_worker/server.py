from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import UUID

from app.core.config import get_settings
from app.url_analysis.traversal import URLTraversalService

app = FastAPI(title="XAI Browser Worker", version="0.1.0")
service = URLTraversalService(get_settings())


class TraversalRequest(BaseModel):
    requested_url: str
    traversal_id: UUID


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/traverse")
def traverse(request: TraversalRequest) -> dict:
    try:
        output = service.traverse(request.requested_url, str(request.traversal_id))
        return asdict(output)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
