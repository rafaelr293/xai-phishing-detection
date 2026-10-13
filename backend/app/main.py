from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.emails import router as emails_router
from app.api.routes.url_analysis import router as url_analysis_router
from app.api.routes.settings import router as settings_router

settings = get_settings()
app = FastAPI(title="XAI Phishing Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ingestion_router)
app.include_router(emails_router)
app.include_router(url_analysis_router)
app.include_router(settings_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/readiness")
def readiness() -> dict[str, str]:
    return {"status": "ready"}
