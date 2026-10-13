from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AttachmentData(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    detected_type: str = "unknown"
    archive_members: list[str] = Field(default_factory=list)


class ParsedEmail(BaseModel):
    content_hash: str
    source_path: str
    sender: Optional[str] = None
    recipients: list[str] = Field(default_factory=list)
    subject: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)
    plain_text: str = ""
    html_body: str = ""
    urls: list[str] = Field(default_factory=list)
    attachments: list[AttachmentData] = Field(default_factory=list)
    received_at: Optional[datetime] = None


class Indicator(BaseModel):
    type: str
    description: str
    severity: str
    evidence: dict = Field(default_factory=dict)


class DetectorResultData(BaseModel):
    detector: str
    status: str
    risk_score: Optional[int] = Field(default=None, ge=0, le=100)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    indicators: list[Indicator] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    error_message: Optional[str] = None


class EmailSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    score: Optional[int] = None
    level: Optional[str] = None
    sender: Optional[str] = None
    subject: Optional[str] = None
    time: Optional[datetime] = None
    status: str
    evidence: list[Indicator] = Field(default_factory=list)


class ScanResponse(BaseModel):
    job_id: UUID
    status: str
    discovered: int
    imported: int
    skipped: int
    failed: int


class RerunResponse(BaseModel):
    status: str
    requested: int
    rerun: int
    failed: int
