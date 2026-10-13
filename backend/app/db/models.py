from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_path: Mapped[str] = mapped_column(Text)
    sender: Mapped[Optional[str]] = mapped_column(String(512))
    recipients: Mapped[list[str]] = mapped_column(JSON, default=list)
    subject: Mapped[Optional[str]] = mapped_column(Text)
    headers: Mapped[dict] = mapped_column(JSON, default=dict)
    plain_text: Mapped[str] = mapped_column(Text, default="")
    html_body: Mapped[str] = mapped_column(Text, default="")
    urls: Mapped[list[str]] = mapped_column(JSON, default=list)
    attachments: Mapped[list] = mapped_column(JSON, default=list)
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analyses: Mapped[list["EmailAnalysis"]] = relationship(back_populates="email", cascade="all, delete-orphan")


class EmailAnalysis(Base):
    __tablename__ = "email_analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email_id: Mapped[UUID] = mapped_column(ForeignKey("emails.id"), index=True)
    status: Mapped[ProcessingStatus] = mapped_column(default=ProcessingStatus.PENDING)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer)
    risk_level: Mapped[Optional[str]] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    email: Mapped[Email] = relationship(back_populates="analyses")
    detector_results: Mapped[list["DetectorResult"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")


class DetectorResult(Base):
    __tablename__ = "detector_results"
    __table_args__ = (UniqueConstraint("analysis_id", "detector", name="uq_analysis_detector"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    analysis_id: Mapped[UUID] = mapped_column(ForeignKey("email_analyses.id"), index=True)
    detector: Mapped[str] = mapped_column(String(128))
    status: Mapped[ProcessingStatus] = mapped_column(default=ProcessingStatus.PENDING)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer)
    confidence: Mapped[Optional[float]]
    indicators: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    analysis: Mapped[EmailAnalysis] = relationship(back_populates="detector_results")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    status: Mapped[ProcessingStatus] = mapped_column(default=ProcessingStatus.RUNNING)
    discovered: Mapped[int] = mapped_column(Integer, default=0)
    imported: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
