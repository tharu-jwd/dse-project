from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.media import MediaFile
    from app.models.quiz import AnswerSubmission
    from app.models.user import User


class Transcript(Base):
    __tablename__ = "transcripts"
    __table_args__ = (
        CheckConstraint(
            "transcript_type IN ('LECTURE', 'NOTE', 'QUIZ_ANSWER')",
            name="ck_transcripts_type",
        ),
        CheckConstraint(
            "status IN ('DRAFT', 'FINALIZED')",
            name="ck_transcripts_status",
        ),
        CheckConstraint(
            "source IS NULL OR source IN ('UPLOAD', 'LIVE')",
            name="ck_transcripts_source",
        ),
    )

    transcript_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    media_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("media_files.media_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    transcript_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        default="DRAFT",
        server_default=text("'DRAFT'"),
        nullable=False,
        index=True,
    )
    source: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    owner: Mapped[User] = relationship(back_populates="transcripts")
    media_file: Mapped[MediaFile | None] = relationship(
        back_populates="transcripts",
    )
    transcription_job: Mapped[TranscriptionJob | None] = relationship(
        back_populates="transcript",
        uselist=False,
    )
    segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.segment_order",
    )
    export_records: Mapped[list[ExportRecord]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
    )
    answer_submissions: Mapped[list[AnswerSubmission]] = relationship(
        back_populates="transcription",
    )


class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"
    __table_args__ = (
        UniqueConstraint(
            "transcript_id",
            name="uq_transcription_jobs_transcript_id",
        ),
        CheckConstraint(
            "status IN "
            "('UPLOADING', 'QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED')",
            name="ck_transcription_jobs_status",
        ),
        CheckConstraint(
            "transcript_type IN ('LECTURE', 'NOTE', 'QUIZ_ANSWER')",
            name="ck_transcription_jobs_type",
        ),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_transcription_jobs_progress",
        ),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    media_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("media_files.media_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    requested_by: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    transcript_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("transcripts.transcript_id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    transcript_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="QUEUED",
        server_default=text("'QUEUED'"),
        nullable=False,
        index=True,
    )
    progress_percent: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    media_file: Mapped[MediaFile] = relationship(
        back_populates="transcription_jobs",
    )
    requester: Mapped[User] = relationship(
        back_populates="transcription_jobs",
    )
    transcript: Mapped[Transcript | None] = relationship(
        back_populates="transcription_job",
        uselist=False,
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint(
            "transcript_id",
            "segment_order",
            name="uq_transcript_segments_order",
        ),
        CheckConstraint(
            "segment_order >= 0",
            name="ck_transcript_segments_order",
        ),
        CheckConstraint(
            "start_time >= 0",
            name="ck_transcript_segments_start_time",
        ),
        CheckConstraint(
            "end_time >= start_time",
            name="ck_transcript_segments_time_range",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_transcript_segments_confidence",
        ),
    )

    segment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    transcript_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("transcripts.transcript_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_order: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_text: Mapped[str] = mapped_column(Text, nullable=False)
    edited_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    word_metadata: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    transcript: Mapped[Transcript] = relationship(back_populates="segments")


class ExportRecord(Base):
    __tablename__ = "export_records"
    __table_args__ = (
        CheckConstraint(
            "export_format IN ('TXT', 'DOCX', 'PDF')",
            name="ck_export_records_format",
        ),
    )

    export_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    transcript_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("transcripts.transcript_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    requested_by: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    export_format: Mapped[str] = mapped_column(String(10), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    exported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    transcript: Mapped[Transcript] = relationship(
        back_populates="export_records",
    )
    requester: Mapped[User] = relationship(
        back_populates="export_records",
    )
