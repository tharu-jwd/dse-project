from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, DateTime, String, UniqueConstraint, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.media import MediaFile
    from app.models.quiz import Quiz, QuizSubmission
    from app.models.transcription import ExportRecord, Transcript, TranscriptionJob
    from app.models.voice_enrollment import CommandEnrollment


class User(Base):
    __tablename__="users"
    __table_args__=(
        UniqueConstraint("email", name="uq_users_email"),
        CheckConstraint(
            "role IN ('STUDENT', 'TEACHER')",
            name="ck_users_role",
        ),
        CheckConstraint(
            "command_language IN ('si', 'en')",
            name="ck_users_command_language",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
    )
    # Which voice-command phrase set (app.streaming.commands) is active
    # for this student - a data setting, not a code one, so switching it
    # never needs a deploy. See app.services.voice_enrollment for how
    # this is kept independent of any particular enrollment bank.
    command_language: Mapped[str] = mapped_column(
        String(2),
        default="si",
        server_default=text("'si'"),
        nullable=False,
    )

    media_files: Mapped[list[MediaFile]] = relationship(
        back_populates="owner",
    )
    transcripts: Mapped[list[Transcript]] = relationship(
        back_populates="owner",
    )
    transcription_jobs: Mapped[list[TranscriptionJob]] = relationship(
        back_populates="requester",
    )
    quizzes: Mapped[list[Quiz]] = relationship(
        back_populates="creator",
    )
    quiz_submissions: Mapped[list[QuizSubmission]] = relationship(
        foreign_keys="QuizSubmission.student_id",
        back_populates="student",
    )
    reviewed_submissions: Mapped[list[QuizSubmission]] = relationship(
        foreign_keys="QuizSubmission.reviewed_by",
        back_populates="reviewer",
    )
    export_records: Mapped[list[ExportRecord]] = relationship(
        back_populates="requester",
    )
    command_enrollments: Mapped[list[CommandEnrollment]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
