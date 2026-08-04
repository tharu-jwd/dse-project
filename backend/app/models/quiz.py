from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
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
    from app.models.transcription import Transcript
    from app.models.user import User


class Quiz(Base):
    __tablename__ = "quizzes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('DRAFT', 'PUBLISHED', 'CLOSED')",
            name="ck_quizzes_status",
        ),
        CheckConstraint(
            "available_until IS NULL "
            "OR available_from IS NULL "
            "OR available_until > available_from",
            name="ck_quizzes_availability",
        ),
    )

    quiz_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        default="DRAFT",
        server_default=text("'DRAFT'"),
        nullable=False,
        index=True,
    )
    available_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    available_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
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

    creator: Mapped[User] = relationship(back_populates="quizzes")
    questions: Mapped[list[Question]] = relationship(
        back_populates="quiz",
        cascade="all, delete-orphan",
        order_by="Question.question_order",
    )
    submissions: Mapped[list[QuizSubmission]] = relationship(
        back_populates="quiz",
        cascade="all, delete-orphan",
    )


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        UniqueConstraint(
            "quiz_id",
            "question_order",
            name="uq_questions_order",
        ),
        CheckConstraint(
            "question_order >= 1",
            name="ck_questions_order",
        ),
    )

    question_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    quiz_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("quizzes.quiz_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_order: Mapped[int] = mapped_column(Integer, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_required: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
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

    quiz: Mapped[Quiz] = relationship(back_populates="questions")
    answer_submissions: Mapped[list[AnswerSubmission]] = relationship(
        back_populates="question",
    )


class QuizSubmission(Base):
    __tablename__ = "quiz_submissions"
    __table_args__ = (
        UniqueConstraint(
            "quiz_id",
            "student_id",
            name="uq_quiz_submissions_student",
        ),
        CheckConstraint(
            "status IN ('IN_PROGRESS', 'SUBMITTED', 'REVIEWED')",
            name="ck_quiz_submissions_status",
        ),
        CheckConstraint(
            "mark IS NULL OR (mark >= 0 AND mark <= 100)",
            name="ck_quiz_submissions_mark",
        ),
    )

    submission_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    quiz_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("quizzes.quiz_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="IN_PROGRESS",
        server_default=text("'IN_PROGRESS'"),
        nullable=False,
        index=True,
    )
    mark: Mapped[float | None] = mapped_column(Float, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    quiz: Mapped[Quiz] = relationship(back_populates="submissions")
    student: Mapped[User] = relationship(
        foreign_keys=[student_id],
        back_populates="quiz_submissions",
    )
    reviewer: Mapped[User | None] = relationship(
        foreign_keys=[reviewed_by],
        back_populates="reviewed_submissions",
    )
    answers: Mapped[list[AnswerSubmission]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
    )


class AnswerSubmission(Base):
    __tablename__ = "answer_submissions"
    __table_args__ = (
        UniqueConstraint(
            "submission_id",
            "question_id",
            name="uq_answer_submissions_question",
        ),
        UniqueConstraint(
            "transcription_id",
            name="uq_answer_submissions_transcription",
        ),
        CheckConstraint(
            "status IN "
            "('NOT_STARTED', 'PROCESSING', 'COMPLETED', 'FAILED')",
            name="ck_answer_submissions_status",
        ),
    )

    answer_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("quiz_submissions.submission_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("questions.question_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transcription_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("transcripts.transcript_id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="NOT_STARTED",
        server_default=text("'NOT_STARTED'"),
        nullable=False,
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

    submission: Mapped[QuizSubmission] = relationship(
        back_populates="answers",
    )
    question: Mapped[Question] = relationship(
        back_populates="answer_submissions",
    )
    transcription: Mapped[Transcript | None] = relationship(
        back_populates="answer_submissions",
    )
