from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class CommandEnrollment(Base):
    """One enrolled voice-command sample: an embedding vector plus enough
    metadata to know whose it is, which command it's for, and whether
    it's still valid for the currently-loaded model.

    `command_id` is deliberately a free-form string - not a foreign key
    or CHECK-constrained enum against app.streaming.commands.COMMANDS.
    That vocabulary can be edited at any time without touching this
    table or any code here; only the enrollment rows already stored
    under an old command id are affected (they simply become unused if
    that id is removed, and nothing reads them until the id is used
    again).
    """

    __tablename__ = "command_enrollments"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "command_id",
            "sample_index",
            name="uq_command_enrollments_slot",
        ),
        CheckConstraint(
            "sample_index >= 0",
            name="ck_command_enrollments_sample_index",
        ),
        CheckConstraint(
            "embedding_dim > 0",
            name="ck_command_enrollments_embedding_dim",
        ),
    )

    enrollment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        # Enrollment samples are convenience/biometric-adjacent data
        # derived from the user's account, not an academic record worth
        # protecting from cascading loss - CASCADE deletes them along
        # with the account, rather than blocking account deletion.
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    command_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    # Identifies which checkpoint produced this embedding (see
    # settings.voice_embedding_model_version) so a later checkpoint swap
    # can detect and invalidate stale enrollments instead of silently
    # comparing embeddings from two different model spaces.
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="command_enrollments")
