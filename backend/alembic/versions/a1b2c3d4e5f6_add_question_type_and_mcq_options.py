"""add question type and mcq options

Revision ID: a1b2c3d4e5f6
Revises: 9349dd15bd77
Create Date: 2026-09-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '9349dd15bd77'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column(
        "questions",
        sa.Column(
            "question_type",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'SPOKEN'"),
        ),
    )
    op.create_check_constraint(
        "ck_questions_type",
        "questions",
        "question_type IN ('MCQ', 'SPOKEN')",
    )

    op.create_table(
        "question_options",
        sa.Column("option_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "question_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("questions.question_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("option_order", sa.Integer(), nullable=False),
        sa.Column("option_text", sa.Text(), nullable=False),
        sa.Column(
            "is_correct",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.UniqueConstraint(
            "question_id",
            "option_order",
            name="uq_question_options_order",
        ),
        sa.CheckConstraint(
            "option_order >= 1 AND option_order <= 4",
            name="ck_question_options_order",
        ),
    )
    op.create_index(
        op.f("ix_question_options_question_id"),
        "question_options",
        ["question_id"],
    )

    op.add_column(
        "answer_submissions",
        sa.Column(
            "selected_option_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "question_options.option_id",
                ondelete="SET NULL",
            ),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column("answer_submissions", "selected_option_id")
    op.drop_index(
        op.f("ix_question_options_question_id"),
        table_name="question_options",
    )
    op.drop_table("question_options")
    op.drop_constraint("ck_questions_type", "questions", type_="check")
    op.drop_column("questions", "question_type")
