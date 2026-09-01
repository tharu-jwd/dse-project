"""add unique owner title index to transcripts

Revision ID: 9349dd15bd77
Revises: 00b5eb9e7a73
Create Date: 2026-09-01 21:10:05.008460

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9349dd15bd77'
down_revision: Union[str, Sequence[str], None] = '00b5eb9e7a73'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Existing rows may already collide (created before this constraint
    # existed) - rename every duplicate but the oldest so the unique index
    # below doesn't fail to build. Application code prevents new
    # collisions going forward (see app.services.transcript_service); this
    # only cleans up what's already in the table.
    op.execute(
        """
        WITH duplicates AS (
            SELECT
                transcript_id,
                row_number() OVER (
                    PARTITION BY owner_id, lower(title)
                    ORDER BY created_at, transcript_id
                ) AS rn
            FROM transcripts
        )
        UPDATE transcripts
        SET title = transcripts.title || ' (' || duplicates.rn || ')'
        FROM duplicates
        WHERE transcripts.transcript_id = duplicates.transcript_id
          AND duplicates.rn > 1
        """
    )

    op.create_index(
        "uq_transcripts_owner_title",
        "transcripts",
        ["owner_id", sa.text("lower(title)")],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index("uq_transcripts_owner_title", table_name="transcripts")
