"""backfill URL records for existing emails

Revision ID: 4c9c5ed4e1d2
Revises: c22312ed429b
"""
from typing import Sequence, Union

from alembic import op


revision: str = "4c9c5ed4e1d2"
down_revision: Union[str, None] = "c22312ed429b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        INSERT INTO email_urls (id, email_id, url, selected_for_traversal)
        SELECT gen_random_uuid(), emails.id, parsed_url.url, false
        FROM emails
        CROSS JOIN LATERAL json_array_elements_text(emails.urls::json) AS parsed_url(url)
        ON CONFLICT (email_id, url) DO NOTHING
        """
    )


def downgrade() -> None:
    pass