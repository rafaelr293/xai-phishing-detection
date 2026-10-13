"""persist visual analysis per traversal

Revision ID: 7e2b2c3f4a5b
Revises: 4c9c5ed4e1d2
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7e2b2c3f4a5b"
down_revision: Union[str, None] = "4c9c5ed4e1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "url_traversals",
        sa.Column("visual_analysis", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False),
    )
    op.alter_column("url_traversals", "visual_analysis", server_default=None)


def downgrade() -> None:
    op.drop_column("url_traversals", "visual_analysis")