"""add persisted risk settings

Revision ID: 8f3a4b5c6d7e
Revises: 7e2b2c3f4a5b
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8f3a4b5c6d7e"
down_revision: Union[str, None] = "7e2b2c3f4a5b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "risk_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("weights", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("risk_settings")