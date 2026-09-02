"""Create the tenant-isolated production platform.

Revision ID: 20260902_0001
Revises:
"""
from alembic import op
from sqlalchemy import text

from backend.database import Base, rls_statements

revision = "20260902_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=bind)
    for statement in rls_statements():
        bind.execute(text(statement))
    bind.execute(text("""
        CREATE OR REPLACE FUNCTION reject_audit_mutation() RETURNS trigger AS $$
        BEGIN RAISE EXCEPTION 'audit events are append-only'; END;
        $$ LANGUAGE plpgsql
    """))
    bind.execute(text("DROP TRIGGER IF EXISTS audit_events_no_update ON audit_events"))
    bind.execute(text("DROP TRIGGER IF EXISTS audit_events_no_delete ON audit_events"))
    bind.execute(text("CREATE TRIGGER audit_events_no_update BEFORE UPDATE ON audit_events FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation()"))
    bind.execute(text("CREATE TRIGGER audit_events_no_delete BEFORE DELETE ON audit_events FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation()"))


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
    bind.execute(text("DROP FUNCTION IF EXISTS reject_audit_mutation()"))
