"""Tenant-scoped grounding review persistence for PostgreSQL."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from .database import DatabaseRuntime, ReviewCase, ReviewerDecision
from .security_models import SecurityContext


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PostgresReviewRegistry:
    def __init__(self, database: DatabaseRuntime, context: SecurityContext):
        self.database, self.context = database, context

    def enqueue(self, payload: dict[str, Any], reason: str, config_fingerprint: str) -> str:
        identity = {"claim": payload.get("text", ""), "evidence": [item.get("chunk_id", "") for item in payload.get("evidence", [])], "config": config_fingerprint}
        fingerprint = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        case_id = "rev_" + fingerprint[:20]
        stored = {"reason": reason, "claim_text": payload.get("text", ""), "verdict": payload.get("verdict", "unsupported"),
                  "payload": payload, "config_fingerprint": config_fingerprint, "updated_at": datetime.now(timezone.utc).isoformat()}
        with self.database.session(self.context, write=True) as session:
            row = session.get(ReviewCase, (self.context.organization_id, case_id))
            if row:
                row.payload_json = stored
                row.status = "open"
            else:
                session.add(ReviewCase(organization_id=self.context.organization_id, case_id=case_id,
                                       status="open", fingerprint=fingerprint, payload_json=stored))
        return case_id

    def list_cases(self, status: str = "open", limit: int = 100):
        with self.database.session(self.context) as session:
            query = select(ReviewCase)
            if status != "all":
                query = query.where(ReviewCase.status == status)
            rows = session.execute(query.order_by(ReviewCase.created_at.desc()).limit(max(1, min(limit, 500)))).scalars().all()
            return [self._decode(row) for row in rows]

    def get_case(self, case_id: str):
        with self.database.session(self.context) as session:
            row = session.get(ReviewCase, (self.context.organization_id, case_id))
            if not row:
                return None
            result = self._decode(row)
            decisions = session.execute(select(ReviewerDecision).where(ReviewerDecision.case_id == case_id).order_by(ReviewerDecision.created_at)).scalars().all()
            result["decisions"] = [{"decision": value.decision, "reviewer": value.reviewer_id, "notes": value.notes,
                                    "created_at": value.created_at.isoformat()} for value in decisions]
            return result

    def decide(self, case_id: str, decision: str, reviewer: str = "local", notes: str = "") -> bool:
        if decision not in {"supported", "unsupported", "contradicted"}:
            raise ValueError("Unknown review decision.")
        with self.database.session(self.context, write=True) as session:
            row = session.get(ReviewCase, (self.context.organization_id, case_id))
            if not row:
                return False
            session.add(ReviewerDecision(organization_id=self.context.organization_id,
                                         decision_id="dec_" + uuid.uuid4().hex, case_id=case_id,
                                         reviewer_id=reviewer.strip() or self.context.user_id,
                                         decision=decision, notes=notes.strip()[:1000]))
            row.status = "resolved"
        return True

    def export_jsonl(self, status: str = "all") -> str:
        return "\n".join(_canonical(self.get_case(row["case_id"])) for row in self.list_cases(status, 500))

    @staticmethod
    def _decode(row: ReviewCase) -> dict[str, Any]:
        payload = dict(row.payload_json)
        return {"case_id": row.case_id, "status": row.status, "reason": payload.get("reason", ""),
                "claim_text": payload.get("claim_text", ""), "verdict": payload.get("verdict", ""),
                "payload": payload.get("payload", {}), "config_fingerprint": payload.get("config_fingerprint", ""),
                "created_at": row.created_at.isoformat(), "updated_at": payload.get("updated_at", row.created_at.isoformat())}
