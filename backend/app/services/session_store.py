"""Persistent storage for generated resume sessions.

LangGraph's in-memory checkpointer is still used for live conversation turns.
This store keeps the final ResumeData available for preview/export after a
server restart or when the graph checkpoint is not present in this process.
"""

import logging
from datetime import UTC, datetime, timedelta

from app.database import SessionLocal
from app.models.models import ResumeSession

logger = logging.getLogger(__name__)


def resume_data_has_content(resume_data: dict | None) -> bool:
    if not resume_data:
        return False
    basic = resume_data.get("basic_info") or {}
    return bool(
        resume_data.get("work_experience")
        or resume_data.get("projects")
        or resume_data.get("education")
        or basic.get("name")
        or basic.get("height")
        or basic.get("age")
        or resume_data.get("self_evaluation")
    )


def save_resume_session(
    session_id: str,
    resume_data: dict,
    resume_draft: str = "",
    stage: str = "",
) -> None:
    if not session_id:
        return
    if not resume_data_has_content(resume_data) and not stage:
        return

    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is None:
            row = ResumeSession(session_id=session_id)
            db.add(row)
        row.resume_data = resume_data
        row.resume_draft = resume_draft or row.resume_draft or ""
        row.stage = stage or row.stage or ""
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(
            "[session_store/save] failed session_id=%s err=%s", session_id, e
        )
    finally:
        db.close()


def load_resume_session(session_id: str) -> dict | None:
    if not session_id:
        return None

    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is None:
            return None
        payload = row.resume_data or {}
        return {
            "session_id": row.session_id,
            "resume_data": payload,
            "resume_draft": row.resume_draft or "",
            "stage": row.stage or "",
            "has_data": resume_data_has_content(payload),
            "updated_at": row.updated_at.isoformat() if row.updated_at else "",
        }
    except Exception as e:
        logger.warning(
            "[session_store/load] failed session_id=%s err=%s", session_id, e
        )
        return None
    finally:
        db.close()


def delete_resume_session(session_id: str) -> None:
    if not session_id:
        return
    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is not None:
            db.delete(row)
            db.commit()
    finally:
        db.close()


def delete_expired_resume_sessions(retention_days: int) -> list[str]:
    """Delete stored resume payloads and return IDs for checkpoint cleanup."""
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(
        days=max(1, retention_days)
    )
    db = SessionLocal()
    try:
        rows = db.query(ResumeSession).filter(ResumeSession.updated_at < cutoff).all()
        session_ids = [row.session_id for row in rows]
        for row in rows:
            db.delete(row)
        if rows:
            db.commit()
        return session_ids
    finally:
        db.close()
