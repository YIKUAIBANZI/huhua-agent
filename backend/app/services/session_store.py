"""Persistence and lifecycle rules for private resume sessions."""

import asyncio
import logging
import weakref
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_

from app.config import get_settings
from app.database import SessionLocal
from app.models.models import ResumeSession

logger = logging.getLogger(__name__)

# Keep a private-data-free marker so an in-flight stream cannot recreate a
# session after its checkpoint and payload have been deleted.
DELETED_STAGE = "__DELETED__"
_session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


def session_lock(session_id: str) -> asyncio.Lock:
    """Serialize stream, explicit deletion, and expiry cleanup in this worker."""
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


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


def _cutoff(retention_days: int | None = None) -> datetime:
    days = (
        retention_days
        if retention_days is not None
        else get_settings().SESSION_RETENTION_DAYS
    )
    return datetime.now(UTC).replace(tzinfo=None) - timedelta(days=max(1, days))


def _is_expired(row: ResumeSession, cutoff: datetime | None = None) -> bool:
    updated_at = row.updated_at
    if updated_at is None:
        return True
    if updated_at.tzinfo is not None:
        updated_at = updated_at.astimezone(UTC).replace(tzinfo=None)
    return updated_at < (cutoff or _cutoff())


def session_is_blocked(session_id: str) -> bool:
    """An old or deleted ID must not start another graph run."""
    if not session_id:
        return True
    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        return bool(row and (row.stage == DELETED_STAGE or _is_expired(row)))
    except Exception:
        logger.exception("session status read failed")
        return True
    finally:
        db.close()


def session_can_read_checkpoint(session_id: str) -> bool:
    """Permit graph reads only for a current, undeleted persisted session."""
    if not session_id:
        return False
    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        return bool(row and row.stage != DELETED_STAGE and not _is_expired(row))
    except Exception:
        logger.exception("session checkpoint permission read failed")
        return False
    finally:
        db.close()


def session_needs_checkpoint_cleanup(session_id: str, retention_days: int) -> bool:
    """Recheck expiry or deletion under the lock before removing a checkpoint."""
    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        return bool(
            row and (row.stage == DELETED_STAGE or _is_expired(row, _cutoff(retention_days)))
        )
    finally:
        db.close()


def ensure_resume_session(session_id: str) -> bool:
    """Track a new graph thread before its first checkpoint is written."""
    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is not None:
            return row.stage != DELETED_STAGE and not _is_expired(row)
        db.add(ResumeSession(session_id=session_id, resume_data={}, stage="JD_INPUT"))
        db.commit()
        return True
    except Exception:
        db.rollback()
        logger.exception("session initialization failed")
        return False
    finally:
        db.close()


def save_resume_session(
    session_id: str,
    resume_data: dict,
    resume_draft: str = "",
    stage: str = "",
) -> None:
    if not session_id or (not resume_data_has_content(resume_data) and not stage):
        return

    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is not None and (row.stage == DELETED_STAGE or _is_expired(row)):
            return
        if row is None:
            row = ResumeSession(session_id=session_id)
            db.add(row)
        row.resume_data = resume_data
        row.resume_draft = resume_draft or row.resume_draft or ""
        row.stage = stage or row.stage or ""
        row.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("session save failed")
    finally:
        db.close()


def load_resume_session(session_id: str) -> dict | None:
    if not session_id:
        return None

    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is None or row.stage == DELETED_STAGE or _is_expired(row):
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
    except Exception:
        logger.exception("session load failed")
        return None
    finally:
        db.close()


def delete_resume_session(session_id: str) -> None:
    """Clear personal content and persist a temporary non-reusable marker."""
    if not session_id:
        return
    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is None:
            row = ResumeSession(session_id=session_id)
            db.add(row)
        row.resume_data = {}
        row.resume_draft = ""
        row.stage = DELETED_STAGE
        row.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("session delete failed")
        raise
    finally:
        db.close()


def list_sessions_needing_checkpoint_cleanup(retention_days: int) -> list[str]:
    """List expired or deleted IDs; never discard a failed-cleanup marker."""
    db = SessionLocal()
    try:
        rows = (
            db.query(ResumeSession.session_id)
            .filter(
                or_(
                    ResumeSession.updated_at < _cutoff(retention_days),
                    ResumeSession.stage == DELETED_STAGE,
                )
            )
            .all()
        )
        return [session_id for (session_id,) in rows]
    finally:
        db.close()


def purge_expired_resume_session(session_id: str, retention_days: int) -> None:
    """Purge only if the session is still expired after its checkpoint is gone."""
    db = SessionLocal()
    try:
        row = db.get(ResumeSession, session_id)
        if row is not None and _is_expired(row, _cutoff(retention_days)):
            db.delete(row)
            db.commit()
    finally:
        db.close()
