import asyncio
import io
import os
import sys
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.pop("LLM_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models.models import ResumeSession  # noqa: E402
from app.services.file_extractor import FileLimitError, extract_text  # noqa: E402
from app.services.session_store import (  # noqa: E402
    delete_resume_session,
    ensure_resume_session,
    load_resume_session,
    save_resume_session,
    session_can_read_checkpoint,
    session_is_blocked,
)


class PrivacyRegressionTest(unittest.TestCase):
    def setUp(self):
        self.session_id = str(uuid.uuid4())
        main._chat_requests.clear()
        main._all_paid_requests.clear()

    def tearDown(self):
        with SessionLocal() as db:
            row = db.get(ResumeSession, self.session_id)
            if row:
                db.delete(row)
                db.commit()

    def _age_session(self, days=8):
        with SessionLocal() as db:
            row = db.get(ResumeSession, self.session_id)
            row.updated_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
            db.commit()

    def test_expired_session_is_unreadable_and_cleanup_retries_failed_checkpoint(self):
        with TestClient(main.app) as client:
            save_resume_session(self.session_id, {"basic_info": {"name": "private"}}, stage="BUILDING")
            self._age_session()
            self.assertIsNone(load_resume_session(self.session_id))
            self.assertFalse(session_can_read_checkpoint(self.session_id))
            self.assertTrue(session_is_blocked(self.session_id))
            current = client.get("/api/resume/current", params={"session_id": self.session_id})
            self.assertEqual(current.status_code, 200)
            self.assertFalse(current.json()["has_data"])

            class FailingSaver:
                async def adelete_thread(self, _):
                    raise RuntimeError("checkpoint unavailable")

            asyncio.run(main._cleanup_expired_sessions(FailingSaver()))
            with SessionLocal() as db:
                self.assertIsNotNone(db.get(ResumeSession, self.session_id))

            removed = []

            class WorkingSaver:
                async def adelete_thread(self, session_id):
                    removed.append(session_id)

            asyncio.run(main._cleanup_expired_sessions(WorkingSaver()))
            self.assertIn(self.session_id, removed)
            with SessionLocal() as db:
                self.assertIsNone(db.get(ResumeSession, self.session_id))

    def test_delete_failure_does_not_report_success_or_allow_resurrection(self):
        with TestClient(main.app) as client:
            save_resume_session(self.session_id, {"basic_info": {"name": "private"}}, stage="BUILDING")

            class FailingSaver:
                async def adelete_thread(self, _):
                    raise RuntimeError("private checkpoint error")

            with patch.object(main.app.state, "checkpointer", FailingSaver()):
                deleted = client.delete(f"/api/chat/session/{self.session_id}")
            self.assertEqual(deleted.status_code, 503)
            self.assertNotIn("private checkpoint error", deleted.text)
            self.assertIsNone(load_resume_session(self.session_id))
            self.assertTrue(session_is_blocked(self.session_id))
            self.assertFalse(ensure_resume_session(self.session_id))
            save_resume_session(self.session_id, {"basic_info": {"name": "resurrected"}}, stage="BUILDING")
            self.assertIsNone(load_resume_session(self.session_id))

            retried = []

            class WorkingSaver:
                async def adelete_thread(self, session_id):
                    retried.append(session_id)

            asyncio.run(main._cleanup_expired_sessions(WorkingSaver()))
            self.assertIn(self.session_id, retried)

            deleted = client.delete(f"/api/chat/session/{self.session_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(deleted.json(), {"deleted": True})

    def test_untrusted_forwarded_header_cannot_reset_paid_request_limit(self):
        with patch.object(main, "CHAT_REQUEST_LIMIT", 2), TestClient(main.app) as client:
            for i in range(2):
                response = client.post(
                    "/api/chat/stream",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Forwarded-For": f"203.0.113.{i + 1}"},
                )
                self.assertNotEqual(response.status_code, 429)
            blocked = client.post(
                "/api/chat/stream",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Forwarded-For": "203.0.113.99"},
            )
            self.assertEqual(blocked.status_code, 429)

    def test_chat_body_and_upload_limits_and_error_redaction(self):
        with TestClient(main.app) as client:
            huge_message = "a" * (main.chat.MAX_CHAT_BODY_BYTES + 1)
            chat = client.post(
                "/api/chat/stream",
                json={"messages": [{"role": "user", "content": huge_message}]},
            )
            self.assertEqual(chat.status_code, 413)
            upload = client.post(
                "/api/chat/upload",
                files={"file": ("resume.txt", b"x" * (10 * 1024 * 1024 + 1))},
            )
            self.assertEqual(upload.status_code, 413)
            early = client.post(
                "/api/chat/upload",
                files={"file": ("resume.txt", b"small")},
                headers={"Content-Length": str(12 * 1024 * 1024)},
            )
            self.assertEqual(early.status_code, 413)
            with patch("app.api.chat.extract_text", side_effect=RuntimeError("secret parse detail")):
                bad = client.post(
                    "/api/chat/upload",
                    files={"file": ("resume.pdf", b"%PDF-1.4")},
                )
            self.assertEqual(bad.status_code, 422)
            self.assertNotIn("secret parse detail", bad.text)

    def test_pdf_pages_and_docx_expansion_are_bounded(self):
        class FakePdf:
            pages = [object()] * 21

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

        with patch("pdfplumber.open", return_value=FakePdf()):
            with self.assertRaises(FileLimitError):
                extract_text("resume.pdf", b"%PDF")

        output = io.BytesIO()
        with ZipFile(output, "w") as archive:
            archive.writestr("word/document.xml", "x" * 101)
        with patch("app.services.file_extractor.MAX_DOCX_UNCOMPRESSED_BYTES", 100):
            with self.assertRaises(FileLimitError):
                extract_text("resume.docx", output.getvalue())


if __name__ == "__main__":
    unittest.main()
