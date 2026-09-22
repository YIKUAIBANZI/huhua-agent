import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.pop("LLM_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from app.agents.editor_chain import _numbers  # noqa: E402
from app.services.resume_review import review_resume  # noqa: E402


class PublicBetaSmokeTest(unittest.TestCase):
    def test_pages_render_and_docx_exports(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/").status_code, 200)
            sample = client.get("/api/resume/sample").json()
            rendered = client.post("/api/resume/render/t001-jianyue", json=sample)
            self.assertEqual(rendered.status_code, 200)
            exported = client.post("/api/resume/export/docx/t001-jianyue", json=sample)
            self.assertEqual(exported.status_code, 200)
            self.assertGreater(len(exported.content), 10_000)

    def test_missing_model_key_fails_cleanly_and_session_can_be_deleted(self):
        with TestClient(app) as client:
            current = client.get("/api/resume/current?session_id=missing")
            self.assertEqual(current.status_code, 200)
            self.assertFalse(current.json()["has_data"])
            chat = client.post(
                "/api/chat/stream",
                json={"messages": [{"role": "user", "content": "你好"}]},
            )
            self.assertEqual(chat.status_code, 503)
            deleted = client.delete("/api/chat/session/test-delete")
            self.assertEqual(deleted.json(), {"deleted": True})

    def test_review_is_evidence_based(self):
        data = {
            "basic_info": {"name": "张三", "email": "a@example.com"},
            "projects": [{"name": "项目 A", "description": "完成 50 条测试"}],
            "skills": ["Python"],
        }
        checks = review_resume(data, {"required_skills": ["Python", "FastAPI"]})
        self.assertTrue(any("FastAPI" in item for item in checks))
        self.assertEqual(_numbers("完成 50 条测试"), {"50"})
        self.assertEqual(_numbers("提升 80%"), {"80%"})


if __name__ == "__main__":
    unittest.main()
