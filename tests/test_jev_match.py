import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.api.resume import MatchRequest, current_resume, match_resume  # noqa: E402
from app.services.jev_match import (  # noqa: E402
    build_jev_payload,
    build_match_context,
    score_jev_response,
)


RESUME = {
    "basic_info": {
        "name": "张三", "phone": "13800138000", "email": "zhang@example.com",
        "photo": "data:image/png;base64,PRIVATEPHOTO",
    },
    "education": [{"school": "清华大学", "degree": "本科", "major": "计算机科学"}],
    "work_experience": [{
        "company": "秘密科技有限公司", "title": "开发工程师",
        "bullets": ["在秘密科技有限公司为张三做 Python 数据清洗，邮箱 zhang@example.com"],
    }],
    "projects": [{"name": "保密项目", "description": "用 SQL 做数据分析"}],
    "skills": ["Python", "SQL"],
}
JD = {
    "company": "目标公司", "required_skills": ["熟悉 Python", "掌握 SQL 数据分析"],
    "key_responsibilities": ["为目标公司做数据清洗"],
}
JD_TEXT = "岗位要求：熟悉 Python；掌握 SQL 数据分析；工作职责：为目标公司做数据清洗。"


def fake_answers(context: dict, source_id: str | None = None) -> dict:
    source_id = source_id or context["evidence"][0]["id"]
    answers = {}
    for row in context["outbound_state"]["requirements"]:
        rid = row["id"]
        answers[f"{rid}_strength"] = {"type": "score", "score": 1.9}
        answers[f"{rid}_source"] = {"type": "choice", "choice": source_id}
    return {"answers": answers}


class JevMatchUnitTest(unittest.TestCase):
    def test_outbound_payload_contains_only_fixed_tags_and_ids(self):
        context = build_match_context(RESUME, JD, JD_TEXT)
        payload = build_jev_payload(context)
        outbound = json.dumps(payload, ensure_ascii=False)
        for secret in (
            "张三", "13800138000", "zhang@example.com", "清华大学",
            "秘密科技有限公司", "目标公司", "PRIVATEPHOTO", "保密项目",
        ):
            self.assertNotIn(secret, outbound)
        self.assertIn("Python", outbound)
        self.assertIn("数据清洗", outbound)
        self.assertEqual(len(payload["questions"]), 6)

    def test_score_requires_existing_relevant_source(self):
        context = build_match_context(RESUME, JD, JD_TEXT)
        with self.assertRaises(ValueError):
            score_jev_response(context, fake_answers(context, "invented_evidence"))

        result = score_jev_response(context, fake_answers(context))
        self.assertEqual(result["status"], "ready")
        self.assertLess(result["score"], 100)
        self.assertEqual(len(result["requirements"]), 3)

    def test_sparse_recognition_has_no_total_score(self):
        jd = {"required_skills": ["Python", "甲", "乙", "丙", "丁"]}
        context = build_match_context(RESUME, jd, "Python；甲；乙；丙；丁")
        result = score_jev_response(context, fake_answers(context))
        self.assertEqual(result["status"], "insufficient")
        self.assertIsNone(result["score"])
        self.assertEqual(sum(r["level"] == "unscored" for r in result["requirements"]), 4)

    def test_decoder_only_requirements_are_not_scored(self):
        context = build_match_context(
            RESUME,
            {"required_skills": ["熟悉 Python", "熟悉 Kubernetes"]},
            "岗位要求：熟悉 SQL。",
        )
        self.assertEqual(context["requirements"][0]["tags"], [])
        self.assertEqual(context["requirements"][1]["tags"], [])
        result = score_jev_response(context, fake_answers(context))
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "insufficient")


class JevMatchFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_current_exposes_match_control_only_with_jd_and_key(self):
        state = SimpleNamespace(values={"jd_info": JD, "jd_text": JD_TEXT, "resume_data": RESUME})
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=state))
        settings = SimpleNamespace(LLM_API_KEY="configured", JEV_API_KEY="test-key")
        with (
            patch("app.api.resume.session_can_read_checkpoint", return_value=True),
            patch("app.api.resume.load_resume_session", return_value={"resume_data": RESUME}),
            patch("app.api.resume.get_settings", return_value=settings),
            patch("app.api.resume._llm_for_graph", return_value=object()),
            patch("app.api.resume.get_graph", return_value=graph),
        ):
            result = await current_resume("session-1")
        self.assertTrue(result["has_data"])
        self.assertTrue(result["has_jd"])
        self.assertTrue(result["match_enabled"])

    async def test_missing_jd_returns_checks_without_score(self):
        settings = SimpleNamespace(LLM_API_KEY="", JEV_API_KEY="", JEV_TIMEOUT_SECONDS=8)
        with (
            patch("app.api.resume.session_can_read_checkpoint", return_value=True),
            patch("app.api.resume.load_resume_session", return_value={"resume_data": RESUME}),
            patch("app.api.resume.get_settings", return_value=settings),
        ):
            result = await match_resume(MatchRequest(session_id="session-1"))
        self.assertEqual(result["status"], "no_jd")
        self.assertIsNone(result["score"])
        self.assertTrue(result["checks"])

    async def test_missing_key_or_remote_failure_has_no_score(self):
        state = SimpleNamespace(values={"jd_info": JD, "jd_text": JD_TEXT, "resume_data": RESUME})
        graph = SimpleNamespace(aget_state=AsyncMock(return_value=state))
        settings = SimpleNamespace(LLM_API_KEY="configured", JEV_API_KEY="", JEV_TIMEOUT_SECONDS=8)
        with (
            patch("app.api.resume.session_can_read_checkpoint", return_value=True),
            patch("app.api.resume.load_resume_session", return_value={"resume_data": RESUME}),
            patch("app.api.resume.get_settings", return_value=settings),
            patch("app.api.resume._llm_for_graph", return_value=object()),
            patch("app.api.resume.get_graph", return_value=graph),
        ):
            missing_key = await match_resume(MatchRequest(session_id="session-1"))
            self.assertEqual(missing_key["status"], "unavailable")
            self.assertIsNone(missing_key["score"])
            self.assertTrue(missing_key["checks"])

            settings.JEV_API_KEY = "test-key"
            with patch("app.api.resume.call_jev", new=AsyncMock(side_effect=RuntimeError("private upstream body"))):
                failed = await match_resume(MatchRequest(session_id="session-1"))
        self.assertEqual(failed["status"], "unavailable")
        self.assertIsNone(failed["score"])
        self.assertNotIn("private upstream body", json.dumps(failed, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
