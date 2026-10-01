import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.editor_chain import EditorOutput, run_editor  # noqa: E402
from app.agents.resume_graph import (  # noqa: E402
    _empty_collected_info,
    _extract_info_from_message,
    _harvest_projects_from_messages,
    _init_state,
    _serialize_to_resume_data,
    collecting_node,
    evaluating_node,
    stream_agent,
)
from app.agents.triage_chain import TriageAction, TriageDecision  # noqa: E402


class _StructuredLLM:
    def __init__(self, output):
        self.output = output

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        return self.output


class ResumeFlowTest(unittest.IsolatedAsyncioTestCase):
    async def _wrap_with_extraction(self, message, snippet, extracted, current=None):
        state = _init_state()
        state["stage"] = "COLLECTING"
        state["messages"] = [HumanMessage(content=message)]
        if current is not None:
            state["collected_info"] = copy.deepcopy(current)
        extraction_response = (
            extracted
            if isinstance(extracted, Exception)
            else AIMessage(content=json.dumps(extracted, ensure_ascii=False))
        )
        llm = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=[extraction_response, AIMessage(content="经历整理建议")]
            )
        )
        decision = TriageDecision(
            action=TriageAction.WRAP_EXPERIENCE,
            reason="user described an experience",
            experience_snippet=snippet,
        )
        with (
            patch("app.agents.resume_graph._make_non_streaming_llm", return_value=llm),
            patch(
                "app.agents.resume_graph.run_triage", AsyncMock(return_value=decision)
            ),
        ):
            return await collecting_node(state, llm)

    def _evaluating_state(self, text: str):
        state = _init_state()
        state["stage"] = "EVALUATING"
        state["collected_info"]["projects"] = [
            {"name": "校招调研", "description": "参与用户调研"}
        ]
        state["resume_data"] = _serialize_to_resume_data(state)
        state["messages"] = [HumanMessage(content=text)]
        return state

    async def test_revision_rebuilds_resume_and_review_in_one_turn(self):
        state = self._evaluating_state(
            "不满意，请补充我整理过 12 份访谈笔记，还会 Python"
        )
        updated = copy.deepcopy(state["collected_info"])
        updated["projects"][0]["description"] = "参与用户调研，整理过 12 份访谈笔记"
        updated["skills"] = ["Python"]

        extractor = AsyncMock(return_value=updated)
        editor = AsyncMock(side_effect=lambda data, _llm: data)
        with (
            patch("app.agents.resume_graph._extract_info_from_message", extractor),
            patch(
                "app.agents.resume_graph._make_non_streaming_llm", return_value=object()
            ),
            patch("app.agents.editor_chain.run_editor", editor),
        ):
            result = await evaluating_node(state, object())

        self.assertEqual(result["stage"], "EVALUATING")
        self.assertEqual(result["collected_info"], updated)
        self.assertEqual(result["resume_data"]["skills"], ["Python"])
        self.assertIn("12 份访谈笔记", result["resume_draft"])
        self.assertIn("投递前核对", result["messages"][0].content)
        self.assertIn("12 份访谈笔记", result["messages"][0].content)
        extractor.assert_awaited_once()
        editor.assert_awaited_once()

    async def test_vague_optimization_asks_for_facts_without_rebuilding(self):
        state = self._evaluating_state("继续优化")
        with patch("app.agents.resume_graph._extract_info_from_message") as extractor:
            result = await evaluating_node(state, object())
        self.assertIn("想调整哪一处", result["messages"][0].content)
        self.assertNotIn("投递前核对", result["messages"][0].content)
        extractor.assert_not_called()

    async def test_standalone_export_does_not_extract(self):
        state = self._evaluating_state("可以导出 Word")
        with patch("app.agents.resume_graph._extract_info_from_message") as extractor:
            result = await evaluating_node(state, object())
        self.assertEqual(result["stage"], "EXPORT")
        extractor.assert_not_called()

    async def test_edit_before_export_is_not_treated_as_confirmation(self):
        state = self._evaluating_state("项目描述有问题，请先补充整理访谈笔记，再导出")
        updated = copy.deepcopy(state["collected_info"])
        updated["projects"][0]["description"] = "参与用户调研，整理访谈笔记"
        extractor = AsyncMock(return_value=updated)
        with (
            patch("app.agents.resume_graph._extract_info_from_message", extractor),
            patch(
                "app.agents.resume_graph._make_non_streaming_llm", return_value=object()
            ),
            patch(
                "app.agents.editor_chain.run_editor",
                AsyncMock(side_effect=lambda data, _llm: data),
            ),
        ):
            result = await evaluating_node(state, object())
        self.assertEqual(result["stage"], "EVALUATING")
        self.assertIn("整理访谈笔记", result["resume_draft"])
        extractor.assert_awaited_once()

    async def test_editor_rejects_semantic_upgrade_and_invented_skills(self):
        source = {
            "skills": ["Excel"],
            "projects": [{"name": "A", "description": "参与调研，整理访谈笔记"}],
        }
        output = EditorOutput(
            skill_groups={"技术栈": ["Python"]},
            project_polishes=[
                {"name": "A", "polished_bullets": ["主导调研并推动上线"]}
            ],
        )
        result = await run_editor(source, _StructuredLLM(output))
        self.assertNotIn("polished_bullets", result["projects"][0])
        self.assertNotIn("skill_groups", result)
        self.assertEqual(
            result["projects"][0]["description"], source["projects"][0]["description"]
        )

    async def test_editor_accepts_exact_source_sentence_and_rejects_new_numbers(self):
        source = {"projects": [{"name": "A", "description": "参与调研；整理访谈笔记"}]}
        exact = EditorOutput(
            project_polishes=[
                {"name": "A", "polished_bullets": ["参与调研", "整理访谈笔记"]}
            ]
        )
        accepted = await run_editor(source, _StructuredLLM(exact))
        self.assertEqual(
            accepted["projects"][0]["polished_bullets"], ["参与调研", "整理访谈笔记"]
        )

        invented_number = EditorOutput(
            project_polishes=[{"name": "A", "polished_bullets": ["完成 80 次调研"]}]
        )
        rejected = await run_editor(source, _StructuredLLM(invented_number))
        self.assertNotIn("polished_bullets", rejected["projects"][0])

    async def test_extractor_rejects_unsupported_ownership_upgrade(self):
        current = _empty_collected_info()
        invented = copy.deepcopy(current)
        invented["projects"] = [{"name": "A", "description": "主导调研推动上线"}]
        llm = SimpleNamespace(
            ainvoke=AsyncMock(return_value=AIMessage(content=json.dumps(invented)))
        )
        rejected = await _extract_info_from_message("我在 A 项目参与调研", current, llm)
        self.assertEqual(rejected, current)

        grounded = copy.deepcopy(current)
        grounded["projects"] = [{"name": "A", "description": "参与调研"}]
        llm.ainvoke.return_value = AIMessage(content=json.dumps(grounded))
        accepted = await _extract_info_from_message("我在 A 项目参与调研", current, llm)
        self.assertEqual(accepted, grounded)

    async def test_extractor_does_not_turn_a_negated_action_into_a_claim(self):
        current = _empty_collected_info()
        invented = copy.deepcopy(current)
        invented["projects"] = [{"name": "A", "description": "负责上线"}]
        llm = SimpleNamespace(
            ainvoke=AsyncMock(return_value=AIMessage(content=json.dumps(invented)))
        )
        rejected = await _extract_info_from_message(
            "我在 A 项目没有负责上线，只参与调研", current, llm
        )
        self.assertEqual(rejected, current)

        grounded = copy.deepcopy(current)
        grounded["projects"] = [{"name": "A", "description": "参与调研"}]
        llm.ainvoke.return_value = AIMessage(content=json.dumps(grounded))
        accepted = await _extract_info_from_message(
            "我在 A 项目没有负责上线，只参与调研", current, llm
        )
        self.assertEqual(accepted, grounded)

    async def test_fallback_uses_user_words_not_ai_bullet(self):
        messages = [
            HumanMessage(content="我在 A 项目参与调研"),
            AIMessage(content="🧷 **试着这么写**：主导调研并推动上线"),
        ]
        fallback = _harvest_projects_from_messages(messages)
        self.assertEqual(fallback[0]["description"], "我在 A 项目参与调研")
        self.assertNotIn("主导", fallback[0]["description"])
        self.assertEqual(_harvest_projects_from_messages(messages[1:]), [])

    async def test_wrap_saves_raw_user_experience_only(self):
        state = _init_state()
        state["stage"] = "COLLECTING"
        state["messages"] = [HumanMessage(content="我在 A 项目参与调研")]
        llm = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=AIMessage(content="🧷 **试着这么写**：主导调研并推动上线")
            )
        )
        decision = TriageDecision(
            action=TriageAction.WRAP_EXPERIENCE,
            reason="user described an experience",
            experience_snippet="我在 A 项目参与调研",
        )
        with (
            patch("app.agents.resume_graph._make_non_streaming_llm", return_value=llm),
            patch(
                "app.agents.resume_graph._extract_info_from_message",
                AsyncMock(return_value=state["collected_info"]),
            ),
            patch(
                "app.agents.resume_graph.run_triage", AsyncMock(return_value=decision)
            ),
        ):
            result = await collecting_node(state, llm)
        self.assertEqual(
            result["collected_info"]["projects"][0]["description"],
            "我在 A 项目参与调研",
        )

    async def test_wrap_does_not_duplicate_an_extracted_chinese_project(self):
        message = (
            "以下均为虚构测试资料，请不要补充未提供的事实。姓名：测试甲。"
            "教育：测试大学，计算机科学本科，2023.09-2027.06。"
            "项目：测试知识库，2026.03-2026.04，我独立用 Python 和 FastAPI "
            "实现文档检索接口，并编写了10条接口测试。"
            "技能：Python、FastAPI、Git。邮箱 test@example.com。"
        )
        snippet = (
            "测试知识库，2026.03-2026.04，我独立用 Python 和 FastAPI "
            "实现文档检索接口，并编写了10条接口测试。技能：Python、FastAPI、Git。"
        )
        extracted = _empty_collected_info()
        extracted["basic_info"].update(name="测试甲", email="test@example.com")
        extracted["education"] = [
            {
                "school": "测试大学",
                "major": "计算机科学",
                "degree": "本科",
                "start_date": "2023.09",
                "end_date": "2027.06",
            }
        ]
        extracted["projects"] = [
            {
                "name": "测试知识库",
                "start_date": "2026.03",
                "end_date": "2026.04",
                "description": (
                    "我独立用 Python 和 FastAPI 实现文档检索接口，"
                    "并编写了10条接口测试"
                ),
            }
        ]
        extracted["skills"] = ["Python", "FastAPI", "Git"]

        result = await self._wrap_with_extraction(message, snippet, extracted)

        self.assertEqual(result["collected_info"], extracted)
        serialized = _serialize_to_resume_data(result)
        self.assertEqual(len(serialized["projects"]), 1)
        self.assertNotIn("技能：", serialized["projects"][0]["description"])

    async def test_wrap_does_not_copy_extracted_work_or_internship_into_projects(self):
        snippet = "我在测试公司用 Python 实现文档检索接口，并编写了10条接口测试"
        for section in ("work_experience", "internship"):
            with self.subTest(section=section):
                extracted = _empty_collected_info()
                extracted[section] = [
                    {
                        "company": "测试公司",
                        "description": "用 Python 实现文档检索接口，并编写了10条接口测试",
                    }
                ]
                result = await self._wrap_with_extraction(snippet, snippet, extracted)
                self.assertEqual(result["collected_info"], extracted)
                self.assertEqual(result["collected_info"]["projects"], [])

    async def test_wrap_keeps_a_new_project_when_extraction_only_found_another(self):
        message = "我在 A 项目整理用户访谈。FastAPI，我实现了文档检索接口"
        snippet = "FastAPI，我实现了文档检索接口"
        extracted = _empty_collected_info()
        extracted["projects"] = [{"name": "A", "description": "整理用户访谈"}]

        result = await self._wrap_with_extraction(message, snippet, extracted)

        self.assertEqual(len(result["collected_info"]["projects"]), 2)
        self.assertEqual(result["collected_info"]["projects"][0], extracted["projects"][0])
        self.assertEqual(result["collected_info"]["projects"][1]["description"], snippet)

    async def test_wrap_keeps_raw_fallback_after_extraction_failure_or_rejection(self):
        current = _empty_collected_info()
        current["projects"] = [{"name": "旧项目", "description": "整理用户访谈"}]
        snippet = "我在新项目参与调研，并整理了12份访谈笔记"
        invented = copy.deepcopy(current)
        invented["projects"].append({"name": "新项目", "description": "主导调研推动上线"})
        for extracted in (RuntimeError("offline extraction failure"), invented):
            with self.subTest(extracted=type(extracted).__name__):
                result = await self._wrap_with_extraction(
                    snippet, snippet, extracted, current=current
                )
                projects = result["collected_info"]["projects"]
                self.assertEqual(len(projects), 2)
                self.assertEqual(projects[0], current["projects"][0])
                self.assertEqual(projects[1]["description"], snippet)
                self.assertNotIn("主导", projects[1]["description"])

    async def test_wrap_uses_existing_chinese_name_for_new_raw_facts(self):
        current = _empty_collected_info()
        current["projects"] = [{"name": "测试知识库", "description": "实现文档检索接口"}]
        snippet = "测试知识库，并编写了10条接口测试"

        result = await self._wrap_with_extraction(
            snippet, snippet, RuntimeError("offline extraction failure"), current=current
        )

        projects = result["collected_info"]["projects"]
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["name"], "测试知识库")
        self.assertIn("实现文档检索接口", projects[0]["description"])
        self.assertIn("编写了10条接口测试", projects[0]["description"])

    async def test_wrap_does_not_append_raw_text_to_an_updated_project(self):
        current = _empty_collected_info()
        current["projects"] = [{"name": "测试知识库", "description": "实现文档检索接口"}]
        extracted = copy.deepcopy(current)
        extracted["projects"][0]["description"] += "，并编写了10条接口测试"
        snippet = "测试知识库，并编写了10条接口测试"

        result = await self._wrap_with_extraction(snippet, snippet, extracted, current)

        self.assertEqual(result["collected_info"], extracted)

    async def test_stream_emits_only_public_node_messages(self):
        class FakeGraph:
            async def astream(self, *_args, **_kwargs):
                yield {"triage": {"messages": [AIMessage(content="INTERNAL_TRIAGE")]}}
                yield {
                    "collecting": {
                        "messages": [AIMessage(content="请讲讲你的项目经历")]
                    }
                }
                yield {"decoder": {"messages": [AIMessage(content="INTERNAL_DECODER")]}}

            async def aget_state(self, _config):
                return SimpleNamespace(
                    values={
                        "messages": [AIMessage(content="请讲讲你的项目经历")],
                        "resume_data": {"basic_info": {"name": "张三"}},
                        "resume_draft": "",
                        "stage": "COLLECTING",
                    }
                )

        with (
            patch("app.agents.resume_graph.get_graph", return_value=FakeGraph()),
            patch("app.services.session_store.save_resume_session") as save,
        ):
            chunks = [
                chunk
                async for chunk in stream_agent([{"content": "你好"}], "s1", object())
            ]
        self.assertEqual(chunks, ["请讲讲你的项目经历"])
        save.assert_called_once()

    async def test_stream_error_hides_internal_exception(self):
        class FailingGraph:
            async def astream(self, *_args, **_kwargs):
                raise RuntimeError("private model diagnostics")
                yield {}  # pragma: no cover

        with patch("app.agents.resume_graph.get_graph", return_value=FailingGraph()):
            chunks = [
                chunk
                async for chunk in stream_agent([{"content": "你好"}], "s1", object())
            ]
        self.assertEqual(chunks, ["生成失败，请稍后重试。"])


if __name__ == "__main__":
    unittest.main()
