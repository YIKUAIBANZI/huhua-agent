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
