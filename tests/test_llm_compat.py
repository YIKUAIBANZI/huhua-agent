import copy
import json
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.editor_chain import run_editor  # noqa: E402
from app.agents.evaluator_chain import run_evaluator_chain  # noqa: E402
from app.agents.triage_chain import TriageAction, run_triage  # noqa: E402
from app.services.llm_compat import with_structured_output  # noqa: E402


class _Answer(BaseModel):
    answer: str


_UNSET = object()


class LLMCompatTest(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def offline_llm(
        self,
        *,
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        extra_body=None,
        output=None,
        root_base_url=None,
    ):
        """Exercise the SDK serializer and parser without storing request headers."""
        requests = []
        output = {"answer": "parsed offline"} if output is None else output

        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            message = {"role": "assistant", "content": json.dumps(output)}
            finish_reason = "stop"
            if body.get("tools"):
                message["content"] = None
                message["tool_calls"] = [
                    {
                        "id": "call_offline",
                        "type": "function",
                        "function": {
                            "name": body["tools"][0]["function"]["name"],
                            "arguments": json.dumps(output),
                        },
                    }
                ]
                finish_reason = "tool_calls"
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-offline",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "message": message,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "total_tokens": 19,
                    },
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ) as http_client:
            kwargs = {
                "model": model,
                "api_key": "offline-test-key",
                "http_async_client": http_client,
                "max_retries": 0,
                "extra_body": extra_body,
            }
            if base_url is not _UNSET:
                kwargs["base_url"] = base_url
            if root_base_url is not None:
                root_client = AsyncOpenAI(
                    api_key="offline-test-key",
                    base_url=root_base_url,
                    http_client=http_client,
                    max_retries=0,
                )
                kwargs["root_async_client"] = root_client
                kwargs["async_client"] = root_client.chat.completions
            # Make the omitted-base-url cases independent of local configuration.
            with patch.dict(os.environ, {"OPENAI_API_BASE": ""}):
                llm = ChatOpenAI(**kwargs)
                yield llm, requests

    def assert_deepseek_request(self, body):
        self.assertNotIn("response_format", body)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["tool_choice"]["type"], "function")
        self.assertEqual(len(body["tools"]), 1)
        self.assertIs(body["tools"][0]["function"].get("strict"), False)

    async def test_deepseek_tool_call_is_parsed_into_pydantic(self):
        async with self.offline_llm() as (llm, requests):
            result = await with_structured_output(llm, _Answer).ainvoke("test")
        self.assertIsInstance(result, _Answer)
        self.assertEqual(result.answer, "parsed offline")
        self.assertEqual(len(requests), 1)
        self.assert_deepseek_request(requests[0])

    async def test_deepseek_endpoint_with_v1_takes_priority_over_model_name(self):
        async with self.offline_llm(
            model="custom-model", base_url="https://api.deepseek.com/v1"
        ) as (llm, requests):
            result = await with_structured_output(llm, _Answer).ainvoke("test")
        self.assertEqual(result.answer, "parsed offline")
        self.assert_deepseek_request(requests[0])

    async def test_structured_call_does_not_mutate_original_model_or_extra_body(self):
        extra_body = {
            "thinking": {"type": "enabled", "budget": 128},
            "provider_option": {"nested": ["preserve"]},
        }
        before = copy.deepcopy(extra_body)
        async with self.offline_llm(extra_body=extra_body) as (llm, requests):
            original_extra_body = llm.extra_body
            original_model_kwargs = copy.deepcopy(llm.model_kwargs)
            original_clients = (llm.client, llm.async_client)

            await with_structured_output(llm, _Answer).ainvoke("structured")
            self.assertIs(llm.extra_body, original_extra_body)
            self.assertEqual(llm.extra_body, before)
            self.assertEqual(extra_body, before)
            self.assertEqual(llm.model_kwargs, original_model_kwargs)
            self.assertEqual((llm.client, llm.async_client), original_clients)

            await llm.ainvoke("ordinary generation")

        self.assertEqual(len(requests), 2)
        self.assert_deepseek_request(requests[0])
        self.assertEqual(requests[0]["provider_option"], before["provider_option"])
        self.assertEqual(requests[1]["thinking"], before["thinking"])
        self.assertEqual(requests[1]["provider_option"], before["provider_option"])
        self.assertNotIn("tools", requests[1])

    async def test_openai_and_dashscope_keep_actual_sdk_defaults(self):
        providers = [
            ("gpt-4o", "https://api.openai.com/v1"),
            ("qwen3-max", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ]
        for model, base_url in providers:
            with self.subTest(model=model):
                async with self.offline_llm(
                    model=model,
                    base_url=base_url,
                    extra_body={"provider_option": {"value": "preserve"}},
                ) as (llm, requests):
                    expected = await llm.with_structured_output(_Answer).ainvoke("test")
                    actual = await with_structured_output(llm, _Answer).ainvoke("test")
                self.assertEqual(actual, expected)
                self.assertEqual(requests[1], requests[0])
                self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
                self.assertNotIn("thinking", requests[1])

    async def test_explicit_non_deepseek_endpoint_preserves_defaults_for_deepseek_model(self):
        endpoints = [
            "https://api.openai.com/v1",
            "https://api.deepseek.com.evil.invalid/v1",
            "https://proxy.invalid/api.deepseek.com/v1",
        ]
        for base_url in endpoints:
            with self.subTest(base_url=base_url):
                async with self.offline_llm(base_url=base_url) as (llm, requests):
                    await llm.with_structured_output(_Answer).ainvoke("test")
                    result = await with_structured_output(llm, _Answer).ainvoke("test")
                self.assertEqual(result.answer, "parsed offline")
                self.assertEqual(requests[1], requests[0])
                self.assertNotIn("thinking", requests[1])

    async def test_official_deepseek_models_are_recognized_without_explicit_base(self):
        for model in ("deepseek-chat", "deepseek-reasoner", "deepseek-flash", "deepseek-v4-pro"):
            with self.subTest(model=model):
                async with self.offline_llm(model=model, base_url=_UNSET) as (llm, requests):
                    self.assertIsNone(llm.openai_api_base)
                    result = await with_structured_output(llm, _Answer).ainvoke("test")
                self.assertEqual(result.answer, "parsed offline")
                self.assert_deepseek_request(requests[0])

    async def test_unknown_deepseek_model_prefix_does_not_change_defaults(self):
        async with self.offline_llm(
            model="deepseek-custom", base_url=_UNSET
        ) as (llm, requests):
            await llm.with_structured_output(_Answer).ainvoke("test")
            await with_structured_output(llm, _Answer).ainvoke("test")
        self.assertEqual(requests[1], requests[0])
        self.assertNotIn("thinking", requests[1])

    async def test_actual_async_root_client_identifies_deepseek_endpoint(self):
        async with self.offline_llm(
            model="custom-model",
            base_url=_UNSET,
            root_base_url="https://api.deepseek.com/v1",
        ) as (llm, requests):
            self.assertIsNone(llm.openai_api_base)
            result = await with_structured_output(llm, _Answer).ainvoke("test")
        self.assertEqual(result.answer, "parsed offline")
        self.assert_deepseek_request(requests[0])

    async def test_fake_llm_with_single_schema_argument_remains_compatible(self):
        class FakeLLM:
            model_name = "deepseek-chat"
            openai_api_base = "https://api.deepseek.com"

            def __init__(self):
                self.schemas = []
                self.structured = RunnableLambda(lambda _: _Answer(answer="fake"))

            def with_structured_output(self, schema):
                self.schemas.append(schema)
                return self.structured

        fake = FakeLLM()
        structured = with_structured_output(fake, _Answer)
        self.assertIs(structured, fake.structured)
        self.assertEqual((await structured.ainvoke("test")).answer, "fake")
        self.assertEqual(fake.schemas, [_Answer])

    async def test_triage_chain_uses_deepseek_tool_call(self):
        output = {"action": "SKIP_JD_OK", "reason": "用户选择通用简历"}
        async with self.offline_llm(output=output) as (llm, requests):
            result = await run_triage(
                user_message="跳过 JD",
                has_jd=False,
                jd_skipped=False,
                experience_count=0,
                dig_attempts=0,
                user_asked_for_suggestion=False,
                stage="JD_INPUT",
                llm=llm,
            )
        self.assertEqual(result.action, TriageAction.SKIP_JD_OK)
        self.assertEqual(result.reason, output["reason"])
        self.assertEqual(len(requests), 1)
        self.assert_deepseek_request(requests[0])

    async def test_editor_chain_uses_deepseek_tool_call(self):
        source = {
            "skills": ["Python"],
            "projects": [{"name": "简历工具", "description": "参与调研；整理访谈笔记"}],
        }
        output = {
            "skill_groups": {"技术栈": ["Python"]},
            "experience_section_title": "项目经历",
            "project_polishes": [
                {"name": "简历工具", "polished_bullets": ["整理访谈笔记", "参与调研"]}
            ],
        }
        async with self.offline_llm(output=output) as (llm, requests):
            result = await run_editor(source, llm)
        self.assertEqual(result["skill_groups"], output["skill_groups"])
        self.assertEqual(result["experience_section_title"], "项目经历")
        self.assertEqual(result["projects"][0]["polished_bullets"], ["整理访谈笔记", "参与调研"])
        self.assertEqual(len(requests), 1)
        self.assert_deepseek_request(requests[0])

    async def test_evaluator_chain_uses_deepseek_tool_call_without_fallback(self):
        output = {"total_score": 73, "ats_score": {"total": 43}, "hr_score": {"total": 30}}
        async with self.offline_llm(output=output) as (llm, requests):
            result = await run_evaluator_chain("参与简历工具调研", None, llm)
        self.assertEqual(result["total_score"], 73)
        self.assertEqual(result["ats_score"]["total"], 43)
        self.assertEqual(result["hr_score"]["total"], 30)
        self.assertEqual(len(requests), 1)
        self.assert_deepseek_request(requests[0])


if __name__ == "__main__":
    unittest.main()
