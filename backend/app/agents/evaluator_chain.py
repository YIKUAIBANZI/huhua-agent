import json
import logging

from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.agents.prompts import EVALUATOR_PROMPT
from app.schemas.evaluator import EvaluateResponse

logger = logging.getLogger(__name__)

_LLM_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    before_sleep=lambda rs: logger.warning(
        f"Evaluator LLM 重试第 {rs.attempt_number} 次"
    ),
)


def _format_jd_context(jd_context: dict | None) -> str:
    """格式化 JD 上下文为文本。"""
    if not jd_context:
        return "未提供 JD，仅做通用质量评估"
    parts: list[str] = []
    for key, label in [
        ("job_title", "岗位"),
        ("industry", "行业"),
        ("experience_years", "经验要求"),
        ("education", "学历要求"),
    ]:
        if jd_context.get(key):
            parts.append(f"{label}：{jd_context[key]}")
    for key, label in [
        ("required_skills", "要求技能"),
        ("key_responsibilities", "核心职责"),
        ("hard_requirements", "硬性门槛"),
    ]:
        if jd_context.get(key):
            parts.append(f"{label}：{', '.join(jd_context[key])}")
    kw_cloud = jd_context.get("keyword_cloud", {})
    if kw_cloud.get("core_keywords"):
        parts.append(f"核心关键词：{', '.join(kw_cloud['core_keywords'])}")
    return "\n".join(parts) if parts else "未提供 JD，仅做通用质量评估"


@_LLM_RETRY
async def run_evaluator_chain(
    resume_text: str,
    jd_context: dict | None,
    llm: ChatOpenAI | None = None,
) -> dict:
    """双阶段简历评分：ATS 模拟(60分) + HR 模拟(40分)。"""
    if llm is None:
        return EvaluateResponse().model_dump()

    jd_str = _format_jd_context(jd_context)
    prompt_vars = {"jd_context": jd_str, "resume_text": resume_text}

    # 尝试结构化输出
    try:
        structured_llm = llm.with_structured_output(EvaluateResponse)
        chain = EVALUATOR_PROMPT | structured_llm
        result = await chain.ainvoke(prompt_vars)
        return result.model_dump()
    except Exception as e:
        logger.info(f"结构化输出不可用，回退手工解析: {e}")

    # 回退：手工解析
    chain = EVALUATOR_PROMPT | llm
    output = await chain.ainvoke(prompt_vars)
    return _parse_output(output.content)


def _parse_output(content: str) -> dict:
    """解析评分 JSON 输出。"""
    text = content.strip()

    # 去掉 markdown 代码块
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    try:
        raw = json.loads(text)
        return EvaluateResponse(**raw).model_dump()
    except (json.JSONDecodeError, Exception):
        pass

    # 回退提取
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            raw = json.loads(text[start : end + 1])
            return EvaluateResponse(**raw).model_dump()
        except (json.JSONDecodeError, Exception):
            pass

    logger.warning(f"简历评分 JSON 解析失败: {text[:300]}")
    return EvaluateResponse().model_dump()
