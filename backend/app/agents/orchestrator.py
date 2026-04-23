"""
主对话 Orchestrator（v0）

职责：
- 维持对话历史，理解用户意图
- 按需调用 decoder / wrapper / evaluator 工具
- 流式返回 token
"""

import logging
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.agents.decoder_chain import run_decoder_chain
from app.agents.evaluator_chain import run_evaluator_chain
from app.agents.wrapper_chain import run_wrapper_chain

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是「胡话简历」AI助手，帮用户打造一份有竞争力的简历。

你的能力：
1. 解码JD黑话 — 用户粘贴招聘JD，你揭露其中的职场潜台词
2. 包装工作经历 — 用户描述自己做过什么，你帮他改写成大厂风格的STAR-V格式
3. 评估简历 — 用户提供简历文本，你从ATS和HR两个维度打分并给改进建议

对话原则：
- 每次只问一个问题，不要同时问多件事
- 用户经历不足时，主动给出具体的写法建议，用「可以这么写：...」格式，并询问是否采用
- 语气直接，带点毒舌但有建设性

当前对话目标：先了解用户的求职意向，再逐步收集信息，最终帮他完成一份简历。
"""


def _to_lc_messages(messages: list[dict]) -> list:
    result = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in messages:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        else:
            result.append(AIMessage(content=m["content"]))
    return result


@tool
async def decode_jd(jd_text: str) -> str:
    """解码招聘JD，揭露职场黑话的真实含义，提取关键词和简历策略。输入：JD原文"""
    try:
        from langchain_openai import ChatOpenAI
        from app.config import get_settings

        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.8,
        )
        result = await run_decoder_chain(jd_text, llm)
        items = result.get("decoded_items", [])
        summary = result.get("summary", "")
        decoded = "\n".join(f"• {i['keyword']}：{i['real_meaning']}" for i in items[:6])
        strategy = result.get("jd_info", {}).get("resume_strategy", {})
        must = "、".join(strategy.get("must_highlight", []))
        return f"**JD解码结果**\n\n{decoded}\n\n**总结：** {summary}\n\n**简历必须突出：** {must}"
    except Exception as e:
        logger.error(f"decode_jd error: {e}")
        return f"解码出错：{e}"


@tool
async def wrap_experience(experience: str, industry: str = "互联网") -> str:
    """将用户的工作经历描述包装成大厂风格的STAR-V格式bullet point。
    输入：experience=用户原始描述，industry=行业"""
    try:
        from langchain_openai import ChatOpenAI
        from app.config import get_settings

        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.7,
        )
        result = await run_wrapper_chain(
            experience=experience, industry=industry, llm=llm
        )
        if result.get("needs_followup"):
            return f"需要更多信息：{result.get('question', '能详细说说吗？')}"
        return result.get("wrapped", "包装失败，请重试")
    except Exception as e:
        logger.error(f"wrap_experience error: {e}")
        return f"包装出错：{e}"


@tool
async def evaluate_resume(resume_text: str, jd_context: str = "") -> str:
    """对简历进行ATS+HR双维度评分，给出通过率预测和改进建议。
    输入：resume_text=简历文本，jd_context=目标JD（可选）"""
    try:
        from langchain_openai import ChatOpenAI
        from app.config import get_settings

        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.2,
        )
        result = await run_evaluator_chain(resume_text, llm, jd_context or None)
        ats = result.get("ats_score", {}).get("total", 0)
        hr = result.get("hr_score", {}).get("total", 0)
        total = result.get("total_score", 0)
        prediction = result.get("pass_prediction", {})
        roadmap = result.get("improvement_roadmap", [])[:3]
        top3 = "\n".join(
            f"{i + 1}. {r.get('action', '')}（预计+{r.get('expected_score_gain', '')}分）"
            for i, r in enumerate(roadmap)
        )
        return (
            f"**简历评分**\n\n"
            f"总分：{total:.0f} | ATS：{ats} | HR：{hr}\n"
            f"ATS通过率：{prediction.get('ats_pass_rate', '-')} | "
            f"面试概率：{prediction.get('interview_probability', '-')}\n\n"
            f"**优先改进：**\n{top3}"
        )
    except Exception as e:
        logger.error(f"evaluate_resume error: {e}")
        return f"评估出错：{e}"


TOOLS = [decode_jd, wrap_experience, evaluate_resume]


async def stream_chat(messages: list[dict], llm: ChatOpenAI) -> AsyncIterator[str]:
    """流式对话，支持 tool call"""
    llm_with_tools = llm.bind_tools(TOOLS)
    lc_messages = _to_lc_messages(messages)

    # 第一次调用，看是否有 tool call
    response = await llm_with_tools.ainvoke(lc_messages)

    if response.tool_calls:
        # 执行工具
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            yield f"\n\n*正在{_tool_action(tool_name)}...*\n\n"

            tool_func = next((t for t in TOOLS if t.name == tool_name), None)
            if tool_func:
                tool_result = await tool_func.ainvoke(tool_args)
            else:
                tool_result = "工具不存在"

            # 把工具结果加入上下文，再让 LLM 生成最终回复
            lc_messages.append(response)
            lc_messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
            )

        # 流式输出最终回复
        async for chunk in llm.astream(lc_messages):
            if chunk.content:
                yield chunk.content
    else:
        # 纯对话，直接流式输出
        async for chunk in llm_with_tools.astream(lc_messages):
            if chunk.content:
                yield chunk.content


def _tool_action(name: str) -> str:
    return {
        "decode_jd": "解码JD",
        "wrap_experience": "包装经历",
        "evaluate_resume": "评估简历",
    }.get(name, "处理")
