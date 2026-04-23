"""
LangGraph 版简历包装 Agent

状态图：
  输入 → evaluate（评估描述长度）
         ├─ 过短 → ask_followup（返回追问）
         └─ 足够 → retrieve（RAG 检索）→ generate（LLM 生成）→ 输出
"""

import logging
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from app.agents.prompts import WRAPPER_PROMPT

logger = logging.getLogger(__name__)

FOLLOW_UP_QUESTION = (
    "你的描述有点简短，帮我补充几点，包装效果会更好：\n"
    "1. 具体负责什么工作？\n"
    "2. 有没有量化成果（数字、比例、规模）？\n"
    "3. 团队规模或项目影响范围？"
)

MIN_LENGTH = 20


class WrapperState(TypedDict):
    experience: str
    industry: str
    follow_up_answer: str | None
    full_experience: str
    context: str
    references: list[str]
    needs_followup: bool
    question: str
    wrapped: str


# ── 节点 ────────────────────────────────────────────────────────


def evaluate(state: WrapperState) -> dict:
    """评估描述是否足够，合并补充信息"""
    exp = state["experience"]
    followup = state.get("follow_up_answer")

    if len(exp.strip()) < MIN_LENGTH and not followup:
        return {"needs_followup": True, "question": FOLLOW_UP_QUESTION}

    full = exp if not followup else f"{exp}\n补充：{followup}"
    return {"needs_followup": False, "full_experience": full}


async def retrieve(state: WrapperState) -> dict:
    """混合检索参考样本"""
    import asyncio

    def _retrieve():
        from app.rag.vector_store import get_retriever

        retriever = get_retriever(collection="golden_resumes", k=5)
        return retriever.invoke(state["full_experience"])

    docs = []
    try:
        _task = asyncio.create_task(asyncio.to_thread(_retrieve))
        done, _ = await asyncio.wait([_task], timeout=8.0)
        if _task in done:
            docs = _task.result()
        else:
            logger.warning("RAG retrieval timed out, proceeding without context")
    except Exception as e:
        logger.warning(f"RAG retrieval skipped: {e}")
    context = "\n".join([f"- {d.page_content}" for d in docs])
    references = [d.page_content for d in docs]
    return {"context": context, "references": references}


# ── 路由 ────────────────────────────────────────────────────────


def route_after_evaluate(state: WrapperState) -> str:
    return "ask_followup" if state.get("needs_followup") else "retrieve"


def ask_followup(state: WrapperState) -> dict:
    """直接返回，不做额外处理"""
    return {}


# ── 对外接口（保持与 service 层签名一致）─────────────────────────


async def run_wrapper_chain(
    experience: str,
    industry: str,
    llm: ChatOpenAI | None = None,
    follow_up_answer: str | None = None,
) -> dict:
    """对外接口不变，内部用 LangGraph 执行。

    LLM 通过闭包注入 generate 节点，避免 StateGraph 丢字段。
    """

    async def generate(state: WrapperState) -> dict:
        """调用 LLM 生成包装结果（通过闭包拿到 llm）"""
        if llm is None:
            return {"wrapped": f"[待接入 LLM] 原始输入：{state['full_experience']}"}

        chain = WRAPPER_PROMPT | llm
        output = await chain.ainvoke(
            {
                "industry": state["industry"],
                "context": state["context"],
                "experience": state["full_experience"],
            }
        )
        return {"wrapped": output.content}

    # 每次调用重新构建图（generate 依赖闭包中的 llm）
    builder = StateGraph(WrapperState)
    builder.add_node("evaluate", evaluate)
    builder.add_node("ask_followup", ask_followup)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.set_entry_point("evaluate")
    builder.add_conditional_edges("evaluate", route_after_evaluate)
    builder.add_edge("ask_followup", END)
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    graph = builder.compile()

    initial_state = {
        "experience": experience,
        "industry": industry,
        "follow_up_answer": follow_up_answer,
        "full_experience": "",
        "context": "",
        "references": [],
        "needs_followup": False,
        "question": "",
        "wrapped": "",
    }
    result = await graph.ainvoke(initial_state)

    if result.get("needs_followup"):
        return {"needs_followup": True, "question": result["question"]}

    return {
        "needs_followup": False,
        "wrapped": result["wrapped"],
        "references": result["references"],
    }
