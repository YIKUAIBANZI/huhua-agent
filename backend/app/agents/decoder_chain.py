import json
import logging

from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.agents.prompts import JD_DECODER_PROMPT

logger = logging.getLogger(__name__)

_LLM_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    before_sleep=lambda rs: logger.warning(
        f"Decoder LLM 重试第 {rs.attempt_number} 次"
    ),
)


@_LLM_RETRY
async def run_decoder_chain(
    jd_text: str,
    llm: ChatOpenAI | None = None,
) -> dict:
    """合并后的 JD 解码：单次 LLM 调用同时输出结构化 JSON + 毒舌解码。"""
    import asyncio

    context = ""
    from app.config import get_settings

    if not get_settings().ENABLE_RAG:
        return await _run_without_rag(jd_text, llm)
    try:

        def _retrieve():
            from app.rag.vector_store import get_retriever

            retriever = get_retriever(collection="jargon_dict", k=5)
            return retriever.invoke(jd_text)

        _task = asyncio.create_task(asyncio.to_thread(_retrieve))
        done, _ = await asyncio.wait([_task], timeout=8.0)
        if _task in done:
            docs = _task.result()
            context = "\n".join([f"- {d.page_content}" for d in docs])
        else:
            logger.warning("RAG retrieval timed out, proceeding without context")
    except Exception as e:
        logger.warning(f"RAG retrieval skipped: {e}")

    if llm is None:
        return {
            "decoded_items": [
                {
                    "keyword": "示例关键词",
                    "surface_meaning": "[待接入 LLM]",
                    "real_meaning": "[待接入 LLM]",
                }
            ],
            "summary": f"[待接入 LLM] JD 长度：{len(jd_text)} 字",
            "jd_info": _empty_jd_info(),
        }

    # 单次调用，统一 Prompt 同时输出 JSON + 毒舌文本
    chain = JD_DECODER_PROMPT | llm
    output = await chain.ainvoke({"context": context, "jd_text": jd_text})

    content = output.content
    jd_info, decode_text = _split_output(content)
    decoded_items, summary = _parse_decode_text(decode_text)

    return {
        "decoded_items": decoded_items,
        "summary": summary,
        "jd_info": jd_info,
    }


async def _run_without_rag(jd_text: str, llm: ChatOpenAI | None) -> dict:
    if llm is None:
        return {
            "decoded_items": [],
            "summary": f"[待接入 LLM] JD 长度：{len(jd_text)} 字",
            "jd_info": _empty_jd_info(),
        }
    chain = JD_DECODER_PROMPT | llm
    output = await chain.ainvoke({"context": "", "jd_text": jd_text})
    jd_info, decode_text = _split_output(output.content)
    decoded_items, summary = _parse_decode_text(decode_text)
    return {"decoded_items": decoded_items, "summary": summary, "jd_info": jd_info}


def _split_output(content: str) -> tuple[dict, str]:
    """将 LLM 输出按 --- 分隔为 JSON 部分和毒舌文本部分。"""
    # 尝试用 --- 分隔
    if "---" in content:
        parts = content.split("---", 1)
        json_part = parts[0].strip()
        decode_part = parts[1].strip() if len(parts) > 1 else ""
    else:
        # 回退：尝试找 JSON 块的结尾，剩余部分为解码文本
        json_end = _find_json_end(content)
        if json_end > 0:
            json_part = content[:json_end].strip()
            decode_part = content[json_end:].strip()
        else:
            json_part = content
            decode_part = ""

    jd_info = _parse_json_part(json_part)
    return jd_info, decode_part


def _find_json_end(text: str) -> int:
    """找到最外层 JSON 对象的结束位置。"""
    start = text.find("{")
    if start == -1:
        return -1
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _parse_json_part(text: str) -> dict:
    """从文本中提取并解析 JSON。"""
    text = text.strip()
    # 去掉 markdown 代码块标记
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline != -1 else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # 直接尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 回退：提取第一个 { 到最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"JD 信息提取 JSON 解析失败: {text[:200]}")
    return _empty_jd_info()


def _parse_decode_text(content: str) -> tuple[list[dict], str]:
    """解析毒舌解码文本部分。"""
    lines = content.strip().split("\n")
    decoded_items: list[dict] = []
    summary = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("【") and "】" in line and "→" in line:
            keyword_end = line.index("】")
            keyword = line[1:keyword_end]
            rest = line[keyword_end + 1 :]
            parts = rest.split("→", 1)
            surface = parts[0].strip()
            real = parts[1].strip() if len(parts) > 1 else ""
            decoded_items.append(
                {
                    "keyword": keyword,
                    "surface_meaning": surface,
                    "real_meaning": real,
                }
            )
        else:
            # 最后一行非格式化的文本作为总结
            summary = line

    if not decoded_items:
        decoded_items = [
            {
                "keyword": "解析失败",
                "surface_meaning": "",
                "real_meaning": content,
            }
        ]

    return decoded_items, summary


def _empty_jd_info() -> dict:
    return {
        "job_title": "",
        "company": "",
        "industry": "",
        "experience_years": "",
        "education": "",
        "required_skills": [],
        "key_responsibilities": [],
        "preferred_qualities": [],
        "salary_range": "",
        "hard_requirements": [],
        "keyword_cloud": {
            "core_keywords": [],
            "secondary_keywords": [],
            "industry_keywords": [],
        },
        "resume_strategy": {
            "must_highlight": [],
            "recommended_format": "",
            "keyword_density_targets": [],
        },
    }
