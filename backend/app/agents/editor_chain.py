"""
编辑 Agent · 简历美化定稿器

职责（单次 LLM 调用，输入 ResumeData 粗稿 → 输出优化后的 ResumeData）：
  1. skills 扁平列表 → skill_groups 分组（技术栈 / 产品方法 / AI 工具 / 内容运营 / 其他）
  2. projects[].description 长段落 → polished_bullets 3-5 条 STAR 句
  3. 根据背景推断 experience_section_title（校园经历 / 工作经历 / 过往经历）

稳定性：15s 硬超时，LLM 任何异常都返回原始 ResumeData（不阻塞主流程）。
"""

import asyncio
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EDITOR_TIMEOUT_SECONDS = 15.0


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?%?", text or ""))


class _ProjectPolish(BaseModel):
    name: str = Field(..., description="项目名，原样返回用于匹配")
    polished_bullets: list[str] = Field(
        default_factory=list,
        description="3-5 条 STAR 格式 bullet，每条 ≤60 字，保留原数字，不编造",
    )


class EditorOutput(BaseModel):
    skill_groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="skills 按类别归组。key 用中文：技术栈 / 产品方法 / AI 工具 / 内容运营 / 设计工具 / 其他。去重，不新增",
    )
    experience_section_title: str = Field(
        default="",
        description=(
            "projects 字段对应的板块标题。按内容性质判断："
            "具体项目/产品/技术实践 → '项目经历'（默认，最常见）；"
            "只有在校奖项、社团、志愿活动等非项目性内容 → '校园经历'；"
            "不确定就返回空串让模板用默认"
        ),
    )
    project_polishes: list[_ProjectPolish] = Field(
        default_factory=list,
        description="对每个项目的 description 压缩结果；name 必须和输入一致以便匹配",
    )


EDITOR_SYSTEM = """你是「胡话简历」的简历美化师。读取下面的简历粗稿 JSON，产出结构化优化结果。

# 任务

## 任务 1：skills 分组
把 skills 扁平列表按下列类别归组，每条只归一组：
- **技术栈**：编程语言、框架、数据库（Java / Python / Spring Boot / MySQL / FastAPI / React 等）
- **产品方法**：PRD 撰写 / 竞品分析 / 用户调研 / 需求优先级排序 / STAR 项目包装 / 数据分析 等
- **AI 工具**：Claude Code / ChatGPT / Cursor / Gemini / OpenClaw / Prompt Engineering / LLM / RAG / Agent / MCP 等
- **设计工具**：Figma / Sketch / Google Stitch / Photoshop 等
- **内容运营**：抖音短视频创作 / 小红书 / 公众号写作 / 产品使用体验报告撰写 等
- **其他**：以上都不合适的才放这里

**规则**：原文照搬，不新增、不改写、不删减；每组内部用 `/` 分隔（由模板负责），这里只返回数组。

## 任务 2：projects 板块标题
判断 projects 数组里的内容性质（**只看 projects 字段，跟 work_experience 无关**）：

- **默认：`"项目经历"`** —— 只要 projects 里是具体的项目/产品/工具/技术实践（含独立开发、课程设计、实验室项目、开源贡献、竞赛作品等）都算
- **`"校园经历"`** —— 当 projects 里**全部**都是在校奖项、社团活动、志愿服务、班干部这类**非项目性**内容时才用

**语义区分举例**：
- "TripAgent 旅游规划 Agent" → 具体项目 → 项目经历
- "Forge Skill 开源工具" → 具体项目 → 项目经历
- "校园十佳歌手" / "学生会外联部副部长" / "支教团队队长" → 非项目 → 校园经历

projects 里既有项目又有活动时，按"项目经历"处理（多数场景用户关心的是项目部分）。
不确定就返回空串，由模板使用默认。

## 任务 3：projects description 压缩
每个项目，把 description 长段落压成 **3-5 条 STAR bullet**：
- 每条 ≤60 字
- **保留原有数字和项目名，绝对不能编造新数字**
- 每条以动词开头（主导 / 设计 / 完成 / 发现 / 推动 / 优化 等）
- bullet 之间逻辑：洞察→方案→动作→结果 的顺序

返回时 `project_polishes[].name` 必须和输入项目的 name 完全一致，便于前端 match。

# 输入 JSON

{resume_json}

# 输出

严格结构化 JSON，无多余文字、无 markdown 代码块。
"""


async def run_editor(resume_data: dict, llm: ChatOpenAI) -> dict:
    """
    输入粗稿 dict（ResumeData 形状），返回优化后的 dict。
    任何错误都原样返回输入（log warning），不阻塞主流程。
    """
    import json as _json

    structured_llm = llm.with_structured_output(EditorOutput)
    prompt = EDITOR_SYSTEM.format(
        resume_json=_json.dumps(resume_data, ensure_ascii=False, indent=2)
    )

    try:
        result: EditorOutput = await asyncio.wait_for(
            structured_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=EDITOR_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[editor/timeout] LLM 超过 {EDITOR_TIMEOUT_SECONDS}s 未返回，跳过美化"
        )
        return resume_data
    except Exception as e:
        logger.warning(f"[editor/error] {type(e).__name__} {e}")
        return resume_data

    # 合并结果回 resume_data
    merged: dict[str, Any] = dict(resume_data)  # 浅拷贝
    if result.skill_groups:
        merged["skill_groups"] = result.skill_groups
    if result.experience_section_title:
        merged["experience_section_title"] = result.experience_section_title

    # projects polished_bullets 按 name 匹配回填
    if result.project_polishes and merged.get("projects"):
        name_to_bullets = {p.name: p.polished_bullets for p in result.project_polishes}
        new_projects = []
        for p in merged["projects"]:
            p_copy = dict(p)
            bullets = name_to_bullets.get(p.get("name", ""), [])
            original_numbers = _numbers(p.get("description", ""))
            introduced_numbers = _numbers("\n".join(bullets)) - original_numbers
            if bullets and not introduced_numbers:
                p_copy["polished_bullets"] = bullets
            elif introduced_numbers:
                logger.warning(
                    "[editor/fact-check] rejected new numbers project=%r numbers=%s",
                    p.get("name", ""),
                    sorted(introduced_numbers),
                )
            new_projects.append(p_copy)
        merged["projects"] = new_projects

    logger.info(
        f"[editor/ok] skill_groups={len(result.skill_groups)} "
        f"section='{result.experience_section_title}' "
        f"polishes={len(result.project_polishes)}"
    )
    return merged
