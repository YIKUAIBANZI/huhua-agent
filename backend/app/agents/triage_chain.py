"""
Triage Agent · 对话路由决策器

职责：看完用户最新消息 + 当前 state，决定下一步动作（ACTION enum）。
不生成用户可见内容，只输出结构化决策。由 resume_graph 路由到对应子 prompt。

决策树优先级（从上到下）：
  1. 用户粘贴 JD 原文 (>50 字)    → DECODE_JD
  2. 用户明确跳过 JD                → SKIP_JD_OK
  3. 用户回复模糊("没有"/"不清楚") → ASK_CLARIFY
  4. 轻量模式无岗位关键词           → LIGHT_MODE_ASK
  5. 首次且未跳过                   → ASK_JD
  6. 用户主动问该做什么 / dig_attempts≥2 → SUGGEST_PROJECT
  7. 用户提到项目/动作/习惯         → WRAP_EXPERIENCE
  8. 经历为空                       → ASK_EXPERIENCE
  9. 经历≥1 且用户说开始/好了       → READY_TO_BUILD
"""

import logging
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TRIAGE_TIMEOUT_SECONDS = 15.0


class TriageAction(str, Enum):
    DECODE_JD = "DECODE_JD"
    SKIP_JD_OK = "SKIP_JD_OK"
    ASK_CLARIFY = "ASK_CLARIFY"
    LIGHT_MODE_ASK = "LIGHT_MODE_ASK"
    ASK_JD = "ASK_JD"
    WRAP_EXPERIENCE = "WRAP_EXPERIENCE"
    ASK_EXPERIENCE = "ASK_EXPERIENCE"
    SUGGEST_PROJECT = "SUGGEST_PROJECT"
    READY_TO_BUILD = "READY_TO_BUILD"


class TriageDecision(BaseModel):
    action: TriageAction = Field(..., description="下一步动作")
    reason: str = Field(..., description="为什么选这个动作（一句话）")
    experience_snippet: str = Field(
        default="", description="若 WRAP_EXPERIENCE，这里是从用户消息抽出的原始经历片段"
    )
    jd_keywords: list[str] = Field(
        default_factory=list,
        description="若 LIGHT_MODE_ASK 后用户给了方向，这里是抽出的 3-5 个岗位关键词",
    )
    dig_attempt_delta: int = Field(
        default=0,
        description="本轮对 dig_attempts 的增量：0（成功/不相关）/+1（挖不出）/+2（用户说没做过）",
    )


TRIAGE_SYSTEM = """你是「胡话简历」的对话路由判官。看完最新用户消息和当前状态，决定下一步走哪个子 agent。

# 当前状态
- 已有 JD 解码：{has_jd}
- 用户已跳过 JD：{jd_skipped}
- 已收集经历数：{experience_count} 条
- 已尝试挖掘失败次数 (dig_attempts)：{dig_attempts}
- 用户是否主动问过"我该做什么"：{user_asked_for_suggestion}
- 当前阶段：{stage}

# 最新用户消息
{user_message}

# 决策规则（按优先级从上到下）

1. 用户消息 >50 字且像岗位描述（含"职位要求/岗位职责/任职要求/薪资"等）→ **DECODE_JD**

2. 用户明确表示跳过 JD（关键词：没JD/不要JD/跳过/直接/不用/先随便/通用）→ **SKIP_JD_OK**
   （即使没用这些原词，只要语义明显"我现在就是不想给JD"也算）

3. 用户回复模糊、只说一两个字（"没有"/"不清楚"/"？"/"嗯"等）且当前在问 JD 阶段 → **ASK_CLARIFY**
   （让用户明确：是没 JD 原文，还是想跳过 JD 做通用简历）

4. 已 jd_skipped=true 但 jd_keywords 为空 → **LIGHT_MODE_ASK**
   （让用户一句话描述心仪岗位）

5. 用户消息包含岗位方向描述（如"找字节 AI 产品实习"/"想投 AI 工程师"）且还在 LIGHT_MODE 阶段 → 填充 jd_keywords 字段，返回 **SKIP_JD_OK**（复用该 action 推进流程）

6. 首次对话 + 未 skipped + 没问过 ASK_CLARIFY → **ASK_JD**

7. 用户**主动问**"我该做什么 / 怎么办 / 没经验怎么写"类 → **SUGGEST_PROJECT**
   （user_asked_for_suggestion 标志打开）

8. dig_attempts ≥ 2 → **SUGGEST_PROJECT**

9. 用户消息提到具体经历信号（触发词很宽松）→ **WRAP_EXPERIENCE**
   识别标准：
   - 明确项目名（如"TripAgent / Forge Skill"）
   - 动作叙述（"我做过 / 我负责 / 我设计过"）
   - 习惯叙述（"我平时会 / 我习惯 / 我通常先"）← 这些也算经历
   - 具体工具使用（"用 Claude 写过"/"用 Python 跑"）
   把用户原文中的经历描述片段填到 experience_snippet 字段

10. 用户明确说"没做过这方面 / 没有相关经验 / 完全没有" → **ASK_EXPERIENCE**
    并设 dig_attempt_delta = +2（直接触发阈值）

11. 经历数 >= 1 且用户说"开始/好了/够了/生成" → **READY_TO_BUILD**

12. 经历数 == 0 且其他都不匹配 → **ASK_EXPERIENCE**

# 挖掘计数规则（填 dig_attempt_delta）

- 用户说"没做过/没相关/完全没有"→ +2（直接到阈值，下轮可 SUGGEST）
- 用户否定上轮 Wrap 结果（"不对/不是这个意思/我没这么做过"）→ +1
- 成功挖出 / 用户接受 bullet → 0
- 其他情况 → 0

# 输出

严格输出结构化 JSON，不要任何 markdown 代码块标记、不要解释文字。
"""


async def run_triage(
    user_message: str,
    has_jd: bool,
    jd_skipped: bool,
    experience_count: int,
    dig_attempts: int,
    user_asked_for_suggestion: bool,
    stage: str,
    llm: ChatOpenAI,
) -> TriageDecision:
    """跑一次 Triage 决策。返回 TriageDecision 结构体。"""
    structured_llm = llm.with_structured_output(TriageDecision)
    prompt = TRIAGE_SYSTEM.format(
        has_jd=has_jd,
        jd_skipped=jd_skipped,
        experience_count=experience_count,
        dig_attempts=dig_attempts,
        user_asked_for_suggestion=user_asked_for_suggestion,
        stage=stage,
        user_message=user_message,
    )
    # 函数内 import 避开 ruff 的 top-level "未使用 import" 激进删除
    from asyncio import TimeoutError as _AsyncTimeout
    from asyncio import wait_for

    try:
        result: TriageDecision = await wait_for(
            structured_llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=TRIAGE_TIMEOUT_SECONDS,
        )
        return result
    except _AsyncTimeout:
        logger.warning(
            f"[triage/timeout] stage={stage} user_message={user_message[:60]!r} "
            f"后端 LLM 超过 {TRIAGE_TIMEOUT_SECONDS}s 未返回，走兜底"
        )
        stage_u = (stage or "").upper()
        if stage_u == "JD_INPUT":
            fallback = TriageAction.ASK_JD
        else:
            fallback = TriageAction.ASK_EXPERIENCE
        return TriageDecision(
            action=fallback,
            reason=f"triage LLM 超时 {TRIAGE_TIMEOUT_SECONDS}s，按 stage={stage} 兜底",
        )
    except Exception as e:
        logger.warning(f"[triage/error] stage={stage} err={e}")
        return TriageDecision(
            action=TriageAction.ASK_EXPERIENCE,
            reason=f"triage 异常，兜底路径: {e}",
        )


def build_state_for_triage(state: dict[str, Any]) -> dict[str, Any]:
    """从 ResumeState 抽出 Triage 需要的字段，方便 node 直接调用。"""
    collected = state.get("collected_info") or {}
    exp_count = (
        len(collected.get("work_experience", []))
        + len(collected.get("internship", []))
        + len(collected.get("projects", []))
    )
    return {
        "has_jd": bool(state.get("jd_info")),
        "jd_skipped": bool(state.get("jd_skipped", False)),
        "experience_count": exp_count,
        "dig_attempts": int(state.get("dig_attempts", 0)),
        "user_asked_for_suggestion": bool(
            state.get("user_asked_for_suggestion", False)
        ),
        "stage": state.get("stage", "JD_INPUT"),
    }
