"""
胡话简历 · LangGraph 状态机 Agent

状态流转：
  JD_INPUT → COLLECTING → BUILDING → EVALUATING → EXPORT

每个阶段有专属系统提示和明确目标。
Agent 用 interrupt() 暂停等待用户输入，用户回复后继续推进。
"""

import json
import logging
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agents.decoder_chain import run_decoder_chain
from app.agents.evaluator_chain import run_evaluator_chain
from app.agents.triage_chain import (
    TriageAction,
    build_state_for_triage,
    run_triage,
)
from app.agents.wrapper_chain import run_wrapper_chain

logger = logging.getLogger(__name__)

# ── 状态定义 ─────────────────────────────────────────────────────────────────

STAGES = ["JD_INPUT", "COLLECTING", "BUILDING", "EVALUATING", "EXPORT"]


class ResumeState(TypedDict):
    messages: Annotated[list, add_messages]
    stage: str
    jd_text: str
    jd_info: dict
    collected_info: dict
    resume_draft: str
    resume_data: dict  # 序列化后的 ResumeData（给 /resume 渲染用），building_node 产出
    eval_result: dict
    # ── Triage 相关字段 ─────────────────────────────────────────────
    jd_skipped: bool  # 用户明确跳过 JD
    jd_keywords: list  # 轻量模式下 Triage 抽取的岗位关键词
    dig_attempts: int  # 挖掘失败累计次数（阈值 2 触发 SUGGEST_PROJECT）
    user_asked_for_suggestion: bool  # 用户主动问"我该做什么"，跳过阈值


def _empty_collected_info() -> dict:
    return {
        "basic_info": {"name": "", "phone": "", "email": "", "location": ""},
        "education": [],
        "work_experience": [],
        "internship": [],
        "projects": [],
        "skills": [],
        "personal_summary": "",
    }


def _init_state() -> ResumeState:
    return {
        "messages": [],
        "stage": "JD_INPUT",
        "jd_text": "",
        "jd_info": {},
        "collected_info": _empty_collected_info(),
        "resume_draft": "",
        "resume_data": {},
        "eval_result": {},
        "jd_skipped": False,
        "jd_keywords": [],
        "dig_attempts": 0,
        "user_asked_for_suggestion": False,
    }


# ── Resume 序列化 ────────────────────────────────────────────────────────────
# 把对话收集到的 state → ResumeData dict，供 /resume 渲染。
# 硬性约束：身高/性别/年龄/籍贯/政治面貌/民族 默认空串，只有用户在对话里
# 主动陈述才填（例如 "我 167cm"、"身高 170"、"我 22 岁"）。不推测、不预填。

import re as _re


def _extract_opt_basic_info(messages: list) -> dict:
    """扫 HumanMessage，只抽取用户明确陈述的可选个人信息。低置信度项不填。"""
    text_parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            text_parts.append(c)
    joined = "\n".join(text_parts)
    out: dict[str, str] = {}
    # 身高：必须带 cm/厘米/米 单位或紧跟"身高"上下文，避免误匹配年龄/金额
    m = _re.search(r"(?:身高[:：\s]*)?(1[4-9]\d|20\d)\s*(?:cm|CM|厘米)", joined)
    if m:
        out["height"] = f"{m.group(1)}cm"
    else:
        m = _re.search(r"身高[:：\s]*(1\.[4-9]\d?|2\.0\d?)\s*米?", joined)
        if m:
            cm = int(round(float(m.group(1)) * 100))
            out["height"] = f"{cm}cm"
    # 年龄：必须带"岁"字，避免误匹配年份 / 数量
    m = _re.search(
        r"(?:我|本人)?\s*(?:今年|现在|)\s*(1[6-9]|[2-4]\d)\s*(?:岁|周岁)", joined
    )
    if m:
        out["age"] = f"{m.group(1)}岁"
    # 政治面貌
    for kw in ("中共党员", "党员", "共青团员", "团员", "群众"):
        if kw in joined:
            out["political"] = "中共党员" if kw in ("党员", "中共党员") else kw
            break
    # 民族（汉族默认不填，避免 LLM 生成简历时误触发"汉"字；只抽少数民族或明确陈述）
    m = _re.search(
        r"(汉族|满族|回族|蒙古族|维吾尔族|藏族|壮族|苗族|彝族|朝鲜族)", joined
    )
    if m:
        out["ethnicity"] = m.group(1)
    return out


def _serialize_to_resume_data(state: "ResumeState") -> dict:
    """state → ResumeData dict（供 /resume 接口返回给前端渲染）。"""
    collected = state.get("collected_info") or _empty_collected_info()
    messages = state.get("messages", [])
    jd_info = state.get("jd_info", {}) or {}
    jd_kw = state.get("jd_keywords") or []

    base_basic = collected.get("basic_info") or {}
    opt = _extract_opt_basic_info(messages)
    basic_info = {
        "name": base_basic.get("name", ""),
        "phone": base_basic.get("phone", ""),
        "email": base_basic.get("email", ""),
        "location": base_basic.get("location", ""),
        "objective": base_basic.get("objective", "")
        or (", ".join(jd_kw) if jd_kw else jd_info.get("target_role", "")),
        "gender": opt.get("gender", ""),
        "age": opt.get("age", ""),
        "birthday": "",
        "hometown": "",
        "political": opt.get("political", ""),
        "ethnicity": opt.get("ethnicity", ""),
        "height": opt.get("height", ""),
    }

    def _edu(e: dict) -> dict:
        return {
            "school": e.get("school", ""),
            "degree": e.get("degree", ""),
            "major": e.get("major", ""),
            "start_date": e.get("start_date", "") or e.get("duration", ""),
            "end_date": e.get("end_date", ""),
            "highlights": e.get("highlights", "") or e.get("description", ""),
        }

    def _work(w: dict) -> dict:
        desc = w.get("description", "")
        bullets = w.get("bullets") or (
            [line.strip("-• ") for line in desc.splitlines() if line.strip()]
            if desc
            else []
        )
        return {
            "company": w.get("company", ""),
            "title": w.get("title", "") or w.get("role", ""),
            "start_date": w.get("start_date", "") or w.get("duration", ""),
            "end_date": w.get("end_date", ""),
            "bullets": bullets,
        }

    def _proj(p: dict) -> dict:
        return {
            "name": p.get("name", ""),
            "role": p.get("role", ""),
            "start_date": p.get("start_date", "") or p.get("duration", ""),
            "end_date": p.get("end_date", ""),
            "description": p.get("description", ""),
        }

    projects = [_proj(p) for p in collected.get("projects", [])]
    work_experience = [_work(w) for w in collected.get("work_experience", [])] + [
        _work(i) for i in collected.get("internship", [])
    ]
    education = [_edu(e) for e in collected.get("education", [])]

    # 兜底：如果结构化抽取失败但对话里有 AI 包装好的 bullet，从消息历史里抽出来塞进 projects
    # 让前端至少能看到内容，不至于白板
    if not projects and not work_experience and not education:
        fallback = _harvest_projects_from_messages(messages)
        projects = fallback

    return {
        "basic_info": basic_info,
        "education": education,
        "work_experience": work_experience,
        "projects": projects,
        "skills": list(collected.get("skills", [])),
        "certificates": list(collected.get("certificates", [])),
        "self_evaluation": collected.get("personal_summary", ""),
    }


def _guess_project_name(text: str) -> str:
    """启发式从用户原话抽项目名（"做X时 / X项目 / 在X做 / 负责X"）。抽不到返回空串。"""
    m = _re.search(
        r"(?:做|在|参与|负责|搞)\s*([A-Za-z\u4e00-\u9fa5][\w\u4e00-\u9fa5\-]{1,30}?)(?:\s*项目|\s*这个|\s*时|\s*的|[，,。\s])",
        text,
    )
    return m.group(1) if m else ""


def _harvest_projects_from_messages(messages: list) -> list[dict]:
    """从对话里兜底抽取经历：
    - AI 按 WRAP prompt 产出的格式里有 "🧷 **试着这么写**：[bullet]"，抽 bullet
    - 用户原话里启发式抽项目名（"做X时/我在X项目/X 项目"）
    没有 AI bullet 时不返回，避免展示未包装的原文。
    """
    user_snippets: list[str] = []
    ai_bullets: list[str] = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if isinstance(m, HumanMessage):
            user_snippets.append(content)
        elif isinstance(m, AIMessage):
            # 匹配 "🧷 **试着这么写**：xxx" 后到下个空行/换行的内容
            for match in _re.finditer(
                r"🧷[^:：]*[:：]\s*\*?\*?([^*\n]+?)(?:\n\n|\n>|$)", content
            ):
                bullet = match.group(1).strip(" *")
                if bullet and len(bullet) > 10:
                    ai_bullets.append(bullet)
            # 备选格式：含 "STAR" 或大段 bullet 的 AI 回复，前 200 字
            if not ai_bullets and ("STAR" in content or "**" in content):
                first_para = content.split("\n\n", 1)[0].strip()
                if len(first_para) > 30:
                    ai_bullets.append(first_para[:240])

    if not ai_bullets:
        return []

    # 启发式项目名
    joined_user = "\n".join(user_snippets)
    project_name = _guess_project_name(joined_user) or "对话产出经历"
    description = "\n".join(f"• {b}" for b in ai_bullets[:3])
    return [
        {
            "name": project_name,
            "role": "",
            "start_date": "",
            "end_date": "",
            "description": description,
        }
    ]


def _has_minimum_info(collected: dict) -> bool:
    """至少有1条工作/实习/项目经历"""
    return bool(
        collected.get("work_experience")
        or collected.get("internship")
        or collected.get("projects")
    )


def _user_wants_to_proceed(text: str) -> bool:
    keywords = [
        "可以了",
        "开始",
        "生成简历",
        "开始生成",
        "差不多了",
        "没有了",
        "就这些",
        "好了",
        "下一步",
        "继续吧",
        "直接生成",
    ]
    return any(kw in text for kw in keywords)


def _collected_summary(collected: dict) -> str:
    parts = []
    if collected.get("education"):
        parts.append(f"教育经历 {len(collected['education'])} 条")
    if collected.get("work_experience"):
        parts.append(f"工作经历 {len(collected['work_experience'])} 条")
    if collected.get("internship"):
        parts.append(f"实习经历 {len(collected['internship'])} 条")
    if collected.get("projects"):
        parts.append(f"项目经历 {len(collected['projects'])} 条")
    if collected.get("skills"):
        parts.append(f"技能 {len(collected['skills'])} 项")
    if collected.get("personal_summary"):
        parts.append("自我评价 ✓")
    return "、".join(parts) if parts else "暂无"


async def _extract_info_from_message(
    message: str, current: dict, llm: ChatOpenAI
) -> dict:
    """用 LLM 从用户消息中提取结构化信息并 merge 进 current"""
    prompt = f"""从用户消息中提取简历信息，合并到已有JSON中。只提取消息明确说到的内容，不要捏造。没有的字段保持原值。

用户消息：
{message}

已有信息（JSON）：
{json.dumps(current, ensure_ascii=False, indent=2)}

直接返回合并后的JSON，格式与已有信息完全一致，不要任何解释。"""
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Info extraction failed: {e}")
        return current


# ── 阶段专属系统提示 ──────────────────────────────────────────────────────────

STAGE_PROMPTS = {
    "JD_INPUT": """你是「胡话简历」AI助手，当前阶段：收集目标JD。

目标：让用户把目标岗位的JD文字粘贴给你。

行为规范：
- 如果用户还没给JD，简短问一句：「请把目标岗位的招聘JD粘贴给我」
- 如果用户给了JD文字（超过50字的大段文字），回复「收到，正在解读...」然后停止，系统会自动切换到解读阶段
- 如果用户只说了岗位方向没给JD（如「产品经理」），告知：「我需要具体JD原文来帮你精准匹配，你能找到一个目标职位的招聘详情页，把文字粘过来吗？」
- 回复简短，一次只问一件事""",
    "COLLECTING": """你是「胡话简历」AI助手，当前阶段：收集用户经历（已废弃，请用 COLLECTING_* 子 prompt）。""",
    # ── Triage 分支后的子 prompt ───────────────────────────────────
    "COLLECTING_WRAP": """你是「胡话简历」的经历挖掘师，当前阶段：把用户刚说的经历**直接包装成可放进简历的 bullet**。

目标岗位要求：{jd_requirements}
用户刚说的经历片段：{experience_snippet}
当前已收集：{collected_summary}

行为规范：
- **第一步**：给一条 STAR-V 格式 bullet（30-60字），用户原话的核心动作+场景+结果+对应 JD 能力点
  - 如果用户没给数字，用定性描述（"显著提升""覆盖多类"），**绝对不要编造数字**
  - 如果确实需要量化，补一个带 `*` 标记的估算数字（例："约 30%*"）
- **第二步**：追问 1 个最关键的量化数字问题（"跑过多少条？发现几类问题？影响多少用户？"）让用户把 `*` 换成真数字
- **第三步**：最后问「这条能用吗？还有哪些项目也能这样挖？」

格式：
> 🧷 **试着这么写**：[bullet]
>
> 👉 **帮我核实一个数字**：[针对这条 bullet 的具体问题]
>
> 还有类似的经历吗？

**硬性禁令**：
- ❌ 不要输出代码、伪代码、README、CLI 命令、requirements.txt、分步实操指南
- ❌ 不要建议用户"做一个新项目"（那是另一个 agent 的事）
- ❌ 不要编造用户没说过的经历、公司、数字""",
    "COLLECTING_ASK_EXP": """你是「胡话简历」AI助手，当前阶段：追问经历。

目标岗位要求：{jd_requirements}
当前已收集：{collected_summary}

行为规范：
- 每次只问一件事
- 优先问 JD 里最重要的能力对应的经历（如 JD 要求 AB 测试就问"做过类似验证吗？"）
- 如果用户之前说过"我平时会 / 我习惯"这类软经历，点名追问（"你刚说会先跑体验看不合理——展开说说这个过程？"）
- 不要一次抛 3 个问题，用户会被淹没
- 已有 ≥1 条经历后，可以问"还想补充点什么？或者直接开始生成？"

**硬性禁令**：
- ❌ 不要推荐"做一个新项目"
- ❌ 不要输出代码、分步指南""",
    "COLLECTING_SUGGEST": """你是「胡话简历」AI助手，当前阶段：兜底推荐方向。

目标岗位要求：{jd_requirements}
已尝试挖掘用户经历但无果（dig_attempts={dig_attempts}）。

行为规范：
- 推荐一个可行方向填补 JD 缺口
- **只给**：方向名称 + 1 句话描述 + 对应 JD 哪条要求
- 格式：
  > 💡 **补缺方向建议**：[方向名]
  > [1 句话描述，不超过 50 字]
  > 对应 JD 里的：[具体要求原话或关键词]
  >
  > 做完了回来告诉我你实际的经历，我帮你包装成 bullet。

**硬性禁令（违反即严重错误）**：
- ❌ 不输出任何代码、伪代码、README、requirements、CLI 命令、安装命令
- ❌ 不给分步实操指南（"第一步…第二步…"）
- ❌ 不超过 3 行文字，只给方向不给实施细节
- ❌ 不写 GitHub 模板、不写 markdown 模板文件内容""",
    "COLLECTING_CLARIFY": """你是「胡话简历」AI助手，当前阶段：澄清用户意图。

用户回复太模糊，需要二选一问清楚：

直接问：
> 想确认一下——你是：
> 1️⃣ 暂时没有 JD 原文（我可以等你找到再来）
> 2️⃣ 就想跳过 JD 做一份通用简历（我直接问你经历）
>
> 选哪个？

不要啰嗦，等用户答。""",
    "LIGHT_MODE_ASK": """你是「胡话简历」AI助手，当前阶段：轻量模式——问用户心仪岗位方向。

用户已选择跳过 JD。现在需要他用一句话描述心仪的岗位，让我们知道往哪个方向包装。

直接问：
> 好，不用 JD 也能做。告诉我你心仪的岗位方向就行，比如：
> - "字节 AI 产品经理实习"
> - "想投后端工程师"
> - "校招运营岗"
>
> 一句话就够，我帮你定位关键能力。

不要问多余问题，等用户一句话回复。""",
    "SKIP_JD_OK": """你是「胡话简历」AI助手，当前阶段：确认跳过 JD。

用户已选择不提供 JD 原文，进入通用模式。

简短回复：
> 好，不用 JD 我也能帮你做。先告诉我你心仪的岗位方向（一句话就行），然后我们开始聊你的经历。

然后停止。""",
    "BUILDING": """你是「胡话简历」AI助手，当前阶段：包装简历。

你已收集了用户的经历信息。现在要把它们包装成高质量的简历内容。

包装规范（必须严格遵守）：
- 只包装用户实际告知的经历，绝对不得凭空添加用户未提及的项目
- 每个项目最多3条bullet（核心动作 + 量化结果 + 复用价值）
- 最相关的项目3条，次相关2条，只有1条则1条，有几条写几条
- 自我评价不超过3行
- 整体控制在1页以内
- 语言：STAR-V格式，动词开头，至少一个量化指标

严禁捏造规则（违反即为严重错误）：
- 姓名、联系方式等基本信息如用户未提供，一律用[待填写]占位，绝对不能编名字
- 所有数字和量化指标必须来自用户原话，不得自行推算或虚构
- 不得添加用户未提及的经历、项目、公司、技能

展示格式：
直接输出简历内容，之后问用户「这个版本你觉得怎么样？要调整什么吗？」""",
    "EVALUATING": """你是「胡话简历」严格评估官，当前阶段：简历评分。

你已经有了简历草稿和目标JD。现在用ATS+HR双维度评分。

评分展示格式：
📊 ATS得分：XX/60 | HR得分：XX/40 | 总分：XX/100
📈 面试通过率预测：XX%

✅ 亮点（2条）
⚠️ 需改进（2-3条，按优先级排序）

之后问用户：「要继续优化还是直接导出？」""",
    "EXPORT": """你是「胡话简历」AI助手，当前阶段：导出简历。

简历已完成。询问用户导出格式：
「你的简历已准备好，选择导出格式：
  1. PDF（推荐，投递用）
  2. Word（还需要自己编辑）」

根据用户选择触发对应导出流程。""",
}


# ── 节点实现 ──────────────────────────────────────────────────────────────────


def _jd_requirements_summary(state: ResumeState) -> str:
    """返回可塞进 prompt 的 JD 关键要求摘要（有 JD 时用 must_highlight，轻量模式用关键词，都没有时兜底）。"""
    if state.get("jd_info"):
        reqs = state["jd_info"].get("resume_strategy", {}).get("must_highlight", [])
        if reqs:
            return "、".join(reqs)
        return "见已解读JD"
    if state.get("jd_keywords"):
        return "（轻量模式）岗位关键词：" + "、".join(state["jd_keywords"])
    return "（暂无 JD，走通用简历规则）"


def _get_stage_system(
    stage: str, state: ResumeState, experience_snippet: str = ""
) -> str:
    """按 stage (或 action key) 返回系统 prompt 文本，已完成模板变量替换。"""
    prompt = STAGE_PROMPTS.get(stage, "")
    collected = state.get("collected_info") or _empty_collected_info()
    fmt_vars = {
        "jd_requirements": _jd_requirements_summary(state),
        "collected_summary": _collected_summary(collected),
        "experience_snippet": experience_snippet or "",
        "dig_attempts": state.get("dig_attempts", 0),
    }
    # 只替换 prompt 里实际用到的占位符，避免 KeyError
    try:
        prompt = prompt.format(**fmt_vars)
    except (KeyError, IndexError):
        pass
    return prompt


def _has_jd_text(text: str) -> bool:
    return len(text.strip()) > 80


def _make_non_streaming_llm(
    llm: ChatOpenAI, temperature: float | None = None
) -> ChatOpenAI:
    """
    基于主 LLM 配置生成一个非流式实例，用于内部决策调用（triage / extract / decoder / editor）。
    如果 settings.LLM_STRUCTURED_MODEL 非空，会切到该模型（通常是结构化输出更快的型号），
    解决 qwen3.6-plus 等模型 with_structured_output 慢的问题。
    """
    from app.config import get_settings

    s = get_settings()
    structured_model = s.LLM_STRUCTURED_MODEL or llm.model_name
    return ChatOpenAI(
        model=structured_model,
        api_key=llm.openai_api_key,
        base_url=llm.openai_api_base,
        temperature=llm.temperature if temperature is None else temperature,
        streaming=False,
    )


async def _run_decoder_and_build_reply(
    last_human: str, llm: ChatOpenAI
) -> tuple[dict, str]:
    """把 JD 原文喂给 decoder_chain，返回 (jd_info, 用户可见的格式化回复)。"""
    decoder_llm = _make_non_streaming_llm(llm)
    try:
        result = await run_decoder_chain(last_human, decoder_llm)
        jd_info = result.get("jd_info", {})
        decoded = result.get("decoded_items", [])
        summary = result.get("summary", "")
        decoded_text = "\n".join(
            f"• **{i['keyword']}**：{i['real_meaning']}" for i in decoded[:5]
        )
        strategy = jd_info.get("resume_strategy", {})
        must = "、".join(strategy.get("must_highlight", []))
        reply = (
            f"**JD解读完毕** 🔍\n\n"
            f"{decoded_text}\n\n"
            f"**总结：** {summary}\n\n"
            f"**简历必须突出：** {must}\n\n"
            f"---\n现在告诉我你的经历。先说说你做过什么工作或项目？"
        )
        return jd_info, reply
    except Exception as e:
        logger.error(f"decoder error: {e}")
        return (
            {},
            "JD解读出了点问题，但没关系，你直接告诉我你的经历也行。做过什么工作或项目？",
        )


async def jd_input_node(state: ResumeState, llm: ChatOpenAI) -> dict:
    """阶段1：等待 JD / 跳过 JD / 轻量模式。由 Triage 决定走哪条分支。"""
    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    # 没用户输入就保持 ASK_JD
    if not last_human:
        sys_msg = SystemMessage(content=_get_stage_system("JD_INPUT", state))
        response = await llm.ainvoke([sys_msg] + messages)
        return {"messages": [response]}

    # 先跑 Triage
    triage_params = build_state_for_triage(state)
    triage_llm = _make_non_streaming_llm(llm, temperature=0)
    decision = await run_triage(
        user_message=last_human, llm=triage_llm, **triage_params
    )
    logger.info(f"[triage/jd_input] action={decision.action} reason={decision.reason}")

    action = decision.action

    # 分支 1：用户粘贴了 JD 原文 → 走解码
    if action == TriageAction.DECODE_JD:
        jd_info, reply = await _run_decoder_and_build_reply(last_human, llm)
        return {
            "messages": [AIMessage(content=reply)],
            "stage": "COLLECTING",
            "jd_text": last_human,
            "jd_info": jd_info,
        }

    # 分支 2：用户明确跳过 JD（可能顺便给了岗位关键词）
    if action == TriageAction.SKIP_JD_OK:
        update: dict[str, Any] = {"jd_skipped": True}
        if decision.jd_keywords:
            update["jd_keywords"] = decision.jd_keywords
            update["stage"] = "COLLECTING"
            reply = (
                f"好，收到。岗位方向：{'、'.join(decision.jd_keywords)}。\n\n"
                f"现在告诉我你的经历——做过什么工作、项目、或者平时有什么习惯跟这个方向相关的都行。"
            )
        else:
            # 跳过了但没给方向，下一轮 LIGHT_MODE_ASK
            sys_msg = SystemMessage(content=_get_stage_system("LIGHT_MODE_ASK", state))
            response = await llm.ainvoke([sys_msg] + messages)
            update["messages"] = [response]
            return update
        update["messages"] = [AIMessage(content=reply)]
        return update

    # 分支 3：用户回复太模糊 → 问清楚
    if action == TriageAction.ASK_CLARIFY:
        sys_msg = SystemMessage(content=_get_stage_system("COLLECTING_CLARIFY", state))
        response = await llm.ainvoke([sys_msg] + messages)
        return {"messages": [response]}

    # 分支 4：轻量模式下问岗位关键词
    if action == TriageAction.LIGHT_MODE_ASK:
        sys_msg = SystemMessage(content=_get_stage_system("LIGHT_MODE_ASK", state))
        response = await llm.ainvoke([sys_msg] + messages)
        return {"messages": [response]}

    # 默认：继续 ASK_JD
    sys_msg = SystemMessage(content=_get_stage_system("JD_INPUT", state))
    response = await llm.ainvoke([sys_msg] + messages)
    return {"messages": [response]}


async def collecting_node(state: ResumeState, llm: ChatOpenAI) -> dict:
    """阶段2：收集用户经历。由 Triage 决定走哪条分支（WRAP / ASK_EXP / SUGGEST / READY_TO_BUILD）。"""
    messages = state["messages"]
    collected = state.get("collected_info") or _empty_collected_info()

    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    # 提取结构化信息（沿用原有逻辑）
    if last_human and len(last_human) > 5:
        extract_llm = _make_non_streaming_llm(llm, temperature=0)
        collected = await _extract_info_from_message(last_human, collected, extract_llm)

    # 跑 Triage
    triage_params = build_state_for_triage(state)
    # 经历提取可能刚更新了 experience_count，用新值覆盖
    triage_params["experience_count"] = (
        len(collected.get("work_experience", []))
        + len(collected.get("internship", []))
        + len(collected.get("projects", []))
    )
    triage_llm = _make_non_streaming_llm(llm, temperature=0)
    decision = await run_triage(
        user_message=last_human, llm=triage_llm, **triage_params
    )
    logger.info(
        f"[triage/collecting] action={decision.action} "
        f"dig_delta={decision.dig_attempt_delta} reason={decision.reason}"
    )

    # 更新计数器
    new_dig_attempts = state.get("dig_attempts", 0) + decision.dig_attempt_delta
    user_asked = state.get("user_asked_for_suggestion", False)
    if decision.action == TriageAction.SUGGEST_PROJECT and _user_explicitly_asks(
        last_human
    ):
        user_asked = True

    base_update: dict[str, Any] = {
        "collected_info": collected,
        "dig_attempts": new_dig_attempts,
        "user_asked_for_suggestion": user_asked,
    }

    action = decision.action

    # 分支 A：开始生成简历
    # 不发硬编码过渡气泡——LangGraph 条件边会直接 chain 到 building_node，
    # 由 building_node 的 LLM 流式产出简历内容，SSE 接力用户就看到持续输出。
    if action == TriageAction.READY_TO_BUILD:
        has_min = _has_minimum_info(collected)
        logger.info(
            f"[collecting/ready_to_build] has_min={has_min} "
            f"projects={len(collected.get('projects', []))} "
            f"work={len(collected.get('work_experience', []))}"
        )
        if has_min:
            base_update["stage"] = "BUILDING"
            return base_update

    # 分支 B：挖掘重叙（核心新动作）
    if action == TriageAction.WRAP_EXPERIENCE:
        snippet = decision.experience_snippet or last_human
        sys_msg = SystemMessage(
            content=_get_stage_system(
                "COLLECTING_WRAP", state, experience_snippet=snippet
            )
        )
        response = await llm.ainvoke([sys_msg] + messages)

        # 规则化落地：直接把这条经历 append 到 collected.projects，不再依赖 _extract_info_from_message
        # 的 LLM 静默提取（它经常失败导致 projects 永远空、Triage 一直触发 ASK_EXPERIENCE）。
        project_name = _guess_project_name(snippet) or "对话产出经历"
        existing_names = {p.get("name") for p in collected.get("projects", [])}
        if project_name not in existing_names:
            collected.setdefault("projects", []).append(
                {
                    "name": project_name,
                    "role": "",
                    "start_date": "",
                    "end_date": "",
                    "description": snippet,
                }
            )
            base_update["collected_info"] = collected

        base_update["messages"] = [response]
        return base_update

    # 分支 C：追问经历
    if action == TriageAction.ASK_EXPERIENCE:
        sys_msg = SystemMessage(content=_get_stage_system("COLLECTING_ASK_EXP", state))
        response = await llm.ainvoke([sys_msg] + messages)
        base_update["messages"] = [response]
        return base_update

    # 分支 D：兜底推荐方向（硬禁令已写进 prompt）
    if action == TriageAction.SUGGEST_PROJECT:
        sys_msg = SystemMessage(content=_get_stage_system("COLLECTING_SUGGEST", state))
        response = await llm.ainvoke([sys_msg] + messages)
        base_update["messages"] = [response]
        return base_update

    # 分支 E：兜底—— triage 返回了 JD 相关 action（用户中途补贴 JD 等）
    if action == TriageAction.DECODE_JD:
        jd_info, reply = await _run_decoder_and_build_reply(last_human, llm)
        base_update["messages"] = [AIMessage(content=reply)]
        base_update["jd_text"] = last_human
        base_update["jd_info"] = jd_info
        return base_update

    # 兜底：按 ASK_EXP 走
    sys_msg = SystemMessage(content=_get_stage_system("COLLECTING_ASK_EXP", state))
    response = await llm.ainvoke([sys_msg] + messages)
    base_update["messages"] = [response]
    return base_update


def _user_explicitly_asks(text: str) -> bool:
    """识别用户主动求建议的关键词（跳过 dig_attempts 阈值的信号）。"""
    keywords = [
        "我该做什么",
        "我该怎么办",
        "没经验怎么写",
        "我没做过怎么办",
        "给我个方向",
        "推荐一下",
        "建议做什么",
    ]
    return any(kw in text for kw in keywords)


async def building_node(state: ResumeState, llm: ChatOpenAI) -> dict:
    """阶段3：包装简历草稿"""
    collected = state.get("collected_info") or _empty_collected_info()
    jd_info = state.get("jd_info", {})
    industry = jd_info.get("industry", "互联网")

    # 合并所有经历类型为待包装列表
    all_experiences = []
    for exp in collected.get("work_experience", []):
        desc = exp.get("description", "") or str(exp)
        if desc:
            all_experiences.append(
                f"【工作经历】{exp.get('company', '')} {exp.get('title', '')} {exp.get('duration', '')}\n{desc}"
            )
    for exp in collected.get("internship", []):
        desc = exp.get("description", "") or str(exp)
        if desc:
            all_experiences.append(
                f"【实习经历】{exp.get('company', '')} {exp.get('title', '')} {exp.get('duration', '')}\n{desc}"
            )
    for exp in collected.get("projects", []):
        desc = exp.get("description", "") or str(exp)
        if desc:
            all_experiences.append(
                f"【项目经历】{exp.get('name', '')} {exp.get('role', '')}\n{desc}"
            )

    # 每条经历过 wrapper_chain 包装（非流式）——走 LLM_STRUCTURED_MODEL，
    # 否则 qwen3.6-plus 的 with_structured_output 单次要 12s，3 条 bullet 就炸了
    logger.info("[building] 开始 wrap 经历")
    wrapper_llm = _make_non_streaming_llm(llm, temperature=llm.temperature)
    wrapped_bullets = []
    for exp_text in all_experiences[:3]:
        try:
            result = await run_wrapper_chain(
                experience=exp_text, industry=industry, llm=wrapper_llm
            )
            wrapped_bullets.append(
                result.get("wrapped", exp_text)
                if not result.get("needs_followup")
                else exp_text
            )
        except Exception as e:
            logger.error(f"wrapper error: {e}")
            wrapped_bullets.append(exp_text)

    # 补充技能和自我评价
    skills_text = "、".join(collected.get("skills", []))
    summary_text = collected.get("personal_summary", "")

    sys_msg = SystemMessage(content=_get_stage_system("BUILDING", state))
    build_prompt = (
        f"以下是用户提供的 {len(wrapped_bullets)} 条经历（只有这些，不得添加），请整合成完整简历：\n\n"
        + "\n\n".join(f"【经历{i + 1}】\n{b}" for i, b in enumerate(wrapped_bullets))
        + (f"\n\n【技能特长】{skills_text}" if skills_text else "")
        + (f"\n\n【自我评价】{summary_text}" if summary_text else "")
        + "\n\n注意：基本信息（姓名/联系方式）用户未提供，请用[待填写]占位。"
        + "\n保留bullet中原有的`*`标记（表示AI估算数字，非用户提供），不要删除它们。"
        + "\n简历末尾加一行：`📝 标*的数字为AI参考建议，请核实后替换为真实数据`"
    )
    response = await llm.ainvoke(
        [sys_msg] + state["messages"] + [HumanMessage(content=build_prompt)]
    )

    # 聊天产出 → ResumeData 粗稿（供 /resume 渲染）
    serialized = _serialize_to_resume_data(
        {
            **state,
            "collected_info": collected,
            "jd_info": jd_info,
        }
    )

    # 编辑 agent：技能分组 + description 压缩 + section 命名
    # 用非流式 LLM 走结构化输出；失败则原样返回 serialized
    from app.agents.editor_chain import run_editor

    editor_llm = _make_non_streaming_llm(llm, temperature=0)
    polished = await run_editor(serialized, editor_llm)
    logger.info(
        f"[building/return] resume_data projects={len(polished.get('projects', []))} "
        f"skill_groups={len(polished.get('skill_groups', {}))} "
        f"section={polished.get('experience_section_title')!r}"
    )

    return {
        "messages": [response],
        "stage": "EVALUATING",
        "resume_draft": response.content,
        "resume_data": polished,
    }


async def evaluating_node(state: ResumeState, llm: ChatOpenAI) -> dict:
    """阶段4：评分"""
    resume_draft = state.get("resume_draft", "")
    jd_text = state.get("jd_text", "")
    messages = state["messages"]

    # 检查用户是否选择继续修改
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    if any(w in last_human for w in ["导出", "好了", "可以", "满意", "不改"]):
        return {"messages": [AIMessage(content="好，准备导出！")], "stage": "EXPORT"}

    # 运行评分
    try:
        result = await run_evaluator_chain(resume_draft, llm, jd_text or None)
        ats = result.get("ats_score", {}).get("total", 0)
        hr = result.get("hr_score", {}).get("total", 0)
        total = result.get("total_score", 0)
        prediction = result.get("pass_prediction", {})
        roadmap = result.get("improvement_roadmap", [])[:3]

        improvements = "\n".join(
            f"{i + 1}. {r.get('action', '')}（预计+{r.get('expected_score_gain', '')}分）"
            for i, r in enumerate(roadmap)
        )
        strengths = result.get("competitive_analysis", {}).get("strengths", [])[:2]
        strengths_text = "\n".join(f"✅ {s}" for s in strengths)

        reply = (
            f"📊 **ATS得分：{ats}/60 | HR得分：{hr}/40 | 总分：{total:.0f}/100**\n"
            f"📈 面试概率：{prediction.get('interview_probability', '-')}\n\n"
            f"{strengths_text}\n\n"
            f"⚠️ **优先改进：**\n{improvements}\n\n"
            f"---\n要继续优化，还是直接导出？"
        )
        return {
            "messages": [AIMessage(content=reply)],
            "eval_result": result,
        }
    except Exception as e:
        logger.error(f"evaluator error: {e}")
        sys_msg = SystemMessage(content=_get_stage_system("EVALUATING", state))
        response = await llm.ainvoke([sys_msg] + messages)
        return {"messages": [response]}


async def export_node(state: ResumeState, llm: ChatOpenAI) -> dict:
    """阶段5：交付（给用户内容 + 模板库引导，不代填）"""
    reply = (
        "**你的简历内容已整理好 ✅**\n\n"
        "上方是完整的简历文字，你可以两种方式使用：\n\n"
        "**方式一：直接下载简约版 Word**\n"
        "`[下载简约版 Word]`（即用即投，不用再排版）\n\n"
        "**方式二：选个好看的模板自己填**\n"
        "`[浏览模板库]` → 挑一个喜欢的下载 → 把上面的内容复制进去\n"
        "（我们准备了 12 个精选模板，风格从简约到双栏都有）\n\n"
        "---\n"
        "💡 **我的角色：** 帮你想清楚简历里该写什么、怎么包装经历。"
        "模板是你自己的选择，简历最终是你的作品。"
    )
    return {"messages": [AIMessage(content=reply)]}


# ── 路由逻辑 ──────────────────────────────────────────────────────────────────


def route_by_stage(state: ResumeState) -> str:
    stage = state.get("stage", "JD_INPUT")
    return stage.lower()


# ── 图构建 ────────────────────────────────────────────────────────────────────


def build_graph(llm: ChatOpenAI) -> Any:
    """构建并编译 LangGraph 状态机"""
    from functools import partial

    graph = StateGraph(ResumeState)

    graph.add_node("jd_input", partial(jd_input_node, llm=llm))
    graph.add_node("collecting", partial(collecting_node, llm=llm))
    graph.add_node("building", partial(building_node, llm=llm))
    graph.add_node("evaluating", partial(evaluating_node, llm=llm))
    graph.add_node("export", partial(export_node, llm=llm))

    graph.set_conditional_entry_point(
        route_by_stage,
        {
            "jd_input": "jd_input",
            "collecting": "collecting",
            "building": "building",
            "evaluating": "evaluating",
            "export": "export",
        },
    )

    # 多数节点执行完就 END（等用户下条消息再决定下一步）；
    # 但 collecting → building 需要在**同一次**请求里自动 chain——
    # 用户说"开始生成/好了"时后端会把 stage 改成 BUILDING，此时 END 就没下文了，
    # 必须直接接 building 跑完让 SSE 流式产出简历文本。
    def _route_after_collecting(state: ResumeState) -> str:
        stage = state.get("stage")
        logger.info(f"[route/after_collecting] stage={stage}")
        return "building" if stage == "BUILDING" else "__end__"

    graph.add_conditional_edges(
        "collecting",
        _route_after_collecting,
        {"building": "building", "__end__": END},
    )
    for node in ["jd_input", "building", "evaluating", "export"]:
        graph.add_edge(node, END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── 对外接口 ──────────────────────────────────────────────────────────────────

_graph_cache: dict[str, Any] = {}


def get_graph(llm: ChatOpenAI) -> Any:
    key = f"{llm.model_name}"
    if key not in _graph_cache:
        _graph_cache[key] = build_graph(llm)
    return _graph_cache[key]


async def stream_agent(messages: list[dict], session_id: str, llm: ChatOpenAI):
    """流式运行 Agent，按 token yield 字符串"""
    graph = get_graph(llm)
    config = {"configurable": {"thread_id": session_id}}

    # 把最新用户消息转为 LangChain 格式
    last_msg = messages[-1] if messages else None
    if not last_msg:
        return

    lc_msg = HumanMessage(content=last_msg["content"])
    input_state = {"messages": [lc_msg]}

    full_response = ""
    try:
        async for event in graph.astream_events(
            input_state, config=config, version="v2"
        ):
            kind = event.get("event", "")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
    except Exception as e:
        logger.error(f"agent stream error: {e}", exc_info=True)
        yield f"\n\n[错误: {e}]"
        return

    # 从 checkpoint 读取节点最终存储的 AI 消息
    # - 无流式时（硬编码回复）：直接输出
    # - 有流式时（decoder 内部 LLM）：追加节点格式化的过渡文字（若与流式内容不同）
    try:
        state = await graph.aget_state(config)
        msgs = state.values.get("messages", []) if state else []
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if ai_msgs:
            last_content = ai_msgs[-1].content
            if not full_response:
                yield last_content
            elif last_content and last_content not in full_response:
                # 节点格式化回复与流式内容不同，追加输出
                yield "\n\n---\n" + last_content
    except Exception as e:
        logger.error(f"fallback state read error: {e}")
