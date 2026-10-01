"""Opt-in JD coverage check using a strictly allowlisted Jev payload.

Only capability labels from TAG_RULES and synthetic IDs leave this server. In
particular, the outbound request never contains resume/JD prose or contact and
organization fields. The original text remains available locally for review.
"""

from __future__ import annotations

import math
import re
from typing import Any

import httpx


JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_REQUIREMENTS = 12
MAX_TOTAL_REQUIREMENTS = 20
MAX_EVIDENCE = 24

# This finite vocabulary is the privacy boundary for outbound text. Add labels
# only if they are generic capabilities, never organization or personal names.
TAG_RULES: tuple[tuple[str, str], ...] = (
    ("Python", r"(?<![A-Za-z])python(?![A-Za-z])"),
    ("Java", r"(?<![A-Za-z])java(?![A-Za-z])"),
    ("JavaScript", r"(?<![A-Za-z])(?:javascript|typescript|node\.js|react|vue)(?![A-Za-z])"),
    ("Go", r"(?<![A-Za-z])golang(?![A-Za-z])"),
    ("C++", r"(?<![A-Za-z])c\+\+(?![A-Za-z])"),
    ("SQL", r"(?<![A-Za-z])(?:sql|mysql|postgresql|sqlite)(?![A-Za-z])"),
    ("数据库", r"数据库|数据建模|(?<![A-Za-z])database(?![A-Za-z])"),
    ("数据分析", r"数据分析|数据洞察|商业分析|(?<![A-Za-z])data analys(?:is|t)(?![A-Za-z])"),
    ("数据清洗", r"数据清洗|数据处理|数据预处理|(?<![A-Za-z])pandas(?![A-Za-z])"),
    ("数据可视化", r"数据可视化|图表分析|(?<![A-Za-z])dashboard(?![A-Za-z])"),
    ("机器学习", r"机器学习|模型训练|(?<![A-Za-z])machine learning(?![A-Za-z])"),
    ("深度学习", r"深度学习|神经网络|(?<![A-Za-z])deep learning(?![A-Za-z])"),
    ("大模型应用", r"大模型|提示词|智能体|(?<![A-Za-z])(?:llm|agent|rag|prompt)(?![A-Za-z])"),
    ("自然语言处理", r"自然语言处理|(?<![A-Za-z])nlp(?![A-Za-z])"),
    ("计算机视觉", r"计算机视觉|图像识别|(?<![A-Za-z])cv(?![A-Za-z])"),
    ("接口开发", r"接口开发|接口设计|后端开发|(?<![A-Za-z])(?:api|fastapi|rest)(?![A-Za-z])"),
    ("前端开发", r"前端开发|网页开发|(?<![A-Za-z])frontend(?![A-Za-z])"),
    ("系统设计", r"系统设计|架构设计|高并发|分布式"),
    ("云服务", r"云服务|云平台|云计算|容器部署|(?<![A-Za-z])(?:aws|kubernetes|docker)(?![A-Za-z])"),
    ("自动化", r"自动化|脚本开发|流程优化|(?<![A-Za-z])automation(?![A-Za-z])"),
    ("测试", r"单元测试|自动化测试|质量保障|测试用例|(?<![A-Za-z])testing(?![A-Za-z])"),
    ("产品设计", r"产品设计|产品规划|需求分析|产品经理|原型设计"),
    ("用户研究", r"用户研究|用户访谈|用户反馈|可用性测试"),
    ("项目管理", r"项目管理|进度管理|跨团队协作|项目协调"),
    ("运营", r"内容运营|用户运营|活动运营|增长运营|运营策略"),
    ("市场分析", r"市场分析|竞品分析|行业研究|市场调研"),
    ("沟通协作", r"沟通协作|团队协作|跨部门沟通|表达能力"),
    ("英语", r"英语|英文|(?<![A-Za-z])english(?![A-Za-z])"),
    ("本科", r"本科|学士|(?<![A-Za-z])bachelor(?![A-Za-z])"),
    ("硕士", r"硕士|研究生|(?<![A-Za-z])master(?:'s)?(?![A-Za-z])"),
    ("博士", r"博士|(?<![A-Za-z])phd(?![A-Za-z])"),
)
_COMPILED_RULES = tuple((label, re.compile(pattern, re.IGNORECASE)) for label, pattern in TAG_RULES)

# Adjacent capabilities can support a weak suggestion but never full coverage.
RELATED_TAGS: dict[str, set[str]] = {
    "数据分析": {"数据清洗", "数据可视化", "SQL", "Python"},
    "数据清洗": {"数据分析", "Python", "SQL"},
    "数据库": {"SQL", "数据分析"},
    "机器学习": {"深度学习", "大模型应用", "Python"},
    "大模型应用": {"机器学习", "自然语言处理", "Python"},
    "接口开发": {"Python", "Java", "JavaScript", "系统设计"},
    "自动化": {"Python", "测试", "项目管理"},
    "产品设计": {"用户研究", "市场分析", "项目管理"},
    "项目管理": {"沟通协作", "产品设计"},
    "运营": {"市场分析", "数据分析", "沟通协作"},
}


def capability_tags(value: Any) -> list[str]:
    """Reduce untrusted free text to a fixed, non-identifying vocabulary."""
    if not isinstance(value, str):
        return []
    text = value[:4000]
    return [label for label, pattern in _COMPILED_RULES if pattern.search(text)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _jd_segments(jd_text: str) -> list[str]:
    # Segmentation is local. The original JD and these segments never enter the
    # outbound state; they only verify the decoder's proposed requirements.
    parts = re.split(r"[\n。；;，,]+", jd_text[:12000])
    return [
        re.sub(r"^\s*(?:[-•*]|\d+[.、)）])\s*", "", part).strip()
        for part in parts if len(part.strip()) >= 3
    ][:100]


def build_match_context(resume_data: dict, jd_info: dict, jd_text: str = "") -> dict:
    """Build local display data and a separate privacy-safe outbound state."""
    requirements: list[dict] = []
    seen: set[str] = set()
    original_segments = _jd_segments(jd_text) if jd_text else []
    categories = (
        ("hard_requirements", "硬性要求", 2),
        ("required_skills", "必备技能", 2),
        ("key_responsibilities", "岗位职责", 1),
        ("preferred_qualities", "加分项", 1),
    )
    for field, category, weight in categories:
        for text in _strings(jd_info.get(field)):
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            proposed_tags = capability_tags(text)
            original_source = next(
                (
                    segment for segment in original_segments
                    if proposed_tags and set(proposed_tags).issubset(capability_tags(segment))
                ),
                None,
            )
            requirements.append({
                "id": f"r{len(requirements) + 1}",
                # Display the source wording when present; otherwise make the
                # decoder's unverified proposal visibly unscored.
                "text": (original_source or f"未在 JD 原文定位：{text}")[:180],
                "category": category,
                "weight": (
                    1 if not original_source or re.search(r"加分|优先|可选", original_source)
                    else weight
                ),
                "tags": proposed_tags if original_source else [],
            })

    verified_sources = {
        row["text"] for row in requirements if not row["text"].startswith("未在 JD 原文定位：")
    }
    for segment in original_segments:
        if segment[:180] in verified_sources:
            continue
        tags = capability_tags(segment)
        if not tags and not re.search(r"熟悉|掌握|经验|能力|负责|职责|要求|学历|专业", segment):
            continue
        requirements.append({
            "id": f"r{len(requirements) + 1}",
            "text": segment[:180],
            "category": "JD 原文",
            "weight": 1,
            "tags": tags,
        })
        if len(requirements) >= MAX_TOTAL_REQUIREMENTS:
            break

    evidence: list[dict] = []

    def add_evidence(kind: str, label: str, raw: Any) -> None:
        tags = capability_tags(raw)
        if tags and len(evidence) < MAX_EVIDENCE:
            evidence.append({
                "id": f"e{len(evidence) + 1}",
                "kind": kind,
                "label": label,
                "tags": tags,
            })

    # A bullet is one source. Its original text and the organization heading
    # are deliberately omitted even from this local context.
    for section, title in (("work_experience", "工作经历"), ("projects", "项目经历")):
        for index, item in enumerate(resume_data.get(section) or [], start=1):
            if not isinstance(item, dict):
                continue
            bullets = _strings(item.get("bullets") or item.get("polished_bullets"))
            if not bullets and isinstance(item.get("description"), str):
                bullets = [item["description"]]
            for bullet_index, bullet in enumerate(bullets, start=1):
                add_evidence("experience", f"{title} {index} · 要点 {bullet_index}", bullet)

    for index, item in enumerate(resume_data.get("education") or [], start=1):
        if isinstance(item, dict):
            degree = item.get("degree") or ""
            major = item.get("major") or ""
            add_evidence("education", f"教育经历 {index}", f"{degree} {major}")

    skills = _strings(resume_data.get("skills"))
    skill_groups = resume_data.get("skill_groups") or {}
    if isinstance(skill_groups, dict):
        for values in skill_groups.values():
            skills.extend(_strings(values))
    for index, skill in enumerate(skills, start=1):
        add_evidence("skill", f"技能 {index}", skill)

    # There is intentionally no pass-through from user strings into this value.
    outbound_state = {
        "requirements": [
            {"id": row["id"], "category": row["category"], "tags": row["tags"]}
            for row in requirements[:MAX_REQUIREMENTS] if row["tags"]
        ],
        "evidence": [
            {"id": row["id"], "kind": row["kind"], "tags": row["tags"]}
            for row in evidence
        ],
    }
    return {"requirements": requirements, "evidence": evidence, "outbound_state": outbound_state}


def build_jev_payload(context: dict) -> dict:
    state = context["outbound_state"]
    choices = {"none": "没有支持这项要求的简历来源"}
    choices.update({
        row["id"]: f"{row['kind']}：{'、'.join(row['tags'])}"
        for row in state["evidence"]
    })
    questions: dict[str, dict] = {}
    for requirement in state["requirements"]:
        rid = requirement["id"]
        questions[f"{rid}_strength"] = {
            "type": "score",
            "instructions": f"根据 `requirements` 中 id={rid} 的能力标签和 `evidence`，判断简历对该要求的覆盖强度。只看当前提供的标签。",
            "criteria": [
                "没有对应的能力标签或来源。",
                "只有技能列表、学历或相邻能力标签，尚无直接任务证据。",
                "工作或项目条目有直接相关的能力和任务标签。",
            ],
        }
        questions[f"{rid}_source"] = {
            "type": "choice",
            "instructions": f"为 `requirements` 中 id={rid} 选择最能支持它的一个 `evidence.id`；无支持则选 none。",
            "criteria": choices,
        }
    return {"state": state, "model": "jev-latest", "questions": questions}


async def call_jev(payload: dict, api_key: str, timeout_seconds: float) -> dict:
    """Send one batched request. The caller handles all failures without a score."""
    timeout = min(20.0, max(2.0, timeout_seconds))
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            JEV_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise ValueError("invalid Jev response")
    return data


def score_jev_response(context: dict, response: dict) -> dict:
    """Reject invented sources and combine bounded judgements in local code."""
    answers = response.get("answers") or {}
    evidence_by_id = {row["id"]: row for row in context["evidence"]}
    scored: list[dict] = []
    total_weight = 0
    covered_weight = 0
    for requirement in context["requirements"]:
        total_weight += requirement["weight"]
        row = {key: requirement[key] for key in ("id", "text", "category", "weight")}
        if not requirement["tags"] or len(scored) >= MAX_REQUIREMENTS:
            row.update(level="unscored", evidence_id=None, evidence_text=None)
            scored.append(row)
            continue
        rid = requirement["id"]
        strength = answers.get(f"{rid}_strength")
        source = answers.get(f"{rid}_source")
        if not isinstance(strength, dict) or not isinstance(source, dict):
            raise ValueError("missing Jev answer")
        raw_score = strength.get("score")
        chosen_id = source.get("choice")
        if (
            strength.get("type") != "score"
            or source.get("type") != "choice"
            or not isinstance(raw_score, (int, float))
            or isinstance(raw_score, bool)
            or not math.isfinite(raw_score)
            or not 0 <= raw_score <= 2
            or not isinstance(chosen_id, str)
            or chosen_id not in ("none", *evidence_by_id)
        ):
            raise ValueError("invalid Jev answer")
        evidence = evidence_by_id.get(chosen_id)
        level = "missing"
        points = 0
        requirement_tags = set(requirement["tags"])
        evidence_tags = set(evidence["tags"]) if evidence else set()
        exact_support = bool(requirement_tags & evidence_tags)
        adjacent_support = any(
            evidence_tags & RELATED_TAGS.get(tag, set()) for tag in requirement_tags
        )
        if evidence is not None and raw_score >= 0.75 and (exact_support or adjacent_support):
            level, points = "weak", 50
            if raw_score >= 1.5 and exact_support and evidence["kind"] == "experience":
                level, points = "supported", 100
        row.update(
            level=level,
            evidence_id=evidence["id"] if points else None,
            evidence_text=(
                f"{evidence['label']}：{'、'.join(evidence['tags'])}"
                if evidence is not None and points else None
            ),
        )
        scored.append(row)
        covered_weight += requirement["weight"] * points

    recognized = sum(row["level"] != "unscored" for row in scored)
    total = len(scored)
    enough_scope = recognized >= 2 and recognized * 5 >= total * 3
    return {
        "status": "ready" if enough_scope else "insufficient",
        "score": round(covered_weight / total_weight) if total_weight and enough_scope else None,
        "requirements": scored,
        "message": (
            f"参考覆盖分仅基于已识别的 {recognized}/{total} 项 JD 要求；"
            "未识别项按 0 留在分母，分数是保守覆盖下界，并不代表能力不足。"
            "Jev 只接收能力标签；结果需结合原简历核对。"
            if enough_scope else
            f"目前只识别到 {recognized}/{total} 项 JD 要求，无法给出有代表性的参考覆盖分。"
        ),
    }
