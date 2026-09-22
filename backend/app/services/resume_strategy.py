"""Small, deterministic resume plan derived from the supplied job description.

The plan guides question order and section order. It must not infer experience
or qualifications that the candidate has not provided.
"""

from typing import Any


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_resume_strategy(
    jd_info: dict | None = None,
    jd_keywords: list[str] | None = None,
    has_jd: bool = False,
) -> dict:
    """Return only priorities supported by a JD or explicitly given role words."""
    jd_info = jd_info or {}
    jd_keywords = _strings(jd_keywords or [])
    decoded = jd_info.get("resume_strategy") or {}
    if not isinstance(decoded, dict):
        decoded = {}
    priorities = _unique(
        _strings(decoded.get("must_highlight"))
        + _strings(jd_info.get("required_skills"))
        + _strings(jd_info.get("key_responsibilities"))
        + jd_keywords
    )[:6]
    mode = "jd_match" if has_jd and jd_info else "capability"
    if not priorities:
        priorities = ["具体行动", "可核实的成果", "与目标岗位相关的能力"]
    return {
        "mode": mode,
        "industry": jd_info.get("industry") or "",
        "target_role": jd_info.get("job_title") or "、".join(jd_keywords),
        "recruiter_priorities": priorities,
        "section_order": ["基本信息", "工作与实习经历", "项目经历", "教育经历", "技能"],
    }


def format_strategy_for_prompt(strategy: dict | None) -> str:
    strategy = strategy or {}
    mode = "岗位匹配" if strategy.get("mode") == "jd_match" else "能力展示"
    role = strategy.get("target_role") or "未指定"
    priorities = "、".join(_strings(strategy.get("recruiter_priorities"))) or "具体行动与真实成果"
    return (
        f"模式：{mode}；目标岗位：{role}；优先呈现：{priorities}。"
        "只选用候选人真实提供的经历和数字，缺少证据时追问或留空。"
    )
