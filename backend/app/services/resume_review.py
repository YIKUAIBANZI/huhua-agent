"""Candidate-facing text and concrete pre-export checks for one ResumeData."""

import re


def resume_to_text(data: dict) -> str:
    basic = data.get("basic_info") or {}
    lines = [f"# {basic.get('name') or '[待填写姓名]'}"]
    contact = " · ".join(str(basic.get(k) or "") for k in ("phone", "email", "location"))
    if contact.strip(" ·"):
        lines.append(contact.strip(" ·"))
    if basic.get("objective"):
        lines.append(f"求职意向：{basic['objective']}")
    for heading, key in (("工作与实习经历", "work_experience"), ("项目经历", "projects"), ("教育经历", "education")):
        items = data.get(key) or []
        if not items:
            continue
        lines.append(f"\n## {heading}")
        for item in items:
            title = item.get("company") or item.get("name") or item.get("school") or "[待填写]"
            detail = item.get("title") or item.get("role") or item.get("degree") or ""
            lines.append(" · ".join(part for part in (title, detail) if part))
            bullets = item.get("bullets") or item.get("polished_bullets") or []
            if not bullets and item.get("description"):
                bullets = [item["description"]]
            for bullet in bullets:
                lines.append(f"- {bullet}")
    skills = data.get("skills") or []
    if skills:
        lines.append("\n## 技能\n" + "、".join(skills))
    if data.get("self_evaluation"):
        lines.append("\n## 自我评价\n" + data["self_evaluation"])
    return "\n".join(lines)


def review_resume(data: dict, jd_info: dict | None = None) -> list[str]:
    """Return checks, never a hiring probability or pretend ATS score."""
    jd_info = jd_info or {}
    basic = data.get("basic_info") or {}
    checks = []
    if not basic.get("name"):
        checks.append("补全姓名。")
    if not basic.get("phone") and not basic.get("email"):
        checks.append("补全至少一种联系方式。")
    if not (data.get("work_experience") or data.get("projects")):
        checks.append("补充至少一段能证明能力的工作或项目经历。")
    text = resume_to_text(data)
    if re.search(r"\d+(?:[.,]\d+)?%?\*", text):
        checks.append("核实并替换带 * 的估算数字。")
    required = [s for s in jd_info.get("required_skills", []) if isinstance(s, str)]
    missing = [s for s in required if s.casefold() not in text.casefold()]
    if missing:
        checks.append("JD 中这些技能在简历文字里尚未出现，请核对是否有真实经历可补：" + "、".join(missing[:5]) + "。")
    if not checks:
        checks.append("基本信息与经历已填写。投递前请逐条核对事实、日期和导出文件的排版。")
    return checks
