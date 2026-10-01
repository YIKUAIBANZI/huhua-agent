"""Shared section order and field paths for preview and document export."""

from typing import Any


DEFAULT_SECTIONS = (
    ("education", "教育背景"),
    ("work_experience", "实习经历"),
    ("projects", "项目经历"),
    ("skills", "技能证书"),
    ("certificates", "证书"),
    ("self_evaluation", "自我评价"),
)


def get_resume_sections(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep explicit metadata intact; missing/None metadata uses legacy defaults.

    Builtin bodies stay in their original fields. Removing a section therefore
    changes visibility without losing the user's source content.
    """
    sections = data.get("sections")
    if sections is not None:
        return [
            {
                "id": section["id"],
                "kind": section["kind"],
                "title": section["title"],
                "content": section.get("content", ""),
                "title_path": f"/sections/{index}/title",
                "content_path": f"/sections/{index}/content" if section["kind"] == "custom" else None,
            }
            for index, section in enumerate(sections)
        ]
    return [
        {
            "id": kind,
            "kind": kind,
            "title": (data.get("experience_section_title") or title) if kind == "projects" else title,
            "content": "",
            "title_path": "/experience_section_title" if kind == "projects" else None,
            "content_path": None,
        }
        for kind, title in DEFAULT_SECTIONS
    ]
