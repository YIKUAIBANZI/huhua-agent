"""HTML 简历渲染服务：Jinja2 模板 + ResumeData → HTML 字符串。"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = ROOT / "data" / "jianlimoban" / "html-template"

# 当前上线的 6 款模板。其余模板文件仍在 templates/ 目录下但不对外暴露，便于随时恢复。
TEMPLATES = [
    {"id": "t001-jianyue", "name": "简约单栏", "style": "极简 / 标准单栏"},
    {"id": "t002-jianyue", "name": "深蓝横幅", "style": "商务单栏"},
    {"id": "t004-jianyue", "name": "灰条夹页", "style": "单栏 / 顶底装饰"},
    {"id": "t1000", "name": "浅蓝斜切", "style": "单栏 / 色块标题"},
    {"id": "t1001", "name": "浅蓝色块", "style": "单栏 / 变体"},
    {"id": "t1002-jianyue", "name": "淡青平行", "style": "单栏 / 平行四边形"},
]

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "htm", "j2"]),
)


def list_templates() -> list[dict[str, str]]:
    return TEMPLATES


def render_resume(template_id: str, data: dict[str, Any]) -> str:
    filename = f"{template_id}.html.j2"
    try:
        template = _env.get_template(filename)
    except TemplateNotFound:
        raise ValueError(f"模板不存在: {template_id}")
    return template.render(**data)
