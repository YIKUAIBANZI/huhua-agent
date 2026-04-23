"""
将简历 markdown 文本转换为格式化的 .docx 文件。

解析规则（与 building_node 输出格式对应）：
  **粗体文本**   → Bold run
  # / ## / ### → Heading 1/2/3
  - 或 • 开头   → List bullet
  --- 分割线     → 段落分隔
  普通行         → Normal paragraph
"""

import io
import re
from docx import Document
from docx.shared import Pt, Inches


def _apply_inline_bold(paragraph, text: str):
    """将 **xxx** 解析为 bold run，其余为普通 run。"""
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part:
            paragraph.add_run(part)


def resume_to_docx(resume_text: str) -> bytes:
    """将简历文本转为 docx bytes，可直接作为 HTTP 响应体返回。"""
    doc = Document()

    # 全局页面边距
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # 默认正文字体
    style = doc.styles["Normal"]
    style.font.name = "微软雅黑"
    style.font.size = Pt(10.5)

    lines = resume_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # 跳过空行（但保留段落间距）
        if not line:
            i += 1
            continue

        # 水平分割线 → 段落间空行
        if line.strip() in ("---", "—--", "——"):
            doc.add_paragraph()
            i += 1
            continue

        # 标题级别
        if line.startswith("### "):
            p = doc.add_heading(line[4:], level=3)
            p.runs[0].font.size = Pt(11)
            i += 1
            continue
        if line.startswith("## "):
            p = doc.add_heading(line[3:], level=2)
            p.runs[0].font.size = Pt(12)
            i += 1
            continue
        if line.startswith("# "):
            p = doc.add_heading(line[2:], level=1)
            p.runs[0].font.size = Pt(14)
            i += 1
            continue

        # Bullet point（- 或 •）
        if line.startswith("- ") or line.startswith("• "):
            content = line[2:].strip()
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Inches(0.25)
            _apply_inline_bold(p, content)
            i += 1
            continue

        # 数字列表（1. 2. 等）
        if re.match(r"^\d+[\.。]\s", line):
            content = re.sub(r"^\d+[\.。]\s*", "", line)
            p = doc.add_paragraph(style="List Number")
            _apply_inline_bold(p, content)
            i += 1
            continue

        # 普通段落（含 **bold**）
        p = doc.add_paragraph()
        _apply_inline_bold(p, line)
        i += 1

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()
