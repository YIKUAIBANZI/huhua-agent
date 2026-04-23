"""
渲染 Demo：JSON 数据 + Jinja2 模板 → HTML（默认）或 PDF（--pdf）。
跑法：
    cd "/Users/yikuaibanz1/Desktop/huhua agent"

    # HTML 预览（浏览器打开）
    uv run --with jinja2 python backend/scripts/render_demo.py t001-jianyue.html.j2

    # PDF 输出（默认应用打开，A4 单页）
    uv run --with jinja2 --with playwright python backend/scripts/render_demo.py t001-jianyue.html.j2 --pdf
"""

import json
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "data" / "jianlimoban" / "html-template"
DATA_PATH = ROOT / "data" / "sample_resume.json"
HTML_OUT = Path("/tmp/resume_preview.html")
PDF_OUT = Path("/tmp/resume_preview.pdf")


def render_html(template_id: str, data_path: Path = DATA_PATH) -> str:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "htm", "j2"]),
    )
    return env.get_template(template_id).render(**data)


def html_to_pdf(html: str, out: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(
            path=str(out),
            format="A4",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        browser.close()


def main() -> None:
    args = sys.argv[1:]
    want_pdf = "--pdf" in args
    args = [a for a in args if a != "--pdf"]

    data_path = DATA_PATH
    remaining = []
    for a in args:
        if a.startswith("--data="):
            data_path = Path(a.split("=", 1)[1]).expanduser().resolve()
        else:
            remaining.append(a)
    args = remaining

    template_id = args[0] if args else "t001-jianyue.html.j2"

    html = render_html(template_id, data_path)

    if want_pdf:
        html_to_pdf(html, PDF_OUT)
        print(f"[ok] {template_id} → {PDF_OUT}")
        subprocess.run(["open", str(PDF_OUT)], check=False)
    else:
        HTML_OUT.write_text(html, encoding="utf-8")
        print(f"[ok] {template_id} → {HTML_OUT}")
        subprocess.run(["open", str(HTML_OUT)], check=False)


if __name__ == "__main__":
    main()
