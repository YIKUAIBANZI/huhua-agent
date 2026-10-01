import copy
import json
import re
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.resume import router  # noqa: E402
from app.schemas.editor import ResumeData  # noqa: E402
from app.services.renderer import list_templates, render_resume  # noqa: E402


class FieldParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.fields = []
        self.current = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-field-path" in attrs:
            self.current = {"path": attrs["data-field-path"], "kind": attrs.get("data-field-kind"), "text": ""}

    def handle_data(self, text):
        if self.current is not None:
            self.current["text"] += text

    def handle_endtag(self, tag):
        if tag == "span" and self.current is not None:
            self.fields.append(self.current)
            self.current = None


def resolve(data, pointer):
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


class TextParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.text = ""
        self.feed(html)

    def handle_data(self, text):
        self.text += text


def section_titles(html):
    titles = [TextParser(match).text for match in re.findall(r'<div class="section-title">(.*?)</div>', html, re.S)]
    return [title for title in titles if title != "基本信息"]


class PreviewMappingTest(unittest.TestCase):
    def setUp(self):
        self.sample = ResumeData.model_validate(json.loads((ROOT / "data/sample_resume.json").read_text())).model_dump()

    def assert_field_values(self, data, template_id):
        fields = FieldParser(render_resume(template_id, data)).fields
        self.assertTrue(fields)
        for field in fields:
            with self.subTest(template=template_id, path=field["path"]):
                value = resolve(data, field["path"])
                if field["kind"] == "group-name":
                    self.assertIsInstance(value, list)
                    key = field["path"].split("/")[-1].replace("~1", "/").replace("~0", "~")
                    self.assertEqual(field["text"], key)
                else:
                    self.assertIsInstance(value, str)
                    if field["path"] == "/experience_section_title" and not value:
                        value = "项目经历"
                    self.assertEqual(field["text"], value)
        return fields

    def test_all_sample_fields_map_to_their_values(self):
        for template in list_templates():
            fields = self.assert_field_values(self.sample, template["id"])
            paths = [field["path"] for field in fields]
            self.assertIn("/projects/0/description", paths)
            self.assertIn("/work_experience/0/bullets/1", paths)
            self.assertIn("/skills/2", paths)
            self.assertIn("/self_evaluation", paths)
            self.assertIn("/certificates/1", paths)
            # Header degree/school aliases must refer to the same education row.
            self.assertGreaterEqual(paths.count("/education/0/degree"), 2)

    def test_multiple_items_and_escaped_group_keys(self):
        data = copy.deepcopy(self.sample)
        for collection in ("education", "work_experience", "projects"):
            for index in (1, 2):
                row = copy.deepcopy(data[collection][0])
                for key, value in row.items():
                    if isinstance(value, str):
                        row[key] = f"{collection}-{index}-{key}"
                    elif isinstance(value, list):
                        row[key] = [f"{collection}-{index}-{key}-{n}" for n in (0, 1)]
                data[collection].append(row)
        data["projects"][0]["polished_bullets"] = ["项目零第一条", "项目零第二条"]
        data["projects"][1]["polished_bullets"] = ["项目一第一条", "项目一第二条"]
        data["projects"][2]["polished_bullets"] = []
        data["experience_section_title"] = "过往经历"
        data["skill_groups"] = {"技术/~栈\"<script>": ["Python", "HTML <b>文本</b>"], "AI 工具": ["Jev", "Codex"]}
        data = ResumeData.model_validate(data).model_dump()
        for template in list_templates():
            fields = self.assert_field_values(data, template["id"])
            paths = {field["path"] for field in fields}
            self.assertIn("/education/2/school", paths)
            self.assertIn("/work_experience/2/bullets/1", paths)
            self.assertIn("/projects/1/polished_bullets/1", paths)
            self.assertIn("/projects/2/description", paths)
            self.assertIn('/skill_groups/技术~1~0栈"<script>/1', paths)
            self.assertNotIn("/skills/0", paths)

    def test_default_sections_and_legacy_project_title(self):
        for template in list_templates():
            data = copy.deepcopy(self.sample)
            data["experience_section_title"] = "过往经历"
            html = render_resume(template["id"], data)
            self.assertEqual(section_titles(html), ["教育背景", "实习经历", "过往经历", "技能证书", "证书", "自我评价"])
            self.assertNotIn('data-field-path="/sections/', html)
            self.assertIn('data-field-path="/experience_section_title"', html)

    def test_explicit_sections_reorder_rename_custom_and_preserve_field_paths(self):
        data = copy.deepcopy(self.sample)
        data["sections"] = [
            {"id": "custom-a", "kind": "custom", "title": "获奖与活动", "content": "第一行\n第二行 <b>纯文本</b>"},
            {"id": "projects", "kind": "projects", "title": "工程实践"},
            {"id": "certificates", "kind": "certificates", "title": "资格认证"},
            {"id": "skills", "kind": "skills", "title": "专业技能"},
            {"id": "education", "kind": "education", "title": "学习经历"},
            {"id": "work", "kind": "work_experience", "title": "工作经历"},
            {"id": "self", "kind": "self_evaluation", "title": "个人介绍"},
            {"id": "custom-b", "kind": "custom", "title": "志愿服务", "content": "参与社区活动"},
        ]
        data = ResumeData.model_validate(data).model_dump()
        expected = [section["title"] for section in data["sections"]]
        for template in list_templates():
            html = render_resume(template["id"], data)
            self.assertEqual(section_titles(html), expected)
            fields = self.assert_field_values(data, template["id"])
            paths = {field["path"] for field in fields}
            self.assertTrue({f"/sections/{index}/title" for index in range(8)} <= paths)
            self.assertIn("/sections/0/content", paths)
            self.assertIn("/sections/7/content", paths)
            self.assertNotIn("/experience_section_title", paths)
            self.assertIn("/projects/0/description", paths)
            self.assertIn("/certificates/0", paths)
            self.assertIn('white-space:pre-line;', html)
            self.assertIn("第二行 &lt;b&gt;纯文本&lt;/b&gt;", html)

    def test_deleted_sections_do_not_reappear_from_existing_body_data(self):
        data = copy.deepcopy(self.sample)
        data["sections"] = [{"id": "skills", "kind": "skills", "title": "仅显示技能"}]
        education_before = copy.deepcopy(data["education"])
        for template in list_templates():
            html = render_resume(template["id"], data)
            paths = {field["path"] for field in FieldParser(html).fields}
            self.assertEqual(section_titles(html), ["仅显示技能"])
            self.assertIn("/skills/0", paths)
            self.assertNotIn("/education/0/highlights", paths)
            self.assertFalse(any(path.startswith("/education/") for path in paths))
            self.assertFalse(any(path.startswith(("/projects/", "/work_experience/", "/certificates/")) for path in paths))
            self.assertNotIn("/self_evaluation", paths)
            self.assertTrue(data["projects"])
            self.assertEqual(data["education"], education_before)

    def test_explicit_empty_section_list_removes_all_body_sections(self):
        data = copy.deepcopy(self.sample)
        data["sections"] = []
        education_before = copy.deepcopy(data["education"])
        for template in list_templates():
            html = render_resume(template["id"], data)
            self.assertEqual(section_titles(html), [])
            paths = {field["path"] for field in FieldParser(html).fields}
            self.assertIn("/basic_info/name", paths)
            self.assertTrue(all(path.startswith("/basic_info/") for path in paths))
            self.assertEqual(data["education"], education_before)

    def test_empty_builtin_and_custom_visibility_and_pure_text(self):
        data = ResumeData.model_validate({
            "basic_info": {"name": "测试"},
            "sections": [
                {"id": "edu", "kind": "education", "title": "隐藏空教育"},
                {"id": "blank", "kind": "custom", "title": "", "content": ""},
                {"id": "content", "kind": "custom", "title": "", "content": "只有正文\n下一行"},
                {"id": "title", "kind": "custom", "title": "只有标题", "content": ""},
                {"id": "xss", "kind": "custom", "title": '<script>alert("title")</script>', "content": '<img src=x onerror="alert(1)">\n<script>alert("body")</script>'},
            ],
        }).model_dump()
        for template in list_templates():
            html = render_resume(template["id"], data)
            self.assertEqual(section_titles(html), ["只有标题", '<script>alert("title")</script>'])
            paths = {field["path"] for field in self.assert_field_values(data, template["id"])}
            self.assertIn("/sections/2/content", paths)
            self.assertIn("/sections/3/title", paths)
            self.assertNotIn("/sections/1/title", paths)
            self.assertNotIn("/sections/2/title", paths)
            self.assertNotIn("<script", html)
            self.assertNotIn("<img src=x", html)
            self.assertIn("&lt;img", html)

    def test_empty_skill_groups_hide_or_fall_back_to_flat_skills(self):
        data = ResumeData.model_validate({
            "basic_info": {"name": "测试"}, "skill_groups": {"空分组": []},
            "sections": [{"id": "skills", "kind": "skills", "title": "技能"}],
        }).model_dump()
        for template in list_templates():
            self.assertEqual(section_titles(render_resume(template["id"], data)), [])
            data["skills"] = ["仍有平铺技能"]
            html = render_resume(template["id"], data)
            self.assertEqual(section_titles(html), ["技能"])
            self.assertIn('data-field-path="/skills/0"', html)
            self.assertIn("仍有平铺技能", html)
            self.assertNotIn('data-field-path="/skill_groups/', html)
            data["skills"] = []

    def test_interactivity_requires_explicit_render_query(self):
        api = FastAPI()
        api.include_router(router, prefix="/api/resume")
        with TestClient(api) as client:
            for template in list_templates():
                for query, interactive in (("", False), ("?interactive=false", False), ("?interactive=true", True)):
                    response = client.post(f'/api/resume/render/{template["id"]}{query}', json=self.sample)
                    self.assertEqual(response.status_code, 200)
                    if interactive:
                        self.assertEqual(response.text.count('<script src="/static/resume-preview.js"></script>'), 1)
                    else:
                        self.assertNotIn("<script", response.text)
                        self.assertNotIn("resume-field-active", response.text)
            self.assertEqual(client.post("/api/resume/render/unknown?interactive=true", json=self.sample).status_code, 404)


if __name__ == "__main__":
    unittest.main()
