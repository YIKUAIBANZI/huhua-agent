import copy
import json
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
