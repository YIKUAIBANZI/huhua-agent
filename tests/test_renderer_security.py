import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.renderer import list_templates, render_resume  # noqa: E402


class RendererSecurityTest(unittest.TestCase):
    def test_photo_must_be_local_image_data_and_text_is_escaped(self):
        data = {
            "basic_info": {
                "name": "<script>alert(1)</script>",
                "photo": 'x" onerror="alert(1)',
            },
            "education": [],
            "work_experience": [],
            "projects": [],
            "skills": [],
        }
        for template in list_templates():
            with self.subTest(template=template["id"]):
                html = render_resume(template["id"], data)
                self.assertNotIn("onerror=", html)
                self.assertNotIn("<script>alert(1)</script>", html)
                self.assertIn("&lt;script&gt;", html)

        data["basic_info"]["photo"] = "https://example.com/track.png"
        html = render_resume("t001-jianyue", data)
        self.assertNotIn("example.com", html)

        data["basic_info"]["photo"] = "data:image/png;base64,aGVsbG8="
        html = render_resume("t001-jianyue", data)
        self.assertIn("data:image/png;base64,aGVsbG8=", html)


if __name__ == "__main__":
    unittest.main()
