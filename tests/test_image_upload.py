import asyncio
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["LLM_API_KEY"] = ""  # Override local credentials; image requests are mocked below.

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
from app.services.image_ocr import (  # noqa: E402
    ImageFormatError,
    _vision_model,
    image_media_type,
    recognize_jd_image,
)


def sample_image(image_format="JPEG"):
    output = io.BytesIO()
    Image.new("RGB", (30, 20), "white").save(output, format=image_format)
    return output.getvalue()


class ImageUploadTest(unittest.TestCase):
    def test_image_content_is_verified(self):
        jpeg = sample_image()
        png = sample_image("PNG")
        self.assertEqual(image_media_type("jd.JPG", jpeg), "image/jpeg")
        self.assertEqual(image_media_type("jd.png", png), "image/png")
        with self.assertRaises(ImageFormatError):
            image_media_type("jd.jpg", png)
        with self.assertRaises(ImageFormatError):
            image_media_type("jd.jpg", b"not an image")

    def test_vision_model_matches_configured_provider(self):
        self.assertEqual(_vision_model("https://api.deepseek.com", "deepseek", ""), "deepseek-flash")
        self.assertEqual(_vision_model("https://dashscope.aliyuncs.com/compatible-mode/v1", "openai", ""), "qwen-vl-plus")
        self.assertEqual(_vision_model(None, "openai", "custom-vision"), "custom-vision")

    def test_deepseek_image_request_returns_only_recognized_text(self):
        settings = SimpleNamespace(
            LLM_API_KEY="test-key",
            LLM_BASE_URL="https://api.deepseek.com",
            LLM_PROVIDER="deepseek",
            LLM_VISION_MODEL="",
        )
        requests = []

        def handle(request):
            requests.append(request)
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            return httpx.Response(200, json={"choices": [{"message": {"content": "岗位职责：开发 API"}}]})

        real_client = httpx.AsyncClient

        def fake_client(**kwargs):
            return real_client(transport=httpx.MockTransport(handle), **kwargs)

        with patch("app.services.image_ocr.get_settings", return_value=settings):
            with patch("app.services.image_ocr.httpx.AsyncClient", side_effect=fake_client):
                result = asyncio.run(recognize_jd_image("jd.jpg", sample_image()))
        self.assertEqual(result, "岗位职责：开发 API")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/chat/completions")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test-key")

    def test_upload_image_returns_editable_ocr_text(self):
        main._chat_requests.clear()
        main._all_paid_requests.clear()
        with patch("app.api.chat.recognize_jd_image", new_callable=AsyncMock) as recognize:
            recognize.return_value = "岗位职责：开发 API"
            with TestClient(main.app) as client:
                response = client.post(
                    "/api/chat/upload",
                    files={"file": ("jd.jpg", sample_image(), "image/jpeg")},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_type"], "image")
        self.assertEqual(response.json()["text"], "岗位职责：开发 API")
        recognize.assert_awaited_once()

    def test_invalid_image_and_missing_model_fail_clearly(self):
        main._chat_requests.clear()
        main._all_paid_requests.clear()
        with TestClient(main.app) as client:
            invalid = client.post(
                "/api/chat/upload",
                files={"file": ("jd.jpg", b"not a jpeg", "image/jpeg")},
            )
            missing_model = client.post(
                "/api/chat/upload",
                files={"file": ("jd.jpg", sample_image(), "image/jpeg")},
            )
        self.assertEqual(invalid.status_code, 415)
        self.assertEqual(missing_model.status_code, 503)
        self.assertIn("尚未配置模型", missing_model.json()["detail"])


if __name__ == "__main__":
    unittest.main()
