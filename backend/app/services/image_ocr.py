"""Read visible text from a JD screenshot with the configured vision model."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.services.file_extractor import MAX_CHARS

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
IMAGE_OCR_TIMEOUT_SECONDS = 45.0


class ImageFormatError(ValueError):
    pass


class ImageRecognitionError(RuntimeError):
    pass


class NoImageTextError(ImageRecognitionError):
    pass


def image_media_type(filename: str, raw: bytes) -> str:
    """Trust image bytes rather than the uploaded filename or MIME header."""
    ext = Path(filename).suffix.lower()
    expected = "JPEG" if ext in {".jpg", ".jpeg"} else "PNG" if ext == ".png" else None
    if expected is None:
        raise ImageFormatError("仅支持 JPG 或 PNG 图片")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != expected or image.width > 8192 or image.height > 8192:
                raise ImageFormatError("图片格式不符或尺寸超过 8192 像素")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageFormatError("图片无法读取，请上传有效的 JPG 或 PNG") from exc
    return "image/jpeg" if expected == "JPEG" else "image/png"


def _vision_model(base_url: str | None, provider: str, configured: str) -> str:
    if configured:
        return configured
    if provider.lower() == "deepseek" or "api.deepseek.com" in (base_url or ""):
        return "deepseek-flash"
    if "dashscope.aliyuncs.com" in (base_url or ""):
        return "qwen-vl-plus"
    return "gpt-4o-mini"


async def recognize_jd_image(filename: str, raw: bytes) -> str:
    import asyncio

    media_type = await asyncio.to_thread(image_media_type, filename, raw)
    settings = get_settings()
    if not settings.LLM_API_KEY:
        raise ImageRecognitionError("图片识别尚未配置模型")

    base_url = (settings.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")
    image_url = f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"
    payload = {
        "model": _vision_model(
            settings.LLM_BASE_URL, settings.LLM_PROVIDER, settings.LLM_VISION_MODEL
        ),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请只转写图片中可见的文字，保留原有段落和顺序。"
                            "不要总结、回答、补全或猜测看不清的内容；"
                            "看不清的片段标为[无法辨认]。如果图片没有文字，返回空内容。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=IMAGE_OCR_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ImageRecognitionError("图片识别暂时不可用，请稍后重试") from exc
    if not isinstance(content, str) or not content.strip():
        raise NoImageTextError("图片中未识别到文字，请换一张清晰图片")
    text = content.strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n\n...（图片文字已截断到 {MAX_CHARS} 字符）"
    return text
