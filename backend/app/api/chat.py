import json
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.agents.resume_graph import get_graph, stream_agent
from app.config import get_settings
from app.schemas.chat import ChatRequest
from app.services.file_extractor import (
    SUPPORTED_EXT,
    UnsupportedFileError,
    extract_text,
)
from app.tools.docx_exporter import resume_to_docx
from app.tools.docx_template_filler import TEMPLATE_DIR, list_templates

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.LLM_MODEL,
        api_key=s.LLM_API_KEY,
        base_url=s.LLM_BASE_URL,
        temperature=0.7,
        streaming=True,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    llm = _get_llm()
    messages = [m.model_dump() for m in req.messages]
    session_id = req.session_id or str(uuid.uuid4())

    async def generate():
        try:
            async for token in stream_agent(messages, session_id, llm):
                yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """接收前端上传的文件（PDF/DOCX/TXT/MD），返回提取后的文本内容。"""
    from pathlib import Path as _Path

    filename = file.filename or "upload"
    ext = _Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(
            status_code=415,
            detail=f"暂不支持此文件类型 {ext}，目前支持：PDF / DOCX / TXT / MD",
        )
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:  # 10MB 上限
        raise HTTPException(status_code=413, detail="文件超过 10MB 上限")
    try:
        text = extract_text(filename, raw)
    except UnsupportedFileError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        logger.error(f"file extract error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件解析失败：{e}")
    return {
        "filename": filename,
        "size": len(raw),
        "text": text,
        "text_length": len(text),
    }


@router.get("/templates")
async def list_available_templates():
    """列出所有可用模版（供前端画廊展示，用户自取）。"""
    return {"templates": list_templates()}


@router.get("/templates/{name}")
async def download_template(name: str):
    """下载原始模版文件（未填充，用户自己复制内容进去）。"""
    # 模糊匹配：'单页001' 匹配 '单页001-简约.docx'
    candidates = [t for t in list_templates() if t == name or t.startswith(name)]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"模版不存在: {name}")
    path = TEMPLATE_DIR / candidates[0]
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=candidates[0],
    )


@router.get("/export/docx")
async def export_docx(session_id: str = Query(...)):
    """导出简约版 Word（从零生成 resume_draft 的内容）。"""
    llm = _get_llm()
    graph = get_graph(llm)
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取会话失败: {e}")

    if not state:
        raise HTTPException(status_code=404, detail="会话不存在")

    resume_draft = state.values.get("resume_draft", "")
    if not resume_draft:
        msgs = state.values.get("messages", [])
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        resume_draft = ai_msgs[-1].content if ai_msgs else ""

    if not resume_draft:
        raise HTTPException(status_code=404, detail="简历草稿不存在，请先完成生成步骤")

    try:
        docx_bytes = resume_to_docx(resume_draft)
    except Exception as e:
        logger.error(f"docx gen error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成 DOCX 失败: {e}")

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="resume_{session_id[:8]}.docx"'
        },
    )
