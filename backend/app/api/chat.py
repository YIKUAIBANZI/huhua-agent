import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.agents.resume_graph import _serialize_to_resume_data, get_graph, stream_agent
from app.config import get_settings
from app.schemas.chat import ChatRequest
from app.services.file_extractor import (
    FileLimitError,
    SUPPORTED_EXT,
    UnsupportedFileError,
    extract_text,
)
from app.services.session_store import (
    delete_resume_session,
    ensure_resume_session,
    load_resume_session,
    resume_data_has_content,
    session_can_read_checkpoint,
    session_is_blocked,
    session_lock,
)
from app.tools.docx_exporter import resume_to_docx
from app.tools.docx_template_filler import TEMPLATE_DIR, list_templates
from app.tools.resume_data_docx import resume_data_to_docx

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024
MAX_CHAT_BODY_BYTES = 128 * 1024
MAX_CHAT_MESSAGES = 30
MAX_CHAT_MESSAGE_CHARS = 10_000
MAX_CHAT_TOTAL_CHARS = 60_000
_extract_slots = asyncio.Semaphore(2)


async def _extract_with_timeout(filename: str, raw: bytes) -> str:
    try:
        await asyncio.wait_for(_extract_slots.acquire(), timeout=1)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="文件解析繁忙，请稍后重试")
    try:
        future = asyncio.get_running_loop().run_in_executor(
            None, extract_text, filename, raw
        )
    except Exception:
        _extract_slots.release()
        raise
    # Timed-out parsing cannot stop a running Python thread. Keep its slot
    # occupied until it actually exits so requests cannot pile up CPU work.
    future.add_done_callback(lambda _: _extract_slots.release())
    return await asyncio.wait_for(asyncio.shield(future), timeout=20)


async def _session_meta(session_id: str, llm: ChatOpenAI) -> dict:
    """Turn-end snapshot for the live resume canvas."""
    stage = ""
    has_data = False
    if not session_can_read_checkpoint(session_id):
        return {
            "type": "meta",
            "session_id": session_id,
            "stage": stage,
            "has_data": has_data,
        }
    try:
        graph = get_graph(llm)
        state = await graph.aget_state({"configurable": {"thread_id": session_id}})
        vals = (state.values if state else {}) or {}
        stage = vals.get("stage") or "JD_INPUT"
        rd = vals.get("resume_data") or {}
        if not resume_data_has_content(rd) and vals:
            rd = _serialize_to_resume_data(vals)
        has_data = resume_data_has_content(rd)
    except Exception as e:
        logger.warning("session meta graph read failed: %s", e)
        stored = load_resume_session(session_id)
        if stored:
            stage = stored.get("stage") or ""
            has_data = bool(stored.get("has_data"))
    return {
        "type": "meta",
        "session_id": session_id,
        "stage": stage,
        "has_data": has_data,
    }


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    if not s.LLM_API_KEY:
        raise HTTPException(status_code=503, detail="服务尚未配置模型，请稍后再试")
    return ChatOpenAI(
        model=s.LLM_MODEL,
        api_key=s.LLM_API_KEY,
        base_url=s.LLM_BASE_URL,
        temperature=0.7,
        streaming=True,
    )


@router.post("/stream")
async def chat_stream(request: Request):
    content_length = request.headers.get("content-length")
    if (
        content_length
        and content_length.isdigit()
        and int(content_length) > MAX_CHAT_BODY_BYTES
    ):
        raise HTTPException(status_code=413, detail="对话内容过长，请缩短后重试")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_CHAT_BODY_BYTES:
            raise HTTPException(status_code=413, detail="对话内容过长，请缩短后重试")
        body.extend(chunk)
    try:
        req = ChatRequest.model_validate_json(bytes(body))
    except ValidationError:
        raise HTTPException(status_code=422, detail="对话内容格式无效")
    if (
        not req.messages
        or len(req.messages) > MAX_CHAT_MESSAGES
        or any(len(m.content) > MAX_CHAT_MESSAGE_CHARS for m in req.messages)
        or sum(len(m.content) for m in req.messages) > MAX_CHAT_TOTAL_CHARS
        or req.messages[-1].role != "user"
        or (req.session_id is not None and len(req.session_id) > 128)
    ):
        raise HTTPException(status_code=413, detail="对话内容过长或格式无效，请精简后重试")
    llm = _get_llm()
    messages = [m.model_dump() for m in req.messages]
    session_id = req.session_id or str(uuid.uuid4())

    async def generate():
        async with session_lock(session_id):
            try:
                if session_is_blocked(session_id):
                    yield f"data: {json.dumps({'error': '会话已结束，请新建对话'}, ensure_ascii=False)}\n\n"
                else:
                    # A checkpoint with no current session row is an orphan
                    # from an old/deleted run. Never continue from its state.
                    if not session_can_read_checkpoint(session_id):
                        old_state = await get_graph(llm).aget_state(
                            {"configurable": {"thread_id": session_id}}
                        )
                        if old_state and old_state.values:
                            yield f"data: {json.dumps({'error': '会话已结束，请新建对话'}, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                    if not ensure_resume_session(session_id):
                        yield f"data: {json.dumps({'error': '会话已结束，请新建对话'}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for token in stream_agent(messages, session_id, llm):
                        yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps(await _session_meta(session_id, llm), ensure_ascii=False)}\n\n"
            except Exception:
                logger.exception("stream error")
                yield f"data: {json.dumps({'error': '生成失败，请稍后重试'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete persisted resume data and the conversation checkpoint."""
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is None or not hasattr(checkpointer, "adelete_thread"):
        raise HTTPException(status_code=503, detail="暂时无法删除会话，请稍后重试")
    async with session_lock(session_id):
        try:
            delete_resume_session(session_id)
            await checkpointer.adelete_thread(session_id)
        except Exception:
            logger.exception("session deletion failed")
            raise HTTPException(status_code=503, detail="暂时无法删除会话，请稍后重试")
    return {"deleted": True}


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
    chunks: list[bytes] = []
    size = 0
    try:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="文件超过 10MB 上限")
            chunks.append(chunk)
        raw = b"".join(chunks)
        text = await _extract_with_timeout(filename, raw)
    except UnsupportedFileError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except FileLimitError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except TimeoutError:
        logger.warning("file extraction timed out")
        raise HTTPException(status_code=422, detail="文件解析超时，请换一份更简短的文件")
    except HTTPException:
        raise
    except Exception:
        logger.exception("file extraction failed")
        raise HTTPException(status_code=422, detail="文件解析失败，请检查文件格式")
    finally:
        await file.close()
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
    async with session_lock(session_id):
        return await _export_docx_data(session_id)


async def _export_docx_data(session_id: str):
    if not session_can_read_checkpoint(session_id):
        raise HTTPException(status_code=404, detail="简历草稿不存在，请先完成生成步骤")
    llm = _get_llm()
    graph = get_graph(llm)
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as e:
        logger.warning(f"graph state read failed, trying persisted session: {e}")
        state = None

    resume_draft = ""
    if state and state.values:
        resume_draft = state.values.get("resume_draft", "")
        if not resume_draft:
            msgs = state.values.get("messages", [])
            ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
            resume_draft = ai_msgs[-1].content if ai_msgs else ""

    stored = load_resume_session(session_id)
    if not resume_draft and stored and stored.get("has_data"):
        try:
            docx_bytes = resume_data_to_docx(stored["resume_data"])
        except Exception:
            logger.exception("persisted docx generation failed")
            raise HTTPException(status_code=500, detail="生成 DOCX 失败，请稍后重试")
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="resume_{session_id[:8]}.docx"'
            },
        )

    if not resume_draft:
        raise HTTPException(status_code=404, detail="简历草稿不存在，请先完成生成步骤")

    try:
        docx_bytes = resume_to_docx(resume_draft)
    except Exception:
        logger.exception("docx generation failed")
        raise HTTPException(status_code=500, detail="生成 DOCX 失败，请稍后重试")

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="resume_{session_id[:8]}.docx"'
        },
    )
