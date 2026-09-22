"""HTML 简历渲染 API：列模板 / 渲染 / 取样例数据 / 取当前 session 数据 / 导出 DOCX。"""

import io
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_openai import ChatOpenAI

from app.agents.resume_graph import _serialize_to_resume_data, get_graph
from app.config import get_settings
from app.schemas.editor import ResumeData
from app.services.renderer import list_templates, render_resume
from app.services.session_store import load_resume_session, resume_data_has_content
from app.tools.resume_data_docx import resume_data_to_docx

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
# 模板预览用中性占位数据（不含任何真实用户信息）
SAMPLE_PATH = ROOT / "data" / "sample_resume.json"


def _llm_for_graph() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.LLM_MODEL,
        api_key=s.LLM_API_KEY,
        base_url=s.LLM_BASE_URL,
        temperature=0.7,
        streaming=True,
    )


@router.get("/templates")
async def templates():
    return {"templates": list_templates()}


@router.get("/sample")
async def sample():
    if not SAMPLE_PATH.exists():
        raise HTTPException(status_code=404, detail="样例数据不存在")
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


@router.get("/current")
async def current_resume(session_id: str = Query(..., description="聊天 session_id")):
    """读聊天 session 当前产出的 ResumeData，内存没有时回退到 SQLite 持久化数据。"""
    config = {"configurable": {"thread_id": session_id}}
    rd: dict = {}
    stage = ""
    stored = load_resume_session(session_id)
    if get_settings().LLM_API_KEY:
        try:
            graph = get_graph(_llm_for_graph())
            state = await graph.aget_state(config)
        except Exception as e:
            state = None
            graph_error = e
        else:
            graph_error = None
    else:
        state = None
        graph_error = None

    if state and state.values:
        vals = state.values
        stage = vals.get("stage") or "JD_INPUT"
        # 优先用 BUILDING 阶段序列化好的 resume_data；否则实时 serialize 一次
        rd = vals.get("resume_data") or {}
        if not resume_data_has_content(rd):
            rd = _serialize_to_resume_data(vals)

    if not resume_data_has_content(rd) and stored:
        rd = stored["resume_data"]
    if not stage and stored:
        stage = stored.get("stage") or ""

    if not resume_data_has_content(rd):
        if graph_error and not stored:
            raise HTTPException(
                status_code=500, detail=f"读取 session 失败: {graph_error}"
            )
        return {"has_data": False, "data": None, "stage": stage}
    return {"has_data": True, "data": rd, "stage": stage}


@router.post("/render/{template_id}", response_class=HTMLResponse)
async def render(template_id: str, data: ResumeData):
    try:
        html = render_resume(template_id, data.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return HTMLResponse(content=html)


@router.post("/export/docx/{template_id}")
async def export_docx(template_id: str, data: ResumeData):
    """把 ResumeData 导出为通用 Word 文件。template_id 仅用于文件名。"""
    payload = data.model_dump()
    docx_bytes = resume_data_to_docx(payload)
    name = (payload.get("basic_info") or {}).get("name") or "简历"
    filename = f"{name}-{template_id}.docx"
    # Content-Disposition 用 filename* 支持中文
    disposition = f"attachment; filename*=UTF-8''{quote(filename)}"
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": disposition},
    )
