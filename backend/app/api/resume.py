"""HTML 简历渲染 API：列模板 / 渲染 / 取样例数据 / 取当前 session 数据 / 导出 DOCX。"""

import io
import json
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.resume_graph import _serialize_to_resume_data, get_graph
from app.config import get_settings
from app.schemas.editor import ResumeData
from app.services.jev_match import (
    build_jev_payload,
    build_match_context,
    call_jev,
    score_jev_response,
)
from app.services.renderer import list_templates, render_resume
from app.services.resume_review import review_resume
from app.services.session_store import (
    load_resume_session,
    resume_data_has_content,
    session_can_read_checkpoint,
)
from app.tools.resume_data_docx import resume_data_to_docx

router = APIRouter()


class MatchRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)

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
    if not session_can_read_checkpoint(session_id):
        return {"has_data": False, "has_jd": False, "match_enabled": False, "data": None, "stage": ""}
    config = {"configurable": {"thread_id": session_id}}
    rd: dict = {}
    stage = ""
    has_jd = False
    stored = load_resume_session(session_id)
    if get_settings().LLM_API_KEY:
        try:
            graph = get_graph(_llm_for_graph())
            state = await graph.aget_state(config)
        except Exception:
            state = None
            graph_error = True
        else:
            graph_error = False
    else:
        state = None
        graph_error = False

    if state and state.values:
        vals = state.values
        stage = vals.get("stage") or "JD_INPUT"
        has_jd = bool(vals.get("jd_info") or vals.get("jd_text"))
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
                status_code=503, detail="暂时无法读取简历，请稍后重试"
            )
        return {"has_data": False, "has_jd": has_jd, "match_enabled": False, "data": None, "stage": stage}
    return {"has_data": True, "has_jd": has_jd, "match_enabled": bool(get_settings().JEV_API_KEY), "data": rd, "stage": stage}


@router.post("/match")
async def match_resume(req: MatchRequest):
    """User-triggered, privacy-minimized Jev JD coverage check."""
    empty = {"score": None, "requirements": [], "checks": []}
    if not session_can_read_checkpoint(req.session_id):
        return {**empty, "status": "no_resume", "message": "当前会话没有可核对的简历。"}

    stored = load_resume_session(req.session_id)
    resume_data = (stored or {}).get("resume_data") or {}
    jd_info: dict = {}
    jd_text = ""
    graph_failed = False
    settings = get_settings()
    if settings.LLM_API_KEY:
        try:
            graph = get_graph(_llm_for_graph())
            state = await graph.aget_state(
                {"configurable": {"thread_id": req.session_id}}
            )
            vals = (state.values if state else {}) or {}
            if not resume_data_has_content(resume_data):
                resume_data = vals.get("resume_data") or {}
                if not resume_data_has_content(resume_data) and vals:
                    resume_data = _serialize_to_resume_data(vals)
            jd_info = vals.get("jd_info") or {}
            jd_text = vals.get("jd_text") or ""
        except Exception:
            graph_failed = True

    if not resume_data_has_content(resume_data):
        return {**empty, "status": "no_resume", "message": "当前会话没有可核对的简历。"}

    checks = review_resume(resume_data, jd_info)
    fallback = {**empty, "checks": checks}
    if not jd_text:
        status = "unavailable" if graph_failed else "no_jd"
        message = "暂时无法读取岗位要求，请稍后重试。" if graph_failed else "当前会话没有可核对的 JD。"
        return {**fallback, "status": status, "message": message}
    if not settings.JEV_API_KEY:
        return {**fallback, "status": "unavailable", "message": "岗位匹配参考功能尚未配置。"}

    context = build_match_context(resume_data, jd_info, jd_text)
    requirements = context["requirements"]
    recognized = context["outbound_state"]["requirements"]
    evidence = context["outbound_state"]["evidence"]
    if not requirements or not recognized or not evidence:
        return {
            **fallback,
            "status": "insufficient",
            "requirements": [
                {"id": row["id"], "text": row["text"], "category": row["category"],
                 "weight": row["weight"], "level": "unscored", "evidence_id": None,
                 "evidence_text": None}
                for row in requirements
            ],
            "message": "JD 或简历中可识别的能力信息不足，暂时不给参考分。",
        }
    try:
        response = await call_jev(
            build_jev_payload(context), settings.JEV_API_KEY, settings.JEV_TIMEOUT_SECONDS
        )
        result = score_jev_response(context, response)
    except Exception:
        return {
            **fallback,
            "status": "unavailable",
            "message": "岗位匹配暂时不可用，请稍后重试。",
        }
    return {**result, "checks": checks}


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
    filename = f"{name}-通用Word.docx"
    # Content-Disposition 用 filename* 支持中文
    disposition = f"attachment; filename*=UTF-8''{quote(filename)}"
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": disposition},
    )
