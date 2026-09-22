import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.api import chat, resume
from app.database import init_db
from app.config import get_settings
from app.agents.resume_graph import configure_checkpointer
from app.services.session_store import delete_expired_resume_sessions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _preload_chromadb():
    """事件循环启动前同步预加载 ChromaDB gRPC（避免首次请求时 GIL 卡住事件循环）"""
    if not get_settings().ENABLE_RAG or not get_settings().LLM_API_KEY:
        return
    try:
        from app.rag.vector_store import get_retriever

        get_retriever(collection="jargon_dict", k=1)
        get_retriever(collection="golden_resumes", k=1)
        logger.info("ChromaDB preloaded successfully")
    except Exception as e:
        logger.warning(f"ChromaDB preload skipped: {e}")


# 在模块导入时（事件循环启动前）同步执行
_preload_chromadb()


async def _cleanup_expired_sessions(checkpointer) -> None:
    session_ids = delete_expired_resume_sessions(get_settings().SESSION_RETENTION_DAYS)
    for session_id in session_ids:
        try:
            await checkpointer.adelete_thread(session_id)
        except Exception:
            logger.warning("expired checkpoint deletion failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    database_url = get_settings().DATABASE_URL
    if database_url.startswith(("postgres://", "postgresql://")):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
            await saver.setup()
            configure_checkpointer(saver)
            await _cleanup_expired_sessions(saver)
            try:
                yield
            finally:
                configure_checkpointer(None)
    else:
        checkpoint_path = Path(get_settings().CHECKPOINT_DB_PATH)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            configure_checkpointer(saver)
            await _cleanup_expired_sessions(saver)
            try:
                yield
            finally:
                configure_checkpointer(None)


app = FastAPI(title="胡话简历 Agent", version="1.0.0", lifespan=lifespan)

_chat_requests: dict[str, deque[float]] = defaultdict(deque)
CHAT_WINDOW_SECONDS = 600
CHAT_REQUEST_LIMIT = 15

@app.middleware("http")
async def security_headers(request: Request, call_next):
    if request.url.path == "/api/chat/stream" and request.method == "POST":
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        client_ip = forwarded or (request.client.host if request.client else "unknown")
        now = time.monotonic()
        timestamps = _chat_requests[client_ip]
        while timestamps and now - timestamps[0] > CHAT_WINDOW_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= CHAT_REQUEST_LIMIT:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=429,
                content={"detail": "请求有点频繁，请稍后再试"},
                headers={"Retry-After": "60"},
            )
        timestamps.append(now)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(resume.router, prefix="/api/resume", tags=["resume"])

# 静态文件（前端）
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/resume")
    async def resume_page():
        return FileResponse(os.path.join(WEB_DIR, "resume.html"))
