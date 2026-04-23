import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.api import chat, resume
from app.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _preload_chromadb():
    """事件循环启动前同步预加载 ChromaDB gRPC（避免首次请求时 GIL 卡住事件循环）"""
    try:
        from app.rag.vector_store import get_retriever

        get_retriever(collection="jargon_dict", k=1)
        get_retriever(collection="golden_resumes", k=1)
        logger.info("ChromaDB preloaded successfully")
    except Exception as e:
        logger.warning(f"ChromaDB preload skipped: {e}")


# 在模块导入时（事件循环启动前）同步执行
_preload_chromadb()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="胡话简历 Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
