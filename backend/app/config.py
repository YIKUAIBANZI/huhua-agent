from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    # 数据库
    DATABASE_URL: str = f"sqlite:///{BACKEND_DIR / 'data' / 'huhua.db'}"
    CHECKPOINT_DB_PATH: str = str(BACKEND_DIR / "data" / "checkpoints.sqlite3")
    SESSION_RETENTION_DAYS: int = 7

    # LLM
    LLM_PROVIDER: str = "openai"  # openai / deepseek
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o"
    # 结构化输出（Triage / editor / decoder / info-extractor）专用模型；
    # qwen3.6-plus 的 with_structured_output 太慢（~12s），这里切到更快的 qwen3-max
    LLM_STRUCTURED_MODEL: str = ""  # 空则回退到 LLM_MODEL
    LLM_VISION_MODEL: str = "qwen-vl-plus"  # 用于 OCR 图片识别
    LLM_BASE_URL: str | None = None  # DeepSeek 等自定义 base_url
    LLM_TEMPERATURE: float = 0.7

    # 用户主动点击岗位匹配时才调用 Jev；凭证仅由服务端环境变量读取。
    JEV_API_KEY: str = ""
    JEV_TIMEOUT_SECONDS: float = 8.0

    # RAG
    CHROMA_DIR: str = str(PROJECT_DIR / "data" / "chroma_db")
    RAG_TOP_K: int = 5
    ENABLE_RAG: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
