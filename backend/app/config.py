from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 数据库
    DATABASE_URL: str = "sqlite:///./data/huhua.db"

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

    # RAG
    CHROMA_DIR: str = "./data/chroma_db"
    RAG_TOP_K: int = 5

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
