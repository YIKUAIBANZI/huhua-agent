from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from pathlib import Path
from app.config import get_settings


def _sqlalchemy_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


class Base(DeclarativeBase):
    pass


engine = create_engine(
    _sqlalchemy_url(get_settings().DATABASE_URL),
    connect_args=(
        {"check_same_thread": False}
        if get_settings().DATABASE_URL.startswith("sqlite")
        else {}
    ),
)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    """建表，启动时调用一次"""
    from app.models import models  # noqa: F401 — 确保 model 被注册
    if engine.dialect.name == "sqlite" and engine.url.database not in (None, "", ":memory:"):
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI 依赖：每个请求一个 session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
