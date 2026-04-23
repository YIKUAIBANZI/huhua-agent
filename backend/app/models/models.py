import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from app.database import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, nullable=True)
    nickname = Column(String, nullable=True)
    industry = Column(String, default="互联网")
    created_at = Column(DateTime, default=datetime.utcnow)


class WrapperRecord(Base):
    __tablename__ = "wrapper_records"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    original_text = Column(Text, nullable=False)
    wrapped_text = Column(Text, nullable=False)
    industry = Column(String, default="互联网")
    references = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)


class DecodeRecord(Base):
    __tablename__ = "decode_records"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    jd_text = Column(Text, nullable=False)
    decoded_result = Column(JSON, default=list)
    summary = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
