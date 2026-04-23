from pydantic import BaseModel
from datetime import datetime


class UserCreate(BaseModel):
    email: str | None = None
    nickname: str | None = None
    industry: str = "互联网"


class UserResponse(BaseModel):
    id: str
    email: str | None
    nickname: str | None
    industry: str
    created_at: datetime

    model_config = {"from_attributes": True}
