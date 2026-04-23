from pydantic import BaseModel
from datetime import datetime


class WrapperRequest(BaseModel):
    experience: str
    industry: str = "互联网"
    user_id: str | None = None
    follow_up_answer: str | None = None  # 追问补充信息


class FollowUpResponse(BaseModel):
    needs_followup: bool = True
    question: str


class WrapperResponse(BaseModel):
    id: str
    original: str
    wrapped: str
    references: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}
