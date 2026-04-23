from pydantic import BaseModel
from datetime import datetime


class DecoderRequest(BaseModel):
    jd_text: str
    user_id: str | None = None


class DecodedItem(BaseModel):
    keyword: str
    surface_meaning: str
    real_meaning: str


class KeywordCloud(BaseModel):
    core_keywords: list[str] = []
    secondary_keywords: list[str] = []
    industry_keywords: list[str] = []


class ResumeStrategy(BaseModel):
    must_highlight: list[str] = []
    recommended_format: str = ""
    keyword_density_targets: list[str] = []


class JdInfo(BaseModel):
    job_title: str = ""
    company: str = ""
    industry: str = ""
    experience_years: str = ""
    education: str = ""
    required_skills: list[str] = []
    key_responsibilities: list[str] = []
    preferred_qualities: list[str] = []
    salary_range: str = ""
    hard_requirements: list[str] = []
    keyword_cloud: KeywordCloud = KeywordCloud()
    resume_strategy: ResumeStrategy = ResumeStrategy()


class DecoderResponse(BaseModel):
    id: str
    decoded_items: list[DecodedItem]
    summary: str
    jd_info: JdInfo | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
