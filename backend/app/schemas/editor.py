from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BasicInfo(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""
    location: str = ""
    objective: str = ""
    gender: str = ""
    age: str = ""
    birthday: str = ""
    hometown: str = ""
    political: str = ""
    ethnicity: str = ""
    # 默认空串，只有用户在对话里主动说才填（"我 167cm"、"身高 170"）
    height: str = ""
    # 头像 data URL（data:image/jpeg;base64,... 或 data:image/png;base64,...）；空则模板显示占位
    photo: str = ""


class EducationItem(BaseModel):
    school: str = ""
    degree: str = ""
    major: str = ""
    start_date: str = ""
    end_date: str = ""
    highlights: str = ""


class WorkItem(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: list[str] = []


class ProjectItem(BaseModel):
    name: str = ""
    role: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""
    # 编辑 agent 压缩后的 STAR bullet（3-5 条）；为空则模板走 description fallback
    polished_bullets: list[str] = []


class ResumeSection(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    kind: Literal[
        "education", "work_experience", "projects", "skills",
        "certificates", "self_evaluation", "custom",
    ]
    title: str = Field(max_length=100)
    content: str = Field(default="", max_length=20000)

    @field_validator("id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("section id must not be blank")
        return value


class ResumeData(BaseModel):
    basic_info: BasicInfo = BasicInfo()
    education: list[EducationItem] = []
    work_experience: list[WorkItem] = []
    projects: list[ProjectItem] = []
    # flat 技能（旧字段，兼容保留）
    skills: list[str] = []
    # 分组技能：{"技术栈": ["Java","Python"], "AI 工具": ["Claude Code", ...]}
    # 编辑 agent 产出；模板优先用 skill_groups，退回 skills
    skill_groups: dict[str, list[str]] = {}
    certificates: list[str] = []
    self_evaluation: str = ""
    # 经历 section 标题（校园经历 / 工作经历 / 过往经历），空则模板用默认
    experience_section_title: str = ""
    # None 兼容旧简历默认栏目；空列表表示用户移除了全部正文栏目。
    sections: list[ResumeSection] | None = Field(default=None, max_length=30)

    @model_validator(mode="after")
    def unique_sections(self) -> "ResumeData":
        ids: set[str] = set()
        builtin_kinds: set[str] = set()
        for section in self.sections or []:
            if section.id in ids:
                raise ValueError("section ids must be unique")
            ids.add(section.id)
            if section.kind != "custom":
                if section.kind in builtin_kinds:
                    raise ValueError("builtin section kinds must be unique")
                builtin_kinds.add(section.kind)
        return self


class GenerateRequest(BaseModel):
    template_id: str
    resume_data: ResumeData


class TemplateInfo(BaseModel):
    id: str
    filename: str
    category: str


class GenerateResponse(BaseModel):
    filename: str
    download_url: str
