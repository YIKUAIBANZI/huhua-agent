from pydantic import BaseModel


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
