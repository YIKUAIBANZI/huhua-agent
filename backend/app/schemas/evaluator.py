from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    resume_text: str
    jd_context: dict | None = None


class KeywordMatchScore(BaseModel):
    score: float = 0
    matched: list[str] = []
    missed: list[str] = []


class DetailScore(BaseModel):
    score: float = 0
    detail: str = ""


class FormatScore(BaseModel):
    score: float = 0
    issues: list[str] = []


class FeedbackScore(BaseModel):
    score: float = 0
    feedback: str = ""


class AtsScore(BaseModel):
    total: float = 0
    keyword_match: KeywordMatchScore = KeywordMatchScore()
    experience_relevance: DetailScore = DetailScore()
    format_compatibility: FormatScore = FormatScore()
    education_match: DetailScore = DetailScore()
    recency: DetailScore = DetailScore()
    soft_skills: DetailScore = DetailScore()


class HrScore(BaseModel):
    total: float = 0
    first_impression: FeedbackScore = FeedbackScore()
    deep_evaluation: FeedbackScore = FeedbackScore()
    differentiation: FeedbackScore = FeedbackScore()


class PassPrediction(BaseModel):
    ats_pass_rate: str = ""
    hr_pass_rate: str = ""
    interview_probability: str = ""


class CompetitiveAnalysis(BaseModel):
    strengths: list[str] = []
    weaknesses: list[str] = []
    market_position: str = ""


class ImprovementItem(BaseModel):
    priority: int = 0
    action: str = ""
    expected_score_gain: str = ""
    effort: str = "medium"


class EvaluateResponse(BaseModel):
    total_score: float = 0
    ats_score: AtsScore = AtsScore()
    hr_score: HrScore = HrScore()
    pass_prediction: PassPrediction = PassPrediction()
    competitive_analysis: CompetitiveAnalysis = CompetitiveAnalysis()
    improvement_roadmap: list[ImprovementItem] = []
