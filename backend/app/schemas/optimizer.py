from pydantic import BaseModel


class DimensionScore(BaseModel):
    score: float = 0
    brief: str = ""


class DimensionScores(BaseModel):
    jd_match: DimensionScore = DimensionScore()
    content_quality: DimensionScore = DimensionScore()
    ats_friendly: DimensionScore = DimensionScore()
    info_architecture: DimensionScore = DimensionScore()
    differentiation: DimensionScore = DimensionScore()


class KeywordAnalysis(BaseModel):
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    keyword_coverage_rate: str = ""


class FieldAdvice(BaseModel):
    field_name: str = ""
    diagnosis: str = ""
    current_value: str = ""
    recommended_value: str = ""
    avoid_value: str = ""
    priority: str = "medium"
    tip: str = ""


class SectionAdvice(BaseModel):
    section_name: str = ""
    current_score: str = ""
    fields: list[FieldAdvice] = []


class OptimizerOutput(BaseModel):
    """AI 调优的结构化输出模型"""

    overall_score: float = 0
    dimension_scores: DimensionScores = DimensionScores()
    keyword_analysis: KeywordAnalysis = KeywordAnalysis()
    sections: list[SectionAdvice] = []
    quick_wins: list[str] = []
    deep_optimization: list[str] = []
