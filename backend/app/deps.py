from fastapi import Depends
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

# 模型路由配置：按任务复杂度选模型
# 可通过 .env 覆盖，格式 LLM_MODEL_DECODER=deepseek-v3.2
_TASK_MODEL_DEFAULTS = {
    "decoder": None,  # 简单：结构化提取 + 创意文案，用默认模型或 deepseek
    "wrapper": None,  # 中等：中文生成质量要求高，用默认模型或 qwen-plus
    "optimizer": None,  # 复杂：推理 + 大上下文，用默认模型或 claude-sonnet
    "evaluator": None,  # 中等：结构化输出，用默认模型或 qwen-plus
}


def _create_llm(
    settings: Settings,
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """创建 LLM 实例。"""
    kwargs = {
        "model": model or settings.LLM_MODEL,
        "api_key": settings.LLM_API_KEY,
        "temperature": temperature
        if temperature is not None
        else settings.LLM_TEMPERATURE,
    }
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    return ChatOpenAI(**kwargs)


def get_llm(settings: Settings = Depends(get_settings)) -> ChatOpenAI:
    """默认 LLM（兼容原有 API 签名）。"""
    return _create_llm(settings)


def get_llm_by_task(task_type: str, settings: Settings | None = None) -> ChatOpenAI:
    """按任务类型选择模型。

    优先级：.env 环境变量 > _TASK_MODEL_DEFAULTS > 默认模型
    例如设置 LLM_MODEL_DECODER=deepseek-v3.2 则 decoder 用 deepseek。
    """
    if settings is None:
        settings = get_settings()

    # 检查环境变量覆盖
    env_key = f"LLM_MODEL_{task_type.upper()}"
    import os

    env_model = os.environ.get(env_key)
    model = env_model or _TASK_MODEL_DEFAULTS.get(task_type) or settings.LLM_MODEL

    # 不同任务的温度微调
    temp_map = {
        "decoder": 0.8,  # 毒舌文案需要更高创意
        "wrapper": 0.7,  # 平衡创意与准确
        "optimizer": 0.3,  # 诊断分析需要更确定
        "evaluator": 0.2,  # 评分需要高一致性
    }
    temperature = temp_map.get(task_type, settings.LLM_TEMPERATURE)

    return _create_llm(settings, model=model, temperature=temperature)
