from langchain_openai import OpenAIEmbeddings
from app.config import get_settings


def get_embeddings():
    settings = get_settings()
    return OpenAIEmbeddings(
        model="text-embedding-v4",
        openai_api_key=settings.LLM_API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
