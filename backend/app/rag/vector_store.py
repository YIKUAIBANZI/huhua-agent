from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.embeddings import get_embeddings

_stores: dict = {}


def get_store(collection: str) -> Chroma:
    if collection not in _stores:
        settings = get_settings()
        _stores[collection] = Chroma(
            collection_name=collection,
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_DIR,
        )
    return _stores[collection]


class HybridRetriever:
    """BM25 + 向量检索混合，各 50% 权重，去重后返回"""

    def __init__(self, bm25: BM25Retriever, vector, k: int):
        self.bm25 = bm25
        self.vector = vector
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        bm25_docs = self.bm25.invoke(query)
        vec_docs = self.vector.invoke(query)

        seen: set[str] = set()
        merged: list[Document] = []
        for doc in bm25_docs + vec_docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged[: self.k]


def get_retriever(
    collection: str,
    k: int | None = None,
    filter_metadata: dict | None = None,
):
    """获取混合检索器。

    Args:
        collection: ChromaDB 集合名
        k: 返回数量
        filter_metadata: 可选 metadata 过滤，如 {"industry": "互联网"} 或 {"role": "产品经理"}
    """
    settings = get_settings()
    k = k or settings.RAG_TOP_K
    store = get_store(collection)

    # 构建 metadata where 子句
    where = None
    if filter_metadata:
        conditions = []
        for key, val in filter_metadata.items():
            if val:
                conditions.append({key: val})
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

    # 从 ChromaDB 取文档构建 BM25（支持 metadata 过滤）
    get_kwargs = {}
    if where:
        get_kwargs["where"] = where
    result = store._collection.get(**get_kwargs)
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]

    # BM25 对过滤后的文档集检索
    if not docs:
        # 无匹配文档时回退到全量
        result_all = store._collection.get()
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result_all["documents"], result_all["metadatas"])
        ]

    if docs:
        bm25 = BM25Retriever.from_documents(docs, k=k)
    else:
        # 集合为空，用空检索器
        class _EmptyRetriever:
            def invoke(self, _query: str) -> list:
                return []

        bm25 = _EmptyRetriever()

    # 向量检索也支持 metadata 过滤
    search_kwargs: dict = {"k": k}
    if where:
        search_kwargs["filter"] = where
    vector = store.as_retriever(search_kwargs=search_kwargs)

    return HybridRetriever(bm25=bm25, vector=vector, k=k)
