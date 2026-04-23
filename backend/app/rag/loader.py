import json
import csv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " "],
)


def load_golden_resumes(path: str) -> list[Document]:
    """加载 golden_resumes.json，每条 bullet point 作为一个 Document"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        text = item.get("bullet", "")
        metadata = {
            "industry": item.get("industry", ""),
            "role": item.get("role", ""),
            "source": "golden_resumes",
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def load_jargon_dict(path: str) -> list[Document]:
    """加载 jargon_dict.csv，每行作为一个 Document"""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = f"{row['keyword']}：{row['real_meaning']}"
            metadata = {
                "keyword": row.get("keyword", ""),
                "source": "jargon_dict",
            }
            docs.append(Document(page_content=text, metadata=metadata))
    return docs
