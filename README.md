# 胡话简历 · Huhua Agent

AI 简历对话工作台：聊天收集经历 → Agent 美化 → 6 款 A4 模板一键出 PDF / Word。

基于 **FastAPI + LangGraph** 构建，不是一次性填表工具——像朋友聊天一样把你的经历抽成结构化简历。

---

## 能干嘛

- **对话式收集**：不硬要 JD，随便聊"最近在忙什么"都能抽出项目素材
- **自动包装**：把"我做了 TripAgent 跑 50 条 query"这种自然语言，用 STAR 法则重写成 ATS 友好的 bullet
- **智能分组**：Java/Python → 技术栈；PRD/竞品 → 产品方法；Claude Code/ChatGPT → AI 工具——自动归类不用手动分
- **6 款模板**：简约单栏 / 深蓝横幅 / 灰条夹页 / 浅蓝斜切 / 浅蓝色块 / 淡青平行，都是 A4 ATS 友好样式
- **头像上传**：base64 内嵌，DOCX 导出也会带图
- **导出**：浏览器打印成 PDF，或下载 Word

---

## 架构

```
用户 ──► SSE 聊天 ──► LangGraph StateGraph
                      │
                      ▼
          JD_INPUT → COLLECTING → BUILDING → EVALUATING → EXPORT
                      │           │
                      │           ├── wrapper_chain（每条经历 STAR 包装）
                      │           └── editor_chain（技能分组 / bullet 压缩 / section 命名）
                      │
                      └── triage_chain（9 个 Action 路由：DECODE_JD / WRAP / ASK_EXP / READY_TO_BUILD / …）

渲染：ResumeData → Jinja2 模板 → HTML（浏览器打印 PDF）/ python-docx（直出 DOCX）
```

**双模型策略**：
- `LLM_MODEL=qwen3.6-plus` → 主对话流，追求自然度
- `LLM_STRUCTURED_MODEL=qwen-plus` → Triage / editor / wrapper 的 `with_structured_output` 调用，快 6 倍

---

## 启动

```bash
# 1. 填 API key
cp .env.example backend/.env
# 编辑 backend/.env 填入你的 LLM_API_KEY

# 2. 启动后端（用 uv 临时注入依赖）
cd backend
uv run --with fastapi --with uvicorn --with jinja2 --with pydantic-settings \
  --with sqlalchemy --with python-docx --with pdfplumber \
  --with langchain --with langchain-openai --with langchain-chroma \
  --with langchain-community --with langgraph --with chromadb \
  --with python-dotenv --with python-multipart --with rank_bm25 \
  uvicorn main:app --host 127.0.0.1 --port 8765

# 3. 打开浏览器
open http://127.0.0.1:8765/
```

---

## 目录

```
backend/
  app/
    agents/           # triage / wrapper / decoder / editor / resume_graph
    api/              # chat.py / resume.py
    services/         # renderer.py（Jinja2）/ file_extractor.py（PDF/DOCX 解析）
    schemas/          # ResumeData / BasicInfo 等 pydantic
    tools/            # resume_data_docx.py（结构化 → Word）
  main.py

web/
  index.html          # 聊天页（Eloquent Canvas 紫色风）
  resume.html         # 模板预览 + 头像上传 + 导出

data/
  jianlimoban/html-template/  # 6 款 Jinja2 模板
  golden_resumes.json          # RAG 参考库
  jargon_dict.csv              # 黑话词典
  sample_resume.json           # 模板预览占位数据（胡小豆）
```

---

## License

MIT

---

Built by [@YIKUAIBANZI](https://github.com/YIKUAIBANZI)
