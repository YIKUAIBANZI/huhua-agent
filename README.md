# 胡话简历 · Huhua Agent

AI 简历对话工作台：聊天收集真实经历 → 针对岗位整理 → 核对事实 → 6 款 A4 模板导出 PDF / Word。

基于 **FastAPI + LangGraph** 构建，不是一次性填表工具——像朋友聊天一样把你的经历抽成结构化简历。

---

## 能干嘛

- **对话式收集**：不硬要 JD，随便聊"最近在忙什么"都能抽出项目素材
- **自动包装**：把"我做了 TripAgent 跑 50 条 query"这种自然语言，用 STAR 法则重写成清晰 bullet，不补用户没说过的数字
- **智能分组**：Java/Python → 技术栈；PRD/竞品 → 产品方法；Claude Code/ChatGPT → AI 工具——自动归类不用手动分
- **投递检查**：指出 JD 证据缺口、待核实数字和导出风险，不虚构面试概率
- **6 款模板**：简约单栏 / 深蓝横幅 / 灰条夹页 / 浅蓝斜切 / 浅蓝色块 / 淡青平行
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

**双模型策略**：`LLM_MODEL` 用于主对话，`LLM_STRUCTURED_MODEL` 用于 Triage / editor 等结构化调用。默认配置使用 `qwen-plus`。

---

## 启动

```bash
# 1. 填 API key
cp .env.example backend/.env
# 编辑 backend/.env 填入你的 LLM_API_KEY

# 2. 安装依赖并启动
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cd backend
../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8765

# 3. 打开浏览器
open http://127.0.0.1:8765/
```

运行回归检查：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 公开 beta

- `Dockerfile` 可直接运行服务，`render.yaml` 会创建 Web Service 和 Postgres。
- 在托管平台把 `LLM_API_KEY` 配成 Secret，不能提交到仓库。
- 会话数据默认保留 7 天；用户点击“新对话”会立即删除当前会话。
- Render 免费 Web 会休眠，免费 Postgres 目前也有期限，仅适合小规模 beta；正式开放前应升级存储或迁移数据库。

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
