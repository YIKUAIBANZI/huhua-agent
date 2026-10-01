# 胡话简历 · Huhua Agent

AI 简历对话工作台：聊天收集真实经历 → 针对岗位整理 → 核对事实 → 6 款 A4 模板导出 PDF / Word。

基于 **FastAPI + LangGraph** 构建，不是一次性填表工具——像朋友聊天一样把你的经历抽成结构化简历。

---

## 能干嘛

- **对话式收集**：不硬要 JD，随便聊"最近在忙什么"都能抽出项目素材
- **截图解码**：上传 JPG/JPEG/PNG 岗位截图后由视觉模型转写文字；识别结果可以先核对、修改，再发送给 JD 解码流程
- **自动包装**：把"我做了 TripAgent 跑 50 条 query"这种自然语言，用 STAR 法则重写成清晰 bullet，不补用户没说过的数字
- **智能分组**：Java/Python → 技术栈；PRD/竞品 → 产品方法；Claude Code/ChatGPT → AI 工具——自动归类不用手动分
- **投递检查**：指出 JD 证据缺口、待核实数字和导出风险，不虚构面试概率
- **可选 Jev 参考分**：用户主动点击后，根据已识别的 JD 要求与简历能力证据给出参考覆盖分；未配置 `JEV_API_KEY` 时不显示入口
- **6 款模板**：简约单栏 / 深蓝横幅 / 灰条夹页 / 浅蓝斜切 / 浅蓝色块 / 淡青平行
- **实时表单编辑**：右侧按栏目填写，输入立即更新 A4 预览；悬停文字显示虚线框并高亮对应输入框，点击可定位编辑。支持增删经历、要点与技能分组
- **自定义栏目**：拖动栏目把手排序，或用上移/下移按钮；可修改标题、删除与撤销，也可添加自定义标题和多行正文。自定义标题留空时只显示正文；六款模板与 PDF / Word 都遵循当前栏目顺序
- **头像上传**：base64 内嵌，DOCX 导出也会带图
- **导出**：所选模板由浏览器打印成 PDF；Word 使用统一排版，便于继续编辑

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

**双模型策略**：`LLM_MODEL` 用于主对话，`LLM_STRUCTURED_MODEL` 用于 Triage / editor 等结构化调用。示例配置分别使用 `qwen3.6-plus` 和 `qwen-plus`。

图片使用 `LLM_VISION_MODEL`。留空时按 API 地址选择：DeepSeek 使用 `deepseek-flash`，DashScope 使用 `qwen-vl-plus`，OpenAI 默认地址使用 `gpt-4o-mini`。图片会发送到配置的视觉模型服务；没有配置 `LLM_API_KEY` 时返回明确错误。图片上传仍受 10MB 限制，并有单独并发与请求频率限制。

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

如果使用 DeepSeek，在 `backend/.env` 中设置 `LLM_PROVIDER=deepseek`、`LLM_BASE_URL=https://api.deepseek.com`、`LLM_MODEL=deepseek-flash`、`LLM_STRUCTURED_MODEL=deepseek-flash`，并填写新生成的 `LLM_API_KEY`。`LLM_VISION_MODEL` 可留空，图片识别会自动使用 `deepseek-flash`。请勿把密钥提交到仓库或贴进聊天。

运行回归检查：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

GitHub PR 会自动运行同一套测试。`JEV_API_KEY` 是可选配置；Jev 接口只发送从 JD 和简历抽出的能力标签，不发送姓名、联系方式、学校、公司或完整对话。参考分是当前简历的证据覆盖提示，不代表招聘系统评分或录用概率。

模板编辑页位于 `/resume`，从当前聊天会话读取简历；没有会话简历时显示样例。此页的修改保留在当前页面，切换模板不会丢失，导出使用最新输入；刷新会重新读取会话数据，尚不回写聊天。

本地服务启动后，可用已安装的 `playwright-cli` 运行交互回归（无需模型密钥）：

```bash
playwright-cli -s=huhua-editor open http://127.0.0.1:8765/resume
playwright-cli -s=huhua-editor run-code --filename=tests/resume_editor.browser.js
playwright-cli -s=huhua-editor run-code --filename=tests/resume_sections.browser.js
```

检查覆盖悬停与定位、连续输入和慢请求、增删后的字段对应、技能分组、原生栏目拖拽/改名/撤销、自定义正文、六款模板、最新内容导出及手机布局。普通导出不包含编辑脚本或高亮样式。

## 公开 beta

- `Dockerfile` 可直接运行服务，`render.yaml` 会创建 Web Service 和 Postgres。
- 在托管平台把 `LLM_API_KEY` 配成 Secret，不能提交到仓库。
- 会话数据约保留 7 天；浏览器会在下次打开时清理超过 7 天的本地聊天记录。用户点击“新对话”会删除当前会话。
- 当前聊天、匹配和上传限流是单进程内存计数；共用代理 IP 的用户可能共用额度。扩到多实例前需要平台或网关限流，不能直接相信客户端传来的 `X-Forwarded-For` 首项。
- 服务端最多接受 10MB 文件，并对常见的超大上传提前拒绝；正式扩量前还需在网关限制分块上传的请求体大小。
- Render 免费 Web 会休眠，免费 Postgres 目前也有期限，仅适合小规模 beta；正式开放前应升级存储或迁移数据库。

---

## 目录

```
backend/
  app/
    agents/           # triage / wrapper / decoder / editor / resume_graph
    api/              # chat.py / resume.py
    services/         # renderer.py（Jinja2）/ resume_sections.py（栏目顺序）/ file_extractor.py / image_ocr.py
    schemas/          # ResumeData / BasicInfo 等 pydantic
    tools/            # resume_data_docx.py（结构化 → Word）
  main.py

web/
  index.html          # 聊天页（Eloquent Canvas 紫色风）
  resume.html         # 模板选择 + 实时表单编辑 + 导出
  resume-editor.js    # 表单与渲染请求、字段联动
  resume-editor.css   # 编辑工作台与移动端布局
  resume-preview.js  # 隔离 iframe 中的字段映射与高亮

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
