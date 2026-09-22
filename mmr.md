# 胡话简历交接

## North star

让秋招求职者把真实经历快速变成一份针对目标岗位、可以核对并直接投递的简历。

## 当前决策

- 尽早开放给真实用户，先做小规模公开 beta。
- 事实可信与导出一致优先于增加 Agent、模板和虚构的 ATS 分数。
- AI 不补估算数字；缺信息时追问或保留定性表达。
- Jev 匹配是用户主动点击的实验功能：只发送固定能力标签，给出有范围说明的参考覆盖分；识别不足不出总分。

## 当前状态

- 2026-09-23 已从远端恢复 55 个缺失文件，原有 8 个本地改动另有完整备份。
- 已补齐缺失的 `resume_strategy`、`ResumeSession`、运行依赖和持久 checkpoint。
- 已加入会话删除、基础限流、安全响应头、健康检查和容器部署文件；线上状态使用 Postgres，避免免费 Web 容器休眠后丢失会话。
- 静态页面、模板列表、样例、HTML 渲染和 DOCX 导出已通过本地 API 冒烟测试。
- 已修复生成后补充事实不更新简历、内部模型流泄露、无来源的职责强化；Word 明示统一排版。
- 已加浏览器本地记录过期、服务端定期清理、删除失败重试标记、上传与请求大小边界、预览隔离和移动端模板布局。
- Jev 可选接口已实现；无 `JEV_API_KEY` 时入口隐藏，缺 JD、模型失败或标签覆盖不足时不给分。
- 当前 28 项自动化测试通过，GitHub Actions 测试工作流已加入；移动端首页与模板页已做浏览器检查。
- 真实 LLM 对话仍需有效的 DashScope/OpenAI 兼容 API key 验证。

## 立即下一步

1. 配置生成模型 API key，跑一遍真实 JD 到 DOCX 的全流程；Jev key 可后配。
2. 用 3 个真实但脱敏的固定案例核对事实没有被新增、预览与导出内容一致。
3. 审查并合并 PR，部署公开 beta；先邀请 3–5 位秋招用户完成一次真实投递。
4. Jev 参考分上线后收集 30–50 份人工标注对照，校准词表、覆盖门槛和分数解释。

## 文件地图

- `backend/app/agents/resume_graph.py`：主状态机。
- `backend/app/services/resume_strategy.py`：确定性简历策略。
- `backend/app/services/resume_review.py`：基于同一份结构化数据的投递检查。
- `backend/app/services/jev_match.py`：可选、脱敏的 JD 参考覆盖分。
- `web/index.html`：聊天与实时画布。
- `Dockerfile` / `render.yaml`：发布配置。
