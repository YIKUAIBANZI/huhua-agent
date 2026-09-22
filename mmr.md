# 胡话简历交接

## North star

让秋招求职者把真实经历快速变成一份针对目标岗位、可以核对并直接投递的简历。

## 当前决策

- 尽早开放给真实用户，先做小规模公开 beta。
- 事实可信与导出一致优先于增加 Agent、模板和虚构的 ATS 分数。
- AI 不补估算数字；缺信息时追问或保留定性表达。

## 当前状态

- 2026-09-23 已从远端恢复 55 个缺失文件，原有 8 个本地改动另有完整备份。
- 已补齐缺失的 `resume_strategy`、`ResumeSession`、运行依赖和持久 checkpoint。
- 已加入会话删除、基础限流、安全响应头、健康检查和容器部署文件；线上状态使用 Postgres，避免免费 Web 容器休眠后丢失会话。
- 静态页面、模板列表、样例、HTML 渲染和 DOCX 导出已通过本地 API 冒烟测试。
- 真实 LLM 对话仍需有效的 DashScope/OpenAI 兼容 API key 验证。

## 立即下一步

1. 配置模型 API key，跑一遍真实 JD 到 DOCX 的全流程。
2. 做 3 个固定回归案例，核对事实没有被新增、预览与导出内容一致。
3. 部署公开 beta，先邀请 3–5 位秋招用户完成一次真实投递。

## 文件地图

- `backend/app/agents/resume_graph.py`：主状态机。
- `backend/app/services/resume_strategy.py`：确定性简历策略。
- `backend/app/services/resume_review.py`：基于同一份结构化数据的投递检查。
- `web/index.html`：聊天与实时画布。
- `Dockerfile` / `render.yaml`：发布配置。
