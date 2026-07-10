# 开发日志 (Development Log)

> 记录每次有意义的修改，便于回顾和面试时讲述演进思路。
>
> 格式：`YYYY-MM-DD | <类型> | <标题> | <分支>`

---

## 2026-07-10 | feat + fix | 工程化基础建设 + ROUTER_PROMPT bug 修复 | test/engineering-foundation

### 背景
项目有 27 个单元测试和 BDD 测试，但缺少：
1. CI/CD 自动化
2. 集成测试（mock LLM 的核心链路覆盖）
3. 代码规范工具（lint）
4. 测试覆盖率度量
5. 编辑器跨平台一致性配置

### 新增文件

| 文件 | 说明 |
|------|------|
| `.github/workflows/ci.yml` | GitHub Actions CI：push/PR 时自动跑 pytest (unit+integration) + behave + ruff lint + 覆盖率上报 Codecov |
| `.editorconfig` | 跨编辑器代码风格一致性（UTF-8 / LF / 缩进规则） |
| `pyproject.toml` | pytest + coverage + ruff 统一配置（覆盖率红线 60%） |
| `tests/integration/__init__.py` | 集成测试包 |
| `tests/integration/conftest.py` | 共享 mock fixtures + 预设 LLM 响应 |
| `tests/integration/test_query_rewriter.py` | QueryRewriter 集成测试 (3 tests)：关键词改写、空输出回退、AgentOutput 接口 |
| `tests/integration/test_question_router.py` | QuestionRouter 集成测试 (4 tests)：trivial/moderate/complex 分类、异常降级 |
| `tests/integration/test_rag_search.py` | RAG Search 集成测试 (4 tests)：三源混合检索、RRF 去重、空后端、排序正确性 |
| `tests/integration/test_orchestrator.py` | QASystem 编排集成测试 (5 tests)：三阶层路由、DirectSolver、Planner 循环、降级、对话历史注入 |
| `DEVLOG.md` | 本文件 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `README.md` | 加入 CI badge + Codecov badge；更新技术栈添加测试工具 |
| `RULE.md` | 更新范式 9 测试分层描述（三层体系 + CI/CD） |
| `src/prompts/qa/router.py` | **Bug fix**: ROUTER_PROMPT 中的 JSON 示例花括号未转义，导致 `.format(question=...)` 每次都抛出 KeyError，Router 静默降级到 moderate。现已转义为 `{{` `}}` |

### Bug 修复详情

**严重程度**：🔴 高（影响所有问题路由准确性）

**问题**：`src/prompts/qa/router.py` 的 `ROUTER_PROMPT` 中 JSON 输出示例使用单花括号：
```python
# 修复前
{
  "difficulty": "...",
  ...
}
```
Python 的 `.format(question=...)` 将 `{\n  "difficulty"` 解析为占位符，找不到对应变量抛出 `KeyError`。Router 的 `except` 捕获后静默降级为 `difficulty="moderate"`，导致**所有问题都被路由到中等难度**，trivial 和 complex 路径永不被触发。

**修复**：JSON 花括号转义 `{` → `{{`，`}` → `}}`。

**影响范围**：Router 现在能正确分类 trivial/complex 问题，直接回答和 Planner 循环路径恢复可用。

### 测试结果

```
43 passed in 18.21s
├── 27 单元测试 (config + graph + chunker)
└── 16 集成测试 (rewriter + router + rag_search + orchestrator)
```

### 下一步建议

- [ ] PR 到 main 分支，验证 GitHub Actions 是否正常触发
- [ ] 在 GitHub 仓库设置中启用 Codecov（需要在 codecov.io 注册）
- [ ] 将 `ci.yml` 中 badge URL 的 `your-username/research-agent` 替换为实际仓库路径
- [ ] 给 README 和 RULE.md 中的变更做一次 commit

---

## 2026-07-09 | feat | 图片PDF章节检测全链路修复 | main

（历史记录，详见 README.md 更新说明）

---

## 2026-07-09 | feat | 项目初始化 (v0.1) | main

大学生自学指导 agent 0.1 版本，含：
- 文档解析 (PDF/TXT/MD/DOCX)
- 章节检测 + 知识提取 + 向量索引
- 基础 QA (ReAct + Reflection)
- 知识图谱 (SQLite)
- 前端 (Vue 3 + Vite)

---
