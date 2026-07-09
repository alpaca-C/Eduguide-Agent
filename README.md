# Document QA System — 项目介绍

> 基于 LLM 的文档知识图谱问答系统。用户上传教材/论文/笔记，系统自动检测章节结构、构建向量索引和知识图谱，支持基于资料的自然语言问答。
>
> 最后更新：2026-07-09（图片PDF章节检测全链路修复：VLM目录提取+本地OCR+页码范围定向导入+增量索引入库）

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+（前端开发）
- Git

### 1. 克隆并进入项目

```bash
git clone https://github.com/your-username/research-agent.git
cd research-agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key（必填）：
#   LLM_API_KEY=sk-your-api-key-here
# 其他服务（Tavily 搜索、MinerU OCR）可选，不影响核心功能
```

### 3. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# 安装 Python 依赖
pip install -r requirements.txt

# 安装前端依赖并构建（可选，FastAPI 会自动 fallback 到旧版前端）
cd frontend && npm install && npm run build && cd ..
```

### 4. 启动服务

```bash
python -m src.main --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，上传一份教材或论文，开始自学问答。

### 前端开发模式（可选）

```bash
cd frontend && npm run dev    # Vite 热重载，端口 5173
```

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈](#2-技术栈)
3. [整体架构](#3-整体架构)
4. [模块详解](#4-模块详解)
   - [4.1 API 层](#41-api-层)
   - [4.2 Agent 层](#42-agent-层)
   - [4.3 工具层](#43-工具层)
   - [4.4 文档处理层](#44-文档处理层)
   - [4.5 知识图谱层](#45-知识图谱层)
   - [4.6 记忆/存储层](#46-记忆存储层)
   - [4.7 监控层](#47-监控层)
   - [4.8 Prompt 层](#48-prompt-层)
   - [4.9 前端](#49-前端)
5. [两条核心数据流](#5-两条核心数据流)
   - [5.1 文档导入流程](#51-文档导入流程)
   - [5.2 问答查询流程](#52-问答查询流程)
6. [现有设计模式（亮点）](#6-现有设计模式亮点)
7. [现有问题清单](#7-现有问题清单)
8. [优先级路线图](#8-优先级路线图)

---

## 1. 项目概述

这是一个**基于 LLM 的文档知识图谱问答系统**。用户上传教材/论文/笔记（PDF、TXT、MD、DOCX），系统自动检测章节结构、构建向量索引和知识图谱，然后用户可以基于这些资料进行自然语言问答。

核心特色：
- **双角色 Reflection Agent**：答疑老师（搜索+回答）→ 审核老师（评判+建议），最多 3 轮迭代
- **RAG 优先，网络兜底**：前 2 轮只查本地教材，第 3 轮才允许网络搜索
- **章节感知处理**：分块时保留章节边界，知识提取时按章节独立批处理（并发 LLM 调用）
- **增量向量索引**：新内容按 hash 去重追加，不重建 ChromaDB collection，按文档精准删除
- **扫描版 PDF 支持**：通过 MinerU 云 API 自动 OCR 无文字层的 PDF，Agent 可主动调用 `mineru_ocr` 工具
- **数学公式渲染**：前端集成 KaTeX，支持 `\(...\)` 行内公式和 `\[...\]` / `$$...$$` 块级公式

> 开发者规范见 [RULE.md](./RULE.md)

---

## 2. 技术栈

| 层 | 技术 |
|---|---|
| LLM | DeepSeek (v4-flash / chat) via OpenAI-compatible API |
| 视觉模型 | 百炼 qwen-vl-ocr（图片PDF章节检测 VLM 兜底，可选） |
| 后端框架 | FastAPI + Uvicorn |
| Agent 编排 | LangChain (ChatOpenAI + bind_tools) |
| 向量存储 | ChromaDB + Qwen3-Embedding-0.6B |
| 知识图谱 | SQLite 自建 |
| 会话存储 | SQLite + Qdrant Cloud（可选） |
| 文档解析 | PyMuPDF / PyPDF2 / python-docx / MinerU OCR / 百炼 VLM |
| 网络搜索 | Tavily API |
| Token 监控 | LangChain Callback + SQLite |
| 前端 | Vue 3 + Vite + Pinia + Vue Router + marked.js + KaTeX |
| 测试 | pytest (unit) + Behave (BDD) + MagicMock |

---

## 3. 整体架构

```
┌──────────────────────────────────────────────────────┐
│                   Frontend (SPA)                      │
│  index.html  /  css/style.css  /  js/app.js          │
│  marked.js (Markdown)  +  KaTeX (数学公式)            │
└──────────────────────┬───────────────────────────────┘
                       │ REST + SSE Streaming
┌──────────────────────▼───────────────────────────────┐
│                API Layer (FastAPI)                     │
│  src/api/ — 模块化路由，6 个 domain router              │
│  run_with_monitor.py — Token 监控包装器                │
│  src/main.py — uvicorn 启动入口                        │
└──────┬───────────────────────────────────────┬────────┘
       │                                       │
┌──────▼──────────┐  ┌─────────────────────────▼────────┐
│ Document         │  │ QA Pipeline                      │
│ Ingestion        │  │                                  │
│                  │  │ src/agents/qa.py                 │
│ src/documents/   │  │ ┌───────────────────────────┐   │
│ ├─ parser.py     │  │ │ ReflectionAgent (双角色)   │   │
│ └─ chunker.py    │  │ │ 答疑老师 → 审核老师         │   │
│                  │  │ │ R1-2: rag_search           │   │
│ src/agents/      │  │ │ R3:   + web_search         │   │
│ ├─ chapterizer   │  │ └───────────────────────────┘   │
│ └─ extractor     │  │                                  │
│                  │  │ src/tools/                       │
│ src/knowledge/   │  │ ├─ rag_search.py (向量+KG)      │
│ └─ graph.py      │  │ ├─ web_search.py (Tavily)       │
│                  │  │ └─ ocr.py (MinerU)              │
│ src/tools/       │  │                                  │
│ └─ ocr.py        │  │ src/prompts/qa.py                │
│                  │  │                                  │
└──────┬───────────┘  └──────────────────────────────────┘
       │
┌──────▼───────────────────────────────────────────────┐
│                Storage Layer                           │
│                                                       │
│  Vector Store       Knowledge Graph     Memory Store  │
│  (ChromaDB +         (SQLite)           (SQLite +      │
│   Qwen3-Embedding)                       Qdrant 可选)  │
│                                                       │
│  Token Monitor                                        │
│  src/monitoring/store.py (SQLite)                     │
└───────────────────────────────────────────────────────┘
```

---

## 4. 模块详解

### 4.1 API 层

**文件**：`src/api/`（模块化包，2026-07-09 从单文件拆分）、`src/main.py`、`run_with_monitor.py`

```
src/api/
├── __init__.py          # FastAPI app 工厂，挂载所有 router + 静态文件
├── deps.py              # 共享单例 (config, store, kg, vs, chapter_agent, caches)
├── schemas.py           # 共享 Pydantic 请求模型 (ChapterDetectRequest 等)
├── router_health.py     # GET  /api/health
├── router_files.py      # POST /api/files/upload | GET list | DELETE /{name}
├── router_chapters.py   # POST /api/chapters/detect | POST save | GET /{name}
├── router_knowledge.py  # POST /api/knowledge/process | DELETE clear | GET stats/documents
├── router_chat.py       # POST /api/chat
└── router_sessions.py   # GET  /api/sessions | GET/DELETE /{id}
```

各 router 按 domain 拆分，通过 `deps.py` 共享同一组单例（`config`, `store`, `kg`, `vs` 等），避免重复初始化。

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/files/upload` | POST | 上传文档（PDF/TXT/MD/DOCX） |
| `/api/files/list` | GET | 列出已上传文件 |
| `/api/files/{filename}` | DELETE | 删除文件 |
| `/api/chapters/detect` | POST | SSE 流式章节检测 |
| `/api/chapters/save` | POST | 持久化章节数据 |
| `/api/chapters/{filename}` | GET | 加载章节缓存 |
| `/api/knowledge/process` | POST | SSE 流式知识处理（解析→分块→索引→提取） |
| `/api/knowledge/clear` | DELETE | 清空知识图谱和向量库 |
| `/api/knowledge/stats` | GET | 知识图谱统计 |
| `/api/knowledge/documents` | GET | 已索引文档列表 |
| `/api/chat` | POST | SSE 流式问答 |
| `/api/sessions` | GET | 会话列表 |
| `/api/sessions/{id}` | GET / DELETE | 会话详情 / 删除 |
| `/api/health` | GET | 健康检查 |

章节检测支持**多文件并发** ✅ (2026-07-09 实现)：`asyncio.gather` + `Semaphore` + `asyncio.Queue` 架构，每个文件独立 worker，事件通过 Queue 汇聚 → SSE 实时推送。并发度通过 `CHAPTER_DETECT_CONCURRENCY` 环境变量调节（默认 3 路）。知识提取的 LLM batch 调用同样并发化（`EXTRACT_CONCURRENCY`）。

前端文件列表支持 **checkbox 多选** + 一键检测所有选中文件，进度和结果按文件名分组显示。

三个长耗时操作（章节检测、知识处理、问答）全部使用 **SSE (Server-Sent Events)** 流式推送进度。

### 4.2 Agent 层

#### QA 模块 — 五 Agent 三阶层路由 + 多跳依赖执行 ✅ (2026-07-09)

`src/agents/qa/` 模块，5 个子 Agent + 1 个 Orchestrator：

```
question → QuestionRouter (1 LLM)
  ├── 输出 decomposition 种子                              ← 🆕
  ├── "trivial"  → direct_answer → 返回
  ├── "moderate" → DirectSolver (think→act→syn, 2-3 LLM) → 返回
  └── "complex"  → Planner(seed=decomposition) → Executor(拓扑分轮) → Reflector (≤3 轮)
                    ├── depends_on 多跳依赖               ← 🆕
                    └── GAPS → 反馈注入 Planner 下一轮
```

| Agent | 文件 | 职责 | LLM |
|-------|------|------|-----|
| QuestionRouter | `qa/router.py` | 问题分类 + 难度判断 (trivial/moderate/complex) + **初步分解种子** 🆕 | 1 |
| DirectSolver | `qa/solver.py` | 中等难度：rewrite→search→synthesize | 2 |
| Planner | `qa/planner.py` | 复杂问题分解子问题（含 `depends_on` 依赖）+ 汇总结果。**接收 Router 分解种子** 🆕 | 1+1 |
| Executor | `qa/executor.py` | **拓扑分轮并发搜索**：按 `depends_on` 分组，前轮结果 enrich 后轮 query 🆕 | 0 |
| Reflector | `qa/reflector.py` | 结构化审核，输出 gaps + suggested_queries JSON | 1 |
| **QASystem** | `qa/orchestrator.py` | 三阶层路由编排 + 对话记忆 + 上下文压缩 + **种子传递** 🆕 | — |

**多跳依赖执行** ✅ (2026-07-09 新增)：
- `Executor._topological_rounds()`：按 `depends_on` 字段将子问题分组为拓扑层级
  - 无依赖 → 全部并发（向后兼容）
  - 线性链 1→2→3 → 3 轮依次执行
  - 钻石形 1→{2,3}→4 → 3 轮（同层并发）
- `Executor._enrich_query_with_context()`：用前轮搜索结果 + QueryRewriter 生成更精准的后续搜索词
- `Executor._summarize_for_context()`：压缩上下文（≤500 字符），避免逐轮膨胀

**Router 分解种子复用** ✅ (2026-07-09 新增)：
- Router 已输出的 `decomposition` 字段不再丢弃，作为 `seed_decomposition` 传入 Planner
- Planner prompt 新增 `{seed_section}` 段落，LLM 基于种子 refine 而非从零分解
- 贯穿所有路径：moderate escalation 和 complex fallback 均传递种子
- **不增加任何 LLM 调用**

**对话记忆** ✅ (2026-07-09)：
- 所有 sub-agent 注入 `chat_history`，多轮对话保持上下文连贯
- `_build_history_context()` 自适应压缩：≤6 条全注入 → 7-12 条截断 → >12 条规则摘要 + 最近 8 条
- `_handle_complex()` 中 observations 截断保护 (MAX_OBS_CHARS=6000)
- QASystem 单例化，请求间复用实例

LLM 调用对比（vs 旧 ReAct+Reflection）：

| 场景 | 旧 | 新 |
|------|----|----|
| 寒暄 | 1 | 1 |
| 简单查找 | 4+ | 3 |
| 跨文档对比 | 4-8 | 5-7 |
| 多跳推理 | 8-12 | 6-9 |

#### ChapterizerAgent (`chapterizer.py`)

负责从文档前 20 页中检测一级章节。支持常规文字 PDF、本地 OCR、VLM 目录提取三种路径：

```
文档前 20 页
  │
  ├── 数字 PDF (有嵌入文字): pymupdf 直接提取 → chapterizer (DeepSeek)
  │
  ├── 图片 PDF (无文字层): 本地 Tesseract OCR (6线程并行) + VLM TOC 扫描
  │   └── VLM 提取目录页中章节标题 + 起始页码，计算各章节页码范围
  │
  ├── VLM 快速通道: 若 metadata 含 pre-detected 章节列表，跳过 LLM 检测
  │   └── 直接 _split_by_meta + 审核，省一次 LLM 调用
  │
  ├── OCR 文本预处理: 超 15000 字智能截断 + 截断提示
  │
  └── Reflection Loop (≤2轮)
       ├── 检测专家 (LLM) → 定位目录、提取章节
       │   └── 优先查找集中目录区，若无则遍历全文搜索"第X章"标题
       ├── 启发式快判 (_fast_check_pass)
       ├── 审核专家 (LLM) → 逐项校验、剔除误检
       └── 不通过 → 收集反馈、下一轮
```

关键函数：`_split_by_meta()` 使用多阶段匹配（精确 → 空格归一化 → 前缀 fallback）将章节标题定位到正文实际位置。`_fast_check_pass()` 启发式快判：≥80% 标题匹配标准格式（`第X章` / `Chapter N` 等）时跳过 LLM 审核，省一次 API 调用。API 层有空内容早期检测（`chars=0` → 立即返回扫描版 PDF 提示，不浪费 LLM 调用）。

#### Extractor (`extractor.py`)

负责从分块文本中提取知识图谱：

```
分块列表
  │
  ├── Phase 1: 按章节边界分批 & 并发 LLM 调用 (Semaphore 控制并发)
  │   └── _process_one_batch() × N (并行)
  │
  └── Phase 2: 顺序合并
       ├── 概念去重 (lowercase name key)
       ├── 关系解析 (source/target 概念 ID 映射)
       └── 批量写入 KnowledgeGraph (SQLite)
```

关键参数：`extract_concurrency = 3`，`MAX_RETRIES = 3`（指数退避），`DEFAULT_CONCURRENCY = 3`

### 4.3 工具层

**文件**：`src/tools/__init__.py`、`rag_search.py`、`web_search.py`、`ocr.py`

采用 **Tool Registry 模式** + **三路混合检索 + RRF 融合** ✅ (2026-07-09)：

```
rag_search(query)
  ├── Dense:  ChromaDB + Qwen3-Embedding-0.6B  (语义, 跨语言)
  ├── Sparse: SQLite FTS5 + BM25               (关键词, 精确匹配)
  └── Graph:  SQLite KnowledgeGraph            (概念 + 1-hop 邻居)
       ↓
  RRF (Reciprocal Rank Fusion, k=60) 融合去重排序 → 结构化 ToolResult
```

**QueryRewriter** ✅ (2026-07-09)：搜索前将自然语言问题转化为 3-5 个优化查询词（中英双语 + 同义词），提升 FTS5 和 KG 召回率。

| 工具 | 功能 | 依赖 |
|---|---|---|
| `rag_search` | 三路混合检索 + RRF 融合 | ChromaDB + SQLite FTS5 + KnowledgeGraph |
| `web_search` | 互联网搜索（兜底） | Tavily API |
| `mineru_ocr` | 扫描版 PDF OCR 文字识别 | MinerU Cloud API |

**MinerU OCR 流程**（三策略容错 + 本地页面截取 + 回退重试）：

```
max_pages 指定时:
  pymupdf insert_pdf 本地截取前 N 页 → 临时 PDF（几 MB）→ 上传小文件

Strategy 1 (首选): POST /api/v4/extract/task     ← 显式创建任务
Strategy 2 (降级): GET  /api/v4/extract/task/batch/{batch_id}  ← 轮询等待自动创建
Strategy 3 (兜底): GET  /api/v4/extract/task?limit=10          ← 列出最近任务
    │
    ▼
GET /api/v4/extract/task/{task_id}  ← 轮询任务状态直到 done/failed
    │
    ▼
下载结果 zip → 提取 markdown 文本 → 清理临时文件
```

- **本地截取**：章节检测只需前 20 页，用 pymupdf `insert_pdf` 从大 PDF 中提取几 MB 的临时文件上传，节省带宽和时间。若子集 PDF 被 MinerU 拒收（`failed to read file`），自动回退上传完整文件 + 服务端 `page_ranges` 参数重试
- 有结果缓存（同文件同 `max_pages` 不重复 OCR），Agent 可通过 `mineru_ocr` 工具名直接调用
- 配置：`MINERU_API_TOKEN`（必填）、`MINERU_POLL_INTERVAL`（默认 3s）、`MINERU_POLL_TIMEOUT`（默认 300s）
- **兼容性**：显式创建失败时自动降级为轮询模式；本地截取失败自动回退完整上传；pymupdf 不可用时退化为完整上传

`rag_search` 的返回结果包含来源标注（教材名 + 章节名），知识图谱概念附带关联邻居节点。

### 4.4 文档处理层

**文件**：`src/documents/parser.py`、`chunker.py`

**解析器** (`parser.py`)：
- PDF 四层 fallback：`pymupdf 嵌入式文字` → `本地 Tesseract OCR` (250 DPI, 6 线程并行) → `VLM TOC 扫描` (qwen-vl-max, 提取章节标题 + 页码) → `MinerU 云 API` (最后手段)
- **VLM 目录提取**：单次调用 20 页 150 DPI ≈ 28K tokens，prompt 定向搜索目录页。返回 "章节标题 | 页码" 格式，自动计算各章页码范围 (第一章 p1-35, 第二章 p36-66…)
- **两阶段分离架构**：VLM 做视觉识别（读目录），DeepSeek 做文本理解（提取章节结构），Tesseract 做正文 OCR。各模型做各自擅长的事
- **知识处理定向 OCR**：根据章节页码范围只 OCR 该章页面（如第一章仅 35 页 ~30s），而非全量 376 页
- **增量索引**：`vs.index_chunks` 按内容 hash 去重追加，导入新章节不删除已有数据
- **导入状态持久化**：章节 `imported` 状态查询 ChromaDB 元数据，重启不丢失
- 支持 TXT / MD / DOCX
- `max_pages` + `page_range` 参数灵活控制解析范围

**分块器** (`chunker.py`)：
- 策略：段落拆分 → 合并至接近 chunk_size → 超长块按句子再拆分
- 重叠：保留 overlap 字符的连接上下文
- 分句正则可同时处理中英文标点（`[。！？.!?]`）

**向量存储** (`qa.py: DocumentVectorStore`)：
- 使用 `CrossLingualEmbeddingFunction`（基于 Qwen3-Embedding-0.6B via sentence-transformers）
- **延迟加载（Lazy Loading）**：模型不在应用启动时下载/加载，而是在首次文档索引或搜索时才触发。启动秒级完成，不因 HuggingFace 网络问题卡死
- 加载时自动重试 + 详细错误提示（含内网镜像和 ModelScope 预下载方案）
- **增量索引**：新 chunks 按内容 hash 去重，只追加不重建 collection。`remove_document()` 按文档名精准删除（`where={"doc_filename": ...}`），不影响其他文档
- `_ensure_collection()` 启动时从 ChromaDB 持久化数据恢复内存状态（`_all_texts`、`_all_metas`、`_content_hashes`），重启后索引不丢失
- 支持 `EMBEDDING_MODEL_PATH` 环境变量指向本地预下载模型
- **场景**：未上传资料的纯问答场景完全不需要模型，零开销

### 4.5 知识图谱层

**文件**：`src/knowledge/graph.py`

SQLite 自建知识图谱，两张表：
- `concepts`：概念节点（name, description, category, source_chunk_id, doc_filename）
- `relations`：关系边（source_id → target_id, relation_type）

关系类型：`prerequisite_of`、`part_of`、`example_of`、`related_to`、`leads_to`

支持：批量写入、文本搜索、按文档过滤搜索、邻居查询（双向 JOIN）

### 4.6 记忆/存储层

**文件**：`src/memory/store.py`

**MemoryStore** — SQLite + Qdrant 混合存储：

| 表/集合 | 用途 | 存储 |
|---|---|---|
| `search_cache` | 搜索结果缓存（7 天 TTL） | SQLite + Qdrant |
| `plan_cache` | 研究计划缓存 | SQLite + Qdrant |
| `sessions` | 会话元数据 | SQLite |
| `session_messages` | 对话消息历史 | SQLite |

采用 **精确匹配（SQLite）优先，语义搜索（Qdrant）补充** 的缓存策略。Qdrant 语义同步读取已因跨洲延迟禁用，当前退化为 SQLite-only。

### 4.7 监控层

**文件**：`src/monitoring/`（`__init__.py`、`store.py`、`reporter.py`）

- 通过 LangChain Callback 自动拦截所有 LLM 调用（`import src.monitoring` 时自动激活，需 `MONITORING_ENABLED=true`）
- 通过 monkey-patching `ChatOpenAI.__init__` 实现零侵入注入
- 记录：token 用量（prompt/completion/total）、模型名、调用类型、耗时
- 存储：`storage/token_usage.db`（WAL 模式、线程安全）
- API：`/api/monitoring/tokens/stats`、`/api/monitoring/tokens/recent`
- LangSmith 追踪：`.env` 中设置 `LANGCHAIN_TRACING_V2=true`（2026-07-09 修复前导空格导致未激活的 bug）

### 4.8 Prompt 层

**文件**：`src/prompts/`（`qa.py`、`chapterizer.py`、`extractor.py`）

所有 prompt 模板集中管理，便于调优：

| Prompt | 用途 |
|---|---|
| `SYSTEM_PROMPT` | 答疑老师系统角色 |
| `SIMPLE_CLASSIFY_PROMPT` | 简单问题分类器 |
| `ANSWERER_THINK_RAG_PROMPT` | Round 1-2 搜索策略（RAG only） |
| `ANSWERER_THINK_WEB_PROMPT` | Round 3 搜索策略（RAG + Web） |
| `ANSWERER_SYNTHESIS_PROMPT` | 回答生成（多教材对比、来源标注） |
| `REVIEWER_PROMPT` | 审核老师评判标准 |
| `CHAPTER_DETECTOR_PROMPT` | 章节检测 |
| `CHAPTER_REVIEWER_PROMPT` | 章节审核 |
| `EXTRACTOR_SYSTEM_PROMPT` | 知识提取 |

### 4.9 前端 ✅ Vue 3 重构 (2026-07-09)

**技术栈**：Vue 3 (Composition API) + Vite + Pinia + Vue Router + marked + KaTeX

```
frontend/
├── package.json          # 构建脚本: dev / build / preview
├── vite.config.js        # Vite 配置 (代理 /api → FastAPI)
├── index.html            # 入口 HTML
└── src/
    ├── main.js           # createApp + Pinia + Router
    ├── App.vue           # 根组件
    ├── router/index.js   # /qa + /docs 路由 (懒加载)
    ├── stores/           # Pinia: app.js (主题/Toast/Loading), chat.js, docs.js
    ├── composables/      # useAPI.js (fetch 封装), useMarkdown.js, useSSE.js
    ├── components/       # 12 个 SFC: NavBar, QAPage, DocsPage, ChatMessage, ...
    └── assets/style.css  # 全局 CSS 变量 + 深色/浅色主题
```

**特性**：
- 响应式数据绑定（Pinia store → 视图自动更新）
- SPA 路由（`/qa` / `/docs`，懒加载代码分割）
- Vite HMR 热重载开发，生产构建压缩 + 摇树
- npm 管理依赖（marked、KaTeX 打包进构建产物，离线可用）
- **章节检测**：per-file 独立检测状态（切换文件不互相干扰）+ 检测进行中可取消（AbortController）+ 后端 `asyncio.gather` 并发
- SSE 流式进度（章节检测 / 知识处理 / 问答）
- 向后兼容：FastAPI 自动检测 `frontend/dist/` 优先，fallback 到旧版

---

## 5. 两条核心数据流

### 5.1 文档导入流程

```
用户上传文件
    │
    ▼
POST /api/files/upload
    │
    ▼
POST /api/chapters/detect  ←── SSE 流式进度（多文件并发）
    │
    ├── 解析前 20 页 (parse_document max_pages=20)
    │   └── 空内容 → 返回扫描版 PDF 提示，跳过 LLM
    ├── ChapterizerAgent.detect_all()
    │   ├── LLM 检测目录 (CHAPTER_DETECTOR_PROMPT)
    │   ├── 启发式快判 (_fast_check_pass) → 通过则跳过 LLM 审核
    │   └── LLM 审核 (CHAPTER_REVIEWER_PROMPT，仅快判未通过时)
    └── 缓存章节到内存 + SQLite
    │
    ▼
用户选择章节 → POST /api/knowledge/process  ←── SSE 流式进度
    │
    ├── Step 1: 全量解析文档 (parse_document)
    ├── Step 2: 按章节切割 (_split_by_meta)
    ├── Step 3: 分块 (chunk_document)
    ├── Step 4: 向量索引 (DocumentVectorStore.index_chunks → ChromaDB)
    └── Step 5: 知识提取 (extract_full_document_async → KnowledgeGraph)
```

### 5.2 问答查询流程

```
用户提问
    │
    ▼
POST /api/chat  ←── SSE 流式回复
    │
    ├── 保存用户消息到 session
    ├── 加载 chat_history
    ├── 创建 QASystem
    │
    └── agent.answer(question, doc_filter, chat_history)
        │
        ├── QuestionRouter: 难度分类 + 输出 decomposition 种子
        │
        ├── trivial → direct_answer (rounds=0)
        │
        ├── moderate → DirectSolver
        │   └── QueryRewriter → rag_search × N → synthesis
        │
        └── complex → Planner → Executor → Reflector loop (≤3 轮)
            │
            ├── PLAN: Planner(seed_decomposition=Router种子) 🆕
            │   └── 输出 sub_questions (含 depends_on 依赖字段)
            │
            ├── EXECUTE: Executor 拓扑分轮执行 🆕
            │   ├── Round 1: depends_on=[] 的子问题并发搜索
            │   ├── Round 2: 依赖前轮的子问题，query 被上下文 enrich 后搜索
            │   └── Round N: ...
            │
            ├── SOLVE: Planner 综合所有搜索结果 → 生成回答
            │
            └── REFLECT: Reflector 结构化审核
                ├── SUFFICIENT → 返回回答
                └── INSUFFICIENT → 反馈 (gaps + suggested_queries) → 回到 PLAN
    │
    ▼
流式推送回复 → 保存助手消息 → 返回
```

---

## 6. 现有设计模式（亮点）

| # | 模式 | 位置 | 评价 |
|---|---|---|---|
| 1 | **Tool Registry** | `src/tools/__init__.py` | 工具可插拔，新增工具只需 `register_tool()` |
| 2 | **Prompt Centralization** | `src/prompts/` | 所有 prompt 独立文件，调优和 A/B 方便 |
| 3 | **5-Agent 三阶层路由** | `src/agents/qa/` | Router→Solver/Planner+Executor+Reflector，按难度自适应 LLM 调用 |
| 4 | **SSE Streaming** | `src/api.py` | 三个长耗时操作全部 SSE 推送进度 |
| 5 | **Configuration from Env** | `src/config.py` | Pydantic model + `from_env()`，12-factor |
| 6 | **Mock-First Testing** | `tests/steps/` | BDD + MagicMock，零 API 消耗的回归测试 |
| 7 | **Graceful Degradation** | `parser.py`, `web_search.py` | PDF 三层 fallback，工具未配置返回提示不崩溃 |
| 8 | **Chapter-Aware Batching** | `extractor.py` | 知识提取 batch 不跨章节边界，避免概念归属混乱 |
| 9 | **Heuristic Fast-Pass** | `chapterizer.py` | 格式规范的文档跳过 LLM 审核，省一次 API 调用 |
| 10 | **Markdown + LaTeX 渲染** | `frontend/js/app.js` | 三阶段管线：保护 → Markdown → KaTeX |
| 11 | **TOC-Skip 章节定位** | `chapterizer.py` | 多策略匹配 + `_find_skip_toc()` 自动识别并跳过目录区域，避免正文章节定位到目录页 |
| 12 | **Cross-Lingual 向量检索** | `qa.py` | Qwen3-Embedding-0.6B 替换 TF-IDF，中文查询可跨语言检索英文文档 |
| 13 | **Lazy Embedding 启动** | `qa.py` | 模型延迟到首次索引/搜索时加载，启动不卡死，纯问答场景零开销 |
| 14 | **Dual-Mode Token 监控** | `src/monitoring/` + LangSmith | 本地 SQLite + 云端 LangSmith 双轨 token 追踪 |
| 15 | **多跳依赖执行** 🆕 | `executor.py` | 子问题按 `depends_on` 拓扑分轮，前轮结果 enrich 后轮 query | 
| 16 | **Router 分解种子复用** 🆕 | `orchestrator.py` → `planner.py` | Router 的 decomposition 传入 Planner 做种子，避免重复分解 |
| 17 | **图片 PDF 全链路支持** 🆕 | `parser.py` + `chapterizer.py` | VLM 目录提取+本地 OCR+页码范围定向导入，图片 PDF 章节检测→知识处理全链路打通 |
| 18 | **VLM 快速通道** 🆕 | `chapterizer.py` | VLM pre-detected 章节跳过 LLM 检测阶段，直接 split + review |
| 19 | **冒号归一化匹配** 🆕 | `chapterizer.py:_collapse_ws` | OCR 文本空格与 `_normalize_title` 冒号差异归一化匹配

---

## 7. 现有问题清单

### 7.1 🔴 严重

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 1 | **全局单例泛滥** | `_agent`, `_vector_store`, `_knowledge_graph`, `_store`, `_qdrant_client` 等模块级全局变量 | 测试隔离困难，多租户不可行 |
| 2 | ~~**错误处理粗糙**~~ ✅ 已修复 (2026-07-09) | `ToolResult` 已加入结构化错误模型，所有 `except: pass` 已消除 | — |

### 7.2 🟡 中等

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 4 | ~~**同步/异步混用**~~ ✅ 已修复 (2026-07-09) | `BaseAgent` + 统一 `ainvoke`，所有 Agent 异步化 | — |
| 5 | ~~**前端状态管理原始**~~ ✅ 已修复 (2026-07-09) | 单例 `State` + `Events` EventBus + 组件化 | — |
| 6 | ~~**前端无模块化**~~ ✅ 已修复 (2026-07-09) | 拆分为 10 文件模块，app.js 777行→40行 | — |
| 7 | **缺少 API 版本化** | 所有端点 `/api/...` 无版本前缀 | 未来 breaking change 无法平滑过渡 |
| 8 | **文档处理是同步阻塞 SSE** | 用户需保持连接，大文件可能超时 | 体验差，不适合生产环境 |

### 7.3 🟢 轻微

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 9 | **前端依赖 CDN** | marked.js、KaTeX 从 jsDelivr 加载 | 离线不可用，加载慢 |
| 10 | ~~**测试覆盖不足**~~ ✅ 已修复 (2026-07-09) | 27 个 pytest 单元测试 + BDD 集成测试 | — |
| 11 | **OCR 异步 API 延迟** | MinerU 异步 API（上传→轮询→下载），单文件 10-30s | 扫描 PDF 需等待，但比本地 OCR 轻量 |
| 12 | **无请求日志/追踪** | 仅 LangSmith 追踪 LLM 调用 | API 层问题难以排查 |

---

## 8. 优先级路线图

### 🔴 P0 — 立即处理（降低风险）

| # | 改进项 | 工作量 | 收益 |
|---|---|---|---|
| 1 | ~~API 路由拆分~~ ✅ 已完成 | — | `src/api/` 已拆为 6 个 domain router |
| 2 | ~~结构化错误模型~~ ✅ 已完成 | — | `ToolResult` 增加 `ToolErrorType` 枚举 + `error`/`error_detail` 字段 |
| 3 | ~~消除 `except: pass`~~ ✅ 已完成 | — | 所有静默异常已替换为 `logger.warning/debug` + 上下文信息 |

### 🟡 P1 — 短期目标（提升工程质量）✅ 全部完成 (2026-07-09)

| # | 改进项 | 工作量 | 收益 |
|---|---|---|---|
| 4 | ~~依赖注入 (AppContext)~~ ✅ 已完成 | — | `src/context.py` + `deps.py` 重构，集中管理 |
| 5 | ~~前端模块化拆分~~ ✅ 已完成 | — | 拆为 10 个文件 (state/events/lib/components)，app.js 从 777 行 → 40 行 |
| 6 | ~~统一异步接口~~ ✅ 已完成 | — | `BaseAgent` + `_make_llm()` + `ainvoke` 全程异步 |
| 7 | ~~单元测试覆盖核心模块~~ ✅ 已完成 | — | 27 个 pytest 测试 (config/graph/chunker) |
| 8 | ~~QA 多跳依赖执行~~ ✅ 已完成 | — | Executor 拓扑分轮 + 上下文 enrich，子问题不再全并发 |
| 9 | ~~Router 分解种子复用~~ ✅ 已完成 | — | Planner 接收 Router 的 decomposition 种子，避免重复分解 |

### 🟢 P2 — 中期目标（提升产品质感）

| # | 改进项 | 工作量 | 收益 |
|---|---|---|---|
| 8 | Prompt 版本化 + A/B | 2-3h | 简历亮点，工程化思维 |
| 9 | 前端构建工具化 (Vite) | 2-3h | 加载性能 + 离线可用 |
| 10 | 任务队列化 | 4-6h | 大文件处理体验 |

### 🔵 P3 — 长期目标（生产级就绪）

| # | 改进项 | 工作量 | 收益 |
|---|---|---|---|
| 11 | API 版本化 (`/api/v1/...`) | 1h | 平滑升级 |
| 12 | 请求日志 + 结构化日志 | 2h | 运维可观测性 |
| 13 | 认证/授权层 | 4-6h | 多用户支持 |
| 14 | OCR 异步化 | 2-3h | 扫描 PDF 处理速度 |

---

> 本文档描述项目现状和设计。开发者编码规范见 [RULE.md](./RULE.md)。
