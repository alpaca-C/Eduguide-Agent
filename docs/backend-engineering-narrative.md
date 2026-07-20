cl# Eduguide-Agent：Python 后端工程视角

> 本文档从**后端工程**角度复述这个项目，用于简历改写。
> AI/LLM 部分弱化为"接入的服务"，重点放在架构、数据、并发、工程实践上。

---

## 一、项目概览

一个基于 **FastAPI** 的文档处理与智能检索平台。用户上传 PDF/TXT/MD/DOCX 等文档后，系统自动解析、分块、建立多级索引，提供高性能的混合检索与问答能力。

**一句话**：大型文档的 ETL 流水线 + 多数据库混合检索引擎 + RESTful API 服务。

---

## 二、系统架构

```
┌──────────────┐     ┌───────────────────────────────────────┐
│   Vue 3 SPA  │────▶│           FastAPI REST API            │
│   (5173)     │     │           (7860)                      │
└──────────────┘     │                                       │
                     │  ┌─────────────────────────────────┐  │
                     │  │  路由层 (6 个独立 Router)        │  │
                     │  │  chat / files / chapters /      │  │
                     │  │  knowledge / sessions / health   │  │
                     │  └──────────────┬──────────────────┘  │
                     │                 │                      │
                     │  ┌──────────────▼──────────────────┐  │
                     │  │  依赖注入容器 (AppContext)       │  │
                     │  │  统一管理所有共享服务实例        │  │
                     │  └──────────────┬──────────────────┘  │
                     │                 │                      │
                     │  ┌──────────────▼──────────────────┐  │
                     │  │       数据层                     │  │
                     │  │  ┌──────┐ ┌──────┐ ┌─────────┐  │  │
                     │  │  │SQlite│ │Chroma│ │FTS5     │  │  │
                     │  │  │会话/  │ │向量库│ │全文索引 │  │  │
                     │  │  │章节   │ │(稠密)│ │(稀疏)   │  │  │
                     │  │  └──────┘ └──────┘ └─────────┘  │  │
                     │  │       + 知识图谱 (KG)            │  │
                     │  └─────────────────────────────────┘  │
                     └───────────────────────────────────────┘
```

### 路由设计

| Router | 职责 |
|--------|------|
| `/api/files` | 文件上传、列表、删除 |
| `/api/chapters` | 章节检测（异步 SSE）、缓存、加载 |
| `/api/knowledge` | 文档处理流水线、索引、KG 提取 |
| `/api/chat` | SSE 流式问答 |
| `/api/sessions` | 会话管理（CRUD） |
| `/api/health` | 健康检查 |

---

## 三、多数据库混合检索架构

这是项目的核心设计亮点。单一检索方式（纯关键词或纯向量）在专业文档场景下存在盲区，因此设计了三路并联检索 + 融合排序：

```
用户查询
    │
    ├──▶ ChromaDB 稠密向量检索 (语义匹配，跨语言)
    ├──▶ SQLite FTS5 稀疏关键词检索 (BM25，精确术语)
    └──▶ 知识图谱扩展检索 (概念关系，召回关联内容)
           │
           ▼
      RRF 融合排序 (Reciprocal Rank Fusion)
           │
           ▼
       最终结果集
```

### 为什么三路？

- **稠密向量**：解决"用户用白话问、文档是学术术语"的语义鸿沟
- **FTS5 关键词**：解决专有名词、公式编号等向量的弱项
- **知识图谱**：用户问"A 和 B 的关系"，即使文档里没同时出现也能召回

### RRF 融合

不依赖单一评分尺度，而是对各路结果的排名做倒数加权求和，解决不同检索器的分数量纲不一致问题。

---

## 四、异步文档处理流水线 (ETL)

这是系统中计算最重的部分，设计为 **SSE 流式长任务**：

```
文件上传 → 章节自动检测 → 用户勾选章节 → 处理流水线启动
                                              │
                   ┌──────────────────────────┘
                   ▼
            Step 1: 文档解析 (PDF OCR / TXT / MD / DOCX)
                   │
                   ▼
            Step 2: 文本分块 (段落边界 + 滑动窗口, chunk_size=800, overlap=150)
                   │
                   ▼
            Step 3: 双索引写入 (ChromaDB 向量 + FTS5 全文)
                   │
                   ▼
            Step 4: 知识图谱提取 (实体 + 关系)
                   │
                   ▼
              全部完成 → SSE 推送最终结果
```

### 工程要点

- **并发控制**：章节检测使用 `asyncio.Semaphore` 限制并发 LLM 调用数
- **SSE 实时进度**：每个步骤完成后推送 `{type: "progress", stage: "...", pct: 45}`
- **增量索引去重**：`content_hashes` 集合避免重复分块写入
- **断点续跑**：章节检测结果持久化到 SQLite，用户可多次进入而不丢失

---

## 五、并发与异步设计

- **asyncio 原生异步**：FastAPI 的 `async def` + `asyncio.gather` 做多文件并行检测
- **线程池混用**：CPU 密集型操作（文档解析、OCR、分块）通过 `asyncio.to_thread` 放入线程池，避免阻塞事件循环
- **Semaphore 限流**：同时检测的文档数由 `CHAPTER_DETECT_CONCURRENCY` 环境变量控制，防止外部 API 过载
- **AbortController**：前端可随时取消正在进行的检测任务

---

## 六、依赖注入 & 工程化

### AppContext 容器

```python
@dataclass
class AppContext:
    config: Configuration        # 环境配置
    memory_store: MemoryStore    # 会话存储 (SQLite)
    knowledge_graph: KnowledgeGraph  # 知识图谱
    chapter_agent: object        # 章节检测服务 (懒加载)
    vector_store: object         # 向量存储服务 (懒加载)
```

- 模块级单例，启动时一次性初始化，各 Router 通过 `get_context()` 获取
- 测试时可独立构造新的 AppContext，天然支持测试隔离

### 配置管理

- `Configuration.from_env()` 从 `.env` 加载全部配置
- 支持 `memory_db_path`、`chunk_size`、`chunk_overlap`、并发数等参数化

### Hook 系统

`src/harness/` 提供 before/after 钩子，可在 LLM 调用或工具调用前后插入日志、权限校验、速率限制等横切逻辑。

---

## 七、API 设计要点

### SSE 流式响应

两个核心接口使用 SSE（非 WebSocket）：

1. **问答接口** `POST /api/chat`：逐 token 推送回复
2. **处理接口** `POST /api/knowledge/process`：推送处理进度

选择 SSE 而非 WebSocket 的原因：单向推送场景，HTTP 兼容性好，无需额外连接管理。

### 章节删除（粒度控制）

`DELETE /api/knowledge/chapters/{label}` 精确删除单个章节的索引数据，而非整个文档。涉及：

- ChromaDB metadata 过滤删除
- FTS5 条件删除
- KG 文档级清理
- 内存缓存同步清理

### 文档级清库

`DELETE /api/knowledge/documents/{filename}` 清除指定文档的全部索引数据，用于重新检测章节的场景。

---

## 八、存储设计

| 存储 | 引擎 | 用途 |
|------|------|------|
| 会话 & 消息 | SQLite (`storage/memory.db`) | 多轮对话历史 |
| 章节元数据 | SQLite (`storage/memory.db`) | 检测结果持久化 |
| 稠密向量索引 | ChromaDB (`data/chroma/`) | 语义检索 |
| 稀疏关键词索引 | SQLite FTS5 (`data/chroma/fts_index.db`) | BM25 关键词检索 |
| 知识图谱 | SQLite (`src/knowledge/graph.py`) | 实体-关系存储 |
| 上传文件 | 文件系统 (`uploads/`) | 原始文档 |

**设计决策**：选择 SQLite 而非 PostgreSQL 是因为轻量部署需求（单机即可运行，无需额外数据库服务），且数据量在 SQLite 承受范围内（百万级 chunks）。

---

## 九、可观测性

- **结构化日志**：Python `logging` 模块，统一格式 `[LEVEL] name: message`
- **Token 用量监控**：`src/monitoring/` 追踪每次 LLM 调用的 token 消耗
- **健康检查**：`GET /api/health` 返回服务状态

---

## 十、最适合在简历中强调的后端亮点

1. **多数据库混合检索架构**：ChromaDB + FTS5 + KG，RRF 融合排序
2. **异步 ETL 流水线**：asyncio + SSE 实时进度，Semaphore 并发控制
3. **依赖注入设计**：AppContext 容器，测试友好
4. **API 设计**：6 个独立 Router，RESTful 风格，SSE 流式响应
5. **数据一致性**：多存储系统间的级联删除与缓存同步
6. **前后端分离**：FastAPI + Vue 3，Vite 代理配置

---

## 十一、面试准备：Java → Python 知识迁移

> 你不需要"学 Python 后端"——你需要的是**用 Python 的术语讲你已经做过的事**。

### 11.1 Java vs Python 核心概念对照表

这是面试中最加分的东西：你能说清楚两种语言的对应关系，说明你是"选合适的工具"而非"只会一个"。

| 概念 | Java | Python |
|------|------|--------|
| Web 框架 | Spring Boot / Spring MVC | FastAPI / Flask / Django |
| 依赖注入 | `@Autowired` / `@Bean` | FastAPI `Depends()` / 手动容器（如本项目 `AppContext`） |
| ORM | MyBatis-Plus / JPA / Hibernate | SQLAlchemy ORM / 原生 sqlite3 |
| 异步编程 | `CompletableFuture` / `@Async` / 线程池 | `asyncio` / `async`-`await` / 事件循环 |
| 并发控制 | `synchronized` / `ReentrantLock` / `Semaphore` | `threading.Lock` / `asyncio.Lock` / `asyncio.Semaphore` |
| 序列化/校验 | Jackson / `@Valid` / `@NotNull` | Pydantic / type hints + Validation |
| 包管理 | Maven (`pom.xml`) / Gradle | pip (`requirements.txt`) / poetry / uv |
| 配置管理 | `application.yml` / `@Value` | `.env` + `os.environ` / pydantic-settings |
| 中间件/拦截器 | Filter / Interceptor / AOP | FastAPI Middleware / `Depends()` |
| 定时任务 | XXL-Job / `@Scheduled` | Celery / APScheduler / cron |
| 消息队列 | RabbitMQ / Kafka | Celery + Redis / arq |
| 测试 | JUnit / Mockito | pytest / unittest.mock |
| 接口文档 | Swagger / Knife4j | FastAPI 自动生成 OpenAPI（开箱即用） |

### 11.2 Python 基础补漏清单（一周够）

这些是面试中被问到的概率最高的 Python 基础问题，每个背后都是你 Java 经验可以直接映射的。

#### GIL（全局解释器锁）

**一句话**：CPython 解释器同一时刻只允许一个线程执行 Python 字节码。

**对你项目的影响**：
- IO 密集型（网络请求、文件读写、SSE 推送）→ GIL 不影响，`async`-`await` 高效
- CPU 密集型（文档解析、OCR、分块）→ 必须用 `asyncio.to_thread()` 放入线程池，或 `ProcessPoolExecutor`
- 这也是为什么你的代码里 `parse_document` 和 `chunk_document` 都套了 `asyncio.to_thread()`

**常见追问："那 Python 的线程有什么用？"**
→ IO 密集型场景下，线程在等待 IO 时会释放 GIL，其他线程可以执行。所以 Python 多线程适合做并发 IO，不适合做并行计算。

#### async/await 与事件循环

**你项目里的实际使用**：

```python
# 并发执行多个文档的章节检测
tasks = [_detect_one_file(...) for ... in resolved]
results = await asyncio.gather(*tasks)

# 线程池执行 CPU 密集任务
full_doc = await asyncio.to_thread(parse_document, full_path)
```

**核心概念**：
- `async def` 定义协程函数，调用它返回一个 coroutine 对象，不会立即执行
- `await` 挂起当前协程，让出控制权给事件循环去执行其他协程
- `asyncio.gather()` 并发执行多个协程，等全部完成
- `asyncio.create_task()` 创建后台任务，不等待（类似 `new Thread().start()`）
- `asyncio.Semaphore(n)` 限制同时执行的协程数 — 等价于 Java 的 `new Semaphore(n)`

**常见追问："事件循环是什么？"**
→ 一个单线程调度器，维护一个任务队列。协程 await 时挂起，事件循环切换到另一个可执行的协程。类似 Java NIO 的 Selector 但更上层。

#### Pydantic 类型校验

**你项目里的使用**（`src/api/schemas.py`）：

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    session_id: str = ""
    doc_filter: list[str] = []
    tutor_mode: bool = False
```

等价于：Spring 的 `@RequestBody @Valid ChatRequest` + `@NotNull` / `@NotEmpty`。

FastAPI 自动根据 Pydantic model 生成 JSON Schema → 自动生成 Swagger 文档。Spring 需要额外引入 Swagger 注解。

#### 装饰器

**一句话**：`@decorator` 本质是 `func = decorator(func)`，一个接收函数、返回新函数的高阶函数。

Java 对照：AOP 的 `@Around` 通知，但 Python 装饰器更轻量，不需要编译时织入。

#### `__init__.py` 和包导入

- 有 `__init__.py` 的目录是 Python package，可以 `import`
- `__init__.py` 为空 → 仅标记包；可以在里面写 `from .module import X` 做 re-export
- 你项目的 `src/skills/__init__.py` 就是一个好例子：注册表 + re-export

---

## 十二、项目追问准备（面试话术）

> 这些是你简历上写了这个项目后，面试官最可能追问的点。提前准备好回答，不要现场编。

### Q1："为什么用 SQLite 不用 PostgreSQL？"

**话术**：这个项目的定位是轻量级文档检索工具，目标用户在自己的机器上就能跑。SQLite 是零配置嵌入式数据库，不需要额外安装和运维。数据量我评估过——百万级文本 chunks，SQLite 的查询性能完全够用。如果是团队协作或更大规模的生产环境，我会换成 PostgreSQL + pgvector 插件替代 ChromaDB，统一到 PG 做向量检索和全文检索。

### Q2："为什么用 SSE 不用 WebSocket？"

**话术**：数据流向决定的。这个项目的所有推送都是服务端到客户端的单向流——问答的逐字输出、文档处理的进度更新。客户端只需要接收，不需要往流里发消息。SSE 基于标准 HTTP，不需要额外的握手协议和连接管理，CDN 和反向代理兼容性也更好。WebSocket 适合像聊天室、协同编辑这种需要客户端频繁主动发消息的场景，这里用不到。

### Q3："ChromaDB 内部原理是什么？为什么不直接用向量数据库？"

**话术**：ChromaDB 并不是一个真正的数据库，它是一个嵌入式的向量检索库。底层索引用的是 hnswlib——一种基于图的近似最近邻搜索算法。元数据过滤层用 SQLite 实现的。它解决的问题很简单：给你一个向量，找出最相似的 k 个。选它的原因是和 SQLite 一样——零配置、嵌入式、pip install 就行。如果需要更高性能，可以换成 Milvus 或者 Qdrant，但部署复杂度会上去。

### Q4："多数据库（ChromaDB + FTS5 + KG）怎么保证数据一致性？"

**话术**：坦白说，当前实现是最终一致性，不是分布式事务。删除操作是先清 ChromaDB，再清 FTS5，再清 KG——中间如果某一步失败，有日志记录，但不会回滚。这是因为项目的并发访问量不大，且删除操作本身是低频的。如果是生产级高并发场景，方案有两个：一是全换到 PostgreSQL（向量用 pgvector，全文用 `tsvector`，知识图谱用关联表），二是引入消息队列做最终一致性的异步保障。

### Q5："你这个 Agent 管道的 Router / Planner / Executor / Reflector 是什么意思？"

**话术**：这是一个请求分级的处理策略，借鉴了 LLM Agent 里的 ReAct 模式，但本质上是一个条件路由 + 任务编排的后端逻辑：

- **Router**：判断请求复杂度，分流到不同处理路径。类比网关层的限流/路由。
- **Planner**：把复杂问题拆成子问题列表。类比 MapReduce 的 Map 阶段。
- **Executor**：拓扑排序后并发执行子问题。类比 DAG 任务调度 + 线程池。
- **Reflector**：审核结果，不合格则回退重试，最多 3 轮。类比重试策略 + 熔断。

你不需要说它是 AI——它就是**一个带重试机制的分级任务处理管道**。

### Q6："FastAPI 比 Flask 好在哪里？为什么选它？"

**话术**：两个原因——原生异步支持和自动 OpenAPI 文档。Flask 是同步框架，要异步需要额外装 Quart 或用 `async_to_sync` 的 hack。FastAPI 从设计第一天就是 async-first，和 asyncio 生态原生配合——我的文档处理流水线需要同时支持异步 SSE 推送和线程池 CPU 密集操作，FastAPI 的 `async def` + `asyncio.to_thread` 这个组合比 Flask 干净得多。另外 FastAPI 基于 Pydantic 自动生成 Swagger 文档，省去了手写接口文档的成本。

### Q7："这个项目有什么技术难点？你遇到的最大的挑战是什么？"

**话术**：
1. **多路检索的融合排序**：三路检索器的分数量纲不同（ChromaDB 返回的是余弦相似度 0-1，FTS5 返回的是 BM25 分数，知识图谱返回的是边权重），直接加权是无意义的。我用 RRF（Reciprocal Rank Fusion）——只关心排名不关心分数绝对值，对每路结果的名次做倒数求和再排序，问题就解决了。
2. **异步长任务的并发控制**：文档处理可能是几分钟的长任务，同时用户可能触发多个文档的检测。做了两层控制：前端 AbortController 可随时取消，后端 asyncio.Semaphore 限制同时处理的文档数，防止外部 API 过载。
3. **多存储系统的级联删除**：删除一个章节需要同步清 ChromaDB（metadata 过滤）+ FTS5（条件 DELETE）+ KG（按文档名删）+ 内存缓存。前端展示的状态必须和底层一致。

---

## 十三、建议的准备路线

| 阶段 | 内容 | 时间 |
|------|------|------|
| 1 | 通读 FastAPI 官方 User Guide（你项目里已经用了大部分，主要是系统化） | 半天 |
| 2 | 对着本文档和你的代码，能口头讲清楚每个模块的设计决策 | 一天 |
| 3 | 看一个 asyncio 的深入文章，理解事件循环 + 协程调度机制 | 半天 |
| 4 | 准备 7 个 Q&A（上面帮你列好了），用自己的话复述，不要背诵 | 一天 |
| 5 | 用 Pydantic + FastAPI 裸写一个 TODO API（CRUD），脱离项目代码独立写 | 半天 |

**核心原则**：面试官不会考你 Python 语法细节。他们关心的是——这个人用 Python 解决了一个什么后端问题？为什么用这些技术选型？决策过程是怎样的？这些问题你能回答好，Python 后端就过了。
