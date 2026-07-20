# Eduguide-Agent — 简历项目文档

> 基于 LLM 的文档知识图谱问答系统 | 2026.07 | 个人项目
>
> GitHub: https://github.com/alpaca-C/Eduguide-Agent
>
> **一句话描述**：上传教材/论文，自动构建知识图谱和向量索引，支持多跳推理的智能问答系统。

---

## 1. 项目含金量评估

### 结论：**足够作为简历核心项目**

| 维度 | 评价 | 说明 |
|---|---|---|
| **技术深度** | ⭐⭐⭐⭐ | 自研 5-Agent 多跳推理架构 + 混合检索(RRF) + 知识图谱 + SSE流式 + 三路PDF解析fallback |
| **工程化程度** | ⭐⭐⭐⭐ | CI/CD + 318个测试(50%覆盖率) + 模块化拆分 + 依赖注入 + 结构化错误模型 |
| **代码规模** | ⭐⭐⭐⭐ | 7409行Python + Vue3前端，17个模块，13个API端点 |
| **完整度** | ⭐⭐⭐⭐ | 从文档解析→章节检测→向量索引→知识图谱→问答的全链路闭环 |
| **简历区分度** | ⭐⭐⭐⭐⭐ | 自研Agent架构非LangChain套壳，有技术文档和开发日志 |

### 对标分析

这个项目比90%的应届生项目强的地方：
- **不是调API套壳** — 自研了5个Agent的协调编排、多跳依赖拓扑排序、三阶层难度路由
- **有完整的工程闭环** — 测试/CI/文档/错误处理/配置管理，不是一次性demo
- **解决了真实问题** — 扫描版PDF处理、中英文混合检索、OCR纠错，不是toy project
- **架构演进有迹可循** — DEVLOG记录了从ReAct→五Agent架构的演进决策

---

## 2. 项目概述（简历用）

### 2.1 一句话版

设计并实现了一个基于多Agent协作的文档知识图谱问答系统，支持教材/论文的自动解析、知识提取和复杂多跳推理问答，从零搭建了完整的工程化体系（CI/CD + 318个测试 + 50%覆盖率）。

### 2.2 段落版（适合简历项目经历）

**Eduguide-Agent — 智能文档问答系统** | Python, FastAPI, LangChain, Vue3

- 设计并实现了**五Agent三阶层路由架构**（Router→Solver/Planner→Executor→Reflector），按问题难度自动选择1~7次LLM调用，相比ReAct方案减少30%+的API消耗
- 自研**多跳依赖执行引擎**：子问题按`depends_on`字段拓扑分轮，前轮结果通过QueryRewriter enrich后续查询，支持线性链/钻石形等复杂依赖拓扑
- 实现**三路混合检索+RRF融合**（ChromaDB向量 + SQLite FTS5关键词 + 知识图谱概念），中文查询可跨语言检索英文文档
- 支持**扫描版PDF全链路处理**：VLM目录检测→本地Tesseract OCR→MinerU云兜底，三层fallback策略
- 从零搭建工程化体系：模块化API路由、AppContext依赖注入、结构化错误模型、GitHub Actions CI/CD、**318个pytest测试用例（50%代码覆盖率）**
- 前端Vue3 + Vite + Pinia，SSE流式推送，支持Markdown + LaTeX数学公式渲染

### 2.3 技能关键词版

```
LLM应用开发 | Multi-Agent架构 | RAG检索增强生成 | 知识图谱 | 向量数据库(ChromaDB) 
FastAPI | LangChain | Prompt Engineering | 混合检索(RRF) | SQLite FTS5 
Python异步编程 | 依赖注入 | pytest单元/集成测试 | CI/CD | Vue3前端
```

---

## 3. 技术架构详解

### 3.1 整体架构图

```
┌──────────────┐     ┌─────────────────────────────────┐
│  Vue3 前端    │────▶│  FastAPI (13个端点, 6个Router)    │
│  SSE 流式     │     │  SSE推送 / RESTful               │
└──────────────┘     └──────────┬──────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
    ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
    │ 文档导入流水线│   │  QA Agent系统 │   │  工具层       │
    │ parse→chunk  │   │              │   │  rag_search   │
    │ →index→      │   │ Router       │   │  web_search   │
    │   extract    │   │  ├─Solver    │   │  mineru_ocr   │
    │              │   │  └─Planner   │   │              │
    │ Chapterizer  │   │     →Executor│   │ QueryRewriter │
    │  (章节检测)   │   │     →Reflector│  │              │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                   │
           └──────────────────┼───────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │        存储层                   │
              │  ChromaDB │ SQLite KG │ FTS5   │
              │  MemoryStore │ Token Monitor   │
              └───────────────────────────────┘
```

### 3.2 五Agent三阶层路由（核心亮点）

这是整个项目最有技术含量的部分：

```
用户提问
    │
    ▼
QuestionRouter (1次LLM)
    │ 输出: difficulty + decomposition种子
    │
    ├── "trivial"  ──▶ 直接回答 (0 tool calls)
    │
    ├── "moderate" ──▶ DirectSolver
    │                  rewrite → rag_search → synthesize (2-3次LLM)
    │
    └── "complex"  ──▶ Planner → Executor → Reflector (≤3轮迭代)
                        │
                        ├─ PLAN: 分解子问题，含depends_on依赖关系
                        ├─ EXECUTE: 拓扑分轮并发搜索，前轮结果enrich后轮
                        ├─ SOLVE: 综合所有结果生成回答
                        └─ REFLECT: 结构化审核
                             ├─ SUFFICIENT → 返回
                             └─ INSUFFICIENT → 反馈注入下一轮PLAN
```

**设计决策**（面试重点）：
- 为什么不用LangGraph？→ 五Agent的协调逻辑用纯Python更可控，LangGraph引入额外复杂度
- 为什么分三阶层？→ 寒暄不需要搜索，简单查找不需要分解，只有跨文档对比才走Planner循环
- Router的decomposition种子复用：避免Planner重复分解，零额外LLM调用

### 3.3 多跳依赖执行引擎

这是区别于普通RAG项目的关键：

```python
# 子问题声明依赖关系
plan = [
    {"id": 1, "question": "什么是库仑定律", "depends_on": []},
    {"id": 2, "question": "库仑定律的数学表达式", "depends_on": [1]},
    {"id": 3, "question": "库仑力与电场强度的关系", "depends_on": [1]},
    {"id": 4, "question": "电场叠加原理", "depends_on": [2, 3]},
]

# _topological_rounds() 输出:
# Round 0: [1]        — 无依赖，先执行
# Round 1: [2, 3]      — 依赖1的结果，并发执行
# Round 2: [4]         — 依赖2和3的结果
```

核心实现：
- `_topological_rounds()`: 拓扑排序分组，处理循环依赖（检测后flush为最终轮）
- `_enrich_query_with_context()`: 前轮搜索结果(≤400字)注入QueryRewriter，生成更精准的搜索词
- `_summarize_for_context()`: 压缩上下文至≤500字符，避免逐轮膨胀

### 3.4 知识提取并发管道

```python
# Phase 1: 按章节边界分批 + 并发LLM调用
batches = _build_chapter_batches(chunks, max_batch_size)  # 不跨章节边界
tasks = [_process_one_batch(...) for batch in batches]     # 每批独立prompt
results = await asyncio.gather(*tasks)                      # Semaphore控制并发

# Phase 2: 顺序合并去重
# concept_map: normalized_name → concept_id
# 跨batch同概念自动合并
```

**设计要点**：分批不跨章节边界，避免"电场"概念被错误归属于"磁场"章节。

---

## 4. HR/面试官可能深入挖掘的点

### Q1: "你这不就是调LangChain吗？有什么技术含量？"

**参考答案**：
> LangChain只用了两个组件：`ChatOpenAI`（LLM调用）和`BaseCallbackHandler`（token监控）。Agent的编排逻辑完全自研——五Agent的协调、三阶层路由、多跳依赖拓扑排序、Reflection循环控制，都是纯Python实现。之所以不用LangGraph，是因为五Agent的协调逻辑用原生async/await更直接，可以精确控制每个Agent的输入/输出格式，而LangGraph的StateGraph会引入不必要的序列化开销。

### Q2: "你的RAG和直接用LangChain的RAG有什么不同？"

**参考答案**：
> 第一，我做了**三路混合检索+RRF融合**而不是单纯的向量搜索。ChromaDB做语义匹配、SQLite FTS5做关键词精确匹配、知识图谱做概念关联扩展，三路结果用RRF(Reciprocal Rank Fusion)融合去重。这解决了纯向量搜索的"精确术语匹配不足"问题。
>
> 第二，搜索结果**按难度分层使用**——简单问题只用RAG，复杂问题RAG+KG联合，最后才允许网络搜索兜底。这是为了减少幻觉：教材里有答案的就不去网上搜。
>
> 第三，有**QueryRewriter**：把自然语言问题改写为3-5个优化搜索词（中英双语+同义词），提升FTS5和KG的召回率。这是很多RAG demo忽略的环节。

### Q3: "50%的测试覆盖率是不是太低了？"

**参考答案**：
> 这个项目有318个测试用例，覆盖率从33%起步提升到50%。核心业务逻辑的实际覆盖率远高于50%——solver 100%、graph 100%、deps 100%、reflector 93%、router_chat 89%、orchestrator 83%、planner 85%。
>
> 剩余没覆盖的主要是三类代码：1) PDF解析的OCR/VLM路径，需要真实PDF文件和外部服务(Tesseract/MinerU)；2) ChromaDB的向量存储层，需要embedding模型；3) 入口启动脚本。这些属于"测试成本高但bug风险低"的基础设施代码。50%是经过ROI权衡后的合理数字，不是做不上去。

### Q4: "你的Agent之间是怎么通信的？有没有考虑过消息队列？"

**参考答案**：
> 当前是**同步调用链**：Orchestrator持有所有子Agent的引用，直接调用子Agent的方法。每个子Agent有明确的输入(AgentInput)和输出(AgentOutput)接口，通过metadata字典传参。
>
> 这是有意为之的设计选择。Agent之间不需要解耦——Planner的输出天然是Executor的输入，Executor的输出天然是Reflector的输入。引入消息队列会增加延迟（序列化/反序列化/网络）和调试复杂度，而收益（解耦、独立部署）在当前场景下不存在。
>
> 如果需要扩展为多用户、长时间任务，我会考虑用asyncio.Queue + FastAPI BackgroundTasks做轻量任务队列，而不是直接上Redis/Celery。

### Q5: "你这个Reflection机制怎么避免无限循环？"

**参考答案**：
> 三重保护：1) **硬上限**：最大3轮Reflection循环；2) **安全兜底**：Reflector的LLM调用失败时默认返回SUFFICIENT（宁可接受不完美回答也不无限循环）；3) **启发式判断**：Reflector输出的JSON解析失败时也返回SUFFICIENT。
>
> 另外，Reflector输出的是**结构化JSON**而非自由文本：`{verdict, missing[], suggested_queries[], issues[]}`。Planner下一轮直接读取suggested_queries和missing字段，不需要再解析自然语言反馈，减少了不确定性。

### Q6: "你怎么处理扫描版PDF？市面上很多方案，你为什么选这个？"

**参考答案**：
> 这是整个项目最工程化的部分。我设计了一个**四层fallback链**：
>
> 1. pymupdf提取嵌入式文字（秒级，覆盖数字PDF）→ 失败则
> 1. 本地Tesseract OCR（250 DPI，6线程并行渲染+OCR，覆盖清晰扫描件）→ 失败则
> 1. VLM(视觉模型)扫描目录页提取章节标题+页码 → 失败则
> 1. MinerU云API（最后手段，覆盖模糊扫描件）
>
> 关键设计点是**VLM和DeepSeek分工**：VLM只做目录页的视觉识别（它擅长的），DeepSeek做全文的章节结构提取（它擅长的），Tesseract做正文OCR（它擅长的）。不是让一个模型做所有事。
>
> 另外做了**定向OCR优化**：根据VLM提取的页码范围只OCR选中章节的页面（如第一章仅35页≈30s），而非全量376页。这在知识处理环节节省了大量时间。

### Q7: "你的项目有什么可以改进的地方？"

**参考答案**（展示技术视野和自我认知）：
> 1. **任务队列化**：当前文档处理是同步SSE（用户必须保持连接），应该改为异步任务队列（提交job→轮询状态），大文件不超时
> 1. **Prompt版本化**：当前prompt是单文件，应该加版本号+A/B测试框架，方便调优
> 1. **流式Token级输出**：当前SSE按8字符分块推送，做不到token-level streaming（因为DeepSeek兼容接口的限制）
> 1. **多模态问答**：目前只能问答文本，图片/公式/表格中的信息无法检索——可以加ColPali或ColQwen做视觉embedding
> 1. **增量知识更新**：当前新文档导入后旧KG数据不更新，缺少"如果新知识跟旧知识冲突怎么办"的处理

### Q8: "如果让你重新设计这个系统，你会怎么做？"

**参考答案**：
> 核心架构不变，但两个地方会重新考虑：
> 1. **向量数据库选型**：ChromaDB适合原型但生产环境扩展性有限，会换成Milvus或Qdrant，支持分布式和更好的过滤查询
> 1. **Agent间通信**：如果未来Agent数量增长到10+，会引入一个轻量的Agent Bus（asyncio.Queue + 事件驱动），而不是现在Orchestrator直连每个子Agent
> 1. **评估体系**：当前只有功能测试，缺少RAG质量评估（如RAGAS框架的faithfulness/answer_relevancy指标），导致prompt调优靠人工判断，效率低

### Q9: "你项目中提到的RRF融合是怎么做的？"

**参考答案**：
> RRF (Reciprocal Rank Fusion) 是一种不需要训练分数的结果融合算法。核心公式：
>
> ```
> RRF_score(d) = Σ 1/(k + rank_i(d))
> ```
>
> 其中k是平滑常数（我用的k=60）。每个文档在三路检索（dense向量、sparse关键词、KG概念）中各有一个排名，RRF对每个排名取倒数求和作为最终分数。
>
> 为什么不用带权重的线性融合？因为dense/sparse/KG三路的分数分布完全不同（cosine similarity vs BM25 vs 图距离），不能直接加权求和。RRF不依赖原始分数绝对值，只需要排名，天然适合异源融合。

### Q10: "你为什么用SQLite做知识图谱而不是Neo4j/NetworkX？"

**参考答案**：
> 三个原因：1) **零依赖**——SQLite是Python标准库，不需要装数据库或图计算库；2) **够用**——这个场景的图规模是几千个节点（一本教材几百个概念），SQLite的JOIN查询完全可以处理，不需要图数据库的遍历算法；3) **部署简单**——一个.db文件，备份/迁移/测试隔离都很方便。
>
> 如果图规模增长到百万级节点或需要复杂的图算法（PageRank、社区发现），会考虑Neo4j。但当前场景下，简单方案能解决问题就是最好的方案。

---

## 5. 简历放置建议

### 5.1 推荐位置

- **项目经历**第一位（如果只有1-2个项目）
- 如果实习经历较弱，放项目经历前面
- 技术栈关键词嵌入"技能"板块

### 5.2 不同岗位的侧重点

| 岗位 | 侧重 |
|---|---|
| **后端/AI应用** | 五Agent架构、RRF融合、异步并发管道、SSE流式 |
| **NLP/算法** | RAG设计决策、混合检索、QueryRewriter、知识图谱构建 |
| **全栈** | FastAPI路由拆分、Vue3前端、SSE实时推送、CI/CD |
| **基础架构** | 依赖注入(AppContext)、测试体系、工程化规范、错误处理 |

### 5.3 面试建议

1. **准备好架构图**：面试时主动画白板，五Agent的调用关系一目了然
1. **讲清楚设计决策**：不是"我用了XX技术"，而是"我面对XX问题，选择了XX方案，因为YY理由"
1. **主动提踩坑经历**：DEVLOG里记录的Router prompt花括号转义bug是最好的"我从错误中学到"的素材
1. **准备好demo**：本地跑起来，现场展示上传PDF→知识提取→多跳问答的完整流程

---

## 6. 项目数据一览

| 指标 | 数值 |
|---|---|
| Python代码量 | 7,409 行 |
| 模块数 | 17 个 Python 模块 |
| API端点 | 13 个 REST + SSE 端点 |
| Agent数量 | 5 个 + 1 Orchestrator |
| LLM调用次数 | 1~7 次/问题（按难度自适应） |
| 测试数量 | 318 个 (pytest) |
| 代码覆盖率 | 50% (核心路径 85%+) |
| 提交记录 | 3 个功能分支 + DEVLOG 演进记录 |
| 技术栈数量 | 20+ 个库/工具 |

---

> 本文档供简历撰写和面试准备使用。项目详细架构见 [README.md](./README.md)，开发者规范见 [RULE.md](./RULE.md)。
