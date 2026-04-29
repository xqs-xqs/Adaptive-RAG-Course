# 📚 Course RAG — 大学选课智能问答系统（自适应 RAG）

<div align="center">


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-异步-009688)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-向量库-FF6B6B)](https://www.trychroma.com/)
[![Qwen](https://img.shields.io/badge/LLM-通义千问%20(DashScope)-FFA500)](https://dashscope.aliyun.com/)
[![License](https://img.shields.io/badge/license-学术用途-green)](#许可证)

大学选课问答 RAG 系统（PolyU），具备意图感知路由、混合检索、对话式 Web 界面和完整的评测框架。



</div>


>**生产级自适应检索增强生成（Adaptive RAG）系统**，实现了**混合检索（BM25 + 稠密向量）**、**倒数排名融合（RRF）**、**父子分块（Parent-Child Chunking）**、**查询拆解（Query Decomposition）**、**多轮对话**，以及完整的**评测体系**（命中率、召回率、MRR、忠实度、接地性）。



**技术栈**：**FastAPI · LangChain · ChromaDB · 通义千问（DashScope）· jieba 中文分词**。同时附带一个 **Claude Mentor Skill**，让这个仓库成为你的私人 RAG 讲师 + 大厂面试陪练。


 **关键词**：自适应 RAG · 混合检索 · BM25 · 稠密向量检索 · 倒数排名融合 RRF · 父子分块 · 查询拆解 · 多查询扩展 · 意图分类 · 两层索引 · FastAPI · LangChain · ChromaDB · 通义千问 · DashScope · jieba · 中文 NLP · 大语言模型 · 检索增强生成 · 选课问答 · 问答系统 · 对话机器人 · SSE 流式 · 忠实度 · 接地性 · 消融实验



---

<br />

<img width="1633" height="1475" alt="香港理工大学选课 RAG 对话界面 — 展示混合检索结果、意图分类与内联引用" src="https://github.com/user-attachments/assets/74caf991-b7df-4c26-a34a-6ed68434fa07" />

---

## 🎯 适合谁用？

- **机器学习 / NLP 工程师**：需要一份完整、可复现、有评测的 RAG 参考实现
- **求职者 / 在读学生**：准备大厂后端或 AI 岗位面试，需要一个有技术深度的简历项目（仓库自带[面试陪练 Skill](#-学习与面试陪练-skill)）
- **研究人员**：在小规模中英双语语料上实验自适应路由、混合检索、消融实验方法
- **落地从业者**：将 RAG 适配到课程目录、知识库问答、文档咨询机器人，或任何"结构化字段 + 自由文本"场景

---

## ✨ 功能亮点

- **自适应 RAG 流水线** — 查询复杂度决定检索深度：简单查询跳过不必要的增强步骤，复杂查询触发完整多阶段检索
- **混合检索（Hybrid Search）** — 稠密向量检索（text-embedding-v4）+ 稀疏 BM25，通过倒数排名融合（RRF）合并排序
- **两层索引** — 课程摘要索引（粗粒度路由定位目标课程）+ 章节级 Chunk 索引（精细检索）
- **快慢双模型策略** — 快模型（Qwen-Turbo）负责意图分类和查询扩展；强模型（Qwen-Plus）负责答案生成
- **父子分块** — 长 Section 切成小 Child Chunk 用于检索，生成时回填父文档完整原文
- **多轮对话** — 基于 Session 的对话历史感知上下文
- **内联引用** — LLM 生成的 `[1]` `[2]` 引用渲染为悬浮提示的绿色小圆点
- **完整评测体系** — 命中率、精确率、NDCG、MRR + 各组件消融实验
- **附带 Mentor Skill** — Claude Skill（`adaptive-rag-mentor/`），让这个仓库成为交互式学习 + 面试陪练伙伴（全中文，讲解 + 拷问混合模式）——详见 [学习与面试陪练 Skill](#-学习与面试陪练-skill)

---

## 🏗️ 系统架构

系统采用**两阶段自适应流水线**：基于 LLM 的**意图分类器**将查询路由到四条路径之一（chitchat / simple_lookup / standard / complex），每条路径检索深度不同。standard 和 complex 路径使用**两层索引**——先用课程摘要索引缩小候选课程范围，再在目标课程的 Chunk 索引内做精细检索——通过**混合检索（BM25 + 向量）**并以 **RRF 融合**排序。

```
┌─────────────────────────────────────────────────────────┐
│                      用户提问                            │
└────────────────────────┬────────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │     意图分类器       │  ← Qwen-Turbo（快模型）
              │   （4 种意图类型）   │
              └────────┬────────────┘
                       │
         ┌─────────────┼──────────────┬──────────────┐
         ▼             ▼              ▼              ▼
      闲聊          精确查询        标准查询       复杂查询
    （直接回复）   （直接检索）    （查询扩展）   （拆解+扩展）
                       │              │              │
                       │         ┌────┘              │
                       │         ▼                   ▼
                       │    摘要索引定位课程      查询拆解
                       │   （粗粒度路由）      + 多查询扩展
                       │         │                   │
                       ▼         ▼                   ▼
              ┌──────────────────────────────────────────┐
              │         异步混合检索                      │
              │   向量检索（ChromaDB）+ BM25 → RRF 融合   │
              └──────────────────┬───────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │      父文档回填 → 多样性过滤               │
              └──────────────────┬───────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │         答案生成 + 内联引用                │  ← Qwen-Plus（强模型）
              └──────────────────────────────────────────┘
```

---

## 📁 项目结构

```
course-rag/
├── .env                    # API Key（已加入 .gitignore）
├── config.py               # 全局配置
├── txt_parser.py           # 课程 TXT 文件解析器
├── chunking.py             # 三层文档分块
├── indexing.py             # Embedding + ChromaDB + BM25 索引构建
├── retrieval.py            # 自适应 RAG 检索流水线
├── generation.py           # LLM 答案生成 + 引用标注
├── evaluation.py           # 评测指标 + 消融实验
├── app.py                  # FastAPI 后端
├── static/
│   └── index.html          # 前端（内联 HTML/CSS/JS）
├── course_docs/            # 源课程 TXT 文件
├── adaptive-rag-mentor/    # 附带的 Claude Skill（交互式讲师 + 面试拷问）
│   ├── SKILL.md
│   └── references/         # 14 个深度 Reference 文件（项目精读、
│                           #   技术栈原理、RAG 领域知识、生产化、坑、题库）
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DashScope API Key：
# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 构建索引

```bash
python indexing.py --doc_dir ./course_docs
# 产出：chroma_db/  bm25_index.pkl  parent_store.json
```

### 4. 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
# 打开 http://localhost:8080
```

### 5. 运行评测（可选）

```bash
python evaluation.py
# 输出：24 题评测结果 + 消融对比 + 延迟分布
# 结果保存到 eval_results.json
```

---

## 📊 评测体系

### 检索指标

| 指标 | 说明 |
|---|---|
| **命中率（Hit Rate）** | top-k 结果中至少包含一个相关课程 |
| **召回率（Recall）** | 所有相关课程中被召回的比例 |
| **MRR（平均倒数排名）** | 第一个相关结果的排名倒数，衡量"前几名准不准" |

### 生成指标

| 指标 | 说明 | 客观/主观 |
|---|---|---|
| **关键词命中率** | 答案包含 `expected_keywords` 的比例 | 客观（字符串匹配） |
| **完整度（Completeness）** | LLM 评判 1-5 分，回答涵盖要点的程度 | LLM 主观 |
| **忠实度（Faithfulness）** | 答案是否正确传达了 ground-truth 事实 | LLM 主观 |
| **接地性（Groundedness）** | 答案是否仅依赖检索到的文档（幻觉检测） | LLM 主观 |

> **忠实度 vs 接地性的本质区别**：忠实度测"答对了没"（对 ground-truth）；接地性测"有没有编"（对检索文档）。一个系统可以接地但不忠实——检索不准但 LLM 老老实实地说错了文档里的内容，没幻觉但答错了。两个指标都需要才能完整评估 RAG 系统。

### 测试集设计（24 题 × 4 类）

| 类别 | 题数 | 考察重点 | top_k |
|---|---|---|---|
| **A — 精确查询** | 6 | 单课程单字段精确命中 | 5 |
| **B — 广泛查询** | 8 | 跨课程广泛检索，关注课程覆盖度 | **15**（特殊放大）|
| **C — 跨字段推理** | 6 | 多字段联合推理，测查询拆解能力 | 5 |
| **D — 防幻觉** | 4 | 测系统能否正确拒绝不存在的信息 | 5 |

### 消融实验

通过 `ablation_config` 三个开关对比 Full Pipeline vs Naive RAG：

| 组件 | Naive RAG | Full Pipeline |
|---|---|---|
| 检索方式 | 单次向量检索 | 混合检索（BM25 + 向量 + RRF） |
| 摘要索引路由 | ✗ | ✓ |
| 多查询扩展 | ✗ | ✓ |

**主要发现**：
- **A 类（精确查询）**：Full Pipeline 显著优于 Naive——元数据双过滤精确锁定 course_code + section_type
- **C 类（跨字段推理）**：差距最大——查询拆解 + 多 section 并行检索体现核心价值
- **B 类（广泛查询）**：两者差距相对小——top_k 覆盖度是瓶颈，而非路由策略
- **D 类（防幻觉）**：Full Pipeline 100% 精准定位，让模型能正确判断并拒绝不存在的信息

---

## 🛠️ 技术栈

| 组件 | 技术 |
|---|---|
| **大语言模型** | Qwen-Plus（答案生成）+ Qwen-Turbo（意图分类、查询扩展、拆解），通过 DashScope OpenAI 兼容 API |
| **嵌入模型** | DashScope `text-embedding-v4`（1536 维，多语言，非对称 query/document 编码） |
| **向量库** | ChromaDB（HNSW 索引，嵌入式模式，SQLite 持久化后端） |
| **稀疏检索** | BM25（rank-bm25 + jieba 中文分词，Okapi BM25，IDF + TF 饱和） |
| **结果融合** | 倒数排名融合（RRF，k=60） |
| **LLM 框架** | LangChain（Embeddings、Document、RecursiveCharacterTextSplitter、ChatOpenAI） |
| **Web 后端** | FastAPI（ASGI，async/await，Pydantic 校验，OpenAPI 自动文档） |
| **流式响应** | Server-Sent Events（SSE），Token 级实时推送 |
| **前端** | 原生 HTML/CSS/JS，带内联引用 `[1]` `[2]` 悬浮渲染 |
| **Token 计数** | tiktoken（cl100k_base）控制 Chunk 大小预算 |
| **并发** | asyncio + ThreadPoolExecutor，支持查询扩展 + 拆解并行执行 |

---

## 📝 数据格式

课程文档存放在 `course_docs/`，每门课一个 `.txt` 文件，格式如下：

```
"Subject Code": "COMP 5422"
"Subject Title": "Multimedia Computing, Systems and Applications"
"Credit Value": "3"
"Level": "5"
"Pre-requisite": "Nil"
"Subject Synopsis/ Indicative Syllabus": "..."
"Intended Learning Outcomes": "..."
"Assessment Methods in Alignment with Intended Learning Outcomes": "..."
"Teaching and Learning Activities": "..."
"Reading List and References": "..."
"Class Schedule": "..."
"Study Effort Expected of Students": "..."
```

`txt_parser.py` 通过两遍匹配（精确 → 模糊容错）将 key 归一化到标准字段名，`chunking.py` 按字段边界切 Chunk 并注入上下文 Prefix。

---

## 💎 未来工作

### 1. 评测增强

- **Precision@k**：(course_code + section_type) 二元组粒度的精确率
- **NDCG**：考虑分级相关性的排名质量指标
- **8 种 Ablation 组合全跑**：三个开关 × 2 = 8 种配置，配对 t-test 显著性检验
- **多次运行取均值**：LLM 评判（忠实度 / 接地性）有随机性，3-5 次跑报均值 ± 标准差
- **测试集扩充**：当前 24 题统计意义弱，扩充到 100+ 题

### 2. 高并发与生产部署

当前为研究/教学原型，生产化需要：
- 异步 LLM 调用（`ainvoke` / `astream`）替换同步阻塞调用
- 对话历史从内存 dict → **Redis**（多 Worker 共享，支持 TTL 自动过期）
- ChromaDB 嵌入式 → **ChromaDB Server 模式**或 **Milvus / Qdrant**（多 Worker 避免 SQLite 锁冲突）
- BM25 → **Elasticsearch**（原生倒排索引，支持元数据过滤，百万级文档性能稳定）
- 多级缓存（本地 LRU + Redis 语义缓存，30-50% 命中率）
- 限流 + 熔断（Circuit Breaker）+ 可观测性（Prometheus + Grafana + OpenTelemetry 链路追踪）

### 3. 增量文档更新

- 文档指纹（Hash）检测变更，仅重新嵌入修改的块
- 版本感知 Chunk ID，避免索引污染
- 支持向量库原子 Swap（上线不中断）

### 4. 向量库替代方案

- **Milvus**：分布式，十亿级，HNSW + IVF+PQ 量化
- **Qdrant**：Rust 实现，强 Payload 索引，性能优秀
- **pgvector**：Postgres 扩展，已有 PG 基础设施时零运维成本

### 5. 嵌入模型替代方案

- **BGE-M3**（BAAI 开源）：单模型同时支持稠密 / 稀疏 / ColBERT 多向量，中英双语强，可本地部署（GPU 推理 ~20ms）
- **Voyage-3**：RAG 优化，非对称检索效果好
- **Cohere Embed v3**：原生 `input_type` 区分 query/document，128 语言支持

### 6. 多模态内容处理

- 基于版式的 PDF 解析器（marker / Unstructured），保留表格、图片位置
- 表格单独作为 Chunk，附带结构化元数据
- 图片通过视觉模型（LLaVA / GPT-4V）转文本后索引

### 7. Redis 语义缓存

- Query → Embedding → Redis 近似最近邻搜索（cosine > 0.95 视为命中）
- 按 course_code 主动失效：文档更新时清除相关 Cache
- 版本 Key（`q:{hash}:v:{doc_version_hash}`）保证文档变更后 Cache 自然失效

### 8. 超越 RRF 的高级混合排序

- **SPLADE**：学习型稀疏检索，BM25 的语义升级版
- **ColBERT**：每 Token 一个向量，后交互精细排序
- **BGE-reranker-v2-m3**：Cross-Encoder 二次精排，top 100 → top 5，准确率 +20-30%
- **LLM-as-Reranker**：用强模型直接对候选排序（慢但最准）

### 9. 本地意图分类器（替代 LLM API 调用）

当前意图分类调用 `qwen-turbo` API（~1-3s/次），是延迟瓶颈：

- **方案**：用 LLM 意图分类器标注 1000-2000 条合成查询，蒸馏到轻量本地分类器（如 BERT-base-chinese，~110M 参数）
- **预期改进**：推理延迟从 ~1-3s（API 调用）→ ~10-50ms（本地 GPU/CPU），总查询延迟降低 30-50%
- **混合兜底**：本地分类器置信度 < 0.8 时回退到 LLM API；记录低置信度样本用于后续训练数据收集

---

## 📖 学习与面试陪练 Skill

本仓库附带一个 [Anthropic Claude Skill](https://docs.claude.com/en/docs/claude-code/skills)——`adaptive-rag-mentor/`——让这个代码库成为你的交互式讲师和面试拷问伙伴。专为深度理解**这个项目**而打造，目标是：在技术面试中，无论面试官指哪一行代码问"为什么"，你都能说出设计动机、对比方案、和踩过的坑。

Skill 全程中文，支持两种模式协同工作：

- **讲解模式（A）**："这段代码/模块做什么"类问题。依次讲解：一句话定位 → 设计意图 → 实现拆解 → 对比方案 → 踩坑预警 → 面试官视角。
- **拷问模式（B）**：模拟面试。一次问一题，按 🟢/🟡/🟠/🔴 四档难度递进，答完给评分 + 点评 + 追问，模拟大厂面试官的逼问风格。

### 内容一览

```
adaptive-rag-mentor/
├── SKILL.md                          # 路由规则 + 教学协议
└── references/
    ├── 00_project_map.md             # 项目鸟瞰、文件依赖、请求数据流
    ├── 01_config_and_app.md          # config.py + app.py 逐行剖析
    ├── 02_parsing_chunking.md        # txt_parser + chunking 策略
    ├── 03_indexing.md                # 索引构建流水线 + Embedding Wrapper
    ├── 04_retrieval.md               # 核心：意图路由、RRF、混合检索、异步（~40KB）
    ├── 05_generation.md              # Prompt 构造 + 多轮对话管理
    ├── 06_evaluation.md              # 4 指标 + 消融 + 忠实度 vs 接地性详解
    ├── tech_fastapi.md               # FastAPI / ASGI / Pydantic / SSE 原理
    ├── tech_langchain.md             # LangChain 抽象与批评
    ├── tech_jieba_bm25.md            # jieba 分词原理 + BM25 公式推导
    ├── tech_chromadb_embedding.md    # HNSW、向量库选型、嵌入模型对比
    ├── tech_asyncio.md               # 协程、GIL、线程池、run_in_executor
    ├── rag_domain.md                 # RAG 通用知识：分块策略、检索范式、
    │                                 #   重排、Agentic RAG、Contextual Retrieval
    ├── production.md                 # 缓存、限流、熔断、可观测性、
    │                                 #   A/B 测试、安全（Prompt 注入、多租户）
    ├── interview_drill.md            # 65 道分级题库（含参考答案和追问话术）
    └── gotchas.md                    # 代码库中 16 个具体 Bug / 设计瑕疵
                                      #   （位置、影响、修法、被问到怎么答）
```

### 如何使用

**方式一 — 上传到 Claude.ai**：
1. 将 `adaptive-rag-mentor/` 打包成 zip
2. 在 Claude.ai 设置中上传为自定义 Skill
3. 新建对话，说"开始复习我的 adaptive-rag-course 项目"或"讲讲 retrieval.py 里的 RRF"，Skill 自动激活

**方式二 — 配合 Claude Code 本地使用**：
将文件夹放到你的 skills 目录（`~/.claude/skills/` 或项目本地），在任何关于这个代码库的对话中直接触发。

**方式三 — 直接当文档读**：
即便不用 Claude，markdown 文件本身就可读。从 `00_project_map.md` 开始看整体架构，再深入 `04_retrieval.md`（最核心）和 `gotchas.md`（已知问题）。

### 对话示例

```
你：讲讲 RRF 是怎么工作的

Skill（讲解模式）：
  【一句话定位】把多路检索结果用排名信息融合，不依赖原始分数
  【设计意图】BM25 分数 (0~∞) 和 cosine 相似度 (-1~1) 量纲不同，
              加权平均要先标准化，太麻烦也不稳定...
  【实现拆解】retrieval.py 第 173-185 行，公式 score = Σ 1/(k+rank+1)，k=60...
  【对比方案】vs CombSUM / 加权平均 / Cross-Encoder Rerank 各自的权衡...
  【踩坑预警】第 179 行用 page_content[:100] 当 doc_id，
              chunks 的 prefix 相同时会误合并！
  【面试官视角】可能追问：k=60 怎么定的？RRF 能被 LLM 替代吗？
  
  准备好让我从面试官视角拷问你吗？

你：拷问我

Skill（拷问模式）：
  🟡 中级第 1 题：为什么用 RRF 不用加权平均？
  [等你回答]
  → 评分：基本正确 / 点评不足之处 / 追问下一层
```

### 为什么做成 Skill 而不只是文档？

读文档是被动的。Skill 通过拷问模式强制**主动回忆**，并根据你的回答动态调整深度。Reference 文件既是可检索的文档（适合快速翻阅），也是 Skill 的知识库（仅在相关时才加载到 Context，控制 Token 用量同时保留深度）。

### 透明说明

`gotchas.md` 完整记录了代码库中发现的 **16 个问题**，包括：
- `config.py:8` 模型名 typo（`qwen3.6-plus` 应为 `qwen-plus`）
- `retrieval.py:378` 三元表达式两边一样的调试残留（`max_per = 2 if is_broad else 2`）
- 架构层问题：内存 `ConversationManager` 不支持多 Worker，`async def` 路由里调同步 LLM 阻塞事件循环

这些问题故意保持透明：Skill 的目的是学习，假装代码完美反而让教训失效。大多数是 MVP 阶段的合理取舍，生产部署前需要逐一修复。

---

## ❓ 常见问题

**Q：自适应 RAG 和朴素 RAG 有什么区别？**

朴素 RAG 无论查询复杂度如何都走同一条路——单次向量检索，固定 top-k，没有任何增强。自适应 RAG 先分类意图（chitchat / simple_lookup / standard / complex），再路由到匹配复杂度的路径：简单查询跳过增强步骤，复杂查询触发查询拆解 + 多查询扩展 + 并行混合检索。本质是让检索深度匹配查询难度，在延迟和准确率之间找最优点。完整路由逻辑见 `retrieval.py`。

**Q：为什么要混合检索（BM25 + 向量）而不只用向量？**

稠密向量检索（语义搜索）能理解语义，但对罕见专有名词、课程代码（如 `COMP5422`）和精确关键词匹配效果差。BM25（稀疏检索）擅长精确词匹配，通过逆文档频率（IDF）赋予罕见词更高权重。通过倒数排名融合（RRF）组合两者，召回率和 MRR 均优于单独任一方法——在 `evaluation.py` 的消融实验中已经验证。

**Q：倒数排名融合（RRF）是什么？为什么不用加权平均？**

RRF 公式：`score(d) = Σ 1/(k + rank_i + 1)`，k=60。它只用排名信息（不用原始分数），因为 BM25 分数（0~∞）和 cosine 相似度（-1~1）量纲不兼容，加权平均要标准化，既麻烦又不稳。RRF 参数少、鲁棒、经过 25 年 IR 实践检验（最初来自 Cormack et al. 2009，Elasticsearch 的 `reciprocal_rank_fusion` API 也用这个）。

**Q：父子分块（Parent-Child Chunking）怎么工作？**

长 Section 被切成小 Child Chunk（~500 token）用于检索——小 Chunk 的 Embedding 聚焦，能精确匹配查询。完整父文档文本单独存储，生成时回填，所以 LLM 看到的是完整上下文。解决的是"小 Chunk 检索准但生成时信息残缺，大 Chunk 生成好但 Embedding 失焦导致检索差"这个经典两难。

**Q：忠实度（Faithfulness）和接地性（Groundedness）有什么区别？**

**忠实度**：答案是否正确传达了 ground-truth 事实（参照 `expected_keywords` 评判）。**接地性**：答案是否仅使用检索文档中的信息（幻觉检测）。一个系统可以"接地但不忠实"——检索没找到对的文档，LLM 老实地说了错误文档里的内容，没幻觉但答错了。完整评测需要两个指标同时监控。

**Q：能把这套系统适配到其他中文课程目录 / 领域问答吗？**

可以，架构是领域无关的。适配步骤：① 用你的语料替换 `course_docs/`；② 更新 `txt_parser.py` 字段映射；③ 调整 `chunking.py` 的 `SECTION_MAPPING`；④ 根据你的查询分布调整 `retrieval.py` 的意图分类 Prompt；⑤ 重新运行 `python indexing.py`。`adaptive-rag-mentor/` Skill 的每个 Reference 文件都覆盖了对应的定制点。

**Q：为什么选 Qwen / DashScope，而不是 OpenAI？**

项目针对中文课程内容，Qwen 在中文上表现强劲。DashScope 提供 OpenAI 兼容的 API 端点（`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`），切换到 OpenAI 只需修改 `LLM_MODEL` 和 `base_url`，LangChain 的 `ChatOpenAI` 包装器完全兼容。

**Q：这套系统可以直接上生产吗？**

这是研究 / 教学原型，有文档记录的局限性（`gotchas.md` 列出了 16 个具体问题）。生产部署需要规划：异步 LLM 调用、Redis 对话状态、ChromaDB Server 模式（或 Milvus/Qdrant 规模化）、Elasticsearch 替代 BM25、多级缓存、限流、熔断器和可观测性。详见上方[未来工作 → 高并发与生产部署](#2-高并发与生产部署)。

---

## 许可证

本项目为香港理工大学学术用途。
