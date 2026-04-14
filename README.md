# 📚 Course RAG — PolyU 选课智能问答系统

An Adaptive-RAG system for course selection at The Hong Kong Polytechnic University, featuring intent-aware retrieval, hybrid search, and a conversational web interface.

<br />

<img width="1633" height="1475" alt="image" src="https://github.com/user-attachments/assets/74caf991-b7df-4c26-a34a-6ed68434fa07" />



***

## ✨ Features

- **Adaptive-RAG Pipeline** — Query complexity drives retrieval depth: simple lookups skip unnecessary enhancement, while complex queries trigger full multi-stage retrieval
- **Hybrid Search** — Combines dense vector retrieval (text-embedding-v4) with sparse BM25, fused via Reciprocal Rank Fusion (RRF)
- **Two-Layer Index** — Course summary index for broad topic routing + section-level chunk index for precise retrieval
- **Dual-Model Strategy** — Fast model (Qwen-Turbo) for intent classification & query expansion; strong model (Qwen-Plus) for answer generation
- **Parent-Child Chunking** — Long sections split into child chunks for retrieval, with parent context backfill for complete answers
- **Multi-Turn Conversation** — Session-based dialogue with history-aware context
- **Inline Citations** — LLM-generated `[1]` `[2]` references rendered as interactive green dots with hover tooltips
- **Comprehensive Evaluation** — Hit Rate, Precision, NDCG, MRR metrics + ablation study across pipeline components

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
└────────────────────────┬────────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Intent Classifier  │  ← Qwen-Turbo (fast)
              │  (4 intent types)   │
              └────────┬────────────┘
                       │
         ┌─────────────┼──────────────┬──────────────┐
         ▼             ▼              ▼              ▼
    chitchat     simple_lookup    standard       complex
    (skip)       (direct search)  (expand)    (decompose)
                       │              │              │
                       │         ┌────┘              │
                       │         ▼                   ▼
                       │   Summary Index      Query Decomposition
                       │   (course routing)   + Multi-Query Expand
                       │         │                   │
                       ▼         ▼                   ▼
              ┌──────────────────────────────────────────┐
              │         Hybrid Search (async)            │
              │   Vector (ChromaDB) + BM25 → RRF Fusion  │
              └──────────────────┬───────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │  Parent Backfill → Diversity Filter       │
              └──────────────────┬───────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │  Answer Generation + Inline Citations     │  ← Qwen-Plus (strong)
              └──────────────────────────────────────────┘
```

## 📁 Project Structure

```
course-rag/
├── .env                   # API Key (gitignored)
├── config.py              # Global configuration
├── txt_parser.py          # Course TXT file parser
├── chunking.py            # Three-layer document chunking
├── indexing.py             # Embedding + ChromaDB + BM25 indexing
├── retrieval.py           # Adaptive-RAG retrieval pipeline
├── generation.py          # LLM answer generation + citations
├── evaluation.py          # Metrics + ablation study
├── app.py                 # FastAPI backend
├── static/
│   └── index.html         # Frontend (inline HTML/CSS/JS)
├── course_docs/           # Source course TXT files
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your DashScope API key:
# DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Build Index

```bash
python indexing.py --doc_dir ./course_docs
```

This parses all course TXT files, generates chunks and embeddings, builds ChromaDB + BM25 indexes, and creates course summaries.

### 4. Launch

```bash
uvicorn app:app --reload --port 8080
```

Open <http://localhost:8080> in your browser.

### 5. Run Evaluation (Optional)

```bash
python evaluation.py --mode full        # Full evaluation
python evaluation.py --mode retrieval   # Retrieval metrics only
python evaluation.py --mode ablation    # Ablation study only
```

## 📊 Evaluation

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| **Hit Rate** | Proportion of queries where at least 1 relevant result appears in top-k |
| **Recall** | Proportion of relevant courses retrieved (critical for multi-course queries) |
| **MRR** | Mean Reciprocal Rank of the first relevant result |
| **Course Coverage** | B-class only: actual relevant courses found / total expected (e.g., 33/53) |

> **Dynamic top-k**: Simple/Reasoning/Boundary queries use k=5; Multi-course queries use k=15 to maximize course coverage.

### Generation Metrics

| Metric | Description |
|--------|-------------|
| **Completeness** | LLM-judged score (1-5) measuring how fully the answer addresses the question |
| **Keyword Hit Rate** | Hard metric: percentage of expected keywords present in the answer (deterministic, reproducible) |
| **Faithfulness** | Whether the answer is correct against ground truth (expected keywords), NOT against retrieved docs |
| **Groundedness** | Whether the answer only contains facts from retrieved documents (hallucination detection) |

> **Faithfulness vs Groundedness**: Faithfulness checks "did you answer correctly?" while Groundedness checks "did you make anything up?". A system can be grounded (no hallucination) but unfaithful (answered wrong course's info because retrieval failed).

### Test Suite Design

24 test cases across 4 categories:

| Category | Count | Purpose |
|----------|-------|---------|
| A: Simple Lookup | 6 | Single course, single section — baseline accuracy |
| B: Multi-Course / Broad | 8 | Cross-course queries including colloquial phrasing — tests summary routing & course coverage |
| C: Advanced Reasoning | 6 | Cross-section, multi-condition queries — tests query expansion & decomposition |
| D: Anti-Hallucination | 4 | Non-existent info, edge cases — tests refusal ability & boundary reasoning |

### Ablation Study

Compares Naive RAG (direct vector search, no enhancement) vs Full Pipeline across all metrics:

| Config | Components |
|--------|------------|
| **Naive RAG** | ChromaDB dense retrieval only — raw question as query, no intent classification, no filtering |
| **Full Pipeline** | Intent classification → Summary index routing → Metadata filtering → Multi-query expansion → Async parallel retrieval → Parent context backfill |

Both retrieval and generation metrics are reported per-category to reveal where each pipeline component contributes most.

### Key Findings

- **A (Simple Lookup)**: Full Pipeline significantly outperforms Naive due to metadata filtering by course code and section type
- **B (Multi-Course)**: Both systems face inherent top-k coverage limits; Full Pipeline achieves higher recall through summary-level routing
- **C (Cross-Section Reasoning)**: Largest performance gap — query decomposition and multi-section retrieval show clear value
- **D (Anti-Hallucination)**: Full Pipeline achieves perfect retrieval; accurate course targeting enables correct refusal



## 🛠️ Tech Stack

| Component     | Technology                                                          |
| :------------ | :------------------------------------------------------------------ |
| LLM           | Qwen-Plus (generation), Qwen-Turbo (intent/expansion) via DashScope |
| Embedding     | text-embedding-v4 (DashScope)                                       |
| Vector Store  | ChromaDB                                                            |
| Sparse Search | BM25 (rank-bm25 + jieba tokenization)                               |
| Framework     | LangChain                                                           |
| Backend       | FastAPI                                                             |
| Frontend      | Vanilla HTML/CSS/JS                                                 |

## 📝 Data Format

Each course is stored as a key-value TXT file:

```
"Subject Code": "COMP 5422"
"Subject Title": "Multimedia Computing, Systems and Applications"
"Credit Value": "3"
"Subject Synopsis/ Indicative Syllabus": "• Multimedia System Primer ..."
"Assessment Methods ...": "1. Assignments 30%; 2. Final Examination 70%"
"Class Time": "Thursday 14:30-17:20"
```

The parser handles fuzzy key matching, whitespace normalization, and missing field tolerance.

## 💎 Future Work
 
### 1. Evaluation Enhancements
 
**Additional Retrieval Metrics**:
- **Precision@5**: Fraction of top-5 results matching the target course
- **Section Precision@5**: Stricter variant — matches both course code and section type
- **NDCG@5**: Normalized Discounted Cumulative Gain — measures ranking quality with graded relevance
 
**Component-Level Ablation**:
Incrementally adding each pipeline component to isolate individual contributions:
 
| Config | Components |
|--------|------------|
| Vector only | ChromaDB dense retrieval |
| + BM25 (RRF) | + Sparse retrieval with Reciprocal Rank Fusion |
| + Summary Index | + Course-level summary routing |
| Full Pipeline | + Multi-Query Expansion (async parallel) |
 
 
### 2. High Concurrency & Production Deployment
 
Current system is single-user. To support campus-wide deployment (10,000+ concurrent users):
 
- **Multi-level caching**: L1 in-process cache (LRU) for hot queries → L2 Redis distributed cache for cross-instance sharing → L3 vector DB as cold storage. Cache keys based on query semantic hash to handle paraphrased questions hitting the same cache
- **Graceful degradation**: If LLM API times out (>5s), fall back to retrieval-only mode returning raw document snippets; if vector DB is overloaded, fall back to BM25-only keyword search; circuit breaker pattern to prevent cascading failures
- **Horizontal scaling**: Stateless FastAPI instances behind load balancer; vector DB and Redis as shared external services
 
### 3. Incremental Document Updates
 
Current pipeline requires full re-indexing when course documents change. Optimizations:
 
- **Document fingerprinting**: Compute MD5/SHA256 hash per document on each sync cycle; only re-chunk, re-embed, and re-index documents whose hash has changed
- **Chunk-level diff**: For partially modified documents, identify changed sections by comparing section-level hashes, only re-embed affected chunks rather than the entire document
- **Summary invalidation**: When a course document updates, automatically regenerate only that course's summary in the summary index, avoiding full summary rebuild
 
### 4. Alternative Vector Database Solutions
 
Current system uses ChromaDB (suitable for prototyping). Production alternatives to explore:
 
- **Milvus**: Distributed architecture, supports billion-scale vectors, GPU-accelerated indexing, better suited for large-scale deployment
- **Qdrant**: Built-in payload filtering (can replace manual metadata filtering logic), supports on-disk storage for cost efficiency
- **Weaviate**: Native hybrid search (vector + keyword in single query), built-in multi-tenancy for isolating different departments' data
- **pgvector**: PostgreSQL extension, avoids introducing a separate vector DB service, simplifies ops for smaller deployments
 
### 5. Alternative Embedding Models
 
Current system uses DashScope `text-embedding-v4`. Alternatives to benchmark:
 
- **BGE-M3** (BAAI): Supports dense, sparse, and multi-vector retrieval in a single model; natively multilingual (Chinese + English course content); can run locally to eliminate API latency
- **Cohere Embed v3**: Leading multilingual performance, built-in input type parameter (`search_document` vs `search_query`) for asymmetric retrieval
- **Local deployment**: Running embedding models on-premise via Ollama or vLLM to eliminate network round-trip (~200ms per call), especially beneficial since embedding is called multiple times per query in multi-query expansion
 
### 6. Multimodal Content Processing
 
Current system only handles text from course descriptions. Many syllabi contain diagrams, formulas, and tables in PDF/image format:
 
- **PDF visual extraction**: Use layout-aware parsers (e.g., marker, Unstructured.io) to extract tables, figures, and mathematical formulas while preserving structure
- **Image understanding**: For diagrams and charts embedded in course materials, use vision-language models (e.g., GPT-4o, Qwen-VL) to generate textual descriptions, then index those descriptions alongside text chunks
- **Table handling**: Extract tables as structured data (JSON/CSV), store both raw table and LLM-generated natural language summary as separate chunks for different query types
 
### 7. Semantic Caching with Redis
 
Current system re-runs full retrieval pipeline for every query, even repeated or similar ones:
 
- **Semantic cache design**: Embed each incoming query → search Redis for cached queries with cosine similarity > 0.95 → if hit, return cached retrieval results directly, skipping vector DB and LLM intent classification
- **Cache structure**: Redis key = query embedding hash, value = serialized retrieval results + generated answer + TTL timestamp
- **Data consistency**: When documents are updated (detected via fingerprinting from §3), invalidate all cache entries whose retrieved chunks reference the modified course code; use Redis pub/sub to broadcast invalidation events across instances
- **Cache warming**: Pre-compute and cache results for the top 50 most frequently asked questions during off-peak hours
 
### 8. Advanced Hybrid Ranking (Beyond RRF)
 
Current system uses static Reciprocal Rank Fusion to merge HNSW (vector) and BM25 (keyword) results. Limitations: fixed weighting ignores query characteristics.
 
- **Learned Sparse-Dense Fusion**: Train a lightweight cross-encoder reranker (e.g., BGE-reranker-v2, ~33M params) that takes query + candidate document pairs and outputs relevance scores, replacing static RRF with learned relevance
- **Query-adaptive weighting**: For keyword-heavy queries (e.g., "COMP5517 assessment") increase BM25 weight; for semantic queries (e.g., "courses about building intelligent systems") increase HNSW weight. Weight selection based on query features (presence of course codes, section keywords, etc.)
- **Concrete approach**: Implement as a simple logistic regression on query features → [w_dense, w_sparse] weights, trained on the evaluation dataset's retrieval judgments
 
### 9. Local Intent Classifier (Replacing LLM API Call)
 
Current intent classification calls `qwen-turbo` via public API (~1-3s per call), which is the latency bottleneck for every query:
 
- **Approach**: Fine-tune a lightweight text classifier (e.g., BERT-base-chinese, ~110M params) on labeled intent data to predict `{chitchat, simple_lookup, standard, complex}` + extract `course_code`, `section_interest`, `is_broad`
- **Training data**: Use the existing LLM intent classifier to label 1000-2000 synthetic queries (generated by LLM from course data), then train the local model on these labels (knowledge distillation)
- **Expected improvement**: Inference time from ~1-3s (API call) → ~10-50ms (local GPU/CPU), reducing overall query latency by 30-50%
- **Hybrid fallback**: If local classifier confidence < 0.8, fall back to LLM API call for difficult queries; log low-confidence cases for future training data collection

## License

This project is for academic purposes at The Hong Kong Polytechnic University.
