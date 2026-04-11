import asyncio
import time
import json
import logging
from tabulate import tabulate

from retrieval import retrieve, vectorstore
from generation import generate_answer, llm

# Disable verbose logging from other modules to keep output clean
logging.getLogger().setLevel(logging.WARNING)

# ============================================================
# 按类别动态调整 top_k
# B 类（多课程广泛问题）需要更大的 k 才能覆盖足够多的课程
# ============================================================
K_DEFAULT = 5
K_BROAD = 15  # B 类用更大的 k

def _get_k(item):
    """根据题目类别返回合适的 top_k。"""
    return K_BROAD if item.get("category") == "B" else K_DEFAULT


EVAL_DATASET = [
    # =========================================================
    # A. 简单查找（6 题）
    # =========================================================
    {
        "question": "What is the exam weighting for COMP5517?",
        "category": "A",
        "relevant_courses": ["COMP5517"],
        "relevant_sections": ["assessment"],
        "expected_keywords": ["70%", "Examination", "Projects and Assignments", "30%"],
    },
    {
        "question": "What topics does COMP5523 cover?",
        "category": "A",
        "relevant_courses": ["COMP5523"],
        "relevant_sections": ["syllabus"],
        "expected_keywords": ["deep learning", "Transformer", "image segmentation", "3D vision", "optical flow"],
    },
    {
        "question": "What are the references for COMP5511?",
        "category": "A",
        "relevant_courses": ["COMP5511"],
        "relevant_sections": ["references"],
        "expected_keywords": ["Russell", "Norvig", "Artificial Intelligence", "PROLOG", "Bratko"],
    },
    {
        "question": "When is COMP5424's class?",
        "category": "A",
        "relevant_courses": ["COMP5424"],
        "relevant_sections": ["class_time"],
        "expected_keywords": ["Wednesday", "8:30", "11:20"],
    },
    {
        "question": "How is COMP5355 assessed?",
        "category": "A",
        "relevant_courses": ["COMP5355"],
        "relevant_sections": ["assessment"],
        "expected_keywords": ["Assignments", "10%", "Class project", "20%", "Examination", "70%"],
    },
    {
        "question": "Does COMP5425 have any prerequisites?",
        "category": "A",
        "relevant_courses": ["COMP5425"],
        "relevant_sections": ["prerequisites"],
        "expected_keywords": ["signals and systems", "recommended", "general knowledge"],
    },

    # =========================================================
    # B. 多课程 / 广泛问题（8 题）
    # =========================================================
    {
        "question": "Which courses are scheduled in the evening (starting at 18:30)?",
        "category": "B",
        "relevant_courses": ["COMP5355", "COMP5423", "COMP5434", "COMP5511", "COMP5513",
                             "COMP5521", "COMP5523", "COMP5311", "COMP5322", "COMP5327",
                             "COMP5425", "COMP5532"],
        "relevant_sections": ["class_time"],
        "expected_keywords": ["18:30", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    },
    {
        "question": "Which courses are related to artificial intelligence or machine learning?",
        "category": "B",
        "relevant_courses": ["COMP5511", "COMP5523", "COMP5423", "COMP5434", "COMP5532"],
        "relevant_sections": ["syllabus"],
        "expected_keywords": ["Artificial Intelligence", "machine learning", "deep learning", "neural network"],
    },
    {
        "question": "Which courses include a group project as part of the assessment?",
        "category": "B",
        "relevant_courses": ["COMP5355", "COMP5424", "COMP5434", "COMP5513", "COMP5521",
                             "COMP5523", "COMP5532", "COMP5327", "COMP5423"],
        "relevant_sections": ["assessment"],
        "expected_keywords": ["project", "group", "30%", "20%"],
    },
    {
        "question": "Which courses have no prerequisites?",
        "category": "B",
        "relevant_courses": ["COMP5422", "COMP5511", "COMP5513", "COMP5521", "COMP5523",
                             "COMP5311", "COMP5322", "COMP5424", "COMP5423"],
        "relevant_sections": ["prerequisites"],
        "expected_keywords": ["Nil", "no prerequisites"],
    },
    {
        "question": "What courses are available in the networking and security area?",
        "category": "B",
        "relevant_courses": ["COMP5311", "COMP5355", "COMP5327"],
        "relevant_sections": ["syllabus"],
        "expected_keywords": ["Internet", "TCP/IP", "security", "wireless", "network"],
    },
    {
        "question": "I'm interested in multimedia and video processing, which courses should I consider?",
        "category": "B",
        "relevant_courses": ["COMP5422", "COMP5425"],
        "relevant_sections": ["syllabus"],
        "expected_keywords": ["multimedia", "JPEG", "MPEG", "video", "compression", "coding"],
    },
    {
        "question": "I don't want to deal with any prerequisites, just give me courses I can jump right into.",
        "category": "B",
        "relevant_courses": ["COMP5422", "COMP5511", "COMP5513", "COMP5521", "COMP5523",
                             "COMP5311", "COMP5322", "COMP5424", "COMP5423"],
        "relevant_sections": ["prerequisites"],
        "expected_keywords": ["Nil", "no prerequisites"],
    },
    {
        "question": "I want to take as few exams as possible, which courses have the lowest exam percentage?",
        "category": "B",
        "relevant_courses": ["COMP5424", "COMP5521", "COMP5513", "COMP5532"],
        "relevant_sections": ["assessment"],
        "expected_keywords": ["40%", "50%", "Examination"],
    },

    # =========================================================
    # C. 进阶推理 / 跨 section（6 题）
    # =========================================================
    {
        "question": "Will there be any guest lectures or industry partners involved in COMP5521?",
        "category": "C",
        "relevant_courses": ["COMP5521"],
        "relevant_sections": ["teaching_methodology"],
        "expected_keywords": ["Practitioners", "industry partners", "Guest lectures"],
    },
    {
        "question": "How many total study hours are expected for COMP5424, including self-study?",
        "category": "C",
        "relevant_courses": ["COMP5424"],
        "relevant_sections": ["study_effort"],
        "expected_keywords": ["115", "76", "39"],
    },
    {
        "question": "Will I learn or use PyTorch in the Computer Vision course?",
        "category": "C",
        "relevant_courses": ["COMP5523"],
        "relevant_sections": ["references", "syllabus"],
        "expected_keywords": ["PyTorch", "Modern Computer Vision with PyTorch", "Ayyadevara"],
    },
    {
        "question": "Does COMP5511 use Prolog for programming exercises?",
        "category": "C",
        "relevant_courses": ["COMP5511"],
        "relevant_sections": ["teaching_methodology", "references"],
        "expected_keywords": ["prolog", "expert system shells", "programming exercises", "Bratko"],
    },
    {
        "question": "Which evening courses related to AI also include a group project?",
        "category": "C",
        "relevant_courses": ["COMP5423", "COMP5434", "COMP5532"],
        "relevant_sections": ["class_time", "syllabus", "assessment"],
        "expected_keywords": ["18:30", "project", "machine learning", "deep learning", "big data"],
    },
    {
        "question": "Are there any courses with no prerequisites that also cover deep learning topics?",
        "category": "C",
        "relevant_courses": ["COMP5523", "COMP5511"],
        "relevant_sections": ["prerequisites", "syllabus"],
        "expected_keywords": ["Nil", "deep learning", "neural network"],
    },

    # =========================================================
    # D. 防幻觉 + 边界测试（4 题）
    # =========================================================
    {
        "question": "Does COMP5521 cover quantum cryptography?",
        "category": "D",
        "relevant_courses": ["COMP5521"],
        "relevant_sections": ["syllabus"],
        "expected_keywords": ["not covered", "no information", "does not cover",
                              "not mentioned", "cannot find"],
    },
    {
        "question": "Is there a midterm exam for COMP5513?",
        "category": "D",
        "relevant_courses": ["COMP5513"],
        "relevant_sections": ["assessment"],
        "expected_keywords": ["Continuous Assessment", "Final Examination",
                              "not explicitly mentioned", "no midterm", "not specified"],
    },
    {
        "question": "Is COMP5434 Big Data Computing strictly requiring database knowledge as prerequisite?",
        "category": "D",
        "relevant_courses": ["COMP5434"],
        "relevant_sections": ["prerequisites"],
        "expected_keywords": ["preferred", "not strictly", "knowledge in database systems",
                              "is preferred", "recommended"],
    },
    {
        "question": "What is the syllabus for COMP5999?",
        "category": "D",
        "relevant_courses": [],
        "relevant_sections": [],
        "expected_keywords": ["not found", "does not exist", "no information",
                              "cannot find", "not available"],
    },
]

CATEGORY_LABELS = {
    "A": "A: Simple Lookup",
    "B": "B: Multi-course / Broad",
    "C": "C: Cross-section Reasoning",
    "D": "D: Anti-hallucination",
}


# ============================================================
# 朴素 RAG（支持动态 k）
# ============================================================

def naive_retrieve(question: str, k: int = 5) -> dict:
    docs = vectorstore.similarity_search(question, k=k)
    return {"intent": "naive", "docs": docs, "parent_contexts": {}}


# ============================================================
# 通用检索指标计算（每题使用各自的 k）
# ============================================================

def _compute_retrieval_metrics(docs_list, dataset):
    """
    检索指标计算。每个 item 根据类别使用不同的 k 截断。
    额外计算 B 类的 Course Coverage（覆盖了多少门不同的课程）。
    """
    hit_rates, recalls, mrrs = [], [], []
    cat_hits, cat_recalls, cat_mrrs = {}, {}, {}
    # B 类专用：课程覆盖数
    cat_coverage = {}

    for item, docs_raw in zip(dataset, docs_list):
        k = _get_k(item)
        docs = docs_raw[:k]
        rel_courses = item.get("relevant_courses", [])
        cat = item.get("category", "?")

        if len(rel_courses) == 0:
            continue

        rel_set = set(rel_courses)
        found = set(d.metadata.get("course_code", "") for d in docs)

        # Hit Rate
        hit = any(d.metadata.get("course_code", "") in rel_set for d in docs)
        hit_val = 1.0 if hit else 0.0
        hit_rates.append(hit_val)
        cat_hits.setdefault(cat, []).append(hit_val)

        # Recall
        recall_val = len(found & rel_set) / len(rel_set)
        recalls.append(recall_val)
        cat_recalls.setdefault(cat, []).append(recall_val)

        # MRR
        rr = 0.0
        for rank, doc in enumerate(docs, 1):
            if doc.metadata.get("course_code", "") in rel_set:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        cat_mrrs.setdefault(cat, []).append(rr)

        # Course Coverage（B 类专用）: 找到了几门不同的相关课程 / 总共应该几门
        if cat == "B":
            coverage = len(found & rel_set)
            cat_coverage.setdefault(cat, []).append(
                {"found": coverage, "total": len(rel_set)}
            )

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    overall = {"hit_rate": _mean(hit_rates), "recall": _mean(recalls), "mrr": _mean(mrrs)}

    by_category = {}
    for cat in sorted(set(list(cat_hits.keys()) + list(cat_recalls.keys()))):
        entry = {
            "hit_rate": _mean(cat_hits.get(cat, [])),
            "recall": _mean(cat_recalls.get(cat, [])),
            "mrr": _mean(cat_mrrs.get(cat, [])),
            "n": len(cat_hits.get(cat, [])),
        }
        # B 类加入 Course Coverage 统计
        if cat == "B" and cat in cat_coverage:
            total_found = sum(c["found"] for c in cat_coverage[cat])
            total_expected = sum(c["total"] for c in cat_coverage[cat])
            entry["course_coverage"] = f"{total_found}/{total_expected}"
        by_category[cat] = entry

    return overall, by_category


# ============================================================
# 检索评测（Full Pipeline + 延迟）
# ============================================================

async def evaluate_retrieval(dataset):
    all_docs, latencies, per_question = [], [], []

    for item in dataset:
        start = time.time()
        result = await retrieve(item["question"])
        lat = (time.time() - start) * 1000
        latencies.append(lat)
        all_docs.append(result["docs"])

        k = _get_k(item)
        per_question.append({
            "question": item["question"],
            "category": item.get("category", "?"),
            "top_k": k,
            "latency_ms": round(lat, 1),
            "retrieved_codes": [d.metadata.get("course_code", "?") for d in result["docs"][:k]],
        })

    overall, by_category = _compute_retrieval_metrics(all_docs, dataset)

    latencies_sorted = sorted(latencies)
    p95_idx = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)

    return {
        **overall,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "p95_latency_ms": round(latencies_sorted[p95_idx]) if latencies_sorted else 0,
        "by_category": by_category,
        "per_question": per_question,
    }


# ============================================================
# 关键词命中率（硬指标）
# ============================================================

def _keyword_hit_rate(answer: str, expected_keywords: list) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


# ============================================================
# 生成质量评测
# 4 指标：Keyword Hit / Completeness / Faithfulness / Groundedness
# ============================================================

def _eval_generation(dataset, retrieval_results):
    total_completeness = 0
    total_faithful = 0
    total_grounded = 0
    total_kw_hit = 0.0

    cat_completeness, cat_kw_hit, cat_faithful, cat_grounded = {}, {}, {}, {}

    for i, item in enumerate(dataset):
        cat = item.get("category", "?")
        answer, _ = generate_answer(item["question"], retrieval_results[i])
        context = "\n".join([doc.page_content for doc in retrieval_results[i]["docs"]])

        # 1. Keyword Hit Rate
        kw_rate = _keyword_hit_rate(answer, item.get("expected_keywords", []))
        total_kw_hit += kw_rate
        cat_kw_hit.setdefault(cat, []).append(kw_rate)

        # 2. Completeness (LLM, strict)
        try:
            c_resp = llm.invoke([{"role": "user", "content": f"""Rate the completeness of this answer (1-5):
1=Completely missed  2=Partial  3=Mostly complete  4=Nearly complete  5=Fully complete

Question: {item['question']}
Answer: {answer}
Key information that should be present: {', '.join(item.get('expected_keywords', []))}

Return ONLY a single number (1-5)."""}])
            c_score = float(c_resp.content.strip())
        except Exception:
            c_score = 0.0
        total_completeness += c_score
        cat_completeness.setdefault(cat, []).append(c_score)

        # 3. Faithfulness (against ground truth, NOT retrieved docs)
        try:
            f_resp = llm.invoke([{"role": "user", "content": f"""You are a strict evaluator for a course advisory chatbot.
Determine whether the answer provides CORRECT and RELEVANT information.

Question: {item['question']}
Answer: {answer}

Ground truth key facts that a correct answer should convey:
{', '.join(item.get('expected_keywords', []))}

Evaluation rules:
- "faithful" if the answer correctly conveys the key facts listed above
- "unfaithful" if the answer provides WRONG information (wrong numbers, wrong days, wrong courses)
- "unfaithful" if the answer says "I don't know" / "not found" when ground truth has real facts
- "unfaithful" if the answer misses MOST of the key facts (covering <30%)
- For boundary/negative questions where expected answer IS "not found"/"not covered",
  answering "not found" or equivalent is "faithful"

Return ONLY "faithful" or "unfaithful"."""}])
            is_faithful = "unfaithful" not in f_resp.content.strip().lower()
        except Exception:
            is_faithful = False
        total_faithful += (1 if is_faithful else 0)
        cat_faithful.setdefault(cat, []).append(1.0 if is_faithful else 0.0)

        # 4. Groundedness (no hallucination)
        try:
            h_resp = llm.invoke([{"role": "user", "content": f"""You are a hallucination detector for a RAG system.
Determine if the answer contains fabricated information not present in the source documents.

Question: {item['question']}
Answer: {answer}

Source documents (the ONLY factual basis available):
{context}

Rules:
- "grounded" if the answer ONLY states facts from the source documents above
- "grounded" if the answer honestly says "I don't know" or "no information found"
- "hallucinated" if the answer includes specific facts, numbers, dates, names, or percentages
  NOT found anywhere in the source documents
- Focus on factual claims, ignore minor phrasing differences
- Be strict with specific details: a wrong percentage or invented book title = hallucinated

Return ONLY "grounded" or "hallucinated"."""}])
            is_grounded = "hallucinated" not in h_resp.content.strip().lower()
        except Exception:
            is_grounded = True
        total_grounded += (1 if is_grounded else 0)
        cat_grounded.setdefault(cat, []).append(1.0 if is_grounded else 0.0)

    num_q = len(dataset)
    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    by_category = {}
    for cat in sorted(set(list(cat_completeness.keys()))):
        by_category[cat] = {
            "completeness": round(_mean(cat_completeness.get(cat, [])), 1),
            "keyword_hit_rate": round(_mean(cat_kw_hit.get(cat, [])), 2),
            "faithfulness": round(_mean(cat_faithful.get(cat, [])), 2),
            "groundedness": round(_mean(cat_grounded.get(cat, [])), 2),
            "n": len(cat_completeness.get(cat, [])),
        }

    return {
        "avg_completeness": round(total_completeness / num_q, 1) if num_q else 0,
        "keyword_hit_rate": round(total_kw_hit / num_q, 2) if num_q else 0,
        "faithfulness_rate": round(total_faithful / num_q, 2) if num_q else 0,
        "groundedness_rate": round(total_grounded / num_q, 2) if num_q else 0,
        "by_category": by_category,
    }


# ============================================================
# 消融实验
# ============================================================

async def run_ablation(dataset):
    # 1. Naive RAG 检索（B 类也用更大的 k，公平对比）
    print("  [1/4] Naive RAG retrieval...")
    naive_results = []
    for item in dataset:
        k = _get_k(item)
        naive_results.append(naive_retrieve(item["question"], k=k))
    naive_docs = [r["docs"] for r in naive_results]

    # 2. Full Pipeline 检索
    print("  [2/4] Full Pipeline retrieval...")
    full_results = []
    for item in dataset:
        result = await retrieve(item["question"])
        full_results.append(result)
    full_docs = [r["docs"] for r in full_results]

    # --- 检索指标对比 ---
    naive_ret_overall, naive_ret_cat = _compute_retrieval_metrics(naive_docs, dataset)
    full_ret_overall, full_ret_cat = _compute_retrieval_metrics(full_docs, dataset)

    retrieval_comparison = {
        "overall": [
            {"Config": "Naive RAG", **{k: f"{v:.2f}" for k, v in naive_ret_overall.items()}},
            {"Config": "Full Pipeline", **{k: f"{v:.2f}" for k, v in full_ret_overall.items()}},
        ],
        "by_category": {},
    }
    # 合并两边的类别，去重
    all_cats = sorted(set(list(naive_ret_cat.keys()) + list(full_ret_cat.keys())))
    for cat in all_cats:
        nc = naive_ret_cat.get(cat, {})
        fc = full_ret_cat.get(cat, {})
        rows = [
            {"Config": "Naive", "Hit Rate": f"{nc.get('hit_rate',0):.2f}",
             "Recall": f"{nc.get('recall',0):.2f}", "MRR": f"{nc.get('mrr',0):.2f}"},
            {"Config": "Full",  "Hit Rate": f"{fc.get('hit_rate',0):.2f}",
             "Recall": f"{fc.get('recall',0):.2f}", "MRR": f"{fc.get('mrr',0):.2f}"},
        ]
        # B 类额外显示 Course Coverage
        if cat == "B":
            rows[0]["Coverage"] = nc.get("course_coverage", "N/A")
            rows[1]["Coverage"] = fc.get("course_coverage", "N/A")
        retrieval_comparison["by_category"][cat] = rows

    # --- 生成指标对比 ---
    print("  [3/4] Evaluating Naive RAG generation...")
    naive_gen = _eval_generation(dataset, naive_results)

    print("  [4/4] Evaluating Full Pipeline generation...")
    full_gen = _eval_generation(dataset, full_results)

    generation_comparison = {
        "overall": [
            {
                "Config": "Naive RAG",
                "Completeness": f"{naive_gen['avg_completeness']:.1f}/5",
                "KW Hit": f"{naive_gen['keyword_hit_rate']*100:.0f}%",
                "Faithfulness": f"{naive_gen['faithfulness_rate']*100:.0f}%",
                "Groundedness": f"{naive_gen['groundedness_rate']*100:.0f}%",
            },
            {
                "Config": "Full Pipeline",
                "Completeness": f"{full_gen['avg_completeness']:.1f}/5",
                "KW Hit": f"{full_gen['keyword_hit_rate']*100:.0f}%",
                "Faithfulness": f"{full_gen['faithfulness_rate']*100:.0f}%",
                "Groundedness": f"{full_gen['groundedness_rate']*100:.0f}%",
            },
        ],
        "by_category": {},
    }
    # 用 all_cats 已包含 D（如果 D 有数据的话），不再额外拼接
    gen_cats = sorted(set(list(naive_gen["by_category"].keys()) + list(full_gen["by_category"].keys())))
    for cat in gen_cats:
        nc = naive_gen["by_category"].get(cat, {})
        fc = full_gen["by_category"].get(cat, {})
        if not nc and not fc:
            continue
        generation_comparison["by_category"][cat] = [
            {"Config": "Naive",
             "Complete": f"{nc.get('completeness',0):.1f}/5",
             "KW Hit": f"{nc.get('keyword_hit_rate',0)*100:.0f}%",
             "Faithful": f"{nc.get('faithfulness',0)*100:.0f}%",
             "Grounded": f"{nc.get('groundedness',0)*100:.0f}%"},
            {"Config": "Full",
             "Complete": f"{fc.get('completeness',0):.1f}/5",
             "KW Hit": f"{fc.get('keyword_hit_rate',0)*100:.0f}%",
             "Faithful": f"{fc.get('faithfulness',0)*100:.0f}%",
             "Grounded": f"{fc.get('groundedness',0)*100:.0f}%"},
        ]

    return {"retrieval": retrieval_comparison, "generation": generation_comparison}


# ============================================================
# CLI
# ============================================================

async def main(mode):
    results_dump = {}

    if mode in ["retrieval", "full"]:
        print("Running Retrieval Evaluation (Full Pipeline)...")
        print(f"  (Using k={K_DEFAULT} for A/C/D, k={K_BROAD} for B)")
        r = await evaluate_retrieval(EVAL_DATASET)
        results_dump["retrieval"] = r

        table = [
            ["Hit Rate",    f"{r['hit_rate']:.2f}"],
            ["Recall",      f"{r['recall']:.2f}"],
            ["MRR",         f"{r['mrr']:.2f}"],
            ["Avg Latency", f"{r['avg_latency_ms']} ms"],
            ["P95 Latency", f"{r['p95_latency_ms']} ms"],
        ]
        print("\n=== Retrieval Metrics (Overall) ===")
        print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

        cat_table = []
        for cat, m in r["by_category"].items():
            label = CATEGORY_LABELS.get(cat, cat)
            k_used = K_BROAD if cat == "B" else K_DEFAULT
            row = [label, m["n"], f"k={k_used}",
                   f"{m['hit_rate']:.2f}", f"{m['recall']:.2f}", f"{m['mrr']:.2f}"]
            # B 类额外显示 Course Coverage
            if "course_coverage" in m:
                row.append(m["course_coverage"])
            else:
                row.append("-")
            cat_table.append(row)
        if cat_table:
            print("\n=== Retrieval Metrics (By Category) ===")
            print(tabulate(cat_table,
                           headers=["Category", "N", "Top-K", "Hit Rate", "Recall", "MRR", "Coverage"],
                           tablefmt="grid"))

    if mode in ["ablation", "full"]:
        print(f"\nRunning Ablation Study (Naive RAG vs Full Pipeline)...")
        print(f"  (Using k={K_DEFAULT} for A/C/D, k={K_BROAD} for B)")
        abl = await run_ablation(EVAL_DATASET)
        results_dump["ablation"] = abl

        print("\n" + "=" * 60)
        print("=== Ablation: Retrieval Comparison (Overall) ===")
        print("=" * 60)
        print(tabulate(abl["retrieval"]["overall"], headers="keys", tablefmt="grid"))

        for cat, rows in abl["retrieval"]["by_category"].items():
            label = CATEGORY_LABELS.get(cat, cat)
            k_used = K_BROAD if cat == "B" else K_DEFAULT
            print(f"\n--- Retrieval: {label} (k={k_used}) ---")
            print(tabulate(rows, headers="keys", tablefmt="grid"))

        print("\n" + "=" * 60)
        print("=== Ablation: Generation Comparison (Overall) ===")
        print("=" * 60)
        print(tabulate(abl["generation"]["overall"], headers="keys", tablefmt="grid"))

        for cat, rows in abl["generation"]["by_category"].items():
            label = CATEGORY_LABELS.get(cat, cat)
            print(f"\n--- Generation: {label} ---")
            print(tabulate(rows, headers="keys", tablefmt="grid"))

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results_dump, f, indent=2, ensure_ascii=False, default=str)
    print("\nResults saved to eval_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "retrieval", "ablation"], default="full")
    args = parser.parse_args()

    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(main(args.mode))
