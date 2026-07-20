"""RAGAS evaluation: compare Dense-only vs Hybrid on answer quality.

Measures: faithfulness, answer_relevancy, context_precision, context_recall.

Usage:
  python scripts/ragas_eval.py           # all 56 questions
  python scripts/ragas_eval.py --limit 5 # quick demo (5 questions)
"""
from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_ragas(limit: int = 0):
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from src.context import init_context, get_context
    from src.config import Configuration
    init_context(Configuration())
    ctx = get_context()

    from src.evaluation.retriever import ProjectRetriever
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    # Load dataset
    data = json.loads((PROJECT_ROOT / "data/eval/merged_ch1ch6.json").read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]

    config = Configuration.from_env()
    llm = ChatOpenAI(
        model=config.llm_model_id,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=0.0,
    )

    print(f"Evaluating {len(data)} questions: Dense vs Hybrid\n")

    eval_rows = []
    for item in data:
        qid = item["question_id"]
        question = item["question"]
        ground_chunks = set(item.get("relevant_chunks", []))

        print(f"[{qid}] {question[:60]}...")

        # ── Dense-only ──
        dense_ret = ProjectRetriever(strategy="dense",
                                      doc_filter="数据库系统概论（第5版） .pdf")
        dense_raw = dense_ret.search({}, {qid: question}, top_k=5)
        dense_chunks = list(dense_raw.get(qid, {}).keys())
        dense_contexts = _chunk_texts(dense_chunks[:5])

        # ── Hybrid ──
        hyb_ret = ProjectRetriever(strategy="hybrid",
                                    doc_filter="数据库系统概论（第5版） .pdf")
        hyb_raw = hyb_ret.search({}, {qid: question}, top_k=5)
        hyb_chunks = list(hyb_raw.get(qid, {}).keys())
        hyb_contexts = _chunk_texts(hyb_chunks[:5])

        # Generate answers
        dense_answer = _generate_answer(llm, question, dense_contexts)
        hyb_answer = _generate_answer(llm, question, hyb_contexts)

        # Record for RAGAS
        eval_rows.append({
            "question": question,
            "answer": dense_answer,
            "contexts": dense_contexts,
            "reference": "",  # ground truth answer (not chunk IDs)
            "strategy": "dense",
            "question_id": qid,
        })
        eval_rows.append({
            "question": question,
            "answer": hyb_answer,
            "contexts": hyb_contexts,
            "reference": "",
            "strategy": "hybrid",
            "question_id": qid,
        })

        time.sleep(0.5)  # rate limit

    # ── RAGAS scoring (monkey-patch vertexai import bug) ──
    import sys as _sys, types as _types
    _fake = _types.ModuleType("langchain_community.chat_models.vertexai")
    class _FakeChatVertexAI: pass
    _fake.ChatVertexAI = _FakeChatVertexAI
    _sys.modules["langchain_community.chat_models.vertexai"] = _fake
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    ds = Dataset.from_list(eval_rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=ctx.vector_store._ef._model,
    )

    # Split results by strategy
    df = result.to_pandas()
    df["strategy"] = [eval_rows[i]["strategy"] for i in range(len(eval_rows))]
    dense_df = df[df["strategy"] == "dense"]
    hyb_df = df[df["strategy"] == "hybrid"]

    print("\n" + "=" * 60)
    print("RAGAS Evaluation: Dense vs Hybrid")
    print("=" * 60)
    metrics_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for m in metrics_names:
        if m in df.columns:
            d_val = dense_df[m].mean()
            h_val = hyb_df[m].mean()
            diff = h_val - d_val
            arrow = "▲" if diff > 0 else "▼" if diff < 0 else "="
            print(f"  {m:<20s}  dense={d_val:.4f}  hybrid={h_val:.4f}  {arrow} {diff:+.4f}")

    print(f"\n  Questions evaluated: {len(data)}")

    return 0


def _chunk_texts(chunk_ids: list[str]) -> list[str]:
    from src.context import get_context
    ctx = get_context()
    data = ctx.vector_store._collection.get(ids=chunk_ids, include=["documents"]) if chunk_ids else {"documents": []}
    return list(data["documents"])


def _generate_answer(llm, question: str, contexts: list[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    ctx_text = "\n\n---\n\n".join(contexts[:5]) if contexts else "(无检索结果)"
    prompt = (
        f"你是一位大学教材答疑助手。请根据以下教材片段回答学生的问题。\n"
        f"如果片段中找不到答案，请诚实地说'教材中未找到相关内容'。\n\n"
        f"教材片段:\n{ctx_text}\n\n"
        f"学生问题: {question}\n\n请回答："
    )
    resp = llm.invoke([SystemMessage(content="你是一个专业的教材答疑助手。"),
                        HumanMessage(content=prompt)])
    return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    sys.exit(evaluate_ragas(limit=args.limit))
