"""LLM-as-Judge: compare Dense vs Supplement answers on faithfulness + completeness.

Usage:
  python scripts/eval_with_llm_judge.py --limit 5
"""
from __future__ import annotations
import argparse, json, sys, time, os, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate(limit: int = 5):
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from src.context import init_context, get_context
    from src.config import Configuration
    init_context(Configuration())
    ctx = get_context()

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    config = Configuration.from_env()
    llm = ChatOpenAI(model=config.llm_model_id, api_key=config.llm_api_key,
                     base_url=config.llm_base_url, temperature=0.0)

    data = json.loads((PROJECT_ROOT / "data/eval/merged_all.json").read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]

    DOC = "数据库系统概论（第5版） .pdf"
    vs = ctx.vector_store
    kg = ctx.knowledge_graph

    # Load reranker once
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    rp = str(PROJECT_ROOT / 'models/reranker/models/BAAI--bge-reranker-v2-m3/snapshots/master/')
    model = AutoModelForSequenceClassification.from_pretrained(rp, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(rp, local_files_only=True)

    results = []
    for item in data:
        qid = item["question_id"]
        question = item["question"]
        print(f"\n[{qid}] {question[:60]}...")

        # ── Dense-only ──
        dense = vs._search_dense(question, 10, filter_docs={DOC}) or []
        dense_contexts = "\n---\n".join(c.get("text", "")[:300] for c in dense[:5])

        # ── Supplement: dense top-10 + sparse top-3 + CE rerank ──
        candidates = list(dense[:10])
        seen = {c["chunk_id"] for c in candidates}
        for s in (vs._search_sparse(question, 3, filter_docs={DOC}) or []):
            if s.get("chunk_id") and s["chunk_id"] not in seen:
                seen.add(s["chunk_id"]); candidates.append(s)
        # CE rerank
        pairs = [(question, c.get("text", "")[:500]) for c in candidates]
        with torch.no_grad():
            inputs = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            scores = model(**inputs, return_dict=True).logits.view(-1).tolist()
        for i, s in enumerate(scores): candidates[i]["ce_score"] = s
        candidates.sort(key=lambda x: x.get("ce_score", 0), reverse=True)
        supp_contexts = "\n---\n".join(c.get("text", "")[:300] for c in candidates[:5])

        # Generate answers
        def gen_answer(ctx_text):
            prompt = f"你是教材答疑助手。根据以下片段回答问题。找不到就说'未找到'。\n\n片段:\n{ctx_text}\n\n问题: {question}\n\n回答："
            resp = llm.invoke([SystemMessage(content="你是专业教材答疑助手。"), HumanMessage(content=prompt)])
            return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

        dense_answer = gen_answer(dense_contexts)
        supp_answer = gen_answer(supp_contexts)

        # LLM judge
        judge_prompt = f"""比较两个AI助手对同一问题的回答质量。

问题: {question}

助手A (dense-only):
{dense_answer[:800]}

助手B (dense+sparse+CE rerank):
{supp_answer[:800]}

请评分(1-5分)并说明理由。输出JSON:
{{"A_faithfulness": 分数, "A_completeness": 分数, "B_faithfulness": 分数, "B_completeness": 分数, "winner": "A"或"B"或"tie", "reason": "理由"}}"""

        judge = llm.invoke([SystemMessage(content="你是评估专家。只输出JSON。"), HumanMessage(content=judge_prompt)])
        judge_text = judge.content if hasattr(judge, "content") else str(judge)
        m = re.search(r'\{.*\}', judge_text, re.DOTALL)
        try:
            scores = json.loads(m.group(0)) if m else {}
        except Exception:
            scores = {}

        results.append({
            "qid": qid, "question": question,
            "dense_contexts": dense_contexts[:500],
            "supp_contexts": supp_contexts[:500],
            "dense_answer": dense_answer[:500],
            "supp_answer": supp_answer[:500],
            "judge": scores,
        })

        winner = scores.get("winner", "?")
        a_f = scores.get("A_faithfulness", "?")
        b_f = scores.get("B_faithfulness", "?")
        print(f"  Dense faithfulness={a_f}, Supp faithfulness={b_f}, winner={winner}")

        time.sleep(0.5)

    # Summary
    a_wins = sum(1 for r in results if r["judge"].get("winner") == "A")
    b_wins = sum(1 for r in results if r["judge"].get("winner") == "B")
    ties = len(results) - a_wins - b_wins
    print(f"\n=== Results ({len(results)} questions) ===")
    print(f"Dense wins: {a_wins}, Supplement wins: {b_wins}, Ties: {ties}")

    out_path = PROJECT_ROOT / "data/eval/llm_judge_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    evaluate(**vars(parser.parse_args()))
