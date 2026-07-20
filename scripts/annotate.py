"""
半自动标注工具：用 dense 搜索辅助标注 relevant_chunks。

用法：
  python scripts/annotate.py --input questions.txt --output annotated.json

输入格式（questions.txt，每行一道题）：
  数据库和DBMS是什么关系？
  关系数据库理论的前置知识有哪些？
  ...

输出（annotated.json）：
[
  {
    "question_id": "q001",
    "question": "数据库和DBMS是什么关系？",
    "relevant_chunks": [],  // <-- 你手动填，候选已列出
    "_candidates": [        // <-- 自动搜索的候选 chunk
      {"chunk_id": "xxx.pdf_24", "score": 0.95, "text": "数据库是..."},
      ...
    ]
  }
]

工作流：
  1. python scripts/annotate.py --input my_questions.txt
  2. 打开输出的 JSON，每个问题下面有 _candidates
  3. 逐个看候选 chunk 的 text 预览，把相关的 chunk_id 填进 relevant_chunks
  4. 删掉 _candidates 字段（可选）
  5. python scripts/run_beir_eval.py --custom annotated.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def annotate(questions_file: str, output_file: str, top_k: int = 10):
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from src.context import init_context
    from src.config import Configuration
    init_context(Configuration())
    from src.context import get_context

    ctx = get_context()
    vs = ctx.vector_store

    # Read questions
    questions_path = Path(questions_file)
    if questions_path.suffix == ".json":
        # JSON array of {"question": "..."} or just strings
        data = json.loads(questions_path.read_text(encoding="utf-8"))
        if isinstance(data[0], str):
            questions = data
        else:
            questions = [item["question"] for item in data]
    else:
        questions = [
            line.strip()
            for line in questions_path.read_text(encoding="utf-8").split("\n")
            if line.strip() and not line.startswith("#")
        ]

    print(f"Loaded {len(questions)} questions")
    print(f"Searching top-{top_k} candidates per question...\n")

    results = []
    for i, q in enumerate(questions):
        qid = f"q{i + 1:03d}"
        print(f"[{qid}] {q[:60]}...")

        # Dense search for candidates (most reliable for finding relevant chunks)
        dense_results = vs._search_dense(q, top_k=top_k)
        candidates = []
        for item in dense_results:
            candidates.append({
                "chunk_id": item.get("chunk_id", ""),
                "score": round(item.get("score", 0), 4),
                "text": (item.get("text", "") or "")[:300],
                "doc": item.get("doc_filename", ""),
                "chapter": item.get("chapter_title", ""),
            })

        # Also try sparse for comparison (may find different chunks)
        sparse_results = vs._search_sparse(q, top_k=5)
        sparse_ids = {item.get("chunk_id") for item in sparse_results}
        for item in sparse_results:
            cid = item.get("chunk_id", "")
            if cid not in {c["chunk_id"] for c in candidates}:
                candidates.append({
                    "chunk_id": cid,
                    "score": round(item.get("score", 0), 4),
                    "text": (item.get("text", "") or "")[:300],
                    "doc": item.get("doc_filename", ""),
                    "chapter": item.get("chapter_title", ""),
                    "source": "sparse",
                })

        # Sort by score desc
        candidates.sort(key=lambda c: c["score"], reverse=True)

        results.append({
            "question_id": qid,
            "question": q,
            "relevant_chunks": [],
            "_candidates": candidates[:top_k + 5],  # keep a few extra
        })

    # Write output
    output_path = Path(output_file)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone! Wrote {len(results)} questions to {output_path}")
    print()
    print("Next steps:")
    print("  1. Open the JSON file")
    print("  2. For each question, look at _candidates[*].text")
    print("  3. Copy the chunk_ids that actually contain the answer into relevant_chunks")
    print(f"  4. Run: python scripts/run_beir_eval.py --custom {output_file}")


def main():
    parser = argparse.ArgumentParser(description="半自动标注 relevant_chunks")
    parser.add_argument("--input", type=str, required=True, help="Questions file (txt or json)")
    parser.add_argument("--output", type=str, default="data/eval/annotated.json", help="Output json")
    parser.add_argument("--top-k", type=int, default=10, help="Candidates per question")
    args = parser.parse_args()
    annotate(args.input, args.output, args.top_k)


if __name__ == "__main__":
    main()
