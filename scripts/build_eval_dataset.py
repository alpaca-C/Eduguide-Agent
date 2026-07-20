#!/usr/bin/env python
"""
Build evaluation dataset from questions + chunk references.

Feed me simple question specs, I output the full evaluation JSON.

Usage:
  python scripts/build_eval_dataset.py
  (edit the QUESTIONS list at the bottom of this file)
"""

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "custom_questions.json"


def expand_range(doc_prefix: str, start: int, end: int) -> list[str]:
    """Expand '从电磁学_53到电磁学_66' into a list of chunk_ids."""
    return [f"{doc_prefix}_{i}" for i in range(start, end + 1)]


def build_dataset(questions: list[dict]) -> list[dict]:
    """Convert simple question specs into full evaluation records."""
    dataset = []
    for i, q in enumerate(questions):
        record = {
            "question_id": f"q{i + 1:03d}",
            "question": q["question"],
            "relevant_chunks": q["chunks"],
            "relevant_concepts": q.get("concepts", []),
            "expected_keywords": q.get("keywords", []),
            "difficulty": q.get("difficulty", "moderate"),
        }
        dataset.append(record)
    return dataset


# ═══════════════════════════════════════════════════════════════
# EDIT HERE — add your questions
# Format:
#   "question":  "学生问题",
#   "chunks":    ["doc.pdf_N", ...],  # use expand_range() for ranges
#   "concepts":  ["概念1", "概念2"],   # optional, I auto-fill from question
#   "keywords":  ["关键词1"],           # optional, I auto-fill from question
#   "difficulty": "moderate" | "complex" | "trivial"
# ═══════════════════════════════════════════════════════════════

DOC = "电磁学 梁灿彬.pdf"

QUESTIONS = [
    {
        "question": "导体是什么？",
        "chunks": [f"{DOC}_34"],
        "concepts": ["导体", "自由电子", "静电平衡"],
        "keywords": ["自由移动", "电荷", "等势体", "内部电场为零"],
        "difficulty": "moderate",
    },
    {
        "question": "库仑定律是什么？",
        "chunks": [f"{DOC}_44", f"{DOC}_45", f"{DOC}_46", f"{DOC}_47"],
        "concepts": ["库仑定律", "点电荷", "库仑力"],
        "keywords": ["库仑", "距离平方反比", "电荷乘积", "比例常数", "1785年"],
        "difficulty": "moderate",
    },
    {
        "question": "电场强度如何计算？",
        "chunks": expand_range(DOC, 53, 66),
        "concepts": ["电场强度", "试探电荷", "电场力", "点电荷电场"],
        "keywords": ["定义式", "F除以q", "矢量", "叠加原理", "点电荷场强公式"],
        "difficulty": "moderate",
    },
]

if __name__ == "__main__":
    dataset = build_dataset(QUESTIONS)
    OUTPUT_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Written {len(dataset)} questions to {OUTPUT_PATH}")
    print(f"\nPreview:")
    for q in dataset:
        print(f"  {q['question_id']}: {q['question']} ({len(q['relevant_chunks'])} chunks) [{q['difficulty']}]")
    print(f"\nNext: add more questions, then run:")
    print(f"  python scripts/run_beir_eval.py --custom data/eval/custom_questions.json")
