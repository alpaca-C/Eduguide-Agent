"""
Build merged test set: 50 concept questions (auto-annotated) + 5 new questions (manual candidates).

Usage:
  python scripts/build_merged_dataset.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
from src.context import init_context, get_context
from src.config import Configuration
init_context(Configuration())
ctx = get_context()
vs = ctx.vector_store

# ── Load both datasets ──
old = json.loads((PROJECT_ROOT / "data/eval/custom_questions.json").read_text(encoding="utf-8"))
new = json.loads((PROJECT_ROOT / "data/eval/my_annotated.json").read_text(encoding="utf-8"))

print(f"Old (concept) questions: {len(old)}")
print(f"New (relation) questions: {len(new)}")

# ── Auto-annotate all 55 questions ──
all_questions = []

# Old 50: auto-fill relevant_chunks with top-3 dense results
for item in old:
    qid = item["question_id"]
    q = item["question"]
    dense = vs._search_dense(q, top_k=5)
    auto_chunks = []
    for r in dense[:3]:
        cid = r.get("chunk_id", "")
        if cid:
            auto_chunks.append(cid)

    all_questions.append({
        "question_id": qid,
        "question": q,
        "relevant_chunks": auto_chunks,
        "difficulty": item.get("difficulty", "moderate"),
        "_source": "auto",
    })
    print(f"  [{qid}] auto-filled {len(auto_chunks)} chunks: {auto_chunks}")

# New 5: leave relevant_chunks empty, keep candidates for manual pick
for item in new:
    qid = item["question_id"]
    q = item["question"]
    all_questions.append({
        "question_id": f"new_{qid}",
        "question": q,
        "relevant_chunks": [],  # user fills these
        "difficulty": "moderate",
        "_candidates": item.get("_candidates", []),
        "_source": "manual",
    })
    print(f"  [new_{qid}] {len(item.get('_candidates', []))} candidates — needs manual pick")

# ── Write merged output ──
out_path = PROJECT_ROOT / "data/eval/merged_55.json"
out_path.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nDone! {len(all_questions)} questions written to {out_path}")
print()
print("Next:")
print("  1. Open data/eval/merged_55.json")
print("  2. For the 5 'new_' questions, review _candidates and fill relevant_chunks")
print("  3. Review the 50 auto-filled questions if needed")
print("  4. Delete _candidates fields (optional)")
print("  5. Run: python scripts/run_beir_eval.py --custom data/eval/merged_55.json")
