"""
Dataset loaders for retrieval evaluation.

Supports:
  1. BEIR datasets (auto-download from public UKP server)
  2. Custom JSONL/JSON datasets

BEIR dataset list (relevant subsets for this project):
  - nfcorpus   : biomedical abstracts (closest to academic docs)
  - scifact    : scientific claim verification
  - fiqa       : financial QA
  - scidocs    : scientific document recommendation

Note: BEIR datasets are English. For Chinese教材 evaluation,
use the custom dataset format.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# BEIR datasets (official benchmarks)
# ═══════════════════════════════════════════════════════════════════

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

BEIR_AVAILABLE = [
    "nfcorpus",
    "scifact",
    "fiqa",
    "scidocs",
    "trec-covid",
    "webis-touche2020",
    "dbpedia-entity",
    "arguana",
    "climate-fever",
    "nq",
    "hotpotqa",
    "fever",
    "msmarco",
]


def _download_beir_dataset(name: str, data_dir: str) -> bool:
    """
    Download a BEIR dataset from UKP server.

    Returns True if download succeeded, False otherwise.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (data_dir / "corpus.jsonl").exists():
        return True

    url = f"{BEIR_URL}/{name}.zip"
    zip_path = data_dir / f"{name}.zip"

    logger.info("Downloading %s from %s ...", name, url)
    print(f"\nDownloading {name} dataset (~50-500 MB)...")
    print(f"URL: {url}")
    print(f"If download fails, manually download from:")
    print(f"  {url}")
    print(f"and extract to: {data_dir}\n")

    try:
        urlretrieve(url, str(zip_path))
        logger.info("Extracting %s ...", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        zip_path.unlink()  # remove zip after extraction
        logger.info("Download complete: %s", name)
        return True
    except Exception as e:
        logger.error("Download failed: %s", e)
        print(f"\n[ERROR] Auto-download failed: {e}")
        print(f"\nManual download instructions:")
        print(f"  1. Download: {url}")
        print(f"  2. Extract to: {data_dir}")
        print(f"  3. Re-run: python scripts/run_beir_eval.py --dataset {name}")
        return False


def load_beir_dataset(
    name: str = "nfcorpus",
    data_dir: str = "",
    split: str = "test",
) -> tuple[dict[str, dict], dict[str, str], dict[str, dict[str, int]]]:
    """
    Load a BEIR benchmark dataset. Auto-downloads on first use.

    Args:
        name: Dataset name (one of BEIR_AVAILABLE).
        data_dir: Custom data directory (default: project_root/data/beir/<name>/).
        split: "train" | "dev" | "test".

    Returns:
        (corpus, queries, qrels)
        - corpus: {doc_id: {"title": str, "text": str}}
        - queries: {query_id: query_text}
        - qrels: {query_id: {doc_id: relevance_score}}
    """
    from beir.datasets.data_loader import GenericDataLoader

    if not data_dir:
        data_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "beir" / name)

    # Auto-download if not present
    if not (Path(data_dir) / "corpus.jsonl").exists():
        ok = _download_beir_dataset(name, data_dir)
        if not ok:
            raise FileNotFoundError(
                f"BEIR dataset '{name}' not found at {data_dir}. "
                f"Download manually from: {BEIR_URL}/{name}.zip"
            )

    logger.info("Loading BEIR dataset '%s' from %s (split=%s)", name, data_dir, split)
    loader = GenericDataLoader(data_folder=data_dir)

    corpus, queries, qrels = loader.load(split=split)

    logger.info(
        "Loaded: %d docs, %d queries, %d qrels pairs",
        len(corpus), len(queries),
        sum(len(v) for v in qrels.values()),
    )

    return corpus, queries, qrels


# ═══════════════════════════════════════════════════════════════════
# Custom datasets (self-annotated, domain-specific)
# ═══════════════════════════════════════════════════════════════════

def load_custom_dataset(
    path: str,
) -> tuple[dict[str, dict], dict[str, str], dict[str, dict[str, int]]]:
    """
    Load a custom evaluation dataset from JSON.

    Expected format (JSON array):
    [
        {
            "question_id": "q001",
            "question": "高斯定理的物理意义是什么？",
            "relevant_chunks": ["physics.pdf_42", "physics.pdf_78"],
            "relevant_concepts": ["高斯定理", "电场强度"],
            "expected_keywords": ["封闭曲面", "通量", "电荷代数和"],
            "difficulty": "moderate"
        },
        ...
    ]

    Returns: (corpus, queries, qrels) in BEIR-compatible format.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}

    # When loading custom data, the corpus is already indexed in the project.
    # We build queries+qrels for evaluation against the existing index.
    for item in data:
        qid = item.get("question_id", item.get("id", f"q{len(queries)}"))
        queries[qid] = item["question"]
        qrels[qid] = {chunk_id: 1 for chunk_id in item.get("relevant_chunks", [])}

    # Build corpus from relevant_chunks (for DummyRetriever pipeline testing).
    # Real evaluation uses the project's pre-indexed ChromaDB + FTS5 + KG corpus.
    corpus: dict[str, dict] = {}
    for item in data:
        for chunk_id in item.get("relevant_chunks", []):
            if chunk_id not in corpus:
                corpus[chunk_id] = {
                    "title": chunk_id,
                    "text": item.get("question", ""),
                }

    logger.info(
        "Loaded custom dataset: %d queries, %d relevant chunks from %s",
        len(queries), len(corpus), path,
    )
    return corpus, queries, qrels


def create_sample_dataset(output_path: str = ""):
    """
    Generate a sample custom evaluation file to show the expected format.

    Run once to get the template, then fill in real data.
    """
    sample = [
        {
            "question_id": "q001",
            "question": "什么是库仑定律？",
            "relevant_chunks": [],
            "relevant_concepts": ["库仑定律", "点电荷", "静电力"],
            "expected_keywords": ["库仑", "电荷", "距离平方反比", "1785年"],
            "difficulty": "moderate",
        },
        {
            "question_id": "q002",
            "question": "高斯定理和库仑定律之间有什么关系？",
            "relevant_chunks": [],
            "relevant_concepts": ["高斯定理", "库仑定律", "电场强度", "电通量"],
            "expected_keywords": ["推导", "等价", "麦克斯韦方程组"],
            "difficulty": "complex",
        },
        {
            "question_id": "q003",
            "question": "怎么用高斯定理计算无限大带电平面的电场？",
            "relevant_chunks": [],
            "relevant_concepts": ["高斯定理", "无限大带电平面", "高斯面"],
            "expected_keywords": ["对称性", "圆柱形高斯面", "通量", "2ε₀", "右侧"],
            "difficulty": "moderate",
        },
        {
            "question_id": "q004",
            "question": "电势和电场强度有什么区别？",
            "relevant_chunks": [],
            "relevant_concepts": ["电势", "电场强度", "电位差"],
            "expected_keywords": ["标量", "矢量", "梯度", "积分"],
            "difficulty": "moderate",
        },
        {
            "question_id": "q005",
            "question": "静电场中的导体有什么特性？",
            "relevant_chunks": [],
            "relevant_concepts": ["静电场", "导体", "静电平衡"],
            "expected_keywords": ["等势体", "内部电场为零", "表面电荷", "屏蔽"],
            "difficulty": "moderate",
        },
    ]

    if not output_path:
        output_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "data" / "eval" / "sample_questions.json"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Created sample dataset: %s (%d questions)", output_path, len(sample))
    print(f"[OK] Sample dataset created: {output_path}")
    print(f"     {len(sample)} template questions — fill in relevant_chunks and adjust as needed.")
    return str(output_path)
