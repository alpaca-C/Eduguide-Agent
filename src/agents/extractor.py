# Knowledge Extractor Agent -- parallel batch extraction for reduced latency

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Configuration
from ..documents.chunker import TextChunk
from ..knowledge.graph import KnowledgeGraph, ConceptNode, RelationEdge

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
DEFAULT_CONCURRENCY = 3   # max parallel LLM calls for extraction

BATCH_SYSTEM_PROMPT = """你是一位知识图谱构建专家。你的任务是从给定的多个文本片段中提取关键概念及其之间的关系。

**提取规则：**
1. 识别所有文本片段中的核心概念（知识点、术语、定义、定理、方法等）
2. 为每个概念提供简洁的描述（1-2句话）
3. 识别概念之间的关系（前置知识、组成部分、示例、相关等）
4. 分类每个概念：definition（定义）、theorem（定理）、method（方法）、example（示例）、concept（概念）
5. 跨片段去重：相同概念只出现一次

**输出格式（严格JSON）：**
\u0060\u0060\u0060json
{
  "concepts": [
    {"name": "概念名称", "description": "简洁描述", "category": "definition|theorem|method|example|concept"}
  ],
  "relations": [
    {"source": "源概念名称", "target": "目标概念名称", "relation_type": "prerequisite_of|part_of|example_of|related_to|leads_to", "description": "关系简述"}
  ]
}
\
**注意：**
- 只提取文本中明确提到的概念，不要编造
- 概念名称保持与原文一致
- 只提取最重要的、最具代表性的概念
- 跨片段去重：相同的概念只出现一次，合并其描述
- 使用中文输出
"""


async def _llm_invoke_with_retry(llm, batch_prompt: str, combined_text: str,
                           n_sections: int, max_concepts: int) -> Optional[str]:
    """Async: Call LLM with exponential backoff retry. Returns content string or None."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.ainvoke([
                SystemMessage(content=batch_prompt),
                HumanMessage(content=f"请从以下 {n_sections} 个文本片段中提取知识概念和关系（最多{max_concepts}个概念）：\n\n{combined_text}"),
            ])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            last_error = e
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Batch LLM attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1, MAX_RETRIES, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Batch LLM failed after %d attempts: %s",
                    MAX_RETRIES, last_error,
                )
    return None


def _build_chapter_batches(chunks: list, max_batch_size: int) -> list[list]:
    """Build batches that never cross chapter boundaries.

    A new batch starts when:
    1. The chapter (or document) changes from the current batch
    2. The current batch reaches max_batch_size
    """
    batches: list[list] = []
    current: list = []
    current_key: str | None = None

    for ch in chunks:
        # Grouping key: chapter_title if available, else doc_filename
        ch_key = ch.chapter_title or ch.doc_filename

        # Flush if chapter changed or batch full
        if current and ch_key != current_key:
            batches.append(current)
            current = []
            current_key = None
        elif len(current) >= max_batch_size:
            batches.append(current)
            current = []
            current_key = None

        if not current:
            current_key = ch_key
        current.append(ch)

    if current:
        batches.append(current)

    return batches


def _parse_json_response(content: str) -> dict | None:
    """Parse JSON from LLM response. Returns dict or None."""
    try:
        start = content.find("{")
        if start < 0:
            return None
        depth = 0
        end = start
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        json_str = content[start:end]
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None


def _build_batch_input(batch: list, chunk_max_chars: int) -> tuple[str, str, str]:
    """Build the combined text for one batch. Returns (combined_text, source_file, source_chapter)."""
    source_chunk = batch[0]
    source_file = source_chunk.doc_filename
    source_chapter = source_chunk.chapter_title

    sections = []
    for ch in batch:
        if not ch.text.strip():
            continue
        if source_chapter:
            sections.append(
                f"[片段 {ch.chunk_index} | {source_file} | {source_chapter}] "
                f"{ch.text[:chunk_max_chars]}"
            )
        else:
            sections.append(
                f"[片段 {ch.chunk_index} | 文件: {source_file}] "
                f"{ch.text[:chunk_max_chars]}"
            )

    combined_text = "\n\n---\n\n".join(sections) if sections else ""
    return combined_text, source_file, source_chapter


async def _process_one_batch(
    llm: ChatOpenAI,
    batch_prompt: str,
    batch: list,
    chunk_max_chars: int,
    max_concepts: int,
    batch_num: int,
    total_batches: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Process a single batch: call LLM → parse JSON. Runs under semaphore for concurrency control."""
    combined_text, source_file, source_chapter = _build_batch_input(batch, chunk_max_chars)
    if not combined_text:
        return None

    async with semaphore:
        logger.info("Batch %d/%d [%s%s] starting...",
                     batch_num, total_batches, source_file,
                     f" / {source_chapter}" if source_chapter else "")
        content = await _llm_invoke_with_retry(
            llm, batch_prompt, combined_text,
            len([ch for ch in batch if ch.text.strip()]), max_concepts,
        )

    if content is None:
        logger.warning("Batch %d/%d failed after retries", batch_num, total_batches)
        return None

    data = _parse_json_response(content)
    if data is None:
        logger.warning("Batch %d/%d JSON parse failed", batch_num, total_batches)
        return None

    # Attach source attribution
    data["_source_file"] = source_file
    data["_source_chunk_id"] = batch[0].chunk_id
    logger.info("Batch %d/%d [%s%s]: %d concepts",
                 batch_num, total_batches, source_file,
                 f" / {source_chapter}" if source_chapter else "",
                 len(data.get("concepts", [])))
    return data


async def extract_full_document_async(
    chunks: list[TextChunk],
    config: Configuration,
    kg: KnowledgeGraph,
    batch_size: int | None = None,
    concurrency: int | None = None,
) -> dict:
    """Extract knowledge graph with parallel batch processing.

    Batches never cross chapter boundaries. All batches are dispatched
    concurrently (limited by `concurrency`), then results are merged
    with cross-batch concept deduplication.
    """
    if not chunks:
        return {"concepts_extracted": 0, "relations_extracted": 0}

    if batch_size is None:
        batch_size = max(50, config.extract_batch_size)
    chunk_max_chars = config.extract_chunk_max_chars
    max_concepts = config.extract_max_concepts_per_batch
    if concurrency is None:
        concurrency = getattr(config, "extract_concurrency", DEFAULT_CONCURRENCY)

    batch_prompt = BATCH_SYSTEM_PROMPT + f"\n\n本批次最多提取{max_concepts} 个概念。\n"

    llm = ChatOpenAI(
        model=config.llm_model_id,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=0.0,
    )

    # ---- Chapter-aware batching ----
    batches = list(_build_chapter_batches(chunks, batch_size))
    total_batches = len(batches)

    # ---- Phase 1: Parallel LLM calls ----
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _process_one_batch(llm, batch_prompt, batch, chunk_max_chars, max_concepts,
                           i + 1, total_batches, semaphore)
        for i, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ---- Phase 2: Sequential merge (dedup concepts, resolve relations) ----
    concept_map: dict[str, str] = {}  # normalized_name -> concept_id
    pending_relations: list[dict] = []
    total_concepts = 0

    for result in results:
        if result is None or isinstance(result, BaseException):
            continue

        source_file = result.get("_source_file", "")
        source_chunk_id = result.get("_source_chunk_id", "")

        for c in result.get("concepts", []):
            name = c.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in concept_map:
                cid = str(uuid.uuid4())[:12]
                concept_map[key] = cid
                kg.add_concept(ConceptNode(
                    id=cid, name=name,
                    description=c.get("description", ""),
                    category=c.get("category", "concept"),
                    source_chunk_id=source_chunk_id,
                    doc_filename=source_file,
                ))
                total_concepts += 1

        for r in result.get("relations", []):
            pending_relations.append(r)

    # ---- Resolve relations ----
    resolved = []
    for r in pending_relations:
        src_key = r.get("source", "").strip().lower()
        tgt_key = r.get("target", "").strip().lower()
        src_id = concept_map.get(src_key)
        tgt_id = concept_map.get(tgt_key)
        if src_id and tgt_id and src_id != tgt_id:
            resolved.append(RelationEdge(
                id=str(uuid.uuid4())[:12],
                source_id=src_id,
                target_id=tgt_id,
                relation_type=r.get("relation_type", "related_to"),
                description=r.get("description", ""),
            ))

    if resolved:
        kg.add_relations_batch(resolved)

    logger.info("Extraction complete: %d concepts, %d relations (%d batches, concurrency=%d)",
                 total_concepts, len(resolved), total_batches, concurrency)

    return {
        "concepts_extracted": total_concepts,
        "relations_extracted": len(resolved),
    }


def extract_full_document(
    chunks: list[TextChunk],
    config: Configuration,
    kg: KnowledgeGraph,
    batch_size: int | None = None,
) -> dict:
    """Synchronous wrapper — calls the async version via asyncio.run."""
    return asyncio.run(extract_full_document_async(chunks, config, kg, batch_size))

