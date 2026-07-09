# Pipeline — document processing workflow (parse -> chunk -> index -> extract)

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END, START

from .config import Configuration
from .documents.parser import parse_document, Document
from .documents.chunker import chunk_document, TextChunk
from .agents.extractor import extract_full_document
from .agents.qa import DocumentVectorStore
from .knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """State for the document processing pipeline."""
    filepaths: list[str]
    documents: list[Document]
    chunks: list[TextChunk]
    message: str
    concepts_extracted: int
    relations_extracted: int
    ready: bool
    error: str


def parse_node(state: PipelineState, config: dict) -> dict:
    """Parse all uploaded documents."""
    filepaths = state.get("filepaths", [])
    if not filepaths:
        return {"error": "没有文件", "ready": False}
    
    documents = []
    for fp in filepaths:
        try:
            doc = parse_document(fp)
            documents.append(doc)
            logger.info("Parsed: %s (%d chars)", doc.filename, len(doc.content))
        except Exception as e:
            logger.error("Parse failed for %s: %s", fp, e)
    
    if not documents:
        return {"error": "所有文件解析失败", "ready": False}
    
    total_chars = sum(len(d.content) for d in documents)
    return {
        "documents": documents,
        "message": f"已解析 {len(documents)} 个文件，共 {total_chars} 字符",
    }


def chunk_node(state: PipelineState, config: dict) -> dict:
    """Chunk all parsed documents."""
    documents = state.get("documents", [])
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    
    logger.info("Chunked %d documents into %d chunks", len(documents), len(all_chunks))
    return {
        "chunks": all_chunks,
        "message": f"已分为 {len(all_chunks)} 个文本片段",
    }


def index_node(state: PipelineState, config: dict) -> dict:
    """Index chunks into local vector store."""
    chunks = state.get("chunks", [])
    
    vs = DocumentVectorStore()
    
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "doc_filename": c.doc_filename,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    vs.index_chunks(chunk_dicts)
    
    return {"message": f"已索引 {len(chunks)} 个片段到向量库"}


def extract_node(state: PipelineState, config: dict) -> dict:
    """Extract knowledge graph from chunks."""
    cfg = Configuration(**(config.get("configurable", {})))
    chunks = state.get("chunks", [])
    
    kg = KnowledgeGraph()
    kg.clear()  # Start fresh
    
    result = extract_full_document(chunks, cfg, kg)
    
    stats = kg.stats()
    return {
        "concepts_extracted": result["concepts_extracted"],
        "relations_extracted": result["relations_extracted"],
        "message": f"知识图谱: {stats['concepts']} 个概念, {stats['relations']} 个关系",
        "ready": True,
    }


def build_pipeline() -> StateGraph:
    """Build document processing pipeline graph."""
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("parse", parse_node)
    workflow.add_node("chunk", chunk_node)
    workflow.add_node("index", index_node)
    workflow.add_node("extract", extract_node)
    
    workflow.add_edge(START, "parse")
    workflow.add_edge("parse", "chunk")
    workflow.add_edge("chunk", "index")
    workflow.add_edge("index", "extract")
    workflow.add_edge("extract", END)
    
    return workflow.compile()
