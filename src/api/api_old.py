"""FastAPI Backend -- Document QA System REST API"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
from src.config import Configuration
from src.documents.parser import parse_document, Document
from src.documents.chunker import chunk_document as chunk_doc_func
from src.agents.extractor import extract_full_document_async
from src.agents.qa import answer_question, DocumentVectorStore, ReflectionAgent
from src.tools.rag_search import init_rag_tool
import src.tools.ocr  # register mineru_ocr tool for agent
from src.agents.chapterizer import ChapterizerAgent, ChapterInfo, _split_by_meta
from src.knowledge.graph import KnowledgeGraph
from src.memory import set_memory_store
from src.memory.store import MemoryStore
import src.monitoring  # activate token tracking (MONITORING_ENABLED=true)
from src.monitoring.reporter import router as monitoring_router
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("doc-qa-api")
app = FastAPI(title="Document QA System", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(monitoring_router)  # /api/monitoring/tokens/stats + /recent
config = Configuration.from_env()
store = MemoryStore(db_path=config.memory_db_path or "")
set_memory_store(store)
kg = KnowledgeGraph()
chapter_agent = ChapterizerAgent(config)
vs = DocumentVectorStore()
# Initialize RAG tool with vector store and knowledge graph
init_rag_tool(vs, kg)
chapters_cache = {}
uploaded_files = {}
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CHAPTERS_DB = PROJECT_ROOT / "storage" / "memory.db"

def _get_chapters_conn():
    import sqlite3
    CHAPTERS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHAPTERS_DB))
    conn.execute("CREATE TABLE IF NOT EXISTS doc_chapters (filename TEXT PRIMARY KEY, chapters_json TEXT, updated_at REAL)")
    conn.commit()
    return conn

def _load_chapters(filename=""):
    conn = _get_chapters_conn()
    if filename:
        row = conn.execute("SELECT chapters_json FROM doc_chapters WHERE filename = ?", (filename,)).fetchone()
        conn.close()
        return json.loads(row[0]) if row else []
    rows = conn.execute("SELECT filename, chapters_json FROM doc_chapters").fetchall()
    conn.close()
    return {r[0]: json.loads(r[1]) for r in rows}

def _save_chapters(filename, chapters):
    conn = _get_chapters_conn()
    conn.execute(
        "INSERT OR REPLACE INTO doc_chapters (filename, chapters_json, updated_at) VALUES (?, ?, ?)",
        (filename, json.dumps(chapters, ensure_ascii=False), __import__("time").time())
    )
    conn.commit()
    conn.close()

class ChapterDetectRequest(BaseModel):
    filepaths: list[str]

class ProcessRequest(BaseModel):
    filepaths: list[str]
    selected_chapters: list[str] = []

class SaveChaptersRequest(BaseModel):
    filename: str
    chapters: list[dict] = []

class ChatRequest(BaseModel):
    question: str
    session_id: str = ""
    doc_filter: list[str] = []

@app.post("/api/files/upload")

async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more document files."""
    saved = []
    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in (".pdf", ".txt", ".md", ".docx"):
            raise HTTPException(400, f"Unsupported format: {suffix}")
        dest = UPLOAD_DIR / f.filename
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        uploaded_files[f.filename] = str(dest)
        saved.append(f.filename)
    return {"uploaded": saved, "total": len(saved)}

@app.get("/api/files/list")

async def list_files():
    """List currently uploaded files (synced with disk)."""
    global uploaded_files
    # Remove entries for files that no longer exist on disk
    to_remove = [fn for fn, fp in uploaded_files.items() if not Path(fp).exists()]
    for fn in to_remove:
        del uploaded_files[fn]
    # Scan uploads dir for orphaned files
    if UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.iterdir():
            if p.is_file() and p.name not in uploaded_files:
                uploaded_files[p.name] = str(p)
    return {"files": list(uploaded_files.keys())}

@app.delete("/api/files/{filename}")

async def delete_file(filename: str):
    """Delete an uploaded file."""
    global uploaded_files
    if filename not in uploaded_files:
        raise HTTPException(404, "File not found")
    # Remove from disk
    filepath = uploaded_files.pop(filename)
    try:
        Path(filepath).unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to delete file on disk: %s", e)
    # Also check uploads dir for duplicates
    alt_path = UPLOAD_DIR / filename
    try:
        alt_path.unlink(missing_ok=True)
    except Exception:
        pass
    return {"deleted": filename, "remaining": list(uploaded_files.keys())}

@app.post("/api/chapters/detect")

async def detect_chapters_api(req: ChapterDetectRequest):
    """Detect chapters with SSE progress streaming. Files processed in parallel."""
    global chapters_cache
    chapters_cache.clear()
    if not req.filepaths:
        return {"chapters": [], "message": "No files uploaded"}

    total = len(req.filepaths)
    max_concurrency = config.chapter_detect_concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    event_queue: asyncio.Queue = asyncio.Queue()
    cache_lock = asyncio.Lock()

    async def _detect_one_file(idx: int, fp: str) -> list[dict]:
        """Process a single file: parse preview → detect chapters → cache. Returns chapter dicts."""
        import time as _time
        chapters_found: list[dict] = []
        full_path = uploaded_files.get(fp, "")
        if not full_path:
            full_path = str(UPLOAD_DIR / fp)
        if not Path(full_path).exists():
            await event_queue.put({"type": "error", "file": fp, "msg": "File not found"})
            return chapters_found

        async with semaphore:
            try:
                _t0 = _time.time()
                fname = Path(full_path).name

                # ---- Step 1: Parse first 20 pages ----
                await event_queue.put({"type": "progress", "file": fname, "stage": "preview parse...",
                                        "file_idx": idx + 1, "file_total": total})
                preview_doc = await asyncio.to_thread(parse_document, full_path, max_pages=20)
                _t1 = _time.time()
                logger.info("[API CHAPTERIZE] %s | preview parse (20p): %.1fs | chars=%d",
                            fname, _t1 - _t0, len(preview_doc.content))

                # ---- 空内容早期检测（扫描版 PDF / 损坏文件） ----
                if not preview_doc.content.strip():
                    msg = "文档无文字内容，可能是扫描版PDF（图片无文字层），请使用OCR处理后重新上传"
                    logger.warning("[API CHAPTERIZE] %s | empty content, likely scanned PDF", fname)
                    await event_queue.put({"type": "error", "file": fp, "msg": msg})
                    return chapters_found

                # ---- Step 2: LLM detection (reflection loop) ----
                await event_queue.put({"type": "progress", "file": fname, "stage": "detecting...",
                                        "file_idx": idx + 1, "file_total": total})
                try:
                    preview_chapters = await chapter_agent.detect_all(preview_doc, None)
                except RuntimeError:
                    logger.error("[API CHAPTERIZE] %s | detection FAILED", fname)
                    await event_queue.put({"type": "error", "file": fp, "msg": "detection failed"})
                    return chapters_found

                _t2 = _time.time()
                logger.info("[API CHAPTERIZE] %s | detect_all: %.1fs | chapters=%d",
                            fname, _t2 - _t1, len(preview_chapters))

                # ---- Step 3: Cache chapters (lock-protected) ----
                async with cache_lock:
                    for ch in preview_chapters:
                        label = f"[{fname}] {ch.title}"
                        chapters_found.append({
                            "label": label,
                            "title": ch.title,
                            "filename": fname,
                            "level": ch.level,
                            "text_preview": ch.text[:200] if ch.text else "",
                            "text_length": len(ch.text) if ch.text else 0,
                        })
                        chapters_cache[label] = {
                            "filename": fname,
                            "title": ch.title,
                            "text": ch.text,
                            "level": ch.level,
                            "start_marker": ch.start_marker,
                            "full_path": full_path,
                        }

                _t3 = _time.time()
                logger.info("[API CHAPTERIZE] %s | TOTAL: %.1fs (parse=%.1fs detect=%.1fs cache=%.1fs)",
                            fname, _t3 - _t0, _t1 - _t0, _t2 - _t1, _t3 - _t2)

                await event_queue.put({"type": "file_done", "file": fname,
                                        "chapters_found": len(preview_chapters),
                                        "file_idx": idx + 1, "file_total": total})

            except Exception as e:
                logger.error("Chapter detection failed for %s: %s", fp, e)
                await event_queue.put({"type": "error", "file": fp, "msg": str(e)})

        return chapters_found

    async def _worker_with_done(i: int, fp: str):
        """Wrap _detect_one_file to guarantee a sentinel is always put."""
        try:
            return await _detect_one_file(i, fp)
        except Exception as e:
            logger.exception("[API CHAPTERIZE] worker %d crashed: %s", i, e)
            await event_queue.put({"type": "error", "file": fp, "msg": f"worker crash: {e}"})
            return []
        finally:
            await event_queue.put(None)  # sentinel — 保证一定发送

    async def event_stream():
        try:
            # 立即发出首事件，建立 SSE 连接
            yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

            # gather() 本身就会调度所有 coroutine 并发执行，返回 Future，无需 create_task
            gather_future = asyncio.gather(
                *[_worker_with_done(i, fp) for i, fp in enumerate(req.filepaths)],
                return_exceptions=True,
            )

            all_chapters: list[dict] = []
            sentinels = 0

            # 阻塞读 queue，直到收齐所有 sentinel
            while sentinels < total:
                evt = await event_queue.get()
                if evt is None:  # sentinel
                    sentinels += 1
                    continue
                try:
                    yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logger.warning("[API CHAPTERIZE] failed to serialize event: %s", e)

            # 收集结果
            results = await gather_future
            for file_chapters in results:
                if isinstance(file_chapters, list):
                    all_chapters.extend(file_chapters)

            # Fallback if nothing found
            if not all_chapters:
                for fp in req.filepaths:
                    name = Path(fp).name
                    label = f"[{name}] all"
                    all_chapters.append({
                        "label": label, "title": "all", "filename": name,
                        "level": 1, "text_preview": "", "text_length": 0,
                    })
                    chapters_cache[label] = {"filename": name, "title": "all", "text": "", "level": 1}

            yield f"data: {json.dumps({'type': 'complete', 'chapters': all_chapters, 'total': len(all_chapters)})}\n\n"

        except Exception as e:
            logger.exception("[API CHAPTERIZE] event_stream crashed")
            yield f"data: {json.dumps({'type': 'error', 'msg': f'server error: {e}'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'chapters': [], 'total': 0})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chapters/save")
async def save_chapters(req: SaveChaptersRequest):
    """Save detected chapters to persistent storage."""
    filename = req.filename
    chapters = req.chapters
    if not filename:
        raise HTTPException(400, "filename is required")
    _save_chapters(filename, chapters)
    return {"status": "ok"}

@app.get("/api/chapters/{filename}")
async def get_chapters(filename: str):
    """Get cached chapters for a file."""
    chapters = _load_chapters(filename)
    return {"chapters": chapters}

@app.post("/api/knowledge/process")

async def process_knowledge(req: ProcessRequest):
    """Process selected chapters with SSE progress streaming."""
    import time as _time
    
    if not req.filepaths:
        raise HTTPException(400, "No files provided")
    
    async def event_stream():
        _total_t0 = _time.time()
        session_id = str(uuid.uuid4())[:12]
        
        def _sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        yield _sse({"type": "status", "text": "正在解析文档...", "session_id": session_id})
        await asyncio.sleep(0.1)
        
        # ---- Step 0: Group chapters by source file ----
        file_chapters: dict[str, list[dict]] = {}
        for label in (req.selected_chapters or []):
            info = chapters_cache.get(label, {})
            fp = info.get("full_path", "")
            if not fp:
                fp = str(UPLOAD_DIR / info.get("filename", ""))
            if fp not in file_chapters:
                file_chapters[fp] = []
            file_chapters[fp].append({
                "label": label,
                "title": info.get("title", ""),
                "start_marker": info.get("start_marker", ""),
            })
        
        # ---- Step 1: Full parse + split ----
        documents: list[Document] = []
        chapter_map: dict[str, str] = {}
        total_files = len(file_chapters)
        
        for idx, (full_path, chap_infos) in enumerate(file_chapters.items()):
            fname = Path(full_path).name
            yield _sse({"type": "progress", "stage": f"解析: {fname}", "pct": int((idx / max(total_files, 1)) * 25)})
            await asyncio.sleep(0.1)
            
            full_doc = await asyncio.to_thread(parse_document, full_path)
            markers = [{"title": ci["title"], "start_marker": ci["start_marker"]} for ci in chap_infos]
            full_chapters = _split_by_meta(full_doc.content, markers) if markers else []
            
            if not full_chapters:
                for ci in chap_infos:
                    info = chapters_cache.get(ci["label"], {})
                    text = info.get("text", "")
                    if text.strip():
                        documents.append(Document(filename=fname, content=text))
                        chapter_map[ci["label"]] = ci["title"]
                continue
            
            for fc in full_chapters:
                matched_ci = next((ci for ci in chap_infos if ci["title"] == fc.title), chap_infos[0] if chap_infos else None)
                if matched_ci and fc.text.strip():
                    documents.append(Document(filename=fname, content=fc.text))
                    chapter_map[matched_ci["label"]] = fc.title
        
        if not documents:
            yield _sse({"type": "error", "msg": "No chapters selected or all failed to parse"})
            return
        
        yield _sse({"type": "progress", "stage": f"已解析 {len(documents)} 个章节, 开始分块...", "pct": 30})
        await asyncio.sleep(0.1)
        
        # ---- Step 2: Chunk ----
        all_chunks = []
        for idx, doc in enumerate(documents):
            chapter_title = ""
            for label, title in chapter_map.items():
                if title and title in doc.filename + doc.content[:50]:
                    chapter_title = title
                    break
            if not chapter_title:
                for label, title in chapter_map.items():
                    cache_info = chapters_cache.get(label, {})
                    cache_text = cache_info.get("text", "")
                    if cache_text and cache_text[:50] == doc.content[:50]:
                        chapter_title = title
                        break
            
            chunks = await asyncio.to_thread(
                chunk_doc_func, doc,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                chapter_title=chapter_title,
            )
            all_chunks.extend(chunks)
            if idx % 3 == 0 or idx == len(documents) - 1:
                yield _sse({"type": "progress", "stage": f"分块: {idx+1}/{len(documents)}", "pct": 35 + int((idx / max(len(documents), 1)) * 15)})
                await asyncio.sleep(0.1)
        
        yield _sse({"type": "progress", "stage": f"共 {len(all_chunks)} 个文本片段, 开始索引...", "pct": 55})
        await asyncio.sleep(0.1)
        
        # ---- Step 3: Index (增量：只移除当前文档旧数据，不重建整个库) ----
        docs_to_refresh = set(c.doc_filename for c in all_chunks)
        for doc_name in docs_to_refresh:
            vs.remove_document(doc_name)
        chunk_dicts = [
            {"chunk_id": c.chunk_id, "text": c.text, "doc_filename": c.doc_filename,
             "chapter_title": c.chapter_title, "chunk_index": c.chunk_index}
            for c in all_chunks
        ]
        await asyncio.to_thread(vs.index_chunks, chunk_dicts)
        
        yield _sse({"type": "progress", "stage": "索引完成, 开始提取知识图谱...", "pct": 65})
        await asyncio.sleep(0.1)
        
        # ---- Step 4: Extract Knowledge Graph (并行 batch) ----
        kg.clear()
        result = await extract_full_document_async(all_chunks, config, kg)
        stats = kg.stats()
        
        yield _sse({"type": "progress", "stage": f"提取完成: {stats['concepts']} 概念, {stats['relations']} 关系", "pct": 95})
        await asyncio.sleep(0.1)
        
        all_concepts = kg.get_all_concepts()
        cats_display = {"definition": "定义", "theorem": "定理", "method": "方法",
                        "example": "示例", "concept": "概念"}
        categories_summary = {}
        for cat, cat_cn in cats_display.items():
            cat_concepts = [c for c in all_concepts if c.category == cat]
            if cat_concepts:
                categories_summary[cat_cn] = [
                    {"name": c.name, "description": c.description[:80]}
                    for c in cat_concepts[:10]
                ]
        
        elapsed = _time.time() - _total_t0
        
        yield _sse({"type": "complete", "session_id": session_id,
                     "documents_processed": len(documents),
                     "chunks_total": len(all_chunks),
                     "concepts_extracted": result["concepts_extracted"],
                     "relations_extracted": result["relations_extracted"],
                     "stats": {"concepts": stats["concepts"], "relations": stats["relations"],
                               "categories": stats.get("categories", {})},
                     "categories_summary": categories_summary,
                     "elapsed": round(elapsed, 1)})
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.delete("/api/knowledge/clear")

async def clear_knowledge():
    """Clear all knowledge graph and vector store data."""
    kg.clear()
    vs.clear()
    chapters_cache.clear()
    return {"status": "cleared"}

@app.get("/api/knowledge/stats")

async def get_stats():
    """Get knowledge graph statistics."""
    stats = kg.stats()
    doc_names = kg.get_doc_names()
    return {
        "concepts": stats["concepts"],
        "relations": stats["relations"],
        "categories": stats.get("categories", {}),
        "documents": doc_names,
    }

@app.get("/api/knowledge/documents")

async def list_documents():
    """List indexed document names for filtering. Clears stale data if no files."""
    global uploaded_files
    if not uploaded_files:
        # No files uploaded, clear any stale KG data
        kg.clear()
        return {"documents": []}
    return {"documents": kg.get_doc_names()}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Answer a question with SSE streaming."""
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    session_id = req.session_id or str(uuid.uuid4())[:12]
    filter_docs = set(req.doc_filter) if req.doc_filter else set()
    try:
        store.add_chat_message(session_id, "user", req.question)
        store.save_session(session_id, req.question[:40], "[]", "")
    except Exception:
        pass

    try:
        chat_history = store.get_chat_history(session_id) if session_id else []
    except Exception:
        chat_history = []

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        yield _sse({"type": "status", "text": "正在分析问题...", "session_id": session_id})
        await asyncio.sleep(0.1)

        reply = ""
        tool_calls_log = []
        rounds = 0
        try:
            agent = ReflectionAgent(config)
            result = await agent.answer(req.question, doc_filter=filter_docs if filter_docs else None, chat_history=chat_history)
            reply = result["reply"]
            rounds = result.get("rounds", 0)
            tool_calls_log = result.get("tool_calls", [])
            if tool_calls_log:
                tools_used = set(tc["tool"] for tc in tool_calls_log)
                yield _sse({"type": "status", "text": f"已检索 {len(tool_calls_log)} 次，正在生成回答...", "session_id": session_id})
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("QA failed: %s", e)
            reply = f"处理出错: {e}"

        # Stream reply character by character
        yield _sse({"type": "reply_start", "session_id": session_id})
        chunk_size = 8
        for i in range(0, len(reply), chunk_size):
            yield _sse({"type": "reply_chunk", "text": reply[i:i+chunk_size]})
            await asyncio.sleep(0.02)

        # 保存助手回复到数据库（在 done 事件前，确保流快速关闭）
        try:
            store.add_chat_message(session_id, "assistant", reply)
        except Exception:
            pass

        yield _sse({"type": "done", "session_id": session_id, "rounds": rounds, "tool_calls": len(tool_calls_log)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/sessions")

async def list_sessions():
    """List all saved sessions."""
    sessions = store.list_sessions()
    return {"sessions": sessions}

@app.get("/api/sessions/{session_id}")

async def get_session(session_id: str):
    """Get session details and chat history."""
    data = store.get_session(session_id)
    if not data:
        raise HTTPException(404, "Session not found")
    chat_history = store.get_chat_history(session_id)
    return {
        "session_id": session_id,
        "topic": data.get("topic", ""),
        "report": data.get("report", ""),
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in chat_history
        ],
    }

@app.delete("/api/sessions/{session_id}")

async def delete_session(session_id: str):
    """Delete a session and its messages."""
    store.delete_session(session_id)
    return {"deleted": session_id}

@app.get("/api/health")

async def health():
    return {"status": "ok"}

frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
