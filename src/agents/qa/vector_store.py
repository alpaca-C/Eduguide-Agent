# DocumentVectorStore — ChromaDB dense + jieba BM25 sparse hybrid search
#
# Dense:  Qwen3-Embedding-0.6B → ChromaDB (semantic, cross-lingual)
# Sparse: jieba BM25 (in-memory, keyword match)
#
# Combined via RRF (Reciprocal Rank Fusion) in rag_search.py

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import hashlib

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

logger = logging.getLogger(__name__)


class CrossLingualEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function using Qwen3-Embedding-0.6B via sentence-transformers.
    Cross-lingual: Chinese queries match English documents and vice versa.
    **Lazy loading**: model not downloaded until first use.
    """

    def __init__(self):
        self._model_source = (
            os.environ.get("EMBEDDING_MODEL_PATH")
            or os.environ.get("BGE_M3_MODEL_PATH")
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self._hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        self._model = None
        self._dim = None
        self._lock = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import threading
        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer
            mirror_info = f" (mirror: {self._hf_endpoint})" if self._hf_endpoint else ""
            try:
                logger.info("Loading Qwen3-Embedding-0.6B from '%s'%s ...", self._model_source, mirror_info)
                self._model = SentenceTransformer(self._model_source)
                self._dim = self._model.get_sentence_embedding_dimension()
                logger.info("Qwen3-Embedding-0.6B loaded, dim=%d", self._dim)
            except Exception as e:
                if any(x in type(e).__name__ or x in str(e) for x in ["Connection", "Timeout", "WinError"]):
                    logger.error("Qwen3-Embedding download failed (network). Set HF_ENDPOINT in .env. Error: %s", e)
                raise

    @property
    def model(self):
        self._ensure_model()
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        self._ensure_model()
        embeddings = self._model.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


class DocumentVectorStore:
    """Hybrid vector store: ChromaDB (dense) + jieba BM25 (sparse).

    Incremental indexing: new chunks appended without rebuilding.
    Each chunk carries doc_filename + chapter_title metadata.
    """

    COLLECTION_NAME = "document_chunks"

    def __init__(self, storage_dir: str = ""):
        if storage_dir:
            persist_dir = str(Path(storage_dir) / "chroma")
        else:
            persist_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "chroma")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # --- Dense: ChromaDB ---
        self._ef = CrossLingualEmbeddingFunction()
        self._all_texts: list[str] = []
        self._all_metas: list[dict] = []
        self._content_hashes: set[str] = set()
        self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self._ensure_collection()

        # --- Sparse: jieba BM25 (built lazily on first search) ---
        self._bm25_valid = False

    @staticmethod
    def _text_hash(text: str) -> str:
        """Deterministic content hash (MD5). Unlike Python's hash(),
        this survives process restarts, making dedup work correctly."""
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    # ── ChromaDB ───────────────────────────────────────────────

    def _ensure_collection(self):
        try:
            self._collection = self._client.get_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._ef,
            )
            existing = self._collection.get(include=["documents", "metadatas"])
            if existing and existing.get("documents"):
                self._all_texts = list(existing["documents"])
                self._all_metas = list(existing.get("metadatas", []))
                self._content_hashes = {self._text_hash(t) for t in self._all_texts}
            logger.info("ChromaDB: loaded %d chunks from existing collection", len(self._all_texts))
        except Exception as _e:
            # Only create a NEW collection if one genuinely doesn't exist.
            # Silently replacing an existing collection would wipe all data.
            existing_names = [c.name for c in self._client.list_collections()]
            if self.COLLECTION_NAME in existing_names:
                logger.error(
                    "ChromaDB: collection '%s' exists but failed to load: %s. "
                    "Data is safe — fix the error above to recover.",
                    self.COLLECTION_NAME, _e,
                )
                raise
            logger.info(
                "ChromaDB: collection '%s' not found, creating new one",
                self.COLLECTION_NAME,
            )
            self._collection = self._client.create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

    def remove_document(self, doc_filename: str):
        if not doc_filename:
            return
        keep_idx = [i for i, m in enumerate(self._all_metas) if m.get("doc_filename") != doc_filename]
        removed = len(self._all_texts) - len(keep_idx)
        if removed == 0:
            return
        self._all_texts = [self._all_texts[i] for i in keep_idx]
        self._all_metas = [self._all_metas[i] for i in keep_idx]
        self._content_hashes = {self._text_hash(t) for t in self._all_texts}
        try:
            self._collection.delete(where={"doc_filename": doc_filename})
            pass  # BM25 rebuilt on next search
        except Exception as e:
            logger.warning("ChromaDB delete doc '%s' failed: %s", doc_filename, e)
        logger.info("ChromaDB+BM25: removed %d chunks for '%s', %d remaining", removed, doc_filename, len(self._all_texts))

    def remove_by_chapter(self, doc_filename: str, chapter_title: str) -> int:
        """Remove all chunks belonging to a specific chapter (granular delete).

        Returns the number of chunks removed.
        """
        if not doc_filename:
            return 0

        # Find chunk IDs to delete (ChromaDB where only supports single-field,
        # so filter in Python and delete by IDs)
        ids_to_delete = []
        data = self._collection.get(include=["metadatas"])
        for cid, meta in zip(data["ids"], data["metadatas"]):
            if (meta or {}).get("doc_filename") == doc_filename and \
               (meta or {}).get("chapter_title") == chapter_title:
                ids_to_delete.append(cid)

        if not ids_to_delete:
            return 0

        removed = len(ids_to_delete)
        try:
            self._collection.delete(ids=ids_to_delete)
            pass  # BM25 rebuilt on next search
        except Exception as e:
            logger.warning("ChromaDB+BM25 chapter delete failed: %s", e)
            return 0

        # Rebuild in-memory state from ChromaDB (source of truth)
        reloaded = self._collection.get(include=["documents", "metadatas"])
        self._all_texts = list(reloaded["documents"])
        self._all_metas = list(reloaded["metadatas"])
        self._content_hashes = {self._text_hash(t) for t in self._all_texts}

        logger.info("ChromaDB+FTS: removed %d chunks for '%s' / '%s', %d remaining",
                     removed, doc_filename, chapter_title, len(self._all_texts))
        return removed

    def get_imported_chapter_titles(self, doc_filename: str) -> set[str]:
        """Return chapter titles that have chunks in the vector store."""
        try:
            results = self._collection.get(
                where={"doc_filename": doc_filename},
                include=["metadatas"],
            )
            titles = set()
            for m in (results.get("metadatas") or []):
                t = (m or {}).get("chapter_title", "")
                if t:
                    titles.add(t)
            logger.info(
                "VS.get_imported_chapter_titles(%s): %d metas → %d titles: %s",
                doc_filename, len(results.get("metadatas") or []),
                len(titles), list(titles)[:5],
            )
            return titles
        except Exception as _e:
            logger.warning("VS.get_imported_chapter_titles failed: %s", _e)
            return set()

    def has_document(self, doc_filename: str) -> bool:
        """Check if any chunks exist for this document."""
        return any(
            m.get("doc_filename") == doc_filename
            for m in self._all_metas
        )

    def index_chunks(self, chunks: list[dict], doc_id: str = ""):
        if not chunks:
            return
        valid = [c for c in chunks if c.get("text", "").strip()]
        if not valid:
            return
        # Dedup
        new_entries: list[dict] = []
        for c in valid:
            text = c["text"][:2000]
            h = self._text_hash(text)
            if h not in self._content_hashes:
                self._content_hashes.add(h)
                new_entries.append({**c, "text": text})
        if not new_entries:
            logger.info("ChromaDB: all %d chunks already indexed", len(valid))
            return
        skipped = len(valid) - len(new_entries)
        new_texts = [c["text"] for c in new_entries]
        new_metas = [{
            "doc_filename": c.get("doc_filename", ""),
            "chapter_title": c.get("chapter_title", ""),
            "chunk_index": c.get("chunk_index", 0),
        } for c in new_entries]

        # Dense: ChromaDB
        # Use provided chunk_id when available (BEIR eval needs BEIR doc_ids),
        # otherwise auto-generate sequential IDs.
        self._all_texts.extend(new_texts)
        self._all_metas.extend(new_metas)
        embeddings = self._ef(new_texts)
        existing_count = self._collection.count()

        # Load existing IDs from ChromaDB to prevent cross-batch collisions
        # (e.g. processing Ch2 after Ch6 already indexed — both start from _0)
        try:
            existing_ids = set(self._collection.get(include=[])["ids"])
        except Exception:
            existing_ids = set()

        # Generate unique IDs — handle multi-chapter processing where
        # the same filename produces duplicate chunk_ids.
        seen_ids: set[str] = set(existing_ids)  # pre-seed with existing
        new_ids = []
        for i, c in enumerate(new_entries):
            base_id = c.get("chunk_id") or f"chunk_{existing_count + i}"
            cid = base_id
            suffix = 1
            while cid in seen_ids:
                cid = f"{base_id}_v{suffix}"
                suffix += 1
            seen_ids.add(cid)
            new_ids.append(cid)

        self._collection.add(ids=new_ids, documents=new_texts, metadatas=new_metas, embeddings=embeddings)

        # Sparse: invalidate BM25 index — rebuilt lazily on next search
        self._bm25_valid = False

        logger.info("ChromaDB+BM25: +%d new (skipped %d), total %d",
                     len(new_ids), skipped, len(self._all_texts))

    # ── Dense search ───────────────────────────────────────────

    def _search_dense(self, query: str, top_k: int = 10, filter_docs: set[str] | None = None) -> list[dict]:
        try:
            where_filter = {"doc_filename": {"$in": list(filter_docs)}} if filter_docs else None
            query_embedding = self._ef.model.encode(query, normalize_embeddings=True, show_progress_bar=False).tolist()
            kwargs = dict(n_results=top_k)
            if where_filter:
                kwargs["where"] = where_filter
            results = self._collection.query(query_embeddings=[query_embedding], **kwargs)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            out = []
            for i, d in enumerate(docs):
                if d:
                    out.append({
                        "chunk_id": ids[i] if i < len(ids) else "",
                        "text": d,
                        "doc_filename": metas[i].get("doc_filename", "") if i < len(metas) else "",
                        "chapter_title": metas[i].get("chapter_title", "") if i < len(metas) else "",
                        "source": "dense",
                    })
            return out
        except Exception as e:
            logger.warning("Dense search failed: %s", e)
            return []

    # ── Sparse search (FTS5) ───────────────────────────────────

        # ── Sparse search (jieba BM25, in-memory) ────────────────────

    def _build_bm25(self):
        """Build in-memory jieba BM25 index from all chunk texts."""
        import math, re
        from collections import defaultdict
        import jieba as _jieba

        idx = defaultdict(lambda: defaultdict(float))
        self._bm25_docs: dict[str, dict] = {}
        doc_count = 0
        total_len = 0

        ids = self._collection.get(include=[])['ids']
        for i, (text, meta) in enumerate(zip(self._all_texts, self._all_metas)):
            cid = ids[i] if i < len(ids) else f"chunk_{i}"
            clean = re.sub(r'(?<=[一-鿿]) +(?=[一-鿿])', '', text or '')
            words = [w.strip() for w in _jieba.cut(clean) if len(w.strip()) >= 2]
            self._bm25_docs[cid] = {
                'text': clean, 'doc_filename': meta.get('doc_filename', ''),
                'chapter_title': meta.get('chapter_title', ''), 'word_count': len(words),
            }
            for w in set(words):
                idx[w][cid] = words.count(w)
            doc_count += 1
            total_len += len(words)

        self._bm25_idx = idx
        self._bm25_N = doc_count
        self._bm25_avgdl = total_len / max(1, doc_count)
        self._bm25_valid = True
        logger.info("BM25 index built: %d docs, avg len=%.1f tokens", doc_count, self._bm25_avgdl)
        # Persist to SQLite for hot restart
        try:
            import pickle, sqlite3
            from pathlib import Path as _P
            db_path = str(_P(__file__).resolve().parent.parent.parent.parent / "data" / "chroma" / "bm25_cache.db")
            db_path = _P(db_path).parent
            db_path = str(_P(db_path) / "bm25_cache.db")
            logger.info("[BM25] saving cache to %s", db_path)
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS bm25 (key TEXT PRIMARY KEY, data BLOB)")
            data_blob = pickle.dumps((dict(self._bm25_idx), self._bm25_docs, self._bm25_N, self._bm25_avgdl), protocol=4)
            conn.execute("INSERT OR REPLACE INTO bm25 (key, data) VALUES (?, ?)", ("index", data_blob))
            conn.commit(); conn.close()
            logger.info("[BM25] cache saved: %d KB", len(data_blob) // 1024)
        except Exception as e:
            logger.warning("[BM25] save failed: %s", e)

    def _search_sparse(self, query: str, top_k: int = 10, filter_docs: set[str] | None = None) -> list[dict]:
        """jieba BM25 keyword search (in-memory)."""
        import math, re
        import jieba as _jieba

        if not getattr(self, '_bm25_valid', False):
            # Try load persisted BM25 first
            try:
                import pickle, sqlite3
                from pathlib import Path as _P
                db_path = str(_P(__file__).resolve().parent.parent.parent.parent / "data" / "chroma" / "bm25_cache.db")
                conn = sqlite3.connect(db_path)
                row = conn.execute("SELECT data FROM bm25 WHERE key='index'").fetchone()
                conn.close()
                if row:
                    self._bm25_idx, self._bm25_docs, self._bm25_N, self._bm25_avgdl = pickle.loads(row[0])
                    self._bm25_valid = True
                    logger.info("BM25 loaded from cache: %d docs", self._bm25_N)
            except Exception: pass
        if not getattr(self, '_bm25_valid', False):
            self._build_bm25()

        clean = re.sub(r'(?<=[一-鿿]) +(?=[一-鿿])', '', query)
        qwords = [w.strip() for w in _jieba.cut(clean) if len(w.strip()) >= 2]
        if not qwords:
            return []

        k1, b = 1.5, 0.75
        scores: dict[str, float] = {}
        for qw in qwords:
            if qw not in self._bm25_idx:
                continue
            idf = math.log((self._bm25_N - len(self._bm25_idx[qw]) + 0.5)
                          / (len(self._bm25_idx[qw]) + 0.5) + 1)
            for cid, tf in self._bm25_idx[qw].items():
                doc = self._bm25_docs.get(cid)
                if not doc:
                    continue
                if filter_docs and doc.get('doc_filename', '') not in filter_docs:
                    continue
                dl = doc['word_count']
                bm25_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._bm25_avgdl))
                scores[cid] = scores.get(cid, 0.0) + bm25_score

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{
            'chunk_id': cid,
            'text': self._bm25_docs[cid]['text'][:200],
            'doc_filename': self._bm25_docs[cid]['doc_filename'],
            'chapter_title': self._bm25_docs[cid]['chapter_title'],
            'score': score, 'source': 'sparse',
        } for cid, score in sorted_items]

    def search(self, query: str, top_k: int = 5, filter_docs: set[str] | None = None) -> list[dict]:
        """Returns dense results only (backward compat). For hybrid, use search_hybrid()."""
        return self._search_dense(query, top_k, filter_docs)

    def search_hybrid(self, query: str, top_k: int = 5, filter_docs: set[str] | None = None) -> dict:
        """Returns dict with 'dense', 'sparse' keyed result lists for RRF fusion."""
        return {
            "dense": self._search_dense(query, top_k * 2, filter_docs),
            "sparse": self._search_sparse(query, top_k * 2, filter_docs),
        }

    def get_doc_names(self) -> list[str]:
        names: set[str] = set()
        for m in self._all_metas:
            fn = m.get("doc_filename", "")
            if fn:
                names.add(fn)
        return sorted(names)

    def clear(self):
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception as e:
            logger.debug("Collection delete: %s", e)
        self._all_texts = []
        self._all_metas = []
        self._content_hashes = set()
        self._bm25_valid = False
        self._ensure_collection()
