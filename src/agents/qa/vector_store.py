# DocumentVectorStore — ChromaDB dense + SQLite FTS5 sparse hybrid search
#
# Dense:  Qwen3-Embedding-0.6B → ChromaDB (semantic, cross-lingual)
# Sparse: SQLite FTS5 with BM25 (keyword, exact term match)
#
# Combined via RRF (Reciprocal Rank Fusion) in rag_search.py

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

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
    """Hybrid vector store: ChromaDB (dense) + SQLite FTS5 (sparse).

    Incremental indexing: new chunks appended without rebuilding.
    Each chunk carries doc_filename + chapter_title metadata.
    """

    COLLECTION_NAME = "document_chunks_qwen3"

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
        self._content_hashes: set[int] = set()
        self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self._ensure_collection()

        # --- Sparse: SQLite FTS5 ---
        fts_path = str(Path(persist_dir) / "fts_index.db")
        self._fts_conn = sqlite3.connect(fts_path, check_same_thread=False)
        self._fts_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id, text, doc_filename, chapter_title,
                tokenize='unicode61 remove_diacritics 2'
            )
        """)
        self._fts_conn.commit()

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
                self._content_hashes = {hash(t) for t in self._all_texts}
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
        self._content_hashes = {hash(t) for t in self._all_texts}
        try:
            self._collection.delete(where={"doc_filename": doc_filename})
            self._fts_conn.execute("DELETE FROM chunk_fts WHERE doc_filename = ?", (doc_filename,))
            self._fts_conn.commit()
        except Exception as e:
            logger.warning("ChromaDB delete doc '%s' failed: %s", doc_filename, e)
        logger.info("ChromaDB+FTS: removed %d chunks for '%s', %d remaining", removed, doc_filename, len(self._all_texts))

    def remove_by_chapter(self, doc_filename: str, chapter_title: str) -> int:
        """Remove all chunks belonging to a specific chapter (granular delete).

        Returns the number of chunks removed.
        """
        if not doc_filename:
            return 0
        keep_idx = [
            i for i, m in enumerate(self._all_metas)
            if not (m.get("doc_filename") == doc_filename and m.get("chapter_title") == chapter_title)
        ]
        removed = len(self._all_texts) - len(keep_idx)
        if removed == 0:
            return 0
        self._all_texts = [self._all_texts[i] for i in keep_idx]
        self._all_metas = [self._all_metas[i] for i in keep_idx]
        self._content_hashes = {hash(t) for t in self._all_texts}
        try:
            self._collection.delete(where={
                "doc_filename": doc_filename,
                "chapter_title": chapter_title,
            })
            self._fts_conn.execute(
                "DELETE FROM chunk_fts WHERE doc_filename = ? AND chapter_title = ?",
                (doc_filename, chapter_title),
            )
            self._fts_conn.commit()
        except Exception as e:
            logger.warning("ChromaDB+FTS chapter delete failed: %s", e)
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
            h = hash(text)
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
        self._all_texts.extend(new_texts)
        self._all_metas.extend(new_metas)
        embeddings = self._ef(new_texts)
        existing_count = self._collection.count()
        new_ids = [f"chunk_{existing_count + i}" for i in range(len(new_texts))]
        self._collection.add(ids=new_ids, documents=new_texts, metadatas=new_metas, embeddings=embeddings)

        # Sparse: FTS5
        fts_rows = [(cid, t, m["doc_filename"], m["chapter_title"])
                      for cid, t, m in zip(new_ids, new_texts, new_metas)]
        self._fts_conn.executemany(
            "INSERT INTO chunk_fts (chunk_id, text, doc_filename, chapter_title) VALUES (?, ?, ?, ?)",
            fts_rows,
        )
        self._fts_conn.commit()

        logger.info("ChromaDB+FTS: +%d new (skipped %d), total %d", len(new_ids), skipped, len(self._all_texts))

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
            out = []
            for i, d in enumerate(docs):
                if d:
                    out.append({
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

    def _search_sparse(self, query: str, top_k: int = 10, filter_docs: set[str] | None = None) -> list[dict]:
        """BM25 keyword search via SQLite FTS5."""
        try:
            # FTS5 query: escape special chars, wrap terms in quotes for phrase matching
            clean = query.replace('"', '').replace("'", "''")
            cursor = self._fts_conn.cursor()
            if filter_docs:
                placeholders = ",".join("?" * len(filter_docs))
                sql = f"""
                    SELECT chunk_id, text, doc_filename, chapter_title, rank
                    FROM chunk_fts WHERE chunk_fts MATCH ? AND doc_filename IN ({placeholders})
                    ORDER BY rank LIMIT ?
                """
                cursor.execute(sql, (clean, *filter_docs, top_k))
            else:
                cursor.execute(
                    "SELECT chunk_id, text, doc_filename, chapter_title, rank FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?",
                    (clean, top_k),
                )
            rows = cursor.fetchall()
            out = []
            for r in rows:
                out.append({
                    "text": r[1],
                    "doc_filename": r[2] or "",
                    "chapter_title": r[3] or "",
                    "source": "sparse",
                })
            logger.debug("FTS5: %d results for '%s'", len(out), query[:60])
            return out
        except Exception as e:
            # FTS5 may fail on malformed queries — fall back gracefully
            logger.debug("FTS5 search failed (may be malformed query): %s", e)
            return []

    # ── Public search API ──────────────────────────────────────

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
        self._fts_conn.execute("DELETE FROM chunk_fts")
        self._fts_conn.commit()
        self._ensure_collection()
