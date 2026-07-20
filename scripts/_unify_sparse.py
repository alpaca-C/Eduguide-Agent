"""Replace FTS5 with jieba BM25 in vector_store.py and retriever.py"""
import re

# ── 1. Replace FTS5 search in vector_store.py ──
with open('src/agents/qa/vector_store.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove FTS5 _search_sparse and replace with jieba BM25
old_search = content.find('def _search_sparse(self')
next_def = content.find('    def search(self', old_search + 1)

new_search = '''    # ── Sparse search (jieba BM25, in-memory) ────────────────────

    def _build_bm25(self):
        """Build in-memory jieba BM25 index from all chunk texts."""
        import math
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

    def _search_sparse(self, query: str, top_k: int = 10, filter_docs: set[str] | None = None) -> list[dict]:
        """jieba BM25 keyword search (in-memory)."""
        import math
        import jieba as _jieba

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

'''

content = content[:old_search] + new_search + content[next_def:]

# Remove FTS5 INSERT in index_chunks
content = content.replace(
    "        # Sparse: FTS5 — normalize CJK text by removing inter-character spaces",
    "        # Sparse: invalidate BM25 index — rebuilt lazily on next search"
)
# Remove the FTS5 INSERT block
old_fts = '''        # Sparse: FTS5 — normalize CJK text by removing inter-character spaces
        # (OCR often inserts spaces between Chinese characters, breaking tokenization)
        def _normalize_cjk(text: str) -> str:
            import re
            return re.sub(r'(?<=[一-鿿]) +(?=[一-鿿])', '', text)
        fts_rows = [(cid, _normalize_cjk(t), m["doc_filename"], m["chapter_title"])
                      for cid, t, m in zip(new_ids, new_texts, new_metas)]
        self._fts_conn.executemany(
            "INSERT INTO chunk_fts (chunk_id, text, doc_filename, chapter_title) VALUES (?, ?, ?, ?)",
            fts_rows,
        )
        self._fts_conn.commit()
'''
content = content.replace(old_fts, "        self._bm25_valid = False  # invalidate BM25 index\n")

# Remove FTS5 connection init
old_fts_init = '''        # --- Sparse: SQLite FTS5 ---
        fts_path = str(Path(persist_dir) / "fts_index.db")
        self._fts_conn = sqlite3.connect(fts_path, check_same_thread=False)
        self._fts_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id, text, doc_filename, chapter_title,
                tokenize='unicode61 remove_diacritics 2'
            )
        """)
        self._fts_conn.commit()
'''
content = content.replace(old_fts_init, "        # --- Sparse: jieba BM25 (built lazily on first search) ---\n        self._bm25_valid = False\n")

# Remove FTS5 DELETE in remove_document
content = content.replace("self._fts_conn.execute(\"DELETE FROM chunk_fts WHERE doc_filename = ?\", (doc_filename,))\n            self._fts_conn.commit()", "pass  # BM25 rebuilt on next search")

# Remove FTS5 DELETE in remove_by_chapter
content = content.replace("self._fts_conn.execute(\n                \"DELETE FROM chunk_fts WHERE doc_filename = ? AND chapter_title = ?\",\n                (doc_filename, chapter_title),\n            )\n            self._fts_conn.commit()", "pass  # BM25 rebuilt on next search")

# Remove FTS5 DELETE in clear
content = content.replace('''self._fts_conn.execute("DELETE FROM chunk_fts")
        self._fts_conn.commit()''', "self._bm25_valid = False")

# Update class docstring
content = content.replace("Hybrid vector store: ChromaDB (dense) + SQLite FTS5 (sparse)", "Hybrid vector store: ChromaDB (dense) + jieba BM25 (sparse)")

# Update file header comment
content = content.replace("# DocumentVectorStore — ChromaDB dense + SQLite FTS5 sparse hybrid search", "# DocumentVectorStore — ChromaDB dense + jieba BM25 sparse hybrid search")
content = content.replace("# Sparse: SQLite FTS5 with BM25 (keyword, exact term match)", "# Sparse: jieba BM25 (in-memory, keyword match)")

with open('src/agents/qa/vector_store.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("vector_store.py: FTS5 → jieba BM25")

# ── 2. Simplify retriever.py: use vs._search_sparse instead of own jieba BM25 ──
with open('src/evaluation/retriever.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove _search_sparse_jieba method
old_jieba_start = content.find('def _search_sparse_jieba(self')
if old_jieba_start > 0:
    next_method = content.find('    def _search_hybrid(self', old_jieba_start + 1)
    if next_method < 0:
        next_method = content.find('    def _search_supplement(self', old_jieba_start + 1)
    if next_method > old_jieba_start:
        content = content[:old_jieba_start] + content[next_method:]
        print("retriever.py: removed _search_sparse_jieba")

# Update _search_single to use vs._search_sparse
content = content.replace(
    'if self.strategy == "sparse":\n            return self._search_sparse_jieba(query, top_k)',
    'if self.strategy == "sparse":\n            return self._vs._search_sparse(query, top_k, filter_docs=f)'
)

with open('src/evaluation/retriever.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("All done")
