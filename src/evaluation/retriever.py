"""
Retriever adapter — wraps the project's search as a BEIR-compatible retriever.

BEIR expects:
    retriever.search(corpus, queries, top_k) → dict[str, dict[str, float]]
    where keys are query_id → {doc_id: score}
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ProjectRetriever:
    """Adapts the project's hybrid search to the standard retriever interface.

    Supports four strategies for ablation comparison:
      - "dense"   : ChromaDB only
      - "sparse"  : FTS5 only
      - "graph"   : Knowledge Graph only
      - "hybrid"  : dense + sparse + KG → RRF fusion (default)
    """

    def __init__(self, strategy: str = "hybrid", doc_filter: str = ""):
        self.strategy = strategy
        self.doc_filter = doc_filter  # e.g. "beir:nfcorpus" — only search docs matching this
        self._vs = None
        self._kg = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        from src.context import get_context
        ctx = get_context()
        self._vs = ctx.vector_store
        self._kg = ctx.knowledge_graph
        self._initialized = True
        # Try loading persisted indices
        ProjectRetriever._load_cache()

    def search(
        self,
        corpus: dict[str, dict],
        queries: dict[str, str],
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Standard BEIR retriever interface."""
        self._ensure_init()

        # Adaptive: batch-classify all queries upfront (1 LLM call vs N)
        if self.strategy == "adaptive" and len(queries) > 1:
            qtexts = list(queries.values())
            self._adaptive_weights = self._classify_queries_batch(qtexts)

        # Supplement: batch-rerank all queries in one Cross-Encoder pass
        if self.strategy == "supplement" and len(queries) > 1:
            return self._search_supplement_batch(queries, top_k)

        # Batch-optimize: if graph-dependent strategy with many queries,
        # pre-compute all concept scoring in one matrix operation.
        _graph_strategies = ("graph", "hybrid", "supplement")  # adaptive does its own fusion
        if self.strategy in _graph_strategies and len(queries) > 1:
            self._graph_batch_precompute(queries, top_k)

        results: dict[str, dict[str, float]] = {}
        for qi, (qid, query_text) in enumerate(queries.items()):
            self._query_index = qi
            self._current_qid = qid
            try:
                # Use precomputed graph scores if available
                if self.strategy in _graph_strategies and hasattr(self, '_graph_batch_scores'):
                    # Build results from precomputed scores
                    scored = self._graph_batch_scores.get(qid, {})
                    results[qid] = dict(list(scored.items())[:top_k])
                    continue

                retrieved = self._search_single(query_text, top_k * 2)
                if retrieved is None:
                    retrieved = []
                scored: dict[str, float] = {}
                for rank, item in enumerate(retrieved[:top_k]):
                    chunk_id = item.get("chunk_id", "")
                    if not chunk_id:
                        continue
                    score = item.get("rrf_score", item.get("score", 1.0 / (60 + rank + 1)))
                    scored[chunk_id] = float(score)
                results[qid] = scored
            except Exception as e:
                logger.warning("Search failed for query '%s': %s", qid, e)
                results[qid] = {}
        return results

    @property
    def _filter_set(self) -> set[str] | None:
        """Convert doc_filter string to a set for VS methods, or None if empty."""
        return {self.doc_filter} if self.doc_filter else None

    def _search_single(self, query: str, top_k: int) -> list[dict]:
        """Execute a single query with the configured strategy."""
        self._ensure_init()
        f = self._filter_set

        if self.strategy == "dense":
            return self._vs._search_dense(query, top_k, filter_docs=f)

        if self.strategy == "sparse":
            return self._vs._search_sparse(query, top_k, filter_docs=f)

        if self.strategy == "graph":
            return self._search_graph(query, top_k)

        if self.strategy in ("hybrid", "adaptive"):
            return self._search_hybrid(query, top_k)

        if self.strategy == "supplement":
            return self._search_supplement(query, top_k)

        return []

    # ── Persistent cache (SQLite-backed, survives process restart) ───

    @classmethod
    def _save_cache(cls, key: str, data):
        """Persist a cache entry to SQLite (BLOB)."""
        try:
            import sqlite3, pickle
            from pathlib import Path as _P
            db = _P(__file__).resolve().parent.parent.parent / 'data' / 'graph_cache.db'
            conn = sqlite3.connect(str(db))
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, data BLOB)")
            conn.execute("INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)",
                         (key, pickle.dumps(data, protocol=4)))
            conn.commit(); conn.close()
            logger.debug("[persist] saved '%s' (%d KB)", key, len(pickle.dumps(data, protocol=4)) // 1024)
        except Exception as e:
            logger.warning("[persist] save '%s' failed: %s", key, e)

    @classmethod
    def _load_cache(cls):
        """Load persisted caches on first init. Returns True if any loaded."""
        if getattr(cls, '_cache_tried', False):
            return
        cls._cache_tried = True
        try:
            import sqlite3, pickle
            from pathlib import Path as _P
            db = _P(__file__).resolve().parent.parent.parent / 'data' / 'graph_cache.db'
            conn = sqlite3.connect(str(db))
            for key in ['concept_embeddings', 'leiden_communities']:
                row = conn.execute("SELECT data FROM cache WHERE key=?", (key,)).fetchone()
                if row:
                    data = pickle.loads(row[0])
                    if key == 'concept_embeddings':
                        cls._concept_cache = data
                        logger.info("[persist] loaded concept embeddings")
                    elif key == 'leiden_communities':
                        cls._global_communities = data
                        logger.info("[persist] loaded leiden communities")
            conn.close()
        except Exception as e:
            logger.debug("[persist] cache load skipped: %s", e)

    def _graph_batch_precompute(self, queries: dict[str, str], top_k: int):
        """Pre-compute graph scores for all queries in one matrix operation.

        Encodes all queries at once → one matmul with concept embeddings →
        distributes chunk scores per query. Cuts graph eval from 108s → ~1s.
        """
        self._ensure_init()
        import numpy as np

        all_concepts = self._kg.get_all_concepts()
        if not all_concepts:
            self._graph_batch_scores = {}
            return

        # ── Ensure concept embeddings (class-level cache) ──
        concept_texts = [f"{c.name or ''}: {c.description or ''}" for c in all_concepts]
        import hashlib
        text_hash = hashlib.md5("|".join(concept_texts).encode()).hexdigest()
        if not hasattr(ProjectRetriever, '_concept_cache'):
            ProjectRetriever._concept_cache = {}
        cache = ProjectRetriever._concept_cache
        if text_hash not in cache:
            logger.info("[graph] encoding %d concepts for semantic search", len(concept_texts))
            emb = np.array(self._vs._ef(concept_texts))
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            cache[text_hash] = emb / norms
            ProjectRetriever._save_cache('concept_embeddings', cache)
        concept_emb = cache[text_hash]

        # ── Batch encode all queries ──
        qids = list(queries.keys())
        qtexts = list(queries.values())
        q_emb = np.array(self._vs._ef(qtexts))
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # ── One matmul: (N_concepts × D) @ (D × N_queries) → (N_concepts × N_queries) ──
        sim_matrix = np.dot(concept_emb, q_emb.T)  # (580, 64)

        # ── Build per-query chunk scores ──
        self._graph_batch_scores = {}
        for qi, qid in enumerate(qids):
            sims = sim_matrix[:, qi]  # 580 scores for this query
            scored: dict[str, float] = {}
            seen: dict[str, str] = {}
            for ci, c in enumerate(all_concepts):
                s = float(sims[ci])
                if s > 0.1:
                    cid = c.source_chunk_id or f"kg:{c.id}"
                    if s > scored.get(cid, 0):
                        scored[cid] = s
                        seen[cid] = f"{c.name}: {(c.description or '')[:80]}"
            # Keep top-k
            top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]
            self._graph_batch_scores[qid] = dict(top)

    def _search_graph(self, query: str, top_k: int) -> list[dict]:
        """KG search with semantic concept matching (embedding-based).

        If batch-precomputed scores are available, uses the per-query cache.
        """
        self._ensure_init()

        # Use batch-precomputed scores if available for this qid
        qid = getattr(self, '_current_qid', '')
        if qid and hasattr(self, '_graph_batch_scores') and qid in self._graph_batch_scores:
            scored = self._graph_batch_scores[qid]
            items = [{"chunk_id": cid, "score": s, "text": ""}
                     for cid, s in scored.items()]
            return items[:top_k]

        try:
            import numpy as np

            scored: dict[str, float] = {}  # chunk_id -> score
            seen: dict[str, dict] = {}     # chunk_id -> best concept info

            all_concepts = self._kg.get_all_concepts()

            # Build concept documents: name + description
            concept_texts = [f"{c.name or ''}: {c.description or ''}" for c in all_concepts]
            text_hash = hash(tuple(concept_texts))

            # Class-level cache: share embeddings across all retriever instances
            if not hasattr(ProjectRetriever, '_concept_cache'):
                ProjectRetriever._concept_cache = {}
            cache = ProjectRetriever._concept_cache

            if text_hash not in cache:
                logger.info("[graph] encoding %d concepts for semantic search", len(concept_texts))
                emb = np.array(self._vs._ef(concept_texts))
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1
                cache[text_hash] = emb / norms
            self._concept_embeddings = cache[text_hash]

            # Encode query
            q_vec = np.array(self._vs._ef([query]))
            q_vec = q_vec / np.linalg.norm(q_vec)

            # Cosine similarity → scores
            similarities = np.dot(self._concept_embeddings, q_vec.T).flatten()

            # Map concept scores to chunk scores
            concept_scores: list[tuple] = []  # (concept, score)
            for i, c in enumerate(all_concepts):
                sim = float(similarities[i])
                if sim > 0.1:  # minimum similarity threshold
                    cid = c.source_chunk_id or f"kg:{c.id}"
                    if sim > scored.get(cid, 0):
                        scored[cid] = sim
                        seen[cid] = {
                            "chunk_id": cid,
                            "text": f"{c.name}: {(c.description or '')[:100]}",
                        }
                    concept_scores.append((c, sim))

            # Sort concepts by similarity for GraphRAG traversal
            concept_scores.sort(key=lambda x: x[1], reverse=True)

            # ── 1:N concept-chunk mapping via related_chunk_ids ──
            # Concepts extracted with related_fragments point to multiple chunks.
            # Add those related chunks with the concept's embedding score × 0.5.
            for c in concept_scores[:15]:  # top-15 concepts only
                concept, score = c
                related = (concept.related_chunk_ids or "").split(",")
                for rcid in related:
                    rcid = rcid.strip()
                    if rcid and rcid not in scored:
                        scored[rcid] = score * 0.5
                        seen[rcid] = {
                            "chunk_id": rcid,
                            "text": f"related: {concept.name}",
                        }

            # ── Cross-chapter expansion ──
            # Use class-level name→concepts lookup (shared across instances).
            if not hasattr(ProjectRetriever, '_global_name_index'):
                ProjectRetriever._global_name_index = {}
                for c in all_concepts:
                    ProjectRetriever._global_name_index.setdefault(c.name, []).append(c)
            self._name_index = ProjectRetriever._global_name_index

            top_ids = {c.id for c, _s in concept_scores[:5]}
            for c, _s in concept_scores[:5]:
                for sibling in self._name_index.get(c.name, []):
                    if sibling.id in top_ids:
                        continue  # skip self-matches
                    cid = sibling.source_chunk_id or f"kg:{sibling.id}"
                    if cid and cid not in scored:
                        scored[cid] = 0.3
                        seen[cid] = {
                            "chunk_id": cid,
                            "text": f"cross-chapter: {sibling.name}",
                        }

            # ── Community-aware expansion (Microsoft GraphRAG Leiden clustering) ──
            # Concepts in the same Leiden community share a semantic topic.
            # "数据库" and "数据独立性" may not be directly connected by an edge,
            # but they belong to the same community → retrieve all their chunks.
            #
            # Tier 1 (direct match): embedding similarity  — done above
            # Tier 2 (community): same-community concepts → score × 0.6
            # Tier 3 (1-hop neighbors): score × 0.4
            # Tier 4 (2-hop): flat 0.12

            # Lazy-build Leiden communities (class-level cache, shared across instances)
            if not hasattr(ProjectRetriever, '_global_communities'):
                ProjectRetriever._global_communities = None
            if ProjectRetriever._global_communities is not None:
                self._concept_community, self._communities = ProjectRetriever._global_communities
            if not hasattr(self, '_communities') or not self._communities:
                self._concept_community = {}  # concept_id → community_id
                self._communities = {}        # community_id → [concept_ids]
                try:
                    import networkx as nx
                    G = nx.Graph()
                    for c in all_concepts:
                        G.add_node(c.id)
                    for c in all_concepts:
                        try:
                            for n in self._kg.get_neighbors(c.id):
                                G.add_edge(c.id, n["concept_id"])
                        except Exception:
                            pass
                    if G.number_of_edges() > 0:
                        import leidenalg as la
                        import igraph as ig
                        # Convert networkx → igraph for leiden
                        ig_g = ig.Graph.TupleList(G.edges(), directed=False)
                        partition = la.find_partition(ig_g, la.ModularityVertexPartition)
                        for community_id, node_ids in enumerate(partition):
                            for nid in node_ids:
                                concept_id = ig_g.vs[nid]["name"]
                                self._concept_community[concept_id] = community_id
                                if community_id not in self._communities:
                                    self._communities[community_id] = []
                                self._communities[community_id].append(concept_id)
                        ProjectRetriever._global_communities = (dict(self._concept_community), dict(self._communities))
                        ProjectRetriever._save_cache('leiden_communities', ProjectRetriever._global_communities)
                        logger.info("[graph] Leiden: %d communities from %d nodes, %d edges",
                                     len(self._communities), G.number_of_nodes(), G.number_of_edges())
                except Exception as e:
                    logger.warning("[graph] Leiden clustering failed: %s — skipping community expansion", e)
                    self._concept_community = {}
                    self._communities = {}

            top_matched = concept_scores[:8]
            visited_concepts: set[str] = {c.id for c, _ in top_matched}

            # Tier 2: same-community concepts
            if self._concept_community:
                seen_communities: set[int] = set()
                for c, base_score in top_matched:
                    comm_id = self._concept_community.get(c.id)
                    if comm_id is not None and comm_id not in seen_communities:
                        seen_communities.add(comm_id)
                        for sibling_id in self._communities.get(comm_id, [])[:10]:
                            if sibling_id in visited_concepts:
                                continue
                            visited_concepts.add(sibling_id)
                            sc = self._kg.get_concept(sibling_id)
                            if sc and sc.source_chunk_id:
                                cid = sc.source_chunk_id
                                if cid not in scored:
                                    scored[cid] = base_score * 0.6
                                    seen[cid] = {
                                        "chunk_id": cid,
                                        "text": f"{sc.name} (same community as {c.name})",
                                    }

            # Tier 3: 1-hop neighbors
            for c, base_score in top_matched:
                try:
                    for n in self._kg.get_neighbors(c.id)[:5]:
                        if n["concept_id"] in visited_concepts:
                            continue
                        visited_concepts.add(n["concept_id"])
                        nc = self._kg.get_concept(n["concept_id"])
                        if nc and nc.source_chunk_id:
                            cid = nc.source_chunk_id
                            if cid not in scored:
                                scored[cid] = base_score * 0.4
                                seen[cid] = {
                                    "chunk_id": cid,
                                    "text": f"{nc.name} ({n.get('relation_type','?')}→{c.name})",
                                }
                except Exception:
                    pass

            # Tier 4: 2-hop
            hop1_ids = list(visited_concepts - {c.id for c, _ in top_matched})
            for hid in hop1_ids[:5]:
                try:
                    for n in self._kg.get_neighbors(hid)[:3]:
                        if n["concept_id"] in visited_concepts:
                            continue
                        visited_concepts.add(n["concept_id"])
                        nc = self._kg.get_concept(n["concept_id"])
                        if nc and nc.source_chunk_id:
                            cid = nc.source_chunk_id
                            if cid not in scored:
                                scored[cid] = 0.12
                                seen[cid] = {
                                    "chunk_id": cid,
                                    "text": f"{nc.name} ({n.get('relation_type','?')}→2hop)",
                                }
                except Exception:
                    pass

            # Sort by score, return top_k
            items = []
            for cid, score in sorted(scored.items(), key=lambda x: x[1], reverse=True):
                item = dict(seen[cid])
                item["score"] = score
                items.append(item)

            return items[:top_k]
        except Exception as e:
            logger.warning("Graph search failed: %s", e)
            return []

    def _classify_query(self, query: str) -> dict:
        """Single query fallback. Delegates to batch."""
        return self._classify_queries_batch([query])[0]

    def _classify_queries_batch(self, queries: list[str]) -> list[dict]:
        """Batch-classify multiple queries in one LLM call."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.config import Configuration
        import json as _json, re as _re

        config = Configuration.from_env()
        llm = ChatOpenAI(model=config.llm_model_id, api_key=config.llm_api_key,
                         base_url=config.llm_base_url, temperature=0.0, max_tokens=1000)
        numbered = "\n".join(f"{i}: {q}" for i, q in enumerate(queries))
        prompt = f"分析以下{len(queries)}个问题，判断检索权重。concept(定义)→dense高，exact(公式/SQL)→sparse高，relational(关系/区别)→graph高。输出JSON数组：[{{\"dense\":0.5,\"sparse\":0.3,\"graph\":0.2}},...]\n\n{numbered}"
        try:
            resp = llm.invoke([SystemMessage(content="只输出JSON数组。"), HumanMessage(content=prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            m = _re.search(r'\[.*\]', text, _re.DOTALL)
            if m:
                arr = _json.loads(m.group(0))
                return [{"dense": float(w.get("dense",0.5)), "sparse": float(w.get("sparse",0.3)), "graph": float(w.get("graph",0.2))} for w in arr]
        except Exception: pass
        return [{"dense": 0.5, "sparse": 0.3, "graph": 0.2}] * len(queries)

    def _search_hybrid(self, query: str, top_k: int) -> list[dict]:
        """Dense + sparse + KG → adaptive weighted RRF fusion."""
        from src.tools.rag_search import _rrf_fuse, _rrf_fuse_weighted  # noqa: F811

        f = self._filter_set
        results_by_source: list[tuple[str, list[dict]]] = []

        hybrid_data = self._vs.search_hybrid(query, top_k, filter_docs=f) or {}
        if hybrid_data.get("dense"):
            results_by_source.append(("dense", hybrid_data["dense"]))
        if hybrid_data.get("sparse"):
            results_by_source.append(("sparse", hybrid_data["sparse"]))

        try:
            kg_concepts = self._kg.search_concepts(query, limit=top_k)
            graph_items = []
            for c in kg_concepts:
                src_cid = c.source_chunk_id or f"kg:{c.id}"
                graph_items.append({
                    "chunk_id": src_cid,
                    "text": f"{c.name or '?'}: {c.description or ''}",
                    "score": 1.0,
                })
            if graph_items:
                results_by_source.append(("graph", graph_items))
        except Exception as e:
            logger.debug("KG search failed: %s", e)

        if not results_by_source:
            return []

        # Adaptive RRF: use pre-classified batch weights
        if self.strategy == "adaptive":
            weights = {"dense": 0.5, "sparse": 0.3, "graph": 0.2}
            if hasattr(self, "_adaptive_weights") and self._adaptive_weights:
                qi = self._query_index if hasattr(self, "_query_index") else 0
                if qi < len(self._adaptive_weights):
                    weights = self._adaptive_weights[qi]
            else:
                weights = self._classify_query(query)

            # Guard: if dense is dominant, skip fusion to avoid RRF
            # dedup noise pushing good dense results out of top-k.
            if weights.get("dense", 0) >= 0.6:
                # Dense-only: return just dense results (no fusion noise)
                for source, items in results_by_source:
                    if source == "dense":
                        return items[:top_k]
                return _rrf_fuse(results_by_source, top_k)

            weighted = []
            for source, items in results_by_source:
                w = weights.get(source, 1.0)
                weighted.append((source, items, w))
            return _rrf_fuse_weighted(weighted, top_k)
        return _rrf_fuse(results_by_source, top_k)

    def _search_supplement(self, query: str, top_k: int) -> list[dict]:
        """Dense-first + sparse/graph supplements → reranker re-scores all.

        1. Dense top-20 (primary)
        2. Sparse top-10 unique chunks (supplement)
        3. Graph top-10 unique chunks (supplement)
        4. Deduplicate → Reranker (Cross-Encoder) re-scores → top-10
        """
        self._ensure_init()
        f = self._filter_set

        # ── Collect candidates ──
        candidates: list[dict] = []
        seen_chunks: set[str] = set()

        # Dense: top-20
        dense_results = self._vs._search_dense(query, 20, filter_docs=f) or []
        for r in dense_results:
            cid = r.get("chunk_id", "")
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                r["source"] = "dense"
                candidates.append(r)

        # Sparse: top-10 unique
        sparse_results = self._vs._search_sparse(query, 10, filter_docs=f) or []
        for r in sparse_results:
            cid = r.get("chunk_id", "")
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                r["source"] = "sparse"
                candidates.append(r)

        # Graph: top-10 unique
        try:
            graph_items = self._search_graph(query, 10)
            for item in graph_items:
                cid = item.get("chunk_id", "")
                if cid and cid not in seen_chunks:
                    seen_chunks.add(cid)
                    item["source"] = "graph"
                    candidates.append(item)
        except Exception as e:
            logger.debug("Graph supplement failed: %s", e)

        # ── Qwen3-Embedding cosine rerank (zero extra model) ──
        if len(candidates) > top_k:
            try:
                import numpy as np
                q_vec = np.array(self._vs._ef([query]))
                q_vec = q_vec / np.linalg.norm(q_vec)
                for c in candidates:
                    txt = c.get("text", "")[:500]
                    if txt.strip():
                        c_vec = np.array(self._vs._ef([txt]))
                        c_vec = c_vec / np.linalg.norm(c_vec)
                        c["rerank_score"] = float(np.dot(q_vec, c_vec.T)[0])
                    else:
                        c["rerank_score"] = 0.0
                candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
            except Exception:
                pass

        return candidates[:top_k]

    def _search_supplement_batch(self, queries: dict[str, str], top_k: int) -> dict[str, dict[str, float]]:
        """Batch version: collect all candidates → one reranker pass → return all results."""
        self._ensure_init()
        f = self._filter_set

        # ── Phase 1: Collect candidates for all queries ──
        all_candidates: dict[str, list[dict]] = {}  # qid → [candidates]
        for qid, query in queries.items():
            seen: set[str] = set()
            cands: list[dict] = []

            for r in (self._vs._search_dense(query, 20, filter_docs=f) or []):
                cid = r.get("chunk_id", "")
                if cid and cid not in seen:
                    seen.add(cid)
                    r["source"] = "dense"
                    cands.append(r)

            for r in (self._vs._search_sparse(query, 10, filter_docs=f) or []):
                cid = r.get("chunk_id", "")
                if cid and cid not in seen:
                    seen.add(cid)
                    r["source"] = "sparse"
                    cands.append(r)

            try:
                for item in (self._search_graph(query, 10) or []):
                    cid = item.get("chunk_id", "")
                    if cid and cid not in seen:
                        seen.add(cid)
                        item["source"] = "graph"
                        cands.append(item)
            except Exception:
                pass

            all_candidates[qid] = cands

        # ── Phase 2: Batch rerank all candidates ──
        try:
            # Load reranker if needed
            if not hasattr(ProjectRetriever, '_reranker'):
                from pathlib import Path as _Path
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                _rp = str(_Path(__file__).resolve().parent.parent.parent
                          / 'models' / 'reranker' / 'models'
                          / 'BAAI--bge-reranker-v2-m3' / 'snapshots' / 'master')
                ProjectRetriever._reranker = AutoModelForSequenceClassification.from_pretrained(
                    _rp, local_files_only=True)
                ProjectRetriever._reranker_tokenizer = AutoTokenizer.from_pretrained(
                    _rp, local_files_only=True)

            # Build all (query, text) pairs
            pairs: list[tuple[str, str]] = []
            pair_map: list[tuple[str, int]] = []  # (qid, cand_index)
            for qid, cands in all_candidates.items():
                query = queries[qid]
                for ci, c in enumerate(cands):
                    pairs.append((query, c.get("text", "")[:500]))
                    pair_map.append((qid, ci))

            if pairs:
                import torch
                with torch.no_grad():
                    inputs = ProjectRetriever._reranker_tokenizer(
                        pairs, padding=True, truncation=True,
                        max_length=512, return_tensors='pt')
                    logits = ProjectRetriever._reranker(**inputs, return_dict=True).logits
                    scores = logits.view(-1).cpu().numpy().tolist()
                if not isinstance(scores, list):
                    scores = [scores]

                for (qid, ci), s in zip(pair_map, scores):
                    all_candidates[qid][ci]["rerank_score"] = float(s)

                # Sort by rerank score
                for qid in all_candidates:
                    all_candidates[qid].sort(
                        key=lambda c: c.get("rerank_score", 0), reverse=True)
        except Exception as e:
            logger.warning("Batch reranker failed: %s", e)

        # ── Phase 3: Format results ──
        results: dict[str, dict[str, float]] = {}
        for qid, cands in all_candidates.items():
            scored: dict[str, float] = {}
            for rank, item in enumerate(cands[:top_k]):
                cid = item.get("chunk_id", "")
                if not cid:
                    continue
                scored[cid] = item.get("rerank_score", item.get("score", 1.0 / (60 + rank + 1)))
            results[qid] = scored
        return results


class DummyRetriever:
    """Minimal retriever for testing the eval pipeline without a real project context."""

    def __init__(self):
        self._corpus: dict[str, str] = {}

    def search(
        self,
        corpus: dict[str, dict],
        queries: dict[str, str],
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        self._corpus = {did: doc.get("text", doc.get("title", "")) for did, doc in corpus.items()}
        results: dict[str, dict[str, float]] = {}
        for qid, query_text in queries.items():
            query_tokens = set(query_text.lower().split())
            if not query_tokens:
                results[qid] = {}
                continue
            scored = {}
            for doc_id, doc_text in self._corpus.items():
                doc_tokens = set(doc_text.lower().split())
                overlap = len(query_tokens & doc_tokens)
                if overlap > 0:
                    union = len(query_tokens | doc_tokens)
                    scored[doc_id] = overlap / union if union > 0 else 0.0
            sorted_items = sorted(scored.items(), key=lambda x: x[1], reverse=True)
            results[qid] = dict(sorted_items[:top_k])
        return results


# ═══════════════════════════════════════════════════════════════════════
# Standalone BEIR Retriever — indexes corpus on-the-fly for true BEIR eval
# ═══════════════════════════════════════════════════════════════════════

class StandaloneRetriever:
    """BEIR-compatible retriever that indexes the passed corpus in-memory.

    Unlike ProjectRetriever, this does NOT use the project's pre-indexed
    vector store. It builds a temporary index from the BEIR corpus at
    search time, ensuring doc_ids match the qrels exactly.

    Supports: dense, sparse, hybrid (no graph — BEIR has no KG).
    """

    def __init__(self, strategy: str = "hybrid"):
        self.strategy = strategy
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._embeddings = None
        self._embed_fn = None
        self._bm25 = None

    def _ensure_indexed(self, corpus: dict[str, dict]):
        """Build in-memory index from corpus (lazy, once per search call)."""
        # Check if corpus changed
        current_ids = list(corpus.keys())
        if current_ids == self._corpus_ids:
            return  # already indexed this corpus

        self._corpus_ids = current_ids
        self._corpus_texts = []
        for doc_id in current_ids:
            doc = corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            self._corpus_texts.append(f"{title}\n{text}" if title else text)

        # Reset cached indices
        self._embeddings = None
        self._bm25 = None

        logger.info("StandaloneRetriever: indexed %d docs in memory", len(self._corpus_ids))

    def _ensure_embeddings(self, corpus: dict[str, dict]):
        """Lazy-load embedding model and encode corpus."""
        if self._embeddings is not None:
            return
        if self._embed_fn is None:
            from src.agents.qa.vector_store import CrossLingualEmbeddingFunction
            self._embed_fn = CrossLingualEmbeddingFunction()
        self._ensure_indexed(corpus)
        import numpy as np
        logger.info("Encoding %d corpus docs (this may take a minute)...", len(self._corpus_texts))
        self._embeddings = np.array(self._embed_fn(self._corpus_texts))
        logger.info("Corpus encoded: shape=%s", self._embeddings.shape)

    def _ensure_bm25(self, corpus: dict[str, dict]):
        """Lazy-build in-memory BM25 index."""
        if self._bm25 is not None:
            return
        self._ensure_indexed(corpus)
        import re
        from collections import defaultdict
        import math

        # Tokenize and build inverted index
        idx: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        doc_lens: list[int] = []
        for i, text in enumerate(self._corpus_texts):
            tokens = re.findall(r'[a-zA-Z0-9]{2,}', text.lower())
            doc_lens.append(len(tokens))
            for t in set(tokens):
                idx[t][i] = tokens.count(t)

        self._bm25_doc_lens = doc_lens
        self._bm25_idx = idx
        self._bm25_N = len(self._corpus_texts)
        self._bm25_avgdl = sum(doc_lens) / max(1, self._bm25_N)
        logger.info("BM25 index built: %d docs, avg len=%.1f tokens", self._bm25_N, self._bm25_avgdl)

    def _search_dense(self, corpus: dict[str, dict], query: str, top_k: int) -> list[dict]:
        import numpy as np
        self._ensure_embeddings(corpus)
        q_vec = np.array(self._embed_fn([query]))
        scores = np.dot(self._embeddings, q_vec.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk_id": self._corpus_ids[i], "score": float(scores[i]),
             "text": self._corpus_texts[i][:200]}
            for i in top_idx if scores[i] > 0
        ]

    def _search_sparse(self, corpus: dict[str, dict], query: str, top_k: int) -> list[dict]:
        import re, math
        self._ensure_bm25(corpus)

        q_tokens = re.findall(r'[a-zA-Z0-9]{2,}', query.lower())
        if not q_tokens:
            return []

        k1, b = 1.5, 0.75
        scores: dict[int, float] = {}
        for qt in q_tokens:
            if qt not in self._bm25_idx:
                continue
            idf = math.log((self._bm25_N - len(self._bm25_idx[qt]) + 0.5)
                          / (len(self._bm25_idx[qt]) + 0.5) + 1)
            for doc_i, tf in self._bm25_idx[qt].items():
                dl = self._bm25_doc_lens[doc_i]
                bm25 = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._bm25_avgdl))
                scores[doc_i] = scores.get(doc_i, 0.0) + bm25

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"chunk_id": self._corpus_ids[i], "score": float(s),
             "text": self._corpus_texts[i][:200]}
            for i, s in sorted_items if s > 0
        ]

    def _search_hybrid(self, corpus: dict[str, dict], query: str, top_k: int) -> list[dict]:
        """Dense + Sparse → RRF fusion."""
        from src.tools.rag_search import _rrf_fuse

        dense_results = self._search_dense(corpus, query, top_k * 2)
        sparse_results = self._search_sparse(corpus, query, top_k * 2)

        results_by_source: list[tuple[str, list[dict]]] = []
        if dense_results:
            results_by_source.append(("dense", dense_results))
        if sparse_results:
            results_by_source.append(("sparse", sparse_results))

        if not results_by_source:
            return []

        return _rrf_fuse(results_by_source, top_k)

    def search(
        self,
        corpus: dict[str, dict],
        queries: dict[str, str],
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Standard BEIR retriever interface.

        Indexes the corpus in-memory on first call, then searches.
        Returns {query_id: {doc_id: score}} where doc_ids match the corpus keys.
        """
        # Determine search method
        if self.strategy == "dense":
            search_fn = self._search_dense
        elif self.strategy == "sparse":
            search_fn = self._search_sparse
        elif self.strategy == "hybrid":
            search_fn = self._search_hybrid
        else:
            search_fn = self._search_hybrid

        results: dict[str, dict[str, float]] = {}
        for qid, query_text in queries.items():
            try:
                retrieved = search_fn(corpus, query_text, top_k * 2)
                scored: dict[str, float] = {}
                for rank, item in enumerate(retrieved[:top_k]):
                    doc_id = item.get("chunk_id", "")
                    if not doc_id:
                        continue
                    score = item.get("rrf_score", item.get("score", 1.0 / (60 + rank + 1)))
                    scored[doc_id] = float(score)
                results[qid] = scored
            except Exception as e:
                logger.warning("Search failed for query '%s': %s", qid, e)
                results[qid] = {}

        return results
