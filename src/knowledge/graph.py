# Knowledge Graph — concept nodes, relation edges, SQLite-backed

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """A knowledge concept extracted from documents."""
    id: str
    name: str
    description: str
    category: str = ""          # e.g. "definition", "theorem", "method", "example"
    source_chunk_id: str = ""   # Which document chunk this came from
    doc_filename: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "source_chunk_id": self.source_chunk_id,
            "doc_filename": self.doc_filename,
            "metadata": self.metadata,
        }


@dataclass
class RelationEdge:
    """A directed relationship between two concepts."""
    id: str
    source_id: str
    target_id: str
    relation_type: str         # e.g. "prerequisite_of", "part_of", "example_of", "related_to"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "description": self.description,
        }


class KnowledgeGraph:
    """SQLite-backed knowledge graph for a document set."""

    def __init__(self, db_path: str = ""):
        if db_path:
            p = Path(db_path)
            if p.is_dir() or p.suffix == "":
                p = p / "knowledge.db"
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(p)
        else:
            default_dir = Path(__file__).resolve().parent.parent.parent / "data"
            default_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(default_dir / "knowledge.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    source_chunk_id TEXT DEFAULT '',
                    doc_filename TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES concepts(id),
                    FOREIGN KEY (target_id) REFERENCES concepts(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_src ON relations(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relation_tgt ON relations(target_id)")
            conn.commit()

    # ---- Concepts ----
    def add_concept(self, concept: ConceptNode):
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO concepts(id, name, description, category, source_chunk_id, doc_filename, metadata_json, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (concept.id, concept.name, concept.description, concept.category,
                 concept.source_chunk_id, concept.doc_filename,
                 json.dumps(concept.metadata, ensure_ascii=False), now),
            )
            conn.commit()

    def add_concepts_batch(self, concepts: list[ConceptNode]):
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO concepts(id, name, description, category, source_chunk_id, doc_filename, metadata_json, created_at) VALUES(?,?,?,?,?,?,?,?)",
                [(c.id, c.name, c.description, c.category, c.source_chunk_id, c.doc_filename, json.dumps(c.metadata, ensure_ascii=False), now) for c in concepts],
            )
            conn.commit()

    def get_concept(self, concept_id: str) -> ConceptNode | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT * FROM concepts WHERE id=?", (concept_id,)).fetchone()
        if row:
            return ConceptNode(id=row[0], name=row[1], description=row[2], category=row[3], source_chunk_id=row[4], doc_filename=row[5], metadata=json.loads(row[6]))
        return None

    def get_all_concepts(self) -> list[ConceptNode]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM concepts ORDER BY name").fetchall()
        return [ConceptNode(id=r[0], name=r[1], description=r[2], category=r[3], source_chunk_id=r[4], doc_filename=r[5], metadata=json.loads(r[6])) for r in rows]

    def search_concepts(self, query: str, limit: int = 10) -> list[ConceptNode]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM concepts WHERE name LIKE ? OR description LIKE ? ORDER BY name LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [ConceptNode(id=r[0], name=r[1], description=r[2], category=r[3], source_chunk_id=r[4], doc_filename=r[5], metadata=json.loads(r[6])) for r in rows]

    # ---- Relations ----
    def add_relation(self, rel: RelationEdge):
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO relations(id, source_id, target_id, relation_type, description, created_at) VALUES(?,?,?,?,?,?)",
                (rel.id, rel.source_id, rel.target_id, rel.relation_type, rel.description, now),
            )
            conn.commit()

    def add_relations_batch(self, relations: list[RelationEdge]):
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO relations(id, source_id, target_id, relation_type, description, created_at) VALUES(?,?,?,?,?,?)",
                [(r.id, r.source_id, r.target_id, r.relation_type, r.description, now) for r in relations],
            )
            conn.commit()

    def get_neighbors(self, concept_id: str) -> list[dict]:
        """Get all concepts connected to this one, with relation info."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("""
                SELECT r.relation_type, r.description, c.id, c.name, c.description, c.category
                FROM relations r JOIN concepts c ON r.target_id = c.id
                WHERE r.source_id = ?
                UNION
                SELECT r.relation_type, r.description, c.id, c.name, c.description, c.category
                FROM relations r JOIN concepts c ON r.source_id = c.id
                WHERE r.target_id = ?
            """, (concept_id, concept_id)).fetchall()
        return [
            {
                "relation_type": r[0], "relation_desc": r[1],
                "concept_id": r[2], "concept_name": r[3],
                "description": r[4], "category": r[5],
            }
            for r in rows
        ]

    def get_all_relations(self) -> list[RelationEdge]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT * FROM relations").fetchall()
        return [RelationEdge(id=r[0], source_id=r[1], target_id=r[2], relation_type=r[3], description=r[4]) for r in rows]

    def remove_by_doc(self, doc_filename: str) -> int:
        """Remove all concepts and their relations for a given document.

        Returns the number of concepts removed.
        """
        if not doc_filename:
            return 0
        with sqlite3.connect(self._db_path) as conn:
            # Find concept IDs to delete
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM concepts WHERE doc_filename = ?", (doc_filename,)
            ).fetchall()]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", ids + ids)
            conn.execute(f"DELETE FROM concepts WHERE id IN ({placeholders})", ids)
            conn.commit()
            logger.info("KG: removed %d concepts for '%s'", len(ids), doc_filename)
            return len(ids)

    def clear(self):
        """Clear all concepts and relations."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM concepts")
            conn.commit()

    def search_concepts_by_docs(self, query: str, doc_filenames: set[str], limit: int = 10) -> list:
        """Search concepts filtering by document filenames."""
        if not doc_filenames:
            return self.search_concepts(query, limit)
        
        with sqlite3.connect(self._db_path) as conn:
            placeholders = ",".join("?" * len(doc_filenames))
            sql = f"SELECT * FROM concepts WHERE (name LIKE ? OR description LIKE ?) AND doc_filename IN ({placeholders}) ORDER BY name LIMIT ?"
            params = [f"%{query}%", f"%{query}%"] + list(doc_filenames) + [limit]
            rows = conn.execute(sql, params).fetchall()
        return [ConceptNode(id=r[0], name=r[1], description=r[2], category=r[3],
                           source_chunk_id=r[4], doc_filename=r[5],
                           metadata=json.loads(r[6])) for r in rows]

    def get_doc_names(self) -> list[str]:
        """Get unique document filenames in the graph."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT DISTINCT doc_filename FROM concepts WHERE doc_filename != ''").fetchall()
        return sorted([r[0] for r in rows])

    def stats_str(self) -> str:
        """Get stats as a formatted string."""
        s = self.stats()
        return f"??: {s['concepts']} | ??: {s['relations']}"

    def stats(self) -> dict:
        with sqlite3.connect(self._db_path) as conn:
            n_concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
            n_relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            cats = conn.execute("SELECT category, COUNT(*) FROM concepts GROUP BY category").fetchall()
        return {
            "concepts": n_concepts,
            "relations": n_relations,
            "categories": {c[0]: c[1] for c in cats},
        }
