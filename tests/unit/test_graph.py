# Unit tests for KnowledgeGraph

from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pytest

from src.knowledge.graph import KnowledgeGraph, ConceptNode, RelationEdge


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph CRUD operations."""

    @pytest.fixture
    def kg(self):
        """Create a KnowledgeGraph with a temporary database."""
        # ignore_cleanup_errors=True needed for Windows (SQLite file locks)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "test_knowledge.db")
            kg = KnowledgeGraph(db_path=db_path)
            yield kg

    def test_init_creates_db(self):
        """KnowledgeGraph should create the database file on init."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "test_init.db")
            kg = KnowledgeGraph(db_path=db_path)
            assert os.path.exists(db_path)

    def test_add_concept(self, kg):
        """Adding a concept should store it and make it retrievable."""
        concept = ConceptNode(
            id="c1", name="Gradient Descent",
            description="An optimization algorithm",
            category="method", doc_filename="ml_book.pdf",
        )
        kg.add_concept(concept)

        retrieved = kg.get_concept("c1")
        assert retrieved is not None
        assert retrieved.name == "Gradient Descent"
        assert retrieved.category == "method"
        assert retrieved.doc_filename == "ml_book.pdf"

    def test_get_nonexistent_concept(self, kg):
        """Getting a non-existent concept should return None."""
        assert kg.get_concept("nonexistent") is None

    def test_add_concepts_batch(self, kg):
        """Batch insertion should store all concepts."""
        concepts = [
            ConceptNode(id="c1", name="Concept 1", description="First"),
            ConceptNode(id="c2", name="Concept 2", description="Second"),
            ConceptNode(id="c3", name="Concept 3", description="Third"),
        ]
        kg.add_concepts_batch(concepts)

        assert kg.get_concept("c1") is not None
        assert kg.get_concept("c2") is not None
        assert kg.get_concept("c3") is not None

    def test_search_concepts(self, kg):
        """Search by name should return matching concepts."""
        kg.add_concept(ConceptNode(id="c1", name="Linear Regression", description="A basic ML model"))
        kg.add_concept(ConceptNode(id="c2", name="Logistic Regression", description="Classification model"))
        kg.add_concept(ConceptNode(id="c3", name="Decision Tree", description="A split-based model"))

        results = kg.search_concepts("Regression")
        assert len(results) == 2
        names = {c.name for c in results}
        assert "Linear Regression" in names
        assert "Logistic Regression" in names

    def test_add_relation(self, kg):
        """Adding a relation should connect two concepts."""
        kg.add_concept(ConceptNode(id="c1", name="A", description="Concept A"))
        kg.add_concept(ConceptNode(id="c2", name="B", description="Concept B"))
        kg.add_relation(RelationEdge(id="r1", source_id="c1", target_id="c2",
                                      relation_type="prerequisite_of", description="A before B"))

        neighbors = kg.get_neighbors("c1")
        assert len(neighbors) == 1
        assert neighbors[0]["concept_name"] == "B"
        assert neighbors[0]["relation_type"] == "prerequisite_of"

    def test_get_neighbors_bidirectional(self, kg):
        """get_neighbors should return both outgoing and incoming relations."""
        kg.add_concept(ConceptNode(id="c1", name="A", description="Concept A"))
        kg.add_concept(ConceptNode(id="c2", name="B", description="Concept B"))
        kg.add_relation(RelationEdge(id="r1", source_id="c1", target_id="c2",
                                      relation_type="leads_to"))

        # From source side
        neighbors_c1 = kg.get_neighbors("c1")
        assert len(neighbors_c1) == 1
        assert neighbors_c1[0]["concept_name"] == "B"

        # From target side (bidirectional lookup)
        neighbors_c2 = kg.get_neighbors("c2")
        assert len(neighbors_c2) == 1
        assert neighbors_c2[0]["concept_name"] == "A"

    def test_stats(self, kg):
        """stats() should return accurate counts."""
        kg.add_concept(ConceptNode(id="c1", name="A", description="Desc A", category="definition"))
        kg.add_concept(ConceptNode(id="c2", name="B", description="Desc B", category="theorem"))
        kg.add_concept(ConceptNode(id="c3", name="C", description="Desc C", category="definition"))
        kg.add_relation(RelationEdge(id="r1", source_id="c1", target_id="c2", relation_type="related_to"))

        stats = kg.stats()
        assert stats["concepts"] == 3
        assert stats["relations"] == 1
        assert stats["categories"] == {"definition": 2, "theorem": 1}

    def test_clear(self, kg):
        """clear() should remove all concepts and relations."""
        kg.add_concept(ConceptNode(id="c1", name="A", description="Desc"))
        kg.clear()

        stats = kg.stats()
        assert stats["concepts"] == 0
        assert stats["relations"] == 0
        assert kg.get_concept("c1") is None

    def test_get_doc_names(self, kg):
        """get_doc_names should return unique document filenames."""
        kg.add_concept(ConceptNode(id="c1", name="A", description="Desc", doc_filename="book1.pdf"))
        kg.add_concept(ConceptNode(id="c2", name="B", description="Desc", doc_filename="book1.pdf"))
        kg.add_concept(ConceptNode(id="c3", name="C", description="Desc", doc_filename="book2.pdf"))

        names = kg.get_doc_names()
        assert len(names) == 2
        assert "book1.pdf" in names
        assert "book2.pdf" in names

    def test_search_concepts_by_docs(self, kg):
        """search_concepts_by_docs should filter by document."""
        kg.add_concept(ConceptNode(id="c1", name="Gradient", description="Desc",
                                    doc_filename="ml.pdf"))
        kg.add_concept(ConceptNode(id="c2", name="Gradient Boosting",
                                    description="Desc", doc_filename="stats.pdf"))

        results = kg.search_concepts_by_docs("Gradient", {"ml.pdf"})
        assert len(results) == 1
        assert results[0].doc_filename == "ml.pdf"
