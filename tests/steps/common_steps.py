# Common BDD step definitions for Document QA System
# All LLM calls are mocked -- zero API cost

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from behave import given, when, then
from behave.runner import Context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.documents.parser import parse_document, Document
from src.documents.chunker import chunk_document
from src.knowledge.graph import KnowledgeGraph, ConceptNode, RelationEdge
from src.agents.qa import DocumentVectorStore


# ============================================================================
# Mock LLM -- canned responses, zero API cost
# ============================================================================

EXTRACTION_RESPONSE = json.dumps({
    "concepts": [
        {"name": "Python", "description": "解释型面向对象编程语言", "category": "concept"},
        {"name": "变量", "description": "不需要声明类型的数据容器", "category": "concept"},
        {"name": "缩进", "description": "Python语法核心，表示代码块层级", "category": "concept"},
        {"name": "机器学习", "description": "AI分支，通过数据训练模型", "category": "concept"},
        {"name": "监督学习", "description": "使用标注数据训练的方法", "category": "method"},
        {"name": "深度学习", "description": "多层神经网络特征提取", "category": "method"},
        {"name": "神经网络", "description": "深度学习核心组件", "category": "concept"},
    ],
    "relations": [
        {"source": "Python", "target": "变量", "relation_type": "part_of", "description": "Python包含变量"},
        {"source": "机器学习", "target": "监督学习", "relation_type": "part_of", "description": "监督学习是ML方法"},
        {"source": "机器学习", "target": "深度学习", "relation_type": "part_of", "description": "深度学习是ML子领域"},
        {"source": "深度学习", "target": "神经网络", "relation_type": "part_of", "description": "使用神经网络"},
    ]
}, ensure_ascii=False)

QA_RESPONSE = "Python是一种解释型面向对象编程语言，语法简洁。"
EXTRACTION_EMPTY_RESPONSE = json.dumps({"concepts": [], "relations": []}, ensure_ascii=False)


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


def mock_llm_factory(response_content: str):
    """Factory that returns a mock ChatOpenAI with canned async response."""
    def _make(*args, **kwargs):
        async def _fake_ainvoke(*_args, **_kwargs):
            return MockLLMResponse(response_content)

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=_fake_ainvoke)
        # Keep sync invoke for backward compat
        mock.invoke.return_value = MockLLMResponse(response_content)
        return mock
    return _make


# ============================================================================
# Given
# ============================================================================

@given("a test document named \"{filename}\"")
def step_given_document(context: Context, filename: str):
    context.doc_path = str(PROJECT_ROOT / "tests" / "fixtures" / filename)
    context.parsed_doc = parse_document(context.doc_path)


@given("an empty knowledge graph")
def step_given_empty_kg(context: Context):
    context.kg = KnowledgeGraph()
    context.kg.clear()


@given("a document vector store")
def step_given_vector_store(context: Context):
    context.vs = DocumentVectorStore()
    context.vs.clear()


@given("the LLM is mocked for extraction")
def step_given_mock_extraction(context: Context):
    # Patch at the import site in extractor.py
    context.mock_patch = patch(
        "src.agents.extractor.ChatOpenAI",
        side_effect=mock_llm_factory(EXTRACTION_RESPONSE),
    )
    context.mock_patch.start()


@given("the LLM is mocked for QA")
def step_given_mock_qa(context: Context):
    context.mock_patch = patch(
        "langchain_openai.ChatOpenAI",
        side_effect=mock_llm_factory(QA_RESPONSE),
    )
    context.mock_patch.start()


@given("the LLM returns empty extraction")
def step_given_mock_empty(context: Context):
    context.mock_patch = patch(
        "src.agents.extractor.ChatOpenAI",
        side_effect=mock_llm_factory(EXTRACTION_EMPTY_RESPONSE),
    )
    context.mock_patch.start()


# ============================================================================
# When
# ============================================================================

@when("I parse the document")
def step_when_parse(context: Context):
    context.parsed_doc = parse_document(context.doc_path)


@when("I chunk the document")
def step_when_chunk(context: Context):
    context.chunks = chunk_document(context.parsed_doc, chunk_size=200, chunk_overlap=30)


@when("I index the chunks into the vector store")
def step_when_index(context: Context):
    chunk_dicts = [
        {"chunk_id": c.chunk_id, "text": c.text,
         "doc_filename": c.doc_filename, "chunk_index": c.chunk_index}
        for c in context.chunks
    ]
    context.vs.index_chunks(chunk_dicts)


@when("I extract knowledge from the chunks")
def step_when_extract(context: Context):
    from src.config import Configuration
    from src.agents.extractor import extract_full_document
    config = Configuration(llm_api_key="mock-key")
    context.extraction_result = extract_full_document(context.chunks, config, context.kg)


@when("I search the vector store for \"{query}\"")
def step_when_search(context: Context, query: str):
    context.search_results = context.vs.search(query)


@when("I search the knowledge graph for \"{query}\"")
def step_when_kg_search(context: Context, query: str):
    context.kg_results = context.kg.search_concepts(query)


@when("I add a concept named \"{name}\" with description \"{desc}\"")
def step_when_add_concept(context: Context, name: str, desc: str):
    context.concept = ConceptNode(id="test_1", name=name, description=desc, category="concept")
    context.kg.add_concept(context.concept)


# ============================================================================
# Then
# ============================================================================

@then("the parsed document should have content")
def step_then_document_has_content(context: Context):
    assert context.parsed_doc is not None
    assert len(context.parsed_doc.content) > 0


@then("the document should have at least {min_chunks:d} chunks")
def step_then_min_chunks(context: Context, min_chunks: int):
    assert len(context.chunks) >= min_chunks, f"Expected >= {min_chunks} chunks, got {len(context.chunks)}"


@then("each chunk should have non-empty text")
def step_then_chunks_non_empty(context: Context):
    for c in context.chunks:
        assert len(c.text.strip()) > 0


@then("the knowledge graph should have at least {min_concepts:d} concepts")
def step_then_min_concepts(context: Context, min_concepts: int):
    stats = context.kg.stats()
    assert stats["concepts"] >= min_concepts, f"Expected >= {min_concepts} concepts, got {stats['concepts']}"


@then("the knowledge graph should have at least {min_relations:d} relations")
def step_then_min_relations(context: Context, min_relations: int):
    stats = context.kg.stats()
    assert stats["relations"] >= min_relations


@then("the search should return at least {min_results:d} result(s)")
def step_then_min_search_results(context: Context, min_results: int):
    assert len(context.search_results) >= min_results


@then("the knowledge graph search should find \"{name}\"")
def step_then_kg_find(context: Context, name: str):
    concepts = context.kg.search_concepts(name)
    assert any(c.name == name for c in concepts), f"Concept '{name}' not found"


@then("the concept should be retrievable by name")
def step_then_concept_retrievable(context: Context):
    concepts = context.kg.search_concepts(context.concept.name)
    assert len(concepts) > 0


@then("I clean up the vector store")
def step_then_cleanup_vs(context: Context):
    context.vs.clear()


@then("I clean up the knowledge graph")
def step_then_cleanup_kg(context: Context):
    context.kg.clear()


@then("I stop the LLM mock")
def step_then_stop_mock(context: Context):
    if hasattr(context, "mock_patch"):
        context.mock_patch.stop()
