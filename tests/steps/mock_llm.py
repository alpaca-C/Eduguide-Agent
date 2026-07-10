# Mock LLM responses for BDD testing (zero API cost)

import json
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Canned responses matching the extractor's JSON format
# ---------------------------------------------------------------------------

EXTRACTION_RESPONSE = json.dumps({
    "concepts": [
        {"name": "Python", "description": "解释型面向对象高级编程语言", "category": "concept"},
        {"name": "变量", "description": "Python中不需要声明类型的数据容器", "category": "concept"},
        {"name": "缩进", "description": "Python语法的核心，表示代码块的层级关系", "category": "concept"},
        {"name": "机器学习", "description": "通过数据和算法让计算机从经验中学习的AI分支", "category": "concept"},
        {"name": "监督学习", "description": "使用标注数据进行训练的机器学习方法", "category": "method"},
        {"name": "深度学习", "description": "使用多层神经网络进行特征提取的机器学习子领域", "category": "method"},
        {"name": "神经网络", "description": "深度学习的核心组件", "category": "concept"},
    ],
    "relations": [
        {"source": "Python", "target": "变量", "relation_type": "part_of", "description": "Python包含变量的概念"},
        {"source": "Python", "target": "缩进", "relation_type": "part_of", "description": "Python使用缩进定义代码块"},
        {"source": "机器学习", "target": "监督学习", "relation_type": "part_of", "description": "监督学习是机器学习的一种方法"},
        {"source": "机器学习", "target": "深度学习", "relation_type": "part_of", "description": "深度学习是机器学习的子领域"},
        {"source": "深度学习", "target": "神经网络", "relation_type": "part_of", "description": "深度学习使用神经网络"},
    ]
}, ensure_ascii=False)

QA_RESPONSE = """Python是一种解释型、面向对象的高级编程语言。根据资料，它的主要特点包括：

1. **简洁的语法** - Python以缩进定义代码块，语法清晰
2. **强大的标准库** - 内置丰富的功能模块
3. **广泛应用** - 在Web开发、数据科学和AI领域都有应用

资料还提到，Python中的变量不需要声明类型，这也是其灵活性的体现。"""

EXTRACTION_EMPTY_RESPONSE = json.dumps({
    "concepts": [],
    "relations": [],
}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------

class MockLLMResponse:
    """Simulates an LLM response object."""
    def __init__(self, content: str):
        self.content = content


def create_mock_llm(response_content: str = EXTRACTION_RESPONSE):
    """Create a mock ChatOpenAI with canned async response."""
    async def _fake_ainvoke(*_args, **_kwargs):
        return MockLLMResponse(response_content)

    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=_fake_ainvoke)
    mock.invoke.return_value = MockLLMResponse(response_content)
    return mock


def patch_llm_with_response(response_content: str = EXTRACTION_RESPONSE):
    """Context manager that patches ChatOpenAI to return a canned response."""
    def mock_init(*args, **kwargs):
        return create_mock_llm(response_content)
    return patch("langchain_openai.ChatOpenAI", side_effect=mock_init)
