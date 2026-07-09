Feature: Knowledge Extraction
  As a user
  I want the system to extract knowledge from my documents
  So that I can explore concepts and their relationships

  Background:
    Given a test document named "sample.txt"
    And an empty knowledge graph

  Scenario: Extract concepts from a document
    Given the LLM is mocked for extraction
    When I parse the document
    And I chunk the document
    And I extract knowledge from the chunks
    Then the knowledge graph should have at least 5 concepts
    And the knowledge graph should have at least 3 relations
    And I stop the LLM mock

  Scenario: Search knowledge graph by concept name
    Given the LLM is mocked for extraction
    When I parse the document
    And I chunk the document
    And I extract knowledge from the chunks
    Then the knowledge graph search should find "Python"
    And the knowledge graph search should find "机器学习"
    And I stop the LLM mock

  Scenario: Handle empty extraction gracefully
    Given the LLM returns empty extraction
    When I parse the document
    And I chunk the document
    And I extract knowledge from the chunks
    Then the knowledge graph should have at least 0 concepts
    And I stop the LLM mock

  Scenario: Add and retrieve concepts manually
    When I add a concept named "贝叶斯定理" with description "描述事件条件概率的定理"
    Then the concept should be retrievable by name
    And I clean up the knowledge graph
