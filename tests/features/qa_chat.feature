Feature: Question Answering
  As a user
  I want to ask questions about my documents
  So that I can learn from the materials I uploaded

  Background:
    Given a test document named "sample.txt"
    And a document vector store
    Given the LLM is mocked for QA

  Scenario: Retrieve relevant chunks for a question
    When I parse the document
    And I chunk the document
    And I index the chunks into the vector store
    And I search the vector store for "什么是Python"
    Then the search should return at least 1 result(s)

  Scenario: Search for content not in the document
    When I parse the document
    And I chunk the document
    And I index the chunks into the vector store
    And I search the vector store for "量子计算"
    Then the search should return at least 0 result(s)

  Scenario: End-to-end QA pipeline mock
    Given a test document named "sample.txt"
    When I parse the document
    And I chunk the document
    And I index the chunks into the vector store
    And I search the vector store for "Python的特点"
    Then the search should return at least 1 result(s)
    And I stop the LLM mock
    And I clean up the vector store

  Scenario: Multi-word Chinese query
    Given a test document named "sample.txt"
    When I parse the document
    And I chunk the document
    And I index the chunks into the vector store
    And I search the vector store for "机器学习的方法有哪些"
    Then the search should return at least 1 result(s)
    And I stop the LLM mock
    And I clean up the vector store
