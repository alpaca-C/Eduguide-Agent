Feature: Document Processing
  As a user
  I want to upload and process documents
  So that the system can prepare them for Q&A

  Scenario: Parse a plain text document
    Given a test document named "sample.txt"
    When I parse the document
    Then the parsed document should have content

  Scenario: Chunk a document into manageable pieces
    Given a test document named "sample.txt"
    When I parse the document
    And I chunk the document
    Then the document should have at least 3 chunks
    And each chunk should have non-empty text

  Scenario: Index chunks into the vector store
    Given a test document named "sample.txt"
    And a document vector store
    When I parse the document
    And I chunk the document
    And I index the chunks into the vector store
    And I search the vector store for "Python"
    Then the search should return at least 1 result(s)
    And I clean up the vector store
