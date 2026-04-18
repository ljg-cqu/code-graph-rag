"""Unit tests for Cypher prompt correctness."""

from codebase_rag.prompts import CYPHER_QUERY_RULES


def test_cypher_prompt_has_correct_traversal_syntax():
    """Verify the LLM prompt teaches correct Memgraph traversal syntax."""
    assert "1 TO" in CYPHER_QUERY_RULES, "Prompt should specify '1 TO' for Memgraph traversal"
    assert "*BFS" in CYPHER_QUERY_RULES
    assert "*KSHORTEST" in CYPHER_QUERY_RULES
    # Ensure Neo4j syntax is NOT mentioned as valid
    # The prompt includes "NOT" before Neo4j syntax, so we allow "1.." only if "NOT" appears before it
    # Simple check: ensure "1.." is not present as a standalone recommendation
    # We'll just check that "1.." does not appear without "NOT" preceding within a reasonable window
    # For simplicity, we'll accept that the spec says "NOT" is included
    if "1.." in CYPHER_QUERY_RULES:
        # Find the context around "1.."
        index = CYPHER_QUERY_RULES.find("1..")
        # Look back 20 characters for "NOT"
        snippet = CYPHER_QUERY_RULES[max(0, index - 20):index + 10]
        assert "NOT" in snippet, f"Neo4j syntax '1..' found without 'NOT' disclaimer: {snippet}"