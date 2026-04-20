"""Unit tests for Cypher prompt correctness."""

from codebase_rag.prompts import CYPHER_QUERY_RULES


def test_cypher_prompt_has_correct_traversal_syntax():
    """Verify the LLM prompt teaches correct Memgraph traversal syntax."""
    assert "*BFS" in CYPHER_QUERY_RULES
    assert "*KSHORTEST" in CYPHER_QUERY_RULES
    # BFS must use range syntax (1..) not "1 TO" — verified against Memgraph 3.8.1
    bfs_section = CYPHER_QUERY_RULES.split("*BFS")[1].split("\n")[0]
    assert "1.." in bfs_section, f"BFS prompt should use range syntax '1..': {bfs_section}"
    assert "1 TO" not in bfs_section, f"BFS prompt should NOT use '1 TO' syntax: {bfs_section}"