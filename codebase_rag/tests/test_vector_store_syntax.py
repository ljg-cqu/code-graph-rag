"""Unit tests for Memgraph syntax correctness.

These tests verify that the codebase uses correct Memgraph Cypher syntax
rather than Neo4j syntax, which would cause runtime failures.
"""

import inspect


def test_bfs_uses_correct_memgraph_syntax():
    """Verify BFS traversal uses Memgraph syntax, not Neo4j."""
    from codebase_rag.vector_store_memgraph import MemgraphBackend

    # Check the source code for correct syntax
    source = inspect.getsource(MemgraphBackend.search)
    assert "*BFS (1.." not in source, "Found Neo4j-style BFS syntax (1..n)"
    assert "*BFS 1 TO" in source or "*BFS " in source, "Expected Memgraph BFS syntax"


def test_kshortest_uses_correct_memgraph_syntax():
    """Verify KSHORTEST uses Memgraph syntax, not Neo4j."""
    from codebase_rag.memgraph_advanced.path_analysis import PathAnalyzer

    source = inspect.getsource(PathAnalyzer.analyze_call_chain)
    assert "..$max_length" not in source, "Found Neo4j-style KSHORTEST syntax"
    assert "1 TO" in source, "Expected Memgraph KSHORTEST syntax"


def test_no_jinja2_in_cypher():
    """Verify no Jinja2 template syntax in Cypher queries."""
    from codebase_rag.memgraph_advanced.path_analysis import PathAnalyzer

    for method_name in ['find_bottlenecks', 'analyze_call_chain', 'find_similar_functions']:
        method = getattr(PathAnalyzer, method_name)
        source = inspect.getsource(method)
        assert "{%" not in source, f"Jinja2 template found in {method_name}"
        assert "%}" not in source, f"Jinja2 template found in {method_name}"