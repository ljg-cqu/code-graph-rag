"""Unit tests for Memgraph syntax correctness.

These tests verify that the codebase uses correct Memgraph Cypher syntax
rather than Neo4j syntax, which would cause runtime failures.
"""

import inspect
import re


def test_bfs_uses_correct_memgraph_syntax():
    """Verify BFS traversal uses Memgraph 3.8.1+ syntax."""
    from codebase_rag.vector_store_memgraph import MemgraphBackend

    # Memgraph 3.8.1 uses range syntax (*BFS 1..n) not "1 TO n"
    source = inspect.getsource(MemgraphBackend.search)
    assert "*BFS 1 TO" not in source, "Found invalid BFS syntax (1 TO n)"
    assert "*BFS 1.." in source, "Expected Memgraph BFS range syntax"


def test_hybrid_retrieval_bfs_syntax():
    """Verify hybrid retrieval BFS uses correct Memgraph syntax."""
    from codebase_rag.memgraph_advanced.hybrid_retrieval import HybridRetriever

    source = inspect.getsource(HybridRetriever._search_atomic)
    assert "*BFS 1 TO" not in source, "Found invalid BFS syntax in hybrid retrieval"
    assert "*BFS 1.." in source, "Expected Memgraph BFS range syntax in hybrid retrieval"


def test_graph_algorithms_bfs_syntax():
    """Verify graph algorithms BFS uses correct Memgraph syntax."""
    from codebase_rag.graph_algorithms import GraphAlgorithms

    source = inspect.getsource(GraphAlgorithms.get_bfs_context)
    assert "*BFS 1 TO" not in source, "Found invalid BFS syntax in graph algorithms"
    assert "*BFS 1.." in source, "Expected Memgraph BFS range syntax in graph algorithms"


def test_no_yield_where_without_with():
    """Verify no standalone CALL has YIELD followed directly by WHERE.

    Memgraph 3.8.1 requires WITH between YIELD and WHERE in standalone CALL.
    """
    from codebase_rag.vector_store_memgraph import MemgraphBackend
    from codebase_rag.memgraph_advanced.hybrid_retrieval import HybridRetriever

    for name, source in [
        ("MemgraphBackend.search", inspect.getsource(MemgraphBackend.search)),
        ("HybridRetriever._search_atomic", inspect.getsource(HybridRetriever._search_atomic)),
    ]:
        # Find all YIELD lines and check the next non-empty line starts a proper clause
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.upper().startswith("YIELD "):
                # Find next non-empty, non-comment line
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith("#"):
                        assert not next_line.upper().startswith("WHERE "), (
                            f"{name}: YIELD followed directly by WHERE (missing WITH). "
                            f"Line {j + 1}: {next_line[:60]}"
                        )
                        break


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