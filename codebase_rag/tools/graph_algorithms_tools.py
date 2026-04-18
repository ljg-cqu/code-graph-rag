"""Tools for advanced graph algorithms (CommunityQFS, PathAnalysis, etc.)."""

from __future__ import annotations

from pydantic_ai import Tool

from ..config import settings


async def community_summary(question: str, top_communities: int = 3) -> str:
    """Generate query-focused summary using community detection.

    Identifies code communities and summarizes the most relevant ones
    for a given question. Best for understanding large codebases.
    """
    from ..memgraph_advanced.qfs import CommunityQFS

    try:
        qfs = CommunityQFS()
        summary = qfs.query_focused_summary(
            question=question,
            top_communities=top_communities,
            min_community_size=settings.qfs_config.min_community_size,
        )
        return summary
    except Exception as e:
        return f"Community summary failed: {e}"


async def analyze_path(
    source: str,
    target: str,
    max_paths: int = 3,
    max_length: int = 10,
) -> str:
    """Analyze call paths between two functions.

    Finds k-shortest paths, identifies critical nodes and bottlenecks.
    Example: 'Analyze path from main() to database.connect()'
    """
    from ..memgraph_advanced.path_analysis import PathAnalyzer

    try:
        analyzer = PathAnalyzer()
        result = analyzer.analyze_call_chain(
            start_qn=source,
            end_qn=target,
            max_paths=max_paths,
            max_path_length=max_length,
        )

        if not result.paths:
            return f"No call paths found between '{source}' and '{target}'."

        lines = [
            f"Call paths from '{source}' to '{target}':",
            f"Shortest path length: {result.shortest_path_length}",
            f"Alternative paths: {result.alternative_paths}",
        ]

        if result.critical_nodes:
            lines.append(
                f"Critical nodes (appear in multiple paths): {', '.join(result.critical_nodes)}"
            )

        if result.bottlenecks:
            lines.append(f"Bottlenecks: {', '.join(result.bottlenecks)}")

        for i, path in enumerate(result.paths, 1):
            nodes = [n.get("qualified_name", "?") for n in path.get("nodes", [])]
            lines.append(f"\nPath {i} (length {path.get('length', '?')}):")
            lines.append(f"  {' -> '.join(nodes)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Path analysis failed: {e}"


async def find_similar_functions(
    function_name: str,
    min_similarity: float = 0.3,
    limit: int = 10,
) -> str:
    """Find functions similar to a given function based on call patterns.

    Uses Jaccard similarity to find functions with similar call patterns.
    Returns similarity scores.
    """
    from ..memgraph_advanced.path_analysis import PathAnalyzer

    try:
        analyzer = PathAnalyzer()
        results = analyzer.find_similar_functions(
            target_qn=function_name,
            min_similarity=min_similarity,
            limit=limit,
        )

        if not results:
            return f"No similar functions found for '{function_name}'."

        lines = [f"Functions similar to '{function_name}':"]
        for r in results:
            similar_fn = r.get("similar_function", "?")
            name = r.get("name", "?")
            score = r.get("jaccard_score", 0)
            lines.append(f"  - {similar_fn} ({name}): {score:.3f}")

        return "\n".join(lines)
    except Exception as e:
        return f"Similar functions search failed: {e}"


async def find_bottlenecks(
    function_qn: str | None = None,
    threshold: float = 0.01,
    limit: int = 20,
) -> str:
    """Find functions that are bottlenecks (high betweenness centrality).

    Identifies critical functions that control flow between different
    parts of the codebase.
    """
    from ..memgraph_advanced.path_analysis import PathAnalyzer

    try:
        analyzer = PathAnalyzer()
        results = analyzer.find_bottlenecks(
            function_qn=function_qn,
            threshold=threshold,
        )

        if not results:
            scope = f" related to '{function_qn}'" if function_qn else ""
            return f"No bottleneck functions found{scope}."

        lines = [f"Bottleneck functions (top {min(limit, len(results))}):"]
        for r in results[:limit]:
            fn = r.get("function_name", "?")
            name = r.get("name", "?")
            score = r.get("centrality_score", 0)
            in_deg = r.get("in_degree", 0)
            out_deg = r.get("out_degree", 0)
            lines.append(
                f"  - {fn} ({name}): centrality={score:.4f}, "
                f"in={in_deg}, out={out_deg}"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Bottleneck analysis failed: {e}"


async def expand_context(
    node_id: int,
    max_depth: int = 3,
) -> str:
    """Expand context around a code entity using BFS traversal.

    Returns related functions, classes, and modules ordered by importance.
    """
    from ..graph_algorithms import get_shared_algorithms

    try:
        algorithms = get_shared_algorithms()
        results = algorithms.get_bfs_context(
            start_node_id=node_id,
            max_depth=max_depth,
        )

        if not results:
            return f"No context found for node {node_id}."

        lines = [f"Context around node {node_id} (depth {max_depth}):"]
        for r in results:
            depth = r.get('depth', '?')
            qn = r.get('qualified_name', '?')
            pr = r.get('pagerank_score', 0)
            lines.append(f"  [depth {depth}, pagerank {pr:.4f}] {qn}")

        return "\n".join(lines)
    except Exception as e:
        return f"Context expansion failed: {e}"


def create_community_summary_tool() -> Tool:
    """Create tool for query-focused community summarization."""
    return Tool(
        community_summary,
        name="community_summary",
        description="Generate query-focused summary using community detection. "
        "Identifies code communities and summarizes the most relevant ones "
        "for a given question. Best for understanding large codebases.",
    )


def create_analyze_path_tool() -> Tool:
    """Create tool for path analysis between code entities."""
    return Tool(
        analyze_path,
        name="analyze_path",
        description="Analyze call paths between two functions. "
        "Finds k-shortest paths, identifies critical nodes and bottlenecks. "
        "Example: 'Analyze path from main() to database.connect()'",
    )


def create_find_similar_functions_tool() -> Tool:
    """Create tool for finding similar functions."""
    return Tool(
        find_similar_functions,
        name="find_similar_functions",
        description="Find functions similar to a given function based on call patterns "
        "using Jaccard similarity. Returns similarity scores.",
    )


def create_find_bottlenecks_tool() -> Tool:
    """Create tool for finding bottleneck functions."""
    return Tool(
        find_bottlenecks,
        name="find_bottlenecks",
        description="Find functions that are bottlenecks (high betweenness centrality). "
        "Identifies critical functions that control flow between different "
        "parts of the codebase.",
    )


def create_expand_context_tool() -> Tool:
    """Create tool for BFS context expansion."""
    return Tool(
        expand_context,
        name="expand_context",
        description="Expand context around a code entity using BFS traversal. "
        "Returns related functions, classes, and modules ordered by importance.",
    )
