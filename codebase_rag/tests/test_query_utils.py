"""Unit tests for query utility functions (deprecated module).

Per the LLM-First Orchestration spec, extract_best_keyword and extract_keywords
have been removed. Use LLMQueryPlanner.expected_entities instead.
"""

import pytest


def test_query_utils_module_empty():
    """Verify that query_utils module no longer exports deprecated functions."""
    from codebase_rag.utils import query_utils

    # Module should have empty __all__
    assert query_utils.__all__ == []

    # Deprecated functions should not be importable
    with pytest.raises(ImportError):
        from codebase_rag.utils.query_utils import extract_best_keyword

    with pytest.raises(ImportError):
        from codebase_rag.utils.query_utils import extract_keywords


def test_llm_planner_replacement():
    """Verify LLMQueryPlanner is available as the replacement."""
    from codebase_rag.orchestrator.llm_query_planner import (
        LLMQueryPlanner,
        QueryPlan,
    )

    # Verify the classes exist and can be instantiated
    planner = LLMQueryPlanner()
    assert planner is not None

    # Verify QueryPlan has expected_entities field
    plan = QueryPlan(
        methods=["semantic_search"],
        reasoning="test",
        expected_entities=["entity1", "entity2"],
    )
    assert plan.expected_entities == ["entity1", "entity2"]
