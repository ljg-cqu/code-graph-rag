from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.services.docstring_inference import (
    DocstringInferenceResult,
    DocstringInferer,
    _build_functions_query,
)


def test_build_functions_query_high_pagerank() -> None:
    query = _build_functions_query("high-pagerank")
    assert "ORDER BY n.pagerank_score DESC" in query
    assert "LIMIT $limit" in query


def test_build_functions_query_alphabetical() -> None:
    query = _build_functions_query("alphabetical")
    assert "ORDER BY n.qualified_name ASC" in query
    assert "LIMIT $limit" in query


def test_docstring_inference_result_model() -> None:
    result = DocstringInferenceResult(
        processed=10, succeeded=8, failed=2, errors=["error1", "error2"]
    )
    assert result.processed == 10
    assert result.succeeded == 8
    assert result.failed == 2
    assert len(result.errors) == 2


def test_infer_for_functions_disabled_by_config() -> None:
    with patch("codebase_rag.config.settings.DOCSTRING_INFERENCE_ENABLED", False):
        inferer = DocstringInferer()
        result = asyncio.run(inferer.infer_for_functions(limit=10))

    assert result.processed == 0
    assert result.succeeded == 0
    assert result.failed == 0
    assert result.errors == []


def test_infer_for_functions_empty_graph() -> None:
    mock_ingestor = MagicMock()
    mock_ingestor.fetch_all.return_value = []
    mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
    mock_ingestor.__exit__ = MagicMock(return_value=False)

    with (
        patch(
            "codebase_rag.services.docstring_inference.MemgraphIngestor",
            return_value=mock_ingestor,
        ),
        patch("codebase_rag.config.settings.DOCSTRING_INFERENCE_ENABLED", True),
    ):
        inferer = DocstringInferer()
        result = asyncio.run(inferer.infer_for_functions(limit=10))

    assert result.processed == 0
    assert result.succeeded == 0
    assert result.failed == 0
    assert result.errors == []


def test_infer_for_functions_success() -> None:
    mock_ingestor = MagicMock()
    mock_ingestor.fetch_all.return_value = [
        {
            "node_id": 1,
            "qualified_name": "pkg.module.func",
            "name": "func",
            "file_path": "/repo/pkg/module.py",
            "start_line": 10,
            "end_line": 20,
            "pagerank_score": 0.5,
        }
    ]
    mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
    mock_ingestor.__exit__ = MagicMock(return_value=False)

    mock_agent_run = MagicMock()
    mock_agent_run.output = '"""A helpful docstring."""'

    with (
        patch(
            "codebase_rag.services.docstring_inference.MemgraphIngestor",
            return_value=mock_ingestor,
        ),
        patch(
            "codebase_rag.services.docstring_inference.Agent.run",
            return_value=mock_agent_run,
        ),
        patch(
            "codebase_rag.services.docstring_inference.extract_source_lines",
            return_value="def func():\n    pass",
        ),
        patch("codebase_rag.config.settings.RATE_LIMIT_ENABLED", False),
        patch("codebase_rag.config.settings.DOCSTRING_INFERENCE_ENABLED", True),
    ):
        inferer = DocstringInferer()
        result = asyncio.run(inferer.infer_for_functions(limit=10))

    assert result.processed == 1
    assert result.succeeded == 1
    assert result.failed == 0
    assert result.errors == []
    mock_ingestor.execute_write.assert_called_once()


def test_infer_for_functions_alphabetical_priority() -> None:
    mock_ingestor = MagicMock()
    mock_ingestor.fetch_all.return_value = [
        {
            "node_id": 1,
            "qualified_name": "alpha.func",
            "name": "func",
            "file_path": "/repo/alpha.py",
            "start_line": 1,
            "end_line": 5,
            "pagerank_score": 0.1,
        }
    ]
    mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
    mock_ingestor.__exit__ = MagicMock(return_value=False)

    mock_agent_run = MagicMock()
    mock_agent_run.output = '"""Docstring."""'

    with (
        patch(
            "codebase_rag.services.docstring_inference.MemgraphIngestor",
            return_value=mock_ingestor,
        ),
        patch(
            "codebase_rag.services.docstring_inference.Agent.run",
            return_value=mock_agent_run,
        ),
        patch(
            "codebase_rag.services.docstring_inference.extract_source_lines",
            return_value="def func(): pass",
        ),
        patch("codebase_rag.config.settings.RATE_LIMIT_ENABLED", False),
        patch("codebase_rag.config.settings.DOCSTRING_INFERENCE_ENABLED", True),
    ):
        inferer = DocstringInferer()
        result = asyncio.run(
            inferer.infer_for_functions(limit=10, priority="alphabetical")
        )

    assert result.processed == 1
    assert result.succeeded == 1
    assert result.failed == 0
    cypher = mock_ingestor.fetch_all.call_args.args[0]
    assert "ORDER BY n.qualified_name ASC" in cypher


def test_infer_for_functions_source_read_failure() -> None:
    mock_ingestor = MagicMock()
    mock_ingestor.fetch_all.return_value = [
        {
            "node_id": 1,
            "qualified_name": "pkg.module.func",
            "name": "func",
            "file_path": "/repo/pkg/module.py",
            "start_line": 10,
            "end_line": 20,
            "pagerank_score": 0.5,
        }
    ]
    mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
    mock_ingestor.__exit__ = MagicMock(return_value=False)

    with (
        patch(
            "codebase_rag.services.docstring_inference.MemgraphIngestor",
            return_value=mock_ingestor,
        ),
        patch(
            "codebase_rag.services.docstring_inference.extract_source_lines",
            return_value=None,
        ),
        patch("codebase_rag.config.settings.RATE_LIMIT_ENABLED", False),
        patch("codebase_rag.config.settings.DOCSTRING_INFERENCE_ENABLED", True),
    ):
        inferer = DocstringInferer()
        result = asyncio.run(inferer.infer_for_functions(limit=10))

    assert result.processed == 1
    assert result.succeeded == 0
    assert result.failed == 1
    assert len(result.errors) == 1
    assert "No source" in result.errors[0]


def test_infer_for_functions_quota_exhausted() -> None:
    mock_ingestor = MagicMock()
    mock_ingestor.fetch_all.return_value = [
        {
            "node_id": 1,
            "qualified_name": "pkg.module.func",
            "name": "func",
            "file_path": "/repo/pkg/module.py",
            "start_line": 10,
            "end_line": 20,
            "pagerank_score": 0.5,
        },
        {
            "node_id": 2,
            "qualified_name": "pkg.module.func2",
            "name": "func2",
            "file_path": "/repo/pkg/module.py",
            "start_line": 22,
            "end_line": 30,
            "pagerank_score": 0.3,
        },
    ]
    mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
    mock_ingestor.__exit__ = MagicMock(return_value=False)

    from codebase_rag.rate_limiter import QuotaStatus

    mock_limiter = MagicMock()
    mock_limiter.check_quota.return_value = QuotaStatus.EXHAUSTED

    with (
        patch(
            "codebase_rag.services.docstring_inference.MemgraphIngestor",
            return_value=mock_ingestor,
        ),
        patch(
            "codebase_rag.services.docstring_inference.get_rate_limiter",
            return_value=mock_limiter,
        ),
        patch("codebase_rag.config.settings.RATE_LIMIT_ENABLED", True),
        patch("codebase_rag.config.settings.DOCSTRING_INFERENCE_ENABLED", True),
    ):
        inferer = DocstringInferer()
        result = asyncio.run(inferer.infer_for_functions(limit=10))

    assert result.processed == 2
    assert result.succeeded == 0
    assert result.failed == 0
    assert len(result.errors) == 1
    assert "Quota exhausted" in result.errors[0]


def test_infer_for_functions_quota_warning_continues() -> None:
    mock_ingestor = MagicMock()
    mock_ingestor.fetch_all.return_value = [
        {
            "node_id": 1,
            "qualified_name": "pkg.module.func",
            "name": "func",
            "file_path": "/repo/pkg/module.py",
            "start_line": 10,
            "end_line": 20,
            "pagerank_score": 0.5,
        }
    ]
    mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
    mock_ingestor.__exit__ = MagicMock(return_value=False)

    mock_agent_run = MagicMock()
    mock_agent_run.output = '"""A docstring."""'

    from codebase_rag.rate_limiter import QuotaStatus

    mock_limiter = MagicMock()
    mock_limiter.check_quota.return_value = QuotaStatus.WARNING

    with (
        patch(
            "codebase_rag.services.docstring_inference.MemgraphIngestor",
            return_value=mock_ingestor,
        ),
        patch(
            "codebase_rag.services.docstring_inference.Agent.run",
            return_value=mock_agent_run,
        ),
        patch(
            "codebase_rag.services.docstring_inference.extract_source_lines",
            return_value="def func():\n    pass",
        ),
        patch(
            "codebase_rag.services.docstring_inference.get_rate_limiter",
            return_value=mock_limiter,
        ),
        patch("codebase_rag.config.settings.RATE_LIMIT_ENABLED", True),
        patch("codebase_rag.config.settings.DOCSTRING_INFERENCE_ENABLED", True),
    ):
        inferer = DocstringInferer()
        result = asyncio.run(inferer.infer_for_functions(limit=10))

    assert result.processed == 1
    assert result.succeeded == 1
    assert result.failed == 0


def test_docstring_inferer_init_registers_rate_limiter() -> None:
    mock_limiter = MagicMock()

    with (
        patch(
            "codebase_rag.services.docstring_inference.get_rate_limiter",
            return_value=mock_limiter,
        ),
        patch("codebase_rag.config.settings.RATE_LIMIT_ENABLED", True),
    ):
        DocstringInferer()

    mock_limiter.register_provider.assert_called_once()


def test_docstring_inferer_init_failure_raises() -> None:
    with (
        patch(
            "codebase_rag.services.docstring_inference.Agent",
            side_effect=RuntimeError("pydantic-ai not available"),
        ),
        pytest.raises(Exception),
    ):
        DocstringInferer()
