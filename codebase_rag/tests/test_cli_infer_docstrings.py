from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from codebase_rag.cli import app

runner = CliRunner()


@pytest.fixture
def mock_memgraph_ingestor() -> Generator[MagicMock, None, None]:
    with patch("codebase_rag.services.graph_service.MemgraphIngestor") as mock_cls:
        mock_ingestor = MagicMock()
        mock_cls.return_value.__enter__ = MagicMock(return_value=mock_ingestor)
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_ingestor


def test_infer_docstrings_help_shows_command() -> None:
    result = runner.invoke(app, ["infer-docstrings", "--help"])
    assert result.exit_code == 0
    assert "--batch-size" in result.output
    assert "--limit" in result.output
    assert "--priority" in result.output
    assert "--dry-run" in result.output


def test_infer_docstrings_invalid_priority_exits_error() -> None:
    result = runner.invoke(
        app,
        ["infer-docstrings", "--priority", "invalid"],
    )
    assert result.exit_code == 1
    assert "Invalid priority" in result.output


def test_infer_docstrings_dry_run_queries_and_reports(
    mock_memgraph_ingestor: MagicMock,
) -> None:
    mock_memgraph_ingestor.fetch_all.return_value = [
        {"node_id": 1, "qualified_name": "pkg.mod.func"}
    ]

    result = runner.invoke(
        app,
        ["infer-docstrings", "--dry-run", "--limit", "10"],
    )

    assert result.exit_code == 0
    assert "Would process 1 functions" in result.output
    mock_memgraph_ingestor.fetch_all.assert_called_once()
