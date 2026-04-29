"""Tests for CLI pre-flight graph availability check."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from codebase_rag.cli import app

runner = CliRunner()


@pytest.fixture
def mock_graph_connections() -> Generator[MagicMock, None, None]:
    with (
        patch("codebase_rag.cli.connect_memgraph") as mock_code,
        patch("codebase_rag.cli.connect_doc_memgraph") as mock_doc,
        patch("codebase_rag.cli.connect_concept_memgraph") as mock_concept,
        patch("codebase_rag.json_ingestion._create_json_ingestor") as mock_json,
    ):
        for mock in (mock_code, mock_doc, mock_concept, mock_json):
            mock_ingestor = MagicMock()
            mock.return_value.__enter__ = MagicMock(return_value=mock_ingestor)
            mock.return_value.__exit__ = MagicMock(return_value=False)
        yield {
            "code": mock_code,
            "doc": mock_doc,
            "concept": mock_concept,
            "json": mock_json,
        }


class TestPreflightAvailability:
    @patch("codebase_rag.cli.GraphUpdater")
    @patch("codebase_rag.cli.load_parsers", return_value=({}, {}))
    @patch("codebase_rag.cli.load_cgrignore_patterns")
    def test_shows_availability_table_for_code_graph(
        self,
        mock_cgrignore: MagicMock,
        mock_load_parsers: MagicMock,
        mock_graph_updater: MagicMock,
        mock_graph_connections: dict[str, MagicMock],
        tmp_path: Path,
    ) -> None:
        from codebase_rag.config import CgrignorePatterns

        mock_cgrignore.return_value = CgrignorePatterns(
            exclude=frozenset(), unignore=frozenset()
        )

        result = runner.invoke(
            app,
            ["start", "--index-code", "--repo-path", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Graph Availability Check" in result.output
        assert "Code Graph" in result.output
        assert "Available" in result.output
        assert mock_graph_connections["code"].call_count >= 1
        mock_graph_connections["doc"].assert_not_called()
        mock_graph_connections["json"].assert_not_called()

    @patch("codebase_rag.cli.GraphUpdater")
    @patch("codebase_rag.cli.load_parsers", return_value=({}, {}))
    @patch("codebase_rag.cli.load_cgrignore_patterns")
    @patch("codebase_rag.document.document_updater.DocumentGraphUpdater")
    def test_shows_unavailable_for_enterprise_license_error(
        self,
        mock_doc_updater: MagicMock,
        mock_cgrignore: MagicMock,
        mock_load_parsers: MagicMock,
        mock_graph_updater: MagicMock,
        mock_graph_connections: dict[str, MagicMock],
        tmp_path: Path,
    ) -> None:
        from codebase_rag.config import CgrignorePatterns

        mock_cgrignore.return_value = CgrignorePatterns(
            exclude=frozenset(), unignore=frozenset()
        )

        # Simulate enterprise license error on document graph
        mock_doc = mock_graph_connections["doc"]
        mock_doc.return_value.__enter__.side_effect = Exception(
            "Your license has an invalid type. To use multi-tenancy "
            "you need to have an enterprise license."
        )

        # Document updater should still be called (graceful degradation)
        mock_doc_updater.return_value.run.return_value = {
            "graph_available": False,
            "graph_error": "Enterprise license required",
        }

        result = runner.invoke(
            app,
            ["start", "--index-code", "--index-docs", "--repo-path", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Graph Availability Check" in result.output
        assert "Document Graph" in result.output
        assert "Unavailable (Enterprise license required)" in result.output


    @patch("codebase_rag.cli.GraphUpdater")
    @patch("codebase_rag.cli.load_parsers", return_value=({}, {}))
    @patch("codebase_rag.cli.load_cgrignore_patterns")
    def test_shows_json_graph_when_ingest_json_flag_passed(
        self,
        mock_cgrignore: MagicMock,
        mock_load_parsers: MagicMock,
        mock_graph_updater: MagicMock,
        mock_graph_connections: dict[str, MagicMock],
        tmp_path: Path,
    ) -> None:
        """Should show JSON Graph availability when --ingest-json is passed."""
        from codebase_rag.config import CgrignorePatterns

        mock_cgrignore.return_value = CgrignorePatterns(
            exclude=frozenset(), unignore=frozenset()
        )

        result = runner.invoke(
            app,
            ["start", "--index-code", "--ingest-json", "--repo-path", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Graph Availability Check" in result.output
        assert "JSON Graph" in result.output
        assert "Available" in result.output
        mock_graph_connections["json"].assert_called()

    @patch("codebase_rag.cli.main_unified_async")
    @patch("codebase_rag.document.document_updater.DocumentGraphUpdater")
    @patch("codebase_rag.cli.GraphUpdater")
    @patch("codebase_rag.cli.load_parsers", return_value=({}, {}))
    @patch("codebase_rag.cli.load_cgrignore_patterns")
    def test_shows_all_requested_graphs(
        self,
        mock_cgrignore: MagicMock,
        mock_load_parsers: MagicMock,
        mock_graph_updater: MagicMock,
        mock_doc_updater: MagicMock,
        mock_main_unified: MagicMock,
        mock_graph_connections: dict[str, MagicMock],
        tmp_path: Path,
    ) -> None:
        """Should show availability for all graphs when multiple flags are passed."""
        from codebase_rag.config import CgrignorePatterns

        mock_cgrignore.return_value = CgrignorePatterns(
            exclude=frozenset(), unignore=frozenset()
        )
        mock_doc_updater.return_value.run.return_value = {"graph_available": True}

        result = runner.invoke(
            app,
            [
                "start",
                "--index-code",
                "--index-docs",
                "--ingest-json",
                "--repo-path",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Graph Availability Check" in result.output
        assert "Code Graph" in result.output
        assert "Document Graph" in result.output
        assert "JSON Graph" in result.output

    @patch("codebase_rag.cli.main_async")
    @patch("codebase_rag.cli._check_graph_freshness", return_value=(True, True, []))
    @patch("codebase_rag.cli.GraphUpdater")
    @patch("codebase_rag.cli.load_parsers", return_value=({}, {}))
    @patch("codebase_rag.cli.load_cgrignore_patterns")
    def test_no_availability_check_when_no_indexing_flags(
        self,
        mock_cgrignore: MagicMock,
        mock_load_parsers: MagicMock,
        mock_graph_updater: MagicMock,
        mock_freshness: MagicMock,
        mock_main_async: MagicMock,
        mock_graph_connections: dict[str, MagicMock],
        tmp_path: Path,
    ) -> None:
        """Should not show availability table when no indexing flags are passed."""
        from codebase_rag.config import CgrignorePatterns

        mock_cgrignore.return_value = CgrignorePatterns(
            exclude=frozenset(), unignore=frozenset()
        )

        result = runner.invoke(
            app,
            ["start", "--repo-path", str(tmp_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Graph Availability Check" not in result.output
        mock_graph_connections["code"].assert_not_called()
        mock_graph_connections["doc"].assert_not_called()
        mock_graph_connections["json"].assert_not_called()
