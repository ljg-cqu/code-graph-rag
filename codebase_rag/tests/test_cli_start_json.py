from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from codebase_rag.cli import app

runner = CliRunner()


def test_start_forwards_json_exclude_to_indexing(tmp_path: Path) -> None:
    with (
        patch("codebase_rag.cli._update_and_validate_models") as mock_validate,
        patch(
            "codebase_rag.cli._handle_indexing",
            return_value=(True, False, False),
        ) as mock_handle_indexing,
    ):
        result = runner.invoke(
            app,
            [
                "start",
                "--repo-path",
                str(tmp_path),
                "--index-code",
                "--ingest-json",
                "--json-exclude",
                "foo/**",
                "--json-exclude",
                "bar.json",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_validate.assert_called_once()
    assert mock_handle_indexing.call_args.kwargs["json_exclude"] == [
        "foo/**",
        "bar.json",
    ]


def test_vector_recreate_indexes_uses_doc_and_json_helpers() -> None:
    with (
        patch("codebase_rag.cli.ensure_document_vector_index") as mock_doc_helper,
        patch("codebase_rag.cli.recreate_json_vector_index") as mock_json_helper,
        patch("codebase_rag.cli.connect_doc_memgraph") as mock_connect_doc,
    ):
        mock_ingestor = MagicMock()
        mock_connect_doc.return_value.__enter__.return_value = mock_ingestor
        mock_connect_doc.return_value.__exit__.return_value = False

        result = runner.invoke(
            app,
            [
                "vector",
                "recreate-indexes",
                "--no-code",
                "--docs",
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_doc_helper.assert_called_once()
    assert mock_doc_helper.call_args.args[0] is mock_ingestor
    mock_json_helper.assert_called_once()
