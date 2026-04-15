from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
