from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from codebase_rag.cli import app

runner = CliRunner()


def test_start_forwards_parallel_flags_to_main_async(tmp_path: Path) -> None:
    with (
        patch("codebase_rag.cli._update_and_validate_models") as mock_validate,
        patch(
            "codebase_rag.cli._handle_indexing",
            return_value=(False, False, False),
        ),
        patch("codebase_rag.cli.main_async", new_callable=AsyncMock) as mock_main_async,
    ):
        result = runner.invoke(
            app,
            [
                "start",
                "--repo-path",
                str(tmp_path),
                "--no-check-freshness",
                "--parallel-workers",
                "7",
                "--no-auto-split",
                "--no-parallel",
                "--parallel-dry-run",
                "--scheduling-strategy",
                "ROUND-ROBIN",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_validate.assert_called_once()
    mock_main_async.assert_awaited_once()
    assert mock_main_async.await_args.args[0] == str(tmp_path)

    parallel_config = mock_main_async.await_args.kwargs["parallel_config"]
    assert parallel_config.worker_count == 7
    assert parallel_config.auto_split is False
    assert parallel_config.no_parallel is True
    assert parallel_config.dry_run is True
    assert parallel_config.scheduling_strategy == "round-robin"


def test_start_forwards_parallel_flags_to_unified_runtime(tmp_path: Path) -> None:
    with (
        patch("codebase_rag.cli._update_and_validate_models") as mock_validate,
        patch(
            "codebase_rag.cli._handle_indexing",
            return_value=(False, False, True),
        ),
        patch(
            "codebase_rag.cli.main_unified_async",
            new_callable=AsyncMock,
        ) as mock_main_unified_async,
    ):
        result = runner.invoke(
            app,
            [
                "start",
                "--repo-path",
                str(tmp_path),
                "--with-docs",
                "--mode",
                "both_merged",
                "--doc-workspace",
                "parallel_docs",
                "--no-check-freshness",
                "--parallel-workers",
                "3",
                "--auto-split",
                "--scheduling-strategy",
                "fifo",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_validate.assert_called_once()
    mock_main_unified_async.assert_awaited_once()
    assert mock_main_unified_async.await_args.kwargs["doc_workspace"] == "parallel_docs"
    assert (
        mock_main_unified_async.await_args.kwargs["query_mode"].value == "both_merged"
    )

    parallel_config = mock_main_unified_async.await_args.kwargs["parallel_config"]
    assert parallel_config.worker_count == 3
    assert parallel_config.auto_split is True
    assert parallel_config.no_parallel is False
    assert parallel_config.dry_run is False
    assert parallel_config.scheduling_strategy == "fifo"
    assert parallel_config.doc_workspace == "parallel_docs"


def test_start_rejects_invalid_parallel_scheduling_strategy(tmp_path: Path) -> None:
    with patch("codebase_rag.cli._update_and_validate_models") as mock_validate:
        result = runner.invoke(
            app,
            [
                "start",
                "--repo-path",
                str(tmp_path),
                "--no-check-freshness",
                "--scheduling-strategy",
                "lifo",
            ],
        )

    assert result.exit_code == 1
    assert "Invalid scheduling strategy" in result.output
    mock_validate.assert_not_called()
