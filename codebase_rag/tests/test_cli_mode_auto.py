"""Unit tests for CLI --mode auto acceptance and forwarding."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from codebase_rag.cli import app

runner = CliRunner()


def test_start_accepts_mode_auto_and_forwards_none(tmp_path: Path) -> None:
    """Auto mode should be accepted and forwarded as None to trigger auto-detection."""
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
                "auto",
                "--no-check-freshness",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_validate.assert_called_once()
    mock_main_unified_async.assert_awaited_once()
    assert mock_main_unified_async.await_args.kwargs["query_mode"] is None


def test_start_rejects_invalid_mode_without_with_docs(tmp_path: Path) -> None:
    """Invalid non-code_only mode without --with-docs should error first."""
    with patch("codebase_rag.cli._update_and_validate_models") as mock_validate:
        result = runner.invoke(
            app,
            [
                "start",
                "--repo-path",
                str(tmp_path),
                "--mode",
                "invalid_mode",
                "--no-check-freshness",
            ],
        )

    assert result.exit_code == 1
    assert "requires document graph" in result.output
    mock_validate.assert_not_called()


def test_start_rejects_invalid_mode_with_with_docs(tmp_path: Path) -> None:
    """Invalid mode with --with-docs should list valid modes including auto."""
    with patch("codebase_rag.cli._update_and_validate_models") as mock_validate:
        result = runner.invoke(
            app,
            [
                "start",
                "--repo-path",
                str(tmp_path),
                "--with-docs",
                "--mode",
                "invalid_mode",
                "--no-check-freshness",
            ],
        )

    assert result.exit_code == 1
    assert "Invalid mode" in result.output
    assert "auto" in result.output
    mock_validate.assert_not_called()


def test_start_explicit_mode_bypasses_auto_detection(tmp_path: Path) -> None:
    """Explicit --mode code_only should bypass auto-detection."""
    from codebase_rag.shared.query_router import QueryMode

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
                "code_only",
                "--no-check-freshness",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_validate.assert_called_once()
    mock_main_unified_async.assert_awaited_once()
    assert mock_main_unified_async.await_args.kwargs["query_mode"] == QueryMode.CODE_ONLY
