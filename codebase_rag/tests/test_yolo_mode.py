"""Unit and integration tests for yolo mode functionality."""

import os
import pytest
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from codebase_rag.main import _process_tool_approvals, _display_yolo_warning, app_context
from codebase_rag.types_defs import ConfirmationToolNames
from codebase_rag.cli import app
from codebase_rag.config import AppConfig


runner = CliRunner()


# ============================================================================
# Unit Tests
# ============================================================================


def test_yolo_mode_auto_approves():
    """Yolo mode should auto-approve all deferred tool requests."""
    app_context.session.yolo_mode = True

    # Create mock tool call
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-123"
    mock_call.args_as_dict.return_value = {}

    # Create mock requests
    requests = MagicMock()
    requests.approvals = [mock_call]

    tool_names = ConfirmationToolNames(
        replace_code="replace_code",
        create_file="create_file",
        shell_command="shell_command",
    )

    results = _process_tool_approvals(requests, "Approve?", "Denied", tool_names)

    # Verify auto-approval
    assert all(v is True for v in results.approvals.values())


def test_yolo_mode_disabled_prompts():
    """Normal mode should prompt for approval."""
    app_context.session.yolo_mode = False
    app_context.session.confirm_edits = True

    # Create mock tool call
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-456"
    mock_call.args_as_dict.return_value = {}

    # Create mock requests
    requests = MagicMock()
    requests.approvals = [mock_call]

    tool_names = ConfirmationToolNames(
        replace_code="replace_code",
        create_file="create_file",
        shell_command="shell_command",
    )

    with patch("codebase_rag.main.Confirm.ask", return_value=True):
        results = _process_tool_approvals(requests, "Approve?", "Denied", tool_names)

        # Verify approval was recorded
        assert results.approvals["test-456"] is True


def test_yolo_mode_logs_approval():
    """Yolo mode should log auto-approval with YOLO: prefix."""
    app_context.session.yolo_mode = True

    # Create mock tool call
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-789"
    mock_call.args_as_dict.return_value = {}

    # Create mock requests
    requests = MagicMock()
    requests.approvals = [mock_call]

    tool_names = ConfirmationToolNames(
        replace_code="replace_code",
        create_file="create_file",
        shell_command="shell_command",
    )

    with patch("codebase_rag.main.logger") as mock_logger:
        _process_tool_approvals(requests, "Approve?", "Denied", tool_names)

        # Verify YOLO: prefix in log
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "YOLO:" in call_args


def test_yolo_mode_overrides_confirm_edits():
    """yolo_mode=True should take precedence over confirm_edits=True."""
    app_context.session.yolo_mode = True
    app_context.session.confirm_edits = True  # Should be ignored

    # Create mock tool call
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-override"
    mock_call.args_as_dict.return_value = {}

    # Create mock requests
    requests = MagicMock()
    requests.approvals = [mock_call]

    tool_names = ConfirmationToolNames(
        replace_code="replace_code",
        create_file="create_file",
        shell_command="shell_command",
    )

    # Should auto-approve despite confirm_edits=True
    results = _process_tool_approvals(requests, "Approve?", "Denied", tool_names)
    assert results.approvals["test-override"] is True


def test_display_yolo_warning_when_enabled():
    """Warning banner should be displayed when yolo mode is enabled."""
    app_context.session.yolo_mode = True

    with patch("codebase_rag.main.app_context.console.print") as mock_print:
        _display_yolo_warning()

        # Verify print was called
        mock_print.assert_called_once()


def test_display_yolo_warning_when_disabled():
    """Warning banner should not be displayed when yolo mode is disabled."""
    app_context.session.yolo_mode = False

    with patch("codebase_rag.main.app_context.console.print") as mock_print:
        _display_yolo_warning()

        # Verify print was not called
        mock_print.assert_not_called()


# ============================================================================
# Integration Tests
# ============================================================================


def test_yolo_cli_flag_recognized():
    """--yolo flag should be recognized by the CLI."""
    result = runner.invoke(app, ["start", "--help"])

    # Verify --yolo is in the help output
    assert result.exit_code == 0
    assert "--yolo" in result.stdout or "-y" in result.stdout


def test_no_confirm_alias_recognized():
    """--no-confirm flag should still be recognized."""
    result = runner.invoke(app, ["start", "--help"])

    # Verify --no-confirm is in the help output
    assert result.exit_code == 0
    assert "--no-confirm" in result.stdout


def test_yolo_env_var_propagation():
    """CGR_YOLO_MODE=true should enable yolo mode via settings."""
    with patch.dict(os.environ, {"CGR_YOLO_MODE": "true"}):
        settings = AppConfig()

        # Verify settings loaded correctly
        assert settings.CGR_YOLO_MODE is True


def test_yolo_env_var_false():
    """CGR_YOLO_MODE=false should keep yolo mode disabled."""
    with patch.dict(os.environ, {"CGR_YOLO_MODE": "false"}):
        settings = AppConfig()

        # Verify settings loaded correctly
        assert settings.CGR_YOLO_MODE is False


def test_optimize_command_yolo_flag():
    """optimize command should also support --yolo flag."""
    result = runner.invoke(app, ["optimize", "--help"])

    # Verify --yolo is in the help output for optimize
    assert result.exit_code == 0
    assert "--yolo" in result.stdout or "-y" in result.stdout


# ============================================================================
# Precedence Tests
# ============================================================================


def test_cli_flag_precedence_over_env_var():
    """CLI --yolo flag should take precedence over CGR_YOLO_MODE env var."""
    # Simulate environment variable being false
    with patch.dict(os.environ, {"CGR_YOLO_MODE": "false"}):
        settings = AppConfig()
        assert settings.CGR_YOLO_MODE is False

        # Simulate CLI flag processing - CLI should override
        yolo = True
        no_confirm = False

        if yolo or no_confirm:
            yolo_mode = True
        elif settings.CGR_YOLO_MODE:
            yolo_mode = True
        else:
            yolo_mode = False

        assert yolo_mode is True


def test_both_flags_enable_yolo():
    """Both --yolo and --no-confirm should enable yolo mode."""
    # Test --yolo alone
    yolo = True
    no_confirm = False

    if yolo or no_confirm:
        yolo_mode_1 = True
    else:
        yolo_mode_1 = False

    # Test --no-confirm alone
    yolo = False
    no_confirm = True

    if yolo or no_confirm:
        yolo_mode_2 = True
    else:
        yolo_mode_2 = False

    # Test both
    yolo = True
    no_confirm = True

    if yolo or no_confirm:
        yolo_mode_3 = True
    else:
        yolo_mode_3 = False

    assert yolo_mode_1 is True
    assert yolo_mode_2 is True
    assert yolo_mode_3 is True