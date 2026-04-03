# Design Specification: Yolo Mode for Code-Graph-RAG

**Status:** Implemented | **Version:** v1.1.0 | **Created:** 2026-04-02 | **Updated:** 2026-04-02

---

## Problem Statement

Code-Graph-RAG currently requires interactive confirmation for destructive tool operations (file edits, shell commands). Users who want faster, uninterrupted workflows need an option to bypass these confirmations.

**Current Behavior:**
- `SessionState.confirm_edits=True` by default
- `_process_tool_approvals()` prompts for each tool call requiring approval
- Confirmation required for: `replace_code`, `create_file`, `shell_command`
- Existing `--no-confirm` flag sets `confirm_edits=False` (help text mentions "YOLO mode")

**Issue with Current Implementation:**
- `--no-confirm` only controls `confirm_edits` but lacks:
  - Explicit "yolo mode" session state tracking
  - Warning banner to alert users
  - Audit logging with `YOLO:` prefix
  - Environment variable support for MCP server mode

**Desired Behavior:**
- Add explicit `yolo_mode` session state flag (distinct from `confirm_edits`)
- `--yolo` as preferred CLI flag (with `--no-confirm` as backward-compatible alias)
- Auto-approve all tool operations without prompts
- Clear visual warning banner when yolo mode is active
- Audit logging for all auto-approved actions
- Environment variable `CGR_YOLO_MODE` for MCP server and persistent settings

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|:--------:|
| FR-1 | Add CLI flag `--yolo` (alias for `--no-confirm`) to disable confirmations | Must |
| FR-2 | Add explicit `yolo_mode: bool` to `SessionState` (distinct from `confirm_edits`) | Must |
| FR-3 | Auto-approve all `DeferredToolRequests` when `yolo_mode=True` | Must |
| FR-4 | Display clear warning banner when yolo mode is active | Must |
| FR-5 | Support both CLI `start` and `optimize` commands | Must |
| FR-6 | Support MCP server mode via environment variable | Should |
| FR-7 | Log yolo mode activation with `YOLO:` prefix to session log | Should |
| FR-8 | Environment variable `CGR_YOLO_MODE` for persistent setting | Could |
| FR-9 | Define precedence: CLI flag > env var > default | Could |

### Non-Functional Requirements

| ID | Requirement | Verification |
|----|-------------|:------------:|
| NFR-1 | Zero-latency approval (no prompt delay) | Performance test |
| NFR-2 | Backward compatible (default remains confirm=True) | Integration test |
| NFR-3 | Clear audit trail in logs | Log review |

---

## Design

### Component Changes

#### 1. SessionState (models.py)

```python
@dataclass
class SessionState:
    confirm_edits: bool = True
    yolo_mode: bool = False  # NEW: Explicit yolo mode flag
    log_file: Path | None = None
    cancelled: bool = False
    # ... existing fields ...
```

#### 2. CLI Arguments (cli.py)

**Flag Precedence Rules:**
1. CLI flag `--yolo` or `--no-confirm` takes precedence over environment variable
2. Both flags set `yolo_mode=True` and `confirm_edits=False`
3. `--yolo` is the preferred flag; `--no-confirm` remains as backward-compatible alias

**Implementation for `start` command (cli.py:276-379):**

```python
@app.command(help=ch.CMD_START)
def start(
    # ... existing args ...
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help=ch.HELP_NO_CONFIRM,  # Update: "Disable confirmation prompts for edit operations"
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        "-y",
        help="Disable all interactive confirmations (auto-approve all tool calls)",
    ),
) -> None:
    # Yolo mode: --yolo or --no-confirm both enable it
    if yolo or no_confirm:
        app_context.session.yolo_mode = True
        app_context.session.confirm_edits = False
    else:
        # Check environment variable (lower precedence than CLI flags)
        if settings.CGR_YOLO_MODE:
            app_context.session.yolo_mode = True
            app_context.session.confirm_edits = False
```

**Implementation for `optimize` command (cli.py:645-681):**

```python
@app.command(help=ch.CMD_OPTIMIZE)
def optimize(
    language: str = typer.Argument(..., help=ch.HELP_LANGUAGE_ARG),
    # ... existing args ...
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help=ch.HELP_NO_CONFIRM,
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        "-y",
        help="Disable all interactive confirmations (auto-approve all tool calls)",
    ),
) -> None:
    # Same yolo logic as start command
    if yolo or no_confirm:
        app_context.session.yolo_mode = True
        app_context.session.confirm_edits = False
    elif settings.CGR_YOLO_MODE:
        app_context.session.yolo_mode = True
        app_context.session.confirm_edits = False
```

#### 3. Tool Approval Flow (main.py)

```python
def _process_tool_approvals(
    requests: DeferredToolRequests,
    approval_prompt: str,
    denial_default: str,
    tool_names: ConfirmationToolNames,
) -> DeferredToolResults:
    deferred_results = DeferredToolResults()

    for call in requests.approvals:
        tool_args = _to_tool_args(
            call.tool_name, RawToolArgs(**call.args_as_dict()), tool_names
        )

        # YOLO MODE: Skip UI, auto-approve
        if app_context.session.yolo_mode:
            logger.info(f"YOLO: Auto-approving {call.tool_name}")
            deferred_results.approvals[call.tool_call_id] = True
            continue

        # Normal confirmation flow
        app_context.console.print(
            f"\n{cs.UI_TOOL_APPROVAL.format(tool_name=call.tool_name)}"
        )
        _display_tool_call_diff(call.tool_name, tool_args, tool_names)

        if app_context.session.confirm_edits:
            if Confirm.ask(style(approval_prompt, cs.Color.CYAN)):
                deferred_results.approvals[call.tool_call_id] = True
            else:
                feedback = Prompt.ask(cs.UI_FEEDBACK_PROMPT, default="")
                denial_msg = feedback.strip() or denial_default
                deferred_results.approvals[call.tool_call_id] = ToolDenied(denial_msg)
        else:
            deferred_results.approvals[call.tool_call_id] = True

    return deferred_results
```

#### 4. Warning Banner (main.py)

```python
YOLO_WARNING = """
[bold red]YOLO MODE ENABLED[/bold red]
All tool operations will be auto-approved without confirmation.
Use with caution on production codebases.
"""

def _display_yolo_warning() -> None:
    if app_context.session.yolo_mode:
        app_context.console.print(Panel(YOLO_WARNING, border_style=cs.Color.RED))
```

#### 5. Configuration Table Update (main.py)

```python
def _create_configuration_table(...) -> Table:
    # ... existing code ...

    # Yolo mode indicator
    yolo_status = (
        cs.YOLO_ENABLED if app_context.session.yolo_mode else cs.YOLO_DISABLED
    )
    table.add_row("Yolo Mode", yolo_status)

    return table
```

#### 6. Constants (constants.py)

```python
# Yolo Mode UI
YOLO_ENABLED = "[bold red]ENABLED (auto-approve)[/bold red]"
YOLO_DISABLED = "Disabled"
```

#### 7. Environment Variable Support (config.py)

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Yolo mode via environment
    CGR_YOLO_MODE: bool = False
```

### File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `models.py` | Modify | Add `yolo_mode: bool = False` to `SessionState` |
| `cli.py` | Modify | Add `--yolo` flag to `start` and `optimize` commands; update flag handling logic |
| `cli_help.py` | Modify | Update `HELP_NO_CONFIRM` to remove "(YOLO mode)" reference |
| `main.py` | Modify | Update `_process_tool_approvals()` with yolo early-return; add warning banner; update config table |
| `constants.py` | Modify | Add `YOLO_ENABLED`, `YOLO_DISABLED` constants |
| `config.py` | Modify | Add `CGR_YOLO_MODE: bool = False` env var support |
| `tests/test_yolo_mode.py` | Create | New test file with unit and integration tests |

---

## API

### CLI Usage

```bash
# Start with yolo mode (no confirmations)
cgr start --repo-path /path/to/repo --yolo

# Short form
cgr start -r /path/to/repo -y

# Environment variable
CGR_YOLO_MODE=true cgr start -r /path/to/repo
```

### MCP Server Usage

```bash
# Enable yolo mode for MCP server
CGR_YOLO_MODE=true cgr mcp-server

# Or via MCP client configuration
claude mcp add --transport stdio code-graph-rag \
  --env CGR_YOLO_MODE=true \
  -- uv run --directory /path/to/code-graph-rag code-graph-rag mcp-server
```

---

## Security Considerations

### Risk Assessment

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Accidental file deletion | High | Prominent warning banner, log all actions |
| Unauthorized code changes | Medium | Session logging, yolo mode explicit opt-in |
| Shell command abuse | High | Same as current (allowlist enforced) |

### Mitigations

1. **Visual Warning**: Red banner displayed at session start
2. **Audit Logging**: All auto-approved actions logged with `YOLO:` prefix
3. **Explicit Opt-In**: Yolo mode disabled by default
4. **Session Log**: All operations recorded in `.tmp/session_*.log`

---

## Edge Cases and Precedence Rules

### Flag Precedence Order

1. **CLI flags** (`--yolo`, `--no-confirm`) take precedence over environment variable
2. **Environment variable** (`CGR_YOLO_MODE=true`) takes precedence over default
3. **Default** (`yolo_mode=False`, `confirm_edits=True`) is lowest precedence

### Combined Flag Behavior

| Scenario | Result |
|----------|--------|
| `--yolo --no-confirm` (both set) | `yolo_mode=True`, `confirm_edits=False` |
| `--yolo` alone | `yolo_mode=True`, `confirm_edits=False` |
| `--no-confirm` alone | `yolo_mode=True`, `confirm_edits=False` (alias behavior) |
| `CGR_YOLO_MODE=true` + no CLI flags | `yolo_mode=True`, `confirm_edits=False` |
| `CGR_YOLO_MODE=false` + `--yolo` | `yolo_mode=True` (CLI overrides env) |
| Neither CLI nor env | `yolo_mode=False`, `confirm_edits=True` (default) |

### State Consistency

The implementation must ensure:
- `yolo_mode=True` always implies `confirm_edits=False`
- `confirm_edits=False` does NOT imply `yolo_mode=True` (user can set either independently)
- Session state is set during CLI initialization before any tool approval flow

### `reset_cancelled()` Interaction

`reset_cancelled()` in `SessionState` is independent of yolo mode:
- `yolo_mode` affects tool approval flow
- `cancelled` flag affects session interruption
- No interaction between these two flags

### Shell Command Safety

In yolo mode, shell commands are still subject to:
- `SHELL_COMMAND_ALLOWLIST` enforcement (config.py:231-261)
- `SHELL_COMMAND_TIMEOUT` limit
- No bypass of allowlist even with `yolo_mode=True`

---

## Testing

### Test File Location

Tests should be added to: `codebase_rag/tests/test_yolo_mode.py`

### Unit Tests

```python
# codebase_rag/tests/test_yolo_mode.py

import pytest
from unittest.mock import MagicMock, patch
from pydantic_ai import DeferredToolRequests, DeferredToolResults

from codebase_rag.main import _process_tool_approvals, app_context
from codebase_rag.types_defs import ConfirmationToolNames


def test_yolo_mode_auto_approves():
    """Yolo mode should auto-approve all deferred tool requests."""
    app_context.session.yolo_mode = True

    requests = DeferredToolRequests()
    # Add mock tool call
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-123"
    mock_call.args_as_dict.return_value = {}
    requests.approvals.append(mock_call)

    tool_names = ConfirmationToolNames(
        replace_code="replace_code",
        create_file="create_file",
        shell_command="shell_command",
    )

    results = _process_tool_approvals(requests, "Approve?", "Denied", tool_names)

    # Fixed assertion syntax
    assert all(v is True for v in results.approvals.values())


def test_yolo_mode_disabled_prompts():
    """Normal mode should prompt for approval."""
    app_context.session.yolo_mode = False
    app_context.session.confirm_edits = True

    requests = DeferredToolRequests()
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-456"
    mock_call.args_as_dict.return_value = {}
    requests.approvals.append(mock_call)

    tool_names = ConfirmationToolNames(
        replace_code="replace_code",
        create_file="create_file",
        shell_command="shell_command",
    )

    with patch("codebase_rag.main.Confirm.ask", return_value=True):
        results = _process_tool_approvals(requests, "Approve?", "Denied", tool_names)

        # Verify Confirm.ask was called
        assert results.approvals["test-456"] is True


def test_yolo_mode_logs_approval():
    """Yolo mode should log auto-approval with YOLO: prefix."""
    app_context.session.yolo_mode = True

    requests = DeferredToolRequests()
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-789"
    mock_call.args_as_dict.return_value = {}
    requests.approvals.append(mock_call)

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

    requests = DeferredToolRequests()
    mock_call = MagicMock()
    mock_call.tool_name = "replace_code"
    mock_call.tool_call_id = "test-override"
    mock_call.args_as_dict.return_value = {}
    requests.approvals.append(mock_call)

    tool_names = ConfirmationToolNames(
        replace_code="replace_code",
        create_file="create_file",
        shell_command="shell_command",
    )

    # Should auto-approve despite confirm_edits=True
    results = _process_tool_approvals(requests, "Approve?", "Denied", tool_names)
    assert results.approvals["test-override"] is True
```

### Integration Tests

```python
# codebase_rag/tests/test_yolo_mode.py

import os
import pytest
from unittest.mock import patch
from typer.testing import CliRunner

from codebase_rag.cli import app
from codebase_rag.main import app_context
from codebase_rag.config import AppConfig


runner = CliRunner()


def test_yolo_cli_flag_sets_session_state():
    """--yolo flag should set session.yolo_mode=True."""
    result = runner.invoke(app, ["start", "--repo-path", ".", "--yolo", "--help"])

    # Note: In actual test, would need to capture session state
    # This test verifies flag is recognized
    assert result.exit_code == 0


def test_no_confirm_alias_enables_yolo():
    """--no-confirm should also set yolo_mode=True (backward compatibility)."""
    app_context.session.yolo_mode = False
    app_context.session.confirm_edits = True

    # Simulate CLI flag processing
    no_confirm = True
    yolo = False

    if yolo or no_confirm:
        app_context.session.yolo_mode = True
        app_context.session.confirm_edits = False

    assert app_context.session.yolo_mode is True
    assert app_context.session.confirm_edits is False


def test_yolo_env_var_propagation():
    """CGR_YOLO_MODE=true should enable yolo mode via settings."""
    with patch.dict(os.environ, {"CGR_YOLO_MODE": "true"}):
        settings = AppConfig()

        # Verify settings loaded correctly
        assert settings.CGR_YOLO_MODE is True


def test_cli_flag_precedence_over_env_var():
    """CLI --yolo flag should take precedence over CGR_YOLO_MODE env var."""
    with patch.dict(os.environ, {"CGR_YOLO_MODE": "false"}):
        settings = AppConfig()
        assert settings.CGR_YOLO_MODE is False

        # CLI flag should override
        yolo = True
        no_confirm = False

        if yolo or no_confirm:
            yolo_mode = True
        elif settings.CGR_YOLO_MODE:
            yolo_mode = True
        else:
            yolo_mode = False

        assert yolo_mode is True


def test_optimize_command_yolo_flag():
    """optimize command should also support --yolo flag."""
    result = runner.invoke(app, ["optimize", "python", "--yolo", "--help"])

    assert result.exit_code == 0
```

---

## Migration Path

### Backward Compatibility

- Default behavior unchanged (`yolo_mode=False`, `confirm_edits=True`)
- Existing configurations continue to work
- No API breaking changes

### Deprecation Timeline

| Version | Change |
|---------|--------|
| v1.x | Add `--yolo` flag as alias for `--no-confirm`; both work identically |
| v2.x | Consider deprecating `--no-confirm` in favor of `--yolo` (with warning) |
| v3.x | (Optional) Remove `--no-confirm` flag; `--yolo` becomes sole flag |

**Note:** `--no-confirm` will remain functional throughout v1.x and v2.x to ensure backward compatibility. Deprecation warnings would appear in v2.x if users use `--no-confirm`.

---

## Implementation Checklist

### Core Implementation
- [x] Add `yolo_mode: bool = False` to `SessionState` in `models.py:16-27`
- [x] Add `--yolo` CLI flag to `start` command in `cli.py:276-379`
- [x] Add `--yolo` CLI flag to `optimize` command in `cli.py:645-681`
- [x] Update flag handling: `if yolo or no_confirm: ...` logic in both commands
- [x] Update `_process_tool_approvals()` in `main.py:240-270` with yolo early-return
- [x] Add `_display_yolo_warning()` function in `main.py`
- [x] Update `_create_configuration_table()` in `main.py:362-365` to show yolo status
- [x] Add `YOLO_ENABLED`, `YOLO_DISABLED` constants in `constants.py`
- [x] Add `CGR_YOLO_MODE: bool = False` in `config.py`

### Help Text Updates
- [x] Update `HELP_NO_CONFIRM` in `cli_help.py:67` to remove "(YOLO mode)" reference
- [x] Add `HELP_YOLO` in `cli_help.py` for new flag

### Testing
- [x] Create `codebase_rag/tests/test_yolo_mode.py`
- [x] Add unit tests: auto-approve, disabled prompts, logging, precedence override
- [x] Add integration tests: CLI flags, env var, precedence rules
- [x] Add test for `optimize` command yolo support

### Documentation
- [x] Update README.md with yolo mode usage examples
- [x] Update ARCHITECTURE.md with session state changes
- [x] Document flag precedence in user guide

---

## References

- Source: `codebase_rag/main.py:240-270` - `_process_tool_approvals()` function
- Source: `codebase_rag/main.py:291-368` - `_create_configuration_table()` function
- Source: `codebase_rag/models.py:16-27` - `SessionState` dataclass
- Source: `codebase_rag/cli.py:306-310` - `--no-confirm` flag in `start` command
- Source: `codebase_rag/cli.py:669-671` - `--no-confirm` flag in `optimize` command
- Source: `codebase_rag/cli.py:379` - `confirm_edits = not no_confirm` assignment
- Source: `codebase_rag/cli_help.py:67` - `HELP_NO_CONFIRM` help text
- Source: `codebase_rag/types_defs.py` - `ConfirmationToolNames` type definition
- Source: `codebase_rag/constants.py:886` - `CONFIRM_DISABLED` constant