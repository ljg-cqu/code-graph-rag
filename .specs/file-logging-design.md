# File Logging Feature Design Specification

## Overview
This feature adds file logging support to the code-graph-rag application, separating internal application logs from user-facing terminal output. All internal logs will be written to a file by default, while the terminal will only display user-focused messages (info, errors, success, progress).

## Current State Analysis
The application currently uses Loguru for logging, with the following behavior:
- All log messages are printed directly to the terminal
- The `--quiet/-q` flag suppresses non-error log messages
- User-facing output uses `app_context.console.print()` separate from the logger
- No persistent log storage exists

## Requirements
1. **Default File Logging**: Enable file logging by default, writing logs to a standard location
2. **Separation of Concerns**: Terminal only shows user-facing output, all internal logs go to file
3. **Configurable**: Allow users to configure log file path, log level, rotation, and retention
4. **Backward Compatible**: No breaking changes to existing CLI behavior
5. **CLI Flags**: Add flags to control logging behavior
6. **Environment Variables**: Support configuration via environment variables

## Design Details

### 1. New Configuration Settings (added to `codebase_rag/config/settings.py`)
| Setting | Type | Default | Description | Env Var |
|---------|------|---------|-------------|---------|
| `LOG_TO_FILE` | bool | `True` | Enable/disable file logging | `CGR_LOG_TO_FILE` |
| `LOG_FILE_PATH` | str | `~/.cache/cgr/cgr.log` | Path to log file | `CGR_LOG_FILE` |
| `LOG_LEVEL` | str | `INFO` | Log level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `CGR_LOG_LEVEL` |
| `LOG_ROTATION` | str | `10 MB` | Log rotation size/time | `CGR_LOG_ROTATION` |
| `LOG_RETENTION` | str | `30 days` | Log retention period | `CGR_LOG_RETENTION` |
| `LOG_COMPRESSION` | str | `zip` | Compression format for rotated logs | `CGR_LOG_COMPRESSION` |
| `LOG_TO_CONSOLE` | bool | `False` | Enable/disable log output to console (for debugging) | `CGR_LOG_TO_CONSOLE` |

### 2. CLI Flag Additions
Add the following global flags to the CLI (available for all commands):
| Flag | Description |
|------|-------------|
| `--log-file PATH` | Override default log file path |
| `--log-level LEVEL` | Set log level (DEBUG, INFO, WARNING, ERROR) |
| `--no-log-file` | Disable file logging entirely |
| `--debug-logs` | Shortcut to enable DEBUG log level and optional console logging |

### 3. Logging Initialization Modifications
Update the `_global_options` function in `codebase_rag/cli.py`:
1. Remove the default Loguru console handler by default (unless `LOG_TO_CONSOLE` is enabled or `--debug-logs` is used)
2. Add a file handler with configured rotation, retention, and compression if `LOG_TO_FILE` is True
3. Ensure log directory exists before initializing the file handler
4. Honor `--quiet` mode: when enabled, still write logs to file at configured level, only suppress terminal user output

### 4. Logging Behavior
- **User-facing output**: Continues to use `_info()`, `_error()`, `_success()` functions which use `console.print()` (unaffected by logging config)
- **Internal logs**: All `logger.*` calls (debug, info, warning, error, exception) are written only to the log file by default
- **Error messages**: Critical errors that require user attention are shown via both `console.print()` for user visibility and logged to file for debugging
- **Debug mode**: When `--debug-logs` is used, logs are printed to both console and file at DEBUG level

### 5. Default Log Location
The default log file will be stored in the user's cache directory:
- Linux: `~/.cache/cgr/cgr.log`
- macOS: `~/Library/Caches/cgr/cgr.log`
- Windows: `%LOCALAPPDATA%\cgr\cgr.log`

The directory will be created automatically if it does not exist.

## Implementation Plan

### Step 1: Update Settings
Add new logging configuration options to `settings.py` with proper validation and default values.

### Step 2: Modify CLI Initialization
Update the `_global_options` callback in `cli.py` to handle new logging flags and initialize the Loguru handlers correctly.

### Step 3: Add Log Directory Creation
Add utility function to create the log directory if it does not exist, with proper permission handling.

### Step 4: Test Integration
Verify that:
- Logs are written to file by default
- Terminal output only shows user-facing messages
- All CLI flags work as expected
- Configuration via environment variables works
- Log rotation and retention work correctly
- Backward compatibility is maintained (existing commands work without changes)

### Step 5: Update Documentation
Add logging configuration details to README.md and CLI help text.

## Impact Assessment
- No breaking changes to existing functionality
- Minimal performance overhead (Loguru file logging is optimized)
- Persistent logs improve debugging for users and support
- No external dependencies added (uses existing Loguru library)