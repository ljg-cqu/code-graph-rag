# Terminal-Only Logging Design Specification

## Overview
The application uses terminal-only logging for all application logs. All log messages are printed directly to the terminal, providing immediate feedback to users without file management overhead.

## Current State Analysis
The application currently uses Loguru for logging with terminal-only behavior:
- All log messages are printed directly to the terminal
- The `--quiet/-q` flag suppresses non-error log messages
- User-facing output uses `app_context.console.print()` separate from logger
- No persistent log storage

## Requirements
1. **Terminal Only**: All logs go to terminal output
2. **Simple Configuration**: Only log level needs to be configured
3. **Backward Compatible**: No breaking changes to existing CLI behavior
4. **CLI Flags**: Support log level control via CLI flag
5. **Environment Variables**: Support log level configuration via environment variables

## Design Details

### 1. Configuration Settings (in `codebase_rag/config/settings.py`)
| Setting | Type | Default | Description | Env Var |
|---------|------|---------|-------------|---------|
| `LOG_LEVEL` | str | `INFO` | Log level for terminal output | `CGR_LOG_LEVEL` |

### 2. CLI Flag Additions
| Flag | Description |
|------|-------------|
| `--log-level LEVEL` | Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### 3. Logging Initialization
The `_global_options` function in `codebase_rag/cli.py`:
1. Removes all default Loguru handlers
2. Adds a console handler with the configured log level
3. Respects `--quiet` mode by using ERROR level when quiet is enabled

### 4. Logging Behavior
- **User-facing output**: Uses `_info()`, `_error()`, `_success()` functions which use `console.print()`
- **Internal logs**: All `logger.*` calls (debug, info, warning, error, exception) are written to the terminal at the configured level
- **Error messages**: Critical errors are shown via `console.print()` and logged
- **Quiet mode**: When `--quiet` is enabled, only ERROR level messages are shown

## Impact Assessment
- No breaking changes to existing functionality
- Zero performance overhead for file operations
- Simplified configuration with fewer options
- No external dependencies beyond the existing Loguru library

## Migration from File Logging

If you were using file logging features (now removed):
- Logs previously written to files are now shown in the terminal
- Use terminal output redirection (`>` operator) to save logs if needed
- Configure log level via `CGR_LOG_LEVEL` or `--log-level` flag
