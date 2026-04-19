# Realtime Update Feature Configuration Fix

## Overview

The realtime update feature has a critical gap between its documented `.env` configuration and actual implementation. The configuration options in `.env` (e.g., `CGR_REALTIME_UPDATER`, `CGR_REALTIME_DEBOUNCE`) are defined in `config.py` but are never used in the CLI or main application logic.

## Severity: **HIGH**

This is a functional bug - users who set these environment variables will not see the expected behavior.

## Issues Identified

### 1. CLI Flags Now Use .env Settings ✅

**Location:** `codebase_rag/cli.py:668-699`

The CLI flags now correctly use `settings` as defaults, allowing environment variable configuration:

```python
# Current (CORRECT)
realtime_updater: bool = typer.Option(
    settings.REALTIME_UPDATER_ENABLED,
    "--realtime-updater/--no-realtime-updater",
    ...
)
realtime_debounce: float = typer.Option(
    settings.REALTIME_DEBOUNCE_SECONDS,
    "--realtime-debounce",
    ...
)
realtime_max_wait: float = typer.Option(
    settings.REALTIME_MAX_WAIT_SECONDS,
    "--realtime-max-wait",
    ...
)
```

**Status:** Fixed as of implementation.

### 2. Configuration Settings Now Used ✅

**Location:** `codebase_rag/config.py:631-643`

The following settings are now properly referenced throughout the codebase:

| Setting | Environment Variable | Default | Status |
|---------|---------------------|---------|--------|
| `REALTIME_UPDATER_ENABLED` | `CGR_REALTIME_UPDATER` | `False` | **USED** |
| `REALTIME_CODE_ENABLED` | `CGR_REALTIME_CODE` | `True` | **USED** |
| `REALTIME_DOCS_ENABLED` | `CGR_REALTIME_DOCS` | `False` | **USED** |
| `REALTIME_JSON_ENABLED` | `CGR_REALTIME_JSON` | `False` | **USED** |
| `REALTIME_BATCH_SIZE` | `CGR_REALTIME_BATCH_SIZE` | `100` | **USED** (standalone script) |

### 3. Standalone Script Now Uses Settings ✅

**Location:** `realtime_updater.py:348-381`

The standalone `realtime_updater.py` script now correctly uses `settings` as defaults:

```python
debounce: Annotated[
    float,
    typer.Option(...),
] = settings.REALTIME_DEBOUNCE_SECONDS,
max_wait: Annotated[
    float,
    typer.Option(...),
] = settings.REALTIME_MAX_WAIT_SECONDS,
```

Additionally, `batch_size` defaults to `settings.REALTIME_BATCH_SIZE`.

### 4. Documentation Updated ✅

**Location:** `docs/guide/realtime-updates.md`

The documentation now includes a dedicated "Environment Configuration" section that explains how to use environment variables:
- `CGR_REALTIME_UPDATER` to enable via env
- `CGR_REALTIME_DEBOUNCE` for debounce config
- `CGR_REALTIME_CODE/DOCS/JSON` toggles

CLI flags override environment settings as expected.

### 5. Integration Now Complete ✅

**Location:** `codebase_rag/main.py:3074-3135`

The `_create_watcher_manager` function uses `realtime_config.enable_code`, `enable_docs`, `enable_json`. These values now come from CLI flags which default to `settings.REALTIME_*_ENABLED` settings, allowing environment variable configuration to flow through.

## Current vs Expected Behavior

## Current Status ✅

The expected behavior described below has been fully implemented. Environment variables now properly configure real-time update defaults, and CLI flags override them as expected.

### Expected Behavior (Now Implemented)

| Configuration | Source | Effect |
|---------------|--------|--------|
| `CGR_REALTIME_UPDATER=true` | `.env` | Watcher auto-enabled on `cgr start` |
| `CGR_REALTIME_DEBOUNCE=10` | `.env` | Default debounce is 10s |
| `CGR_REALTIME_CODE=false` | `.env` | Code file watching disabled by default |
| `CGR_REALTIME_DOCS=true` | `.env` | Doc file watching enabled by default |

All settings are now respected via `settings.REALTIME_*` defaults in CLI flags and `RealtimeConfig`.

## Implementation Status

The recommended fixes have been largely implemented. Below is the original plan with current status.

### Phase 1: CLI Flag Defaults (Priority: HIGH) ✅ **COMPLETED**

Update `codebase_rag/cli.py` to use settings as defaults. This has been implemented as shown below.

### Phase 2: Standalone Script Update (Priority: MEDIUM) ✅ **COMPLETED**

Update `realtime_updater.py` to use settings. This has been implemented.

### Phase 3: Documentation Update (Priority: MEDIUM) ✅ **COMPLETED**

Update `docs/guide/realtime-updates.md` to include environment variable configuration. This has been implemented.

### Phase 4: REALTIME_BATCH_SIZE Usage (Priority: LOW) ✅ **IMPLEMENTED**

`REALTIME_BATCH_SIZE` is now used in the watcher manager when a separate ingestor is created for real-time updates. The `_create_watcher_manager` function creates a dedicated `MemgraphIngestor` with `batch_size=settings.REALTIME_BATCH_SIZE` when no shared ingestor is provided. The watcher manager closes this ingestor when stopped.

## Testing Requirements

1. **Unit Tests**: Added tests for environment variable configuration and RealtimeConfig defaults – ✅ **COMPLETED**
2. **Integration Tests**: Added mock tests for env‑configured watcher startup and CLI flag precedence – ✅ **COMPLETED**
3. **Manual Testing**: Verify `.env` configuration takes effect – ⏳ **PENDING** (recommended for final validation)

## Files to Modify

| File | Change Type | Status |
|------|-------------|--------|
| `codebase_rag/cli.py` | Update CLI defaults | ✅ **Completed** |
| `realtime_updater.py` | Update standalone script defaults | ✅ **Completed** |
| `realtime_updater.py` | Add ingestor lifecycle management | ✅ **Completed** |
| `codebase_rag/main.py` | Modify `_create_watcher_manager` to use REALTIME_BATCH_SIZE | ✅ **Completed** |
| `docs/guide/realtime-updates.md` | Add env config documentation | ✅ **Completed** |

## Compatibility

- **Backward Compatible**: Yes – existing CLI usage unchanged
- **Breaking Changes**: None – only adds functionality
- **Migration**: None required

## Implementation Priority

1. **Immediate**: All unit and integration tests completed
2. **Short‑term**: Manual validation of `.env` configuration (optional)
3. **Medium‑term**: (none)
4. **Low**: (none)
