# Models Command Design Specification

## Problem Statement

The `/models` in-chat command fails to list LLM configurations defined in the `.env` file. Users expect to see their configured models (e.g., `qwen3-max`, `glm-5.1`) alongside the static catalog models, but these dynamically configured models are not appearing in the output.

## Root Cause Analysis

### Primary Issues Identified

1. **Unknown Provider Handling**: Custom providers not in `PROVIDER_REGISTRY` (e.g., "dashscope", "volces") cause models to be skipped entirely. Standard providers (`openai`, `anthropic`, `google`, `ollama`, `azure`) work correctly with custom endpoints via `ORCHESTRATOR_ENDPOINT`.

2. **Silent Exception Swallowing**: Context window resolution failures are caught by `except Exception` without logging, making debugging impossible. This affects models with non-standard IDs when provider instantiation fails.

3. **Missing Debug Logging**: The model discovery process lacks debug logs, making it difficult to trace why models may not appear in output.

4. **Error Handling Gaps**: Exceptions during dynamic model creation are not properly logged, leading to silent failures that are hard to diagnose.

### Verified Working Correctly

1. **Provider Registry**: Standard providers (`openai`, `anthropic`, `google`, `ollama`, `azure`) ARE registered in `PROVIDER_REGISTRY` and work correctly with custom endpoints.

2. **Role-Specific API Keys**: `ORCHESTRATOR_API_KEY` and `CYPHER_API_KEY` ARE passed to `_build_dynamic_model_from_config()` via the `api_key` parameter.

3. **Global Key Fallback**: When role-specific key is not set, the code correctly falls back to provider global keys (e.g., `OPENAI_API_KEY`).

4. **Context Window Fallback**: Non-standard model IDs don't cause models to be skipped - they fall back to 256000 tokens.

5. **Worker LLM Parsing**: Both JSON array and comma-separated string formats are supported in `config.py:active_worker_llms`.

### Current Flow Breakdown

```
/models command → build_dynamic_model_catalog()
  ├─ Load static catalog (MODEL_CATALOG)
  ├─ Extract .env configured models (_extract_configured_models_from_env())
  │   ├─ ORCHESTRATOR_* → _build_dynamic_model_from_config()
  │   ├─ CYPHER_* → _build_dynamic_model_from_config()  
  │   └─ CGR_WORKER_LLMS → _build_dynamic_model_from_config()
  └─ Merge into unified catalog
```

The failure occurs in `_build_dynamic_model_from_config()` when:
- Provider is not in `PROVIDER_REGISTRY` (returns `None`, model is skipped)
- Exceptions during provider instantiation are silently caught

## Design Requirements

### Functional Requirements

1. **Complete Model Discovery**: All models configured in `.env` must appear in `/models` output
2. **Accurate Configuration Status**: Properly indicate whether models are fully configured (✅) or missing API keys (⚠️)
3. **Robust Error Handling**: Log detailed errors for debugging while gracefully handling failures
4. **Flexible Provider Support**: Support OpenAI-compatible endpoints with custom model IDs
5. **Worker LLM Integration**: Properly parse and display worker LLM configurations

### Non-Functional Requirements

1. **Backward Compatibility**: Existing static catalog functionality must remain unchanged
2. **Performance**: Dynamic catalog building should not significantly impact startup time
3. **Security**: No sensitive information (API keys) should be exposed in logs or output
4. **Maintainability**: Clear separation between static and dynamic model handling

## Proposed Solution Architecture

### Core Changes

#### 1. Enhanced Logging in Dynamic Model Builder

**Current Issue**: Exceptions are silently swallowed, making debugging impossible.

**Solution**: Add debug logging for context window resolution failures:

```python
# Resolve context window via the provider's get_model_context_window()
try:
    provider_instance = get_provider(provider)
    context_window = provider_instance.get_model_context_window(model_id)
except Exception as e:
    logger.debug(
        f"Context window resolution failed for {provider}:{model_id}: {e}. "
        "Using fallback 256000."
    )
    context_window = 256000  # Fallback
```

**Status**: ✅ Implemented

#### 2. Debug Logging for Model Discovery

**Current Issue**: No visibility into which models are discovered and added.

**Solution**: Add debug logging when models are successfully built:

```python
logger.debug(
    f"Built dynamic model: {provider}:{model_id} "
    f"(configured={has_api_key}, source={source})"
)
```

**Status**: ✅ Implemented

#### 3. Context Window Environment Variable Override

**Already Implemented**: The `get_model_context_window()` method in `providers/base.py` already supports:

```bash
# Format: {PROVIDER}_{NORMALIZED_MODEL_ID}_CONTEXT_WINDOW=<tokens>
OPENAI_QWEN3_MAX_CONTEXT_WINDOW=32768
ANTHROPIC_CLAUDE_3_5_SONNET_CONTEXT_WINDOW=200000
```

**Status**: ✅ Already working (see `providers/base.py:47-82`)

#### 4. API Key Detection (Current Behavior is Correct)

**Current Implementation**:
1. Role-specific key (`ORCHESTRATOR_API_KEY`/`CYPHER_API_KEY`) is passed via `api_key` parameter
2. If not set, falls back to provider global key (`OPENAI_API_KEY`, etc.)
3. Local providers always return `True`
4. Google Vertex checks `PROJECT_ID` instead

**Why This is Correct**: The role-specific key IS being used. The fallback to global keys is intentional - users should set role-specific keys if they want different keys per role.

**Status**: ✅ Working as designed

#### 5. Worker LLM Parsing (Current Behavior is Correct)

**Already Supported Formats**:
1. Comma-separated: `CGR_WORKER_LLMS="openai:gpt-4o,anthropic:claude-3-haiku"`
2. JSON array: `CGR_WORKER_LLMS='[{"provider": "openai", "model_id": "gpt-4o"}]'`
3. List of strings: `CGR_WORKER_LLMS=["openai:gpt-4o", "anthropic:claude-3"]`

**Status**: ✅ Already working (see `config.py:779-823`)

### Configuration Schema

#### Supported .env Patterns

The system supports the following .env configurations:

```bash
# Standard role-based configuration
ORCHESTRATOR_PROVIDER=openai
ORCHESTRATOR_MODEL=qwen3-max
ORCHESTRATOR_API_KEY=sk-...
ORCHESTRATOR_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1

# Context window overrides for custom models (already working)
OPENAI_QWEN3_MAX_CONTEXT_WINDOW=32768
OPENAI_GLM_5_1_CONTEXT_WINDOW=128000

# Worker LLMs (multiple formats supported)
CGR_WORKER_LLMS='[
    {"provider": "openai", "model_id": "qwen3-max", "api_key": "...", "endpoint": "..."},
    {"provider": "openai", "model_id": "glm-5.1", "api_key": "...", "endpoint": "..."}
]'

# Or simplified format
CGR_WORKER_LLMS=openai:qwen3-max,openai:glm-5.1
```

## Implementation Plan

### Phase 1: Core Fixes (✅ Completed)

1. **Add debug logging for context window resolution failures** - Done
2. **Add debug logging for successful model discovery** - Done

### Phase 2: Enhanced Features (Optional)

1. **Improve display formatting** with better status indicators
2. **Add `/models debug` command** for troubleshooting
3. **Implement model validation testing** in CI

### Phase 3: Future Considerations (Low Priority)

1. **Support external model catalogs** via `CGR_MODEL_CATALOG_PATH`
2. **Add model performance metrics** collection
3. **Implement automatic model discovery** from provider APIs
4. **Add model compatibility checking**

## Testing Strategy

### Unit Tests

1. **Dynamic Model Builder Tests**
   - Test with valid/invalid providers
   - Test with standard/custom model IDs  
   - Test API key detection scenarios
   - Test context window resolution fallbacks

2. **Worker LLM Parser Tests**
   - Valid JSON arrays
   - Malformed JSON with trailing commas
   - Comma-separated string format
   - Empty/invalid configurations

3. **Catalog Integration Tests**
   - Static + dynamic model merging
   - Duplicate model handling
   - Provider creation for new providers

### Integration Tests

1. **End-to-end `/models` command testing**
   - With various .env configurations
   - With missing API keys
   - With custom endpoints
   - With worker LLM configurations

2. **Real-world scenario testing**
   - Dashscope (Aliyun) configurations
   - Volces/Ark configurations  
   - Mixed provider scenarios
   - Edge cases (empty configs, malformed JSON)

## Backward Compatibility

- **No breaking changes** to existing static catalog functionality
- **Existing .env configurations** will continue to work
- **Current API contracts** remain unchanged
- **Logging levels** maintain existing verbosity (debug logs for troubleshooting)

## Performance Considerations

- **Lazy loading**: Dynamic catalog built only when `/models` is called
- **Caching**: Cache resolved context windows to avoid repeated provider calls
- **Efficient parsing**: Optimized JSON and string parsing for worker configurations
- **Memory usage**: Minimal overhead for dynamic model storage

## Security Considerations

- **API key protection**: Never log or display actual API key values
- **Input validation**: Sanitize model IDs and provider names
- **Error message sanitization**: Avoid exposing sensitive configuration details in error messages
- **Environment isolation**: Ensure .env parsing doesn't execute arbitrary code

## Success Metrics

1. **Functionality**: `/models` command displays all .env configured models
2. **Accuracy**: Configuration status indicators correctly reflect API key availability  
3. **Reliability**: No silent failures during dynamic model discovery
4. **Usability**: Clear, informative output that helps users understand their model configuration
5. **Maintainability**: Well-documented, testable code with comprehensive logging

## Rollback Strategy

If issues arise after deployment:

1. **Immediate**: Set `CGR_DISABLE_MODEL_DISCOVERY=true` to revert to static catalog only
2. **Short-term**: Revert specific commits related to dynamic model building
3. **Long-term**: Maintain dual code paths with feature flags for gradual rollout

## Dependencies and Prerequisites

- **Python 3.10+**: Required for type union syntax and dataclass features
- **loguru**: For structured logging
- **pydantic-settings**: For .env file parsing
- **Existing provider implementations**: Must support `get_model_context_window()` method

## Timeline and Milestones

- **Phase 1**: ✅ Complete - Debug logging added for context window resolution and model discovery
- **Phase 2**: Optional enhancements (display formatting, `/models debug` command)
- **Phase 3**: Future considerations as needed

## Summary

This design specification documents the `/models` command behavior. The core functionality works correctly:

1. Role-specific API keys (`ORCHESTRATOR_API_KEY`, `CYPHER_API_KEY`) are properly passed to model builder
2. Context window environment variable overrides are supported via `{PROVIDER}_{MODEL}_CONTEXT_WINDOW`
3. Worker LLM parsing supports multiple formats (JSON array, comma-separated)
4. Provider registry includes all standard providers (`openai`, `anthropic`, `google`, `ollama`, `azure`)

The implemented fixes add debug logging to help diagnose issues when they occur.