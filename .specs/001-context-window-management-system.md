# Context Window Management System Design Specification
**Version**: 1.0
**Date**: 2024-04-14
**Status**: Final
**Author**: Code Graph RAG Team

## Revision History
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2024-04-14 | Initial draft with complete design | Team |

## 1. Purpose
This document specifies the design and implementation requirements for the Context Window Management System in Code Graph RAG. The system addresses:
1. Updating the default context window from 128k to 256k to align with modern LLM capabilities
2. Fixing the reported `model_override_config is not defined` error
3. Adding flexible, multi-level context window configuration options
4. Providing automatic context window detection for common LLM models

## 2. Scope
### 2.1 In Scope
- Global default context window configuration
- Role-specific context window overrides (orchestrator, cypher)
- Provider/model-specific context window overrides
- Automatic context window detection for popular LLM models across all supported providers
- Fix for the out-of-scope `model_override_config` error in the `/compress` command
- Integration with existing context compression logic
- Complete documentation of configuration options

### 2.2 Out of Scope
- Dynamic context window adjustment based on real-time token usage
- Runtime API endpoints for modifying context window values
- Support for custom model context windows outside of env var configuration

## 3. Requirements
### 3.1 Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Default context window shall be 256,000 tokens | Critical |
| FR2 | Users shall be able to set a global default context window via environment variable | High |
| FR3 | Users shall be able to set role-specific context window overrides for orchestrator and cypher models | High |
| FR4 | Users shall be able to set provider/model-specific context window overrides via environment variables | High |
| FR5 | System shall automatically detect context window values for common LLM models per provider | Medium |
| FR6 | System shall gracefully fall back to lower-precedence values when detection fails | Critical |
| FR7 | The `/compress` command shall no longer throw `model_override_config is not defined` error | Critical |
| FR8 | Context compression logic shall use the correct context window value for both default and model override scenarios | High |
| FR9 | All new configuration options shall be documented in `.env.example` | High |

### 3.2 Non-Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| NFR1 | All changes shall be backwards compatible (no breaking changes to existing configurations) | Critical |
| NFR2 | No new external dependencies shall be added | Medium |
| NFR3 | Context window retrieval shall add <1ms latency to request processing | Medium |
| NFR4 | Configuration precedence shall be clearly documented and predictable | High |

## 4. System Architecture
### 4.1 Component Overview
The Context Window Management System integrates with 4 existing core components:

```
┌─────────────────────────┐    ┌─────────────────────────┐
│      Config Layer       │    │    Model Provider       │
│  (loads env vars,        │    │  (per-provider model    │
│   stores defaults)       │    │   context maps)         │
└─────────────────────────┘    └─────────────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│    Main Agent Loop      │    │  Context Compressor     │
│  (retrieves context      │    │  (uses context window   │
│   window for requests)   │    │   to trigger compression│
└─────────────────────────┘    └─────────────────────────┘
```

### 4.2 Configuration Precedence
Context window values are resolved in the following order (highest to lowest priority):
1. **Role-specific override**: `ORCHESTRATOR_CONTEXT_WINDOW` or `CYPHER_CONTEXT_WINDOW` env vars
2. **Provider/model-specific override**: `{PROVIDER}_{MODEL_ID}_CONTEXT_WINDOW` env var (e.g. `OPENAI_GPT_4O_CONTEXT_WINDOW`)
3. **Auto-detected value**: From the provider's pre-defined model context window map
4. **Global default**: `DEFAULT_CONTEXT_WINDOW` env var, or 256,000 if not set

## 5. Implementation Details
### 5.1 Config Layer Changes
**File**: `codebase_rag/config.py`
1. Add new constant to `AppConfig`:
   ```python
   CONTEXT_WINDOW_DEFAULT: int = 256000
   ```
2. Add support for new environment variables:
   - `DEFAULT_CONTEXT_WINDOW`: Global fallback value
   - `ORCHESTRATOR_CONTEXT_WINDOW`: Override for main agent model
   - `CYPHER_CONTEXT_WINDOW`: Override for Cypher generation model

### 5.2 Model Provider Layer Changes
**File**: `codebase_rag/providers/base.py`
1. Add new abstract method to `ModelProvider` base class:
   ```python
   def get_model_context_window(self, model_id: str) -> int:
       """
       Retrieve context window size for a given model ID.
       First checks for provider/model-specific env var override,
       then falls back to pre-defined model map, then global default.
       """
   ```
2. Add pre-defined context window maps for each provider:
   - **OpenAI**: GPT-4o (128k), GPT-3.5-turbo (128k), GPT-4 Turbo (128k), etc.
   - **Anthropic**: Claude 3.5 Sonnet (200k), Claude 3 Opus (200k), Claude 3 Sonnet (200k), etc.
   - **Google**: Gemini 2.5 Pro (1M), Gemini 1.5 Pro (2M), Gemini 2.5 Flash (1M), etc.
   - **Ollama**: Llama 3.1 (128k), Mistral Nemo (128k), Gemma 2 (128k), etc.
   - **Azure**: Same as OpenAI mapping
3. Implement env var override parsing:
   - Normalize model ID: replace spaces, hyphens, and periods with underscores, convert to uppercase
   - Check for env var in format: `{PROVIDER}_{NORMALIZED_MODEL_ID}_CONTEXT_WINDOW`
   - Return integer value if set and valid

### 5.3 Fix for `model_override_config` Error
**Root Cause**: The `/compress` command handler was trying to access the `model_override_config` variable which was out of scope, leading to the undefined error when retrieving context window.

**Solution**:
1. Modify `ContextCompressor` class to accept `max_context` as an explicit constructor parameter instead of relying on external scope variables
2. Update the `/compress` command handler to retrieve the context window using the same logic as the main agent loop:
   a. Check for `ORCHESTRATOR_CONTEXT_WINDOW` env var first
   b. Get current provider from active orchestrator config
   c. Call `get_model_context_window()` with the current model ID
   d. Fall back to global default if all previous steps fail
3. Pass the retrieved `max_context` value to the ContextCompressor constructor

### 5.4 Main Agent Loop Integration
**File**: `codebase_rag/main.py`
1. Update context window retrieval logic in `_run_agent_response_loop()`:
   a. Check for role-specific env var override first (based on current role: orchestrator/cypher)
   b. If using a model override, retrieve context window from the override model's provider
   c. Otherwise, use the active config's provider to get the context window
   d. Fall back to global default on any failure
2. Update log messages to show correct default value (256k) when retrieval fails:
   ```python
   logger.debug(f"Failed to retrieve model context window, using default {settings.CONTEXT_WINDOW_DEFAULT}: {e}")
   ```

### 5.5 Context Compressor Updates
**File**: `codebase_rag/context_compressor.py`
1. Add `max_context` parameter to `__init__` method:
   ```python
   def __init__(
       self,
       context: list[dict],
       max_context: int,
       aggressive_mode: bool = False,
       preserve_pattern: str | None = None,
       worker_count: int = settings.CONTEXT_COMPRESSION_PARALLEL_WORKERS,
   ):
       self.max_context = max_context
       # ... rest of initialization
   ```
2. Update compression threshold calculation to use `self.max_context` instead of hardcoded value

## 6. Configuration Specification
### 6.1 Supported Environment Variables
| Variable Name | Format | Description | Example |
|---------------|--------|-------------|---------|
| `DEFAULT_CONTEXT_WINDOW` | Integer | Global fallback context window size (default: 256000) | `DEFAULT_CONTEXT_WINDOW=256000` |
| `ORCHESTRATOR_CONTEXT_WINDOW` | Integer | Override context window for main agent model (highest precedence) | `ORCHESTRATOR_CONTEXT_WINDOW=256000` |
| `CYPHER_CONTEXT_WINDOW` | Integer | Override context window for Cypher generation model | `CYPHER_CONTEXT_WINDOW=128000` |
| `{PROVIDER}_{MODEL_ID}_CONTEXT_WINDOW` | Integer | Provider/model-specific override. Model ID is normalized to uppercase with underscores replacing spaces/hyphens/periods | `OPENAI_GPT_4O_CONTEXT_WINDOW=128000`<br>`ANTHROPIC_CLAUDE_3_5_SONNET_CONTEXT_WINDOW=200000`<br>`GOOGLE_GEMINI_2_5_PRO_CONTEXT_WINDOW=1048576` |

### 6.2 Model ID Normalization Rules
1. Convert to uppercase
2. Replace spaces, hyphens (`-`), and periods (`.`) with underscores (`_`)
3. Remove any special characters

Examples:
- `gpt-4o` → `GPT_4O`
- `claude 3.5 sonnet` → `CLAUDE_3_5_SONNET`
- `gemini-2.5-pro-exp-0325` → `GEMINI_2_5_PRO_EXP_0325`

## 7. API Specification
### 7.1 ModelProvider.get_model_context_window()
```python
def get_model_context_window(self, model_id: str) -> int:
    """
    Retrieve context window size for a given model ID.
    
    Args:
        model_id: ID of the model to get context window for
        
    Returns:
        Context window size in tokens (integer)
        
    Precedence:
    1. Provider/model-specific env var override
    2. Pre-defined model map value for the provider
    3. Global default context window
    """
```

### 7.2 ContextCompressor Constructor
```python
def __init__(
    self,
    context: list[dict],
    max_context: int,
    aggressive_mode: bool = False,
    preserve_pattern: str | None = None,
    worker_count: int = settings.CONTEXT_COMPRESSION_PARALLEL_WORKERS,
):
    """
    Initialize context compressor.
    
    Args:
        context: List of message dicts to compress
        max_context: Maximum context window size in tokens
        aggressive_mode: Enable more aggressive compression (lower retention threshold)
        preserve_pattern: Regex pattern for content to preserve during compression
        worker_count: Number of parallel workers to use for compression
    """
```

## 8. Testing Plan
### 8.1 Unit Tests
1. **Config Tests**: Verify all new env vars are loaded correctly, default value is 256000
2. **Model Provider Tests**:
   - Test context window detection for common models across all providers
   - Test provider/model-specific env var overrides work correctly
   - Test fallback to global default for unknown models
3. **Context Compressor Tests**: Verify compression threshold calculation uses the provided `max_context` parameter correctly
4. **Error Fix Test**: Verify `/compress` command no longer throws `model_override_config` undefined error

### 8.2 Integration Tests
1. Test context window retrieval in main agent loop with all precedence levels
2. Test context compression triggers correctly at the expected threshold for different context window sizes
3. Test role-specific overrides work correctly for both orchestrator and cypher models

### 8.3 Edge Case Tests
1. Test invalid env var values (non-integer, negative numbers) fall back to next precedence level
2. Test unknown model IDs fall back to global default
3. Test model override scenario uses the correct context window for the overridden model

## 9. Migration Guide
This release is fully backwards compatible with no breaking changes:
- Existing configurations will continue to work without modification
- The only default behavior change is the context window fallback value increasing from 128k to 256k
- Users who want to keep the old 128k default can set:
  ```env
  DEFAULT_CONTEXT_WINDOW=128000
  ```
- Users can take advantage of new configuration options by adding the appropriate env vars to their `.env` file as documented

## 10. Documentation Updates
1. Update `.env.example` with all new configuration options and examples
2. Add context window configuration section to main README
3. Update error messages and logging to reflect new default value

## 11. Future Enhancements
1. Add runtime API endpoints to modify context window values without restarting
2. Add dynamic context window adjustment based on real-time token usage metrics
3. Expand default model context window maps with more community-contributed models
4. Add support for context window limits per request type
