# Dynamic Environment Models Integration Specification

## Problem Statement

The current implementation has a critical usability gap: models configured via `.env` variables are not visible or accessible through the `/models` command interface. Users see only static built-in models from `MODEL_CATALOG`, many of which lack proper API key configuration and cannot function. This creates confusion and prevents users from leveraging their properly configured models.

### Current Issues
1. **Invisible Configured Models**: Models like `qwen3-max`, `glm-5.1`, `MiMo-V2-Pro` configured in `.env` don't appear in `/models` output
2. **Non-functional Built-ins**: Static catalog contains models without API keys that fail at runtime
3. **Poor User Experience**: Users cannot discover or switch to their working models via chat interface
4. **Security Risk**: Static catalog could potentially expose API key patterns if not handled properly

## Design Goals

1. **Dynamic Integration**: Automatically incorporate `.env` configured models into the model catalog
2. **Zero API Key Exposure**: Never store or display API keys in the catalog (security first)
3. **Context Window Awareness**: Support per-model context window configuration from `.env`
4. **Backward Compatibility**: Maintain existing static catalog for discovery while prioritizing configured models
5. **Provider Agnostic**: Work with any provider (OpenAI-compatible, Google, Anthropic, Ollama, etc.)
6. **Runtime Validation**: Only show models that can actually be used (have valid configuration)

## Solution Architecture

### Core Components

#### 1. Dynamic Model Catalog Builder
- **Location**: New module `codebase_rag/models_dynamic.py`
- **Purpose**: Build runtime model catalog by combining static catalog with `.env` configurations
- **Input Sources**:
  - Static `MODEL_CATALOG` from `models_catalog.py`
  - `.env` configured models (`ORCHESTRATOR_*`, `CYPHER_*`, `CGR_WORKER_LLMS`)
  - Context window overrides from `.env` (`*_CONTEXT_WINDOW` variables)

#### 2. Model Configuration Parser
- **Location**: Enhanced logic in `config.py`
- **Purpose**: Parse and validate model configurations from environment variables
- **Key Features**:
  - Extract provider, model_id, endpoint, context_window from `.env`
  - Validate API key presence without storing it
  - Support both role-based (`ORCHESTRATOR_*`) and array-based (`CGR_WORKER_LLMS`) configurations

#### 3. Unified Model Registry
- **Location**: New class in `models_dynamic.py`
- **Purpose**: Maintain single source of truth for all available models
- **Design Note**: The existing `ModelInfo` is a `NamedTuple`, which is immutable and does not support subclassing with new fields having defaults. Therefore, `DynamicModelInfo` must be a **`dataclass(frozen=True)`** to preserve immutability semantics while allowing default values on new fields. It does NOT inherit from `ModelInfo`; instead it contains all the same fields plus new ones, and provides a `from_model_info()` factory for converting static entries.
- **Data Structure**: 
  ```python
  @dataclass(frozen=True)
  class DynamicModelInfo:
      """Enhanced model info with dynamic configuration.
      
      NOT a subclass of ModelInfo (NamedTuple) — uses frozen dataclass
      to allow new fields with defaults while preserving immutability.
      """
      provider: str
      model_id: str  
      display_name: str
      context_window: int
      description: str
      requires_api_key: bool
      is_local: bool
      pricing_tier: Literal["free", "low", "medium", "high"]
      # New fields for dynamic models
      is_configured: bool = False  # True if API key is available or provider is local
      endpoint: str | None = None  # Custom endpoint if configured
      source: Literal["static", "env_orchestrator", "env_cypher", "env_worker"] = "static"

      @classmethod
      def from_model_info(cls, info: ModelInfo, **overrides) -> DynamicModelInfo:
          """Convert a static ModelInfo (NamedTuple) to DynamicModelInfo.
          
          Note: context_window should be resolved via get_model_context_window()
          at build time, not copied from the static ModelInfo. The static
          context_window is the baseline but may be overridden by env vars
          (e.g., OPENAI_GPT_4O_CONTEXT_WINDOW) or provider MODEL_CONTEXT_WINDOWS
          dict entries. Callers should pass the resolved value in **overrides.
          """
          return cls(
              provider=info.provider,
              model_id=info.model_id,
              display_name=info.display_name,
              context_window=overrides.pop("context_window", info.context_window),
              description=info.description,
              requires_api_key=info.requires_api_key,
              is_local=info.is_local,
              pricing_tier=info.pricing_tier,
              **overrides,
          )
  ```

### Implementation Details

#### Environment Variable Parsing Strategy

The system will parse these `.env` configurations to discover dynamically configured models:

1. **Role-based configurations** (already exist in `.env.example`):
   - `ORCHESTRATOR_PROVIDER`, `ORCHESTRATOR_MODEL`, `ORCHESTRATOR_ENDPOINT`, `ORCHESTRATOR_API_KEY`
   - `CYPHER_PROVIDER`, `CYPHER_MODEL`, `CYPHER_ENDPOINT`, `CYPHER_API_KEY`

2. **Worker LLM array** (already exists in `.env.example`):
   - `CGR_WORKER_LLMS` (JSON array or comma-separated format of model configurations)
   - Parsed by the existing `AppConfig.active_worker_llms` property

3. **Context window overrides** (already exist in `.env.example` — reused for catalog display):
   - Provider/model-specific: `{PROVIDER}_{NORMALIZED_MODEL_ID}_CONTEXT_WINDOW` 
     (e.g., `OPENAI_QWEN3_MAX_CONTEXT_WINDOW`, already supported by `ModelProvider.get_model_context_window()`)
   - Global default: `DEFAULT_CONTEXT_WINDOW` (already in `.env.example`)

> **Important**: `ORCHESTRATOR_CONTEXT_WINDOW` and `CYPHER_CONTEXT_WINDOW` are NOT used 
> for catalog display context windows. They are role-specific overrides used only for 
> context compression thresholds (see Context Window Resolution section).

The `_extract_configured_models_from_env()` function (referenced in the discovery algorithm) 
will be implemented in `models_dynamic.py` and will:

1. Read `settings.active_orchestrator_config` — **only include if explicitly configured by the user** 
   (i.e., `ORCHESTRATOR_PROVIDER` and `ORCHESTRATOR_MODEL` are both set in `.env`, not using the 
   Ollama/llama3.2 default fallback from `_get_default_config()`). Detect this by checking 
   `settings.ORCHESTRATOR_PROVIDER` and `settings.ORCHESTRATOR_MODEL` (both must be non-empty strings; 
   empty strings indicate the default fallback is active). Convert to `DynamicModelInfo` 
   with `source="env_orchestrator"` and `is_configured=True`
2. Read `settings.active_cypher_config` — same logic: only include if `settings.CYPHER_PROVIDER` 
   and `settings.CYPHER_MODEL` are both set. Convert to `DynamicModelInfo` 
   with `source="env_cypher"` and `is_configured=True`
3. Read `settings.active_worker_llms` and convert each to a `DynamicModelInfo` 
   with `source="env_worker"` and `is_configured=True` (worker LLMs are always explicitly 
   configured since they default to an empty list if `CGR_WORKER_LLMS` is not set)
4. For each model, resolve its `context_window` by calling the appropriate 
   `ModelProvider.get_model_context_window(model_id)` method (via `get_provider_from_config()`)
5. Skip models with providers not in `PROVIDER_REGISTRY` (e.g., `cohere`, `local`, `vllm`) 
   with a warning log
6. Deduplicate: if the same provider+model_id appears in multiple roles, keep the 
   entry with the richest configuration (preferring entries that have explicit endpoints)

> **Critical**: Default fallback models (Ollama/llama3.2 when no ORCHESTRATOR_*/CYPHER_* vars 
> are set) must NOT be included as dynamic entries. They already exist in the static catalog's 
> `ollama` provider section and would create duplicates. Only user-explicitly-configured models 
> should appear as `source="env_*"`.

#### Model Discovery Algorithm

> **Design Note**: The static `MODEL_CATALOG` stores `ModelInfo` (NamedTuple) entries, 
> but the dynamic catalog must use `DynamicModelInfo` (frozen dataclass) everywhere. 
> Therefore, the algorithm must **convert all static entries first** before merging 
> dynamic entries — `deepcopy(MODEL_CATALOG)` alone would produce mixed-type lists 
> that cannot be processed uniformly.

```python
def build_dynamic_model_catalog() -> dict[str, list[DynamicModelInfo]]:
    """
    Build unified model catalog combining static and dynamic (.env) models.
    
    All entries are converted to DynamicModelInfo for uniform processing.
    Context windows are resolved dynamically via get_model_context_window()
    at build time (env var overrides and MODEL_CONTEXT_WINDOWS dict are applied).
    
    Priority order:
    1. .env configured models (validated and working) — marked is_configured=True
    2. Static catalog models (for discovery) — marked is_configured based on API key availability
    
    Returns:
        Dictionary mapping providers to lists of DynamicModelInfo objects
    """
    # Convert all static catalog entries to DynamicModelInfo first,
    # resolving context windows dynamically via get_model_context_window()
    catalog: dict[str, list[DynamicModelInfo]] = {}
    for provider, models in MODEL_CATALOG.items():
        provider_instance = get_provider(provider)
        catalog[provider] = [
            DynamicModelInfo.from_model_info(
                m,
                is_configured=_check_api_key_available(m),
                context_window=provider_instance.get_model_context_window(m.model_id),
            )
            for m in models
        ]
    
    # Add/override with .env configured models
    configured_models = _extract_configured_models_from_env()
    
    for model_info in configured_models:
        provider = model_info.provider
        if provider not in catalog:
            catalog[provider] = []
        
        # Check if this exact model already exists in catalog
        existing_idx = _find_model_in_list(catalog[provider], model_info.model_id)
        if existing_idx >= 0:
            # Update existing entry with dynamic info (preserving is_configured=True)
            catalog[provider][existing_idx] = model_info
        else:
            # Add new model to provider list
            catalog[provider].append(model_info)
    
    return catalog


def _check_api_key_available(model_info: ModelInfo) -> bool:
    """Check whether an API key is available for a static catalog model.
    
    Reuses the existing API_KEY_INFO dict from config.py (which already maps
    providers to their env var names and URLs) instead of creating a new mapping.
    
    Uses the same logic as ModelConfig.validate_api_key() and LOCAL_PROVIDERS:
    - Local providers (ollama, local, vllm) never require API keys → True
    - Google Vertex (provider_type=vertex) uses project_id → check project_id instead
    - All other providers: check their environment variable
    
    Returns True if the model can be used without additional user configuration.
    """
    from .config import API_KEY_INFO, LOCAL_PROVIDERS
    
    provider = model_info.provider.lower()
    
    # Local providers never need API keys
    if provider in LOCAL_PROVIDERS:
        return True
    
    # Non-local providers without API key requirement in catalog are always unconfigured
    if not model_info.requires_api_key:
        return True
    
    # Reuse existing API_KEY_INFO mapping (already maps providers → env var names)
    info = API_KEY_INFO.get(provider)
    if info and os.environ.get(info["env_var"]):
        return True
    
    return False
```

#### Context Window Resolution

Context window resolution must reuse the existing `ModelProvider.get_model_context_window()` 
method in `providers/base.py`, which already implements a proven precedence chain:

**Existing resolution (in `ModelProvider.get_model_context_window()`) — reuse for catalog display:**
1. **Provider/model-specific env var**: `{PROVIDER}_{NORMALIZED_MODEL_ID}_CONTEXT_WINDOW` (e.g., `OPENAI_QWEN3_MAX_CONTEXT_WINDOW`)
2. **Provider's `MODEL_CONTEXT_WINDOWS` dict**: Wildcard prefix matching (e.g., `gemini-2.5-pro*` → 1,048,576)
3. **Global default**: `DEFAULT_CONTEXT_WINDOW` (256,000 tokens)

**Separate resolution (in `main.py` compression hooks) — NOT used for catalog display:**
- `ORCHESTRATOR_CONTEXT_WINDOW` and `CYPHER_CONTEXT_WINDOW` are role-specific overrides 
  used **only** for context compression trigger thresholds in `_run_agent_response_loop()` 
  and the `/compress` command. They should NOT appear in the model catalog's context window 
  display because context window is an inherent property of the model, not a role-specific 
  runtime parameter.

Example resolution for `qwen3-max` under the `openai` provider:
- Check `OPENAI_QWEN3_MAX_CONTEXT_WINDOW` env var → not set
- Check `OpenAIProvider.MODEL_CONTEXT_WINDOWS` → not present (only gpt-* models listed)
- Use `DEFAULT_CONTEXT_WINDOW` → 256,000

> **Note**: `AppConfig` currently has two overlapping fields: `CONTEXT_WINDOW_DEFAULT: int = 256000` 
> (a plain attribute) and `DEFAULT_CONTEXT_WINDOW: int = Field(default=256000, gt=0)` (a validated 
> pydantic field). The implementation must use `DEFAULT_CONTEXT_WINDOW` (the validated field) 
> everywhere and should deprecate or remove `CONTEXT_WINDOW_DEFAULT` to eliminate the duplication 
> and potential for inconsistency.

#### Security Considerations

- **API Keys**: Never stored in model catalog. Only validation status (`requires_api_key=True/False`) is tracked
- **Endpoints**: Custom endpoints are stored but sanitized (no authentication tokens)
- **Validation**: Models are only added to catalog if they pass basic configuration validation
- **Error Handling**: Failed validations log warnings but don't crash the application

### User Interface Changes

#### Enhanced `/models` Command

> **Existing Code**: `_handle_models_command()` and `_display_models_table()` already 
> exist in `main.py` (lines ~931 and ~956). These functions must be **modified** to 
> use the dynamic catalog instead of creating new ones. The current `_handle_models_command` 
> imports `MODEL_CATALOG` directly from `models_catalog.py`; it must be changed to call 
> `build_dynamic_model_catalog()` from `models_dynamic.py`. The current 
> `_display_models_table` accepts `dict[str, list[ModelInfo]]`; its signature must be 
> changed to accept `dict[str, list[DynamicModelInfo]]`.

The `/models` command output will be enhanced to show:

1. **Configured Models First**: Models from `.env` appear at the top of each provider section
2. **Visual Indicators**: 
   - ✅ for configured/working models (`is_configured=True`)
   - ⚠️ for static models that may need configuration (`is_configured=False`, `requires_api_key=True`)
   - ✓ (checkmark) for current active model (in addition to ✅/⚠️)
3. **Context Window Display**: Show actual resolved context window via `ModelProvider.get_model_context_window()` (not just static value)
4. **Source Attribution**: Indicate if model comes from `.env` vs static catalog via `source` field

> **Display Name Accuracy**: `PROVIDER_DISPLAY_NAMES` in `models_catalog.py` maps 
> provider keys to human-readable names: `openai` → "OpenAI", `google` → "Google AI", 
> etc. The example output below uses these exact names. For dynamic models with 
> providers not in `PROVIDER_DISPLAY_NAMES` (e.g., custom OpenAI-compatible endpoints), 
> the display name falls back to `provider.title()`.

Example output:
```
  OpenAI
  ✅ qwen3-max (Context: 256K tokens) - Configured from .env ✓ [Active]
  ✅ glm-5.1 (Context: 256K tokens) - Configured from .env  
  ⚠️ gpt-4o (Context: 128K tokens) - Requires API key configuration
  ⚠️ gpt-4o-mini (Context: 128K tokens) - Requires API key configuration

Current Model: openai:qwen3-max
Usage: /model <provider>:<model_id> to switch
```

#### Model Switching Enhancement

The `/model` command already supports switching via `_create_model_from_string()` in `main.py`. 
For dynamic catalog models, the existing `_create_model_from_string` must be enhanced to:

- Look up `DynamicModelInfo.endpoint` from the catalog and pass it to `ModelConfig` for 
  OpenAI-compatible providers with custom endpoints (e.g., Aliyun/Dashscope)
- Look up `DynamicModelInfo.is_configured` to warn users if attempting to switch to an 
  unconfigured model (missing API key)
- Maintain the existing inheritance logic: same provider uses base config's API key; 
  Ollama uses default endpoint; other providers create fresh `ModelConfig`

### Configuration File Updates

#### `.env` Variable Support

The dynamic catalog leverages **existing** `.env` patterns that are already defined in `.env.example`. 
No new environment variables are required — the change is that these existing variables are now 
surfaced in the `/models` command output:

```env
# Model-specific context windows (normalized model ID format — ALREADY in .env.example)
OPENAI_QWEN3_MAX_CONTEXT_WINDOW=32768
OPENAI_GLM_5_1_CONTEXT_WINDOW=32768  
GOOGLE_GEMINI_2_5_PRO_CONTEXT_WINDOW=1048576

# Role-specific context windows (ALREADY in .env.example — used for compression, NOT catalog display)
ORCHESTRATOR_CONTEXT_WINDOW=32768
CYPHER_CONTEXT_WINDOW=16384

# Global default fallback (ALREADY in .env.example)
DEFAULT_CONTEXT_WINDOW=256000

# Role-based model configurations (ALREADY in .env.example — now surfaced in /models)
ORCHESTRATOR_PROVIDER=openai
ORCHESTRATOR_MODEL=qwen3-max
ORCHESTRATOR_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
ORCHESTRATOR_API_KEY=sk-your-key
```

The only `.env.example` change needed is adding a comment explaining that role-configured 
models now appear in `/models` output automatically.

#### Backward Compatibility

- Existing `.env` configurations continue to work unchanged
- Static `MODEL_CATALOG` remains as fallback/discovery mechanism
- No breaking changes to existing CLI flags or APIs

### Error Handling and Validation

#### Model Configuration Validation

Before adding a model to the dynamic catalog, validate:

1. **Provider Support**: Provider must be in `PROVIDER_REGISTRY` (`openai`, `google`, `anthropic`, `ollama`, `azure`) for full validation. Note: `cohere`, `local`, and `vllm` exist in the `Provider` enum but are NOT in `PROVIDER_REGISTRY` — they lack `ModelProvider` implementations and cannot be used at runtime. Models with unrecognized providers should be skipped with a warning.
2. **Model ID Format**: Non-empty string with valid characters
3. **Endpoint Format**: If provided, must be valid URL
4. **Context Window**: Must be positive integer
5. **API Key Availability**: For non-local providers, check if API key is set. For Google Vertex (`provider_type=vertex`), check `GOOGLE_PROJECT_ID` or `ORCHESTRATOR_PROJECT_ID`/`CYPHER_PROJECT_ID` instead of `GOOGLE_API_KEY`

#### Graceful Degradation

- Invalid `.env` model configurations are skipped with warning logs
- Static catalog models are always available for discovery
- Application continues to function even if dynamic catalog building fails

### Performance Considerations

#### Caching Strategy

- **Build Once**: Dynamic catalog is built once at application startup
- **No Runtime Overhead**: Catalog building doesn't impact query performance
- **Memory Efficient**: Only stores metadata, not API keys or large configurations

#### Lazy Loading

- Model validation occurs only when catalog is built
- Provider-specific validation happens when model is actually used
- No unnecessary API calls during catalog building

## Implementation Roadmap

### Phase 1: Core Infrastructure (High Priority)
1. Create `models_dynamic.py` module with `DynamicModelInfo` and catalog builder
2. Enhance `config.py` to extract and validate model configurations from `.env`
3. Integrate existing context window resolution logic (`ModelProvider.get_model_context_window()`) into catalog builder — the resolution already exists in `providers/base.py`, just needs to be called during catalog construction
4. Update `/models` command to use dynamic catalog (modify existing `_handle_models_command()` and `_display_models_table()` in `main.py`)

### Phase 2: User Experience Enhancements (Medium Priority)  
1. Add visual indicators and source attribution to `/models` output
2. Enhance `/model` command to support dynamic model switching
3. Add comprehensive error messages for configuration issues
4. Implement logging for model discovery and validation

### Phase 3: Advanced Features (Low Priority)
1. Implement `CGR_MODEL_CATALOG_PATH` support for external model catalog files — this field already exists as a placeholder in `AppConfig` (`CGR_MODEL_CATALOG_PATH: str | None = None`) but has no runtime logic; Phase 3 adds the loading and validation implementation
2. Implement `CGR_DISABLE_MODEL_DISCOVERY` — this field already exists as a placeholder in `AppConfig` (`CGR_DISABLE_MODEL_DISCOVERY: bool = False`) but has no runtime effect; Phase 3 wires it up to skip dynamic model discovery, keeping only the static `MODEL_CATALOG`
3. Add model health checking (test connectivity at startup)
4. Implement automatic model discovery from Ollama/local providers
5. Add model usage analytics and recommendations

## Testing Strategy

### Unit Tests
- Test `.env` parsing for various configuration formats
- Validate context window resolution precedence
- Test dynamic catalog building with mixed static/dynamic models
- Verify security (no API key leakage in catalog)

### Integration Tests  
- End-to-end testing of `/models` command with real `.env` configurations
- Model switching functionality with dynamic models
- Error handling for invalid configurations
- Backward compatibility with existing setups

### Manual Testing Scenarios
1. **Basic Configuration**: Single orchestrator model in `.env`
2. **Complex Setup**: Multiple worker models with custom endpoints
3. **Mixed Providers**: OpenAI-compatible + Ollama models
4. **Context Windows**: Various context window override scenarios
5. **Edge Cases**: Invalid model IDs, missing API keys, malformed JSON

## Migration and Backward Compatibility

### Zero-Downtime Migration
- Existing users experience no breaking changes
- New features are opt-in via `.env` configuration
- Static catalog continues to work as before

### Documentation Updates
- Add a comment in `.env.example` explaining that role-configured models (ORCHESTRATOR_*, CYPHER_*) now automatically appear in `/models` output
- Add examples for configuring multiple models (already supported via `.env` but not documented for `/models` visibility)
- Document the dynamic catalog behavior in README
- Provide troubleshooting guide for common configuration issues

## Security Review

### Data Flow Analysis
- **Input**: `.env` variables (trusted configuration)
- **Processing**: Validation and normalization (no external input)
- **Output**: Model metadata (no secrets, no user data)
- **Storage**: In-memory only (no persistent storage of configurations)

### Threat Modeling
- **Threat**: Malicious `.env` configuration → **Mitigation**: Input validation and sanitization
- **Threat**: API key exposure → **Mitigation**: Never store API keys in catalog
- **Threat**: SSRF via custom endpoints → **Mitigation**: URL validation and allowlist providers

## Success Metrics

### Technical Metrics
- 100% of `.env` configured models appear in `/models` output
- Zero API key exposure in model catalog
- < 100ms catalog building time at startup
- 100% backward compatibility with existing configurations

### User Experience Metrics  
- Users can discover and switch to their configured models via chat
- Reduced confusion about non-working static models
- Clear indication of which models are properly configured
- Improved success rate for model switching operations

---

This specification provides a comprehensive, secure, and user-friendly solution to integrate `.env` configured models into the built-in model catalog while maintaining all existing functionality and security guarantees.