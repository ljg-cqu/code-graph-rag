# Model Management and Command Discovery Feature Specification

## Overview
This specification defines the implementation of enhanced model management capabilities and interactive command discovery within the in-chat terminal interface. The feature enables users to view available models, switch between models, and discover supported commands through an intuitive interactive interface.

## Current State Analysis
- Existing `/model` command allows switching to specific provider:model combinations
- Existing `/help` command displays static list of available commands
- No mechanism exists for discovering available models or providers
- No interactive command suggestions when typing `/`

## Feature Requirements

### 1. `/models` Command (Primary Feature)
**Purpose**: Display comprehensive information about available models and allow switching.

#### 1.1 Command Syntax
- `/models` - Show all available models grouped by provider
- `/models <provider>` - Show models for specific provider  
- `/models help` - Show usage information

#### 1.2 Display Format
```
Available Models:

[Google AI]
• gemini-2.5-pro (Context: 1M tokens) - Advanced reasoning
• gemini-2.5-flash (Context: 1M tokens) - Fast responses
• gemini-1.5-pro (Context: 2M tokens) - Balanced performance
• gemini-1.5-flash (Context: 1M tokens) - Cost-effective

[OpenAI]
• gpt-4o (Context: 128K tokens) - Multimodal, fast
• gpt-4o-mini (Context: 128K tokens) - Lightweight, economical
• gpt-4-turbo (Context: 128K tokens) - High intelligence
• gpt-4 (Context: 8K tokens) - Legacy model, limited context
• gpt-3.5-turbo (Context: 128K tokens) - Budget-friendly, legacy

[Anthropic]
• claude-3-5-sonnet (Context: 200K tokens) - Best overall
• claude-3-opus (Context: 200K tokens) - Maximum intelligence
• claude-3-sonnet (Context: 200K tokens) - Balanced legacy model
• claude-3-haiku (Context: 200K tokens) - Fastest responses
• claude-2.1 (Context: 200K tokens) - Legacy, long context
• claude-2.0 (Context: 100K tokens) - Legacy, shorter context

[Azure OpenAI]
• gpt-4o (Context: 128K tokens) - Multimodal, fast
• gpt-4o-mini (Context: 128K tokens) - Lightweight, economical
• gpt-4-turbo (Context: 128K tokens) - High intelligence

[Ollama (Local)]
• llama3.2 (Context: 8K tokens) - Default local model
• llama3.1 (Context: 128K tokens) - Extended context
• mistral-nemo (Context: 128K tokens) - High performance
• gemma2 (Context: 128K tokens) - Google's lightweight model
• qwen2 (Context: 128K tokens) - Alibaba's multilingual model
• phi3 (Context: 128K tokens) - Microsoft's compact model

Current Model: google:gemini-2.5-flash
Usage: /model <provider>:<model_id> to switch
```

> **Note**: Azure OpenAI shares the same model context windows as regular OpenAI
> (inherited via `OpenAIProvider.MODEL_CONTEXT_WINDOWS`). Cohere, Local, and vLLM
> are listed in the `Provider` enum but have no provider implementations in
> `PROVIDER_REGISTRY`. They should be hidden from the catalog display (or shown as
> "available but not yet configured" if desired).

#### 1.3 Model Information Sources
- **Predefined Model Catalog**: Static mapping of known models with metadata
- **Provider API Integration**: Dynamic fetching from provider APIs (where supported)
- **Local Ollama Models**: Query local Ollama instance for available models
- **Context Window Data**: Leverage existing `MODEL_CONTEXT_WINDOWS` mappings

#### 1.4 Model Metadata Structure
```python
from typing import Literal, NamedTuple

class ModelInfo(NamedTuple):
    """Metadata for a single model entry in the catalog.

    Uses NamedTuple to match the project's existing pattern (e.g., AgentLoopUI,
    LanguageMetadata, PyInstallerPackage all use NamedTuple or dataclass).
    """
    provider: str           # e.g. "google", "openai", "anthropic", "azure", "ollama"
    model_id: str           # e.g. "gemini-2.5-pro" — a real, valid model ID usable with /model
    display_name: str        # e.g. "Gemini 2.5 Pro" — human-readable name for display
    context_window: int      # Context window in tokens (e.g. 1048576)
    description: str         # e.g. "Advanced reasoning" — short description for display
    requires_api_key: bool   # True for cloud providers, False for Ollama
    is_local: bool          # True for Ollama/vLLM, False for cloud providers
    pricing_tier: Literal["free", "low", "medium", "high"]  # Constrained to prevent inconsistent values
```

> **Design decision**: `pricing_tier` uses `Literal["free", "low", "medium", "high"]`
> rather than bare `str` to prevent inconsistent values across catalog entries
> (e.g., "cheap" vs "low"). This follows the project's pattern of constraining
> string choices via `StrEnum` or `Literal` types (see `Color`, `StyleModifier`,
> `GoogleProviderType` in `constants.py`). Note: `pricing_tier` is not displayed
> in Phase 1's `_display_models_table()` — it's included for future filtering
> and sorting capabilities (Phase 3+).

> **Important**: `model_id` must be a **real, valid model ID** that can be passed directly to
> `/model <provider>:<model_id>` — NOT a wildcard pattern from `MODEL_CONTEXT_WINDOWS`.
> The static `MODEL_CATALOG` (§4.1) is the authoritative source for displayable model IDs.
> `MODEL_CONTEXT_WINDOWS` is only used for context window size lookup (§1.3).

### 2. Interactive Command Discovery
**Purpose**: Provide real-time command suggestions when users type `/`.

#### 2.1 Trigger Mechanism (Phase 1: TAB Completion)
- Completion is triggered by **TAB** key (prompt_toolkit default behavior)
- When user types `/` and presses TAB, a dropdown list of matching commands appears
- ENTER always inserts a newline (existing behavior, unchanged)
- CTRL+J submits the input (existing behavior, unchanged)
- CTRL+C interrupts (existing behavior, unchanged)

> **Phase 2/3 Enhancement**: An overlay panel with arrow-key (↑/↓) navigation and
> Enter-to-select semantics may be added later using `prompt_toolkit`'s
> `PromptSession` with custom `NestedCompleter`. This requires careful refactoring
> of `get_multiline_input()` and is **not** part of Phase 1 because it risks
> breaking the existing multiline input flow (ENTER must remain newline, not submit).

#### 2.2 Command Categories
```
Model Management:
  /models        - Browse available models
  /model         - Switch to specific model (e.g., /model google:gemini-2.5-pro)
  /model <p:m>   - Switch to provider:model

Query Modes:
  /mode code_only      - Code graph only (default)
  /mode document_only  - Document graph only  
  /mode both_merged    - Both graphs, merged results
  /mode code_vs_doc    - Validate code against docs
  /mode doc_vs_code    - Validate docs against code

Session Management:
  /help          - Show available commands
  /compress      - Manually compress context (also /compress --aggressive)
  exit, quit     - Exit chat session
```

#### 2.3 Implementation Approach
- Extend existing `get_multiline_input()` function
- Integrate with `prompt_toolkit` completion system
- Use `prompt_toolkit.completion.WordCompleter` for command suggestions
- Implement custom completer for dynamic model lists

### 3. Enhanced `/model` Command Integration
**Purpose**: Maintain backward compatibility while enhancing functionality.

#### 3.1 Backward Compatibility
- Existing `/model <provider>:<model>` syntax remains unchanged
- Existing `/model` (no args) shows current model
- All error handling and validation preserved

#### 3.2 Enhanced Validation
- Validate model exists in catalog before attempting switch
- Provide helpful error messages for unknown models
- Suggest similar models when typos detected
- Check API key availability before model switch

### 4. Technical Implementation Details

#### 4.1 Model Catalog Structure

> **Design decision**: The static `MODEL_CATALOG` is the **primary and authoritative** data source
> for the `/models` display. It contains only **real, valid model IDs** that users can pass to
> `/model <provider>:<model_id>`. It must NOT be derived from `MODEL_CONTEXT_WINDOWS` wildcard
> patterns (see §1.4 note). `MODEL_CONTEXT_WINDOWS` is used only for context window size lookup
> when a model is not found in the catalog, or for resolving wildcard patterns during actual
> model switching (existing `get_model_context_window()` method on providers).

```python
# In new file: codebase_rag/models_catalog.py
# (Keeping large data structures out of constants.py, which is already 1000+ lines)

from .types_defs import ModelInfo  # or define inline if preferred

# Static catalog with known, valid model IDs and full metadata.
# Keys MUST match PROVIDER_REGISTRY keys (cs.Provider enum values).
MODEL_CATALOG: dict[str, list[ModelInfo]] = {
    cs.Provider.GOOGLE: [
        ModelInfo(provider="google", model_id="gemini-2.5-pro",
                   display_name="Gemini 2.5 Pro", context_window=1048576,
                   description="Advanced reasoning", requires_api_key=True,
                   is_local=False, pricing_tier="medium"),
        ModelInfo(provider="google", model_id="gemini-2.5-flash",
                   display_name="Gemini 2.5 Flash", context_window=1048576,
                   description="Fast responses", requires_api_key=True,
                   is_local=False, pricing_tier="low"),
        ModelInfo(provider="google", model_id="gemini-1.5-pro",
                   display_name="Gemini 1.5 Pro", context_window=2097152,
                   description="Balanced performance", requires_api_key=True,
                   is_local=False, pricing_tier="medium"),
        ModelInfo(provider="google", model_id="gemini-1.5-flash",
                   display_name="Gemini 1.5 Flash", context_window=1048576,
                   description="Cost-effective", requires_api_key=True,
                   is_local=False, pricing_tier="low"),
    ],
    cs.Provider.OPENAI: [
        ModelInfo(provider="openai", model_id="gpt-4o",
                   display_name="GPT-4o", context_window=128000,
                   description="Multimodal, fast", requires_api_key=True,
                   is_local=False, pricing_tier="medium"),
        ModelInfo(provider="openai", model_id="gpt-4o-mini",
                   display_name="GPT-4o Mini", context_window=128000,
                   description="Lightweight, economical", requires_api_key=True,
                   is_local=False, pricing_tier="low"),
        ModelInfo(provider="openai", model_id="gpt-4-turbo",
                   display_name="GPT-4 Turbo", context_window=128000,
                   description="High intelligence", requires_api_key=True,
                   is_local=False, pricing_tier="high"),
        ModelInfo(provider="openai", model_id="gpt-4",
                   display_name="GPT-4", context_window=8192,
                   description="Legacy model, limited context", requires_api_key=True,
                   is_local=False, pricing_tier="high"),
        ModelInfo(provider="openai", model_id="gpt-3.5-turbo",
                   display_name="GPT-3.5 Turbo", context_window=128000,
                   description="Budget-friendly, legacy", requires_api_key=True,
                   is_local=False, pricing_tier="low"),
    ],
    cs.Provider.ANTHROPIC: [
        ModelInfo(provider="anthropic", model_id="claude-3-5-sonnet",
                   display_name="Claude 3.5 Sonnet", context_window=200000,
                   description="Best overall", requires_api_key=True,
                   is_local=False, pricing_tier="medium"),
        ModelInfo(provider="anthropic", model_id="claude-3-opus",
                   display_name="Claude 3 Opus", context_window=200000,
                   description="Maximum intelligence", requires_api_key=True,
                   is_local=False, pricing_tier="high"),
        ModelInfo(provider="anthropic", model_id="claude-3-sonnet",
                   display_name="Claude 3 Sonnet", context_window=200000,
                   description="Balanced legacy model", requires_api_key=True,
                   is_local=False, pricing_tier="medium"),
        ModelInfo(provider="anthropic", model_id="claude-3-haiku",
                   display_name="Claude 3 Haiku", context_window=200000,
                   description="Fastest responses", requires_api_key=True,
                   is_local=False, pricing_tier="low"),
        ModelInfo(provider="anthropic", model_id="claude-2.1",
                   display_name="Claude 2.1", context_window=200000,
                   description="Legacy, long context", requires_api_key=True,
                   is_local=False, pricing_tier="medium"),
        ModelInfo(provider="anthropic", model_id="claude-2.0",
                   display_name="Claude 2.0", context_window=100000,
                   description="Legacy, shorter context", requires_api_key=True,
                   is_local=False, pricing_tier="low"),
    ],
    cs.Provider.AZURE: [
        # Azure OpenAI uses the same models as OpenAI but accessed via Azure endpoints.
        # AzureOpenAIProvider inherits MODEL_CONTEXT_WINDOWS from OpenAIProvider.
        ModelInfo(provider="azure", model_id="gpt-4o",
                   display_name="GPT-4o (Azure)", context_window=128000,
                   description="Multimodal, fast", requires_api_key=True,
                   is_local=False, pricing_tier="medium"),
        ModelInfo(provider="azure", model_id="gpt-4o-mini",
                   display_name="GPT-4o Mini (Azure)", context_window=128000,
                   description="Lightweight, economical", requires_api_key=True,
                   is_local=False, pricing_tier="low"),
        ModelInfo(provider="azure", model_id="gpt-4-turbo",
                   display_name="GPT-4 Turbo (Azure)", context_window=128000,
                   description="High intelligence", requires_api_key=True,
                   is_local=False, pricing_tier="high"),
    ],
    cs.Provider.OLLAMA: [
        # Ollama model IDs must match actual pull names (e.g., "llama3.2", not "llama3").
        # These are the known defaults; the actual available models are determined
        # by querying the running Ollama instance (Phase 3: Dynamic Discovery).
        ModelInfo(provider="ollama", model_id="llama3.2",
                   display_name="Llama 3.2", context_window=8192,
                   description="Default local model", requires_api_key=False,
                   is_local=True, pricing_tier="free"),
        ModelInfo(provider="ollama", model_id="llama3.1",
                   display_name="Llama 3.1", context_window=128000,
                   description="Extended context", requires_api_key=False,
                   is_local=True, pricing_tier="free"),
        ModelInfo(provider="ollama", model_id="mistral-nemo",
                   display_name="Mistral Nemo", context_window=128000,
                   description="High performance", requires_api_key=False,
                   is_local=True, pricing_tier="free"),
        ModelInfo(provider="ollama", model_id="gemma2",
                   display_name="Gemma 2", context_window=128000,
                   description="Google's lightweight model", requires_api_key=False,
                   is_local=True, pricing_tier="free"),
        ModelInfo(provider="ollama", model_id="qwen2",
                   display_name="Qwen 2", context_window=128000,
                   description="Alibaba's multilingual model", requires_api_key=False,
                   is_local=True, pricing_tier="free"),
        ModelInfo(provider="ollama", model_id="phi3",
                   display_name="Phi 3", context_window=128000,
                   description="Microsoft's compact model", requires_api_key=False,
                   is_local=True, pricing_tier="free"),
    ],
}

# Display names for providers (used in section headers)
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    cs.Provider.GOOGLE: "Google AI",
    cs.Provider.OPENAI: "OpenAI",
    cs.Provider.ANTHROPIC: "Anthropic",
    cs.Provider.AZURE: "Azure OpenAI",
    cs.Provider.OLLAMA: "Ollama (Local)",
}
```

> **Note on unimplemented providers**: The `Provider` enum also includes `cohere`, `local`,
> and `vllm`, but these have no entries in `PROVIDER_REGISTRY` and should be excluded from
> `MODEL_CATALOG`. The catalog builder should iterate over `PROVIDER_REGISTRY` keys only
> (not the full `Provider` enum) to avoid showing models for providers that can't actually
> be used.

#### 4.2 New Functions in main.py
```python
def _handle_models_command(command: str) -> None:
    """Handle /models command to display available models.

    Follows the same pattern as existing _handle_model_command():
    - No arg → show all providers
    - "help" → show usage (cs.HELP_ARG == "help")
    - Valid provider → show that provider only
    - Invalid provider → explicit error with available provider list
    """
    from .models_catalog import MODEL_CATALOG
    # Note: PROVIDER_DISPLAY_NAMES is not imported here because it's only
    # needed by _display_models_table(). The error message uses MODEL_CATALOG.keys()
    # (the authoritative data source) rather than PROVIDER_DISPLAY_NAMES.keys().

    parts = command.strip().split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else None

    if arg == cs.HELP_ARG:
        app_context.console.print(cs.UI_MODELS_USAGE)
        return

    if arg is None:
        # Show all providers
        _display_models_table(MODEL_CATALOG)
        return

    # Check if arg is a valid provider key
    if arg in MODEL_CATALOG:
        provider_models = {arg: MODEL_CATALOG[arg]}
        _display_models_table(provider_models)
    else:
        # Invalid provider — explicit error, NOT silent fallback
        # Use MODEL_CATALOG.keys() (the authoritative data source) for the
        # valid-provider list, not PROVIDER_DISPLAY_NAMES.keys().
        valid_providers = ", ".join(MODEL_CATALOG.keys())
        app_context.console.print(
            cs.UI_MODELS_INVALID_PROVIDER.format(provider=arg, available=valid_providers)
        )

def _display_models_table(models_by_provider: dict[str, list[ModelInfo]]) -> None:
    """Display formatted table of available models using Rich Text for safe markup."""
    # Implementation details... (see Implementation Guide §3 for full code)
```

#### 4.3 Enhanced Input Handling

> **Critical**: Do NOT rename or replace `get_multiline_input()`. All callers in `_run_interactive_loop()`
> use `await asyncio.to_thread(get_multiline_input, input_prompt)` — renaming would break these call sites.
> Instead, add command completion to the **existing** function by passing a `completer` parameter to
> `prompt()`. Completion triggers on **TAB** (prompt_toolkit default), not on ENTER — ENTER must always
> insert a newline (current behavior). All existing key bindings (CTRL_J submit, ENTER newline,
> CTRL_C interrupt) must be preserved unchanged.

```python
def get_multiline_input(prompt_text: str = cs.PROMPT_ASK_QUESTION) -> str:
    """Get multiline input with command completion via TAB key.

    Completion is triggered by TAB (prompt_toolkit default behavior).
    ENTER always inserts a newline. CTRL+J submits. CTRL+C interrupts.
    All existing behavior is preserved — only TAB completion is added.
    """
    # WordCompleter is imported at module level alongside other prompt_toolkit imports
    # (prompt, KeyBindings, HTML, print_formatted_text) for consistency with existing code.
    bindings = KeyBindings()

    @bindings.add(cs.KeyBinding.CTRL_J)
    def submit(event: KeyPressEvent) -> None:
        event.app.exit(result=event.app.current_buffer.text)

    @bindings.add(cs.KeyBinding.ENTER)
    def new_line(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add(cs.KeyBinding.CTRL_C)
    def keyboard_interrupt(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    # Command completion — TAB triggers suggestions
    command_completer = WordCompleter(
        [cs.MODELS_COMMAND_PREFIX, cs.MODEL_COMMAND_PREFIX,
         cs.MODE_COMMAND_PREFIX, cs.HELP_COMMAND, cs.COMPRESS_COMMAND_PREFIX],
        ignore_case=True,
    )

    clean_prompt = Text.from_markup(prompt_text).plain

    print_formatted_text(
        HTML(cs.UI_INPUT_PROMPT_HTML.format(prompt=clean_prompt, hint=cs.MULTILINE_INPUT_HINT))
    )

    result = prompt(
        "",
        multiline=True,
        key_bindings=bindings,
        completer=command_completer,   # NEW: TAB-triggered completion
        wrap_lines=True,
        style=ORANGE_STYLE,
    )
    if result is None:
        raise EOFError
    return result.strip()
```

#### 4.4 Integration with Existing Command System
- Add `/models` to `UI_HELP_COMMANDS` constant
- Update command routing in `_run_interactive_loop()`
- Maintain consistent error handling patterns

### 5. User Experience Flow

#### 5.1 Basic Usage (Phase 1: TAB Completion)
1. User types `/` in chat input
2. User presses TAB — prompt_toolkit displays a dropdown of matching commands
3. User selects a command from the dropdown (or continues typing to filter)
4. Command text is inserted into the input line
5. User can modify parameters and press Ctrl+J to execute

> **Phase 2/3 Enhancement**: An overlay panel with arrow-key navigation and Enter-to-select
> semantics may replace or supplement TAB completion in later phases (see §2.1).

#### 5.2 Model Browsing Flow
1. User types `/models`
2. System displays categorized model list with metadata
3. User sees current model highlighted
4. User can type `/model <desired_model>` based on displayed options
5. System validates and switches model with confirmation

#### 5.3 Error Handling
- Invalid provider: "Provider '{provider}' not found. Available providers: google, openai, anthropic, azure, ollama" — uses `MODEL_CATALOG.keys()` for the valid-provider list (see §4.2 implementation)
- Invalid model: "Model 'gpt-5' not available for OpenAI. Available: gpt-4o, gpt-4o-mini, gpt-4-turbo"
- Missing API key: Clear instructions on setting up required API keys (using existing `format_missing_api_key_errors` from `config.py`)
- Local model not running: "Ollama not running. Start with 'ollama serve'"

> **Note on unimplemented providers**: When a user types `/models cohere`, the `/models`
> command checks `MODEL_CATALOG` keys (which only contains registered providers). Since
> `cohere` is in the `Provider` enum but not in `PROVIDER_REGISTRY`/`MODEL_CATALOG`, it
> triggers the **same generic** `UI_MODELS_INVALID_PROVIDER` error as any invalid string
> like `/models foobar`. A specialized "defined but not yet implemented" message would
> require checking `Provider` enum membership separately, which adds complexity without
> much user benefit. The generic error already lists the available providers, making it
> clear that `cohere` is not among them. If desired, this can be enhanced in Phase 2.

### 6. Configuration and Extensibility

#### 6.1 Environment Variables

> **Implementation note**: All new environment variables must be added as fields to the
> `AppConfig(BaseSettings)` class in `codebase_rag/config.py` with proper types, defaults,
> and validators, following the existing `SettingsConfigDict(env_file=".env")` pattern.

- `CGR_MODEL_CATALOG_PATH: str | None = None`: Custom model catalog file path (JSON/YAML).
  When set, the catalog is loaded from this file instead of the static `MODEL_CATALOG`.
  Must be validated as a valid file path at startup if provided.

  > **Phase 3 — Not implementation-ready for Phase 1**: The JSON/YAML schema for custom
  > catalogs is not yet defined. A custom catalog file must map to `ModelInfo` NamedTuple
  > fields, but the exact schema (required vs optional fields, validation rules, YAML
  > structure) needs separate specification. This env var should be added to `AppConfig`
  > only when the schema is finalized and a `_load_custom_catalog()` function is implemented.
  > For Phase 1, the static `MODEL_CATALOG` dict in `models_catalog.py` is the sole source.

- `CGR_DISABLE_MODEL_DISCOVERY: bool = False`: Disable dynamic model discovery (Ollama
  querying, provider API integration). Only the static `MODEL_CATALOG` is used.

  > **Phase 3 placeholder**: This setting has no effect in Phase 1 since dynamic discovery
  > is not yet implemented. It can be added to `AppConfig` as a no-op field for forward
  > compatibility, but actual functionality requires Phase 3's Ollama querying logic.

> **Removed**: `CGR_DEFAULT_MODEL_PROVIDER` — this overlaps with existing `ORCHESTRATOR_PROVIDER`
> and `CYPHER_PROVIDER` settings in `AppConfig`. The default provider for model suggestions is
> already determined by `settings.active_orchestrator_config.provider`. Adding a separate
> environment variable would create configuration ambiguity.

#### 6.2 Extensibility Points
- Plugin system for adding new providers
- Custom model metadata via configuration files
- Integration with model benchmarking services

### 7. Testing Strategy

#### 7.1 Unit Tests
- Model catalog loading and filtering
- Command parsing and validation
- Input completion functionality
- Error message formatting

#### 7.2 Integration Tests
- End-to-end command execution flow
- Model switching with different providers
- Interactive completion in terminal environment

#### 7.3 Edge Cases
- Empty model catalogs
- Network failures during dynamic discovery
- Malformed model specifications
- Unicode model names

### 8. Performance Considerations

#### 8.1 Caching Strategy

> **Phase 3 — Not applicable to Phase 1**: In Phase 1, `MODEL_CATALOG` is a static
> Python dict loaded once at module import time. There is zero runtime cost and no
> need for caching. The strategies below apply only when dynamic model discovery
> (Ollama querying, provider API integration) is introduced in Phase 3.

- Cache model catalogs for 1 hour (Phase 3: dynamic provider API responses)
- Cache Ollama model lists for 5 minutes (Phase 3: local instance queries)
- Lazy loading of detailed model metadata (Phase 3: optional detailed info)

#### 8.2 Memory Usage
- Stream model display for large catalogs
- Limit displayed models to top 20 per provider
- Virtual scrolling for long command lists

### 9. Security Considerations

#### 9.1 Input Validation
- Sanitize model names and provider strings
- Prevent command injection in model parameters
- Validate URLs for custom model endpoints

#### 9.2 API Key Handling
- Never log API keys in model switching operations
- Secure storage of provider credentials
- Clear error messages without exposing sensitive data

### 10. Implementation Roadmap

#### Phase 1: Core Model Catalog (Week 1)
- Implement static model catalog
- Add `/models` command handler
- Basic model display formatting

#### Phase 2: Interactive Completion (Week 2)  
- Implement command suggestion system
- Integrate with prompt_toolkit
- Add arrow key navigation support

#### Phase 3: Dynamic Discovery (Week 3)
- Add Ollama model querying
- Implement provider API integration
- Add caching and error handling

#### Phase 4: Polish and Testing (Week 4)
- Comprehensive testing suite
- Performance optimization
- Documentation and examples

### 11. Backward Compatibility
- All existing `/model` functionality preserved
- No breaking changes to CLI interface
- Existing scripts and automation continue to work
- Configuration file format unchanged

### 12. Dependencies
- `prompt_toolkit>=3.0.0` (already in dependencies)
- `httpx` (already in dependencies)  
- No additional external dependencies required

This specification provides a comprehensive, implementation-ready design that enhances the existing model management capabilities while maintaining full backward compatibility and aligning with the current codebase architecture.