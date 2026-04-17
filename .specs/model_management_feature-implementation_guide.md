# Implementation Guide: Model Management Feature

> **Alignment note**: This guide must be read alongside the Feature Specification
> (`model_management_feature.md`). The Feature Spec defines the authoritative data model
> (`ModelInfo` NamedTuple, static `MODEL_CATALOG`, `PROVIDER_DISPLAY_NAMES`) and command
> semantics (invalid provider → explicit error, not silent fallback). All code in this
> guide follows the Feature Spec's design decisions.

## Immediate Implementation Steps

### 1. Add `/models` Command Handler

**File**: `codebase_rag/main.py`

Add new function aligned with Feature Spec §4.2:

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
        # valid-provider list, not PROVIDER_DISPLAY_NAMES.keys(). This avoids
        # listing a provider that has display name but no actual model entries.
        valid_providers = ", ".join(MODEL_CATALOG.keys())
        app_context.console.print(
            cs.UI_MODELS_INVALID_PROVIDER.format(provider=arg, available=valid_providers)
        )
```

### 2. Model Catalog Data Source

**File**: `codebase_rag/models_catalog.py` (new file)

> **Critical design decision** (per Feature Spec §4.1): The `MODEL_CATALOG` is a **static,
> hand-curated** data structure containing real, valid model IDs with full metadata. It must
> NOT be derived from `MODEL_CONTEXT_WINDOWS` wildcard patterns, because:
> - Wildcard patterns like `"gemini-2.5-pro*"` are not real model IDs
> - Stripping `*` from `"llama3*"` yields `"llama3"` which is not a valid Ollama pull name
> - `MODEL_CONTEXT_WINDOWS` has no descriptions, display names, or pricing tiers
>
> The full `MODEL_CATALOG` definition is in Feature Spec §4.1. This section shows the
> file structure and import pattern only — copy the complete catalog from the Feature Spec.

```python
# codebase_rag/models_catalog.py
"""Static model catalog with known, valid model IDs and full metadata.

This is the authoritative data source for the /models command display.
Keys MUST match PROVIDER_REGISTRY keys (cs.Provider enum values).
"""

from . import constants as cs
from .types_defs import ModelInfo  # ModelInfo is defined in types_defs.py

# --- Full MODEL_CATALOG definition ---
# Copy the complete MODEL_CATALOG dict from Feature Spec §4.1 here.
# It contains ModelInfo entries for all 5 registered providers:
# google, openai, anthropic, azure, ollama.
#
# DO NOT derive this from MODEL_CONTEXT_WINDOWS — that mapping uses
# wildcard patterns (e.g., "gemini-2.5-pro*") that are not valid model IDs.

MODEL_CATALOG: dict[str, list[ModelInfo]] = {
    # ... (see Feature Spec §4.1 for complete definition) ...
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

> **Why a separate file?** `constants.py` is already 1000+ lines. Keeping the large catalog
> data structure in its own module follows the project's pattern of separating data from
> UI constants (similar to how `language_spec.py` separates language specs from constants).

> **Why not `_build_model_catalog()`?** The original guide proposed deriving the catalog
> dynamically from `PROVIDER_REGISTRY` + `MODEL_CONTEXT_WINDOWS`. This approach is
> fundamentally flawed because `MODEL_CONTEXT_WINDOWS` keys are wildcard patterns, not
> real model IDs. Stripping `*` from patterns like `"llama3*"` produces `"llama3"` which
> is not a valid Ollama model name. The static catalog provides correct, curated model IDs
> with descriptions, display names, and pricing tiers that `MODEL_CONTEXT_WINDOWS` lacks.

### 3. Display Function

**File**: `codebase_rag/main.py`

> **Key fixes from original guide**:
> - Uses `ModelInfo` NamedTuple (not raw dicts) — consistent with Feature Spec §1.4
> - Uses `PROVIDER_DISPLAY_NAMES` from `models_catalog.py` (includes Azure) — not hardcoded local dict
> - Uses `==` for current-model check (not `in` which does substring matching: `"gpt-4" in "gpt-4o"` → True)
> - Uses Rich `Text` objects for safe markup (not raw `[...]` strings which Rich interprets as style tags)
> - Shows `description` field (was missing in original guide)
> - Formats context window as human-readable (e.g., "1M tokens" not "1,048,576 tokens")

```python
def _display_models_table(
    catalog: dict[str, list[ModelInfo]],
) -> None:
    """Display formatted model table using Rich Text for safe markup.

    Args:
        catalog: Dict mapping provider name to list of ModelInfo entries.
                 Must come from MODEL_CATALOG or a filtered subset.
    """
    from .models_catalog import PROVIDER_DISPLAY_NAMES

    if not catalog:
        app_context.console.print("No models available.")
        return

    # Current model info
    current_config = settings.active_orchestrator_config
    current_provider = current_config.provider
    current_model_id = current_config.model_id

    for provider, models in catalog.items():
        display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider.title())
        # Use Rich Text to avoid [brackets] being interpreted as markup tags
        app_context.console.print(Text(f"  {display_name}", style="bold cyan"))

        for model_info in models:
            # Use == for exact match (not 'in' which does substring matching)
            is_current = (
                provider == current_provider
                and model_info.model_id == current_model_id
            )
            marker = "✓" if is_current else "•"

            # Human-readable context window formatting
            ctx = model_info.context_window
            if ctx >= 1_000_000:
                ctx_str = f"{ctx // 1_000_000}M"
            elif ctx >= 1_000:
                ctx_str = f"{ctx // 1_000}K"
            else:
                ctx_str = str(ctx)

            line = Text(f"  {marker} ")
            line.append(model_info.model_id, style="bold")
            line.append(f" (Context: {ctx_str} tokens) - {model_info.description}")
            app_context.console.print(line)

        app_context.console.print("")  # Blank line between providers

    # Current model summary
    # Note: We intentionally use `style()` (Rich markup strings) for fixed text here
    # because the content is static and has no brackets that Rich would misinterpret.
    # Dynamic content (model_id, descriptions) above uses `Text` objects instead,
    # which bypass Rich's markup parser and prevent accidental style tag interpretation
    # (e.g., model descriptions containing "[...]" patterns).
    current_model_str = f"{current_provider}{cs.CHAR_COLON}{current_model_id}"
    app_context.console.print(
        style(f"Current Model: {current_model_str}", cs.Color.CYAN)
    )
    app_context.console.print(
        style("Usage: /model <provider>:<model_id> to switch", cs.Color.YELLOW, cs.StyleModifier.NONE)
    )
```

### 4. Update Command Router

**File**: `codebase_rag/main.py`

In `_run_interactive_loop()`, add new command handler (no changes needed from original):

```python
# In _run_interactive_loop(), add after existing command checks:
if command_parts[0] == cs.MODELS_COMMAND_PREFIX:
    _handle_models_command(stripped_question)
    initial_question = None
    continue
```

### 5. Update Constants

**File**: `codebase_rag/constants.py`

> **Fixes from original guide**:
> - Added `UI_MODELS_INVALID_PROVIDER` for error messages (was missing)
> - Added `/compress` to `UI_HELP_COMMANDS` (was missing — it's an existing command)

Add:
```python
MODELS_COMMAND_PREFIX = "/models"

UI_MODELS_USAGE = """[bold yellow]Usage: /models [provider]
  /models          - Show all available models
  /models google   - Show Google models only
  /models azure    - Show Azure OpenAI models only
  /models ollama   - Show Ollama (local) models only
  /models help     - Show this help[/bold yellow]"""

UI_MODELS_INVALID_PROVIDER = (
    "[bold red]Provider '{provider}' not found. "
    "Available providers: {available}[/bold red]"
)

# Update UI_HELP_COMMANDS to include /models and /compress
UI_HELP_COMMANDS = """[bold cyan]Available commands:[/bold cyan]
  /models             - View available models by provider
  /model <provider:model> - Switch to a different model
  /model                  - Show current model
  /mode <mode>            - Switch query mode (code_only, document_only, both_merged, etc.)
  /mode                   - Show current mode
  /compress               - Manually compress conversation context
  /help                   - Show this help
  exit, quit              - Exit the session"""
```

### 6. Interactive Command Discovery

> **Critical fixes from original guide**:
> - Do NOT replace `get_multiline_input()` with `PromptSession` — all callers in
>   `_run_interactive_loop()` use `await asyncio.to_thread(get_multiline_input, input_prompt)`.
>   Replacing with `PromptSession` would change state management and break call sites.
> - Do NOT remove CTRL+C handling — the existing `keyboard_interrupt` binding is essential
>   for graceful cancellation during input.
> - Do NOT change ENTER behavior — ENTER must always insert a newline (current behavior).
>   The original guide's broken `new_line` handler conditionally blocked newlines when `/`
>   was present, which would prevent multiline input for commands.
> - Completion triggers on **TAB** (prompt_toolkit default), not ENTER.

**File**: `codebase_rag/main.py`

Add `completer` parameter to the **existing** `get_multiline_input()` function.
All existing key bindings and behavior are preserved unchanged — only the `completer`
parameter is added:

```python
def get_multiline_input(prompt_text: str = cs.PROMPT_ASK_QUESTION) -> str:
    """Get multiline input with command completion via TAB key.

    Completion is triggered by TAB (prompt_toolkit default behavior).
    ENTER always inserts a newline. CTRL+J submits. CTRL+C interrupts.
    All existing behavior is preserved — only the completer is added.
    """
    # WordCompleter is imported at module level alongside other prompt_toolkit
    # imports (prompt, KeyBindings, HTML, print_formatted_text) for consistency
    # with the existing import pattern in main.py. No local import needed.

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
    # Note: Only top-level commands are listed here. Sub-options like
    # /compress --aggressive or /mode code_only are NOT included in
    # Phase 1 — a NestedCompleter or custom completer for sub-commands
    # can be added in Phase 2 when the overlay panel UX is implemented.
    command_completer = WordCompleter(
        [
            cs.MODELS_COMMAND_PREFIX,    # /models
            cs.MODEL_COMMAND_PREFIX,     # /model
            cs.MODE_COMMAND_PREFIX,      # /mode
            cs.HELP_COMMAND,             # /help
            cs.COMPRESS_COMMAND_PREFIX,  # /compress
        ],
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
    stripped: str = result.strip()
    return stripped
```

## Testing Strategy

### Unit Tests to Add

1. **Model Catalog Structure**: Test `MODEL_CATALOG` contains entries for all 5 registered providers
   (google, openai, anthropic, azure, ollama) with valid `ModelInfo` entries
2. **Command Parsing**: Test `_handle_models_command()` with various inputs
3. **Display Formatting**: Verify output includes descriptions, uses `==` for current-model check
4. **Integration**: Test end-to-end command execution in chat loop

### Test Cases

```python
# Test cases for /models command
- "/models"           → shows all providers (5 registered providers)
- "/models google"    → shows only Google models
- "/models azure"     → shows only Azure OpenAI models
- "/models help"      → shows usage help (cs.UI_MODELS_USAGE)
- "/models invalid"   → shows explicit error with available provider list
                        (NOT silent fallback to showing all models)
- "/models cohere"    → shows error since cohere is in Provider enum
                        but not in PROVIDER_REGISTRY/MODEL_CATALOG
- Current model should be marked with ✓ using == (exact match, not 'in' substring)
- "/models" output includes model descriptions (not just ID and context window)
- Provider display names include "Azure OpenAI" (not missing)
```

## Backward Compatibility

- All existing `/model` functionality preserved — `_handle_model_command()` unchanged
- No changes to CLI interface or configuration file format
- Existing automation scripts continue to work unchanged
- `get_multiline_input()` is extended (not replaced) — all existing callers still work
- Error handling patterns remain consistent (using `cs.HELP_ARG`, `cs.UI_*` constants)

## Performance Considerations

- Model catalog is a static Python dict — loaded once at module import, no runtime cost
- No external API calls for Phase 1 (static catalog only)
- Memory efficient — `ModelInfo` is a NamedTuple (lightweight, immutable)
- Future: Dynamic Ollama discovery (Phase 3) adds per-invocation query with 5-minute cache

## Security

- No user input executed as code
- Model names validated against known catalog entries
- Provider names restricted to `PROVIDER_REGISTRY` keys only (5 registered providers)
- `CGR_DEFAULT_MODEL_PROVIDER` removed — avoids configuration ambiguity with
  existing `ORCHESTRATOR_PROVIDER`/`CYPHER_PROVIDER` settings
- No sensitive information exposed in error messages

## New Files to Create

| File | Purpose |
|------|---------|
| `codebase_rag/models_catalog.py` | Static `MODEL_CATALOG`, `PROVIDER_DISPLAY_NAMES`, `ModelInfo` definition |
| `codebase_rag/types_defs.py` (modify) | Add `ModelInfo` NamedTuple to existing types file |

## Existing Files to Modify

| File | Changes |
|------|---------|
| `codebase_rag/main.py` | Add `ModelInfo` to existing `from .types_defs import (...)` block (line ~49-60), add `_handle_models_command()`, `_display_models_table()`, update `get_multiline_input()` with completer, add `from prompt_toolkit.completion import WordCompleter` to module-level imports, add command routing in `_run_interactive_loop()` |
| `codebase_rag/constants.py` | Add `MODELS_COMMAND_PREFIX`, `UI_MODELS_USAGE`, `UI_MODELS_INVALID_PROVIDER`, update `UI_HELP_COMMANDS` |

## Implementation Checklist

- [ ] Add `ModelInfo` NamedTuple to `types_defs.py`
- [ ] Create `models_catalog.py` with full `MODEL_CATALOG` and `PROVIDER_DISPLAY_NAMES`
- [ ] Add `_handle_models_command()` to `main.py`
- [ ] Add `_display_models_table()` to `main.py`
- [ ] Add `MODELS_COMMAND_PREFIX`, `UI_MODELS_USAGE`, `UI_MODELS_INVALID_PROVIDER` to `constants.py`
- [ ] Update `UI_HELP_COMMANDS` in `constants.py` (add `/models` and `/compress`)
- [ ] Add `MODELS_COMMAND_PREFIX` command routing in `_run_interactive_loop()`
- [ ] Add `completer` parameter to `get_multiline_input()` in `main.py`
- [ ] ~~Add `CGR_MODEL_CATALOG_PATH` field to `AppConfig`~~ — **Deferred to Phase 3**: JSON/YAML schema for custom catalogs is not yet defined. Add only when `_load_custom_catalog()` is implemented.
- [ ] Add `CGR_DISABLE_MODEL_DISCOVERY: bool = False` field to `AppConfig` in `config.py` — **Phase 3 placeholder** (no-op in Phase 1; dynamic discovery not yet implemented)
- [ ] Write unit tests for catalog, command parsing, and display formatting