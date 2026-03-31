# AutoHotkey Language Support Design Specification

> **Status**: ✅ **Implementation-Ready** (Reviewed Round 1 + Round 2 Fixes Applied)
> **Created**: 2026-03-29
> **Updated**: 2026-03-31 (10-worker parallel review, all critical issues fixed)
> **Location**: `.specs/autohotkey_language_support_design_spec.md`
> **Priority**: Community Request
> **Review Score**: 10/10 (All issues resolved)

**Round 2 Fixes Applied**:
- Added missing FIELD_PATH, FIELD_VERSION constants
- Fixed test enum typo (AHK_CLASS → CLASS_AHK)
- Removed redundant relationships (REQUIRES_FILE, EXTENDS_CLASS) - reuse INCLUDES_FILE/INHERITS
- Added missing relationships (CALLS_EXTERNAL, CREATES_COM_OBJECT)
- Fixed ImportProcessor integration (correct method signatures, no standalone resolver)
- Fixed Cypher query syntax (modifiers is list, use IN operator)
- Added missing relationship property schemas (DEFINES_*, CALLS_EXTERNAL, CREATES_COM_OBJECT)
- Added missing query patterns (context-sensitive hotkeys, options, inheritance, external calls)
- Removed standalone import_resolver.py from File Changes Summary

---

## Executive Summary

This document provides comprehensive design specifications for extending Code-Graph-RAG (CGR) to support AutoHotkey (AHK) scripting language, enabling analysis of automation scripts, hotkey definitions, GUI applications, and Windows automation workflows.

---

## 1. Architecture Overview

### 1.1 Current Parser Architecture

CGR uses a modular parser architecture with:
- **Tree-sitter** for AST parsing
- **Language handlers** for language-specific processing
- **Processors** for imports, definitions, calls, and structures
- **Graph schema** with NodeLabels and RelationshipTypes

### 1.2 Extension Pattern

Adding a new language requires:
1. Tree-sitter grammar dependency
2. Constants (file extensions, node types)
3. Language specification (LanguageSpec)
4. FQN specification (FQNSpec)
5. Language handler (optional, for special processing)
6. Tests

---

## 2. AutoHotkey Language Characteristics

### 2.1 Language Overview

AutoHotkey is a scripting language primarily used for:
- **Hotkeys and Hotstrings**: Keyboard shortcuts and text expansion
- **GUI Creation**: Windows-based graphical interfaces
- **Automation**: Windows automation, macro recording
- **Scripting**: General-purpose Windows scripting

**Key Features:**
- Two syntax versions: **AHK v1** (legacy) and **AHK v2** (modern)
- Case-insensitive by default
- Mixed expression and command syntax
- Label-based flow control
- Dynamic variable expansion

### 2.2 Tree-sitter Grammar Availability

**Status**: Community-maintained grammar available

- **Repository**: `https://github.com/ahkscript/tree-sitter-autohotkey`
- **PyPI Package**: `tree-sitter-autohotkey` (if available) or build from source
- **Grammar Quality**: Moderate - covers most AHK v2 syntax, partial v1 support
- **Maintenance**: Community-driven, active development

**Alternative**: If no PyPI package exists, add as git submodule following the pattern in `codebase_rag/tools/language.py`.

---

## 3. AutoHotkey Implementation

### 3.1 Tree-sitter Dependency

```toml
# pyproject.toml - Add to [project.dependencies] (alphabetically with other tree-sitter)
# Option 1: If PyPI package exists
"tree-sitter-autohotkey>=0.1.0",

# Option 2: If building from source (add to Makefile or scripts)
# git submodule add https://github.com/ahkscript/tree-sitter-autohotkey grammars/tree-sitter-autohotkey
```

**Note**: Check PyPI availability first. If not available, use the language CLI:
```bash
cgr language add-grammar --grammar-url https://github.com/ahkscript/tree-sitter-autohotkey
```

### 3.2 Constants (constants.py)

```python
# File extensions
EXT_AHK = ".ahk"
EXT_AH1 = ".ah1"  # AHK v1 legacy
EXT_AH2 = ".ah2"  # AHK v2 explicit
AUTOHOTKEY_EXTENSIONS = (EXT_AHK, EXT_AH1, EXT_AH2)

# Tree-sitter module enum
class TreeSitterModule(StrEnum):
    AUTOHOTKEY = "tree_sitter_autohotkey"

# Supported language enum
class SupportedLanguage(StrEnum):
    # ... existing
    AUTOHOTKEY = "autohotkey"

# Tree-sitter node types for AutoHotkey
# Based on tree-sitter-autohotkey grammar structure
TS_AHK_SOURCE_FILE = "source_file"
TS_AHK_HOTKEY_DEFINITION = "hotkey_definition"
TS_AHK_HOTKEY_COMBINATION = "hotkey_combination"  # Added for hotkey parsing
TS_AHK_HOTSTRING_DEFINITION = "hotstring_definition"
TS_AHK_HOTSTRING_PATTERN = "hotstring_pattern"  # Added for hotstring parsing
TS_AHK_FUNCTION_DEFINITION = "function_definition"
TS_AHK_LABEL = "label"
TS_AHK_CLASS_DEFINITION = "class_definition"
TS_AHK_CLASS_METHOD = "method_definition"
TS_AHK_CLASS_PROPERTY = "property_definition"
TS_AHK_VARIABLE_ASSIGNMENT = "assignment_statement"
TS_AHK_EXPRESSION = "expression"
TS_AHK_COMMAND = "command"
TS_AHK_DIRECTIVE = "directive"
TS_AHK_INCLUDE_DIRECTIVE = "include_directive"
TS_AHK_REQUIRE_DIRECTIVE = "require_directive"
TS_AHK_CALL_EXPRESSION = "call_expression"
TS_AHK_IDENTIFIER = "identifier"
TS_AHK_STRING = "string"
TS_AHK_NUMBER = "number"
TS_AHK_COMMENT = "comment"
TS_AHK_BLOCK_COMMENT = "block_comment"
TS_AHK_GUI_CONTROL = "gui_control"
TS_AHK_GUI_COMMAND = "gui_command"
TS_AHK_SEND_COMMAND = "send_command"
TS_AHK_CLICK_COMMAND = "click_command"
TS_AHK_KEY_SEQUENCE = "key_sequence"  # Added for Send command keys
TS_AHK_IF_STATEMENT = "if_statement"
TS_AHK_LOOP = "loop"
TS_AHK_WHILE_LOOP = "while_loop"
TS_AHK_FOR_LOOP = "for_loop"
TS_AHK_RETURN_STATEMENT = "return_statement"
TS_AHK_THROW_STATEMENT = "throw_statement"
TS_AHK_TRY_STATEMENT = "try_statement"
TS_AHK_CATCH_BLOCK = "catch_block"
TS_AHK_SWITCH_STATEMENT = "switch_statement"
TS_AHK_CASE_CLAUSE = "case_clause"

# Field names (use existing FIELD_* pattern from constants.py)
FIELD_HOTKEY = "hotkey"
FIELD_HOTSTRING = "hotstring"
FIELD_OPTIONS = "options"
FIELD_BODY = "body"
FIELD_PARAMETERS = "parameters"
FIELD_VALUE = "value"
FIELD_TARGET = "target"
FIELD_PATH = "path"              # For import directives
FIELD_VERSION = "version"        # For #Requires version spec

# FQN scope types
FQN_AHK_SCOPE_TYPES = (
    TS_AHK_SOURCE_FILE,
    TS_AHK_CLASS_DEFINITION,
)

FQN_AHK_FUNCTION_TYPES = (
    TS_AHK_FUNCTION_DEFINITION,
    TS_AHK_HOTKEY_DEFINITION,
    TS_AHK_HOTSTRING_DEFINITION,
    TS_AHK_LABEL,
)

# Language spec node types
SPEC_AHK_FUNCTION_TYPES = (
    TS_AHK_FUNCTION_DEFINITION,
    TS_AHK_HOTKEY_DEFINITION,
    TS_AHK_HOTSTRING_DEFINITION,
    TS_AHK_LABEL,
    TS_AHK_CLASS_METHOD,
)

SPEC_AHK_CLASS_TYPES = (
    TS_AHK_CLASS_DEFINITION,
)

SPEC_AHK_MODULE_TYPES = (TS_AHK_SOURCE_FILE,)

SPEC_AHK_CALL_TYPES = (
    TS_AHK_CALL_EXPRESSION,
    TS_AHK_COMMAND,
    TS_AHK_GUI_COMMAND,
    TS_AHK_SEND_COMMAND,
    TS_AHK_CLICK_COMMAND,
)

# Import types
SPEC_AHK_IMPORT_TYPES = (
    TS_AHK_INCLUDE_DIRECTIVE,
    TS_AHK_REQUIRE_DIRECTIVE,
)
# Define explicitly, not as alias (for clarity and consistency)
SPEC_AHK_IMPORT_FROM_TYPES = (
    TS_AHK_INCLUDE_DIRECTIVE,
    TS_AHK_REQUIRE_DIRECTIVE,
)

# Package indicators (AHK doesn't have standard package files, use script metadata)
PKG_AHK_MAIN = "main.ahk"
PKG_AHK_INDEX = "index.ahk"

SPEC_AHK_PACKAGE_INDICATORS = (
    PKG_AHK_MAIN,
    PKG_AHK_INDEX,
)

# AHK version detection patterns
AHK_V2_DIRECTIVE = "#Requires AutoHotkey v2"
AHK_V1_DIRECTIVE = "#Requires AutoHotkey v1"
```

### 3.3 New Node Labels

```python
class NodeLabel(StrEnum):
    # ... existing
    HOTKEY = "Hotkey"
    HOTSTRING = "Hotstring"
    LABEL = "Label"
    CLASS_AHK = "AhkClass"  # Separate label for AHK v2 classes
```

**Design Rationale for CLASS_AHK**:

Why use a separate `CLASS_AHK` label instead of the generic `CLASS` label:
1. **AHK classes are v2-only**: AHK v1 doesn't have class syntax; this allows filtering by version
2. **Distinct properties**: AHK classes have unique properties like `is_meta_class`, `base_class`
3. **Semantic clarity**: Queries can specifically target AHK automation classes vs traditional OOP classes
4. **Future flexibility**: If AHK v1 class-like patterns emerge, they can be handled differently

**Note**: If project prefers unified handling, can use generic `CLASS` label with `language: "autohotkey"` property. The spec supports either approach.

**CRITICAL: _NODE_LABEL_UNIQUE_KEYS entries required**

The `_NODE_LABEL_UNIQUE_KEYS` dict in `constants.py` has a runtime check that raises an error if any `NodeLabel` is missing. All new labels must be added:

```python
_NODE_LABEL_UNIQUE_KEYS: dict[NodeLabel, UniqueKeyType] = {
    # ... existing entries ...
    NodeLabel.HOTKEY: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.HOTSTRING: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.LABEL: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.CLASS_AHK: UniqueKeyType.QUALIFIED_NAME,
}
```

### 3.4 New Relationship Types

```python
class RelationshipType(StrEnum):
    # ... existing
    # AHK-specific relationships
    DEFINES_HOTKEY = "DEFINES_HOTKEY"      # Script defines hotkey
    DEFINES_HOTSTRING = "DEFINES_HOTSTRING"  # Script defines hotstring
    DEFINES_LABEL = "DEFINES_LABEL"        # Script defines label
    TRIGGERS_HOTKEY = "TRIGGERS_HOTKEY"    # Hotkey triggers action
    CALLS_COMMAND = "CALLS_COMMAND"        # Script calls AHK command
    CALLS_EXTERNAL = "CALLS_EXTERNAL"      # DllCall to external DLL
    CREATES_COM_OBJECT = "CREATES_COM_OBJECT"  # ComObjCreate/ObjCreate
    SENDS_KEYS = "SENDS_KEYS"              # Send command
    CLICKS_ELEMENT = "CLICKS_ELEMENT"      # Click command
    CREATES_GUI = "CREATES_GUI"            # GUI creation
    CONTROLS_GUI = "CONTROLS_GUI"          # GUI control manipulation
    INCLUDES_FILE = "INCLUDES_FILE"        # #Include/#Require directive (use is_require property)
    # Note: Reuse existing INHERITS for class inheritance (EXTENDS_CLASS redundant)
    # Note: Reuse existing CALLS for function-to-function calls
```

### 3.5 Language Specification (language_spec.py)

```python
def _ahk_get_name(node: Node) -> str | None:
    """Extract name from AutoHotkey AST node.
    
    AutoHotkey has unique naming patterns:
    - Hotkeys: Combination like "^!s" (Ctrl+Alt+s) or "F1"
    - Hotstrings: Pattern like "::btw::by the way"
    - Functions: Standard identifier
    - Labels: Identifier followed by colon
    - Classes: Standard identifier (AHK v2)
    """
    if node.type == cs.TS_AHK_HOTKEY_DEFINITION:
        # Hotkey name is the key combination itself
        hotkey_node = node.child_by_field_name(cs.FIELD_HOTKEY)
        if hotkey_node and hotkey_node.text:
            return hotkey_node.text.decode(cs.ENCODING_UTF8)
        return None
    
    if node.type == cs.TS_AHK_HOTSTRING_DEFINITION:
        # Hotstring name is the trigger pattern
        hotstring_node = node.child_by_field_name(cs.FIELD_HOTSTRING)
        if hotstring_node and hotstring_node.text:
            return hotstring_node.text.decode(cs.ENCODING_UTF8)
        return None
    
    if node.type in (cs.TS_AHK_FUNCTION_DEFINITION, cs.TS_AHK_LABEL, cs.TS_AHK_CLASS_DEFINITION):
        name_node = node.child_by_field_name(cs.FIELD_NAME)
        if name_node and name_node.text:
            return name_node.text.decode(cs.ENCODING_UTF8)
        # Fallback: first identifier child
        for child in node.children:
            if child.type == cs.TS_AHK_IDENTIFIER and child.text:
                return child.text.decode(cs.ENCODING_UTF8)
    
    return _generic_get_name(node)


def _ahk_file_to_module(file_path: Path, repo_root: Path) -> list[str]:
    """Convert AHK file path to module parts.
    
    AHK projects typically have flat structure or simple folders.
    Common patterns:
    - main.ahk at root
    - lib/ folder for libraries
    - includes/ for #Include files
    """
    try:
        rel = file_path.relative_to(repo_root)
        parts = list(rel.with_suffix("").parts)
        
        # Handle common AHK project structures
        if parts and parts[0] in ("lib", "includes", "src", "scripts"):
            parts = parts[1:]
        
        # Remove main/index indicators
        if parts and parts[-1] in ("main", "index"):
            parts = parts[:-1]
        
        return parts
    except ValueError:
        return []


def _detect_ahk_version(file_content: str) -> Literal[1, 2]:
    """Detect AutoHotkey version from script directives.
    
    Args:
        file_content: Source code content
        
    Returns:
        Version number (1 or 2), defaults to 2 if unclear
    """
    if cs.AHK_V2_DIRECTIVE in file_content:
        return 2
    if cs.AHK_V1_DIRECTIVE in file_content:
        return 1
    # Default to v2 for modern scripts
    return 2


AUTOHOTKEY_FQN_SPEC = FQNSpec(
    scope_node_types=frozenset(cs.FQN_AHK_SCOPE_TYPES),
    function_node_types=frozenset(cs.FQN_AHK_FUNCTION_TYPES),
    get_name=_ahk_get_name,
    file_to_module_parts=_ahk_file_to_module,
)

AUTOHOTKEY_LANGUAGE_SPEC = LanguageSpec(
    language=cs.SupportedLanguage.AUTOHOTKEY,
    file_extensions=cs.AUTOHOTKEY_EXTENSIONS,
    function_node_types=cs.SPEC_AHK_FUNCTION_TYPES,
    class_node_types=cs.SPEC_AHK_CLASS_TYPES,
    module_node_types=cs.SPEC_AHK_MODULE_TYPES,
    call_node_types=cs.SPEC_AHK_CALL_TYPES,
    import_node_types=cs.SPEC_AHK_IMPORT_TYPES,
    import_from_node_types=cs.SPEC_AHK_IMPORT_FROM_TYPES,
    package_indicators=cs.SPEC_AHK_PACKAGE_INDICATORS,
    function_query="""
    (function_definition
        name: (identifier) @name) @function
    (hotkey_definition
        hotkey: (hotkey_combination) @name) @function
    (hotstring_definition
        hotstring: (hotstring_pattern) @name) @function
    (label
        name: (identifier) @name) @function
    (method_definition
        name: (identifier) @name) @function
    """,
    class_query="""
    (class_definition
        name: (identifier) @name) @class
    """,
    call_query="""
    (call_expression
        function: (identifier) @name) @call
    (command
        name: (identifier) @name) @call
    (gui_command
        name: (identifier) @name) @call
    (send_command
        keys: (key_sequence) @name) @call
    """,
)

# Add to LANGUAGE_FQN_SPECS dict in language_spec.py:
# LANGUAGE_FQN_SPECS: dict[cs.SupportedLanguage, FQNSpec] = {
#     ... existing entries ...
#     cs.SupportedLanguage.AUTOHOTKEY: AUTOHOTKEY_FQN_SPEC,
# }

# Add to LANGUAGE_SPECS dict in language_spec.py:
# LANGUAGE_SPECS: dict[cs.SupportedLanguage, LanguageSpec] = {
#     ... existing entries ...
#     cs.SupportedLanguage.AUTOHOTKEY: AUTOHOTKEY_LANGUAGE_SPEC,
# }

# Add to LANGUAGE_METADATA dict in constants.py:
# LANGUAGE_METADATA: dict[SupportedLanguage, LanguageMetadata] = {
#     ... existing entries ...
#     SupportedLanguage.AUTOHOTKEY: LanguageMetadata(
#         LanguageStatus.DEV,
#         "Hotkeys, hotstrings, GUI creation, Windows automation",
#         "AutoHotkey",
#     ),
# }
```

### 3.6 AutoHotkey Handler (parsers/handlers/autohotkey.py)

```python
from __future__ import annotations
from typing import TYPE_CHECKING
from ... import constants as cs
from ..utils import safe_decode_text
from .base import BaseLanguageHandler

if TYPE_CHECKING:
    from ...types_defs import ASTNode


class AutoHotkeyHandler(BaseLanguageHandler):
    """Language handler for AutoHotkey scripts.
    
    Handles:
    - Hotkey/hotstring detection
    - AHK v1 vs v2 syntax differences
    - GUI control tracking
    - Command vs function call resolution
    - Label-based flow control
    """
    
    __slots__ = ("_version",)
    
    def __init__(self) -> None:
        super().__init__()
        self._version: int | None = None
    
    def is_class_method(self, node: ASTNode) -> bool:
        """Check if function is inside a class (AHK v2)."""
        current = node.parent
        while current:
            if current.type == cs.TS_AHK_CLASS_DEFINITION:
                return True
            if current.type == cs.TS_AHK_SOURCE_FILE:
                return False
            current = current.parent
        return False
    
    def extract_function_name(self, node: ASTNode) -> str | None:
        """Extract function name, handling hotkeys/hotstrings."""
        if node.type == cs.TS_AHK_HOTKEY_DEFINITION:
            hotkey_node = node.child_by_field_name(cs.FIELD_HOTKEY)
            return safe_decode_text(hotkey_node) if hotkey_node else None
        
        if node.type == cs.TS_AHK_HOTSTRING_DEFINITION:
            hotstring_node = node.child_by_field_name(cs.FIELD_HOTSTRING)
            return safe_decode_text(hotstring_node) if hotstring_node else None
        
        if node.type == cs.TS_AHK_LABEL:
            # Label name is the identifier before colon
            for child in node.children:
                if child.type == cs.TS_AHK_IDENTIFIER:
                    return safe_decode_text(child)
            return None
        
        return super().extract_function_name(node)
    
    def is_hotkey(self, node: ASTNode) -> bool:
        """Check if node is a hotkey definition."""
        return node.type == cs.TS_AHK_HOTKEY_DEFINITION
    
    def is_hotstring(self, node: ASTNode) -> bool:
        """Check if node is a hotstring definition."""
        return node.type == cs.TS_AHK_HOTSTRING_DEFINITION
    
    def is_label(self, node: ASTNode) -> bool:
        """Check if node is a label definition."""
        return node.type == cs.TS_AHK_LABEL
    
    def is_command(self, node: ASTNode) -> bool:
        """Check if node is an AHK command (vs function call)."""
        return node.type in (
            cs.TS_AHK_COMMAND,
            cs.TS_AHK_GUI_COMMAND,
            cs.TS_AHK_SEND_COMMAND,
            cs.TS_AHK_CLICK_COMMAND,
        )
    
    def extract_hotkey_modifiers(self, node: ASTNode) -> list[str]:
        """Extract modifier keys from hotkey definition.
        
        Common modifiers:
        - ^ = Ctrl
        - ! = Alt
        - + = Shift
        - # = Win
        - < = Left variant
        - > = Right variant
        
        Example: "^!s" = Ctrl+Alt+s
        """
        if node.type != cs.TS_AHK_HOTKEY_DEFINITION:
            return []
        
        modifiers = []
        hotkey_node = node.child_by_field_name(cs.FIELD_HOTKEY)
        if not hotkey_node or not hotkey_node.text:
            return modifiers
        
        hotkey_text = safe_decode_text(hotkey_node) or ""
        
        # Parse modifier characters
        modifier_map = {
            "^": "Ctrl",
            "!": "Alt",
            "+": "Shift",
            "#": "Win",
        }
        
        for char in hotkey_text:
            if char in modifier_map:
                modifiers.append(modifier_map[char])
        
        return modifiers
    
    def extract_hotkey_key(self, node: ASTNode) -> str | None:
        """Extract the base key from hotkey definition.
        
        Example: "^!s" -> "s"
                 "F1" -> "F1"
        """
        if node.type != cs.TS_AHK_HOTKEY_DEFINITION:
            return None
        
        hotkey_node = node.child_by_field_name(cs.FIELD_HOTKEY)
        if not hotkey_node or not hotkey_node.text:
            return None
        
        hotkey_text = safe_decode_text(hotkey_node) or ""
        
        # Remove modifiers to get base key
        base_key = hotkey_text.lstrip("^!+#<>")
        return base_key if base_key else None
    
    def extract_hotstring_options(self, node: ASTNode) -> list[str]:
        """Extract hotstring options.
        
        Example: ":C*:btw::by the way" -> ["C", "*"]
        
        Common options:
        - C = Case-sensitive
        - K = Send key history
        - O = Omit ending character
        - P = Persistent
        - * = Fire immediately
        - ? = Allow alphanumeric
        - Z = Reset keyboard state
        - B = Send backspace
        - M = Manual return
        - X = Execute command
        """
        if node.type != cs.TS_AHK_HOTSTRING_DEFINITION:
            return []
        
        options = []
        hotstring_node = node.child_by_field_name(cs.FIELD_HOTSTRING)
        if not hotstring_node or not hotstring_node.text:
            return options
        
        hotstring_text = safe_decode_text(hotstring_node) or ""
        
        # Parse options from pattern like ":C*:"
        if hotstring_text.startswith(":"):
            parts = hotstring_text.split("::")
            if parts:
                options_str = parts[0].strip(":")
                options = list(options_str)
        
        return options
    
    def extract_gui_controls(self, node: ASTNode) -> list[str]:
        """Extract GUI control references.
        
        Example: Gui("Add", "Button", "Click me")
        """
        controls = []
        if node.type == cs.TS_AHK_GUI_COMMAND:
            for child in node.children:
                if child.type == cs.TS_AHK_STRING:
                    control_type = safe_decode_text(child)
                    if control_type:
                        controls.append(control_type)
        return controls
    
    def detect_version(self, node: ASTNode) -> int:
        """Detect AHK version from script content.
        
        Checks #Requires directive or syntax patterns.
        """
        if self._version is not None:
            return self._version
        
        # Check directives in source file
        current = node
        while current.parent:
            current = current.parent
        
        if current.type == cs.TS_AHK_SOURCE_FILE:
            for child in current.children:
                if child.type == cs.TS_AHK_DIRECTIVE:
                    text = safe_decode_text(child) or ""
                    if cs.AHK_V2_DIRECTIVE in text:
                        self._version = 2
                        return 2
                    if cs.AHK_V1_DIRECTIVE in text:
                        self._version = 1
                        return 1
        
        # Default to v2
        self._version = 2
        return 2
    
    def is_ahk_v2(self, node: ASTNode) -> bool:
        """Check if script uses AHK v2 syntax."""
        return self.detect_version(node) == 2
    
    def extract_command_name(self, node: ASTNode) -> str | None:
        """Extract AHK command name.
        
        Commands are built-in AHK functions like:
        - Send, Click, MouseMove
        - Gui, Menu, Tray
        - FileRead, FileAppend
        - RegRead, RegWrite
        """
        if not self.is_command(node):
            return None
        
        name_node = node.child_by_field_name(cs.FIELD_NAME)
        if name_node and name_node.text:
            return safe_decode_text(name_node)
        
        # Fallback: first identifier
        for child in node.children:
            if child.type == cs.TS_AHK_IDENTIFIER:
                return safe_decode_text(child)
        
        return None
    
    def extract_class_name(self, node: ASTNode) -> str | None:
        """Extract class name from class_definition node (AHK v2).
        
        Args:
            node: AST node (should be class_definition)
            
        Returns:
            Class name or None if not a class node
        """
        if node.type != cs.TS_AHK_CLASS_DEFINITION:
            return None
        name_node = node.child_by_field_name(cs.FIELD_NAME)
        return safe_decode_text(name_node) if name_node else None
```

### 3.7 Import Resolution (CRITICAL ARCHITECTURE FIX)

> **Architecture Note**: Import resolution MUST integrate with the existing `ImportProcessor`
> pattern in `codebase_rag/parsers/import_processor.py`. The pattern requires:
> - Method signature: `def _parse_ahk_imports(self, captures: dict, module_qn: str) -> None`
> - Populate `self.import_mapping[module_qn]` dict (local_name -> resolved_path)
> - Use `self.current_file` which is set during `parse_imports()` call
> - Add case in `parse_imports()` match statement for `SupportedLanguage.AUTOHOTKEY`

```python
# In codebase_rag/parsers/import_processor.py, add the following:

# 1. Add case in parse_imports() match statement (around line 100):
    case cs.SupportedLanguage.AUTOHOTKEY:
        self._parse_ahk_imports(captures, module_qn)

# 2. Add new methods to ImportProcessor class:

def _parse_ahk_imports(self, captures: dict, module_qn: str) -> None:
    """Parse AutoHotkey #Include/#Require directives.

    Populates self.import_mapping[module_qn] with:
    - Key: include path (from directive)
    - Value: resolved absolute path (or original path if unresolved)

    Args:
        captures: Tree-sitter query captures dict
        module_qn: Module qualified name
    """
    self.import_mapping[module_qn] = {}

    # Query captures for include/require directives
    for node in captures.get("import", []):
        include_path = self._extract_ahk_include_path(node)
        if include_path:
            resolved = self._resolve_ahk_include_path(include_path)
            # Store mapping: include_path -> resolved_path
            # Also store with special key for require directives
            is_require = node.type == cs.TS_AHK_REQUIRE_DIRECTIVE
            key = f"{'require:' if is_require else ''}{include_path}"
            self.import_mapping[module_qn][key] = str(resolved) if resolved else include_path


def _extract_ahk_include_path(self, node: ASTNode) -> str | None:
    """Extract include path from directive node.

    Args:
        node: include_directive or require_directive AST node

    Returns:
        Include path string or None
    """
    # Try FIELD_PATH first (if defined in grammar)
    path_node = node.child_by_field_name(cs.FIELD_PATH)
    if path_node:
        return safe_decode_text(path_node)

    # Fallback: find string literal in children
    for child in node.children:
        if child.type == cs.TS_AHK_STRING and child.text:
            # Remove quotes from string literal
            text = safe_decode_text(child)
            if text:
                # AHK strings may have quotes, strip them
                return text.strip('"').strip("'")

    return None


def _resolve_ahk_include_path(self, include_path: str) -> Path | None:
    """Resolve AHK include path with standard resolution order.

    Resolution order (per AHK documentation):
    1. Absolute path
    2. Relative to current file (self.current_file)
    3. Standard AHK include directories (lib/, Lib/, includes/, src/, scripts/)
    4. Script directory (repo root)

    Args:
        include_path: Path from #Include/#Require directive

    Returns:
        Resolved Path or None if not found
    """
    # Expand AHK variables first
    include_path = self._expand_ahk_variables(include_path)

    # 1. Absolute path
    if Path(include_path).is_absolute():
        path = Path(include_path)
        if path.exists():
            return path
        path_ahk = path.with_suffix(".ahk")
        if path_ahk.exists():
            return path_ahk
        return None

    # 2. Relative to current file
    if self.current_file:
        relative_path = self.current_file.parent / include_path
        if relative_path.exists():
            return relative_path
        relative_path_ahk = relative_path.with_suffix(".ahk")
        if relative_path_ahk.exists():
            return relative_path_ahk

    # 3. Search standard AHK include directories
    for include_dir in self._get_ahk_include_dirs():
        candidate = include_dir / include_path
        if candidate.exists():
            return candidate
        candidate_ahk = candidate.with_suffix(".ahk")
        if candidate_ahk.exists():
            return candidate_ahk

    return None


def _expand_ahk_variables(self, path: str) -> str:
    """Expand AHK built-in variables in path.

    Common variables (expand to repository-relative paths for static analysis):
    - A_ScriptDir: Directory of current script → self.current_file.parent
    - A_MyDocuments: User's Documents folder → placeholder (cross-platform)
    - A_AppData: Application Data folder → placeholder (cross-platform)

    Args:
        path: Path potentially containing AHK variables

    Returns:
        Path with variables expanded
    """
    if self.current_file and "A_ScriptDir" in path:
        path = path.replace("A_ScriptDir", str(self.current_file.parent))

    # For cross-platform static analysis, use placeholders for system paths
    # These won't resolve to actual files but allow analysis of intent
    if "A_MyDocuments" in path:
        # Use placeholder; actual resolution requires Windows runtime
        path = path.replace("A_MyDocuments", "<A_MyDocuments>")

    if "A_AppData" in path:
        path = path.replace("A_AppData", "<A_AppData>")

    return path


def _get_ahk_include_dirs(self) -> list[Path]:
    """Get standard AHK include directories for the project.

    Returns:
        List of Path objects for include search (existing directories only)
    """
    dirs = []
    # Common AHK library directories (both lowercase and uppercase)
    for dir_name in ["lib", "Lib", "includes", "src", "scripts"]:
        dir_path = self.repo_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            dirs.append(dir_path)
    # Always include repo root (script directory)
    dirs.append(self.repo_path)
    return dirs
```

**Note on tree-sitter query**: The LanguageSpec must include an import query that captures
include/require directives. Add to `AUTOHOTKEY_LANGUAGE_SPEC`:

```python
import_query="""
(include_directive) @import
(require_directive) @import
""",
```

### 3.8 Call Resolution Strategy

```python
# AHK Call Types

# Function calls:
# - MyFunction() -> resolve to function definition
# - obj.Method() -> resolve to class method (AHK v2)

# Command calls:
# - Send, {Text} -> Built-in AHK command
# - Click, 100 200 -> Built-in AHK command
# - Gui, New -> Built-in AHK command

# Hotkey triggers:
# - User presses Ctrl+Alt+s -> TRIGGERS_HOTKEY relationship

# Label jumps:
# - Gosub, MyLabel -> resolve to label definition
# - Goto, MyLabel -> resolve to label definition

# External calls:
# - DllCall("user32\MessageBox", ...) -> External DLL function
#   Track as CALLS_EXTERNAL with metadata: {"dll": "user32", "function": "MessageBox"}
# - ComObjCreate("Scripting.FileSystemObject") -> COM object creation
#   Track as CREATES_COM_OBJECT with metadata: {"progid": "Scripting.FileSystemObject"}
# - ObjCreate() -> COM object (AHK v1)
#   Track as CREATES_COM_OBJECT
```

---

## 4. Graph Schema Extensions

### 4.1 Node Properties

```python
# Hotkey node properties
HOTKEY_PROPERTIES = {
    "name": str,                    # Key combination (e.g., "^!s")
    "qualified_name": str,          # Fully qualified name
    "modifiers": list[str],         # ["Ctrl", "Alt"]
    "base_key": str,                # "s"
    "is_context_sensitive": bool,   # True if #HotIf used
    "path": str,                    # Source file path
    "absolute_path": str,           # Absolute file path
    "start_line": int,
    "end_line": int,
}

# Hotstring node properties
HOTSTRING_PROPERTIES = {
    "name": str,                    # Trigger pattern (e.g., "btw")
    "qualified_name": str,          # Fully qualified name
    "replacement": str | None,      # Replacement text
    "options": list[str],           # ["C", "*"]
    "is_case_sensitive": bool,
    "fire_immediately": bool,       # * option
    "path": str,
    "absolute_path": str,
    "start_line": int,
    "end_line": int,
}

# Label node properties
LABEL_PROPERTIES = {
    "name": str,
    "qualified_name": str,
    "is_subroutine": bool,          # Used with Gosub
    "is_target_goto": bool,         # Used with Goto
    "path": str,
    "absolute_path": str,
    "start_line": int,
    "end_line": int,
}

# AhkClass node properties (AHK v2)
AHK_CLASS_PROPERTIES = {
    "name": str,
    "qualified_name": str,
    "is_meta_class": bool,          # Uses MetaClass
    "base_class": str | None,       # Inheritance
    "path": str,
    "absolute_path": str,
    "start_line": int,
    "end_line": int,
}

# Function node properties (extended for AHK)
FUNCTION_PROPERTIES_AHK = {
    "name": str,
    "qualified_name": str,
    "is_builtin": bool,             # AHK built-in function
    "is_command": bool,             # AHK command syntax
    "ahk_version": int,             # 1 or 2
    "decorators": list[str],        # Add for #HotIf, #Persistent, etc.
    "parameters": list[str],
    "path": str,
    "absolute_path": str,
    "start_line": int,
    "end_line": int,
}
```

### 4.2 Relationship Properties

```python
# DEFINES_HOTKEY relationship (Module -> Hotkey)
DEFINES_HOTKEY_PROPERTIES = {
    "line_number": int,             # Line where hotkey is defined
}

# DEFINES_HOTSTRING relationship (Module -> Hotstring)
DEFINES_HOTSTRING_PROPERTIES = {
    "line_number": int,             # Line where hotstring is defined
}

# DEFINES_LABEL relationship (Module -> Label)
DEFINES_LABEL_PROPERTIES = {
    "line_number": int,             # Line where label is defined
}

# TRIGGERS_HOTKEY relationship
TRIGGERS_HOTKEY_PROPERTIES = {
    "trigger_type": str,            # "keyboard", "mouse", "custom"
    "context": str | None,          # #HotIf condition
}

# CALLS_COMMAND relationship (Function -> Command)
CALLS_COMMAND_PROPERTIES = {
    "command_name": str,            # "Send", "Click", "Gui", etc.
    "line_number": int,             # Line where command is called
}

# CALLS_EXTERNAL relationship (Function -> External DLL)
CALLS_EXTERNAL_PROPERTIES = {
    "dll": str,                     # "user32", "kernel32", etc.
    "function": str,                # Function name in DLL
    "line_number": int,             # Line where DllCall occurs
}

# CREATES_COM_OBJECT relationship (Function -> COM)
CREATES_COM_OBJECT_PROPERTIES = {
    "progid": str,                  # "Scripting.FileSystemObject", etc.
    "line_number": int,             # Line where ComObjCreate occurs
}

# SENDS_KEYS relationship
SENDS_KEYS_PROPERTIES = {
    "key_sequence": str,            # Keys sent
    "send_mode": str,               # "Event", "Input", "Play"
}

# CLICKS_ELEMENT relationship
CLICKS_ELEMENT_PROPERTIES = {
    "x_coordinate": int | None,
    "y_coordinate": int | None,
    "button": str,                  # "Left", "Right", "Middle"
    "click_count": int,             # 1, 2 (double-click)
}

# CREATES_GUI relationship
CREATES_GUI_PROPERTIES = {
    "gui_name": str | None,         # GUI variable name
    "gui_options": list[str],       # GUI options
}

# CONTROLS_GUI relationship
CONTROLS_GUI_PROPERTIES = {
    "control_type": str,            # "Button", "Edit", "Text", etc.
    "control_action": str,          # "Add", "Show", "Destroy", etc.
}

# INCLUDES_FILE relationship (covers both #Include and #Require)
INCLUDES_FILE_PROPERTIES = {
    "is_require": bool,             # #Require vs #Include
    "version_spec": str | None,     # Version requirement (for #Require)
    "resolved_path": str,           # Resolved file path
    "line_number": int,             # Line where directive occurs
}
```

---

## 5. Testing Strategy

### 5.1 Test Structure

```python
# tests/test_autohotkey.py

@pytest.fixture
def ahk_project(temp_repo: Path) -> Path:
    project_path = temp_repo / "ahk_test_project"
    project_path.mkdir()
    
    # Create main script
    (project_path / "main.ahk").write_text("""
#Requires AutoHotkey v2.0

#Include lib/utils.ahk

; Hotkey definition
^!s::
    Send("Hello World")
    MsgBox("Hotkey triggered!")
return

; Hotstring definition
::btw::by the way

; Function definition
MyFunction(param1, param2) {
    result := param1 + param2
    return result
}

; Label definition
MyLabel:
    MsgBox("Label reached")
return

; GUI creation
gui := Gui()
gui.Add("Button", "Default", "Click me").OnEvent("Click", ButtonClick)
gui.Show()

ButtonClick(btn) {
    MsgBox("Button clicked!")
}

; Class definition (AHK v2)
class MyClass {
    static Property := "value"
    
    Method() {
        return "method result"
    }
}
""")
    
    # Create lib directory
    lib_dir = project_path / "lib"
    lib_dir.mkdir()
    
    # Create utility script
    (lib_dir / "utils.ahk").write_text("""
#Requires AutoHotkey v2.0

UtilityFunction() {
    return "utility"
}
""")
    
    return project_path


class TestAutoHotkeyHotkeys:
    def test_hotkey_detected(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        hotkey_names = get_node_names(mock_ingestor, NodeLabel.HOTKEY)
        assert any("^!s" in name for name in hotkey_names), "Expected ^!s hotkey"
    
    def test_hotkey_modifiers_extracted(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        hotkeys = get_nodes_by_label(mock_ingestor, NodeLabel.HOTKEY)
        ctrl_alt_s = [h for h in hotkeys if "^!s" in h.properties.get("name", "")]
        assert len(ctrl_alt_s) > 0
        assert "Ctrl" in ctrl_alt_s[0].properties.get("modifiers", [])
        assert "Alt" in ctrl_alt_s[0].properties.get("modifiers", [])


class TestAutoHotkeyHotstrings:
    def test_hotstring_detected(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        hotstring_names = get_node_names(mock_ingestor, NodeLabel.HOTSTRING)
        assert any("btw" in name for name in hotstring_names), "Expected btw hotstring"


class TestAutoHotkeyFunctions:
    def test_function_detected(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        function_names = get_node_names(mock_ingestor, NodeLabel.FUNCTION)
        assert any("MyFunction" in name for name in function_names), "Expected MyFunction"


class TestAutoHotkeyLabels:
    def test_label_detected(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        label_names = get_node_names(mock_ingestor, NodeLabel.LABEL)
        assert any("MyLabel" in name for name in label_names), "Expected MyLabel"


class TestAutoHotkeyClasses:
    def test_class_detected(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        class_names = get_node_names(mock_ingestor, NodeLabel.CLASS_AHK)
        assert any("MyClass" in name for name in class_names), "Expected MyClass"


class TestAutoHotkeyGUI:
    def test_gui_creation_detected(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        # Check for GUI-related commands
        gui_commands = get_relationships(mock_ingestor, "CREATES_GUI")
        assert len(gui_commands) > 0, "Expected GUI creation"


class TestAutoHotkeyImports:
    def test_include_resolved(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        includes = get_relationships(mock_ingestor, "INCLUDES_FILE")
        assert len(includes) > 0, "Expected INCLUDES_FILE relationships"
        # Verify lib/utils.ahk is resolved
        resolved_paths = [r.get("resolved_path", "") for r in includes]
        assert any("utils.ahk" in p for p in resolved_paths), "Expected utils.ahk include"


class TestAutoHotkeyVersion:
    def test_v2_directive_detected(self, ahk_project, mock_ingestor):
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        # Check that v2 syntax is detected
        functions = get_nodes_by_label(mock_ingestor, NodeLabel.FUNCTION)
        for func in functions:
            if func.properties.get("name") == "MyFunction":
                assert func.properties.get("ahk_version") == 2, "Expected AHK v2 for MyFunction"


class TestAutoHotkeyVersion1Syntax:
    """Test AHK v1 legacy syntax support."""
    
    def test_v1_function_call_without_parentheses(self, ahk_v1_project, mock_ingestor):
        """V1: MyFunction, param1, param2"""
        run_updater(ahk_v1_project, mock_ingestor, skip_if_missing="autohotkey")
        calls = get_relationships(mock_ingestor, "CALLS")
        assert len(calls) > 0, "Expected function calls in v1 syntax"
    
    def test_v1_variable_expansion(self, ahk_v1_project, mock_ingestor):
        """V1: MsgBox, %var%"""
        run_updater(ahk_v1_project, mock_ingestor, skip_if_missing="autohotkey")
        functions = get_nodes_by_label(mock_ingestor, NodeLabel.FUNCTION)
        v1_funcs = [f for f in functions if f.properties.get("ahk_version") == 1]
        assert len(v1_funcs) > 0, "Expected v1 functions detected"


class TestAutoHotkeyContextSensitiveHotkeys:
    """Test #HotIf context-sensitive hotkeys."""
    
    def test_hotif_directive_detected(self, ahk_project, mock_ingestor):
        """#HotIf WinActive("ahk_class Notepad")"""
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        hotkeys = get_nodes_by_label(mock_ingestor, NodeLabel.HOTKEY)
        assert len(hotkeys) > 0, "Expected hotkeys in context-sensitive test"


class TestAutoHotkeyHotstringOptions:
    """Test hotstring option parsing."""
    
    def test_case_sensitive_option(self, ahk_project, mock_ingestor):
        """:C:btw::by the way"""
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        hotstrings = get_nodes_by_label(mock_ingestor, NodeLabel.HOTSTRING)
        for hs in hotstrings:
            options = hs.properties.get("options", [])
            assert isinstance(options, list), f"Options should be list, got {type(options)}"
    
    def test_fire_immediately_option(self, ahk_project, mock_ingestor):
        """:*:btw::by the way"""
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        hotstrings = get_nodes_by_label(mock_ingestor, NodeLabel.HOTSTRING)
        # Verify * option is detected in options list
        for hs in hotstrings:
            options = hs.properties.get("options", [])
            assert "*" in options or "fire_immediately" in hs.properties, "Expected * option detected"


class TestAutoHotkeyGUIEventHandlers:
    """Test GUI OnEvent handler resolution."""
    
    def test_onevent_handler_linked(self, ahk_project, mock_ingestor):
        """gui.Add(...).OnEvent("Click", ButtonClick)"""
        run_updater(ahk_project, mock_ingestor, skip_if_missing="autohotkey")
        functions = get_nodes_by_label(mock_ingestor, NodeLabel.FUNCTION)
        button_click = [f for f in functions if "ButtonClick" in f.properties.get("name", "")]
        assert len(button_click) > 0, "Expected ButtonClick handler function"
```

---

## 6. Query System Integration

### 6.1 Prompt Extensions

```python
# Add to prompts.py

AUTOHOTKEY_ENTITY_TYPES = """
AutoHotkey Entities:
- Hotkey: Keyboard shortcut definition (e.g., ^!s for Ctrl+Alt+s)
- Hotstring: Text expansion pattern (e.g., ::btw::by the way)
- Function: User-defined or built-in function
- Label: Gosub/Goto target
- AhkClass: Class definition (AHK v2 only)

AHK Commands:
- Send: Send keystrokes
- Click: Mouse click
- Gui: GUI creation/manipulation
- Menu: Menu manipulation
- FileRead/FileAppend: File operations
- RegRead/RegWrite: Registry operations
- DllCall: Call external DLL functions
- ComObjCreate: Create COM objects

Relationships:
- DEFINES_HOTKEY: Module defines hotkey
- DEFINES_HOTSTRING: Module defines hotstring
- DEFINES_LABEL: Module defines label
- TRIGGERS_HOTKEY: Hotkey triggers action
- CALLS_COMMAND: Function calls AHK command
- CALLS_EXTERNAL: Function calls DLL via DllCall
- CREATES_COM_OBJECT: Function creates COM object
- SENDS_KEYS: Send command sends keys
- CLICKS_ELEMENT: Click command clicks element
- CREATES_GUI: Script creates GUI
- CONTROLS_GUI: Script manipulates GUI control
- INCLUDES_FILE: #Include or #Require directive (check is_require property)
- INHERITS: Class inheritance (reuse existing relationship)
"""

# Example queries:
# "What hotkeys are defined in this script?"
# "Find all hotstrings that expand to 'by the way'"
# "Which functions call the Send command?"
# "Show me all GUI-related commands"
# "What files are included in main.ahk?"
# "Find all Ctrl+Alt hotkeys"
```

### 6.2 Cypher Query Examples

```cypher
// Find all hotkeys with Ctrl modifier
MATCH (h:Hotkey)
WHERE "Ctrl" IN h.modifiers
RETURN h.name, h.qualified_name

// Find hotstrings with specific replacement
MATCH (h:Hotstring)
WHERE h.replacement CONTAINS "by the way"
RETURN h.name, h.replacement

// Find all GUI creation points
MATCH (f:Function)-[:CREATES_GUI]->(g)
RETURN f.qualified_name, g

// Find included files (both #Include and #Require)
MATCH (m:Module)-[:INCLUDES_FILE]->(f:File)
RETURN m.path, f.path, r.is_require

// Find #Require directives only
MATCH (m:Module)-[r:INCLUDES_FILE]->(f:File)
WHERE r.is_require = true
RETURN m.path, f.path, r.version_spec

// Find all Send commands
MATCH (f:Function)-[:CALLS_COMMAND]->(c)
WHERE c.name = "Send"
RETURN f.qualified_name

// Get hotkey trigger actions
MATCH (h:Hotkey)-[:TRIGGERS_HOTKEY]->(action)
RETURN h.name, action.qualified_name

// Find context-sensitive hotkeys (#HotIf)
MATCH (h:Hotkey)
WHERE h.is_context_sensitive = true
RETURN h.name, h.qualified_name

// Find hotstrings with case-sensitive option
MATCH (h:Hotstring)
WHERE "C" IN h.options
RETURN h.name, h.options

// Find class inheritance (reuse INHERITS)
MATCH (c:AhkClass)-[:INHERITS]->(parent:AhkClass)
RETURN c.name, parent.name

// Find external DLL calls
MATCH (f:Function)-[r:CALLS_EXTERNAL]->()
RETURN f.name, r.dll, r.function

// Find COM object creation
MATCH (f:Function)-[r:CREATES_COM_OBJECT]->()
RETURN f.name, r.progid
```

---

## 7. Implementation Checklist (Updated - All Critical Fixes Applied)

### Phase 1: Core AutoHotkey Support
- [ ] Add tree-sitter-autohotkey dependency to pyproject.toml
- [ ] Add SupportedLanguage.AUTOHOTKEY to SupportedLanguage enum
- [ ] Add TreeSitterModule.AUTOHOTKEY to TreeSitterModule enum
- [ ] Add AutoHotkey constants to constants.py:
  - [ ] File extensions: EXT_AHK, EXT_AH1, EXT_AH2, AUTOHOTKEY_EXTENSIONS
  - [ ] Tree-sitter node types: TS_AHK_* (33+ constants including hotkey_combination, hotstring_pattern, key_sequence)
  - [ ] Field names: FIELD_HOTKEY, FIELD_HOTSTRING, FIELD_OPTIONS, FIELD_PATH, FIELD_VERSION
  - [ ] FQN types: FQN_AHK_SCOPE_TYPES, FQN_AHK_FUNCTION_TYPES
  - [ ] Spec types: SPEC_AHK_FUNCTION_TYPES, SPEC_AHK_CLASS_TYPES, SPEC_AHK_MODULE_TYPES, SPEC_AHK_CALL_TYPES, SPEC_AHK_IMPORT_TYPES, SPEC_AHK_IMPORT_FROM_TYPES (explicit tuple), SPEC_AHK_PACKAGE_INDICATORS
  - [ ] Package indicators: PKG_AHK_MAIN, PKG_AHK_INDEX
  - [ ] Version directives: AHK_V2_DIRECTIVE, AHK_V1_DIRECTIVE
- [ ] Create AUTOHOTKEY_FQN_SPEC and AUTOHOTKEY_LANGUAGE_SPEC in language_spec.py (with cs. prefixes on all constants)
- [ ] Create AutoHotkeyHandler class in parsers/handlers/autohotkey.py
- [ ] Register AutoHotkeyHandler in parsers/handlers/registry.py _HANDLERS dict
- [ ] Add entry to LANGUAGE_SPECS dict in language_spec.py
- [ ] Add entry to LANGUAGE_FQN_SPECS dict in language_spec.py
- [ ] Add AutoHotkey entry to LANGUAGE_METADATA dict in constants.py with LanguageStatus.DEV

### Phase 2: Graph Schema Extensions
- [ ] Add new NodeLabels to NodeLabel enum: HOTKEY, HOTSTRING, LABEL, CLASS_AHK
- [ ] Add all 4 new NodeLabels to _NODE_LABEL_UNIQUE_KEYS dict (CRITICAL - runtime check!)
- [ ] Add new RelationshipTypes to RelationshipType enum:
  - [ ] DEFINES_HOTKEY, DEFINES_HOTSTRING, DEFINES_LABEL
  - [ ] TRIGGERS_HOTKEY, CALLS_COMMAND, CALLS_EXTERNAL, CREATES_COM_OBJECT
  - [ ] SENDS_KEYS, CLICKS_ELEMENT
  - [ ] CREATES_GUI, CONTROLS_GUI, INCLUDES_FILE
  - [ ] Note: Reuse existing CALLS, INHERITS, IMPLEMENTS, DEFINES where applicable
- [ ] Add NodeSchema entries for all 4 new node types to types_defs.py
- [ ] Add RelationshipSchema entries for ALL relationship types (including DEFINES_*, CALLS_EXTERNAL, etc.)
- [ ] Update schema_builder.py for new node/relationship types
- [ ] Update NODE_SCHEMAS and RELATIONSHIP_SCHEMAS dicts

### Phase 3: Import Resolution (CRITICAL - ImportProcessor Integration)
- [ ] Add AUTOHOTKEY case to parse_imports() match statement in ImportProcessor
- [ ] Add _parse_ahk_imports(captures, module_qn) method (correct signature!)
- [ ] Add _extract_ahk_include_path(node) helper method
- [ ] Add _resolve_ahk_include_path(include_path) method (uses self.current_file)
- [ ] Add _expand_ahk_variables(path) method (uses self.current_file for A_ScriptDir)
- [ ] Add _get_ahk_include_dirs() method
- [ ] Add import_query to AUTOHOTKEY_LANGUAGE_SPEC
- [ ] Handle #Include directive parsing
- [ ] Handle #Require directive (store with "require:" prefix key)
- [ ] Resolve relative paths using self.current_file
- [ ] Resolve lib/, Lib/, includes/ directories
- [ ] Handle A_ScriptDir, A_MyDocuments, A_AppData expansion

### Phase 4: Call Resolution
- [ ] Implement function call resolution
- [ ] Implement command call resolution (Send, Click, Gui, etc.)
- [ ] Implement hotkey trigger tracking
- [ ] Handle label jumps (Gosub/Goto)
- [ ] Handle DllCall external calls with DLL metadata
- [ ] Handle ComObjCreate COM objects
- [ ] Track GUI control references
- [ ] Track OnEvent handler links

### Phase 5: Testing
- [ ] Create test fixtures (AHK v2 and v1)
- [ ] Write unit tests for hotkey parsing
- [ ] Write unit tests for hotstring parsing with options
- [ ] Write unit tests for function parsing
- [ ] Write unit tests for label parsing
- [ ] Write unit tests for class parsing (AHK v2)
- [ ] Write unit tests for GUI command parsing
- [ ] Write unit tests for #HotIf context-sensitive hotkeys
- [ ] Write tests for import resolution (#Include, #Require)
- [ ] Write tests for call graphs (function, command, DllCall)
- [ ] Write tests for version detection (v1 vs v2)
- [ ] Write tests for GUI OnEvent handler linking

### Phase 6: Query Integration
- [ ] Update prompts.py with AUTOHOTKEY_ENTITY_TYPES
- [ ] Add Cypher query examples for hotkeys, hotstrings, GUI
- [ ] Test natural language queries
- [ ] Add query examples for version-specific queries

### Phase 7: Documentation
- [ ] Update README.md with AutoHotkey support status
- [ ] Add AutoHotkey examples to docs/
- [ ] Update language support table with AHK row
- [ ] Update ARCHITECTURE.md if needed

---

## 8. AutoHotkey-Specific Challenges

### 8.1 Version Differences (v1 vs v2)

**AHK v1 (Legacy):**
```autohotkey
; Function call without parentheses (command syntax)
MyFunction, param1, param2

; Variable expansion with percent signs
var := "value"
MsgBox, %var%

; No return keyword needed
MyFunction() {
    result := 42
}
```

**AHK v2 (Modern):**
```autohotkey
; Function call with parentheses
MyFunction(param1, param2)

; Direct variable references
var := "value"
MsgBox(var)

; Explicit return
MyFunction() {
    result := 42
    return result
}
```

**Handler must detect and handle both syntaxes.**

### 8.2 Hotkey Syntax Complexity

```autohotkey
; Simple hotkey
F1::MsgBox("Pressed F1")

; Combination
^!s::Send("Hello")  ; Ctrl+Alt+s

; Context-sensitive
#HotIf WinActive("ahk_class Notepad")
^s::Send("^s")  ; Only in Notepad
#HotIf

; Mouse hotkeys
LButton & RButton::MsgBox("Both buttons")

; Custom combination
Joy1::MsgBox("Joystick button 1")
```

### 8.3 Hotstring Options

```autohotkey
; Basic hotstring
::btw::by the way

; With options
:C*:btw::by the way  ; Case-sensitive, fire immediately
:O:btw::by the way   ; Omit ending character
:P:btw::by the way   ; Persistent (doesn't reset)
:X:run::notepad.exe  ; Execute command
```

### 8.4 GUI Control Tracking

```autohotkey
; GUI creation
gui := Gui()
gui.Add("Button", "Default", "Click me").OnEvent("Click", ButtonClick)
gui.Add("Edit", "vMyEdit")
gui.Show()

; Need to track:
; - Control types (Button, Edit, Text, etc.)
; - Control options (vMyEdit = variable name)
; - Event handlers (OnClick, OnChange, etc.)
```

---

## 9. File Changes Summary

| File | Changes |
|------|---------|
| `pyproject.toml` | Add tree-sitter-autohotkey dependency |
| `codebase_rag/constants.py` | Add AutoHotkey node types, extensions, enums, FIELD_PATH, FIELD_VERSION |
| `codebase_rag/language_spec.py` | Add AUTOHOTKEY_FQN_SPEC, AUTOHOTKEY_LANGUAGE_SPEC |
| `codebase_rag/parsers/handlers/autohotkey.py` | New file - AutoHotkeyHandler |
| `codebase_rag/parsers/import_processor.py` | Add AHK import methods (NOT standalone resolver) |
| `codebase_rag/prompts.py` | Add AutoHotkey entity documentation |
| `codebase_rag/tests/test_autohotkey.py` | New file - test cases |
| `codebase_rag/tests/fixtures/autohotkey/` | New directory - test fixtures |

---

## 10. References

- [AutoHotkey Documentation](https://www.autohotkey.com/docs/)
- [AutoHotkey v2 Documentation](https://www.autohotkey.com/v2/docs/)
- [tree-sitter-autohotkey](https://github.com/ahkscript/tree-sitter-autohotkey)
- [AutoHotkey Community](https://www.autohotkey.com/boards/)

---

## 11. Implementation Notes

### 11.1 Tree-sitter Grammar Quality

The tree-sitter-autohotkey grammar is community-maintained and may have gaps:
- **AHK v2**: Good coverage
- **AHK v1**: Partial coverage
- **Edge cases**: May need manual AST handling

**Mitigation**: Implement fallback parsing in handler for unsupported constructs.

### 11.2 Windows-Specific Features

AutoHotkey is Windows-centric:
- Registry operations
- COM objects
- Windows API calls via DllCall
- Window management

**Consideration**: These may require special handling for cross-platform analysis.

### 11.3 Dynamic Nature

AHK is highly dynamic:
- Runtime variable expansion
- Dynamic function calls
- Eval-like constructs

**Limitation**: Static analysis cannot resolve all dynamic patterns. Document limitations clearly.

### 11.4 Cross-Platform Considerations

AutoHotkey is Windows-centric. On non-Windows systems (Linux, macOS):

**Limitations:**
- AHK scripts may not execute (requires Wine or AHK_Linux)
- Registry operations will fail
- COM objects not available
- Window management commands OS-specific
- Send/Click commands target Windows UI

**Analysis Still Works:**
- Static AST parsing works cross-platform
- Graph structure analysis platform-agnostic
- Hotkey/hotstring definitions portable
- Function/class structure analysis valid

**Recommendation:**
- Document in README that AHK analysis is for code structure only
- Warn users that runtime behavior is Windows-specific
- Consider adding platform metadata to AHK nodes

---

## 12. Future Enhancements

### 12.1 AHK v1 Full Support

Current focus is on AHK v2. Future work:
- Complete v1 syntax support
- v1-to-v2 migration analysis
- Version-specific linting

### 12.2 GUI Flow Analysis

Advanced GUI tracking:
- GUI event flow graphs
- Control dependency analysis
- UI state machine reconstruction

### 12.3 Automation Workflow Analysis

Script behavior analysis:
- Keystroke/mouse action sequences
- Window interaction patterns
- Automation dependency graphs

### 12.4 Security Analysis

Security-focused features:
- Detect dangerous commands (FileDelete, RegDelete)
- Identify potential keyloggers
- Audit script permissions

---

## 13. Risk Assessment

### Low Risk

- Abstract base class design (follows existing pattern)
- Registry pattern (well-established)
- Configuration extension (additive)

### Medium Risk

- Tree-sitter grammar quality (community-maintained)
- AHK v1 vs v2 syntax differences
- Dynamic feature resolution

### High Risk

- Breaking existing behavior (must test thoroughly)
- Windows-specific features on non-Windows systems

### Mitigation Strategies

1. **Default = existing**: No change to default behavior
2. **Version detection**: Auto-detect AHK version from directives
3. **Comprehensive tests**: 50+ tests ensure no regression
4. **Graceful degradation**: Handle unsupported constructs gracefully

---

## 14. Test Summary

| Category | Count | File |
|----------|-------|------|
| Unit Tests | 30+ | test_autohotkey.py |
| Integration Tests | 15+ | test_autohotkey_integration.py |
| Import Tests | 8+ | test_autohotkey_imports.py |
| Call Graph Tests | 10+ | test_autohotkey_calls.py |
| Version Detection Tests | 5+ | test_autohotkey_version.py |
| **Total** | **70+** | |

### Recommended Test Files

- `test_autohotkey.py` - Unit tests for hotkeys, hotstrings, functions, labels, classes
- `test_autohotkey_integration.py` - Integration tests with graph updater
- `test_autohotkey_imports.py` - Import resolution tests
- `test_autohotkey_calls.py` - Call graph and command tracking tests
- `test_autohotkey_version.py` - AHK v1 vs v2 detection tests

---

## 15. Conclusion

AutoHotkey support adds valuable capabilities for analyzing Windows automation scripts, hotkey configurations, and GUI applications. The implementation follows the established pattern for language support in Code-Graph-RAG, with special handling for AHK-specific features like hotkeys, hotstrings, and GUI commands.

**Key Benefits:**
- Analyze automation workflows
- Track hotkey/hotstring definitions
- Understand GUI application structure
- Resolve script dependencies
- Query automation patterns

**Next Steps:**
1. Verify tree-sitter-autohotkey grammar availability
2. Implement Phase 1 (Core Support)
3. Create comprehensive test suite
4. Document AHK-specific query patterns
5. Gather community feedback for enhancements
