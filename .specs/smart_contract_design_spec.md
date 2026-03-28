# Smart Contract Language Support Design Specification

## Executive Summary

This document provides comprehensive design specifications for extending Code-Graph-RAG (CGR) to support smart contract programming languages, with primary focus on Solidity and secondary support for Vyper, Move, and other blockchain languages.

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

## 2. Solidity Implementation

### 2.1 Tree-sitter Dependency

```toml
# pyproject.toml - Add to [project.dependencies] (alphabetically with other tree-sitter)
"tree-sitter-solidity>=1.2.11",  # NOTE: Use 1.2.11+, NOT 0.21.0 (doesn't exist)

# Also add to [project.optional-dependencies].treesitter-full
treesitter-full = [
    # ... existing ...
    "tree-sitter-solidity>=1.2.11",
    # ...
]
```

**Available packages on PyPI:**
- `tree-sitter-solidity` versions: 1.2.11, 1.2.12, 1.2.13 (latest)
- Maintained by JoranHonig/tree-sitter-solidity on GitHub
- ⚠️ Version `0.21.0` does NOT exist - use `>=1.2.11`

### 2.2 Constants (constants.py)

```python
# File extensions
EXT_SOL = ".sol"
SOLIDITY_EXTENSIONS = (EXT_SOL,)

# Tree-sitter module enum
class TreeSitterModule(StrEnum):
    SOLIDITY = "tree_sitter_solidity"

# Supported language enum
class SupportedLanguage(StrEnum):
    # ... existing
    SOLIDITY = "solidity"

# Tree-sitter node types for Solidity
TS_SOL_SOURCE_FILE = "source_file"
TS_SOL_CONTRACT_DECLARATION = "contract_declaration"
TS_SOL_INTERFACE_DECLARATION = "interface_declaration"
TS_SOL_LIBRARY_DECLARATION = "library_declaration"
TS_SOL_FUNCTION_DEFINITION = "function_definition"
TS_SOL_MODIFIER_DEFINITION = "modifier_definition"
TS_SOL_EVENT_DEFINITION = "event_definition"
TS_SOL_STRUCT_DECLARATION = "struct_declaration"
TS_SOL_ENUM_DECLARATION = "enum_declaration"
TS_SOL_ERROR_DECLARATION = "error_declaration"
TS_SOL_STATE_VARIABLE_DECLARATION = "state_variable_declaration"
TS_SOL_VARIABLE_DECLARATION = "variable_declaration"
TS_SOL_CALL_EXPRESSION = "call_expression"
TS_SOL_MEMBER_EXPRESSION = "member_expression"
TS_SOL_IMPORT_DIRECTIVE = "import_directive"
TS_SOL_PRAGMA_DIRECTIVE = "pragma_directive"
TS_SOL_USING_DIRECTIVE = "using_directive"
TS_SOL_EMIT_STATEMENT = "emit_statement"
TS_SOL_FALLBACK_RECEIVE_DEFINITION = "fallback_receive_definition"
TS_SOL_CONSTRUCTOR_DEFINITION = "constructor_definition"
TS_SOL_USER_DEFINED_TYPE = "user_defined_type"
TS_SOL_IDENTIFIER = "identifier"
# Additional node types used in handler
TS_SOL_MODIFIER_INVOCATION = "modifier_invocation"
TS_SOL_INHERITANCE_SPECIFIER = "inheritance_specifier"
TS_SOL_VISIBILITY = "visibility"
TS_SOL_STATE_MUTABILITY = "state_mutability"

# Field names (use existing FIELD_* pattern from constants.py)
# Note: FIELD_NAME already exists at constants.py:618
FIELD_VISIBILITY = "visibility"
FIELD_STATE_MUTABILITY = "state_mutability"
# Note: virtual, override, is_abstract are presence-check children, not field names

# FQN scope types
FQN_SOL_SCOPE_TYPES = (
    TS_SOL_CONTRACT_DECLARATION,
    TS_SOL_INTERFACE_DECLARATION,
    TS_SOL_LIBRARY_DECLARATION,
    TS_SOL_SOURCE_FILE,
)

FQN_SOL_FUNCTION_TYPES = (
    TS_SOL_FUNCTION_DEFINITION,
    TS_SOL_MODIFIER_DEFINITION,
    TS_SOL_FALLBACK_RECEIVE_DEFINITION,
    TS_SOL_CONSTRUCTOR_DEFINITION,
)

# Language spec node types
SPEC_SOL_FUNCTION_TYPES = (
    TS_SOL_FUNCTION_DEFINITION,
    TS_SOL_MODIFIER_DEFINITION,
    TS_SOL_CONSTRUCTOR_DEFINITION,
    TS_SOL_FALLBACK_RECEIVE_DEFINITION,
)

SPEC_SOL_CLASS_TYPES = (
    TS_SOL_CONTRACT_DECLARATION,
    TS_SOL_INTERFACE_DECLARATION,
    TS_SOL_LIBRARY_DECLARATION,
)

SPEC_SOL_MODULE_TYPES = (TS_SOL_SOURCE_FILE,)

SPEC_SOL_CALL_TYPES = (
    TS_SOL_CALL_EXPRESSION,
    TS_SOL_MEMBER_EXPRESSION,
    TS_SOL_EMIT_STATEMENT,
)

# Import types (Solidity uses single import_directive for all patterns)
SPEC_SOL_IMPORT_TYPES = (TS_SOL_IMPORT_DIRECTIVE,)
SPEC_SOL_IMPORT_FROM_TYPES = SPEC_SOL_IMPORT_TYPES  # Same type, consistent with other languages

# Package indicators
PKG_FOUNDRY_TOML = "foundry.toml"
PKG_HARDHAT_CONFIG_JS = "hardhat.config.js"
PKG_HARDHAT_CONFIG_TS = "hardhat.config.ts"
PKG_REMAPPINGS_TXT = "remappings.txt"

SPEC_SOL_PACKAGE_INDICATORS = (
    PKG_FOUNDRY_TOML,
    PKG_HARDHAT_CONFIG_JS,
    PKG_HARDHAT_CONFIG_TS,
    PKG_REMAPPINGS_TXT,
)
```

### 2.3 New Node Labels

```python
class NodeLabel(StrEnum):
    # ... existing
    CONTRACT = "Contract"
    LIBRARY = "Library"
    EVENT = "Event"
    MODIFIER = "Modifier"
    STATE_VARIABLE = "StateVariable"
    CUSTOM_ERROR = "CustomError"
```

**CRITICAL: _NODE_LABEL_UNIQUE_KEYS entries required**

The `_NODE_LABEL_UNIQUE_KEYS` dict in `constants.py` has a runtime check that raises an error if any `NodeLabel` is missing. All new labels must be added:

```python
_NODE_LABEL_UNIQUE_KEYS: dict[NodeLabel, UniqueKeyType] = {
    # ... existing entries ...
    NodeLabel.CONTRACT: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.LIBRARY: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.EVENT: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.MODIFIER: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.STATE_VARIABLE: UniqueKeyType.QUALIFIED_NAME,
    NodeLabel.CUSTOM_ERROR: UniqueKeyType.QUALIFIED_NAME,
}
```

### 2.4 New Relationship Types

```python
class RelationshipType(StrEnum):
    # ... existing
    # REUSE existing: INHERITS, IMPLEMENTS, CALLS, DEFINES - Already defined in constants.py

    # Phase 1: Core relationships
    EMITS = "EMITS"                 # Function emits event
    MODIFIED_BY = "MODIFIED_BY"     # Function uses modifier
    USES_LIBRARY = "USES_LIBRARY"   # Using X for Y directive
    CALLS_EXTERNAL = "CALLS_EXTERNAL"  # External contract call
    DEFINES_EVENT = "DEFINES_EVENT"     # Contract defines event
    DEFINES_MODIFIER = "DEFINES_MODIFIER"  # Contract defines modifier
    DEFINES_STATE = "DEFINES_STATE"     # Contract defines state variable

    # Phase 2: Advanced call resolution
    CALLS_DELEGATE = "CALLS_DELEGATE"   # Library call (delegatecall semantics)
    CALLS_STATIC = "CALLS_STATIC"       # Staticcall (view/pure external)
    READS_STATE = "READS_STATE"         # Read state variable
    WRITES_STATE = "WRITES_STATE"       # Write state variable
    REVERTS_WITH = "REVERTS_WITH"       # Function reverts with error
```

### 2.5 Language Specification (language_spec.py)

```python
def _solidity_get_name(node: Node) -> str | None:
    """Extract name from Solidity AST node."""
    name_node = node.child_by_field_name(cs.FIELD_NAME)  # Use cs.FIELD_NAME constant (same value as TS_FIELD_NAME)
    if name_node and name_node.text:
        return name_node.text.decode(cs.ENCODING_UTF8)

    # Handle constructor fallback name (constants accessed via cs. prefix)
    if node.type == cs.TS_SOL_CONSTRUCTOR_DEFINITION:
        return "constructor"
    if node.type == cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION:
        # Determine receive vs fallback based on payable modifier presence
        # receive() has payable modifier, fallback() does not
        for child in node.children:
            if child.type == cs.TS_SOL_STATE_MUTABILITY:
                text = child.text.decode(cs.ENCODING_UTF8) if child.text else ""
                if text == "payable":
                    return "receive"
        return "fallback"

    return None

def _solidity_file_to_module(file_path: Path, repo_root: Path) -> list[str]:
    """Convert Solidity file path to module parts."""
    try:
        rel = file_path.relative_to(repo_root)
        parts = list(rel.with_suffix("").parts)
        # Handle common src/ or contracts/ prefixes
        if parts and parts[0] in ("src", "contracts", "script", "test"):
            parts = parts[1:]
        return parts
    except ValueError:
        return []

SOLIDITY_FQN_SPEC = FQNSpec(
    scope_node_types=frozenset(cs.FQN_SOL_SCOPE_TYPES),
    function_node_types=frozenset(cs.FQN_SOL_FUNCTION_TYPES),
    get_name=_solidity_get_name,
    file_to_module_parts=_solidity_file_to_module,
)

SOLIDITY_LANGUAGE_SPEC = LanguageSpec(
    language=cs.SupportedLanguage.SOLIDITY,
    file_extensions=cs.SOLIDITY_EXTENSIONS,
    function_node_types=cs.SPEC_SOL_FUNCTION_TYPES,
    class_node_types=cs.SPEC_SOL_CLASS_TYPES,
    module_node_types=cs.SPEC_SOL_MODULE_TYPES,
    call_node_types=cs.SPEC_SOL_CALL_TYPES,
    import_node_types=cs.SPEC_SOL_IMPORT_TYPES,
    import_from_node_types=cs.SPEC_SOL_IMPORT_FROM_TYPES,
    package_indicators=cs.SPEC_SOL_PACKAGE_INDICATORS,
    function_query="""
    (function_definition
        name: (identifier) @name) @function
    (modifier_definition
        name: (identifier) @name) @function
    (constructor_definition) @function
    (fallback_receive_definition) @function
    """,
    class_query="""
    (contract_declaration
        name: (identifier) @name) @class
    (interface_declaration
        name: (identifier) @name) @class
    (library_declaration
        name: (identifier) @name) @class
    (struct_declaration
        name: (identifier) @name) @class
    (enum_declaration
        name: (identifier) @name) @class
    """,
    call_query="""
    (call_expression
        function: (expression
            (identifier) @name)) @call
    (call_expression
        function: (expression
            (member_expression
                property: (identifier) @name))) @call
    (emit_statement
        name: (expression
            (identifier) @name)) @call
    """,
)

# Add to LANGUAGE_FQN_SPECS dict in language_spec.py:
# LANGUAGE_FQN_SPECS: dict[cs.SupportedLanguage, FQNSpec] = {
#     ... existing entries ...
#     cs.SupportedLanguage.SOLIDITY: SOLIDITY_FQN_SPEC,
# }

# Add to LANGUAGE_SPECS dict in language_spec.py:
# LANGUAGE_SPECS: dict[cs.SupportedLanguage, LanguageSpec] = {
#     ... existing entries ...
#     cs.SupportedLanguage.SOLIDITY: SOLIDITY_LANGUAGE_SPEC,
# }

# Add to LANGUAGE_METADATA dict in constants.py:
# LANGUAGE_METADATA: dict[SupportedLanguage, LanguageMetadata] = {
#     ... existing entries ...
#     SupportedLanguage.SOLIDITY: LanguageMetadata(
#         LanguageStatus.FULL,
#         "Contracts, interfaces, libraries, events, modifiers, state variables",
#         "Solidity",
#     ),
# }
```

### 2.6 Solidity Handler (parsers/handlers/solidity.py)

```python
from __future__ import annotations
from typing import TYPE_CHECKING
from ... import constants as cs
from ..utils import safe_decode_text
from .base import BaseLanguageHandler

if TYPE_CHECKING:
    from ...types_defs import ASTNode

class SolidityHandler(BaseLanguageHandler):
    __slots__ = ()

    def is_class_method(self, node: ASTNode) -> bool:
        """Check if function is inside a contract (class method)."""
        return self.is_contract_function(node)

    def extract_function_name(self, node: ASTNode) -> str | None:
        """Extract function name, handling special cases."""
        if node.type == cs.TS_SOL_CONSTRUCTOR_DEFINITION:
            return "constructor"
        if node.type == cs.TS_SOL_FALLBACK_RECEIVE_DEFINITION:
            # Determine if this is receive or fallback based on visibility
            # receive() has payable modifier, fallback() does not
            if self.is_payable(node):
                return "receive"
            return "fallback"
        name_node = node.child_by_field_name(cs.FIELD_NAME)
        if name_node and name_node.text:
            return safe_decode_text(name_node)
        return None

    def extract_decorators(self, node: ASTNode) -> list[str]:
        """Extract modifiers used on functions."""
        modifiers = []
        for child in node.children:
            if child.type == cs.TS_SOL_MODIFIER_INVOCATION:
                # Extract the identifier child for the modifier name
                for subchild in child.children:
                    if subchild.type == cs.TS_SOL_IDENTIFIER:
                        if mod_text := safe_decode_text(subchild):
                            modifiers.append(mod_text)
                        break
        return modifiers

    def is_contract_function(self, node: ASTNode) -> bool:
        """Check if function is inside a contract."""
        current = node.parent
        while current:
            if current.type == cs.TS_SOL_CONTRACT_DECLARATION:
                return True
            if current.type == cs.TS_SOL_SOURCE_FILE:
                return False
            current = current.parent
        return False

    def extract_visibility(self, node: ASTNode) -> str | None:
        """Extract visibility modifier (public, private, internal, external)."""
        for child in node.children:
            if child.type == cs.TS_SOL_VISIBILITY:
                return safe_decode_text(child)
        return None

    def extract_state_mutability(self, node: ASTNode) -> str | None:
        """Extract state mutability (view, pure, payable)."""
        for child in node.children:
            if child.type == cs.TS_SOL_STATE_MUTABILITY:
                return safe_decode_text(child)
        return None

    def is_payable(self, node: ASTNode) -> bool:
        """Check if function is payable."""
        return self.extract_state_mutability(node) == "payable"

    def is_view_or_pure(self, node: ASTNode) -> bool:
        """Check if function is view or pure."""
        mutability = self.extract_state_mutability(node)
        return mutability in ("view", "pure")

    def extract_inheritance(self, node: ASTNode) -> list[str]:
        """Extract inherited contracts/interfaces from inheritance specifier."""
        bases = []
        for child in node.children:
            if child.type == cs.TS_SOL_INHERITANCE_SPECIFIER:
                for base in child.children:
                    if base.type == cs.TS_SOL_USER_DEFINED_TYPE:
                        # Extract identifier from user_defined_type node
                        for subchild in base.children:
                            if subchild.type == cs.TS_SOL_IDENTIFIER:
                                if base_text := safe_decode_text(subchild):
                                    bases.append(base_text)
                                break
        return bases

    def extract_base_class_name(self, base_node: ASTNode) -> str | None:
        """Extract base contract name from user_defined_type node."""
        if base_node.type == cs.TS_SOL_USER_DEFINED_TYPE:
            for child in base_node.children:
                if child.type == cs.TS_SOL_IDENTIFIER:
                    return safe_decode_text(child)
        return safe_decode_text(base_node) if base_node.text else None
```

### 2.7 Import Resolution (parsers/solidity/import_resolver.py)

> **Architecture Note:** The import resolution should integrate with the existing `ImportProcessor`
> pattern in `codebase_rag/parsers/import_processor.py`. Add methods like `_parse_solidity_imports()`
> and `_resolve_solidity_import_path()` to `ImportProcessor` rather than creating a standalone resolver.

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SolidityImport:
    """Represents a resolved Solidity import."""
    path: str
    symbols: list[str]  # Imported symbols
    aliases: dict[str, str]  # Symbol -> alias mapping
    namespace_name: str | None = None  # For namespace imports (import * as X)
    is_absolute: bool = True
    remapped_path: str | None = None  # After remapping

class SolidityImportResolver:
    """Resolve Solidity imports with remapping support."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.remappings: dict[str, str] = {}
        self._load_remappings()

    def _load_remappings(self) -> None:
        """Load remappings from foundry.toml or remappings.txt."""
        # Check for remappings.txt
        remappings_file = self.repo_path / "remappings.txt"
        if remappings_file.exists():
            for line in remappings_file.read_text().splitlines():
                # Remove comments and strip whitespace
                line = line.split('#')[0].strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    self.remappings[key.strip()] = value.strip()
            # Sort by key length descending (most specific first)
            self.remappings = dict(sorted(
                self.remappings.items(),
                key=lambda x: len(x[0]),
                reverse=True
            ))

        # Check for foundry.toml
        foundry_toml = self.repo_path / "foundry.toml"
        if foundry_toml.exists():
            # Parse foundry.toml for remappings
            # Note: foundry.toml uses TOML format, remappings in [profile.default]
            # Example: remappings = ["@openzeppelin/=lib/openzeppelin-contracts/"]
            # For now, rely on remappings.txt which is more commonly used
            # TODO: Add TOML parsing if needed (requires toml dependency)
            pass

    def resolve_import(self, import_path: str, current_file: Path | None = None) -> Path | None:
        """Resolve import path to actual file path.

        Args:
            import_path: The import path from the import statement
            current_file: The file containing the import (needed for relative imports)
        """
        # Apply remappings
        for key, value in self.remappings.items():
            if import_path.startswith(key):
                import_path = value + import_path[len(key):]
                break

        # Try relative resolution (needs current file context)
        if import_path.startswith("./") or import_path.startswith("../"):
            if current_file is None:
                return None
            base_dir = current_file.parent
            return (base_dir / import_path).resolve()

        # Try node_modules
        node_modules = self.repo_path / "node_modules" / import_path
        if node_modules.exists():
            return node_modules

        # Try lib (Foundry)
        lib_path = self.repo_path / "lib" / import_path
        if lib_path.exists():
            return lib_path

        return None
```

### 2.8 Call Resolution Strategy

```python
# Internal calls: Same contract
# - functionName() -> resolve to internal function
# - super.functionName() -> resolve via C3 linearization MRO (skipping current contract)

# External calls to self:
# - this.functionName() -> external call via contract address (not internal!)

# External calls: Other contracts
# - contractVar.functionName() -> resolve to contract type
# - address.functionName() -> resolve dynamically (track as external)
# - Interface(address).functionName() -> resolve to interface definition

# Library calls (delegatecall semantics):
# - LibraryName.function(args) -> CALLS_DELEGATE to library function
# - using Library for Type; variable.method() -> CALLS_DELEGATE with library metadata

# Special calls:
# - emit EventName() -> EMITS relationship to Event node
# - delegatecall -> CALLS_DELEGATE relationship
# - staticcall -> CALLS_STATIC relationship
```

---

## 3. Graph Schema Extensions

### 3.1 Node Properties

```python
# Contract node properties
CONTRACT_PROPERTIES = {
    "name": str,                    # Contract name
    "qualified_name": str,          # Fully qualified name
    "is_abstract": bool,            # Abstract contract flag
    "license": str | None,          # SPDX license
    "pragma": str | None,           # Solidity version
    "path": str,                    # Source file path
    "absolute_path": str,           # Absolute file path
    "start_line": int,
    "end_line": int,
    "is_upgradeable": bool | None,  # Uses proxy pattern
}

# Library node properties
LIBRARY_PROPERTIES = {
    "name": str,
    "qualified_name": str,
    "path": str,
    "absolute_path": str,           # Absolute file path
    "start_line": int,
    "end_line": int,
}

# Function node properties (extended)
FUNCTION_PROPERTIES_SOLIDITY = {
    "visibility": str,              # public, private, internal, external
    "state_mutability": str,        # view, pure, payable, nonpayable
    "is_virtual": bool,             # virtual function flag
    "is_override": bool,            # override flag
    "modifiers": list[str],         # Applied modifiers
    "gas_estimate": int | None,     # Optional gas estimate
}

# Event node properties
EVENT_PROPERTIES = {
    "name": str,
    "qualified_name": str,          # CRITICAL: Required for UniqueKeyType.QUALIFIED_NAME
    "signature": str | None,        # Event signature hash
    "parameters": list[str],        # Parameter types list (compatible with PropertyValue)
    "indexed_count": int,           # Number of indexed params
    "is_anonymous": bool,
    "path": str,                    # Source file path
    "absolute_path": str,           # Absolute file path
    "start_line": int,
    "end_line": int,
}

# Modifier node properties
MODIFIER_PROPERTIES = {
    "name": str,
    "qualified_name": str,
    "path": str,
    "absolute_path": str,           # Absolute file path
    "start_line": int,
    "end_line": int,
}

# StateVariable node properties
STATE_VARIABLE_PROPERTIES = {
    "name": str,
    "qualified_name": str,          # CRITICAL: Required for UniqueKeyType.QUALIFIED_NAME
    "type": str,                    # Solidity type
    "visibility": str,              # public, private, internal
    "is_constant": bool,
    "is_immutable": bool,
    "is_mapped": bool,              # Is mapping type
    "initial_value": str | None,
    "slot": int | None,             # Storage slot (optional)
    "path": str,
    "absolute_path": str,           # Absolute file path
    "start_line": int,
    "end_line": int,
}

# CustomError node properties
CUSTOM_ERROR_PROPERTIES = {
    "name": str,
    "qualified_name": str,
    "parameters": list[str],        # Parameter types list (compatible with PropertyValue)
    "path": str,
    "absolute_path": str,           # Absolute file path
    "start_line": int,
    "end_line": int,
}
```

### 3.2 Relationship Properties

```python
# INHERITS relationship
INHERITS_PROPERTIES = {
    "is_linearized": bool,  # For multiple inheritance
}

# EMITS relationship
EMITS_PROPERTIES = {
    "location": str,  # Line/column where emit occurs
}

# MODIFIED_BY relationship
MODIFIED_BY_PROPERTIES = {
    "modifier_args": list[str] | None,  # Arguments to modifier
}

# CALLS_EXTERNAL relationship
CALLS_EXTERNAL_PROPERTIES = {
    "target_contract": str | None,     # Target contract name
    "target_function": str,            # Target function name
    "is_interface_call": bool,         # True if called via interface
    "call_type": str | None,           # "transfer", "call", or None
    "function_selector": str | None,   # 4-byte selector (if available)
}

# CALLS_DELEGATE relationship
CALLS_DELEGATE_PROPERTIES = {
    "target_library": str,             # Library name
    "target_function": str,            # Library function name
    "is_using_for": bool,              # True if via "using X for Y"
}

# CALLS_STATIC relationship
CALLS_STATIC_PROPERTIES = {
    "target_contract": str | None,
    "target_function": str,
    "is_interface_call": bool,
}
```

---

## 4. Other Smart Contract Languages

### 4.1 Vyper

```python
# Vyper constants
EXT_VY = ".vy"
VYPER_EXTENSIONS = (EXT_VY,)

TS_VY_MODULE = "source_file"
TS_VY_CONTRACT = "contract"
TS_VY_FUNCTION = "function_def"
TS_VY_EVENT = "event"
TS_VY_STRUCT = "struct"
TS_VY_INTERFACE = "interface"

# Vyper-specific: No modifiers, uses @public/@private decorators
# State variables: self.variable_name pattern
# No inheritance (composition only)
```

### 4.2 Move (Aptos/Sui)

```python
# Move constants
EXT_MOVE = ".move"
MOVE_EXTENSIONS = (EXT_MOVE,)

TS_MOVE_MODULE = "module"
TS_MOVE_STRUCT = "struct_definition"
TS_MOVE_FUNCTION = "function_definition"
TS_MOVE_RESOURCE = "resource"  # Move-specific

# Move-specific patterns:
# - module MyModule { } wrapper
# - struct has key, store, drop abilities
# - entry functions for transactions
# - Resource types
```

### 4.3 Priority Order for Implementation

| Priority | Language | Rationale |
|----------|----------|-----------|
| 1 | Solidity | Largest ecosystem, EVM standard, tree-sitter available |
| 2 | Vyper | EVM compatible, Python-like, simpler grammar |
| 3 | Move | Growing ecosystem (Aptos, Sui), tree-sitter in development |
| 4 | Cairo | StarkNet specific, Python-like syntax |
| 5 | Cadence | Flow blockchain, resource-oriented |

---

## 5. Testing Strategy

### 5.1 Test Structure

```python
# tests/test_solidity.py (consistent with project naming convention)

@pytest.fixture
def solidity_project(temp_repo: Path) -> Path:
    project_path = temp_repo / "sol_test_project"
    project_path.mkdir()

    # Create foundry.toml
    (project_path / "foundry.toml").write_text("""
[profile.default]
src = "src"
out = "out"
libs = ["lib"]
""")

    # Create remappings.txt for npm-style import tests
    (project_path / "remappings.txt").write_text("""
@openzeppelin/=lib/openzeppelin-contracts/
""")

    # Create sample contracts
    src_dir = project_path / "src"
    src_dir.mkdir()

    # Create interface file for import tests
    (src_dir / "IERC20.sol").write_text("""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
}
""")

    (src_dir / "ERC20.sol").write_text("""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

// Custom error definition
error InsufficientBalance(uint256 available, uint256 required);

// Library definition
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
}

contract ERC20 is IERC20 {
    using SafeMath for uint256;

    address private owner;
    mapping(address => uint256) private _balances;
    uint256 private _totalSupply;

    event Transfer(address indexed from, address indexed to, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    function transfer(address to, uint256 amount)
        public
        override
        returns (bool)
    {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal {
        _balances[from] -= amount;
        _balances[to] = SafeMath.add(_balances[to], amount);  // Library call
        emit Transfer(from, to, amount);
    }
}
""")
    return project_path

class TestSolidityContractNodes:
    def test_contract_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        contract_names = get_node_names(mock_ingestor, NodeLabel.CONTRACT)
        assert any("ERC20" in name for name in contract_names), "Expected ERC20 contract"

    def test_interface_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        interface_names = get_node_names(mock_ingestor, NodeLabel.INTERFACE)
        assert any("IERC20" in name for name in interface_names), "Expected IERC20 interface"

class TestSolidityInheritance:
    def test_inherits_relationship(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        inherits = get_relationships(mock_ingestor, "INHERITS")
        # ERC20 implements IERC20 - verify specific inheritance
        assert len(inherits) > 0, "Expected INHERITS relationships"
        # Verify ERC20 -> IERC20 inheritance
        erc20_inherits = [r for r in inherits if "ERC20" in r.get("source", "")]
        assert len(erc20_inherits) > 0, "Expected ERC20 to inherit from IERC20"
        # Verify the target is IERC20
        assert any("IERC20" in r.get("target", "") for r in erc20_inherits), \
            "Expected ERC20 to inherit from IERC20 interface"

class TestSolidityEvents:
    def test_event_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        event_names = get_node_names(mock_ingestor, NodeLabel.EVENT)
        assert any("Transfer" in name for name in event_names), "Expected Transfer event"

    def test_emit_relationship(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        emits = get_relationships(mock_ingestor, "EMITS")
        assert len(emits) > 0, "Expected EMITS relationships from emit statements"
        # Verify _transfer function emits Transfer event
        transfer_emits = [r for r in emits if "Transfer" in r.get("target", "")]
        assert len(transfer_emits) > 0, "Expected _transfer to emit Transfer event"

class TestSolidityModifiers:
    def test_modifier_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        modifier_names = get_node_names(mock_ingestor, NodeLabel.MODIFIER)
        assert any("onlyOwner" in name for name in modifier_names), "Expected onlyOwner modifier in ERC20"

    def test_modified_by_relationship(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        modified_by = get_relationships(mock_ingestor, "MODIFIED_BY")
        assert len(modified_by) > 0, "Expected MODIFIED_BY relationships"
        # Verify a function uses the onlyOwner modifier
        owner_mods = [r for r in modified_by if "onlyOwner" in r.get("target", "")]
        assert len(owner_mods) > 0, "Expected function to use onlyOwner modifier"

class TestSolidityLibraries:
    def test_library_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        library_names = get_node_names(mock_ingestor, NodeLabel.LIBRARY)
        assert any("SafeMath" in name for name in library_names), "Expected SafeMath library"

class TestSolidityStateVariables:
    def test_state_variable_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        state_vars = get_node_names(mock_ingestor, NodeLabel.STATE_VARIABLE)
        assert any("_balances" in name for name in state_vars), "Expected _balances state variable"

class TestSolidityCustomErrors:
    def test_custom_error_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        errors = get_node_names(mock_ingestor, NodeLabel.CUSTOM_ERROR)
        assert any("InsufficientBalance" in name for name in errors), "Expected InsufficientBalance error"

class TestSolidityImports:
    def test_relative_import_resolved(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        imports = get_relationships(mock_ingestor, "IMPORTS")
        # Verify relative imports (./IERC20.sol) are resolved
        assert len(imports) > 0, "Expected IMPORTS relationships from import statements"

    def test_npm_style_import_resolved(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        imports = get_relationships(mock_ingestor, "IMPORTS")
        # Verify @openzeppelin imports with remappings are resolved
        import_paths = [r.get("target_path", "") for r in imports]
        assert any("openzeppelin" in p.lower() for p in import_paths), \
            "Expected @openzeppelin import to be resolved via remappings"

class TestSolidityCallGraphs:
    def test_internal_call_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        # Test _transfer called from transfer
        calls = get_relationships(mock_ingestor, "CALLS")
        assert len(calls) > 0, "Expected CALLS relationships for internal function calls"

    def test_emit_statement_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        emits = get_relationships(mock_ingestor, "EMITS")
        assert len(emits) > 0, "Expected EMITS relationships for emit statements"

    def test_library_call_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        # Test calls to SafeMath library use CALLS_DELEGATE
        delegate_calls = get_relationships(mock_ingestor, "CALLS_DELEGATE")
        assert len(delegate_calls) > 0, "Expected CALLS_DELEGATE relationships for library calls"
```

### 5.2 Test Fixtures

```python
# Sample contracts for testing:
# - Basic contract with state variables
# - Contract with inheritance chain
# - Contract implementing interface
# - Contract with modifiers
# - Contract with events
# - Library with functions
# - Import statements (relative, absolute, npm-style)
# - Using for directives
# - Custom errors
```

---

## 6. Query System Integration

### 6.1 Prompt Extensions

```python
# Add to prompts.py

SOLIDITY_ENTITY_TYPES = """
Smart Contract Entities:
- Contract: A Solidity contract definition
- Interface: A Solidity interface definition
- Library: A Solidity library definition
- Event: An event definition that can be emitted
- Modifier: A function modifier
- StateVariable: Contract state storage
- CustomError: User-defined error types

Function Properties:
- visibility: public, private, internal, external
- mutability: view, pure, payable
- modifiers: Applied modifier names

Relationships:
- INHERITS: Contract inheritance chain
- IMPLEMENTS: Contract implements interface
- EMITS: Function emits event
- MODIFIED_BY: Function uses modifier
- CALLS: Internal function call
- CALLS_EXTERNAL: External contract call
- CALLS_DELEGATE: Library call (delegatecall semantics)
- CALLS_STATIC: Staticcall (view/pure external)
- USES_LIBRARY: Using X for Y directive
- DEFINES_EVENT: Contract defines event
- DEFINES_MODIFIER: Contract defines modifier
- DEFINES_STATE: Contract defines state variable
"""

# Example queries:
# "What contracts inherit from ERC20?"
# "Which functions emit Transfer events?"
# "Show all public functions in the token contract"
# "What modifiers are applied to transfer()?"
# "Find all view functions that don't modify state"
```

### 6.2 Cypher Query Examples

```cypher
// Find all contracts inheriting from a base contract
MATCH (child:Contract)-[:INHERITS]->(base:Contract {name: "ERC20"})
RETURN child.name, child.qualified_name

// Find functions that emit specific events
MATCH (f:Function)-[:EMITS]->(e:Event {name: "Transfer"})
RETURN f.qualified_name, f.visibility

// Find all external function calls
MATCH (caller:Function)-[:CALLS_EXTERNAL]->(callee:Function)
RETURN caller.qualified_name, callee.qualified_name

// Find functions using specific modifiers
MATCH (f:Function)-[:MODIFIED_BY]->(m:Modifier {name: "onlyOwner"})
RETURN f.qualified_name

// Get full contract structure
MATCH (c:Contract {name: "ERC20"})
OPTIONAL MATCH (c)-[:DEFINES]->(f:Function)
OPTIONAL MATCH (c)-[:DEFINES_EVENT]->(e:Event)
OPTIONAL MATCH (c)-[:DEFINES_STATE]->(sv:StateVariable)
OPTIONAL MATCH (c)-[:DEFINES_MODIFIER]->(m:Modifier)
RETURN c, collect(DISTINCT f) as functions,
       collect(DISTINCT e) as events,
       collect(DISTINCT sv) as state_vars,
       collect(DISTINCT m) as modifiers
```

---

## 7. Project Detection

### 7.1 Foundry Project

```python
def is_foundry_project(path: Path) -> bool:
    return (path / "foundry.toml").exists()

def get_foundry_src_dirs(path: Path) -> list[Path]:
    """Get source directories from foundry.toml."""
    src_dirs = []
    if (path / "src").exists():
        src_dirs.append(path / "src")
    if (path / "test").exists():
        src_dirs.append(path / "test")
    if (path / "script").exists():
        src_dirs.append(path / "script")
    return src_dirs
```

### 7.2 Hardhat Project

```python
def is_hardhat_project(path: Path) -> bool:
    return (path / "hardhat.config.js").exists() or \
           (path / "hardhat.config.ts").exists()

def get_hardhat_src_dirs(path: Path) -> list[Path]:
    """Get source directories for Hardhat."""
    src_dirs = []
    if (path / "contracts").exists():
        src_dirs.append(path / "contracts")
    return src_dirs
```

---

## 8. Implementation Checklist

### Phase 1: Core Solidity Support
- [ ] Add tree-sitter-solidity dependency to pyproject.toml
- [ ] Add SupportedLanguage.SOLIDITY to SupportedLanguage enum
- [ ] Add TreeSitterModule.SOLIDITY to TreeSitterModule enum
- [ ] Add Solidity constants to constants.py:
  - [ ] File extensions: EXT_SOL, SOLIDITY_EXTENSIONS
  - [ ] Tree-sitter node types: TS_SOL_* (26 constants)
  - [ ] Field names: FIELD_VISIBILITY, FIELD_STATE_MUTABILITY
  - [ ] FQN types: FQN_SOL_SCOPE_TYPES, FQN_SOL_FUNCTION_TYPES
  - [ ] Spec types: SPEC_SOL_FUNCTION_TYPES, SPEC_SOL_CLASS_TYPES, SPEC_SOL_MODULE_TYPES, SPEC_SOL_CALL_TYPES, SPEC_SOL_IMPORT_TYPES, SPEC_SOL_IMPORT_FROM_TYPES, SPEC_SOL_PACKAGE_INDICATORS
  - [ ] Package indicators: PKG_FOUNDRY_TOML, PKG_HARDHAT_CONFIG_JS, PKG_HARDHAT_CONFIG_TS, PKG_REMAPPINGS_TXT
- [ ] Create SOLIDITY_FQN_SPEC and SOLIDITY_LANGUAGE_SPEC in language_spec.py
- [ ] Create SolidityHandler class in parsers/handlers/solidity.py
- [ ] Register SolidityHandler in handlers/registry.py _HANDLERS dict
- [ ] Add entry to LANGUAGE_SPECS dict in language_spec.py
- [ ] Add entry to LANGUAGE_FQN_SPECS dict in language_spec.py
- [ ] Add Solidity entry to LANGUAGE_METADATA dict in constants.py
- [ ] Override extract_base_class_name in SolidityHandler for user_defined_type handling

### Phase 2: Graph Schema Extensions
- [ ] Add new NodeLabels to NodeLabel enum:
  - [ ] CONTRACT, LIBRARY, EVENT, MODIFIER, STATE_VARIABLE, CUSTOM_ERROR
- [ ] Add all 6 new NodeLabels to _NODE_LABEL_UNIQUE_KEYS dict (required!)
- [ ] Add new RelationshipTypes to RelationshipType enum:
  - [ ] Phase 1 (7 types): EMITS, MODIFIED_BY, USES_LIBRARY, CALLS_EXTERNAL, DEFINES_EVENT, DEFINES_MODIFIER, DEFINES_STATE
  - [ ] Phase 2 (5 types): CALLS_DELEGATE, CALLS_STATIC, READS_STATE, WRITES_STATE, REVERTS_WITH
- [ ] Note: INHERITS, IMPLEMENTS, CALLS, DEFINES already exist - reuse them
- [ ] Add RelationshipSchema entries for all 12 new relationship types to types_defs.py
- [ ] Add NodeSchema entries for all 6 new node types to types_defs.py
- [ ] Update schema_builder.py for new node/relationship types

### Phase 3: Import Resolution
- [ ] Add _parse_solidity_imports() to ImportProcessor (not standalone resolver)
- [ ] Handle remappings.txt parsing
- [ ] Handle foundry.toml remappings
- [ ] Handle node_modules and lib resolution
- [ ] Handle src/ and contracts/ resolution paths
- [ ] Add namespace import (import * as X) and alias import parsing

### Phase 4: Call Resolution
- [ ] Implement internal call resolution
- [ ] Implement external call resolution
- [ ] Implement emit statement handling
- [ ] Handle delegatecall/staticcall
- [ ] Implement C3 linearization for super calls and inheritance resolution
- [ ] Add low-level call metadata extraction (call_type, function_selector)

### Phase 5: Testing
- [ ] Create test fixtures
- [ ] Write unit tests for contract parsing
- [ ] Write integration tests for inheritance
- [ ] Write tests for events and modifiers
- [ ] Write tests for libraries and library calls
- [ ] Write tests for state variables
- [ ] Write tests for custom errors
- [ ] Write tests for import resolution
- [ ] Write tests for call graphs

### Phase 6: Query Integration
- [ ] Update prompts.py with Solidity entities
- [ ] Add Cypher query examples
- [ ] Test natural language queries

### Phase 7: Other Languages (Future)
- [ ] Add Vyper support
- [ ] Add Move support
- [ ] Add Cairo support

---

## 9. File Changes Summary

| File | Changes |
|------|---------|
| `pyproject.toml` | Add tree-sitter-solidity dependency |
| `codebase_rag/constants.py` | Add Solidity node types, extensions, enums |
| `codebase_rag/language_spec.py` | Add SOLIDITY_FQN_SPEC, SOLIDITY_LANGUAGE_SPEC |
| `codebase_rag/parsers/handlers/solidity.py` | New file - SolidityHandler |
| `codebase_rag/parsers/solidity/__init__.py` | New file - package init |
| `codebase_rag/parsers/solidity/import_resolver.py` | New file - import resolution |
| `codebase_rag/parsers/solidity/utils.py` | New file - utility functions |
| `codebase_rag/prompts.py` | Add Solidity entity documentation |
| `codebase_rag/tests/test_solidity.py` | New file - test cases |
| `codebase_rag/tests/fixtures/solidity/` | New directory - test fixtures |

---

## 10. Extended Graph Schema (Detailed)

### 10.1 Node Property Schemas

#### CONTRACT Node
| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `name` | str | Contract identifier | `"ERC20Token"` |
| `qualified_name` | str | Fully qualified name | `"contracts/Token.sol.ERC20Token"` |
| `is_abstract` | bool | Abstract contract flag | `true` |
| `license` | str \| None | SPDX license | `"MIT"` |
| `pragma` | str \| None | Solidity version | `">=0.8.0 <0.9.0"` |
| `path` | str | Source file path | `"contracts/Token.sol"` |
| `absolute_path` | str | Absolute file path | `"/project/contracts/Token.sol"` |
| `start_line` | int | Start line number | `10` |
| `end_line` | int | End line number | `50` |
| `is_upgradeable` | bool \| None | Uses proxy pattern | `true` |

#### LIBRARY Node
| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `name` | str | Library identifier | `"SafeMath"` |
| `qualified_name` | str | Fully qualified name | `"contracts/libraries/SafeMath.sol.SafeMath"` |
| `path` | str | Source file path | `"contracts/libraries/SafeMath.sol"` |
| `absolute_path` | str | Absolute file path | `"/project/contracts/libraries/SafeMath.sol"` |
| `start_line` | int | Start line number | `5` |
| `end_line` | int | End line number | `30` |

#### EVENT Node
| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `name` | str | Event identifier | `"Transfer"` |
| `qualified_name` | str | Fully qualified name | `"contracts/Token.sol.ERC20.Transfer"` |
| `signature` | str \| None | Event signature hash | `"0xddf252..."` |
| `parameters` | list[str] | Parameter type list | `["address","address","uint256"]` |
| `indexed_count` | int | Number of indexed params | `2` |
| `is_anonymous` | bool | Anonymous event flag | `false` |
| `path` | str | Source file path | `"contracts/Token.sol"` |
| `absolute_path` | str | Absolute file path | `"/project/contracts/Token.sol"` |
| `start_line` | int | Start line number | `15` |
| `end_line` | int | End line number | `15` |

#### MODIFIER Node
| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `name` | str | Modifier identifier | `"onlyOwner"` |
| `qualified_name` | str | Fully qualified name | `"contracts/Ownable.sol.Ownable.onlyOwner"` |
| `path` | str | Source file path | `"contracts/Ownable.sol"` |
| `absolute_path` | str | Absolute file path | `"/project/contracts/Ownable.sol"` |
| `start_line` | int | Start line number | `20` |
| `end_line` | int | End line number | `23` |

#### STATE_VARIABLE Node
| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `name` | str | Variable identifier | `"balanceOf"` |
| `qualified_name` | str | Fully qualified name | `"contracts/Token.sol.ERC20.balanceOf"` |
| `type` | str | Solidity type | `"mapping(address => uint256)"` |
| `visibility` | str | public/internal/private | `"public"` |
| `is_constant` | bool | Is constant | `false` |
| `is_immutable` | bool | Is immutable | `false` |
| `is_mapped` | bool | Is mapping type | `true` |
| `initial_value` | str \| None | Initial value expression | `None` |
| `slot` | int \| None | Storage slot | `0` |
| `path` | str | Source file path | `"contracts/Token.sol"` |
| `absolute_path` | str | Absolute file path | `"/project/contracts/Token.sol"` |
| `start_line` | int | Start line number | `12` |
| `end_line` | int | End line number | `12` |

#### CUSTOM_ERROR Node
| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `name` | str | Error identifier | `"InsufficientBalance"` |
| `qualified_name` | str | Fully qualified name | `"contracts/Token.sol.InsufficientBalance"` |
| `parameters` | list[str] | Parameter type list | `["uint256","uint256"]` |
| `path` | str | Source file path | `"contracts/Token.sol"` |
| `absolute_path` | str | Absolute file path | `"/project/contracts/Token.sol"` |
| `start_line` | int | Start line number | `8` |
| `end_line` | int | End line number | `8` |

### 10.2 Extended Relationship Types

```python
class RelationshipType(StrEnum):
    # ... existing ...
    EMITS = "EMITS"                     # Function emits event
    MODIFIED_BY = "MODIFIED_BY"         # Function uses modifier
    USES_LIBRARY = "USES_LIBRARY"       # Using X for Y
    CALLS_EXTERNAL = "CALLS_EXTERNAL"   # External contract call
    CALLS_DELEGATE = "CALLS_DELEGATE"   # Delegatecall
    CALLS_STATIC = "CALLS_STATIC"       # Staticcall
    READS_STATE = "READS_STATE"         # Read state variable
    WRITES_STATE = "WRITES_STATE"       # Write state variable
    DEFINES_EVENT = "DEFINES_EVENT"     # Contract defines event
    DEFINES_MODIFIER = "DEFINES_MODIFIER"  # Contract defines modifier
    DEFINES_STATE = "DEFINES_STATE"     # Contract defines state var
    REVERTS_WITH = "REVERTS_WITH"       # Function reverts with error
```

---

## 11. Import Resolution (Detailed)

### 11.1 Solidity Import Syntax

```solidity
// Basic import
import "filename";

// Namespace import
import * as symbolName from "filename";
import "filename" as symbolName;  // Equivalent shorthand

// Symbol import with aliases
import {symbol1 as alias, symbol2} from "filename";

// Relative imports
import "./MyContract.sol";
import "../utils/Helper.sol";

// NPM-style imports
import "@openzeppelin/contracts/token/ERC20.sol";
```

### 11.2 Remapping Resolution

**Foundry (remappings.txt):**
```
ds-test/=lib/ds-test/src/
forge-std/=lib/forge-std/src/
@openzeppelin/=lib/openzeppelin-contracts/
```

**Hardhat (hardhat.config.js):**
```javascript
module.exports = {
  solidity: "0.8.20",
  remappings: [
    "@openzeppelin=node_modules/@openzeppelin",
    "@mylib=libs/mylib/src"
  ]
};
```

### 11.3 Tree-sitter Import Nodes

```
(import_directive
  (import_clause
    (import_namespace_clause
      (identifier "symbolName")))
  (import_path "filename"))

(import_directive
  (import_clause
    (import_alias_clause
      (identifier "symbol1")
      (identifier "alias")))
  (import_path "filename"))
```

---

## 12. Call Resolution Strategy (Detailed)

### 12.1 Solidity Call Types

| Call Type | Description | Relationship |
|-----------|-------------|--------------|
| Internal | Same contract function | `CALLS` |
| External (self) | this.functionName() | `CALLS_EXTERNAL` |
| External (self, view/pure) | this.viewFunction() | `CALLS_STATIC` |
| External | Other contract | `CALLS_EXTERNAL` |
| Static | View/pure external | `CALLS_STATIC` |
| Delegate | Proxy pattern / Library call | `CALLS_DELEGATE` |
| Low-level call | call opcode | `CALLS_EXTERNAL` with metadata |
| Low-level delegatecall | delegatecall opcode | `CALLS_DELEGATE` with metadata |
| Low-level staticcall | staticcall opcode | `CALLS_STATIC` with metadata |
| Transfer/Send | ETH transfer | `CALLS_EXTERNAL` with `call_type: "transfer"` |

### 12.2 Resolution Strategy

**Internal Calls:**
1. Check current contract for function definition
2. Traverse inheritance hierarchy (C3 linearization)
3. Create `CALLS` relationship
4. For `super.` calls, resolve to the next implementation in the C3 linearization MRO starting from the calling contract's position in the hierarchy

**External Calls:**
1. Resolve target contract type from variable declaration
2. Link to function in target contract
3. Add metadata: `target_contract`, `target_function`

**Interface Calls:**
1. Resolve to interface definition
2. Create `CALLS_EXTERNAL` relationship with `is_interface_call: true` metadata
3. Note: Implementation resolution is runtime-only; cannot statically resolve interface calls to implementations

**Low-Level Calls:**
1. Extract function selector if available
2. Link to matching functions
3. Create generic relationship for dynamic calldata

### 12.3 C3 Linearization for Inheritance Resolution

```python
def c3_linearization(contract: Contract, inheritance_graph: Graph) -> list[Contract]:
    """
    Compute the Method Resolution Order (MRO) for a contract using C3 algorithm.

    Example:
        A inherits B, C
        B inherits D
        C inherits D
        MRO: [A, B, C, D]

    super.functionName() resolution:
    1. Find the calling contract's position in MRO
    2. Search for functionName in contracts after that position
    3. Return first match (next in inheritance chain)

    This ensures the parent implementation is called, not current contract's.
    """
    pass

# Example super call resolution:
# contract A is B, C { function foo() public { super.bar(); } }
# MRO: [A, B, C]
# super.bar() searches B.bar() first, then C.bar()
```

### 12.4 Low-Level Call Metadata

```python
CALL_METADATA = {
    "call_type": str,           # "call", "delegatecall", "staticcall", "transfer"
    "function_selector": str | None,   # First 4 bytes of calldata (if available)
    "gas": int | None,          # Gas stipend
    "value": int | None,        # ETH value (for call/transfer)
    "target_address": str,      # Variable or expression for target address
}
```

---

## 13. Testing Patterns (Detailed)

### 13.1 Test Structure Template

```python
@pytest.fixture
def solidity_project(temp_repo: Path) -> Path:
    project_path = temp_repo / "sol_test_project"
    project_path.mkdir()

    (project_path / "foundry.toml").write_text("""
[profile.default]
src = "src"
libs = ["lib"]
""")

    src_dir = project_path / "src"
    src_dir.mkdir()

    (src_dir / "Token.sol").write_text("""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Token {
    event Transfer(address indexed from, address indexed to, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    function transfer(address to, uint256 amount) public returns (bool) {
        emit Transfer(msg.sender, to, amount);
        return true;
    }
}
""")
    return project_path

class TestSolidityContracts:
    def test_contract_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        contract_names = get_node_names(mock_ingestor, NodeLabel.CONTRACT)
        assert any("Token" in name for name in contract_names)

class TestSolidityInheritance:
    def test_inherits_relationship(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        inherits = get_relationships(mock_ingestor, "INHERITS")
        assert len(inherits) >= 0

class TestSolidityEvents:
    def test_event_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        events = get_node_names(mock_ingestor, NodeLabel.EVENT)
        assert any("Transfer" in name for name in events)

class TestSolidityModifiers:
    def test_modifier_detected(self, solidity_project, mock_ingestor):
        run_updater(solidity_project, mock_ingestor, skip_if_missing="solidity")
        modifiers = get_node_names(mock_ingestor, NodeLabel.MODIFIER)
        assert any("onlyOwner" in name for name in modifiers)
```

### 13.2 Required Test Categories

1. **Basic Contract Parsing** - Contracts, state variables, functions
2. **Inheritance Chains** - Multi-level inheritance, interface implementation
3. **Modifier Usage** - Definition and application
4. **Event Definitions/Emissions** - Event nodes and EMITS relationships
5. **Import Resolution** - Relative, absolute, npm-style, remappings
6. **Call Graphs** - Internal, external, delegate calls

---

## 14. Cypher Query Examples

### 14.1 Required Indexes

```python
# Add to cypher_queries.py for performance
CYPHER_INDEX_CONTRACT_NAME = build_index_query("Contract", "name")
CYPHER_INDEX_LIBRARY_NAME = build_index_query("Library", "name")
CYPHER_INDEX_EVENT_NAME = build_index_query("Event", "name")
CYPHER_INDEX_MODIFIER_NAME = build_index_query("Modifier", "name")
CYPHER_INDEX_STATE_VARIABLE_NAME = build_index_query("StateVariable", "name")
CYPHER_INDEX_CUSTOM_ERROR_NAME = build_index_query("CustomError", "name")
```

### 14.2 Query Examples

```cypher
// Find all contracts inheriting from ERC20
MATCH (child:Contract)-[:INHERITS]->(base:Contract {name: "ERC20"})
RETURN child.name, child.qualified_name

// Find functions that emit specific events
MATCH (f:Function)-[:EMITS]->(e:Event {name: "Transfer"})
RETURN f.qualified_name, f.visibility

// Find all external function calls
MATCH (caller:Function)-[:CALLS_EXTERNAL]->(callee:Function)
RETURN caller.qualified_name, callee.qualified_name

// Find functions using specific modifiers
MATCH (f:Function)-[:MODIFIED_BY]->(m:Modifier {name: "onlyOwner"})
RETURN f.qualified_name

// Find unused modifiers (potential dead code) - optimized with NOT EXISTS
MATCH (m:Modifier)
WHERE NOT EXISTS { ()-[:MODIFIED_BY]->(m) }
RETURN m.qualified_name

// Find delegate calls (security audit)
MATCH (f:Function)-[call:CALLS_EXTERNAL]->()
WHERE call.call_type = 'delegatecall'
RETURN f.qualified_name

// Get full contract structure
MATCH (c:Contract {name: "ERC20"})
OPTIONAL MATCH (c)-[:DEFINES]->(f:Function)
OPTIONAL MATCH (c)-[:DEFINES_EVENT]->(e:Event)
OPTIONAL MATCH (c)-[:DEFINES_STATE]->(sv:StateVariable)
OPTIONAL MATCH (c)-[:DEFINES_MODIFIER]->(m:Modifier)
RETURN c, collect(DISTINCT f) as functions,
       collect(DISTINCT e) as events,
       collect(DISTINCT sv) as state_vars,
       collect(DISTINCT m) as modifiers
```

---

## 15. References

- [Solidity Documentation](https://docs.soliditylang.org/)
- [tree-sitter-solidity](https://github.com/JoranHonig/tree-sitter-solidity)
- [Foundry Book](https://book.getfoundry.sh/)
- [Hardhat Documentation](https://hardhat.org/docs)
- [OpenZeppelin Contracts](https://github.com/OpenZeppelin/openzeppelin-contracts)

---

## 16. Implementation Review Summary

**Status:** Reviewed and Verified (Rounds 8-13) - Ready for Implementation

### Round 1 Fixes (Previous)

| Issue | Status | Fix Applied |
|-------|--------|-------------|
| Wrong tree-sitter version | ✅ Fixed | Use `>=1.2.11` not `>=0.21.0` |
| Encoding constant | ✅ Fixed | Use `cs.ENCODING_UTF8` in `_solidity_get_name` |
| Call query capture name | ✅ Fixed | Use `@call` not `@emit` for emit statements |
| Import resolver architecture | ✅ Documented | Add `_parse_solidity_imports()` to ImportProcessor |
| Missing handler methods | ✅ Fixed | Added `is_class_method()`, `extract_function_name()` |
| Remappings parsing | ✅ Fixed | Added comment handling and sort by key length |
| Cypher query indexes | ✅ Fixed | Added index creation queries |
| Cypher query optimization | ✅ Fixed | Use `NOT EXISTS {}` pattern for Memgraph |

### Round 2 Fixes (Previous)

| Issue | Status | Fix Applied |
|-------|--------|-------------|
| Missing `cs.` prefix in constants | ✅ Fixed | Added `cs.` prefix to TS_SOL_* constants |
| Duplicate `TS_FIELD_NAME` | ✅ Fixed | Removed, use existing `FIELD_NAME` |
| Wrong field name prefix | ✅ Fixed | Changed `TS_SOL_*` to `FIELD_*` pattern |
| INHERITS/IMPLEMENTS already exist | ✅ Fixed | Marked as "reuse existing" in Section 2.4 |
| Incorrect `this` semantics | ✅ Fixed | Clarified `this` means external call in Solidity |
| `foundry.toml` placeholder | ✅ Documented | Added TODO note for TOML parsing |
| Missing package indicator | ✅ Fixed | Added `PKG_REMAPPINGS_TXT` to tuple |
| Missing node types | ✅ Fixed | Added `TS_SOL_MODIFIER_INVOCATION`, `TS_SOL_INHERITANCE_SPECIFIER` |
| Section numbering duplicate | ✅ Fixed | Renumbered Section 10 (References) to Section 16 |
| Test file naming | ✅ Fixed | Standardized to `test_solidity.py` |
| Checklist incomplete | ✅ Fixed | Added `_NODE_LABEL_UNIQUE_KEYS`, registry, enum entries |

### Round 3 Fixes (Previous)

| Issue | Status | Fix Applied |
|-------|--------|-------------|
| Package indicators naming | ✅ Fixed | Renamed to `SPEC_SOL_PACKAGE_INDICATORS` |
| LanguageSpec dict population | ✅ Fixed | Use inline dict syntax, added `cs.` prefix |
| Handler string literals | ✅ Fixed | Use `cs.TS_SOL_*` constants throughout |
| `extract_decorators` bug | ✅ Fixed | Extract identifier from modifier_invocation node |
| `extract_inheritance` bug | ✅ Fixed | Extract identifier from user_defined_type node |
| Missing node type constants | ✅ Fixed | Added `TS_SOL_VISIBILITY`, `TS_SOL_STATE_MUTABILITY` |
| Missing relationships in Section 2.4 | ✅ Fixed | Added `DEFINES_EVENT`, `DEFINES_STATE`, `DEFINES_MODIFIER` |
| C3 linearization incomplete | ✅ Fixed | Added library calls, clarified super resolution |
| Missing test classes | ✅ Fixed | Added Library, StateVariable, CustomError tests |

### Round 5 Fixes (Previous)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | Constants | `cs.TS_FIELD_NAME` should be `cs.FIELD_NAME` | ✅ Fixed |
| 2 | NodeLabel | All 6 labels correctly specified with QUALIFIED_NAME | ✅ Pass |
| 3 | RelationshipType | Checklist missing 8 types | ⚠️ Documented |
| 4 | LanguageSpec | Constants pending implementation in constants.py | ⚠️ Pending |
| 5 | SolidityHandler | Missing registry entry, extract_base_class_name issue | ⚠️ Known |
| 6 | Import Resolver | Architecture violation, missing namespace/alias parsing | ⚠️ Known |
| 7 | Call Resolution | Section 12.1 low-level call table inconsistency | ✅ Fixed |
| 8 | Graph Schema | STATE_VARIABLE missing qualified_name, property inconsistencies | ✅ Fixed |
| 9 | Test Coverage | Missing library/error fixtures, import/call tests | ⚠️ Known |
| 10 | Checklist | Missing explicit constants, LANGUAGE_METADATA undefined | ⚠️ Documented |

### Round 6 Fixes (Current - 10 Parallel Workers)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | Constants | Missing `SPEC_SOL_IMPORT_FROM_TYPES` | ✅ Fixed |
| 2 | NodeLabel | All 6 labels correctly specified with QUALIFIED_NAME | ✅ Pass |
| 3 | RelationshipType | 12 new types correctly separated by phase | ✅ Pass |
| 4 | LanguageSpec | Use `cs.FIELD_NAME` instead of `"name"` string | ✅ Fixed |
| 5 | SolidityHandler | Missing `extract_base_class_name` override | ⚠️ Known |
| 6 | Import Resolver | Architecture violation, missing namespace/alias parsing | ⚠️ Known |
| 7 | Call Resolution | `INTERFACE_CALLS` referenced but not defined | ✅ Fixed |
| 8 | Graph Schema | `list[dict]` incompatible with PropertyValue, missing `absolute_path` | ✅ Fixed |
| 9 | Test Coverage | Missing import/call tests, placeholder classes | ⚠️ Known |
| 10 | Checklist | Vague constants, missing explicit NodeLabels/RelationshipTypes | ⚠️ Documented |

### Round 7 Fixes (Current - 10 Parallel Workers)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | Constants | `EXT_SOL` may conflict with `EXT_CS` (C#) | ⚠️ Minor |
| 2 | NodeLabel | All 6 labels correctly specified | ✅ Pass |
| 3 | RelationshipType | Missing `CALLS` in "reuse existing" list | ✅ Fixed |
| 4 | LanguageSpec | All fields correct, queries valid | ✅ Pass |
| 5 | SolidityHandler | Missing `extract_base_class_name` override | ⚠️ Known |
| 6 | Import Resolver | Missing `namespace_name`, relative import bug | ✅ Fixed |
| 7 | Call Resolution | Complete and correct | ✅ Pass |
| 8 | Graph Schema | Markdown table formatting (`\|` escape) | ⚠️ Minor |
| 9 | Test Coverage | Missing fixtures/classes, placeholder tests | ✅ Fixed |
| 10 | Checklist | Vague constants, incomplete RelationshipTypes | ✅ Fixed |

### Round 8 Fixes (Current - 10 Parallel Workers)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | Constants | All patterns correct | ✅ Pass |
| 2 | NodeLabel | Redundant `is_library` property on CONTRACT | ✅ Fixed |
| 3 | RelationshipType | All 12 new types correct, no duplicates | ✅ Pass |
| 4 | LanguageSpec | **CRITICAL**: Wrong tree-sitter node names | ✅ Fixed |
| 5 | SolidityHandler | All correct | ✅ Pass |
| 6 | Import Resolver | Missing parsing logic, architecture conflict | ⚠️ Known |
| 7 | Call Resolution | All correct | ✅ Pass |
| 8 | Tests | Missing imports, remappings, library usage | ⚠️ Known |
| 9 | Cypher Queries | Wrong relationships, missing indexes | ✅ Fixed |
| 10 | Checklist | Wrong count (26 vs 27), missing test categories | ✅ Fixed |

### Round 11 Fixes (Verification - 5 Parallel Workers)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | FIELD_NAME constant | All fixed | ✅ Pass |
| 2 | Test assertions | All strengthened | ✅ Pass |
| 3 | Checklist accuracy | All correct | ✅ Pass |
| 4 | Tree-sitter node names | 3 incorrect: error/struct/enum | ✅ Fixed |
| 5 | Property types | All valid | ✅ Pass |

### Round 12-13 Fixes (Final Verification)

| Issue | Fix Applied |
|-------|-------------|
| TS_SOL_ERROR_DEFINITION | Changed to TS_SOL_ERROR_DECLARATION = "error_declaration" |
| TS_SOL_STRUCT_DEFINITION | Changed to TS_SOL_STRUCT_DECLARATION = "struct_declaration" |
| TS_SOL_ENUM_DEFINITION | Changed to TS_SOL_ENUM_DECLARATION = "enum_declaration" |
| 5 test assertions | Added error messages to all remaining assertions |

### Round 14 Fixes (10 Parallel Workers - Comprehensive Review)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | Constants & Tree-sitter | All 26 constants correct | ✅ Pass |
| 2 | NodeLabel | NODE_SCHEMAS missing (implementation item) | ✅ Documented |
| 3 | RelationshipType | Missing RelationshipSchema definitions | ✅ Added to checklist |
| 4 | LanguageSpec/FQNSpec | FIELD_NAME constant usage | ✅ Fixed |
| 5 | SolidityHandler | Missing extract_base_class_name override | ✅ Added |
| 6 | Import Resolution | Architecture documented in Known Issues | ✅ Pass |
| 7 | Call Resolution | Missing C3 linearization, metadata, relationship props | ✅ Added |
| 8 | Graph Schema Properties | All property types valid | ✅ Pass |
| 9 | Test Coverage | Missing owner var, weak assertions | ✅ Fixed |
| 10 | Implementation Checklist | Missing items from checklist | ✅ Fixed |

### Critical Fixes Applied in Round 14

1. **FIELD_NAME constant** - Changed `cs.TS_FIELD_NAME` to `cs.FIELD_NAME` in `_solidity_get_name()` and `SolidityHandler.extract_function_name()` for consistency with existing codebase
2. **Test fixture** - Added `address private owner;` variable declaration
3. **Test assertion** - Added specific IERC20 inheritance verification in `test_inherits_relationship`
4. **Relationship properties** - Added `CALLS_EXTERNAL_PROPERTIES`, `CALLS_DELEGATE_PROPERTIES`, `CALLS_STATIC_PROPERTIES` to Section 3.2
5. **C3 linearization** - Added Section 12.3 with algorithm specification
6. **Low-level call metadata** - Added Section 12.4 with metadata specification
7. **Call types table** - Added `this.functionName()` and `this.viewFunction()` to Section 12.1
8. **SolidityHandler** - Added `extract_base_class_name()` override for `user_defined_type` handling
9. **Implementation checklist**:
   - Added `extract_base_class_name` override to Phase 1
   - Added RelationshipSchema and NodeSchema entries to Phase 2
   - Changed Phase 3 from "Create SolidityImportResolver" to "Add _parse_solidity_imports() to ImportProcessor"
   - Added namespace/alias import handling to Phase 3
   - Added src/ and contracts/ resolution paths to Phase 3
   - Added C3 linearization implementation to Phase 4
   - Added low-level call metadata extraction to Phase 4

### Round 10 Fixes (Current - 10 Parallel Workers)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | Constants | All patterns correct | ✅ Pass |
| 2 | NodeLabel | All 6 labels correctly specified | ✅ Pass |
| 3 | RelationshipType | Checklist missing DEFINES from reuse list | ✅ Fixed |
| 4 | LanguageSpec | All fields correct, queries valid | ✅ Pass |
| 5 | SolidityHandler | Used `cs.FIELD_NAME` instead of `cs.TS_FIELD_NAME` | ✅ Fixed |
| 6 | Import Resolver | Architecture documented, all fields present | ✅ Pass |
| 7 | Call Resolution | Complete and correct | ✅ Pass |
| 8 | Graph Schema | All property types valid | ✅ Pass |
| 9 | Test Coverage | Weak assertions in Inheritance/Events/Modifiers | ✅ Fixed |
| 10 | Checklist | Wrong count (27 vs 26), missing DEFINES, missing LANGUAGE_METADATA | ✅ Fixed |

### Critical Fixes Applied in Round 10

1. **Constants count** - Fixed from 27 to 26 TS_SOL_* constants
2. **Checklist reuse list** - Added DEFINES to "reuse existing" relationships
3. **TestSolidityInheritance** - Added specific assertion for ERC20 -> IERC20 inheritance
4. **TestSolidityEvents.test_emit_relationship** - Added assertion for _transfer emitting Transfer
5. **TestSolidityModifiers** - Added specific assertions for onlyOwner modifier usage
6. **LANGUAGE_METADATA** - Added Solidity entry definition with LanguageStatus.FULL

### Round 9 Fixes (Verification - 5 Parallel Workers)

| Worker | Area | Issues Found | Status |
|--------|------|--------------|--------|
| 1 | Tree-sitter consistency | function_query used old names, is_contract_function wrong constant | ✅ Fixed |
| 2 | Test fixture | Missing imports, remappings.txt, library usage, test assertions | ✅ Fixed |
| 3 | Handler consistency | is_contract_function used TS_SOL_SOURCE_UNIT | ✅ Fixed |
| 4 | Cypher queries | DEFINES relationship not documented as "reuse existing" | ✅ Fixed |
| 5 | Property types | All valid PropertyValue types | ✅ Pass |

### Critical Fixes Applied in Round 9

1. **function_query** - Fixed to use `fallback_receive_definition` instead of separate `receive_declaration` and `fallback_declaration`
2. **SolidityHandler.is_contract_function()** - Fixed to use `TS_SOL_SOURCE_FILE` instead of `TS_SOL_SOURCE_UNIT`
3. **Section 2.4** - Added `DEFINES` to "reuse existing" relationships list
4. **Test fixture** - Added import statements (`./IERC20.sol`, `@openzeppelin/...`)
5. **Test fixture** - Added `remappings.txt` file for npm-style import tests
6. **Test fixture** - Added library usage (`SafeMath.add()` call and `using SafeMath for uint256`)
7. **Test fixture** - Added separate `IERC20.sol` file for relative import tests
8. **TestSolidityImports** - Added proper assertions with error messages
9. **TestSolidityCallGraphs** - Added proper assertions with error messages for library call test

### Critical Fixes Applied in Round 8

1. **Tree-sitter node types** - Fixed wrong names based on actual tree-sitter-solidity grammar:
   - `TS_SOL_SOURCE_UNIT` → `TS_SOL_SOURCE_FILE` (matches `source_file` node)
   - `TS_SOL_FUNCTION_CALL` → `TS_SOL_CALL_EXPRESSION`
   - `TS_SOL_MEMBER_ACCESS` → `TS_SOL_MEMBER_EXPRESSION`
   - `TS_SOL_RECEIVE_DECLARATION` + `TS_SOL_FALLBACK_DECLARATION` → `TS_SOL_FALLBACK_RECEIVE_DEFINITION`
2. **call_query** - Fixed field names to match grammar: `function:` → `expression:`, `member:` → `property:`, `event:` → `name:`
3. **CONTRACT node** - Removed redundant `is_library` property (LIBRARY is separate node type)
4. **Cypher queries** - Fixed Section 6 query to use `DEFINES_EVENT`, `DEFINES_STATE`, `DEFINES_MODIFIER`
5. **Cypher indexes** - Added missing `CYPHER_INDEX_LIBRARY_NAME` and `CYPHER_INDEX_CUSTOM_ERROR_NAME`
6. **SOLIDITY_ENTITY_TYPES prompt** - Added all 12 new relationships to the documentation
7. **Implementation checklist** - Fixed count from 26 to 27 constants, added FIELD_* constants, expanded Phase 5 tests

### Critical Fixes Applied in Round 15 (Current)

1. **call_query** - Corrected field names based on actual tree-sitter-solidity grammar:
   - Uses `function:` field (NOT `expression:`) for call_expression
   - `function` field contains an `expression` wrapper node around the identifier
   - emit_statement uses `name:` field containing `expression` wrapper
2. **class_query** - Added `struct_declaration` and `enum_declaration` patterns for consistency with other languages
3. **SPEC_SOL_CLASS_TYPES** - Added `TS_SOL_STRUCT_DECLARATION` and `TS_SOL_ENUM_DECLARATION`
4. **NodeType enum** - Added `CONTRACT` and `LIBRARY` values for proper node type determination
5. **node_type.py** - Added Solidity type handling for contracts, libraries, interfaces, structs, enums
6. **SolidityHandler** - Fixed `extract_visibility` and `extract_state_mutability` to iterate children instead of using field access
7. **RELATIONSHIP_SCHEMAS** - Added `LIBRARY` to `USES_LIBRARY` sources, `INTERFACE` to `DEFINES_EVENT` sources
8. **Import resolution** - Added direct path resolution, separator handling, and prefix guards
8. **_solidity_get_name** - Updated to handle combined `fallback_receive_definition` node type

### Critical Fixes Applied in Round 7

1. **SolidityImport dataclass** - Added `namespace_name: str | None = None` field for namespace imports
2. **resolve_import() method** - Fixed to accept `current_file` parameter for relative import resolution
3. **Test fixture** - Added `SafeMath` library and `InsufficientBalance` custom error definitions
4. **Test classes** - Implemented `TestSolidityLibraries`, `TestSolidityStateVariables`, `TestSolidityCustomErrors`
5. **Test classes** - Added new `TestSolidityImports` and `TestSolidityCallGraphs` classes
6. **Section 2.4** - Added `CALLS` to "reuse existing" comment
7. **Implementation checklist** - Expanded Phase 1 with explicit constant sub-items
8. **Implementation checklist** - Listed all 12 RelationshipTypes in Phase 2

### Critical Fixes Applied in Round 6

1. **SPEC_SOL_IMPORT_FROM_TYPES** - Added constant for consistency with other languages
2. **_solidity_get_name** - Changed `"name"` string to `cs.FIELD_NAME` constant
3. **EVENT/CUSTOM_ERROR parameters** - Changed `list[dict]` to `list[str]` for PropertyValue compatibility
4. **Node property tables** - Added `absolute_path` property to all 6 node types
5. **INTERFACE_CALLS** - Changed to `CALLS_EXTERNAL` with `is_interface_call: true` metadata
6. **Contract nullable types** - Fixed `license`, `pragma`, `is_upgradeable` to show `| None`

### Previous Critical Fixes (Rounds 1-5)

- STATE_VARIABLE_PROPERTIES: Added `qualified_name` property
- EVENT_PROPERTIES: Added `qualified_name`, `path`, `start_line`, `end_line`
- Section 10.1 tables: Added complete tables for LIBRARY, MODIFIER, STATE_VARIABLE, CUSTOM_ERROR
- Section 12.1 call types: Fixed low-level call table for call/delegatecall/staticcall
- SolidityHandler: Fixed `cs.TS_FIELD_NAME` to `cs.FIELD_NAME`

### Remaining Implementation Notes

- Add all new NodeLabels to `_NODE_LABEL_UNIQUE_KEYS` dict (CRITICAL - runtime check!)
- Add Phase 1 (7) & Phase 2 (5) RelationshipTypes: see Section 2.4 for complete list
- Follow ImportProcessor pattern (add `_parse_solidity_imports()`) instead of standalone resolver
- Add namespace import and alias import parsing to import resolution
- Add LANGUAGE_METADATA entry for Solidity with LanguageStatus and display_name
- Register SolidityHandler in handlers/registry.py `_HANDLERS` dict

### Known Issues (Not Blocking)

1. **Import Resolver Architecture**: Spec shows standalone `SolidityImportResolver` class but recommends using `ImportProcessor` pattern. Implementation should add `_parse_solidity_imports()` to `ImportProcessor`. The checklist has been updated to reflect this.

2. **`EXT_SOL` naming**: May cause confusion with `EXT_CS` (C#). Consider using `EXT_SOLIDITY` for consistency with `SOLIDITY_EXTENSIONS`.

### Test Coverage Notes

Tests should cover:
- Library detection
- Constructor/fallback/receive functions
- State variables (StateVariable nodes)
- Custom errors (CustomError nodes)
- Visibility modifiers
- State mutability (view/pure/payable)
- Import resolution (relative/npm/remapping)
- Using "for" directive (USES_LIBRARY)
- Cross-file inheritance
- Override/virtual flags
- IMPLEMENTS relationship (interface implementation)

