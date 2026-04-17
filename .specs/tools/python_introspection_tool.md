# Python Introspection Tool — Implementation Specification

## Problem

The LLM agent needs to inspect Python modules, packages, and objects at runtime
but is blocked by shell-command restrictions:

| Blocked command | Reason |
|----------------|--------|
| `python3 -c "import mgclient; help(mgclient.connect)"` | `python3` not in allowlist |
| `pip show mgclient` | `pip` not in allowlist |
| `python3 -c "import inspect; inspect.signature(mgclient.connect)"` | `python3` not in allowlist |

Shell workarounds are fragile (parse unstructured text) and unsafe (would require
expanding the allowlist). A native Python API tool eliminates both problems.

## Design

### Single tool: `inspect_python_object`

One tool with a single parameter covers all use cases. Complexity (detail levels,
operations) is unnecessary — the tool always returns everything it can safely extract.

```python
async def inspect_python_object(object_path: str) -> str
```

**Parameter**: `object_path` — fully-qualified Python path
(`mgclient`, `mgclient.connect`, `codebase_rag.main`).

**Returns**: Plain string (matching existing tool pattern — see `create_file_reader_tool`,
`create_code_retrieval_tool`). On success, formatted text with type, signature, docstring,
file path, and members. On failure, `Error: <message>` (matching `te.ERROR_WRAPPER`).

### PythonObjectInfo schema (in `schemas.py`)

```python
class PythonObjectInfo(BaseModel):
    object_path: str
    object_type: str | None = None   # "module", "function", "class", "method", "builtin_function", "object"
    name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    members: list[str] | None = None  # public attributes for modules/classes
    is_builtin: bool = False
    error_message: str | None = None

    @model_validator(mode="after")
    def _set_success_on_error(self) -> PythonObjectInfo:
        if self.error_message is not None:
            # error_message presence signals failure (follows EditResult/FileCreationResult pattern)
            pass
        return self
```

### Core class: `PythonObjectInspector` (in `tools/python_inspector.py`)

```python
from __future__ import annotations

import asyncio
import inspect
import importlib
import re
from pathlib import Path

from loguru import logger
from pydantic_ai import Tool

from .. import logs as ls
from .. import tool_errors as te
from ..schemas import PythonObjectInfo
from . import tool_descriptions as td


# Targeted dunders blocklist — NOT a blanket "__" ban.
# Blocks known-dangerous patterns while allowing legitimate ones like __init__, __main__.
_BLOCKED_DUNDERS: frozenset[str] = frozenset({
    "__import__",
    "__subclasses__",
    "__builtins__",
    "__builtins__",
    "__loader__",
    "__reduce__",
    "__reduce_ex__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
})


def _validate_object_path(object_path: str) -> str | None:
    """Return an error message if path is unsafe, None if OK."""
    if not object_path or not object_path.strip():
        return "Empty object path."
    if ".." in object_path:
        return "Path traversal ('..') is not allowed."
    if object_path.startswith("/") or object_path.startswith("\\"):
        return "Absolute paths are not allowed."
    # Check each dotted segment against targeted blocklist
    for segment in object_path.split("."):
        if segment in _BLOCKED_DUNDERS:
            return f"Access to '{segment}' is blocked for security."
    # Validate characters: must be valid Python identifiers plus dots
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", object_path):
        return f"Invalid Python object path: '{object_path}'."
    return None


def _get_object_type(obj: object) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isbuiltin(obj):
        return "builtin_function"
    if inspect.isfunction(obj):
        return "function"
    if inspect.isclass(obj):
        return "class"
    if inspect.ismethod(obj):
        return "method"
    return "object"


class PythonObjectInspector:
    __slots__ = ("project_root",)

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        if self.project_root.is_file():
            self.project_root = self.project_root.parent

    async def inspect(self, object_path: str) -> PythonObjectInfo:
        # Validation
        validation_error = _validate_object_path(object_path)
        if validation_error:
            return PythonObjectInfo(
                object_path=object_path,
                error_message=validation_error,
            )

        # Run blocking importlib/inspect work in a thread to avoid blocking the event loop
        return await asyncio.to_thread(self._inspect_sync, object_path)

    def _inspect_sync(self, object_path: str) -> PythonObjectInfo:
        try:
            # Import and traverse
            if "." in object_path:
                module_name, attr_name = object_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                obj = getattr(module, attr_name)
            else:
                obj = importlib.import_module(object_path)
                module_name = object_path

            # Determine file path
            file_path = None
            try:
                file_path = inspect.getfile(obj)
            except (TypeError, OSError):
                # Built-in or C extension — no source file
                pass

            # Signature (only for callables)
            signature = None
            if callable(obj):
                try:
                    signature = str(inspect.signature(obj))
                except (ValueError, TypeError):
                    pass

            # Docstring
            docstring = inspect.getdoc(obj)

            # Source location
            line_start = None
            line_end = None
            try:
                src_lines, line_start = inspect.getsourcelines(obj)
                line_end = line_start + len(src_lines) - 1
            except (OSError, TypeError):
                pass

            # Public members (modules and classes only)
            members = None
            if inspect.ismodule(obj) or inspect.isclass(obj):
                members = sorted(
                    name for name in dir(obj) if not name.startswith("_")
                )

            return PythonObjectInfo(
                object_path=object_path,
                object_type=_get_object_type(obj),
                name=getattr(obj, "__name__", object_path.rsplit(".", 1)[-1]),
                signature=signature,
                docstring=docstring,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                members=members,
                is_builtin=inspect.isbuiltin(obj),
            )

        except ModuleNotFoundError as e:
            return PythonObjectInfo(
                object_path=object_path,
                error_message=f"Module not found: {e}",
            )
        except AttributeError as e:
            return PythonObjectInfo(
                object_path=object_path,
                error_message=f"Object not found in module: {e}",
            )
        except Exception as e:
            logger.exception(f"Unexpected error inspecting '{object_path}': {e}")
            return PythonObjectInfo(
                object_path=object_path,
                error_message=f"Unexpected error: {e}",
            )


def create_inspect_python_object_tool(inspector: PythonObjectInspector) -> Tool:
    async def inspect_python_object(object_path: str) -> str:
        result = await inspector.inspect(object_path)
        if result.error_message:
            return te.ERROR_WRAPPER.format(message=result.error_message)

        lines = [f"Object: {result.object_path}"]
        lines.append(f"Type: {result.object_type}")
        if result.signature:
            lines.append(f"Signature: {result.signature}")
        if result.docstring:
            lines.append(f"Docstring:\n{result.docstring}")
        if result.file_path:
            location = result.file_path
            if result.line_start:
                location += f":{result.line_start}-{result.line_end}"
            lines.append(f"Location: {location}")
        if result.members:
            lines.append(f"Members ({len(result.members)}): {', '.join(result.members)}")
        return "\n".join(lines)

    return Tool(
        function=inspect_python_object,
        name=td.AgenticToolName.INSPECT_PYTHON_OBJECT,
        description=td.INSPECT_PYTHON_OBJECT,
    )
```

## Integration

### 1. Add schema to `schemas.py`

Append the `PythonObjectInfo` class shown above (without the docstring comment).

### 2. Register tool name in `tools/tool_descriptions.py`

```python
# In AgenticToolName enum:
INSPECT_PYTHON_OBJECT = "inspect_python_object"

# In AGENTIC_TOOLS dict:
AgenticToolName.INSPECT_PYTHON_OBJECT: INSPECT_PYTHON_OBJECT_DESC,
```

Add description constant:

```python
INSPECT_PYTHON_OBJECT_DESC = (
    "Inspect Python modules, classes, functions, and objects at runtime. "
    "Provide a fully-qualified Python path (e.g., 'mgclient.connect', "
    "'codebase_rag.config.settings'). Returns type, signature, docstring, "
    "file location, and public members."
)
```

### 3. Wire up in `main.py` (`_initialize_services_and_agent`)

```python
from .tools.python_inspector import PythonObjectInspector, create_inspect_python_object_tool

# After existing tool initializations:
python_inspector = PythonObjectInspector(project_root=repo_path)
inspect_python_tool = create_inspect_python_object_tool(python_inspector)
tools.append(inspect_python_tool)
```

### 4. Optionally add MCP tool name in `constants.py`

```python
# In MCPToolName enum:
INSPECT_PYTHON_OBJECT = "inspect_python_object"
```

### 5. Optionally add MCP tool description in `tools/tool_descriptions.py`

```python
MCP_INSPECT_PYTHON_OBJECT = INSPECT_PYTHON_OBJECT_DESC

# In MCP_TOOLS dict:
MCPToolName.INSPECT_PYTHON_OBJECT: MCP_INSPECT_PYTHON_OBJECT,
```

### 6. Add log constants in `logs.py`

```python
# (H) Python inspector logs
PYTHON_INSPECT_INIT = "PythonObjectInspector initialized with root: {root}"
PYTHON_INSPECT_SEARCH = "Inspecting Python object: {path}"
PYTHON_INSPECT_SUCCESS = "Successfully inspected: {path}"
PYTHON_INSPECT_FAILED = "Failed to inspect '{path}': {error}"
```

## Security Model

- **Read-only**: Auto-approved, no user confirmation needed.
- **No arbitrary code execution beyond imports**: `importlib.import_module()` executes
  module-level code (including `__init__.py`) at import time — this is inherent to Python's
  import system and identical to what happens when any module is imported in the application.
- **Targeted dunders blocklist**: Blocks known-dangerous patterns (`__import__`,
  `__subclasses__`, `__builtins__`, `__reduce__`) while allowing legitimate dunders
  (`__init__`, `__main__`, `__version__`).
- **Path validation**: Blocks `..`, absolute paths, and invalid Python identifiers.
- **Thread offloading**: Uses `asyncio.to_thread()` to avoid blocking the event loop
  during import and inspection.

## Use Cases

| Blocked shell command | Tool call |
|----------------------|-----------|
| `python3 -c "import mgclient; help(mgclient.connect)"` | `inspect_python_object("mgclient.connect")` |
| `pip show mgclient` | `inspect_python_object("mgclient")` — returns `file_path` showing install location |
| `python3 -c "print(dir(mgclient))"` | `inspect_python_object("mgclient")` — returns `members` list |
| `python3 -c "import inspect; inspect.signature(fn)"` | `inspect_python_object("module.fn")` — returns `signature` |

## Testing

```python
def test_inspect_builtin():
    inspector = PythonObjectInspector(".")
    result = asyncio.run(inspector.inspect("len"))
    assert result.error_message is None
    assert result.object_type == "builtin_function"
    assert result.signature == "(obj, /)"

def test_inspect_project_module():
    inspector = PythonObjectInspector(".")
    result = asyncio.run(inspector.inspect("codebase_rag.config"))
    assert result.error_message is None
    assert result.object_type == "module"
    assert result.members is not None
    assert "settings" in result.members

def test_inspect_nonexistent():
    inspector = PythonObjectInspector(".")
    result = asyncio.run(inspector.inspect("nonexistent.module"))
    assert result.error_message is not None
    assert "Module not found" in result.error_message

def test_rejects_traversal():
    inspector = PythonObjectInspector(".")
    result = asyncio.run(inspector.inspect("../etc/passwd"))
    assert result.error_message is not None
    assert "traversal" in result.error_message.lower()

def test_rejects_blocked_dunder():
    inspector = PythonObjectInspector(".")
    result = asyncio.run(inspector.inspect("os.__import__"))
    assert result.error_message is not None
    assert "blocked" in result.error_message.lower()

def test_allows_legitimate_dunder():
    inspector = PythonObjectInspector(".")
    result = asyncio.run(inspector.inspect("codebase_rag.__init__"))
    # __init__ is not in the blocklist — should succeed or fail with
    # "no attribute" (not a security rejection)
    assert "blocked" not in (result.error_message or "").lower()
```
