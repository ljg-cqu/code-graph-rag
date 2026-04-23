from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from pathlib import Path

from loguru import logger
from codebase_rag.compat.pydantic_ai import Tool

from .. import logs as ls
from .. import tool_errors as te
from ..config import settings
from ..schemas import PythonObjectInfo
from . import tool_descriptions as td


# Targeted dunders blocklist — NOT a blanket "__" ban.
# Blocks known-dangerous patterns while allowing legitimate ones like __init__, __main__.
_BLOCKED_DUNDERS: frozenset[str] = frozenset({
    "__import__",
    "__subclasses__",
    "__builtins__",
    "__loader__",
    "__reduce__",
    "__reduce_ex__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__globals__",
    "__code__",
    "__closure__",
    "__defaults__",
    "__bases__",
    "__class__",
    "__dict__",
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

        logger.info(ls.PYTHON_INSPECT_SEARCH.format(path=object_path))

        # Run blocking importlib/inspect work in a thread with timeout to avoid blocking the event loop
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._inspect_sync, object_path),
                timeout=settings.PYTHON_INSPECT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(ls.PYTHON_INSPECT_FAILED.format(path=object_path, error="timeout"))
            return PythonObjectInfo(
                object_path=object_path,
                error_message=f"Inspection timed out after {settings.PYTHON_INSPECT_TIMEOUT} seconds",
            )

    def _inspect_sync(self, object_path: str) -> PythonObjectInfo:
        try:
            # Import and traverse
            if "." in object_path:
                module_name, attr_name = object_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                obj = getattr(module, attr_name)
            else:
                # Try as module first, then as builtin
                try:
                    obj = importlib.import_module(object_path)
                    module_name = object_path
                except ModuleNotFoundError:
                    # Check builtins
                    import builtins
                    obj = getattr(builtins, object_path, None)
                    if obj is None:
                        raise
                    module_name = "builtins"

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

            logger.info(ls.PYTHON_INSPECT_SUCCESS.format(path=object_path))

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
            logger.warning(ls.PYTHON_INSPECT_FAILED.format(path=object_path, error=e))
            return PythonObjectInfo(
                object_path=object_path,
                error_message=f"Module not found: {e}",
            )
        except AttributeError as e:
            logger.warning(ls.PYTHON_INSPECT_FAILED.format(path=object_path, error=e))
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
