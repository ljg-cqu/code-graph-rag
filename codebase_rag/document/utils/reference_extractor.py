"""Code reference extraction utilities.

Extracts function/class/module references from document content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ReferenceType(StrEnum):
    """Type of code reference."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    VARIABLE = "variable"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


@dataclass
class CodeReference:
    """A extracted code reference."""

    name: str
    qualified_name: str  # May be same as name if no module path
    reference_type: ReferenceType
    context: str  # Surrounding context
    line_number: int  # Approximate line in document


def extract_code_references(content: str) -> list[CodeReference]:
    """
    Extract code references from document content.

    Uses deterministic patterns (regex), not LLM.

    Args:
        content: Document content

    Returns:
        List of extracted code references
    """
    references: list[CodeReference] = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        # Find all potential references on this line
        refs = _extract_references_from_line(line, i)
        references.extend(refs)

    # Deduplicate by qualified_name
    seen: set[str] = set()
    unique_refs: list[CodeReference] = []
    for ref in references:
        if ref.qualified_name not in seen:
            seen.add(ref.qualified_name)
            unique_refs.append(ref)

    return unique_refs


def _extract_references_from_line(line: str, line_num: int) -> list[CodeReference]:
    """Extract references from a single line."""
    refs: list[CodeReference] = []

    # Pattern 1: Backtick-quoted function calls
    for match in re.finditer(r"`([a-zA-Z_][\w]*(?:\.[\w]+)*)\(\)`", line):
        name = match.group(1)
        refs.append(
            CodeReference(
                name=name.split(".")[-1],
                qualified_name=name,
                reference_type=ReferenceType.FUNCTION,
                context=line.strip(),
                line_number=line_num,
            )
        )

    # Pattern 2: "class `ClassName`" references
    for match in re.finditer(r"class\s+`(\w+)`", line, re.IGNORECASE):
        name = match.group(1)
        refs.append(
            CodeReference(
                name=name,
                qualified_name=name,
                reference_type=ReferenceType.CLASS,
                context=line.strip(),
                line_number=line_num,
            )
        )

    # Pattern 3: "method `MethodName`" references
    for match in re.finditer(r"method\s+`(\w+)`", line, re.IGNORECASE):
        name = match.group(1)
        refs.append(
            CodeReference(
                name=name,
                qualified_name=name,
                reference_type=ReferenceType.METHOD,
                context=line.strip(),
                line_number=line_num,
            )
        )

    # Pattern 4: "function `funcName`" references
    for match in re.finditer(r"function\s+`(\w+)`", line, re.IGNORECASE):
        name = match.group(1)
        refs.append(
            CodeReference(
                name=name,
                qualified_name=name,
                reference_type=ReferenceType.FUNCTION,
                context=line.strip(),
                line_number=line_num,
            )
        )

    # Pattern 5: "module `module.path`" references
    for match in re.finditer(r"module\s+`([a-zA-Z_][\w]*(?:\.[\w]+)*)`", line, re.IGNORECASE):
        name = match.group(1)
        refs.append(
            CodeReference(
                name=name.split(".")[-1],
                qualified_name=name,
                reference_type=ReferenceType.MODULE,
                context=line.strip(),
                line_number=line_num,
            )
        )

    # Pattern 6: Generic backtick-quoted identifiers
    for match in re.finditer(r"`([a-zA-Z_][\w]*(?:\.[\w]+)*)`", line):
        name = match.group(1)
        # Skip if already captured by other patterns
        if not any(r.qualified_name == name for r in refs):
            refs.append(
                CodeReference(
                    name=name.split(".")[-1],
                    qualified_name=name,
                    reference_type=ReferenceType.UNKNOWN,
                    context=line.strip(),
                    line_number=line_num,
                )
            )

    return refs


def resolve_reference_to_fqn(
    reference: CodeReference,
    known_fqns: set[str],
) -> str | None:
    """
    Try to resolve a reference to a known fully qualified name.

    Args:
        reference: The extracted reference
        known_fqns: Set of known FQNs from code graph

    Returns:
        Resolved FQN or None if not found
    """
    # If already a potential FQN (has dots), check directly
    if "." in reference.qualified_name:
        if reference.qualified_name in known_fqns:
            return reference.qualified_name

    # Try to match by short name
    for fqn in known_fqns:
        if fqn.endswith(f".{reference.name}") or fqn == reference.name:
            return fqn

    return None


def extract_import_like_references(content: str) -> list[str]:
    """
    Extract import-like patterns from content.

    Looks for patterns like:
    - import module.path
    - from module import name

    Args:
        content: Document content

    Returns:
        List of module/import paths
    """
    refs: list[str] = []

    # Pattern: import X
    for match in re.finditer(r"import\s+([a-zA-Z_][\w]*(?:\.[\w]+)*)", content):
        refs.append(match.group(1))

    # Pattern: from X import Y
    for match in re.finditer(r"from\s+([a-zA-Z_][\w]*(?:\.[\w]+)*)\s+import", content):
        refs.append(match.group(1))

    return refs


__all__ = [
    "ReferenceType",
    "CodeReference",
    "extract_code_references",
    "resolve_reference_to_fqn",
    "extract_import_like_references",
]