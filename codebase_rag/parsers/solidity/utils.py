"""Solidity import resolution utilities.

Architecture Note: The main import resolution should integrate with the existing
ImportProcessor pattern in codebase_rag/parsers/import_processor.py.
Add methods like _parse_solidity_imports() and _resolve_solidity_import_path()
to ImportProcessor rather than using this standalone module.

This module provides utility functions that can be called by ImportProcessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SolidityImport:
    """Represents a resolved Solidity import."""

    path: str
    symbols: list[str]  # Imported symbols
    aliases: dict[str, str]  # Symbol -> alias mapping
    namespace_name: str | None = None  # For namespace imports (import * as X)
    is_absolute: bool = True
    remapped_path: str | None = None  # After remapping


def load_remappings(repo_path: Path) -> dict[str, str]:
    """Load remappings from foundry.toml or remappings.txt.

    Args:
        repo_path: Root path of the Solidity project

    Returns:
        Dictionary mapping import prefixes to remapped paths
    """
    remappings: dict[str, str] = {}

    # Check for remappings.txt
    remappings_file = repo_path / "remappings.txt"
    if remappings_file.exists():
        for line in remappings_file.read_text().splitlines():
            # Remove comments and strip whitespace
            line = line.split("#")[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                remappings[key.strip()] = value.strip()

    # Sort by key length descending (most specific first)
    return dict(
        sorted(remappings.items(), key=lambda x: len(x[0]), reverse=True)
    )


def extract_solidity_import_info(import_path: str) -> SolidityImport:
    """Extract import information from a Solidity import path.

    Args:
        import_path: The import path string from the import statement

    Returns:
        SolidityImport with parsed information
    """
    # Basic implementation - full parsing would require AST inspection
    return SolidityImport(
        path=import_path,
        symbols=[],
        aliases={},
        namespace_name=None,
        is_absolute=not import_path.startswith(("./", "../")),
    )


def resolve_solidity_import_path(
    import_path: str,
    repo_path: Path,
    current_file: Path | None = None,
    remappings: dict[str, str] | None = None,
) -> Path | None:
    """Resolve import path to actual file path.

    Args:
        import_path: The import path from the import statement
        repo_path: Root path of the Solidity project
        current_file: The file containing the import (needed for relative imports)
        remappings: Optional remappings dict (loaded from remappings.txt if not provided)

    Returns:
        Resolved file path or None if not found
    """
    # Handle empty or whitespace-only import paths
    if not import_path or not import_path.strip():
        return None

    if remappings is None:
        remappings = load_remappings(repo_path)
    else:
        # Sort by key length descending (most specific first) even when passed directly
        remappings = dict(
            sorted(remappings.items(), key=lambda x: len(x[0]), reverse=True)
        )

    # Apply remappings
    for key, value in remappings.items():
        if import_path.startswith(key):
            # Ensure separator between value and remaining path
            remaining = import_path[len(key) :]
            if remaining and not value.endswith("/") and not remaining.startswith("/"):
                import_path = value + "/" + remaining
            else:
                import_path = value + remaining
            break

    # Try direct resolution first (important after remapping)
    direct_path = repo_path / import_path
    if direct_path.exists():
        return direct_path
    direct_path_sol = direct_path.with_suffix(".sol")
    if direct_path_sol.exists():
        return direct_path_sol

    # Try relative resolution (needs current file context)
    if import_path.startswith("./") or import_path.startswith("../"):
        if current_file is None:
            return None
        base_dir = current_file.parent
        resolved = (base_dir / import_path).resolve()
        if resolved.exists():
            return resolved
        # Try with .sol extension
        resolved_sol = resolved.with_suffix(".sol")
        if resolved_sol.exists():
            return resolved_sol
        return None

    # Try node_modules (only if not already resolved via remapping)
    if not import_path.startswith(("lib/", "src/", "contracts/")):
        node_modules = repo_path / "node_modules" / import_path
        if node_modules.exists():
            return node_modules
        node_modules_sol = node_modules.with_suffix(".sol")
        if node_modules_sol.exists():
            return node_modules_sol

        # Try lib (Foundry)
        lib_path = repo_path / "lib" / import_path
        if lib_path.exists():
            return lib_path
        lib_path_sol = lib_path.with_suffix(".sol")
        if lib_path_sol.exists():
            return lib_path_sol

    # Try src directory (if not already in src/)
    if not import_path.startswith("src/"):
        src_path = repo_path / "src" / import_path
        if src_path.exists():
            return src_path
        src_path_sol = src_path.with_suffix(".sol")
        if src_path_sol.exists():
            return src_path_sol

    # Try contracts directory (Hardhat, if not already in contracts/)
    if not import_path.startswith("contracts/"):
        contracts_path = repo_path / "contracts" / import_path
        if contracts_path.exists():
            return contracts_path
        contracts_path_sol = contracts_path.with_suffix(".sol")
        if contracts_path_sol.exists():
            return contracts_path_sol

    return None
