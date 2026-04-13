from pathlib import Path

from .. import constants as cs
from ..config import settings


def should_skip_path(
    path: Path,
    repo_path: Path,
    exclude_paths: frozenset[str] | None = None,
    unignore_paths: frozenset[str] | None = None,
) -> bool:
    if path.is_file() and path.suffix in cs.IGNORE_SUFFIXES:
        return True
    rel_path = path.relative_to(repo_path)
    rel_path_str = rel_path.as_posix()
    dir_parts = rel_path.parent.parts if path.is_file() else rel_path.parts
    if exclude_paths and (
        not exclude_paths.isdisjoint(dir_parts)
        or rel_path_str in exclude_paths
        or any(rel_path_str.startswith(f"{p}/") for p in exclude_paths)
    ):
        return True
    if unignore_paths and any(
        rel_path_str == p or rel_path_str.startswith(f"{p}/") for p in unignore_paths
    ):
        return False
    return not cs.IGNORE_PATTERNS.isdisjoint(dir_parts)


def get_all_code_files(
    repo_path: Path,
    exclude_paths: frozenset[str] | None = None,
    unignore_paths: frozenset[str] | None = None,
) -> list[Path]:
    code_files = []
    for path in repo_path.rglob("*"):
        if path.is_file() and not should_skip_path(
            path, repo_path, exclude_paths, unignore_paths
        ):
            code_files.append(path)
    return code_files


def is_path_allowed(path: Path, project_root: Path) -> bool:
    """Check if a path is allowed for file operations.

    Args:
        path: Path to check
        project_root: Original project root path

    Returns:
        True if path is allowed, False otherwise
    """

    # Resolve both paths to absolute paths, follow symlinks
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()

    # Global access enabled: all paths are allowed
    if settings.ENABLE_GLOBAL_FILE_ACCESS:
        return True

    # Check if path is inside project root
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def is_path_outside_project_root(path: Path, project_root: Path) -> bool:
    """Check if a path is outside the project root, regardless of global access setting.

    Args:
        path: Path to check
        project_root: Original project root path

    Returns:
        True if path is outside project root, False otherwise
    """
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()

    try:
        resolved_path.relative_to(resolved_root)
        return False
    except ValueError:
        return True
