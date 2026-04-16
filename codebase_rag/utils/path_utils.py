from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from .. import constants as cs
from ..config import settings


def _directory_candidates(rel_path: PurePosixPath, *, is_dir: bool) -> tuple[str, ...]:
    directory_path = rel_path if is_dir else rel_path.parent
    if not directory_path.parts:
        return ()

    return tuple(
        PurePosixPath(*directory_path.parts[: index + 1]).as_posix()
        for index in range(len(directory_path.parts))
    )


def _matches_ignore_pattern(
    rel_path: PurePosixPath,
    pattern: str,
    *,
    is_dir: bool,
) -> bool:
    normalized = pattern.strip().replace("\\", "/")
    if not normalized:
        return False

    is_anchored = normalized.startswith("/")
    if is_anchored:
        normalized = normalized.lstrip("/")

    is_directory_pattern = normalized.endswith("/")
    if is_directory_pattern:
        normalized = normalized.rstrip("/")

    if not normalized:
        return False

    rel_path_str = rel_path.as_posix()
    directory_candidates = _directory_candidates(rel_path, is_dir=is_dir)

    if is_directory_pattern:
        if is_anchored:
            return any(
                fnmatchcase(candidate, normalized)
                for candidate in directory_candidates
            )

        return any(
            fnmatchcase(candidate, normalized)
            or fnmatchcase(PurePosixPath(candidate).name, normalized)
            for candidate in directory_candidates
        )

    candidates = {rel_path_str}
    if not is_anchored:
        candidates.add(rel_path.name)
        candidates.update(rel_path.parts)
        candidates.update(directory_candidates)

    return any(fnmatchcase(candidate, normalized) for candidate in candidates)


def should_skip_path(
    path: Path,
    repo_path: Path,
    exclude_paths: frozenset[str] | None = None,
    unignore_paths: frozenset[str] | None = None,
) -> bool:
    if path.is_file() and path.suffix in cs.IGNORE_SUFFIXES:
        return True
    rel_path = path.relative_to(repo_path)
    pure_rel_path = PurePosixPath(rel_path.as_posix())
    if exclude_paths and any(
        _matches_ignore_pattern(pure_rel_path, pattern, is_dir=path.is_dir())
        for pattern in exclude_paths
    ):
        return True
    if unignore_paths and any(
        _matches_ignore_pattern(pure_rel_path, pattern, is_dir=path.is_dir())
        for pattern in unignore_paths
    ):
        return False
    dir_parts = rel_path.parent.parts if path.is_file() else rel_path.parts
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
