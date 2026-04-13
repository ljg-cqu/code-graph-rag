from pathlib import Path

from .. import constants as cs


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
