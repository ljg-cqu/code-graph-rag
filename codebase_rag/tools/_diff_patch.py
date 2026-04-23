"""Import guard for diff-match-patch package.

This module provides graceful fallback when diff-match-patch is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diff_match_patch import diff_match_patch as DiffMatchPatchClass

try:
    from diff_match_patch import diff_match_patch

    DIFF_PATCH_AVAILABLE = True
except ImportError:
    diff_match_patch = None  # type: ignore
    DIFF_PATCH_AVAILABLE = False


def get_diff_match_patch() -> "DiffMatchPatchClass":
    """Get diff_match_patch instance with proper error handling.

    Raises:
        ImportError: If diff-match-patch package is not available.
    """
    if not DIFF_PATCH_AVAILABLE or diff_match_patch is None:
        msg = (
            "diff-match-patch package is required for this feature. "
            "Install with: pip install diff-match-patch>=20241021"
        )
        raise ImportError(msg)
    return diff_match_patch()
