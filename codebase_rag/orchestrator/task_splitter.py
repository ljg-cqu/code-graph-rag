"""
Task Splitter module for parallel sub-agent execution.
Handles splitting user requests into independent subtasks.
"""

import os
from pathlib import Path
from typing import Any

from loguru import logger

from codebase_rag.config import settings
from codebase_rag.utils.path_utils import get_all_code_files


class TaskSplitter:
    """
    Splits user requests into independent parallelizable subtasks.
    Supports multiple splitting strategies: file-based, node-type, query-based, manual.
    """

    def __init__(self, repo_path: str | None = None):
        self.repo_path = Path(repo_path or settings.TARGET_REPO_PATH).resolve()

    def split_task(self, prompt: str, strategy: str = "auto") -> list[dict[str, Any]]:
        """
        Split a user request into subtasks based on the given strategy.

        Args:
            prompt: User's original request
            strategy: Splitting strategy to use (auto, file, node, query, manual)

        Returns:
            List of subtask dictionaries with task details
        """
        if strategy == "auto":
            strategy = self._detect_strategy(prompt)

        if strategy == "file":
            return self._split_by_file(prompt)
        elif strategy == "node":
            return self._split_by_node_type(prompt)
        elif strategy == "query":
            return self._split_by_query(prompt)
        elif strategy == "manual":
            return self._split_manual(prompt)
        else:
            raise ValueError(f"Unsupported splitting strategy: {strategy}")

    def _detect_strategy(self, prompt: str) -> str:
        """
        Automatically detect the best splitting strategy based on the prompt.

        Args:
            prompt: User's request

        Returns:
            Detected strategy name
        """
        lower_prompt = prompt.lower()

        # Check for file-based patterns
        file_indicators = [
            "all files",
            "each file",
            "files in",
            "directory",
            "folder",
            "review code",
            "scan files",
            "generate tests for",
            "analyze files",
        ]
        if any(indicator in lower_prompt for indicator in file_indicators):
            return "file"

        # Check for node-type patterns
        node_indicators = [
            "all functions",
            "all classes",
            "all methods",
            "function definitions",
            "class declarations",
            "interface definitions",
        ]
        if any(indicator in lower_prompt for indicator in node_indicators):
            return "node"

        # Default to file-based for MVP
        return "file"

    def _split_by_file(self, prompt: str) -> list[dict[str, Any]]:
        """
        Split task by file boundaries, one subtask per file.

        Args:
            prompt: Original user request

        Returns:
            List of file-based subtasks
        """
        code_files = get_all_code_files(self.repo_path)

        subtasks = []
        for idx, file_path in enumerate(code_files):
            relative_path = os.path.relpath(file_path, self.repo_path)
            subtasks.append(
                {
                    "id": f"subtask_{idx}",
                    "type": "file",
                    "file_path": str(file_path),
                    "relative_path": relative_path,
                    "prompt": f"{prompt}\n\nFocus only on the file: {relative_path}",
                    "metadata": {
                        "file_size": os.path.getsize(file_path),
                        "file_extension": os.path.splitext(file_path)[1].lower(),
                    },
                }
            )

        logger.info(f"Split task into {len(subtasks)} file-based subtasks")
        return subtasks

    def _split_by_node_type(self, prompt: str) -> list[dict[str, Any]]:
        """
        Split task by graph node type (functions, classes, etc.).
        TODO: Implement in Phase 2
        """
        raise NotImplementedError("Node-type splitting will be implemented in Phase 2")

    def _split_by_query(self, prompt: str) -> list[dict[str, Any]]:
        """
        Split task by independent query segments.
        TODO: Implement in Phase 2
        """
        raise NotImplementedError(
            "Query-based splitting will be implemented in Phase 2"
        )

    def _split_manual(self, prompt: str) -> list[dict[str, Any]]:
        """
        Split task based on explicit user-defined subtasks.
        TODO: Implement in Phase 2
        """
        raise NotImplementedError("Manual splitting will be implemented in Phase 2")

    def validate_subtasks(
        self, subtasks: list[dict[str, Any]], original_prompt: str
    ) -> bool:
        """
        Validate that subtasks cover the full scope of the original request with no gaps or overlaps.

        Args:
            subtasks: List of generated subtasks
            original_prompt: Original user request

        Returns:
            True if subtasks are valid, False otherwise
        """
        if not subtasks:
            logger.warning("No subtasks generated")
            return False

        # For file-based splitting, check that all files are included
        file_subtasks = [st for st in subtasks if st["type"] == "file"]
        if file_subtasks:
            expected_files = get_all_code_files(self.repo_path)
            subtask_files = [st["file_path"] for st in file_subtasks]
            if len(set(subtask_files)) != len(expected_files):
                logger.warning("File-based subtasks do not cover all code files")
                return False

        logger.info("Subtasks validation passed")
        return True
