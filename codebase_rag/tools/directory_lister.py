from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from pydantic_ai import Tool

from .. import exceptions as ex
from .. import logs as ls
from .. import tool_errors as te
from ..utils.path_utils import is_path_allowed
from . import tool_descriptions as td


class DirectoryLister:
    __slots__ = ("project_root",)

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        # If project_root is a file, use its parent directory
        if self.project_root.is_file():
            self.project_root = self.project_root.parent

    def list_directory_contents(self, directory_path: str) -> str:
        try:
            target_path = self._get_safe_path(directory_path)
        except PermissionError:
            return te.DIRECTORY_PATH_OUTSIDE_ROOT.format(
                path=directory_path, root=self.project_root
            )

        logger.info(ls.DIR_LISTING.format(path=target_path))

        try:
            if not target_path.is_dir():
                return te.DIRECTORY_INVALID.format(path=directory_path)

            if contents := os.listdir(target_path):
                return "\n".join(contents)
            return te.DIRECTORY_EMPTY.format(path=directory_path)

        except Exception as e:
            logger.error(ls.DIR_LIST_ERROR.format(path=directory_path, error=e))
            return te.DIRECTORY_LIST_FAILED.format(path=directory_path)

    def _get_safe_path(self, file_path: str) -> Path:
        if Path(file_path).is_absolute():
            safe_path = Path(file_path).resolve()
        else:
            safe_path = (self.project_root / file_path).resolve()

        if not is_path_allowed(safe_path, self.project_root):
            raise PermissionError(
                f"{ex.ACCESS_DENIED} To allow access to files outside the project root, set ENABLE_GLOBAL_FILE_ACCESS=true in your environment or .env file."
            )

        return safe_path


def create_directory_lister_tool(directory_lister: DirectoryLister) -> Tool:
    return Tool(
        function=directory_lister.list_directory_contents,
        name=td.AgenticToolName.LIST_DIRECTORY,
        description=td.DIRECTORY_LISTER,
    )
