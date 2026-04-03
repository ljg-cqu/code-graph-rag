"""Document indexing tool for interactive agent.

Provides Pydantic AI Tool factory for indexing documents into the document graph.
"""

from __future__ import annotations

from loguru import logger
from pydantic_ai import Tool

from .. import constants as cs
from ..config import settings
from . import tool_descriptions as td


def create_index_documents_tool() -> Tool:
    """Create index_documents tool for Pydantic AI agent.

    Usage: Index documents into the document graph.
    Example: "Index all markdown files in the docs folder"
    """

    async def index_documents(
        clean: bool = False,
        force: bool = False,
    ) -> str:
        """Index documents into document graph.

        Args:
            clean: Clear document database before indexing
            force: Force re-indexing (ignore version cache)

        Returns:
            Indexing statistics
        """
        logger.info(f"Indexing documents (clean={clean}, force={force})")

        try:
            from ..document.document_updater import DocumentGraphUpdater

            updater = DocumentGraphUpdater(
                repo_path=settings.TARGET_REPO_PATH,
                host=settings.DOC_MEMGRAPH_HOST,
                port=settings.DOC_MEMGRAPH_PORT,
            )

            stats = updater.run(clean=clean, force=force)

            return (
                f"**Document Indexing Complete**\n\n"
                f"- Total documents: {stats.get('total_documents', 0)}\n"
                f"- Indexed: {stats.get('indexed', 0)}\n"
                f"- Skipped: {stats.get('skipped', 0)}\n"
                f"- Failed: {stats.get('failed', 0)}\n"
                f"- Sections created: {stats.get('sections_created', 0)}\n"
                f"- Chunks created: {stats.get('chunks_created', 0)}\n"
            )

        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return f"Document indexing failed: {e}"

    return Tool(
        index_documents,
        name="index_documents",
        description=td.MCP_INDEX_DOCUMENTS,
    )


__all__ = [
    "create_index_documents_tool",
]