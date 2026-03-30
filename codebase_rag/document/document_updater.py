"""Document graph updater for indexing documents into the document graph.

Pattern follows GraphUpdater from codebase_rag/graph_updater.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .. import constants as cs
from ..embeddings import get_embedding_provider
from .chunking import DocumentChunk, SemanticDocumentChunker
from .error_handling import DeadLetterQueue, ErrorType, ExtractionError
from .extractors import ExtractedDocument, get_extractor_for_file
from .versioning import ContentVersionTracker, DocumentVersion, VersionCache

if TYPE_CHECKING:
    from ..services.graph_service import MemgraphIngestor


class DocumentGraphUpdater:
    """
    Handles document graph ingestion and updates.

    Pattern follows GraphUpdater from codebase_rag/graph_updater.py
    """

    # Directories to exclude from document indexing
    EXCLUDED_DIRS = frozenset(
        {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".env",
            "env",
            "dist",
            "build",
            ".cache",
            ".pytest_cache",
            ".ruff_cache",
            ".qdrant_code_embeddings",
            ".embedding_cache",
        }
    )

    def __init__(
        self,
        host: str,
        port: int,
        repo_path: Path,
        batch_size: int = 1000,
        workspace: str = "default",
    ) -> None:
        self.host = host
        self.port = port
        self.repo_path = repo_path.resolve()
        self.batch_size = batch_size
        self.workspace = workspace

        self.version_tracker = ContentVersionTracker()
        self.version_cache = VersionCache(
            self.repo_path / ".cgr" / "doc_versions.json"
        )
        self.dead_letter_queue = DeadLetterQueue(
            self.repo_path / ".cgr" / "doc_errors"
        )
        self.chunker = SemanticDocumentChunker()

    def run(self, force: bool = False) -> dict:
        """
        Ingest all documents into document graph.

        Args:
            force: If True, re-index all documents (ignore cache)

        Returns:
            Dict with indexing statistics
        """
        stats = {
            "total_documents": 0,
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "sections_created": 0,
            "chunks_created": 0,
        }

        # TODO: Use actual MemgraphIngestor
        # For now, this is a placeholder implementation
        documents = self._collect_documents()
        stats["total_documents"] = len(documents)

        logger.info(f"Found {len(documents)} documents to index")

        for doc_path in documents:
            try:
                result = self._process_document(doc_path, force=force)
                if result == "indexed":
                    stats["indexed"] += 1
                elif result == "skipped":
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {e}")
                stats["failed"] += 1
                self.dead_letter_queue.enqueue(
                    ExtractionError(
                        path=str(doc_path),
                        error_type=ErrorType.UNKNOWN,
                        message=str(e),
                    )
                )

        logger.info(f"Document indexing complete: {stats}")
        return stats

    async def run_async(self, force: bool = False) -> dict:
        """Async version of run() for concurrent processing."""
        stats = {
            "total_documents": 0,
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "sections_created": 0,
            "chunks_created": 0,
        }

        documents = self._collect_documents()
        stats["total_documents"] = len(documents)

        # Process documents concurrently
        semaphore = asyncio.Semaphore(10)  # Limit concurrent processing

        async def process_one(doc_path: Path) -> str:
            async with semaphore:
                try:
                    return await self._process_document_async(doc_path, force=force)
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {e}")
                    self.dead_letter_queue.enqueue(
                        ExtractionError(
                            path=str(doc_path),
                            error_type=ErrorType.UNKNOWN,
                            message=str(e),
                        )
                    )
                    return "failed"

        results = await asyncio.gather(
            *[process_one(d) for d in documents], return_exceptions=True
        )

        for result in results:
            if result == "indexed":
                stats["indexed"] += 1
            elif result == "skipped":
                stats["skipped"] += 1
            elif result == "failed" or isinstance(result, Exception):
                stats["failed"] += 1

        return stats

    def _collect_documents(self) -> list[Path]:
        """Collect all eligible document files."""
        documents: list[Path] = []

        # Get supported extensions from config or use defaults
        supported_extensions = {".md", ".rst", ".txt", ".pdf", ".docx"}

        for ext in supported_extensions:
            for doc_path in self.repo_path.rglob(f"*{ext}"):
                # Check if path is in excluded directory
                if any(excl in str(doc_path) for excl in self.EXCLUDED_DIRS):
                    continue

                # Check if path is a file
                if doc_path.is_file():
                    documents.append(doc_path)

        return documents

    def _process_document(
        self,
        file_path: Path,
        force: bool = False,
    ) -> str:
        """
        Process single document.

        Returns:
            "indexed", "skipped", or "failed"
        """
        # Find appropriate extractor
        extractor = get_extractor_for_file(file_path)
        if not extractor:
            logger.warning(f"No extractor for {file_path}")
            return "skipped"

        # Check if re-indexing needed
        if not force:
            stored_version = self.version_cache.get(str(file_path))
            needs_reindex, _ = self.version_tracker.needs_reindex(
                file_path, stored_version
            )
            if not needs_reindex:
                logger.debug(f"Skipping unchanged document: {file_path}")
                return "skipped"

        # Extract content
        doc = extractor.extract(file_path)

        # Store in graph
        self._store_document(doc)

        # Generate and store embeddings
        self._generate_and_store_embeddings(doc)

        # Update version cache
        version = self.version_tracker.create_version(doc)
        self.version_cache.set(version)

        return "indexed"

    async def _process_document_async(
        self,
        file_path: Path,
        force: bool = False,
    ) -> str:
        """Async version of _process_document."""
        extractor = get_extractor_for_file(file_path)
        if not extractor:
            return "skipped"

        if not force:
            stored_version = self.version_cache.get(str(file_path))
            needs_reindex, _ = self.version_tracker.needs_reindex(
                file_path, stored_version
            )
            if not needs_reindex:
                return "skipped"

        # Async extraction
        doc = await extractor.extract_async(file_path)

        # Store and embed
        self._store_document(doc)
        await self._generate_and_store_embeddings_async(doc)

        # Update version
        version = self.version_tracker.create_version(doc)
        self.version_cache.set(version)

        return "indexed"

    def _store_document(self, doc: ExtractedDocument) -> None:
        """Store document and sections in graph."""
        # TODO: Implement actual MemgraphIngestor integration
        # This would create Document, Section, and Chunk nodes

        logger.debug(f"Storing document: {doc.path}")

        # Create document node
        # ingestor.ensure_node_batch('Document', {
        #     'path': doc.path,
        #     'workspace': self.workspace,
        #     'file_type': doc.file_type,
        #     'section_count': len(doc.sections),
        #     'code_block_count': len(doc.code_blocks),
        #     'code_references': doc.code_references,
        #     'word_count': doc.word_count,
        #     'modified_date': doc.modified_date,
        #     'indexed_at': datetime.now(UTC).isoformat(),
        #     'content_hash': doc.content_hash,
        # })

        # Create section nodes
        for i, section in enumerate(doc.sections):
            section_qn = f"{doc.path}#{section.title}"
            # ingestor.ensure_node_batch('Section', {...})
            # ingestor.ensure_relationship_batch(...)
            pass

    def _generate_and_store_embeddings(self, doc: ExtractedDocument) -> None:
        """Generate embeddings using existing provider system."""
        from ..config import settings

        provider = get_embedding_provider(
            provider=settings.EMBEDDING_PROVIDER,
            model_id=settings.EMBEDDING_MODEL,
        )

        # Chunk document semantically
        chunks = list(self.chunker.chunk_document(doc))

        if not chunks:
            # Fallback for empty documents
            doc_embedding = provider.embed(doc.content[:1000])
            # Store document-level embedding
            return

        # Embed all chunks
        chunk_contents = [c.content for c in chunks]
        embeddings = provider.embed_batch(chunk_contents)

        # Store chunk embeddings
        for chunk, embedding in zip(chunks, embeddings):
            # Store in Chunk node with embedding property
            pass

    async def _generate_and_store_embeddings_async(
        self, doc: ExtractedDocument
    ) -> None:
        """Async version of _generate_and_store_embeddings."""
        # For now, run sync version in thread
        await asyncio.to_thread(self._generate_and_store_embeddings, doc)

    def update_file(self, file_path: Path) -> str:
        """
        Update a single document file.

        Called by real-time updater when file changes.
        """
        return self._process_document(file_path, force=True)


__all__ = ["DocumentGraphUpdater"]