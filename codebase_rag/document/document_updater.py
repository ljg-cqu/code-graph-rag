"""Document graph updater for indexing documents into the document graph.

Pattern follows GraphUpdater from codebase_rag/graph_updater.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .. import constants as cs
from ..config import settings
from ..embeddings import get_embedding_provider
from ..services.graph_service import MemgraphIngestor
from .chunking import SemanticDocumentChunker
from .error_handling import DeadLetterQueue, ErrorType, ExtractionError, ExtractionException
from .extractors import ExtractedDocument, ExtractedSection, get_extractor_for_file
from .versioning import ContentVersionTracker, VersionCache


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

        # Determine base path for metadata files
        # If repo_path is a file, use its parent directory
        if self.repo_path.is_file():
            self.base_path = self.repo_path.parent
        else:
            self.base_path = self.repo_path

        # Security: Validate repo_path is within expected boundaries
        resolved_repo = self.repo_path.resolve()
        # Use base_path for boundary check (handles both file and directory cases)
        # Validate that the resolved path is within base_path boundaries
        if not self._is_path_within_boundary(resolved_repo):
            raise ValueError(f"repo_path {repo_path} is outside allowed boundaries")

        self.batch_size = batch_size
        self.workspace = workspace

        # Ensure metadata directory exists before initializing caches
        cgr_dir = self.base_path / ".cgr"
        cgr_dir.mkdir(parents=True, exist_ok=True)

        self.version_tracker = ContentVersionTracker()
        self.version_cache = VersionCache(
            cgr_dir / "doc_versions.json"
        )
        self.dead_letter_queue = DeadLetterQueue(
            cgr_dir / "doc_errors"
        )
        self.chunker = SemanticDocumentChunker()

        # Cache embedding provider to avoid recreation per document
        config = settings.active_embedding_config
        self._embedding_provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
            api_key=config.api_key,
            endpoint=config.endpoint,
            keep_alive=config.keep_alive,
            project_id=config.project_id,
            region=config.region,
            provider_type=config.provider_type,
            service_account_file=config.service_account_file,
            device=config.device,
        )

        # Cache supported extensions from config
        self._supported_extensions = set(settings.DOC_SUPPORTED_EXTENSIONS)

    def _is_path_within_boundary(self, path: Path) -> bool:
        """
        Check if a path is within the repository boundary.

        Uses proper path comparison, not string prefix which can be flawed
        (e.g., '/repo2/file.txt' incorrectly matches '/repo' prefix).

        Args:
            path: Path to validate (should be resolved)

        Returns:
            True if path is within base_path boundaries
        """
        try:
            # is_relative_to() is the correct way to check path containment
            # It handles edge cases like /repo vs /repo2 correctly
            return path.is_relative_to(self.base_path)
        except (OSError, ValueError):
            # is_relative_to may raise on some edge cases
            # Fallback to checking if relative_to succeeds
            try:
                path.relative_to(self.base_path)
                return True
            except ValueError:
                return False

    def _map_error_type(self, exc: Exception) -> ErrorType:
        """Map common exception types to ErrorType for better classification."""
        if isinstance(exc, FileNotFoundError):
            return ErrorType.FILE_NOT_FOUND
        if isinstance(exc, PermissionError):
            return ErrorType.PERMISSION_DENIED
        if isinstance(exc, UnicodeDecodeError | UnicodeEncodeError):
            return ErrorType.ENCODING_ERROR
        if isinstance(exc, OSError | IOError):
            # Generic I/O error - could be disk, network, resource issues
            # Not necessarily malformed file content
            return ErrorType.UNKNOWN
        return ErrorType.UNKNOWN

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

        with MemgraphIngestor(
            host=self.host,
            port=self.port,
            batch_size=self.batch_size,
        ) as ingestor:
            ingestor.ensure_constraints()
            documents = self._collect_documents()
            stats["total_documents"] = len(documents)

            logger.info(f"Found {len(documents)} documents to index")

            for doc_path in documents:
                try:
                    result = self._process_document(doc_path, ingestor, force=force)
                    if result == "indexed":
                        stats["indexed"] += 1
                    elif result == "skipped":
                        stats["skipped"] += 1
                except ExtractionException as e:
                    logger.error(f"Failed to process {doc_path}: {type(e).__name__}: {e}")
                    stats["failed"] += 1
                    try:
                        self.dead_letter_queue.enqueue(e.to_extraction_error())
                    except Exception as dlq_error:
                        logger.warning(f"Could not enqueue error for {doc_path}: {dlq_error}")
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {type(e).__name__}: {e}")
                    stats["failed"] += 1
                    try:
                        self.dead_letter_queue.enqueue(
                            ExtractionError(
                                path=str(doc_path),
                                error_type=self._map_error_type(e),
                                message=str(e),
                            )
                        )
                    except Exception as dlq_error:
                        logger.warning(f"Could not enqueue error for {doc_path}: {dlq_error}")

            # Flush all pending nodes and relationships
            try:
                ingestor.flush_all()
            except Exception as e:
                logger.error(f"Failed to flush batch to graph: {type(e).__name__}: {e}")
                # Mark all indexed documents as failed since data wasn't persisted
                stats["failed"] += stats["indexed"]
                stats["indexed"] = 0
                # Clear in-memory version cache to prevent stale skips on retry
                self.version_cache.clear()
                raise  # Re-raise to trigger context manager cleanup

            # Query counts from graph for workspace-filtered stats
            # Note: These are total counts for this workspace, not deltas from this run
            try:
                section_result = ingestor.fetch_all(
                    "MATCH (s:Section {workspace: $ws}) RETURN count(s) as count",
                    {"ws": self.workspace}
                )
                chunk_result = ingestor.fetch_all(
                    "MATCH (c:Chunk {workspace: $ws}) RETURN count(c) as count",
                    {"ws": self.workspace}
                )
                if section_result:
                    stats["sections_created"] = section_result[0].get("count", 0)
                if chunk_result:
                    stats["chunks_created"] = chunk_result[0].get("count", 0)
            except Exception as e:
                logger.warning(f"Could not query stats from graph: {e}")

            # Persist version cache to disk
            try:
                logger.debug("Saving version cache to disk")
                self.version_cache.save()
            except Exception as e:
                logger.warning(f"Could not save version cache: {e}")

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

        async with MemgraphIngestor(
            host=self.host,
            port=self.port,
            batch_size=self.batch_size,
        ) as ingestor:
            await asyncio.to_thread(ingestor.ensure_constraints)
            documents = await asyncio.to_thread(self._collect_documents)
            stats["total_documents"] = len(documents)

            logger.info(f"Found {len(documents)} documents to index")

            # Process documents with async extraction but sequential graph writes
            for doc_path in documents:
                try:
                    result = await self._process_document_async(doc_path, ingestor, force=force)
                    if result == "indexed":
                        stats["indexed"] += 1
                    elif result == "skipped":
                        stats["skipped"] += 1
                except ExtractionException as e:
                    logger.error(f"Failed to process {doc_path}: {type(e).__name__}: {e}")
                    stats["failed"] += 1
                    try:
                        await asyncio.to_thread(self.dead_letter_queue.enqueue, e.to_extraction_error())
                    except Exception as dlq_error:
                        logger.warning(f"Could not enqueue error for {doc_path}: {dlq_error}")
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {type(e).__name__}: {e}")
                    stats["failed"] += 1
                    try:
                        await asyncio.to_thread(
                            self.dead_letter_queue.enqueue,
                            ExtractionError(
                                path=str(doc_path),
                                error_type=self._map_error_type(e),
                                message=str(e),
                            ),
                        )
                    except Exception as dlq_error:
                        logger.warning(f"Could not enqueue error for {doc_path}: {dlq_error}")

            # Flush all pending nodes and relationships
            try:
                await asyncio.to_thread(ingestor.flush_all)
            except Exception as e:
                logger.error(f"Failed to flush batch to graph: {type(e).__name__}: {e}")
                # Mark all indexed documents as failed since data wasn't persisted
                stats["failed"] += stats["indexed"]
                stats["indexed"] = 0
                # Clear in-memory version cache to prevent stale skips on retry
                self.version_cache.clear()
                raise  # Re-raise to trigger context manager cleanup

            # Query counts from graph for workspace-filtered stats
            # Note: These are total counts for this workspace, not deltas from this run
            try:
                section_result = await asyncio.to_thread(
                    ingestor.fetch_all,
                    "MATCH (s:Section {workspace: $ws}) RETURN count(s) as count",
                    {"ws": self.workspace}
                )
                chunk_result = await asyncio.to_thread(
                    ingestor.fetch_all,
                    "MATCH (c:Chunk {workspace: $ws}) RETURN count(c) as count",
                    {"ws": self.workspace}
                )
                if section_result:
                    stats["sections_created"] = section_result[0].get("count", 0)
                if chunk_result:
                    stats["chunks_created"] = chunk_result[0].get("count", 0)
            except Exception as e:
                logger.warning(f"Could not query stats from graph: {e}")

            # Persist version cache to disk
            try:
                logger.debug("Saving version cache to disk")
                await asyncio.to_thread(self.version_cache.save)
            except Exception as e:
                logger.warning(f"Could not save version cache: {e}")

        logger.info(f"Document indexing complete: {stats}")
        return stats

    def _collect_documents(self) -> list[Path]:
        """Collect all eligible document files."""
        documents: list[Path] = []

        # Use cached supported extensions from config
        supported_extensions = self._supported_extensions

        # Handle single file path
        if self.repo_path.is_file():
            # Security: Check excluded directories for single file
            if any(part in self.EXCLUDED_DIRS for part in self.repo_path.parts):
                logger.debug(f"Skipping file in excluded directory: {self.repo_path}")
                return documents

            # Security: Check extension
            if self.repo_path.suffix.lower() not in supported_extensions:
                logger.debug(f"Skipping unsupported file type: {self.repo_path}")
                return documents

            # Security: Check symlink escape
            if self.repo_path.is_symlink():
                resolved = self.repo_path.resolve()
                if not self._is_path_within_boundary(resolved):
                    logger.debug(f"Skipping symlink pointing outside repo: {self.repo_path}")
                    return documents

            documents.append(self.repo_path)
            return documents

        # Handle directory path
        for ext in supported_extensions:
            for doc_path in self.repo_path.rglob(f"*{ext}"):
                # Check if any path component is in excluded directories
                if any(part in self.EXCLUDED_DIRS for part in doc_path.parts):
                    continue

                # Check if path is a file
                if not doc_path.is_file():
                    continue

                # Security: Skip symlinks pointing outside repo
                if doc_path.is_symlink():
                    resolved = doc_path.resolve()
                    if not self._is_path_within_boundary(resolved):
                        logger.debug(f"Skipping symlink pointing outside repo: {doc_path}")
                        continue

                documents.append(doc_path)

        return documents

    def _process_document(
        self,
        file_path: Path,
        ingestor: MemgraphIngestor,
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

        # Generate embeddings BEFORE deleting existing nodes
        # This ensures rollback safety: if embedding fails, old data is preserved
        chunks = list(self.chunker.chunk_document(doc))
        embeddings_data = self._prepare_embeddings(doc, chunks)

        # Only delete existing nodes after embeddings are validated
        self._delete_document_nodes(doc.path, ingestor)

        # Store document and sections in graph
        store_stats = self._store_document(doc, ingestor)

        # Store pre-computed chunks with embeddings
        chunk_count = self._store_chunks_with_embeddings(doc, chunks, embeddings_data, ingestor)

        # Update version cache
        version = self.version_tracker.create_version(doc)
        self.version_cache.set(version)

        logger.debug(
            f"Indexed {file_path}: {store_stats['sections']} sections, {chunk_count} chunks"
        )
        return "indexed"

    def _delete_document_nodes(self, doc_path: str, ingestor: MemgraphIngestor) -> None:
        """
        Delete existing Section and Chunk nodes for a document.

        This ensures clean re-indexing without duplicates or stale data.
        Handles both old-format (#{title}) and new-format (#L{line}:{title}) nodes.

        Args:
            doc_path: Document path to clean up
            ingestor: MemgraphIngestor instance

        Raises:
            ExtractionException: If deletion fails (critical for data integrity)
        """
        try:
            # Delete all Sections connected to this document (any format)
            ingestor.execute_write(
                """
                MATCH (d:Document {path: $path})-[:CONTAINS_SECTION]->(s:Section)
                DETACH DELETE s
                """,
                {"path": doc_path},
            )

            # Delete all Chunks connected to this document
            ingestor.execute_write(
                """
                MATCH (d:Document {path: $path})-[:CONTAINS_CHUNK]->(c:Chunk)
                DETACH DELETE c
                """,
                {"path": doc_path},
            )

            # Also delete orphaned sections/chunks that might match by path prefix
            # (handles old data where document node might not exist)
            ingestor.execute_write(
                """
                MATCH (s:Section)
                WHERE s.qualified_name STARTS WITH $path_prefix
                DETACH DELETE s
                """,
                {"path_prefix": f"{doc_path}#"},
            )

            ingestor.execute_write(
                """
                MATCH (c:Chunk)
                WHERE c.qualified_name STARTS WITH $path_prefix
                DETACH DELETE c
                """,
                {"path_prefix": f"{doc_path}#"},
            )

            logger.debug(f"Cleaned existing nodes for document: {doc_path}")
        except Exception as e:
            raise ExtractionException(
                path=doc_path,
                error_type=ErrorType.UNKNOWN,
                message=f"Failed to delete existing document nodes: {type(e).__name__}: {e}",
            ) from e

    async def _process_document_async(
        self,
        file_path: Path,
        ingestor: MemgraphIngestor,
        force: bool = False,
    ) -> str:
        """Async version of _process_document."""
        extractor = get_extractor_for_file(file_path)
        if not extractor:
            logger.warning(f"No extractor for {file_path}")
            return "skipped"

        if not force:
            stored_version = self.version_cache.get(str(file_path))
            needs_reindex, _ = await asyncio.to_thread(
                self.version_tracker.needs_reindex, file_path, stored_version
            )
            if not needs_reindex:
                logger.debug(f"Skipping unchanged document: {file_path}")
                return "skipped"

        # Async extraction
        doc = await extractor.extract_async(file_path)

        # Generate embeddings BEFORE deleting existing nodes (rollback safety)
        chunks = list(self.chunker.chunk_document(doc))
        embeddings_data = await asyncio.to_thread(self._prepare_embeddings, doc, chunks)

        # Only delete existing nodes after embeddings are validated
        await asyncio.to_thread(self._delete_document_nodes, doc.path, ingestor)

        # Store document and sections (run in thread to avoid blocking)
        store_stats = await asyncio.to_thread(self._store_document, doc, ingestor)
        chunk_count = await asyncio.to_thread(
            self._store_chunks_with_embeddings, doc, chunks, embeddings_data, ingestor
        )

        # Update version
        version = self.version_tracker.create_version(doc)
        self.version_cache.set(version)

        logger.debug(
            f"Indexed {file_path}: {store_stats['sections']} sections, {chunk_count} chunks"
        )
        return "indexed"

    def _store_document(self, doc: ExtractedDocument, ingestor: MemgraphIngestor) -> dict:
        """Store document and sections in graph."""
        stats = {"sections": 0}

        logger.debug(f"Storing document: {doc.path}")

        # Create Document node
        ingestor.ensure_node_batch(
            cs.NodeLabel.DOCUMENT.value,
            {
                cs.UniqueKeyType.PATH.value: doc.path,
                "workspace": self.workspace,
                "file_type": doc.file_type,
                "section_count": len(doc.sections),
                "code_block_count": len(doc.code_blocks),
                "code_references": doc.code_references,
                "word_count": doc.word_count,
                "modified_date": doc.modified_date,
                "indexed_at": datetime.now(UTC).isoformat(),
                "content_hash": doc.content_hash,
            },
        )

        # Create Section nodes and relationships
        for section in doc.sections:
            self._store_section(doc.path, section, doc.path, None, ingestor, stats)

        return stats

    def _store_section(
        self,
        doc_path: str,
        section: ExtractedSection,
        parent_path: str,
        parent_qn: str | None,
        ingestor: MemgraphIngestor,
        stats: dict,
    ) -> None:
        """
        Recursively store section and its subsections.

        Args:
            doc_path: Document path
            section: Section to store
            parent_path: Path for qualified name construction
            parent_qn: Parent section's qualified name (None for top-level)
            ingestor: MemgraphIngestor instance
            stats: Stats dict to update
        """
        # Create hierarchical qualified name with line number to avoid collisions
        section_qn = f"{parent_path}#L{section.start_line}:{section.title}"
        ingestor.ensure_node_batch(
            cs.NodeLabel.SECTION.value,
            {
                cs.UniqueKeyType.QUALIFIED_NAME.value: section_qn,
                "workspace": self.workspace,
                "title": section.title,
                "level": section.level,
                "start_line": section.start_line,
                "end_line": section.end_line,
                "content_snippet": section.content[:500] if section.content else "",
            },
        )

        # Create relationship to parent (Document or Section)
        if parent_qn is None:
            # Top-level section: Document -> Section
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc_path),
                cs.RelationshipType.CONTAINS_SECTION.value,
                (cs.NodeLabel.SECTION.value, cs.UniqueKeyType.QUALIFIED_NAME.value, section_qn),
            )
        else:
            # Subsection: Section -> Section
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.SECTION.value, cs.UniqueKeyType.QUALIFIED_NAME.value, parent_qn),
                cs.RelationshipType.HAS_SUBSECTION.value,
                (cs.NodeLabel.SECTION.value, cs.UniqueKeyType.QUALIFIED_NAME.value, section_qn),
            )
        stats["sections"] += 1

        # Recursively process subsections
        for subsection in section.subsections:
            self._store_section(
                doc_path, subsection, section_qn, section_qn, ingestor, stats
            )

    def _prepare_embeddings(
        self,
        doc: ExtractedDocument,
        chunks: list,
    ) -> tuple[list, list[list[float]]]:
        """Generate embeddings before deleting existing nodes.

        This ensures rollback safety: if embedding fails, old data is preserved.

        Args:
            doc: Extracted document
            chunks: List of DocumentChunk objects

        Returns:
            Tuple of (non_empty_chunks list, embeddings list)

        Raises:
            ExtractionException: If embedding generation fails
        """
        provider = self._embedding_provider

        if not chunks:
            # Fallback for empty documents
            if not doc.content or not doc.content.strip():
                logger.warning(f"Document {doc.path} has no content, skipping embedding")
                return ([], [])
            try:
                doc_embedding = provider.embed(doc.content[:1000])
            except Exception as e:
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.UNKNOWN,
                    message=f"Embedding generation failed: {type(e).__name__}: {e}",
                ) from e
            # Return a pseudo-chunk for the fallback case
            from .chunking import DocumentChunk
            fallback_chunk = DocumentChunk(
                content=doc.content[:1000],
                section_title="",
                start_line=0,
                end_line=0,
                token_count=self.chunker.count_tokens(doc.content[:1000]),
                document_path=doc.path,
                chunk_index=0,
            )
            return ([fallback_chunk], [doc_embedding])

        # Filter out empty chunks to avoid API errors
        non_empty_chunks = [(i, c) for i, c in enumerate(chunks) if c.content.strip()]
        if not non_empty_chunks:
            logger.warning(f"All chunks in {doc.path} are empty, skipping embedding")
            return ([], [])

        chunk_contents = [c.content for i, c in non_empty_chunks]
        batch_size = settings.VECTOR_EMBEDDING_BATCH_SIZE
        try:
            embeddings = provider.embed_batch(chunk_contents, batch_size=batch_size)
        except Exception as e:
            raise ExtractionException(
                path=doc.path,
                error_type=ErrorType.UNKNOWN,
                message=f"Embedding batch generation failed: {type(e).__name__}: {e}",
            ) from e

        # Validate embedding count matches non-empty chunk count
        if len(embeddings) != len(non_empty_chunks):
            raise ExtractionException(
                path=doc.path,
                error_type=ErrorType.UNKNOWN,
                message=f"Embedding provider returned {len(embeddings)} embeddings for {len(non_empty_chunks)} non-empty chunks",
            )

        # Return chunks and embeddings (without original indices)
        chunks_list = [c for i, c in non_empty_chunks]
        return (chunks_list, embeddings)

    def _store_chunks_with_embeddings(
        self,
        doc: ExtractedDocument,
        chunks: list,
        embeddings_data: tuple[list, list[list[float]]],
        ingestor: MemgraphIngestor,
    ) -> int:
        """Store chunks with pre-computed embeddings.

        Args:
            doc: Extracted document
            chunks: Original chunk list (for reference)
            embeddings_data: Tuple of (non_empty_chunks, embeddings) from _prepare_embeddings
            ingestor: MemgraphIngestor instance

        Returns:
            Number of chunks stored
        """
        non_empty_chunks, embeddings = embeddings_data

        if not non_empty_chunks:
            return 0

        for chunk, embedding in zip(non_empty_chunks, embeddings):
            ingestor.ensure_node_batch(
                cs.NodeLabel.CHUNK.value,
                {
                    cs.UniqueKeyType.QUALIFIED_NAME.value: chunk.qualified_name,
                    "workspace": self.workspace,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "section_title": chunk.section_title,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "embedding": embedding,
                },
            )
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                cs.RelationshipType.CONTAINS_CHUNK.value,
                (cs.NodeLabel.CHUNK.value, cs.UniqueKeyType.QUALIFIED_NAME.value, chunk.qualified_name),
            )

        return len(non_empty_chunks)

    def _generate_and_store_embeddings(
        self, doc: ExtractedDocument, ingestor: MemgraphIngestor
    ) -> int:
        """Generate embeddings using cached provider.

        Raises:
            ExtractionException: If embedding generation fails
        """
        # Use cached embedding provider from __init__
        provider = self._embedding_provider

        # Chunk document semantically
        chunks = list(self.chunker.chunk_document(doc))

        if not chunks:
            # Fallback for empty documents - store document-level embedding
            if not doc.content or not doc.content.strip():
                logger.warning(f"Document {doc.path} has no content, skipping embedding")
                return 0
            # Wrap embedding operation to handle provider errors
            try:
                doc_embedding = provider.embed(doc.content[:1000])
            except Exception as e:
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.UNKNOWN,
                    message=f"Embedding generation failed: {type(e).__name__}: {e}",
                ) from e
            ingestor.ensure_node_batch(
                cs.NodeLabel.CHUNK.value,
                {
                    cs.UniqueKeyType.QUALIFIED_NAME.value: f"{doc.path}#chunk_0",
                    "workspace": self.workspace,
                    "content": doc.content[:1000],
                    "token_count": self.chunker.count_tokens(doc.content[:1000]),
                    "section_title": "",
                    "start_line": 0,
                    "end_line": 0,
                    "embedding": doc_embedding,
                },
            )
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                cs.RelationshipType.CONTAINS_CHUNK.value,
                (cs.NodeLabel.CHUNK.value, cs.UniqueKeyType.QUALIFIED_NAME.value, f"{doc.path}#chunk_0"),
            )
            return 1

        # Embed all chunks using config batch size
        # Filter out empty chunks to avoid API errors (DashScope rejects empty strings)
        non_empty_chunks = [(i, c) for i, c in enumerate(chunks) if c.content.strip()]
        if not non_empty_chunks:
            logger.warning(f"All chunks in {doc.path} are empty, skipping embedding")
            return 0

        chunk_contents = [c.content for i, c in non_empty_chunks]
        batch_size = settings.VECTOR_EMBEDDING_BATCH_SIZE
        # Wrap embedding operation to handle provider errors
        try:
            embeddings = provider.embed_batch(chunk_contents, batch_size=batch_size)
        except Exception as e:
            raise ExtractionException(
                path=doc.path,
                error_type=ErrorType.UNKNOWN,
                message=f"Embedding batch generation failed: {type(e).__name__}: {e}",
            ) from e

        # Validate embedding count matches non-empty chunk count
        if len(embeddings) != len(non_empty_chunks):
            raise ExtractionException(
                path=doc.path,
                error_type=ErrorType.UNKNOWN,
                message=f"Embedding provider returned {len(embeddings)} embeddings for {len(non_empty_chunks)} non-empty chunks",
            )

        # Store chunk embeddings (only non-empty chunks have embeddings)
        for (orig_idx, chunk), embedding in zip(non_empty_chunks, embeddings):
            ingestor.ensure_node_batch(
                cs.NodeLabel.CHUNK.value,
                {
                    cs.UniqueKeyType.QUALIFIED_NAME.value: chunk.qualified_name,
                    "workspace": self.workspace,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "section_title": chunk.section_title,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "embedding": embedding,
                },
            )
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                cs.RelationshipType.CONTAINS_CHUNK.value,
                (cs.NodeLabel.CHUNK.value, cs.UniqueKeyType.QUALIFIED_NAME.value, chunk.qualified_name),
            )

        return len(non_empty_chunks)

    def update_file(self, file_path: Path) -> str:
        """
        Update a single document file.

        Called by real-time updater when file changes.

        Returns:
            "indexed", "skipped", or "failed"
        """
        # Security: Check excluded directories (same as _collect_documents)
        if any(part in self.EXCLUDED_DIRS for part in file_path.parts):
            logger.debug(f"Skipping file in excluded directory: {file_path}")
            return "skipped"

        # Security: Validate extension (same as _collect_documents)
        if file_path.suffix.lower() not in self._supported_extensions:
            logger.debug(f"Skipping unsupported file type: {file_path}")
            return "skipped"

        # Security: Check symlink escape (same as _collect_documents)
        if file_path.is_symlink():
            resolved = file_path.resolve()
            if not self._is_path_within_boundary(resolved):
                logger.debug(f"Skipping symlink pointing outside repo: {file_path}")
                return "skipped"

        # Security: Validate path is within repo boundaries
        # Use base_path for check (handles both file and directory repo paths)
        resolved_path = file_path.resolve()
        if not self._is_path_within_boundary(resolved_path):
            logger.error(f"Path traversal attempt: {file_path} is outside repo {self.base_path}")
            self.dead_letter_queue.enqueue(
                ExtractionError(
                    path=str(file_path),
                    error_type=ErrorType.PATH_TRAVERSAL,
                    message=f"Path is outside repository boundaries",
                )
            )
            return "failed"

        try:
            with MemgraphIngestor(
                host=self.host,
                port=self.port,
                batch_size=self.batch_size,
            ) as ingestor:
                ingestor.ensure_constraints()
                result = self._process_document(file_path, ingestor, force=True)
                ingestor.flush_all()
                logger.debug("Saving version cache to disk")
                self.version_cache.save()
                return result
        except ExtractionException as e:
            logger.error(f"Failed to update file {file_path}: {type(e).__name__}: {e}")
            self.dead_letter_queue.enqueue(e.to_extraction_error())
            self.version_cache.remove(str(file_path))  # Rollback stale version
            return "failed"
        except Exception as e:
            logger.error(f"Failed to update file {file_path}: {type(e).__name__}: {e}")
            self.dead_letter_queue.enqueue(
                ExtractionError(
                    path=str(file_path),
                    error_type=self._map_error_type(e),
                    message=str(e),
                )
            )
            self.version_cache.remove(str(file_path))  # Rollback stale version
            return "failed"


__all__ = ["DocumentGraphUpdater"]