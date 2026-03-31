"""Document graph updater for indexing documents into the document graph.

Pattern follows GraphUpdater from codebase_rag/graph_updater.py
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .. import constants as cs
from ..config import settings
from ..embeddings import get_embedding_provider
from ..services.graph_service import MemgraphIngestor
from .chunking import DocumentChunk, SemanticDocumentChunker
from .error_handling import DeadLetterQueue, ErrorType, ExtractionError, ExtractionException
from .extractors import ExtractedDocument, ExtractedSection, get_extractor_for_file
from .versioning import ContentVersionTracker, VersionCache

# Workspace must be a safe identifier (alphanumeric, underscore, hyphen)
# Same validation pattern as document_query.py for consistency
WORKSPACE_PATTERN = re.compile(r"^[\w\-]+$")


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

        # Validate workspace identifier (same pattern as document_query.py)
        if not WORKSPACE_PATTERN.match(workspace):
            raise ValueError(
                f"Invalid workspace identifier: '{workspace}'. "
                "Must contain only alphanumeric characters, underscores, and hyphens."
            )
        self.workspace = workspace

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
            self._ensure_vector_index(ingestor)
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
                    # Remove stale version cache entry so retry will re-process
                    self.version_cache.remove(str(doc_path))
                    try:
                        self.dead_letter_queue.enqueue(e.to_extraction_error())
                    except Exception as dlq_error:
                        logger.warning(f"Could not enqueue error for {doc_path}: {dlq_error}")
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {type(e).__name__}: {e}")
                    stats["failed"] += 1
                    # Remove stale version cache entry so retry will re-process
                    self.version_cache.remove(str(doc_path))
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
                if section_result and len(section_result) > 0:
                    stats["sections_created"] = section_result[0].get("count", 0)
                if chunk_result and len(chunk_result) > 0:
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
                    # Remove stale version cache entry so retry will re-process
                    self.version_cache.remove(str(doc_path))
                    try:
                        await asyncio.to_thread(self.dead_letter_queue.enqueue, e.to_extraction_error())
                    except Exception as dlq_error:
                        logger.warning(f"Could not enqueue error for {doc_path}: {dlq_error}")
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {type(e).__name__}: {e}")
                    stats["failed"] += 1
                    # Remove stale version cache entry so retry will re-process
                    self.version_cache.remove(str(doc_path))
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
                if section_result and len(section_result) > 0:
                    stats["sections_created"] = section_result[0].get("count", 0)
                if chunk_result and len(chunk_result) > 0:
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

    def _ensure_vector_index(self, ingestor: MemgraphIngestor) -> None:
        """Create vector index for Chunk embeddings.

        Vector indexes enable efficient similarity search on embeddings.
        Uses Memgraph's CREATE VECTOR INDEX syntax with capacity from config.
        """
        # Use the embedding provider's dimension property (already has correct mapping)
        # The provider's dimension is determined from known model dimensions
        dimension = self._embedding_provider.dimension

        capacity = settings.DOC_MEMGRAPH_VECTOR_CAPACITY
        index_name = settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME

        cypher = f"""
        CREATE VECTOR INDEX {index_name}
        ON :Chunk(embedding)
        WITH CONFIG {{
            "dimension": {dimension},
            "capacity": {capacity},
            "metric": "cos"
        }};
        """

        try:
            ingestor.execute_write(cypher, {})
            logger.info(f"Created vector index '{index_name}' for Chunk nodes (dim={dimension}, capacity={capacity})")
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate" in error_str:
                logger.info(f"Vector index '{index_name}' already exists")
            else:
                logger.error(f"Failed to create vector index '{index_name}': {e}")
                # Non-fatal: indexing can proceed without vector index

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

        # Store document and sections in graph, get section info for chunk matching
        store_stats, section_info, indexed_at = self._store_document(doc, ingestor)

        # Store pre-computed chunks with embeddings and section relationships
        chunk_count = self._store_chunks_with_embeddings(doc, embeddings_data, section_info, ingestor, indexed_at)

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

        Note:
            All queries include workspace filter to prevent multi-tenancy data leaks.
            Uses path prefix matching to clean up orphaned nodes.
        """
        workspace = self.workspace
        path_prefix = f"{doc_path}#"

        try:
            # Combined deletion query for all nodes related to this document
            # This reduces the number of round-trips and is more atomic
            ingestor.execute_write(
                """
                MATCH (d:Document {path: $path, workspace: $workspace})
                OPTIONAL MATCH (d)-[:CONTAINS_SECTION]->(s:Section)
                OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
                DETACH DELETE s
                DETACH DELETE c
                """,
                {"path": doc_path, "workspace": workspace},
            )

            # Clean up orphaned sections/chunks by path prefix (handles edge cases)
            ingestor.execute_write(
                """
                MATCH (s:Section {workspace: $workspace})
                WHERE s.qualified_name STARTS WITH $path_prefix
                DETACH DELETE s
                """,
                {"path_prefix": path_prefix, "workspace": workspace},
            )

            ingestor.execute_write(
                """
                MATCH (c:Chunk {workspace: $workspace})
                WHERE c.qualified_name STARTS WITH $path_prefix
                DETACH DELETE c
                """,
                {"path_prefix": path_prefix, "workspace": workspace},
            )

            logger.debug(f"Cleaned existing nodes for document: {doc_path}")
        except Exception as e:
            raise ExtractionException(
                path=doc_path,
                error_type=ErrorType.GRAPH_ERROR,
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
        store_stats, section_info, indexed_at = await asyncio.to_thread(self._store_document, doc, ingestor)
        chunk_count = await asyncio.to_thread(
            self._store_chunks_with_embeddings, doc, embeddings_data, section_info, ingestor, indexed_at
        )

        # Update version
        version = self.version_tracker.create_version(doc)
        self.version_cache.set(version)

        logger.debug(
            f"Indexed {file_path}: {store_stats['sections']} sections, {chunk_count} chunks"
        )
        return "indexed"

    def _store_document(self, doc: ExtractedDocument, ingestor: MemgraphIngestor) -> tuple[dict, list[dict], str]:
        """Store document and sections in graph.

        Returns:
            Tuple of (stats dict, section_info list for chunk matching, indexed_at timestamp)
        """
        stats = {"sections": 0}
        all_section_info: list[dict] = []
        indexed_at = datetime.now(UTC).isoformat()

        logger.debug(f"Storing document: {doc.path}")

        # Check for preamble content (content before first section)
        # Cache lines to avoid redundant split operations
        preamble_line_count = 0
        has_preamble = False
        preamble_content = ""
        doc_lines = doc.content.split("\n") if doc.content else []

        if doc.sections and doc_lines:
            first_section_start = doc.sections[0].start_line
            if first_section_start > 0:
                preamble_content = "\n".join(doc_lines[:first_section_start])
                if preamble_content.strip():
                    has_preamble = True
                    preamble_line_count = first_section_start

        # Track if we'll create a synthetic section (for documents without sections)
        # Note: Synthetic section and preamble are mutually exclusive
        will_create_synthetic = not doc.sections and doc.content and doc.content.strip() and not has_preamble

        # Create Document node
        # Note: section_count includes synthetic sections for plain text files
        # and counts ALL sections (root + nested) for accurate total
        # Also includes preamble section if present
        preamble_count = 1 if has_preamble else 0
        ingestor.ensure_node_batch(
            cs.NodeLabel.DOCUMENT.value,
            {
                cs.UniqueKeyType.PATH.value: doc.path,
                "workspace": self.workspace,
                "file_type": doc.file_type,
                "section_count": doc.total_section_count() + (1 if will_create_synthetic else 0) + preamble_count,
                "code_block_count": len(doc.code_blocks),
                "code_references": doc.code_references,
                "word_count": doc.word_count,
                "modified_date": doc.modified_date,
                "indexed_at": indexed_at,
                "content_hash": doc.content_hash,
            },
        )

        # Create preamble section if there's content before the first section
        if has_preamble:
            preamble_qn = f"{doc.path}#synthetic:Preamble"
            logger.debug(f"Creating preamble section for {doc.path} ({preamble_line_count} lines)")
            ingestor.ensure_node_batch(
                cs.NodeLabel.SECTION.value,
                {
                    cs.UniqueKeyType.QUALIFIED_NAME.value: preamble_qn,
                    "workspace": self.workspace,
                    "title": "Preamble",
                    "level": 0,  # Level 0 to indicate it's before the document structure
                    "start_line": 0,
                    "end_line": max(0, preamble_line_count - 1),
                    "content_snippet": preamble_content[:500],
                    "indexed_at": indexed_at,
                },
            )
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                cs.RelationshipType.CONTAINS_SECTION.value,
                (cs.NodeLabel.SECTION.value, cs.UniqueKeyType.QUALIFIED_NAME.value, preamble_qn),
            )
            all_section_info.append({
                "qualified_name": preamble_qn,
                "title": "Preamble",
                "start_line": 0,
                "end_line": max(0, preamble_line_count - 1),
                "level": 0,
            })
            stats["sections"] += 1

        # Create Section nodes and relationships, collect section info
        for section in doc.sections:
            section_infos = self._store_section(doc.path, section, doc.path, None, ingestor, stats, indexed_at)
            all_section_info.extend(section_infos)

        # For documents without sections (e.g., plain text files),
        # create a synthetic "Document Content" section so chunks have a section to belong to.
        # Use "#synthetic:" prefix instead of "#L0:" to avoid collision with real sections
        # that might start at line 0 (edge case for malformed documents).
        # Note: This is mutually exclusive with preamble section creation.
        if will_create_synthetic:
            synthetic_qn = f"{doc.path}#synthetic:Document Content"
            ingestor.ensure_node_batch(
                cs.NodeLabel.SECTION.value,
                {
                    cs.UniqueKeyType.QUALIFIED_NAME.value: synthetic_qn,
                    "workspace": self.workspace,
                    "title": "Document Content",
                    "level": 1,
                    "start_line": 0,
                    "end_line": doc.content.count("\n"),
                    "content_snippet": doc.content[:500],
                    "indexed_at": indexed_at,
                },
            )
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                cs.RelationshipType.CONTAINS_SECTION.value,
                (cs.NodeLabel.SECTION.value, cs.UniqueKeyType.QUALIFIED_NAME.value, synthetic_qn),
            )
            all_section_info.append({
                "qualified_name": synthetic_qn,
                "title": "Document Content",
                "start_line": 0,
                "end_line": doc.content.count("\n"),
                "level": 1,
            })
            stats["sections"] = 1

        return stats, all_section_info, indexed_at

    def _store_section(
        self,
        doc_path: str,
        section: ExtractedSection,
        parent_path: str,
        parent_qn: str | None,
        ingestor: MemgraphIngestor,
        stats: dict,
        indexed_at: str,
    ) -> list[dict]:
        """
        Recursively store section and its subsections.

        Args:
            doc_path: Document path
            section: Section to store
            parent_path: Path for qualified name construction
            parent_qn: Parent section's qualified name (None for top-level)
            ingestor: MemgraphIngestor instance
            stats: Stats dict to update
            indexed_at: ISO timestamp for the indexing operation

        Returns:
            List of section info dicts with qualified_name, title, start_line, end_line, level
            (includes this section and all subsections)
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
                "indexed_at": indexed_at,
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

        # Collect section info for chunk-to-section matching
        section_info: list[dict] = [{
            "qualified_name": section_qn,
            "title": section.title,
            "start_line": section.start_line,
            "end_line": section.end_line,
            "level": section.level,
        }]

        # Recursively process subsections
        for subsection in section.subsections:
            subsection_infos = self._store_section(
                doc_path, subsection, section_qn, section_qn, ingestor, stats, indexed_at
            )
            section_info.extend(subsection_infos)

        return section_info

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
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=f"Embedding generation failed: {type(e).__name__}: {e}",
                ) from e
            # Return a pseudo-chunk for the fallback case
            fallback_chunk = DocumentChunk(
                content=doc.content[:1000],
                section_title="",
                start_line=0,  # 0-indexed to match chunking.py convention
                end_line=self.chunker._count_lines(doc.content[:1000]) - 1,
                token_count=self.chunker.count_tokens(doc.content[:1000]),
                document_path=doc.path,
                chunk_index=0,
            )
            return ([fallback_chunk], [doc_embedding])

        # Filter out empty and tiny chunks to avoid API errors and meaningless embeddings
        # Tiny chunks (<10 tokens) like "```" or "```python" provide no semantic value
        MIN_CHUNK_TOKENS = 10
        non_empty_chunks = [
            (i, c) for i, c in enumerate(chunks)
            if c.content.strip() and c.token_count >= MIN_CHUNK_TOKENS
        ]
        if not non_empty_chunks:
            logger.warning(
                f"All chunks in {doc.path} are empty or too small (<{MIN_CHUNK_TOKENS} tokens), "
                "skipping embedding"
            )
            return ([], [])

        chunk_contents = [c.content for i, c in non_empty_chunks]
        batch_size = settings.VECTOR_EMBEDDING_BATCH_SIZE
        try:
            embeddings = provider.embed_batch(chunk_contents, batch_size=batch_size)
        except Exception as e:
            raise ExtractionException(
                path=doc.path,
                error_type=ErrorType.EMBEDDING_ERROR,
                message=f"Embedding batch generation failed: {type(e).__name__}: {e}",
            ) from e

        # Validate embedding count matches non-empty chunk count
        if len(embeddings) != len(non_empty_chunks):
            raise ExtractionException(
                path=doc.path,
                error_type=ErrorType.EMBEDDING_ERROR,
                message=f"Embedding provider returned {len(embeddings)} embeddings for {len(non_empty_chunks)} non-empty chunks",
            )

        # Validate embedding quality (check for NaN, None, and zero vectors)
        validated_embeddings = []
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=f"Embedding provider returned None for chunk {i}",
                )
            # Check for NaN values
            if any(isinstance(v, float) and math.isnan(v) for v in embedding):
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=f"Embedding for chunk {i} contains NaN values",
                )
            # Check for all-zero embedding (indicates failure)
            if all(v == 0.0 for v in embedding):
                logger.warning(f"Embedding for chunk {i} is all zeros, may indicate embedding failure")
            validated_embeddings.append(embedding)

        # Return chunks and validated embeddings (without original indices)
        chunks_list = [c for i, c in non_empty_chunks]
        return (chunks_list, validated_embeddings)

    def _store_chunks_with_embeddings(
        self,
        doc: ExtractedDocument,
        embeddings_data: tuple[list, list[list[float]]],
        section_info: list[dict],
        ingestor: MemgraphIngestor,
        indexed_at: str,
    ) -> int:
        """Store chunks with pre-computed embeddings.

        Args:
            doc: Extracted document
            embeddings_data: Tuple of (non_empty_chunks, embeddings) from _prepare_embeddings
            section_info: List of section info dicts for chunk-to-section matching
            ingestor: MemgraphIngestor instance
            indexed_at: ISO timestamp for the indexing operation

        Returns:
            Number of chunks stored

        Note:
            Every chunk MUST get a BELONGS_TO_SECTION relationship. If _find_section_for_chunk
            returns None (edge case for malformed documents), we use the first available section
            as a fallback to prevent orphaned chunks.
        """
        non_empty_chunks, embeddings = embeddings_data

        if not non_empty_chunks:
            return 0

        # Safety check: ensure we have sections for chunk-to-section mapping
        if not section_info:
            logger.error(
                f"No sections available for chunks in {doc.path}. "
                "This indicates a bug in section creation. Skipping chunk storage."
            )
            return 0

        # Determine fallback section for chunks that don't match any section
        # (shouldn't happen with synthetic section creation, but safety fallback)
        fallback_section = section_info[0]

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
                    "indexed_at": indexed_at,
                },
            )
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                cs.RelationshipType.CONTAINS_CHUNK.value,
                (cs.NodeLabel.CHUNK.value, cs.UniqueKeyType.QUALIFIED_NAME.value, chunk.qualified_name),
            )

            # Find matching section for this chunk and create relationship
            # GUARANTEE: Every chunk gets BELONGS_TO_SECTION relationship
            matching_section = self._find_section_for_chunk(chunk, section_info)
            if matching_section:
                ingestor.ensure_relationship_batch(
                    (cs.NodeLabel.CHUNK.value, cs.UniqueKeyType.QUALIFIED_NAME.value, chunk.qualified_name),
                    cs.RelationshipType.BELONGS_TO_SECTION.value,
                    (cs.NodeLabel.SECTION.value, cs.UniqueKeyType.QUALIFIED_NAME.value, matching_section["qualified_name"]),
                )
            else:
                # Fallback: Use first section (typically synthetic section for plain text)
                # This branch is reached when matching_section is None
                logger.warning(
                    f"Chunk {chunk.qualified_name} has no overlapping section, "
                    f"using fallback section: {fallback_section['qualified_name']}"
                )
                ingestor.ensure_relationship_batch(
                    (cs.NodeLabel.CHUNK.value, cs.UniqueKeyType.QUALIFIED_NAME.value, chunk.qualified_name),
                    cs.RelationshipType.BELONGS_TO_SECTION.value,
                    (cs.NodeLabel.SECTION.value, cs.UniqueKeyType.QUALIFIED_NAME.value, fallback_section["qualified_name"]),
                )

        return len(non_empty_chunks)

    def _find_section_for_chunk(
        self,
        chunk: DocumentChunk,
        section_info: list[dict],
    ) -> dict | None:
        """Find the most specific section that contains a chunk.

        Matching logic:
        1. Use line overlap to find candidate sections
        2. Prefer sections with highest overlap (most content in common)
        3. Among equal overlap, prefer deepest (highest level) section
        4. Title match used as tiebreaker only

        Args:
            chunk: DocumentChunk to match
            section_info: List of section info dicts with qualified_name, title,
                          start_line, end_line, level

        Returns:
            Best matching section dict, or None if no overlap found
        """
        candidates: list[tuple[dict, int]] = []  # (section, overlap_lines)

        for section in section_info:
            # Calculate line overlap (inclusive ranges)
            overlap_start = max(chunk.start_line, section["start_line"])
            overlap_end = min(chunk.end_line, section["end_line"])
            overlap_lines = overlap_end - overlap_start + 1 if overlap_start <= overlap_end else 0

            if overlap_lines > 0:
                candidates.append((section, overlap_lines))

        if not candidates:
            return None

        # Sort by: (overlap lines, level descending, title match)
        # Prioritize overlap and level (specificity) over title match since
        # section_title is set from root section during chunking and may not
        # reflect the actual subsection where the chunk content belongs.
        def sort_key(item: tuple[dict, int]) -> tuple[int, int, int]:
            section, overlap = item
            # Level: higher is deeper (more specific)
            # Title match: used as tiebreaker only
            title_match = 1 if chunk.section_title == section["title"] else 0
            return (overlap, section["level"], title_match)

        candidates.sort(key=sort_key, reverse=True)
        return candidates[0][0]

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