"""Document graph updater for indexing documents into the document graph.

Pattern follows GraphUpdater from codebase_rag/graph_updater.py
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import cast

from loguru import logger

from .. import constants as cs
from .. import logs as ls
from ..config import load_cgrignore_patterns, settings
from ..embeddings import get_embedding_provider
from ..services import (
    ErrorContext,
    FailureType,
    LLMErrorGuidance,
    UserExpertiseLevel,
    classify_memgraph_failure,
)
from ..services.graph_service import MemgraphIngestor
from ..types_defs import ResultRow
from ..utils.path_utils import should_skip_path
from . import logs as doc_ls
from .chunking import DocumentChunk, SemanticDocumentChunker
from .concept_extraction import ConceptExtractor, ExtractionResult, LLMConceptExtractor
from .error_handling import (
    DeadLetterQueue,
    ErrorType,
    ExtractionError,
    ExtractionException,
)
from .extractors import ExtractedDocument, ExtractedSection, get_extractor_for_file
from .utils.reference_extractor import extract_code_references
from .versioning import ContentVersionTracker, VersionCache


class DocumentGraphUnavailableError(Exception):
    """Raised when the document graph is not available."""

    def __init__(
        self,
        message: str,
        *,
        suggested_action: str | None = None,
        original_error: Exception | None = None,
        failure_type: FailureType | None = None,
        should_retry: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.suggested_action = suggested_action
        self.original_error = original_error
        self.failure_type = failure_type
        self.should_retry = should_retry


# Workspace must be a safe identifier (alphanumeric, underscore, hyphen)
# Same validation pattern as document_query.py for consistency
WORKSPACE_PATTERN = re.compile(r"^[\w\-]+$")


def _find_vector_index(
    rows: list[ResultRow],
    index_name: str,
) -> ResultRow | None:
    for row in rows:
        if str(row.get("index_name") or "") == index_name:
            return row
    return None


def _read_vector_index_dimension(
    index_info: ResultRow | None,
    index_name: str = "unknown",
) -> int | None:
    """Read dimension from vector index info with improved type handling.

    Args:
        index_info: Row from database query
        index_name: Name of the index for logging context

    Returns:
        Validated dimension as int, or None if not found/invalid
    """
    if index_info is None:
        logger.debug(f"Index '{index_name}': no info available")
        return None

    raw_dimension = index_info.get("dimension")
    if raw_dimension is None:
        logger.debug(f"Index '{index_name}': dimension field not found")
        return None

    # Handle int directly
    if isinstance(raw_dimension, int):
        return _validate_dimension(raw_dimension, index_name)

    # Handle float (e.g., 768.0 from some database configs)
    if isinstance(raw_dimension, float):
        if raw_dimension.is_integer():
            dimension = int(raw_dimension)
            logger.debug(
                f"Index '{index_name}': converted float {raw_dimension} to int {dimension}"
            )
            return _validate_dimension(dimension, index_name)
        logger.warning(
            f"Index '{index_name}': non-integer float dimension {raw_dimension}"
        )
        return None

    # Handle string
    if isinstance(raw_dimension, str):
        try:
            dimension = int(raw_dimension)
            return _validate_dimension(dimension, index_name)
        except ValueError:
            logger.warning(
                f"Index '{index_name}': invalid string dimension '{raw_dimension}'"
            )
            return None

    # Unknown type
    logger.warning(
        f"Index '{index_name}': unexpected dimension type {type(raw_dimension).__name__}: {raw_dimension!r}"
    )
    return None


def _validate_dimension(dimension: int, index_name: str) -> int | None:
    """Validate dimension value range."""
    if dimension <= 0:
        logger.warning(
            f"Index '{index_name}': invalid dimension {dimension} (must be positive)"
        )
        return None
    if dimension > 10000:
        logger.warning(
            f"Index '{index_name}': dimension {dimension} exceeds max (10000)"
        )
        return None
    return dimension


def _check_graph_availability(
    ingestor: MemgraphIngestor,
    graph_type: str = "document",
) -> None:
    """Check if the graph database is available (sync version with static fallback).

    Performs a simple health check query and raises DocumentGraphUnavailableError
    with actionable guidance if the graph is not available.

    Per LLM-First design:
    - Classification is deterministic (via classify_memgraph_failure)
    - User-facing guidance uses static fallback (for sync contexts)

    For async version with LLM guidance, use _check_graph_availability_async.

    Args:
        ingestor: MemgraphIngestor instance to test
        graph_type: "code" or "document" for error messages

    Raises:
        DocumentGraphUnavailableError: If the graph is not available
    """
    try:
        ingestor.fetch_all("RETURN 1 as health")
    except Exception as e:
        # 1. Deterministic classification (fast, reliable)
        classification = classify_memgraph_failure(e)

        # 2. Use static fallback guidance (no LLM in sync context)
        guidance_generator = LLMErrorGuidance(model_call=None)
        context = ErrorContext(
            operation_type="document_indexing" if graph_type == "document" else "code_ingestion",
            error_category=classification.failure_type,
            graph_type=graph_type,
            embedding_provider=settings.EMBEDDING_PROVIDER,
            user_expertise=UserExpertiseLevel.INTERMEDIATE,
        )
        # Note: Using _get_static_guidance directly to avoid async
        guidance = guidance_generator._get_static_guidance(e, context, classification)

        raise DocumentGraphUnavailableError(
            message=guidance.explanation,
            suggested_action=guidance.suggested_fix,
            original_error=e,
            failure_type=classification.failure_type,
            should_retry=guidance.should_retry,
        ) from e


async def _check_graph_availability_async(
    ingestor: MemgraphIngestor,
    graph_type: str = "document",
    llm_guidance: LLMErrorGuidance | None = None,
) -> None:
    """Check if the graph database is available (async version with LLM guidance).

    Performs a simple health check query and raises DocumentGraphUnavailableError
    with LLM-generated actionable guidance if the graph is not available.

    Per LLM-First design:
    - Classification is deterministic (via classify_memgraph_failure)
    - User-facing guidance is LLM-generated (contextual, user-friendly)

    Args:
        ingestor: MemgraphIngestor instance to test
        graph_type: "code" or "document" for error messages
        llm_guidance: Optional LLM guidance generator. If None, uses static fallback.

    Raises:
        DocumentGraphUnavailableError: If the graph is not available
    """
    try:
        ingestor.fetch_all("RETURN 1 as health")
    except Exception as e:
        # 1. Deterministic classification (fast, reliable)
        classification = classify_memgraph_failure(e)

        # 2. Build error context for LLM guidance
        context = ErrorContext(
            operation_type="document_indexing" if graph_type == "document" else "code_ingestion",
            error_category=classification.failure_type,
            graph_type=graph_type,
            embedding_provider=settings.EMBEDDING_PROVIDER,
            user_expertise=UserExpertiseLevel.INTERMEDIATE,
        )

        # 3. Generate LLM-based guidance (or fallback to static)
        guidance_generator = llm_guidance or LLMErrorGuidance(model_call=None)
        guidance = await guidance_generator.generate_guidance(e, context, classification)

        raise DocumentGraphUnavailableError(
            message=guidance.explanation,
            suggested_action=guidance.suggested_fix,
            original_error=e,
            failure_type=classification.failure_type,
            should_retry=guidance.should_retry,
        ) from e


def ensure_document_vector_index(
    ingestor: MemgraphIngestor,
    dimension: int,
    clear_existing_embeddings: bool = True,
    force_recreate: bool = False,
) -> None:
    if not dimension or dimension <= 0:
        raise ExtractionException(
            path="",
            error_type=ErrorType.EMBEDDING_ERROR,
            message=f"Invalid embedding dimension: {dimension}. "
            "Check EMBEDDING_MODEL configuration.",
        )

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

    existing_index = None
    try:
        existing_indexes = ingestor.fetch_all("SHOW VECTOR INDEX INFO;")
        existing_index = _find_vector_index(existing_indexes, index_name)
    except Exception:
        existing_index = None

    existing_dimension = _read_vector_index_dimension(existing_index, index_name)
    needs_recreate = force_recreate

    if existing_index is not None and not force_recreate:
        if existing_dimension == dimension:
            logger.info(f"Vector index '{index_name}' already exists")
            return

        logger.warning(
            f"Vector index '{index_name}' has dimension {existing_dimension}, "
            f"recreating it for dimension {dimension}"
        )
        needs_recreate = True

    if existing_index is not None and needs_recreate:
        if clear_existing_embeddings:
            ingestor.execute_write(
                """
                MATCH (n:Chunk)
                SET n.embedding = NULL
                """,
                {},
            )
        try:
            ingestor.execute_write(f"DROP VECTOR INDEX {index_name};", {})
        except Exception as exc:
            logger.debug(f"Failed to drop vector index '{index_name}': {exc}")

    try:
        ingestor.execute_write(cypher, {})
        logger.info(
            f"Created vector index '{index_name}' for Chunk nodes "
            f"(dim={dimension}, capacity={capacity})"
        )
    except Exception as exc:
        error_str = str(exc).lower()
        if "already exists" in error_str or "duplicate" in error_str:
            logger.info(f"Vector index '{index_name}' already exists")
        else:
            logger.error(f"Failed to create vector index '{index_name}': {exc}")


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
            ".cgr",
            ".embedding_cache",
            "grammars",  # Tree-sitter grammar submodules (not user code)
        }
    )

    def __init__(
        self,
        host: str,
        port: int,
        repo_path: Path,
        batch_size: int = 1000,
        workspace: str = "default",
        exclude_paths: frozenset[str] | None = None,
        unignore_paths: frozenset[str] | None = None,
        concept_extractor: ConceptExtractor | None = None,
        embeddings_enabled: bool | None = None,
        embeddings_required: bool | None = None,
        username: str | None = None,
        password: str | None = None,
        concept_host: str | None = None,
        concept_port: int | None = None,
        concept_username: str | None = None,
        concept_password: str | None = None,
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
        self.concept_extractor = concept_extractor or (
            LLMConceptExtractor(
                timeout=settings.DOC_CONCEPT_BASE_TIMEOUT,
                max_timeout=settings.DOC_CONCEPT_MAX_TIMEOUT,
            )
            if settings.DOC_CONCEPT_EXTRACTION_ENABLED
            else None
        )
        self._concept_indexes_ensured = False

        # Embedding mode configuration (graceful degradation support)
        self.embeddings_enabled = (
            embeddings_enabled if embeddings_enabled is not None
            else settings.DOC_EMBEDDINGS_ENABLED
        )
        self.embeddings_required = (
            embeddings_required if embeddings_required is not None
            else settings.DOC_EMBEDDINGS_REQUIRED
        )

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
        self.username = username if username is not None else settings.DOC_MEMGRAPH_USERNAME
        self.password = password if password is not None else settings.DOC_MEMGRAPH_PASSWORD
        self.concept_host = concept_host if concept_host is not None else settings.CONCEPT_MEMGRAPH_HOST
        self.concept_port = concept_port if concept_port is not None else settings.CONCEPT_MEMGRAPH_PORT
        self.concept_username = concept_username if concept_username is not None else settings.CONCEPT_MEMGRAPH_USERNAME
        self.concept_password = concept_password if concept_password is not None else settings.CONCEPT_MEMGRAPH_PASSWORD

        # Ensure metadata directory exists before initializing caches
        cgr_dir = self.base_path / ".cgr"
        cgr_dir.mkdir(parents=True, exist_ok=True)

        self.version_tracker = ContentVersionTracker()
        self.version_cache = VersionCache(cgr_dir / "doc_versions.json")
        self.dead_letter_queue = DeadLetterQueue(cgr_dir / "doc_errors")
        self.chunker = SemanticDocumentChunker()

        cgrignore = load_cgrignore_patterns(self.base_path)
        combined_excludes = set(cgrignore.exclude)
        if exclude_paths:
            combined_excludes.update(exclude_paths)
        self.exclude_paths = frozenset(combined_excludes) if combined_excludes else None

        combined_unignores = set(cgrignore.unignore)
        if unignore_paths:
            combined_unignores.update(unignore_paths)
        self.unignore_paths = (
            frozenset(combined_unignores) if combined_unignores else None
        )

        # Cache embedding provider to avoid recreation per document
        config = settings.active_embedding_config
        self._embedding_provider = get_embedding_provider(config=config)

        # Cache supported extensions from config
        self._supported_extensions = set(settings.DOC_SUPPORTED_EXTENSIONS)
        self._code_reference_qns: set[str] = set()
        self._code_reference_simple_lookup: dict[str, tuple[str, ...]] = {}

    def _is_excluded_path(self, file_path: Path) -> bool:
        return any(
            part in self.EXCLUDED_DIRS or part.endswith(".egg-info")
            for part in file_path.parts
        )

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
        from ..exceptions import EmbeddingError

        if isinstance(exc, EmbeddingError):
            return ErrorType.EMBEDDING_ERROR
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

    def _handle_extraction_error(
        self, doc_path: Path, e: ExtractionException, stats: dict[str, object]
    ) -> None:
        if e.error_type in (ErrorType.FILE_NOT_FOUND, ErrorType.NOT_A_FILE):
            logger.warning(doc_ls.DOC_FILE_MISSING_SKIP.format(path=doc_path, error=e))
        else:
            logger.opt(exception=True).error(
                doc_ls.DOC_EXTRACTION_FAILED.format(path=doc_path, error=e)
            )
        stats["failed"] += 1
        self.version_cache.remove(str(doc_path))
        try:
            self.dead_letter_queue.enqueue(e.to_extraction_error())
        except Exception as dlq_error:
            logger.warning(
                doc_ls.DOC_DLQ_ENQUEUE_FAILED.format(path=doc_path, error=dlq_error)
            )

    def _handle_generic_error(
        self, doc_path: Path, e: Exception, stats: dict[str, object]
    ) -> None:
        logger.opt(exception=True).error(
            doc_ls.DOC_EXTRACTION_FAILED.format(path=doc_path, error=e)
        )
        stats["failed"] += 1
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
            logger.warning(
                doc_ls.DOC_DLQ_ENQUEUE_FAILED.format(path=doc_path, error=dlq_error)
            )

    def _cleanup_dlq(self) -> None:
        from codebase_rag.config import settings

        self.dead_letter_queue.cleanup_stale_errors()
        queue_size = self.dead_letter_queue.size()
        if queue_size > settings.DOC_ERRORS_WARNING_THRESHOLD:
            logger.warning(doc_ls.DOC_DLQ_SIZE_WARNING.format(count=queue_size))

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
            "graph_available": True,
        }

        try:
            try:
                ingestor = MemgraphIngestor(
                    host=self.host,
                    port=self.port,
                    batch_size=self.batch_size,
                    connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
                ).__enter__()
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.error(doc_ls.DOC_GRAPH_CONNECT_FAILED.format(error=e))
                stats["graph_available"] = False
                stats["graph_error"] = str(e)
                return stats

            # Create concept ingestor if concept extraction is enabled
            concept_ingestor = None
            if self.concept_extractor and settings.CONCEPT_MEMGRAPH_ENABLED:
                try:
                    concept_ingestor = MemgraphIngestor(
                        host=self.concept_host,
                        port=self.concept_port,
                        batch_size=settings.CONCEPT_MEMGRAPH_BATCH_SIZE,
                        connection_timeout=settings.CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT,
                        username=self.concept_username,
                        password=self.concept_password,
                    ).__enter__()
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.warning(
                        f"Concept extraction enabled but concept graph instance unavailable "
                        f"({self.concept_host}:{self.concept_port}): {e} — "
                        "concepts will not be stored"
                    )

            try:
                try:
                    _check_graph_availability(ingestor, graph_type="document")
                except DocumentGraphUnavailableError as e:
                    logger.error(doc_ls.DOC_GRAPH_UNAVAILABLE.format(error=e.message))
                    stats["graph_available"] = False
                    stats["graph_error"] = e.message
                    return stats

                setup_ops = [
                    ("ensure_constraints", lambda: ingestor.ensure_constraints()),
                    ("ensure_vector_index", lambda: self._ensure_vector_index(ingestor)),
                    ("ensure_document_indexes", lambda: self._ensure_document_indexes(ingestor)),
                    ("refresh_code_references", lambda: self._refresh_code_reference_index()),
                ]
                for op_name, op_fn in setup_ops:
                    try:
                        op_fn()
                    except Exception as e:
                        logger.error(doc_ls.DOC_SETUP_OP_FAILED.format(op=op_name, error=e))

                documents = self._collect_documents()

                current_paths = {str(d) for d in documents}
                for cached_path in self.version_cache.keys():
                    if cached_path not in current_paths and not Path(cached_path).exists():
                        logger.debug(doc_ls.DOC_CACHE_PRUNE.format(path=cached_path))
                        self.version_cache.remove(cached_path)
                self.version_cache.save()

                original_count = len(documents)
                documents = [d for d in documents if d.exists()]
                if len(documents) < original_count:
                    logger.info(
                        doc_ls.DOC_PRE_VERIFICATION_FILTERED.format(
                            count=original_count - len(documents)
                        )
                    )

                try:
                    deleted_stale = self._delete_stale_documents(documents, ingestor, concept_ingestor=concept_ingestor)
                except Exception as e:
                    logger.warning(doc_ls.DOC_STALE_CLEANUP_FAILED.format(error=e))
                    deleted_stale = 0

                try:
                    deleted_excluded = self._delete_excluded_documents(documents, ingestor, concept_ingestor=concept_ingestor)
                except Exception as e:
                    logger.warning(doc_ls.DOC_EXCLUDED_CLEANUP_FAILED.format(error=e))
                    deleted_excluded = 0

                stats["total_documents"] = len(documents)
                stats["cleanup_deleted_files"] = deleted_stale
                stats["cleanup_excluded_files"] = deleted_excluded

                if deleted_stale > 0 or deleted_excluded > 0:
                    logger.info(
                        f"Cleanup: {deleted_stale} deleted files, {deleted_excluded} excluded files removed"
                    )

                total_documents = len(documents)
                logger.info(f"Found {total_documents} documents to index")

                for index, doc_path in enumerate(documents, start=1):
                    logger.info(
                        f"Indexing document {index}/{total_documents}: {doc_path}"
                    )
                    try:
                        result = self._process_document(doc_path, ingestor, concept_ingestor=concept_ingestor, force=force, stats=stats)
                        if result == "indexed":
                            stats["indexed"] += 1
                        elif result == "skipped":
                            stats["skipped"] += 1
                    except ExtractionException as e:
                        self._handle_extraction_error(doc_path, e, stats)
                    except Exception as e:
                        self._handle_generic_error(doc_path, e, stats)

                    if (
                        settings.DOC_INCREMENTAL_FLUSH_INTERVAL > 0
                        and index % settings.DOC_INCREMENTAL_FLUSH_INTERVAL == 0
                    ):
                        try:
                            flush_stats = ingestor.flush_all_with_stats()
                            if flush_stats["nodes_failed"] > 0 or flush_stats["relationships_failed"] > 0:
                                logger.warning(
                                    doc_ls.DOC_INCREMENTAL_FLUSH_PARTIAL.format(
                                        index=index, stats=flush_stats
                                    )
                                )
                            else:
                                logger.debug(
                                    doc_ls.DOC_INCREMENTAL_FLUSH_OK.format(
                                        index=index, stats=flush_stats
                                    )
                                )
                            self.version_cache.save()
                        except Exception as e:
                            logger.error(
                                doc_ls.DOC_INCREMENTAL_FLUSH_FAILED.format(
                                    index=index, error=e
                                )
                            )

                try:
                    flush_stats = ingestor.flush_all_with_stats()
                    if flush_stats["nodes_failed"] > 0 or flush_stats["relationships_failed"] > 0:
                        logger.warning(
                            doc_ls.DOC_FINAL_FLUSH_PARTIAL.format(stats=flush_stats)
                        )
                    else:
                        logger.info(doc_ls.DOC_FINAL_FLUSH_OK.format(stats=flush_stats))
                except Exception as e:
                    logger.opt(exception=True).error(
                        doc_ls.DOC_FINAL_FLUSH_FAILED.format(error=e)
                    )
                    stats["flush_error"] = str(e)

                try:
                    section_result = ingestor.fetch_all(
                        "MATCH (s:Section {workspace: $ws}) RETURN count(s) as count",
                        {"ws": self.workspace},
                    )
                    chunk_result = ingestor.fetch_all(
                        "MATCH (c:Chunk {workspace: $ws}) RETURN count(c) as count",
                        {"ws": self.workspace},
                    )
                    if section_result and len(section_result) > 0:
                        stats["sections_created"] = section_result[0].get("count", 0)
                    if chunk_result and len(chunk_result) > 0:
                        stats["chunks_created"] = chunk_result[0].get("count", 0)
                except Exception as e:
                    logger.warning(f"Could not query stats from graph: {e}")

                try:
                    logger.debug("Saving version cache to disk")
                    self.version_cache.save()
                except Exception as e:
                    logger.warning(f"Could not save version cache: {e}")

                self._cleanup_dlq()

                logger.info(f"Document indexing complete: {stats}")
                return stats
            finally:
                if concept_ingestor is not None:
                    try:
                        concept_ingestor.__exit__(None, None, None)
                    except Exception as e:
                        logger.warning(f"Error closing concept ingestor: {e}")
                ingestor.__exit__(None, None, None)
        finally:
            try:
                self._embedding_provider.close()
            except Exception as e:
                logger.warning(doc_ls.EMBEDDING_PROVIDER_CLOSE_FAILED.format(error=e))

    async def run_async(self, force: bool = False) -> dict:
        """Async version of run() for concurrent processing."""
        stats = {
            "total_documents": 0,
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "sections_created": 0,
            "chunks_created": 0,
            "graph_available": True,
        }

        try:
            try:
                ingestor = await asyncio.to_thread(
                    MemgraphIngestor(
                        host=self.host,
                        port=self.port,
                        batch_size=self.batch_size,
                        connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
                    ).__enter__
                )
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.error(doc_ls.DOC_GRAPH_CONNECT_FAILED.format(error=e))
                stats["graph_available"] = False
                stats["graph_error"] = str(e)
                return stats

            # Create concept ingestor if concept extraction is enabled
            concept_ingestor = None
            if self.concept_extractor and settings.CONCEPT_MEMGRAPH_ENABLED:
                try:
                    concept_ingestor = await asyncio.to_thread(
                        MemgraphIngestor(
                            host=self.concept_host,
                            port=self.concept_port,
                            batch_size=settings.CONCEPT_MEMGRAPH_BATCH_SIZE,
                            connection_timeout=settings.CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT,
                            username=self.concept_username,
                            password=self.concept_password,
                        ).__enter__
                    )
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.warning(
                        f"Concept extraction enabled but concept graph instance unavailable "
                        f"({self.concept_host}:{self.concept_port}): {e} — "
                        "concepts will not be stored"
                    )

            try:
                try:
                    await _check_graph_availability_async(ingestor, graph_type="document")
                except DocumentGraphUnavailableError as e:
                    logger.error(doc_ls.DOC_GRAPH_UNAVAILABLE.format(error=e.message))
                    stats["graph_available"] = False
                    stats["graph_error"] = e.message
                    return stats

                setup_ops = [
                    ("ensure_constraints", lambda: ingestor.ensure_constraints()),
                    ("ensure_vector_index", lambda: self._ensure_vector_index(ingestor)),
                    ("ensure_document_indexes", lambda: self._ensure_document_indexes(ingestor)),
                    ("refresh_code_references", lambda: self._refresh_code_reference_index()),
                ]
                for op_name, op_fn in setup_ops:
                    try:
                        await asyncio.to_thread(op_fn)
                    except Exception as e:
                        logger.error(doc_ls.DOC_SETUP_OP_FAILED.format(op=op_name, error=e))

                try:
                    documents = await asyncio.to_thread(self._collect_documents)
                except Exception as e:
                    logger.error(f"Failed to collect documents: {e}")
                    documents = []

                current_paths = {str(d) for d in documents}
                for cached_path in self.version_cache.keys():
                    if cached_path not in current_paths and not Path(cached_path).exists():
                        logger.debug(doc_ls.DOC_CACHE_PRUNE.format(path=cached_path))
                        self.version_cache.remove(cached_path)
                self.version_cache.save()

                original_count = len(documents)
                documents = [d for d in documents if d.exists()]
                if len(documents) < original_count:
                    logger.info(
                        doc_ls.DOC_PRE_VERIFICATION_FILTERED.format(
                            count=original_count - len(documents)
                        )
                    )

                try:
                    deleted_stale = await asyncio.to_thread(
                        self._delete_stale_documents, documents, ingestor, concept_ingestor
                    )
                except Exception as e:
                    logger.warning(doc_ls.DOC_STALE_CLEANUP_FAILED.format(error=e))
                    deleted_stale = 0

                try:
                    deleted_excluded = await asyncio.to_thread(
                        self._delete_excluded_documents, documents, ingestor, concept_ingestor
                    )
                except Exception as e:
                    logger.warning(doc_ls.DOC_EXCLUDED_CLEANUP_FAILED.format(error=e))
                    deleted_excluded = 0

                stats["total_documents"] = len(documents)
                stats["cleanup_deleted_files"] = deleted_stale
                stats["cleanup_excluded_files"] = deleted_excluded

                total_documents = len(documents)
                logger.info(f"Found {total_documents} documents to index")

                for index, doc_path in enumerate(documents, start=1):
                    logger.info(
                        f"Indexing document {index}/{total_documents}: {doc_path}"
                    )
                    try:
                        result = await self._process_document_async(
                            doc_path, ingestor, concept_ingestor=concept_ingestor, force=force, stats=stats
                        )
                        if result == "indexed":
                            stats["indexed"] += 1
                        elif result == "skipped":
                            stats["skipped"] += 1
                    except ExtractionException as e:
                        self._handle_extraction_error(doc_path, e, stats)
                    except Exception as e:
                        self._handle_generic_error(doc_path, e, stats)

                    if (
                        settings.DOC_INCREMENTAL_FLUSH_INTERVAL > 0
                        and index % settings.DOC_INCREMENTAL_FLUSH_INTERVAL == 0
                    ):
                        try:
                            flush_stats = await asyncio.to_thread(
                                ingestor.flush_all_with_stats
                            )
                            if flush_stats["nodes_failed"] > 0 or flush_stats["relationships_failed"] > 0:
                                logger.warning(
                                    doc_ls.DOC_INCREMENTAL_FLUSH_PARTIAL.format(
                                        index=index, stats=flush_stats
                                    )
                                )
                            else:
                                logger.debug(
                                    doc_ls.DOC_INCREMENTAL_FLUSH_OK.format(
                                        index=index, stats=flush_stats
                                    )
                                )
                            await asyncio.to_thread(self.version_cache.save)
                        except Exception as e:
                            logger.error(
                                doc_ls.DOC_INCREMENTAL_FLUSH_FAILED.format(
                                    index=index, error=e
                                )
                            )

                try:
                    flush_stats = await asyncio.to_thread(ingestor.flush_all_with_stats)
                    if flush_stats["nodes_failed"] > 0 or flush_stats["relationships_failed"] > 0:
                        logger.warning(
                            doc_ls.DOC_FINAL_FLUSH_PARTIAL.format(stats=flush_stats)
                        )
                    else:
                        logger.info(doc_ls.DOC_FINAL_FLUSH_OK.format(stats=flush_stats))
                except Exception as e:
                    logger.opt(exception=True).error(
                        doc_ls.DOC_FINAL_FLUSH_FAILED.format(error=e)
                    )
                    stats["flush_error"] = str(e)

                try:
                    section_result = await asyncio.to_thread(
                        ingestor.fetch_all,
                        "MATCH (s:Section {workspace: $ws}) RETURN count(s) as count",
                        {"ws": self.workspace},
                    )
                    chunk_result = await asyncio.to_thread(
                        ingestor.fetch_all,
                        "MATCH (c:Chunk {workspace: $ws}) RETURN count(c) as count",
                        {"ws": self.workspace},
                    )
                    if section_result and len(section_result) > 0:
                        stats["sections_created"] = section_result[0].get(
                            "count", 0
                        )
                    if chunk_result and len(chunk_result) > 0:
                        stats["chunks_created"] = chunk_result[0].get("count", 0)
                except Exception as e:
                    logger.warning(f"Could not query stats from graph: {e}")

                try:
                    logger.debug("Saving version cache to disk")
                    await asyncio.to_thread(self.version_cache.save)
                except Exception as e:
                    logger.warning(f"Could not save version cache: {e}")

                self._cleanup_dlq()

                logger.info(f"Document indexing complete: {stats}")
                return stats
            finally:
                if concept_ingestor is not None:
                    try:
                        await asyncio.to_thread(concept_ingestor.__exit__, None, None, None)
                    except Exception as e:
                        logger.warning(f"Error closing concept ingestor: {e}")
                await asyncio.to_thread(ingestor.__exit__, None, None, None)
        finally:
            try:
                self._embedding_provider.close()
            except Exception as e:
                logger.warning(doc_ls.EMBEDDING_PROVIDER_CLOSE_FAILED.format(error=e))

    def _ensure_vector_index(self, ingestor: MemgraphIngestor) -> None:
        """Ensure vector index for Chunk embeddings exists.

        Uses the shared document vector backend for index creation.
        This provides consistent index management with clear separation of concerns:
        - VectorBackend handles all vector index creation/management
        - DocumentGraphUpdater handles graph structure only

        Raises:
            ExtractionException: If embedding dimension is invalid (0 or negative).
        """
        from ..vector_backend import get_shared_backend_for_documents

        # Get the shared document backend - this will initialize and create the index
        # if it doesn't exist. The backend manages all vector index creation.
        backend = get_shared_backend_for_documents()
        logger.debug(
            f"Document vector backend initialized for index management "
            f"(backend type: {type(backend).__name__})"
        )

    def _ensure_document_indexes(self, ingestor: MemgraphIngestor) -> None:
        """Create indexes for document graph queries.

        Creates indexes for:
        - workspace property on Document, Section, Chunk nodes (multi-tenant queries)
        - start_line property on Section, Chunk nodes (ORDER BY queries)
        - Composite indexes for workspace+start_line (optimized query patterns)

        Note: These are non-unique indexes for query performance.
        """
        # Single-property indexes
        index_specs = [
            ("Document", "workspace"),
            ("Section", "workspace"),
            ("Chunk", "workspace"),
            ("Section", "start_line"),
            ("Chunk", "start_line"),
        ]

        for label, prop in index_specs:
            try:
                cypher = f"CREATE INDEX ON :{label}({prop});"
                ingestor.execute_write(cypher, {})
                logger.debug(f"Created index on :{label}({prop})")
            except Exception as e:
                error_str = str(e).lower()
                if "already exists" in error_str or "duplicate" in error_str:
                    logger.debug(f"Index on :{label}({prop}) already exists")
                else:
                    logger.warning(f"Failed to create index on :{label}({prop}): {e}")
                    # Non-fatal: indexing can proceed without these indexes

        # Composite indexes for optimized query patterns
        # These improve queries that filter by workspace AND order by start_line
        composite_index_specs = [
            ("Section", ["workspace", "start_line"]),
            ("Chunk", ["workspace", "start_line"]),
        ]

        for label, props in composite_index_specs:
            try:
                props_str = ", ".join(props)
                cypher = f"CREATE INDEX ON :{label}({props_str});"
                ingestor.execute_write(cypher, {})
                logger.debug(f"Created composite index on :{label}({props_str})")
            except Exception as e:
                error_str = str(e).lower()
                if "already exists" in error_str or "duplicate" in error_str:
                    logger.debug(
                        f"Composite index on :{label}({props_str}) already exists"
                    )
                else:
                    logger.warning(
                        f"Failed to create composite index on :{label}({props_str}): {e}"
                    )
                    # Non-fatal: indexing can proceed without these indexes

        logger.info("Document graph indexes ensured")

    def _collect_documents(self) -> list[Path]:
        """Collect all eligible document files."""
        documents: list[Path] = []

        # Use cached supported extensions from config
        supported_extensions = self._supported_extensions

        # Handle single file path
        if self.repo_path.is_file():
            # Security: Check excluded directories for single file
            if self._is_excluded_path(self.repo_path):
                logger.debug(f"Skipping file in excluded directory: {self.repo_path}")
                return documents

            if should_skip_path(
                self.repo_path,
                self.base_path,
                exclude_paths=self.exclude_paths,
                unignore_paths=self.unignore_paths,
            ):
                logger.debug(f"Skipping excluded document path: {self.repo_path}")
                return documents

            # Security: Check extension
            if self.repo_path.suffix.lower() not in supported_extensions:
                logger.debug(f"Skipping unsupported file type: {self.repo_path}")
                return documents

            # Security: Check symlink escape
            if self.repo_path.is_symlink():
                resolved = self.repo_path.resolve()
                if not self._is_path_within_boundary(resolved):
                    logger.debug(
                        f"Skipping symlink pointing outside repo: {self.repo_path}"
                    )
                    return documents
            documents.append(self.repo_path)
            return documents

        # Handle directory path
        for ext in supported_extensions:
            for doc_path in self.repo_path.rglob(f"*{ext}"):
                # Check if any path component is in excluded directories
                if self._is_excluded_path(doc_path):
                    continue

                if should_skip_path(
                    doc_path,
                    self.base_path,
                    exclude_paths=self.exclude_paths,
                    unignore_paths=self.unignore_paths,
                ):
                    continue

                # Check if path is a file
                if not doc_path.is_file():
                    continue

                # Security: Skip symlinks pointing outside repo
                if doc_path.is_symlink():
                    resolved = doc_path.resolve()
                    if not self._is_path_within_boundary(resolved):
                        logger.debug(
                            f"Skipping symlink pointing outside repo: {doc_path}"
                        )
                        continue

                documents.append(doc_path)

        return documents

    def _delete_stale_documents(
        self, documents: list[Path], ingestor: MemgraphIngestor,
        concept_ingestor: MemgraphIngestor | None = None,
    ) -> int:
        if not self.repo_path.is_dir():
            return 0

        current_paths = {str(doc_path) for doc_path in documents}
        stored_documents = ingestor.fetch_all(
            "MATCH (d:Document {workspace: $workspace}) RETURN d.path AS path",
            {"workspace": self.workspace},
        )

        stale_paths: list[str] = []
        for row in stored_documents:
            stored_path = row.get("path")
            if not isinstance(stored_path, str) or not stored_path:
                continue
            if not self._is_document_in_scope(stored_path):
                continue
            if stored_path in current_paths:
                continue
            stale_paths.append(stored_path)

        for stale_path in stale_paths:
            self._delete_document_nodes(stale_path, ingestor, concept_ingestor=concept_ingestor)
            self.version_cache.remove(stale_path)

        if stale_paths:
            logger.info(
                f"Removed {len(stale_paths)} stale documents from graph for workspace {self.workspace}"
            )

        return len(stale_paths)

    def _delete_excluded_documents(
        self,
        current_documents: list[Path],
        ingestor: MemgraphIngestor,
        concept_ingestor: MemgraphIngestor | None = None,
    ) -> int:
        """Remove documents that are now excluded by .cgrignore patterns.

        This handles the case where:
        - Files were previously indexed
        - User added patterns to .cgrignore to exclude them
        - Files still exist on disk

        Args:
            current_documents: List of documents that should be indexed (after filtering)
            ingestor: MemgraphIngestor instance

        Returns:
            Number of excluded documents removed from the graph.
        """
        if not self.repo_path.is_dir():
            return 0

        # Get all stored documents for this workspace
        stored_documents = ingestor.fetch_all(
            "MATCH (d:Document {workspace: $workspace}) RETURN d.path AS path",
            {"workspace": self.workspace},
        )

        current_paths = {str(doc_path) for doc_path in current_documents}
        excluded_count = 0

        for row in stored_documents:
            stored_path = row.get("path")
            if not isinstance(stored_path, str) or not stored_path:
                continue

            # Check if document is in scope
            if not self._is_document_in_scope(stored_path):
                continue

            # If document is in current_paths, it will be processed/re-indexed
            if stored_path in current_paths:
                continue

            # Document exists on disk but is not in current_paths
            # This means it was excluded by .cgrignore or other filters
            stored_path_obj = Path(stored_path)
            if stored_path_obj.exists():
                # File exists but is excluded - remove from graph
                logger.info(
                    f"Removing excluded document from graph: {stored_path} "
                    f"(matched by .cgrignore pattern)"
                )
                self._delete_document_nodes(stored_path, ingestor, concept_ingestor=concept_ingestor)
                self.version_cache.remove(stored_path)
                excluded_count += 1

        if excluded_count > 0:
            logger.info(
                f"Removed {excluded_count} excluded documents from graph "
                f"for workspace {self.workspace}"
            )

        return excluded_count

    def _is_document_in_scope(self, doc_path: str) -> bool:
        stored_path = Path(doc_path)
        if not stored_path.is_absolute():
            return True

        try:
            stored_path.relative_to(self.base_path)
        except ValueError:
            return False

        return True

    def preview_excluded_documents(self) -> list[tuple[str, str | None]]:
        """Preview documents that would be removed by cleanup.

        Returns:
            List of (path, pattern) tuples for documents that would be removed.
            Pattern is None if no specific pattern matched.
        """

        excluded: list[tuple[str, str | None]] = []

        with MemgraphIngestor(
            host=self.host,
            port=self.port,
            batch_size=self.batch_size,
            username=self.username,
            password=self.password,
            connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
        ) as ingestor:
            stored_documents = ingestor.fetch_all(
                "MATCH (d:Document {workspace: $workspace}) RETURN d.path AS path",
                {"workspace": self.workspace},
            )

            current_documents = self._collect_documents()
            current_paths = {str(doc_path) for doc_path in current_documents}

            for row in stored_documents:
                stored_path = row.get("path")
                if not isinstance(stored_path, str) or not stored_path:
                    continue
                if not self._is_document_in_scope(stored_path):
                    continue
                if stored_path in current_paths:
                    continue

                stored_path_obj = Path(stored_path)
                if stored_path_obj.exists():
                    excluded.append((stored_path, "matched .cgrignore pattern"))

        return excluded

    def cleanup_excluded_documents(self) -> int:
        """Remove excluded documents from graph (standalone cleanup).

        Returns:
            Number of documents removed.
        """
        with MemgraphIngestor(
            host=self.host,
            port=self.port,
            batch_size=self.batch_size,
            username=self.username,
            password=self.password,
            connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
        ) as ingestor:
            documents = self._collect_documents()
            return self._delete_excluded_documents(documents, ingestor)

    def check_for_large_excluded_documents(
        self,
        chunk_threshold: int = 100,
        max_display: int = 5,
    ) -> list[tuple[str, int]]:
        """Check if large documents that should be excluded are still in the graph.

        LLM-First Design: This is deterministic infrastructure checking.
        Uses existing .cgrignore patterns and graph queries - no semantic analysis.

        Args:
            chunk_threshold: Minimum chunks to consider a document "large"
            max_display: Maximum number of documents to return

        Returns:
            List of (path, chunk_count) for documents that should be excluded.
        """
        from ..utils.path_utils import should_skip_path

        excluded_large_docs: list[tuple[str, int]] = []

        with MemgraphIngestor(
            host=self.host,
            port=self.port,
            batch_size=self.batch_size,
            username=self.username,
            password=self.password,
            connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
        ) as ingestor:
            # Find large documents in graph
            large_docs = ingestor.fetch_all(
                """
                MATCH (d:Document {workspace: $workspace})
                OPTIONAL MATCH (d)<-[:BELONGS_TO_DOCUMENT]-(c:Chunk)
                WITH d, count(c) AS chunk_count
                WHERE chunk_count >= $threshold
                RETURN d.path AS path, chunk_count
                ORDER BY chunk_count DESC
                LIMIT $limit
                """,
                {
                    "workspace": self.workspace,
                    "threshold": chunk_threshold,
                    "limit": max_display * 2,  # Get extra for filtering
                },
            )

            for row in large_docs:
                doc_path = row.get("path")
                chunk_count = row.get("chunk_count", 0)

                if not doc_path:
                    continue

                path_obj = Path(doc_path)
                if not path_obj.exists():
                    continue

                # Check if this document should now be excluded
                if should_skip_path(
                    path_obj,
                    self.base_path,
                    exclude_paths=self.exclude_paths,
                    unignore_paths=self.unignore_paths,
                ):
                    excluded_large_docs.append((doc_path, chunk_count))
                    if len(excluded_large_docs) >= max_display:
                        break

        return excluded_large_docs

    def _refresh_code_reference_index(self) -> None:
        self._code_reference_qns = set()
        self._code_reference_simple_lookup = {}

        try:
            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=self.batch_size,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
                connection_timeout=settings.MEMGRAPH_CONNECTION_TIMEOUT,
            ) as code_ingestor:
                rows = code_ingestor.fetch_all(
                    """
                    MATCH (n)
                    WHERE n:Function OR n:Method OR n:Class OR n:Module
                    RETURN n.qualified_name AS qualified_name, n.name AS name
                    """
                )
        except Exception as e:
            logger.warning(f"Could not load code reference index from code graph: {e}")
            return

        simple_lookup: dict[str, set[str]] = {}

        for row in rows:
            qualified_name = row.get("qualified_name")
            if not isinstance(qualified_name, str) or not qualified_name:
                continue

            self._code_reference_qns.add(qualified_name)

            simple_name = qualified_name.rsplit(".", 1)[-1]
            if simple_name:
                simple_lookup.setdefault(simple_name, set()).add(qualified_name)

            display_name = row.get("name")
            if isinstance(display_name, str) and display_name:
                simple_lookup.setdefault(display_name, set()).add(qualified_name)

        self._code_reference_simple_lookup = {
            name: tuple(sorted(values)) for name, values in simple_lookup.items()
        }

    def _resolve_code_reference_names(self, reference_names: list[str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()

        for reference_name in reference_names:
            candidate = reference_name.strip()
            if not candidate:
                continue

            resolved_qn: str | None = None
            if candidate in self._code_reference_qns:
                resolved_qn = candidate
            else:
                matches = self._code_reference_simple_lookup.get(candidate, ())
                if len(matches) == 1:
                    resolved_qn = matches[0]
                else:
                    simple_name = candidate.rsplit(".", 1)[-1]
                    simple_matches = self._code_reference_simple_lookup.get(
                        simple_name, ()
                    )
                    if len(simple_matches) == 1:
                        resolved_qn = simple_matches[0]

            if resolved_qn and resolved_qn not in seen:
                seen.add(resolved_qn)
                resolved.append(resolved_qn)

        return resolved

    def _extract_chunk_reference_names(self, content: str) -> list[str]:
        return [
            reference.qualified_name for reference in extract_code_references(content)
        ]

    def _process_document(
        self,
        file_path: Path,
        ingestor: MemgraphIngestor,
        force: bool = False,
        stats: dict[str, object] | None = None,
        concept_ingestor: MemgraphIngestor | None = None,
    ) -> str:
        """
        Process single document.

        Returns:
            "indexed", "skipped", or "failed"
        """
        if not file_path.exists():
            logger.debug(doc_ls.DOC_PREFLIGHT_MISSING.format(path=file_path))
            self.version_cache.remove(str(file_path))
            return "skipped"

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
        resolved_code_references = self._resolve_code_reference_names(
            doc.code_references
        )

        # Generate embeddings BEFORE deleting existing nodes
        # This ensures rollback safety: if embedding fails, old data is preserved
        # Uses graceful degradation: continues without embeddings if unavailable
        chunks = list(self.chunker.chunk_document(doc))
        embeddings_data = self._prepare_embeddings_with_fallback(doc, chunks)

        # Only delete existing nodes after embeddings are validated
        self._delete_document_nodes(doc.path, ingestor, concept_ingestor=concept_ingestor)

        # Store document and sections in graph, get section info for chunk matching
        store_stats, section_info, indexed_at = self._store_document(
            doc,
            ingestor,
            resolved_code_references,
        )

        # Store pre-computed chunks with embeddings and section relationships
        chunk_count = self._store_chunks_with_embeddings(
            doc, embeddings_data, section_info, ingestor, indexed_at
        )

        # Extract and store concepts from chunks (sync wrapper)
        if self.concept_extractor and chunks:
            try:
                asyncio.get_running_loop()
                logger.debug(
                    "Skipping concept extraction in running event loop (sync path)"
                )
            except RuntimeError:
                asyncio.run(
                    self._extract_and_store_concepts(
                        chunks, ingestor, self.workspace, stats, concept_ingestor=concept_ingestor
                    )
                )

        # Update version cache
        version = self.version_tracker.create_version(doc)
        self.version_cache.set(version)

        logger.debug(
            f"Indexed {file_path}: {store_stats['sections']} sections, {chunk_count} chunks"
        )
        return "indexed"

    def _delete_document_nodes(self, doc_path: str, ingestor: MemgraphIngestor, concept_ingestor: MemgraphIngestor | None = None) -> None:
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
            # Clean up orphaned concepts before deleting chunks
            if self.concept_extractor:
                self._cleanup_concepts_for_document(doc_path, ingestor, concept_ingestor=concept_ingestor)

            ingestor.execute_write(
                """
                MATCH (d:Document {path: $path, workspace: $workspace})
                    -[:CONTAINS_SECTION]->(s:Section)
                DETACH DELETE s
                """,
                {"path": doc_path, "workspace": workspace},
            )

            ingestor.execute_write(
                """
                MATCH (d:Document {path: $path, workspace: $workspace})
                    -[:CONTAINS_CHUNK]->(c:Chunk)
                DETACH DELETE c
                """,
                {"path": doc_path, "workspace": workspace},
            )

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
        stats: dict[str, object] | None = None,
        concept_ingestor: MemgraphIngestor | None = None,
    ) -> str:
        """Async version of _process_document."""
        if not file_path.exists():
            logger.debug(doc_ls.DOC_PREFLIGHT_MISSING.format(path=file_path))
            self.version_cache.remove(str(file_path))
            return "skipped"

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
        resolved_code_references = self._resolve_code_reference_names(
            doc.code_references
        )

        # Generate embeddings BEFORE deleting existing nodes (rollback safety)
        # Uses graceful degradation: continues without embeddings if unavailable
        chunks = list(self.chunker.chunk_document(doc))
        embeddings_data = await asyncio.to_thread(
            self._prepare_embeddings_with_fallback, doc, chunks
        )

        # Only delete existing nodes after embeddings are validated
        await asyncio.to_thread(self._delete_document_nodes, doc.path, ingestor, concept_ingestor=concept_ingestor)

        # Store document and sections (run in thread to avoid blocking)
        store_stats, section_info, indexed_at = await asyncio.to_thread(
            self._store_document,
            doc,
            ingestor,
            resolved_code_references,
        )
        chunk_count = await asyncio.to_thread(
            self._store_chunks_with_embeddings,
            doc,
            embeddings_data,
            section_info,
            ingestor,
            indexed_at,
        )

        # Extract and store concepts from chunks
        if self.concept_extractor and chunks:
            await self._extract_and_store_concepts(chunks, ingestor, self.workspace, stats, concept_ingestor=concept_ingestor)

        # Update version
        version = self.version_tracker.create_version(doc)
        self.version_cache.set(version)

        logger.debug(
            f"Indexed {file_path}: {store_stats['sections']} sections, {chunk_count} chunks"
        )
        return "indexed"

    def _store_document(
        self,
        doc: ExtractedDocument,
        ingestor: MemgraphIngestor,
        resolved_code_references: list[str],
    ) -> tuple[dict, list[dict], str]:
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
        will_create_synthetic = (
            not doc.sections
            and doc.content
            and doc.content.strip()
            and not has_preamble
        )

        # Create Document node
        # Note: total_section_count includes synthetic sections for plain text files
        # and counts ALL sections (root + nested) for accurate total.
        # Also includes preamble section if present.
        # This differs from CONTAINS_SECTION relationships which only connect
        # top-level sections (nested sections use HAS_SUBSECTION relationships).
        preamble_count = 1 if has_preamble else 0
        ingestor.ensure_node_batch(
            cs.NodeLabel.DOCUMENT.value,
            {
                cs.UniqueKeyType.PATH.value: doc.path,
                "name": Path(doc.path).name,
                "workspace": self.workspace,
                "file_type": doc.file_type,
                "total_section_count": doc.total_section_count()
                + (1 if will_create_synthetic else 0)
                + preamble_count,
                "code_block_count": len(doc.code_blocks),
                "code_references": doc.code_references,
                "resolved_code_references": resolved_code_references,
                "resolved_code_reference_count": len(resolved_code_references),
                "word_count": doc.word_count,
                "modified_date": doc.modified_date,
                "indexed_at": indexed_at,
                "content_hash": doc.content_hash,
            },
        )

        # Create preamble section if there's content before the first section
        if has_preamble:
            preamble_qn = f"{doc.path}#synthetic:Preamble"
            logger.debug(
                f"Creating preamble section for {doc.path} ({preamble_line_count} lines)"
            )
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
                (
                    cs.NodeLabel.SECTION.value,
                    cs.UniqueKeyType.QUALIFIED_NAME.value,
                    preamble_qn,
                ),
            )
            all_section_info.append(
                {
                    "qualified_name": preamble_qn,
                    "title": "Preamble",
                    "start_line": 0,
                    "end_line": max(0, preamble_line_count - 1),
                    "level": 0,
                }
            )
            stats["sections"] += 1

        # Create Section nodes and relationships, collect section info
        for section in doc.sections:
            section_infos = self._store_section(
                doc.path, section, doc.path, None, ingestor, stats, indexed_at
            )
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
                (
                    cs.NodeLabel.SECTION.value,
                    cs.UniqueKeyType.QUALIFIED_NAME.value,
                    synthetic_qn,
                ),
            )
            all_section_info.append(
                {
                    "qualified_name": synthetic_qn,
                    "title": "Document Content",
                    "start_line": 0,
                    "end_line": doc.content.count("\n"),
                    "level": 1,
                }
            )
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
        # Create hierarchical qualified name with title hash for stability across edits
        # Format: {parent_path}#sec_{title_hash[:8]}:{section.title}
        title_hash = hashlib.sha256(section.title.encode()).hexdigest()[:8]
        section_qn = f"{parent_path}#sec_{title_hash}:{section.title}"
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
                (
                    cs.NodeLabel.SECTION.value,
                    cs.UniqueKeyType.QUALIFIED_NAME.value,
                    section_qn,
                ),
            )
        else:
            # Subsection: Section -> Section
            ingestor.ensure_relationship_batch(
                (
                    cs.NodeLabel.SECTION.value,
                    cs.UniqueKeyType.QUALIFIED_NAME.value,
                    parent_qn,
                ),
                cs.RelationshipType.HAS_SUBSECTION.value,
                (
                    cs.NodeLabel.SECTION.value,
                    cs.UniqueKeyType.QUALIFIED_NAME.value,
                    section_qn,
                ),
            )
        stats["sections"] += 1

        # Collect section info for chunk-to-section matching
        section_info: list[dict] = [
            {
                "qualified_name": section_qn,
                "title": section.title,
                "start_line": section.start_line,
                "end_line": section.end_line,
                "level": section.level,
            }
        ]

        # Recursively process subsections
        for subsection in section.subsections:
            subsection_infos = self._store_section(
                doc_path,
                subsection,
                section_qn,
                section_qn,
                ingestor,
                stats,
                indexed_at,
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
                logger.warning(ls.DOC_EMBEDDING_NO_CONTENT.format(path=doc.path))
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
            (i, c)
            for i, c in enumerate(chunks)
            if c.content.strip() and c.token_count >= MIN_CHUNK_TOKENS
        ]
        if not non_empty_chunks:
            logger.warning(
                ls.DOC_EMBEDDING_NO_VALID_CHUNKS.format(
                    path=doc.path,
                    min_tokens=MIN_CHUNK_TOKENS,
                )
            )
            return ([], [])

        chunk_contents = [c.content for i, c in non_empty_chunks]
        batch_size = max(1, settings.VECTOR_EMBEDDING_BATCH_SIZE)
        chunk_count = len(non_empty_chunks)
        total_tokens = sum(c.token_count for i, c in non_empty_chunks)
        total_batches = math.ceil(chunk_count / batch_size)

        if chunk_count >= 100:
            logger.info(
                ls.DOC_EMBEDDING_LARGE_DOCUMENT.format(
                    path=doc.path,
                    count=chunk_count,
                    tokens=total_tokens,
                )
            )
        if total_batches > 1:
            logger.info(
                ls.DOC_EMBEDDING_BATCH_START.format(
                    path=doc.path,
                    count=chunk_count,
                    batches=total_batches,
                )
            )

        embeddings: list[list[float]] = []
        log_interval = 1 if total_batches <= 20 else max(1, total_batches // 20)
        for batch_index, start in enumerate(range(0, chunk_count, batch_size), start=1):
            batch_contents = chunk_contents[start : start + batch_size]
            try:
                batch_embeddings = provider.embed_batch(
                    batch_contents, batch_size=len(batch_contents)
                )
            except Exception as e:
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=f"Embedding batch generation failed: {type(e).__name__}: {e}",
                ) from e

            if len(batch_embeddings) != len(batch_contents):
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=(
                        f"Embedding provider returned {len(batch_embeddings)} embeddings "
                        f"for batch {batch_index} with {len(batch_contents)} chunks"
                    ),
                )

            embeddings.extend(batch_embeddings)

            should_log_progress = (
                batch_index in {1, total_batches} or batch_index % log_interval == 0
            )
            if total_batches > 1 and (should_log_progress):
                processed_chunks = start + len(batch_contents)
                logger.info(
                    ls.DOC_EMBEDDING_BATCH_PROGRESS.format(
                        path=doc.path,
                        batch=batch_index,
                        total_batches=total_batches,
                        processed=processed_chunks,
                        count=chunk_count,
                    )
                )

        # Validate embedding quality (check for NaN, None, zero vectors, and dimension mismatch)
        expected_dimension = self._embedding_provider.dimension
        validated_embeddings = []
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=f"Embedding provider returned None for chunk {i}",
                )
            # Check for dimension mismatch
            if expected_dimension and len(embedding) != expected_dimension:
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=f"Embedding dimension mismatch for chunk {i}: "
                    f"expected {expected_dimension}, got {len(embedding)}. "
                    f"Check that EMBEDDING_MODEL matches the vector index configuration.",
                )
            # Check for NaN or infinity values (invalid for vector operations)
            if any(
                isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                for v in embedding
            ):
                raise ExtractionException(
                    path=doc.path,
                    error_type=ErrorType.EMBEDDING_ERROR,
                    message=f"Embedding for chunk {i} contains NaN or infinity values",
                )
            # Check for all-zero embedding (indicates failure)
            if all(v == 0.0 for v in embedding):
                logger.warning(ls.DOC_EMBEDDING_ZERO_VECTOR.format(index=i))
            validated_embeddings.append(embedding)

        # Return chunks and validated embeddings (without original indices)
        chunks_list = [c for i, c in non_empty_chunks]
        return (chunks_list, validated_embeddings)

    def _filter_valid_chunks(self, chunks: list) -> list:
        """Filter out chunks that are too small for meaningful indexing.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            Filtered list of chunks with meaningful content.
        """
        MIN_CHUNK_TOKENS = 10
        return [
            c for c in chunks
            if c.content.strip() and getattr(c, 'token_count', 0) >= MIN_CHUNK_TOKENS
        ]

    def _prepare_embeddings_with_fallback(
        self,
        doc: ExtractedDocument,
        chunks: list,
    ) -> tuple[list, list[list[float]] | None]:
        """Generate embeddings with graceful degradation.

        LLM-First: Fallback decision is deterministic (Python).
        User messaging is handled by logging and error templates.

        Args:
            doc: Extracted document.
            chunks: List of DocumentChunk objects.

        Returns:
            Tuple of (chunks, embeddings or None if unavailable).
        """
        if not self.embeddings_enabled:
            logger.info(f"Embeddings disabled for {doc.path}, storing structure only")
            return (self._filter_valid_chunks(chunks), None)

        try:
            return self._prepare_embeddings(doc, chunks)
        except Exception as e:
            if self.embeddings_required:
                # User explicitly requires embeddings - fail
                raise

            # Try fallback to local if configured and not already using local
            if settings.EMBEDDING_FALLBACK_TO_LOCAL:
                provider_name = self._embedding_provider.__class__.__name__
                if provider_name != "LocalEmbeddingProvider":
                    try:
                        logger.warning(
                            f"Primary embedding provider failed for {doc.path}, "
                            f"attempting local fallback: {type(e).__name__}: {e}"
                        )
                        from ..embeddings.local import get_local_embedding_provider
                        fallback_provider = get_local_embedding_provider(
                            model_id=settings.EMBEDDING_FALLBACK_MODEL
                        )
                        # Retry with fallback provider using same logic as _prepare_embeddings
                        return self._embed_with_fallback_provider(
                            doc, self._filter_valid_chunks(chunks), fallback_provider
                        )
                    except Exception as fallback_error:
                        logger.warning(
                            f"Local fallback also failed: {type(fallback_error).__name__}: {fallback_error}"
                        )

            # Graceful degradation: continue without embeddings
            logger.warning(
                f"Embedding generation failed for {doc.path}, "
                f"continuing with structural-only indexing: {type(e).__name__}: {e}"
            )
            return (self._filter_valid_chunks(chunks), None)

    def _embed_with_fallback_provider(
        self,
        doc: ExtractedDocument,
        chunks: list,
        provider,
    ) -> tuple[list, list[list[float]]]:
        """Embed chunks with a fallback provider.

        Args:
            doc: Extracted document.
            chunks: Pre-filtered list of valid chunks.
            provider: Fallback embedding provider.

        Returns:
            Tuple of (chunks, embeddings).
        """
        if not chunks:
            return ([], [])

        chunk_contents = [c.content for c in chunks]
        batch_size = max(1, settings.VECTOR_EMBEDDING_BATCH_SIZE)
        chunk_count = len(chunks)

        embeddings: list[list[float]] = []
        for start in range(0, chunk_count, batch_size):
            batch_contents = chunk_contents[start : start + batch_size]
            batch_embeddings = provider.embed_batch(
                batch_contents, batch_size=len(batch_contents)
            )
            embeddings.extend(batch_embeddings)

        logger.info(
            f"Successfully embedded {len(chunks)} chunks for {doc.path} using fallback provider"
        )
        return (chunks, embeddings)

    def _store_chunks_with_embeddings(
        self,
        doc: ExtractedDocument,
        embeddings_data: tuple[list, list[list[float]] | None],
        section_info: list[dict],
        ingestor: MemgraphIngestor,
        indexed_at: str,
    ) -> int:
        """Store chunks with pre-computed embeddings.

        Args:
            doc: Extracted document
            embeddings_data: Tuple of (non_empty_chunks, embeddings or None) from _prepare_embeddings_with_fallback
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

        # Handle no-embeddings mode (graceful degradation)
        if embeddings is None:
            logger.info(
                f"Storing {len(non_empty_chunks)} chunks without embeddings for {doc.path} "
                f"(structural-only indexing)"
            )
            for chunk in non_empty_chunks:
                chunk_reference_names = self._extract_chunk_reference_names(chunk.content)
                resolved_chunk_references = self._resolve_code_reference_names(
                    chunk_reference_names
                )
                # Store chunk without embedding property
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
                        "code_references": chunk_reference_names,
                        "resolved_code_references": resolved_chunk_references,
                        "resolved_code_reference_count": len(resolved_chunk_references),
                        # No embedding property - structural only
                        "indexed_at": indexed_at,
                    },
                )
                ingestor.ensure_relationship_batch(
                    (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                    cs.RelationshipType.CONTAINS_CHUNK.value,
                    (
                        cs.NodeLabel.CHUNK.value,
                        cs.UniqueKeyType.QUALIFIED_NAME.value,
                        chunk.qualified_name,
                    ),
                )

                # Find matching section for this chunk
                matching_section = self._find_section_for_chunk(chunk, section_info)
                target_section = matching_section or fallback_section
                ingestor.ensure_relationship_batch(
                    (
                        cs.NodeLabel.CHUNK.value,
                        cs.UniqueKeyType.QUALIFIED_NAME.value,
                        chunk.qualified_name,
                    ),
                    cs.RelationshipType.BELONGS_TO_SECTION.value,
                    (
                        cs.NodeLabel.SECTION.value,
                        cs.UniqueKeyType.QUALIFIED_NAME.value,
                        target_section["qualified_name"],
                    ),
                )

            return len(non_empty_chunks)

        # Standard path: store chunks with embeddings
        for chunk, embedding in zip(non_empty_chunks, embeddings):
            chunk_reference_names = self._extract_chunk_reference_names(chunk.content)
            resolved_chunk_references = self._resolve_code_reference_names(
                chunk_reference_names
            )
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
                    "code_references": chunk_reference_names,
                    "resolved_code_references": resolved_chunk_references,
                    "resolved_code_reference_count": len(resolved_chunk_references),
                    "embedding": embedding,
                    "indexed_at": indexed_at,
                },
            )
            ingestor.ensure_relationship_batch(
                (cs.NodeLabel.DOCUMENT.value, cs.UniqueKeyType.PATH.value, doc.path),
                cs.RelationshipType.CONTAINS_CHUNK.value,
                (
                    cs.NodeLabel.CHUNK.value,
                    cs.UniqueKeyType.QUALIFIED_NAME.value,
                    chunk.qualified_name,
                ),
            )

            # Find matching section for this chunk and create relationship
            # GUARANTEE: Every chunk gets BELONGS_TO_SECTION relationship
            matching_section = self._find_section_for_chunk(chunk, section_info)
            if matching_section:
                ingestor.ensure_relationship_batch(
                    (
                        cs.NodeLabel.CHUNK.value,
                        cs.UniqueKeyType.QUALIFIED_NAME.value,
                        chunk.qualified_name,
                    ),
                    cs.RelationshipType.BELONGS_TO_SECTION.value,
                    (
                        cs.NodeLabel.SECTION.value,
                        cs.UniqueKeyType.QUALIFIED_NAME.value,
                        matching_section["qualified_name"],
                    ),
                )
            else:
                # Fallback: Use first section (typically synthetic section for plain text)
                # This branch is reached when matching_section is None
                logger.warning(
                    f"Chunk {chunk.qualified_name} has no overlapping section, "
                    f"using fallback section: {fallback_section['qualified_name']}"
                )
                ingestor.ensure_relationship_batch(
                    (
                        cs.NodeLabel.CHUNK.value,
                        cs.UniqueKeyType.QUALIFIED_NAME.value,
                        chunk.qualified_name,
                    ),
                    cs.RelationshipType.BELONGS_TO_SECTION.value,
                    (
                        cs.NodeLabel.SECTION.value,
                        cs.UniqueKeyType.QUALIFIED_NAME.value,
                        fallback_section["qualified_name"],
                    ),
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
            overlap_lines = (
                overlap_end - overlap_start + 1 if overlap_start <= overlap_end else 0
            )

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
        if self._is_excluded_path(file_path):
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
            logger.error(
                f"Path traversal attempt: {file_path} is outside repo {self.base_path}"
            )
            self.dead_letter_queue.enqueue(
                ExtractionError(
                    path=str(file_path),
                    error_type=ErrorType.PATH_TRAVERSAL,
                    message="Path is outside repository boundaries",
                )
            )
            return "failed"

        concept_ingestor = None
        ingestor = None
        try:
            ingestor = MemgraphIngestor(
                host=self.host,
                port=self.port,
                batch_size=self.batch_size,
                connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
            ).__enter__()

            # Create concept ingestor if concept extraction is enabled
            if self.concept_extractor and settings.CONCEPT_MEMGRAPH_ENABLED:
                try:
                    concept_ingestor = MemgraphIngestor(
                        host=self.concept_host,
                        port=self.concept_port,
                        batch_size=settings.CONCEPT_MEMGRAPH_BATCH_SIZE,
                        connection_timeout=settings.CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT,
                        username=self.concept_username,
                        password=self.concept_password,
                    ).__enter__()
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.warning(
                        f"Concept extraction enabled but concept graph instance unavailable "
                        f"({self.concept_host}:{self.concept_port}): {e} — "
                        "concepts will not be stored"
                    )

            ingestor.ensure_constraints()
            result = self._process_document(file_path, ingestor, force=True, concept_ingestor=concept_ingestor)
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
        finally:
            if concept_ingestor is not None:
                try:
                    concept_ingestor.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing concept ingestor: {e}")
            if ingestor is not None:
                try:
                    ingestor.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing ingestor: {e}")

    def delete_file(self, file_path: Path) -> str:
        """
        Delete a document and all related nodes from the graph.

        Called by real-time updater when a document file is deleted.

        Args:
            file_path: Path to the deleted document file

        Returns:
            "deleted", "skipped", or "failed"
        """
        if self._is_excluded_path(file_path):
            return "skipped"

        resolved_path = file_path.resolve()
        if not self._is_path_within_boundary(resolved_path):
            logger.error(
                f"Path traversal attempt: {file_path} is outside repo {self.base_path}"
            )
            self.dead_letter_queue.enqueue(
                ExtractionError(
                    path=str(file_path),
                    error_type=ErrorType.PATH_TRAVERSAL,
                    message="Path is outside repository boundaries",
                )
            )
            return "failed"

        doc_path = str(file_path.relative_to(self.repo_path))

        concept_ingestor = None
        try:
            ingestor = MemgraphIngestor(
                host=self.host,
                port=self.port,
                batch_size=self.batch_size,
                connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
            ).__enter__()

            # Create concept ingestor for cross-instance cleanup
            if settings.CONCEPT_MEMGRAPH_ENABLED:
                try:
                    concept_ingestor = MemgraphIngestor(
                        host=self.concept_host,
                        port=self.concept_port,
                        batch_size=settings.CONCEPT_MEMGRAPH_BATCH_SIZE,
                        connection_timeout=settings.CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT,
                        username=self.concept_username,
                        password=self.concept_password,
                    ).__enter__()
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.warning(
                        f"Concept graph instance unavailable during file deletion "
                        f"({self.concept_host}:{self.concept_port}): {e} — "
                        "concept cleanup will be skipped"
                    )

            ingestor.ensure_constraints()
            self._delete_document_nodes(doc_path, ingestor, concept_ingestor=concept_ingestor)
            ingestor.execute_write(
                """
                MATCH (d:Document {path: $path, workspace: $workspace})
                DETACH DELETE d
                """,
                {"path": doc_path, "workspace": self.workspace},
            )
            ingestor.flush_all()
            logger.debug("Saving version cache to disk")
            self.version_cache.remove(doc_path)
            self.version_cache.save()
            return "deleted"
        except ExtractionException as e:
            logger.error(f"Failed to delete file {file_path}: {type(e).__name__}: {e}")
            self.dead_letter_queue.enqueue(e.to_extraction_error())
            return "failed"
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {type(e).__name__}: {e}")
            self.dead_letter_queue.enqueue(
                ExtractionError(
                    path=str(file_path),
                    error_type=self._map_error_type(e),
                    message=str(e),
                )
            )
            return "failed"
        finally:
            if concept_ingestor is not None:
                try:
                    concept_ingestor.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing concept ingestor: {e}")
            try:
                ingestor.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing ingestor: {e}")


    def _ensure_concept_indexes(self, ingestor: MemgraphIngestor) -> None:
        """Create indexes for Concept and Topic nodes if they do not exist.

        Uses Memgraph-compatible syntax: CREATE INDEX ON :Label(property)
        Handles existence check in Python since Memgraph doesn't support IF NOT EXISTS.
        """
        if self._concept_indexes_ensured:
            return

        # List of (label, property) tuples for indexes
        indexes_to_create = [
            ("Concept", "qualified_name"),
            ("Concept", "workspace"),
            ("Concept", "entity_category"),
            ("Concept", "entity_subtype"),
            ("ChunkRef", "qualified_name"),
            ("ChunkRef", "workspace"),
            ("Topic", "qualified_name"),
            ("Topic", "workspace"),
        ]

        # Edge label indexes for concept relationship categories
        edge_indexes_to_create = [
            ("HIERARCHICAL", "verb"),
            ("COMPOSITIONAL", "verb"),
            ("CONTEXTUAL", "verb"),
            ("ATTRIBUTIVE", "verb"),
            ("COMPARATIVE", "verb"),
            ("SEQUENTIAL", "verb"),
            ("CAUSAL", "verb"),
            ("ANALOGICAL", "verb"),
            ("RELATED_TO", "verb"),
        ]

        for label, prop in indexes_to_create:
            cypher = f"CREATE INDEX ON :{label}({prop});"
            try:
                ingestor.fetch_all(cypher)
                logger.debug(f"Created index on :{label}({prop})")
            except Exception as e:
                msg = str(e).lower()
                # Memgraph error messages for existing indexes
                if any(x in msg for x in ["already exists", "duplicate", "existing", "already created"]):
                    logger.debug(f"Index on :{label}({prop}) already exists")
                else:
                    logger.warning(f"Failed to create index on :{label}({prop}): {e}")

        for label, prop in edge_indexes_to_create:
            cypher = f"CREATE INDEX ON :{label}({prop});"
            try:
                ingestor.fetch_all(cypher)
                logger.debug(f"Created index on :{label}({prop})")
            except Exception as e:
                msg = str(e).lower()
                # Memgraph error messages for existing indexes
                if any(x in msg for x in ["already exists", "duplicate", "existing", "already created"]):
                    logger.debug(f"Index on :{label}({prop}) already exists")
                else:
                    logger.warning(f"Failed to create index on :{label}({prop}): {e}")

        self._concept_indexes_ensured = True

    async def _extract_and_store_concepts(
        self,
        chunks: list[DocumentChunk],
        ingestor: MemgraphIngestor,
        workspace: str,
        stats: dict[str, object] | None = None,
        concept_ingestor: MemgraphIngestor | None = None,
    ) -> None:
        """Extract concepts from chunks and store in graph via batch MERGE.

        Uses LLM extraction (not regex) for semantic concept identification.
        Extraction runs concurrently with a configurable semaphore to limit
        parallel LLM calls. Partial failures are tolerated: successful chunks
        are stored and failed chunks are logged at debug level.
        """
        if not self.concept_extractor:
            return

        # Graceful degradation: skip if concept instance unavailable
        if concept_ingestor is None:
            logger.warning("Skipping concept storage: concept graph instance unavailable")
            return

        extractor = cast(LLMConceptExtractor, self.concept_extractor)
        if extractor._circuit_breaker is not None and not extractor._circuit_breaker.can_execute():
            remaining = 0.0
            if extractor._circuit_breaker.last_failure_time is not None:
                remaining = max(
                    0.0,
                    extractor._circuit_breaker.config.timeout_seconds
                    - (datetime.now(UTC) - extractor._circuit_breaker.last_failure_time).total_seconds(),
                )
            logger.warning(
                doc_ls.DOC_CONCEPT_BREAKER_SKIP_DOC.format(
                    doc=workspace,
                    remaining=remaining,
                )
            )
            if stats is not None:
                stats["concepts_skipped_circuit_breaker"] = stats.get("concepts_skipped_circuit_breaker", 0) + len(chunks)
            return

        self._ensure_concept_indexes(concept_ingestor)
        logger.info(doc_ls.DOC_CONCEPT_EXTRACT_START.format(chunk_count=len(chunks)))

        semaphore = asyncio.Semaphore(
            getattr(settings, "DOC_CONCEPT_EXTRACTION_CONCURRENCY", 10)
        )

        async def _extract_one(chunk: DocumentChunk) -> ExtractionResult:
            async with semaphore:
                extractor = cast(LLMConceptExtractor, self.concept_extractor)
                return await extractor.extract_with_retry(
                    chunk.content,
                    chunk.qualified_name,
                    dead_letter_queue=self.dead_letter_queue,
                )

        extraction_results = await asyncio.gather(
            *[_extract_one(c) for c in chunks],
            return_exceptions=True,
        )

        concept_nodes: list[dict[str, object]] = []
        mention_rels: list[dict[str, object]] = []
        concept_relationships: list[tuple[str, str, str, str, str, float]] = []
        failed_count = 0

        for idx, result in enumerate(extraction_results):
            chunk = chunks[idx]
            if isinstance(result, Exception):
                failed_count += 1
                logger.debug(
                    f"Concept extraction failed for {chunk.qualified_name}: {result}"
                )
                continue

            for concept in result.concepts:
                concept_qn = f"{workspace}:{concept.name}"
                concept_nodes.append({
                    "qualified_name": concept_qn,
                    "workspace": workspace,
                    "name": concept.name,
                    "aliases": concept.aliases,
                    "type": concept.type,
                    "definition": concept.definition,
                    "confidence": concept.confidence,
                    "source_chunk_qn": concept.source_chunk_qn,
                    "entity_category": concept.entity_category,
                    "entity_subtype": concept.entity_subtype,
                    "entity_emoji": concept.entity_emoji,
                })
                frequency = chunk.content.lower().count(concept.name.lower())
                if not frequency:
                    frequency = 1
                mention_rels.append({
                    "chunk_qn": chunk.qualified_name,
                    "concept_qn": concept_qn,
                    "frequency": frequency,
                    "context": concept.context,
                })

            for rel in result.relationships:
                concept_relationships.append(
                    (
                        f"{workspace}:{rel.from_concept}",
                        f"{workspace}:{rel.to_concept}",
                        rel.verb,
                        rel.emoji,
                        rel.category,
                        rel.strength,
                    )
                )

        if failed_count:
            logger.warning(
                f"Concept extraction partial success: "
                f"{len(chunks) - failed_count}/{len(chunks)} chunks succeeded"
            )

        logger.info(
            doc_ls.DOC_CONCEPT_EXTRACT_DONE.format(concept_count=len(concept_nodes))
        )

        if concept_nodes:
            self._merge_concept_nodes_batch(concept_ingestor, concept_nodes)
        if mention_rels:
            self._create_mentions_batch(concept_ingestor, mention_rels, workspace)
        if concept_relationships:
            self._store_concept_relationships_batch(concept_ingestor, concept_relationships, workspace)

    def _merge_concept_nodes_batch(
        self,
        ingestor: MemgraphIngestor,
        concept_nodes: list[dict[str, object]],
    ) -> None:
        """Batch merge Concept nodes using UNWIND for efficiency."""
        batch_size = settings.CONCEPT_MEMGRAPH_BATCH_SIZE
        for batch in batched(concept_nodes, batch_size):
            cypher = """
            UNWIND $nodes as node
            MERGE (c:Concept {qualified_name: node.qualified_name})
            SET c.workspace = node.workspace,
                c.name = node.name,
                c.aliases = node.aliases,
                c.type = node.type,
                c.definition = node.definition,
                c.confidence = node.confidence,
                c.source_chunk_qn = node.source_chunk_qn,
                c.entity_category = node.entity_category,
                c.entity_subtype = node.entity_subtype,
                c.entity_emoji = node.entity_emoji
            """
            ingestor.fetch_all(cypher, {"nodes": list(batch)})

    def _create_mentions_batch(
        self,
        ingestor: MemgraphIngestor,
        mention_rels: list[dict[str, object]],
        workspace: str,
    ) -> None:
        """Batch create MENTIONS relationships from ChunkRef proxies to concepts."""
        batch_size = settings.CONCEPT_MEMGRAPH_BATCH_SIZE
        for batch in batched(mention_rels, batch_size):
            cypher = """
            UNWIND $rels as rel
            MERGE (c:ChunkRef {qualified_name: rel.chunk_qn, workspace: $workspace})
            MERGE (concept:Concept {qualified_name: rel.concept_qn, workspace: $workspace})
            MERGE (c)-[m:MENTIONS]->(concept)
            SET m.frequency = rel.frequency, m.context = rel.context
            """
            ingestor.fetch_all(cypher, {"rels": list(batch), "workspace": workspace})

    def _store_concept_relationships_batch(
        self,
        ingestor: MemgraphIngestor,
        relationships: list[tuple[str, str, str, str, str, float]],
        workspace: str,
    ) -> None:
        """Batch create relationships between concepts.

        Memgraph does not support parameterized relationship types, so we use
        explicit FOREACH branches per category (8 canonical + RELATED_TO fallback).
        """
        rel_maps: list[dict[str, object]] = [
            {
                "from_qn": from_qn,
                "to_qn": to_qn,
                "verb": verb,
                "emoji": emoji,
                "category": category,
                "strength": strength,
            }
            for from_qn, to_qn, verb, emoji, category, strength in relationships
        ]
        batch_size = settings.CONCEPT_MEMGRAPH_BATCH_SIZE
        for batch in batched(rel_maps, batch_size):
            cypher = """
            UNWIND $rels as rel
            MATCH (a:Concept {qualified_name: rel.from_qn, workspace: $workspace})
            MATCH (b:Concept {qualified_name: rel.to_qn, workspace: $workspace})
            FOREACH (_ IN CASE WHEN rel.category = 'HIERARCHICAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:HIERARCHICAL]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'COMPOSITIONAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:COMPOSITIONAL]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'CONTEXTUAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:CONTEXTUAL]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'ATTRIBUTIVE' THEN [1] ELSE [] END |
                MERGE (a)-[r:ATTRIBUTIVE]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'COMPARATIVE' THEN [1] ELSE [] END |
                MERGE (a)-[r:COMPARATIVE]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'SEQUENTIAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:SEQUENTIAL]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'CAUSAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:CAUSAL]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'ANALOGICAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:ANALOGICAL]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'RELATED_TO' THEN [1] ELSE [] END |
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
            )
            """
            ingestor.fetch_all(cypher, {"rels": list(batch), "workspace": workspace})

    def _get_chunk_qns_for_document(
        self,
        document_path: str,
        doc_ingestor: MemgraphIngestor,
    ) -> list[str]:
        """Get chunk qualified names for a document from the doc instance."""
        cypher = """
        MATCH (d:Document {path: $doc_path, workspace: $workspace})
              -[:CONTAINS_CHUNK]->(c:Chunk)
        RETURN c.qualified_name AS chunk_qn
        """
        result = doc_ingestor.fetch_all(cypher, {
            "doc_path": document_path,
            "workspace": self.workspace,
        })
        return [r["chunk_qn"] for r in result]

    def _cleanup_concepts_for_document(
        self,
        document_path: str,
        doc_ingestor: MemgraphIngestor,
        concept_ingestor: MemgraphIngestor | None = None,
    ) -> None:
        """Remove orphaned concepts after document chunks are deleted.

        Step 1: Queries doc instance for chunk QNs belonging to the document.
        Step 2: Deletes ChunkRefs and orphaned Concepts from concept instance.
        """
        if concept_ingestor is None or not settings.CONCEPT_MEMGRAPH_ENABLED:
            return

        logger.info(doc_ls.DOC_CONCEPT_CLEANUP_START.format(doc_path=document_path))

        # Step 1: Get chunk QNs from doc instance
        chunk_qns = self._get_chunk_qns_for_document(document_path, doc_ingestor)
        if not chunk_qns:
            logger.debug(f"No chunks found for document: {document_path}")
            return

        # Step 2: Clean up concept instance
        cypher = """
        UNWIND $chunk_qns as chunk_qn
        MATCH (cr:ChunkRef {qualified_name: chunk_qn, workspace: $workspace})
        OPTIONAL MATCH (cr)-[m:MENTIONS]->(concept:Concept {workspace: $workspace})
        DELETE m, cr
        WITH DISTINCT concept
        WHERE concept IS NOT NULL
        WITH collect(DISTINCT concept) as concepts
        UNWIND concepts as concept
        OPTIONAL MATCH (:ChunkRef)-[remaining:MENTIONS]->(concept)
        WITH concept, remaining
        WHERE remaining IS NULL
        DETACH DELETE concept
        RETURN count(concept) as removed_count
        """
        result = concept_ingestor.fetch_all(
            cypher,
            {"chunk_qns": chunk_qns, "workspace": self.workspace},
        )

        removed_count = result[0].get("removed_count", 0) if result else 0
        logger.info(doc_ls.DOC_CONCEPT_CLEANUP_DONE.format(count=removed_count))


def migrate_section_count_property(
    host: str = "localhost",
    port: int = 7688,
    workspace: str = "default",
    connection_timeout: int | None = None,
) -> dict:
    """
    Migrate old 'section_count' property to 'total_section_count' on Document nodes.

    This migration handles the property rename from section_count to total_section_count.
    It renames the property on existing Document nodes to maintain consistency.

    Handles two cases:
    1. Documents with only old property -> rename to new property
    2. Documents with both properties -> remove old property

    Args:
        host: Memgraph host
        port: Memgraph port
        workspace: Workspace to migrate (default: all workspaces if None)
        connection_timeout: Connection timeout in seconds (uses default if None)

    Returns:
        Dict with migration statistics
    """
    stats = {"migrated": 0, "cleaned": 0, "errors": 0}

    # Use configured timeout if not provided
    if connection_timeout is None:
        connection_timeout = settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT

    with MemgraphIngestor(
        host=host, port=port, connection_timeout=connection_timeout
    ) as ingestor:
        # Case 1: Documents with only old property -> rename to new
        if workspace:
            query1 = """
            MATCH (d:Document)
            WHERE d.section_count IS NOT NULL AND d.total_section_count IS NULL
            AND d.workspace = $workspace
            SET d.total_section_count = d.section_count
            REMOVE d.section_count
            RETURN count(d) as migrated
            """
            params1 = {"workspace": workspace}
        else:
            query1 = """
            MATCH (d:Document)
            WHERE d.section_count IS NOT NULL AND d.total_section_count IS NULL
            SET d.total_section_count = d.section_count
            REMOVE d.section_count
            RETURN count(d) as migrated
            """
            params1 = {}

        try:
            result = ingestor.fetch_all(query1, params1)
            if result:
                stats["migrated"] = result[0].get("migrated", 0)
            logger.info(
                f"Migrated {stats['migrated']} Document nodes from section_count to total_section_count"
            )
        except Exception as e:
            logger.error(f"Migration (case 1) failed: {e}")
            stats["errors"] += 1

        # Case 2: Documents with both properties -> remove old
        if workspace:
            query2 = """
            MATCH (d:Document)
            WHERE d.section_count IS NOT NULL AND d.total_section_count IS NOT NULL
            AND d.workspace = $workspace
            REMOVE d.section_count
            RETURN count(d) as cleaned
            """
            params2 = {"workspace": workspace}
        else:
            query2 = """
            MATCH (d:Document)
            WHERE d.section_count IS NOT NULL AND d.total_section_count IS NOT NULL
            REMOVE d.section_count
            RETURN count(d) as cleaned
            """
            params2 = {}

        try:
            result = ingestor.fetch_all(query2, params2)
            if result:
                stats["cleaned"] = result[0].get("cleaned", 0)
            if stats["cleaned"] > 0:
                logger.info(
                    f"Cleaned {stats['cleaned']} Document nodes with duplicate property"
                )
        except Exception as e:
            logger.error(f"Migration (case 2) failed: {e}")
            stats["errors"] += 1

    return stats


__all__ = [
    "DocumentGraphUpdater",
    "DocumentGraphUnavailableError",
    "_check_graph_availability",
    "_check_graph_availability_async",
    "migrate_section_count_property",
]
