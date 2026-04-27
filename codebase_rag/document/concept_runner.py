"""Standalone concept extraction runner for existing document chunks.

Architecture:
    - Document Graph (port 7688): Source of Chunk content
    - Concept Graph (port 7690): Target for Concept/ChunkRef storage

Reuses existing infrastructure:
    - LLMConceptExtractor from concept_extraction.py
    - Batch storage methods from document_updater.py
    - DeadLetterQueue from error_handling.py
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ..config import settings
from ..main import connect_concept_memgraph, connect_doc_memgraph
from ..services.graph_service import MemgraphIngestor
from . import logs as doc_ls
from .chunk_splitting import _split_chunk
from .concept_consolidator import ConceptConsolidator
from .concept_extraction import (
    ExtractedConcept,
    ExtractionResult,
    LLMConceptExtractor,
    calculate_adaptive_max_tokens,
)
from .error_handling import DeadLetterQueue, ErrorType, ExtractionError

if TYPE_CHECKING:
    pass

_WORKSPACE_PATTERN = re.compile(r"^[\w\-]+$")


@dataclass
class StandaloneExtractionStats:
    """Statistics for standalone concept extraction session.

    Extends the base ConceptExtractionStats with additional fields for
    tracking incremental processing and skipped chunks.
    """
    total_chunks: int = 0
    processed_chunks: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    concepts_created: int = 0
    relationships_created: int = 0
    skipped_existing: int = 0
    skewed_chunks_count: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class AdaptiveRetryResult:
    """Result of an adaptive-token DLQ retry attempt."""

    success: bool
    concepts: int = 0
    relationships: int = 0
    reason: str | None = None
    tokens_used: int | None = None
    result: ExtractionResult | None = None


class ConceptExtractionRunner:
    """Run concept extraction on existing document chunks.

    Handles dual-graph architecture:
    - Reads Chunk content from document graph (port 7688)
    - Writes Concept/ChunkRef to concept graph (port 7690)

    Reuses batch storage methods from DocumentGraphUpdater for consistency.
    """

    def __init__(
        self,
        repo_path: Path,
        workspace: str = "default",
        batch_size: int | None = None,
        concurrency: int | None = None,
    ):
        self.repo_path = repo_path

        # Validate workspace identifier (same pattern as DocumentGraphUpdater)
        if not _WORKSPACE_PATTERN.match(workspace):
            raise ValueError(
                f"Invalid workspace identifier: '{workspace}'. "
                "Must contain only alphanumeric characters, underscores, and hyphens."
            )
        self.workspace = workspace
        self.batch_size = batch_size or settings.CONCEPT_MEMGRAPH_BATCH_SIZE
        # Default concurrency from config, capped at 50
        self.concurrency = min(
            concurrency if concurrency is not None
            else settings.DOC_CONCEPT_EXTRACTION_CONCURRENCY,
            50
        )

        # Dual-graph connections (managed via context manager)
        self._doc_ingestor: MemgraphIngestor | None = None
        self._concept_ingestor: MemgraphIngestor | None = None

        # Reusable components
        self.extractor: LLMConceptExtractor | None = None
        self.dead_letter_queue: DeadLetterQueue | None = None

    @contextmanager
    def _connect_graphs(self) -> Generator[tuple[MemgraphIngestor, MemgraphIngestor] | None, None, None]:
        """Context manager for dual-graph connections.

        Yields:
            Tuple of (doc_ingestor, concept_ingestor) if both connected, None otherwise.
        """
        doc_ingestor = None
        concept_ingestor = None
        try:
            try:
                doc_ingestor = connect_doc_memgraph(batch_size=self.batch_size).__enter__()
            except Exception as e:
                logger.error(doc_ls.DOC_GRAPH_CONNECT_FAILED.format(error=e))
                yield None
                return

            if not settings.CONCEPT_MEMGRAPH_ENABLED:
                logger.warning(doc_ls.CONCEPT_GRAPH_DISABLED)
                yield None
                return

            try:
                concept_ingestor = connect_concept_memgraph(batch_size=self.batch_size).__enter__()
            except Exception as e:
                logger.error(doc_ls.CONCEPT_GRAPH_CONNECT_FAILED.format(error=e))
                yield None
                return

            self._doc_ingestor = doc_ingestor
            self._concept_ingestor = concept_ingestor
            yield (doc_ingestor, concept_ingestor)
        finally:
            if concept_ingestor is not None:
                try:
                    concept_ingestor.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(doc_ls.CONCEPT_GRAPH_CLOSE_FAILED.format(error=e))
            if doc_ingestor is not None:
                try:
                    doc_ingestor.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(doc_ls.DOC_GRAPH_CLOSE_FAILED.format(error=e))
            self._doc_ingestor = None
            self._concept_ingestor = None

    def _initialize_extractor(self) -> bool:
        """Initialize LLM concept extractor.

        Returns:
            True if initialized successfully, False otherwise.
        """
        if not settings.DOC_CONCEPT_EXTRACTION_ENABLED:
            logger.warning(doc_ls.DOC_CONCEPT_EXTRACTION_DISABLED)
            return False

        try:
            self.extractor = LLMConceptExtractor()
            cgr_dir = self.repo_path / ".cgr"
            cgr_dir.mkdir(parents=True, exist_ok=True)
            self.dead_letter_queue = DeadLetterQueue(cgr_dir / "doc_errors")
            return True
        except Exception as e:
            logger.error(doc_ls.DOC_CONCEPT_INIT_FAILED.format(error=e))
            return False

    def _get_all_chunks(self, doc_ingestor: MemgraphIngestor) -> list[dict]:
        """Query document graph for all chunks in workspace.

        Args:
            doc_ingestor: Document graph connection.

        Returns:
            List of dicts with keys: qualified_name, content
        """
        cypher = """
        MATCH (c:Chunk {workspace: $workspace})
        RETURN c.qualified_name AS qualified_name,
               c.content AS content
        ORDER BY c.qualified_name
        """
        results = doc_ingestor.fetch_all(cypher, {"workspace": self.workspace})
        return [dict(r) for r in results]

    def _get_extracted_chunk_qns(self, concept_ingestor: MemgraphIngestor) -> set[str]:
        """Query concept graph for chunks that already have concepts.

        Args:
            concept_ingestor: Concept graph connection.

        Returns:
            Set of qualified names for chunks with existing MENTIONS relationships.
        """
        cypher = """
        MATCH (cr:ChunkRef {workspace: $workspace})-[:MENTIONS]->(:Concept)
        RETURN DISTINCT cr.qualified_name AS qn
        """
        results = concept_ingestor.fetch_all(cypher, {"workspace": self.workspace})
        return {qn for r in results if (qn := r.get("qn")) is not None}

    def _get_chunks_needing_extraction(
        self,
        doc_ingestor: MemgraphIngestor,
        concept_ingestor: MemgraphIngestor,
        force: bool,
    ) -> tuple[list[dict], int]:
        """Get chunks that need concept extraction.

        Args:
            doc_ingestor: Document graph connection.
            concept_ingestor: Concept graph connection.
            force: If True, return all chunks. If False, exclude already-extracted.

        Returns:
            Tuple of (list of chunk dicts needing extraction, count of skipped existing)
        """
        all_chunks = self._get_all_chunks(doc_ingestor)

        if force:
            return (all_chunks, 0)

        extracted_qns = self._get_extracted_chunk_qns(concept_ingestor)
        needing_extraction = [c for c in all_chunks if c["qualified_name"] not in extracted_qns]
        skipped = len(all_chunks) - len(needing_extraction)
        return (needing_extraction, skipped)

    def _cleanup_existing_concepts(
        self,
        chunk_qns: list[str],
        concept_ingestor: MemgraphIngestor,
    ) -> None:
        """Remove existing ChunkRefs, MENTIONS, and concept relationships for chunks.

        Args:
            chunk_qns: Qualified names of chunks to clean up.
            concept_ingestor: Concept graph connection.
        """
        if not chunk_qns:
            return
        cypher = """
        UNWIND $chunk_qns AS qn
        MATCH (cr:ChunkRef {qualified_name: qn, workspace: $workspace})
        OPTIONAL MATCH (cr)-[m:MENTIONS]->(concept:Concept)
        OPTIONAL MATCH (concept)-[r]->(other:Concept)
        WHERE other.workspace = $workspace
        DELETE m, r, cr
        """
        concept_ingestor.fetch_all(cypher, {
            "chunk_qns": chunk_qns,
            "workspace": self.workspace,
        })

    def run(
        self,
        force: bool = False,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> StandaloneExtractionStats:
        """Extract concepts from existing chunks.

        Args:
            force: Re-extract even if chunk already has concepts
            dry_run: Preview mode, no LLM calls
            limit: Max chunks to process

        Returns:
            Stats object with extraction results
        """
        stats = StandaloneExtractionStats()

        with self._connect_graphs() as connections:
            if connections is None:
                logger.error(doc_ls.DUAL_GRAPH_CONNECT_FAILED)
                return stats

            doc_ingestor, concept_ingestor = connections

            if not dry_run and not self._initialize_extractor():
                logger.error(doc_ls.DOC_CONCEPT_INIT_FAILED.format(error="extractor init returned False"))
                return stats

            chunks, skipped = self._get_chunks_needing_extraction(
                doc_ingestor, concept_ingestor, force
            )

            if not chunks:
                stats.skipped_existing = skipped
                logger.info(doc_ls.DOC_CONCEPT_NO_CHUNKS)
                if skipped > 0:
                    logger.info(doc_ls.DOC_CONCEPT_ALL_EXISTING.format(count=skipped))
                return stats

            stats.total_chunks = len(chunks)
            stats.skipped_existing = skipped

            if limit and limit > 0:
                chunks = chunks[:limit]
                logger.info(doc_ls.DOC_CONCEPT_LIMIT_APPLIED.format(limit=limit))

            if dry_run:
                logger.info(doc_ls.DOC_CONCEPT_DRY_RUN.format(count=len(chunks)))
                stats.processed_chunks = len(chunks)
                index_status = self._check_indexes(concept_ingestor)
                logger.info(doc_ls.DOC_CONCEPT_INDEX_STATUS.format(status=index_status))
                return stats

            if force:
                chunk_qns = [c["qualified_name"] for c in chunks]
                logger.info(doc_ls.DOC_CONCEPT_CLEANUP_START.format(count=len(chunk_qns)))
                self._cleanup_existing_concepts(chunk_qns, concept_ingestor)

            self._ensure_indexes(concept_ingestor)

            logger.info(doc_ls.DOC_CONCEPT_EXTRACT_START.format(chunk_count=len(chunks)))

            stats = asyncio.run(self._process_chunks_async(
                chunks, stats, concept_ingestor
            ))

            return stats

    async def _process_chunks_async(
        self,
        chunks: list[dict],
        stats: StandaloneExtractionStats,
        concept_ingestor: MemgraphIngestor,
    ) -> StandaloneExtractionStats:
        """Process chunks with concurrent LLM extraction.

        Uses ConceptConsolidator with periodic intermediate flushes to
        bound memory usage and preserve progress on crash.

        Args:
            chunks: List of chunk dicts with content
            stats: Stats object to update
            concept_ingestor: Concept graph connection for storage

        Returns:
            Updated stats
        """
        if not self.extractor:
            return stats

        semaphore = asyncio.Semaphore(self.concurrency)

        async def _extract_one(chunk: dict) -> ExtractionResult:
            async with semaphore:
                return await self.extractor.extract_with_retry(
                    chunk["content"],
                    chunk["qualified_name"],
                    dead_letter_queue=self.dead_letter_queue,
                )

        results = await asyncio.gather(
            *[_extract_one(c) for c in chunks],
            return_exceptions=True,
        )

        consolidator = ConceptConsolidator()
        all_mention_rels: list[dict] = []
        all_relationships: list[tuple] = []
        flush_interval = getattr(settings, "DOC_CONCEPT_FLUSH_INTERVAL", 100)

        def _flush() -> None:
            deduped = consolidator.consolidate()
            if deduped:
                self._merge_concept_nodes_batch(concept_ingestor, deduped)
            if all_mention_rels:
                self._create_mentions_batch(concept_ingestor, all_mention_rels, self.workspace)
            if all_relationships:
                self._store_concept_relationships_batch(
                    concept_ingestor, all_relationships, self.workspace
                )

        for idx, result in enumerate(results):
            chunk = chunks[idx]
            stats.processed_chunks += 1

            if isinstance(result, Exception):
                stats.failed_extractions += 1
                error_type = type(result).__name__
                stats.errors_by_type[error_type] = stats.errors_by_type.get(error_type, 0) + 1
                logger.debug(doc_ls.DOC_CONCEPT_EXTRACT_FAILED.format(
                    chunk_qn=chunk["qualified_name"], error=result
                ))
                continue

            stats.successful_extractions += 1
            stats.concepts_created += len(result.concepts)
            stats.relationships_created += len(result.relationships)
            if getattr(result, "was_rebalanced", False):
                stats.skewed_chunks_count += 1

            chunk_qn = chunk["qualified_name"]
            for concept in result.concepts:
                concept_qn = f"{self.workspace}:{concept.name}"
                node = {
                    "qualified_name": concept_qn,
                    "workspace": self.workspace,
                    "name": concept.name,
                    "aliases": concept.aliases,
                    "type": concept.type,
                    "definition": concept.definition,
                    "confidence": concept.confidence,
                    "entity_category": concept.entity_category,
                    "entity_subtype": concept.entity_subtype,
                    "entity_emoji": concept.entity_emoji,
                    "context": concept.context,
                }
                consolidator.add(node, chunk_qn)
                frequency = chunk["content"].lower().count(concept.name.lower()) or 1
                all_mention_rels.append({
                    "chunk_qn": chunk_qn,
                    "concept_qn": concept_qn,
                    "frequency": frequency,
                    "context": concept.context,
                })

            for rel in result.relationships:
                all_relationships.append(
                    (
                        f"{self.workspace}:{rel.from_concept}",
                        f"{self.workspace}:{rel.to_concept}",
                        rel.verb,
                        rel.emoji,
                        rel.category,
                        rel.strength,
                    )
                )

            if (idx + 1) % flush_interval == 0:
                _flush()
                consolidator = ConceptConsolidator()
                all_mention_rels = []
                all_relationships = []

        _flush()
        return stats

    def _ensure_indexes(self, concept_ingestor: MemgraphIngestor) -> None:
        """Ensure required indexes exist in concept graph.

        Args:
            concept_ingestor: Concept graph connection.
        """
        node_indexes = [
            ("Concept", "qualified_name"),
            ("Concept", "workspace"),
            ("Concept", "entity_category"),
            ("Concept", "entity_subtype"),
            ("ChunkRef", "qualified_name"),
            ("ChunkRef", "workspace"),
        ]
        edge_indexes = [
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

        for label, prop in node_indexes + edge_indexes:
            try:
                concept_ingestor.fetch_all(f"CREATE INDEX ON :{label}({prop});")
            except Exception as e:
                msg = str(e).lower()
                if any(x in msg for x in ["already exists", "duplicate", "existing", "already created"]):
                    logger.debug(doc_ls.DOC_CONCEPT_INDEX_EXISTS.format(label=label, prop=prop))
                else:
                    logger.warning(doc_ls.DOC_CONCEPT_INDEX_FAILED.format(label=label, prop=prop, error=e))

    def _check_indexes(self, concept_ingestor: MemgraphIngestor) -> dict[str, bool]:
        """Check if required indexes exist in concept graph.

        Args:
            concept_ingestor: Concept graph connection.

        Returns:
            Dict mapping index name to existence status.
        """
        indexes_to_check = [
            ("Concept", "qualified_name"),
            ("Concept", "workspace"),
            ("ChunkRef", "qualified_name"),
            ("ChunkRef", "workspace"),
        ]

        status: dict[str, bool] = {}
        try:
            result = concept_ingestor.fetch_all("SHOW INDEX INFO;")
        except Exception:
            return {f"{label}_{prop}": False for label, prop in indexes_to_check}

        for label, prop in indexes_to_check:
            index_name = f"{label}_{prop}"
            exists = any(
                r.get("index label") == label and r.get("index property") == prop
                for r in result
            )
            status[index_name] = exists

        return status

    def _store_extraction_results(
        self,
        chunk: dict,
        result: ExtractionResult,
        concept_ingestor: MemgraphIngestor,
    ) -> None:
        """Store extracted concepts and relationships in concept graph.

        Reuses batch storage logic from DocumentGraphUpdater for consistency.

        Args:
            chunk: Source chunk dict
            result: Extraction result with concepts and relationships
            concept_ingestor: Concept graph connection
        """
        # 1. Create ChunkRef proxy
        chunk_qn = chunk["qualified_name"]
        concept_ingestor.fetch_all(
            """
            MERGE (cr:ChunkRef {qualified_name: $qn, workspace: $ws})
            """,
            {"qn": chunk_qn, "ws": self.workspace},
        )

        # 2. Create Concept nodes
        concept_nodes = []
        for concept in result.concepts:
            concept_qn = f"{self.workspace}:{concept.name}"
            concept_nodes.append({
                "qualified_name": concept_qn,
                "workspace": self.workspace,
                "name": concept.name,
                "aliases": concept.aliases,
                "type": concept.type,
                "definition": concept.definition,
                "confidence": concept.confidence,
                "entity_category": concept.entity_category,
                "entity_subtype": concept.entity_subtype,
                "entity_emoji": concept.entity_emoji,
            })

        if concept_nodes:
            self._merge_concept_nodes_batch(concept_ingestor, concept_nodes)

        # 3. Create MENTIONS relationships
        mention_rels = []
        for concept in result.concepts:
            concept_qn = f"{self.workspace}:{concept.name}"
            frequency = chunk["content"].lower().count(concept.name.lower()) or 1
            mention_rels.append({
                "chunk_qn": chunk_qn,
                "concept_qn": concept_qn,
                "frequency": frequency,
                "context": concept.context,
            })

        if mention_rels:
            self._create_mentions_batch(concept_ingestor, mention_rels, self.workspace)

        # 4. Create concept-to-concept relationships
        if result.relationships:
            concept_relationships = [
                (
                    f"{self.workspace}:{r.from_concept}",
                    f"{self.workspace}:{r.to_concept}",
                    r.verb,
                    r.emoji,
                    r.category,
                    r.strength,
                )
                for r in result.relationships
            ]
            self._store_concept_relationships_batch(
                concept_ingestor, concept_relationships, self.workspace
            )

    # ------------------------------------------------------------------
    # Batch storage methods - mirrors DocumentGraphUpdater implementations
    # These could be refactored into a shared module for code reuse.
    # ------------------------------------------------------------------

    def _merge_concept_nodes_batch(
        self,
        ingestor: MemgraphIngestor,
        concept_nodes: list[dict[str, object]],
    ) -> None:
        """Batch merge Concept nodes using UNWIND for efficiency.

        Mirrors DocumentGraphUpdater._merge_concept_nodes_batch() for consistency.
        """
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
        """Batch create MENTIONS relationships from ChunkRef proxies to concepts.

        Mirrors DocumentGraphUpdater._create_mentions_batch() for consistency.
        """
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

    async def _retry_with_adaptive_tokens(
        self,
        error: ExtractionError,
    ) -> AdaptiveRetryResult:
        """Retry a context-overflow chunk with adaptive max_tokens."""
        if not error.chunk_content:
            return AdaptiveRetryResult(
                success=False, reason=doc_ls.DOC_DLQ_ADAPTIVE_MISSING_CONTENT
            )

        adaptive_tokens = calculate_adaptive_max_tokens(
            error.chunk_content,
            base_tokens=settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS,
            min_tokens=settings.DOC_CONCEPT_MIN_OUTPUT_TOKENS,
            max_tokens=settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS,
            tokens_per_char=settings.DOC_CONCEPT_TOKENS_PER_CHAR,
        )

        model_max = self._get_model_max_tokens()
        dlq_multiplier = getattr(settings, "DOC_CONCEPT_DLQ_TOKEN_MULTIPLIER", 4.0)
        retry_tokens = min(int(adaptive_tokens * dlq_multiplier), model_max)

        try:
            result = await self.extractor.extract(
                error.chunk_content,
                error.chunk_qn or error.path,
                max_tokens=retry_tokens,
            )
            return AdaptiveRetryResult(
                success=True,
                concepts=len(result.concepts),
                relationships=len(result.relationships),
                tokens_used=retry_tokens,
                result=result,
            )
        except Exception as e:
            return AdaptiveRetryResult(
                success=False,
                reason=str(e),
                tokens_used=retry_tokens,
            )

    def _get_model_max_tokens(self) -> int:
        """Return the maximum output tokens supported by the active model."""
        return getattr(settings, "DOC_CONCEPT_MAX_OUTPUT_TOKENS", 16384)

    def retry_dlq(self, force: bool = False) -> StandaloneExtractionStats:
        """Retry concept extraction for DLQ entries.

        For CONCEPT_CONTEXT_OVERFLOW, first retries with adaptive max_tokens,
        then falls back to splitting the chunk content into smaller sub-chunks.

        Args:
            force: If True, retry even if chunk already has concepts.

        Returns:
            Stats object with retry results.
        """
        stats = StandaloneExtractionStats()

        with self._connect_graphs() as connections:
            if connections is None:
                logger.error(doc_ls.DUAL_GRAPH_CONNECT_FAILED)
                return stats

            doc_ingestor, concept_ingestor = connections

            if not self._initialize_extractor():
                logger.error(doc_ls.DOC_CONCEPT_INIT_FAILED.format(error="extractor init returned False"))
                return stats

            if self.dead_letter_queue is None:
                logger.warning(doc_ls.DOC_DLQ_UNAVAILABLE)
                return stats

            pending = self.dead_letter_queue.get_pending(include_scheduled=True)
            overflow_errors = [
                e for e in pending
                if e.error_type == ErrorType.CONCEPT_CONTEXT_OVERFLOW
            ]

            if not overflow_errors:
                logger.info(doc_ls.DOC_DLQ_NO_OVERFLOW_ERRORS)
                return stats

            stats.total_chunks = len(overflow_errors)
            logger.info(
                doc_ls.DOC_DLQ_RETRY_START.format(count=len(overflow_errors))
            )

            for error in overflow_errors:
                stats.processed_chunks += 1

                content = error.chunk_content
                if not content and error.chunk_qn:
                    # Fetch from doc graph if content not stored
                    results = doc_ingestor.fetch_all(
                        "MATCH (c:Chunk {qualified_name: $qn}) RETURN c.content AS content",
                        {"qn": error.chunk_qn},
                    )
                    if results:
                        content = results[0].get("content", "")

                if not content:
                    logger.warning(
                        doc_ls.DOC_DLQ_NO_CONTENT.format(chunk_qn=error.chunk_qn)
                    )
                    stats.failed_extractions += 1
                    continue

                chunk_qn = error.chunk_qn or error.path

                # Strategy 1: Adaptive max_tokens
                adaptive_result = asyncio.run(self._retry_with_adaptive_tokens(error))
                if adaptive_result.success and adaptive_result.result is not None:
                    if force:
                        self._cleanup_existing_concepts([chunk_qn], concept_ingestor)
                    self._store_extraction_results(
                        {"qualified_name": chunk_qn, "content": content},
                        adaptive_result.result,
                        concept_ingestor,
                    )
                    self.dead_letter_queue.remove(error)
                    stats.successful_extractions += 1
                    stats.concepts_created += adaptive_result.concepts
                    stats.relationships_created += adaptive_result.relationships
                    continue

                if adaptive_result.reason:
                    logger.info(
                        doc_ls.DOC_DLQ_ADAPTIVE_FAILED.format(
                            chunk_qn=chunk_qn, reason=adaptive_result.reason
                        )
                    )

                # Strategy 2: Split-based retry
                sub_chunks = _split_chunk(content)
                sub_results: list[ExtractionResult] = []
                any_success = False

                for sub in sub_chunks:
                    try:
                        result = asyncio.run(self.extractor.extract_with_retry(
                            sub,
                            chunk_qn,
                            dead_letter_queue=None,
                        ))
                        if result.concepts or result.relationships:
                            any_success = True
                        sub_results.append(result)
                    except Exception as e:
                        logger.debug(f"Sub-chunk retry failed: {e}")
                        sub_results.append(ExtractionResult())

                if not any_success:
                    stats.failed_extractions += 1
                    updated = self.dead_letter_queue.mark_retry_attempt(error)
                    self.dead_letter_queue.enqueue(updated)
                    continue

                # Merge and deduplicate sub-chunk results
                merged = ExtractionResult()
                consolidator = ConceptConsolidator()
                for result in sub_results:
                    for concept in result.concepts:
                        node = {
                            "qualified_name": f"{self.workspace}:{concept.name}",
                            "workspace": self.workspace,
                            "name": concept.name,
                            "aliases": concept.aliases,
                            "type": concept.type,
                            "definition": concept.definition,
                            "confidence": concept.confidence,
                            "entity_category": concept.entity_category,
                            "entity_subtype": concept.entity_subtype,
                            "entity_emoji": concept.entity_emoji,
                            "context": concept.context,
                        }
                        consolidator.add(node, chunk_qn)
                    merged.relationships.extend(result.relationships)

                deduped_nodes = consolidator.consolidate()
                merged.concepts = [
                    ExtractedConcept(
                        name=n["name"],
                        aliases=n["aliases"],
                        type=n["type"],
                        definition=n["definition"],
                        confidence=n["confidence"],
                        context=n.get("context", ""),
                        entity_category=n["entity_category"],
                        entity_subtype=n["entity_subtype"],
                        entity_emoji=n["entity_emoji"],
                    )
                    for n in deduped_nodes
                ]

                # Remove duplicates from merged relationships
                seen_rels: set[tuple] = set()
                unique_rels = []
                for rel in merged.relationships:
                    key = (rel.from_concept, rel.to_concept, rel.verb, rel.category)
                    if key not in seen_rels:
                        seen_rels.add(key)
                        unique_rels.append(rel)
                merged.relationships = unique_rels

                if force:
                    self._cleanup_existing_concepts([chunk_qn], concept_ingestor)
                self._store_extraction_results(
                    {"qualified_name": chunk_qn, "content": content},
                    merged,
                    concept_ingestor,
                )
                self.dead_letter_queue.remove(error)
                stats.successful_extractions += 1
                stats.concepts_created += len(merged.concepts)
                stats.relationships_created += len(merged.relationships)

        return stats

    def _store_concept_relationships_batch(
        self,
        ingestor: MemgraphIngestor,
        relationships: list[tuple[str, str, str, str, str, float]],
        workspace: str,
    ) -> None:
        """Batch create relationships between concepts.

        Mirrors DocumentGraphUpdater._store_concept_relationships_batch() for consistency.
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
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'COMPOSITIONAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:COMPOSITIONAL]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'CONTEXTUAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:CONTEXTUAL]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'ATTRIBUTIVE' THEN [1] ELSE [] END |
                MERGE (a)-[r:ATTRIBUTIVE]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'COMPARATIVE' THEN [1] ELSE [] END |
                MERGE (a)-[r:COMPARATIVE]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'SEQUENTIAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:SEQUENTIAL]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'CAUSAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:CAUSAL]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'ANALOGICAL' THEN [1] ELSE [] END |
                MERGE (a)-[r:ANALOGICAL]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            FOREACH (_ IN CASE WHEN rel.category = 'RELATED_TO' THEN [1] ELSE [] END |
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.verb = rel.verb, r.category = rel.category, r.emoji = rel.emoji, r.strength = rel.strength
            )
            """
            ingestor.fetch_all(cypher, {"rels": list(batch), "workspace": workspace})


__all__ = [
    "AdaptiveRetryResult",
    "StandaloneExtractionStats",
    "ConceptExtractionRunner",
]
