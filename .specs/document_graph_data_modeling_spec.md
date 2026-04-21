# Document Graph Data Modeling and Query Optimization Design Spec

## Executive Summary

The current document graph architecture treats documents as flat collections of `Section` and `Chunk` nodes with no semantic relationships between concepts. This limits the system to keyword/semantic search only, preventing graph-based queries like "shortest path between concepts" or "related concepts."

This spec addresses:
1. Document graph schema enhancements for concept relationships
2. Integration with unified LLM-First query orchestration
3. Graph algorithm support for document knowledge graphs

## Prerequisites

The following must be implemented **before** this work begins:

| Prerequisite | Status | Notes |
|-------------|--------|-------|
| `llm_first_orchestration_spec.md` | **Required** | This spec uses `LLMQueryPlanner` and `QueryPlan` from that spec for all intent classification and entity extraction |

**Key Integration Points:**
- `QueryIntent` enum is defined in `llm_query_planner.py` (includes document intents)
- `QueryPlan.expected_entities` provides LLM-extracted entities (replaces regex extraction)
- `LLMSufficiencyAnalyzer` handles result completeness assessment

**Do NOT implement:**
- A separate keyword-based `QueryIntentClassifier`
- Regex-based concept extraction from queries

---

## Problem Analysis

### Issue 1: Missing Concept-to-Concept Relationships

**Evidence from logs:**
```
Querying document graph: what is the shortest path for categorical thinking?
Querying document graph: shortest path for categorical thinking
Querying document graph: categorical thinking quick start or beginner guide
```

The query "shortest path for categorical thinking" was:
1. Classified as `document_conceptual_query`
2. Routed to semantic search
3. Answered by reading files, not graph traversal

**Root cause:** The document graph schema only has:
```
Document -[:CONTAINS_SECTION]-> Section -[:HAS_SUBSECTION]-> Section
Document -[:CONTAINS_CHUNK]-> Chunk
Chunk -[:BELONGS_TO_SECTION]-> Section
```

There are no relationships between concepts across documents/sections.

### Issue 2: Query Misclassification

**Evidence from logs:**
```
Task not eligible for parallel execution: DOCUMENT_ONLY mode with conceptual question - routing to semantic search
Parallel execution skipped: task_type=document_conceptual_query, confidence=0.00
Routing document conceptual query to semantic search
```

**Current classification logic** (`concurrency_eligibility_classifier.py:374-386`):
```python
if query_mode == QueryMode.DOCUMENT_ONLY:
    if self._is_conceptual_question(prompt):
        return EligibilityResult(
            False,
            "document_conceptual_query",
            0.0,
            fallback_action="semantic_search",
        )
```

The classifier treats "shortest path" as a conceptual question, not a graph traversal request.

**Resolution:** This is fixed by `llm_first_orchestration_spec.md` which removes all regex-based classification. The `LLMQueryPlanner` correctly identifies graph traversal intent.

### Issue 3: No Graph Algorithms for Documents

The system supports PageRank and community detection for code graphs (`graph_algorithms.py`), but these are not available for document graphs.

---

## Proposed Solution

### Part 1: Enhanced Document Graph Schema

#### 1.1 New Node Types

```cypher
// Concept nodes extracted from document content
CREATE (:Concept {
    name: string,              // e.g., "categorical thinking"
    qualified_name: string,    // "{workspace}:{name}" for multi-tenancy
    aliases: list[string],     // e.g., ["category theory", "lens switching"]
    definition: string,        // Brief definition
    confidence: float,         // Extraction confidence
    source_chunk_qn: string,   // Qualified name of source chunk
    workspace: string          // Multi-tenant workspace identifier
})

// Topic nodes for high-level categorization
CREATE (:Topic {
    name: string,
    qualified_name: string,    // "{workspace}:{name}"
    description: string,
    workspace: string
})
```

#### 1.2 New Relationship Types

```cypher
// Concept relationships (extracted from content)
(:Concept)-[:RELATED_TO {strength: float, context: string}]->(:Concept)
(:Concept)-[:IS_A]->(:Concept)           // Hierarchical
(:Concept)-[:PART_OF]->(:Concept)        // Compositional
(:Concept)-[:CAUSES]->(:Concept)         // Causal

// Document-Concept links
(:Chunk)-[:MENTIONS {frequency: int, context: string}]->(:Concept)
(:Section)-[:DISCUSSES]->(:Concept)
(:Document)-[:COVERS]->(:Topic)

// Concept-Topic categorization
(:Concept)-[:BELONGS_TO_TOPIC]->(:Topic)
```

#### 1.3 Constants Update

**Modify:** `codebase_rag/constants.py`

```python
class NodeLabel(StrEnum):
    # ... existing values ...
    CONCEPT = "Concept"
    TOPIC = "Topic"


class RelationshipType(StrEnum):
    # ... existing values ...
    RELATED_TO = "RELATED_TO"
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    CAUSES = "CAUSES"
    MENTIONS = "MENTIONS"
    DISCUSSES = "DISCUSSES"
    COVERS = "COVERS"
    BELONGS_TO_TOPIC = "BELONGS_TO_TOPIC"
```

Update `_NODE_LABEL_UNIQUE_KEYS`:
```python
NodeLabel.CONCEPT: UniqueKeyType.QUALIFIED_NAME,
NodeLabel.TOPIC: UniqueKeyType.QUALIFIED_NAME,
```

Update `_NODE_LABEL_UNIQUE_KEYS`:
```python
NodeLabel.CONCEPT: UniqueKeyType.QUALIFIED_NAME,
NodeLabel.TOPIC: UniqueKeyType.QUALIFIED_NAME,
```

**Note:** `NODE_UNIQUE_CONSTRAINTS` is auto-derived from `_NODE_LABEL_UNIQUE_KEYS`
in `constants.py` (`{label.value: key.value for label, key in _NODE_LABEL_UNIQUE_KEYS.items()}`),
so no separate update is needed.

#### 1.4 Document Log Constants

**New file:** `codebase_rag/document/logs.py`

```python
"""Log message constants for document graph operations."""

# Concept extraction
DOC_CONCEPT_EXTRACT_START = "Extracting concepts from {chunk_count} chunks"
DOC_CONCEPT_EXTRACT_DONE = "Extracted {concept_count} concepts from document"
DOC_CONCEPT_STORE_BATCH = "Storing {count} concept nodes via batch MERGE"
DOC_MENTIONS_STORE_BATCH = "Creating {count} MENTIONS relationships"
DOC_REL_STORE_BATCH = "Creating {count} concept-to-concept relationships"

# Graph algorithms
DOC_SHORTEST_PATH_QUERY = "Finding shortest path: {source} -> {target}"
DOC_SHORTEST_PATH_FOUND = "Found path of length {length} between {source} and {target}"
DOC_SHORTEST_PATH_NONE = "No path found between {source} and {target}"
DOC_RELATED_CONCEPTS_QUERY = "Finding concepts related to: {concept}"
DOC_NEIGHBORS_QUERY = "Finding neighbors of: {concept} at depth {depth}"
DOC_GRAPH_ALGO_ERROR = "Document graph algorithm failed: {error}"

# Real-time updates
DOC_CONCEPT_CLEANUP_START = "Cleaning up orphaned concepts for document: {doc_path}"
DOC_CONCEPT_CLEANUP_DONE = "Removed {count} orphaned concepts"
```

#### 1.5 Schema Implementation

**New file:** `codebase_rag/document/concept_extraction.py`

```python
"""Concept extraction from document chunks using LLM."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Protocol


class ExtractedConcept(BaseModel):
    """A concept extracted from document content."""

    name: str = Field(..., description="Concept name")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names for this concept",
    )
    definition: str = Field(..., description="Brief definition")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence",
    )
    source_chunk_qn: str = Field(..., description="Source chunk qualified name")
    context: str = Field(default="", description="Surrounding context")


class ConceptRelationship(BaseModel):
    """A relationship between two concepts."""

    from_concept: str = Field(..., description="Source concept name")
    to_concept: str = Field(..., description="Target concept name")
    relationship_type: str = Field(
        ...,
        description="Type: RELATED_TO, IS_A, PART_OF, CAUSES",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relationship strength",
    )


class ExtractionResult(BaseModel):
    """Result of concept extraction from a chunk."""

    concepts: list[ExtractedConcept] = Field(default_factory=list)
    relationships: list[ConceptRelationship] = Field(default_factory=list)


class ConceptExtractor(Protocol):
    """Protocol for concept extraction strategies."""

    async def extract(self, chunk_content: str, chunk_qn: str) -> ExtractionResult:
        """Extract concepts and relationships from a text chunk.

        Args:
            chunk_content: The text content of the chunk.
            chunk_qn: Qualified name of the chunk for attribution.

        Returns:
            ExtractionResult with concepts and relationships.
        """
        ...


class LLMConceptExtractor:
    """LLM-based concept extraction implementation."""

    __slots__ = ("agent",)

    SYSTEM_PROMPT = """You are a concept extractor for technical documentation.
Given a document chunk, identify:
1. Key concepts mentioned (with definitions if available)
2. Relationships between concepts (RELATED_TO, IS_A, PART_OF, CAUSES)

Respond with JSON matching this structure:
{
  "concepts": [
    {
      "name": "concept name",
      "aliases": ["alternative name"],
      "definition": "brief definition",
      "confidence": 0.9
    }
  ],
  "relationships": [
    {
      "from_concept": "source",
      "to_concept": "target",
      "relationship_type": "RELATED_TO",
      "strength": 0.8
    }
  ]
}

Rules:
- Only extract concepts that are clearly defined or important in the text
- Confidence should reflect how clearly the concept is presented
- Relationship types must be one of: RELATED_TO, IS_A, PART_OF, CAUSES
- Strength reflects how explicitly the relationship is stated"""

    def __init__(self) -> None:
        self.agent = None

    def _initialize_agent(self) -> None:
        """Lazy initialization."""
        if self.agent is None:
            from pydantic_ai import Agent
            from codebase_rag.config import settings
            from codebase_rag.providers import _create_provider_model

            config = settings.active_orchestrator_config
            llm = _create_provider_model(config)

            self.agent = Agent(
                model=llm,
                system_prompt=self.SYSTEM_PROMPT,
                output_type=ExtractionResult,
                retries=1,
            )

    async def extract(self, chunk_content: str, chunk_qn: str) -> ExtractionResult:
        """Extract concepts from chunk content."""
        self._initialize_agent()

        try:
            result = await self.agent.run(chunk_content)
            # Add source attribution
            for concept in result.output.concepts:
                concept.source_chunk_qn = chunk_qn
            return result.output
        except Exception as e:
            from loguru import logger
            logger.warning(f"Concept extraction failed for {chunk_qn}: {e}")
            return ExtractionResult()
```

**Update:** `codebase_rag/document/document_updater.py`

Add imports:
```python
from itertools import batched

from . import logs as doc_ls
from .concept_extraction import ConceptExtractor, LLMConceptExtractor, ExtractionResult
```

Add `concept_extractor` to `__init__`:
```python
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
) -> None:
    # ... existing init ...
    self.concept_extractor = concept_extractor or (
        LLMConceptExtractor() if settings.DOC_CONCEPT_EXTRACTION_ENABLED else None
    )
```

Add concept extraction phase after chunk creation:

```python
async def _extract_and_store_concepts(
    self,
    chunks: list[DocumentChunk],
    ingestor: MemgraphIngestor,
    workspace: str,
) -> None:
    """Extract concepts from chunks and store in graph via batch MERGE.

    Uses LLM extraction (not regex) for semantic concept identification.
    Extraction runs concurrently with a configurable semaphore to limit
    parallel LLM calls.
    """
    if not self.concept_extractor:
        return

    logger.info(doc_ls.DOC_CONCEPT_EXTRACT_START.format(chunk_count=len(chunks)))

    import asyncio

    semaphore = asyncio.Semaphore(
        getattr(settings, "DOC_CONCEPT_EXTRACTION_CONCURRENCY", 10)
    )

    async def _extract_one(chunk: DocumentChunk) -> ExtractionResult:
        async with semaphore:
            return await self.concept_extractor.extract(
                chunk.content, chunk.qualified_name
            )

    extraction_results = await asyncio.gather(
        *[_extract_one(c) for c in chunks]
    )

    concept_nodes: list[dict[str, object]] = []
    mention_rels: list[dict[str, object]] = []
    concept_relationships: list[tuple[str, str, str, float]] = []

    for chunk, result in zip(chunks, extraction_results):
        for concept in result.concepts:
            concept_qn = f"{workspace}:{concept.name}"
            concept_nodes.append({
                "qualified_name": concept_qn,
                "workspace": workspace,
                "name": concept.name,
                "aliases": concept.aliases,
                "definition": concept.definition,
                "confidence": concept.confidence,
                "source_chunk_qn": concept.source_chunk_qn,
            })
            # Compute actual mention frequency in chunk content
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
                    rel.relationship_type,
                    rel.strength,
                )
            )

    logger.info(
        doc_ls.DOC_CONCEPT_EXTRACT_DONE.format(concept_count=len(concept_nodes))
    )

    # Batch MERGE concept nodes via UNWIND (idempotent)
    if concept_nodes:
        self._merge_concept_nodes_batch(ingestor, concept_nodes)

    # Batch CREATE MENTIONS relationships
    if mention_rels:
        self._create_mentions_batch(ingestor, mention_rels, workspace)

    # Batch CREATE concept-to-concept relationships
    if concept_relationships:
        self._store_concept_relationships_batch(ingestor, concept_relationships, workspace)


def _merge_concept_nodes_batch(
    self,
    ingestor: MemgraphIngestor,
    concept_nodes: list[dict[str, object]],
) -> None:
    """Batch merge Concept nodes using UNWIND for efficiency."""
    batch_size = settings.DOC_MEMGRAPH_BATCH_SIZE
    for batch in batched(concept_nodes, batch_size):
        cypher = """
        UNWIND $nodes as node
        MERGE (c:Concept {qualified_name: node.qualified_name})
        SET c.workspace = node.workspace,
            c.name = node.name,
            c.aliases = node.aliases,
            c.definition = node.definition,
            c.confidence = node.confidence,
            c.source_chunk_qn = node.source_chunk_qn
        """
        ingestor.fetch_all(cypher, {"nodes": list(batch)})


def _create_mentions_batch(
    self,
    ingestor: MemgraphIngestor,
    mention_rels: list[dict[str, object]],
    workspace: str,
) -> None:
    """Batch create MENTIONS relationships from chunks to concepts."""
    batch_size = settings.DOC_MEMGRAPH_BATCH_SIZE
    for batch in batched(mention_rels, batch_size):
        cypher = """
        UNWIND $rels as rel
        MATCH (c:Chunk {qualified_name: rel.chunk_qn, workspace: $workspace})
        MATCH (concept:Concept {qualified_name: rel.concept_qn, workspace: $workspace})
        MERGE (c)-[m:MENTIONS]->(concept)
        SET m.frequency = rel.frequency, m.context = rel.context
        """
        ingestor.fetch_all(cypher, {"rels": list(batch), "workspace": workspace})


def _store_concept_relationships_batch(
    self,
    ingestor: MemgraphIngestor,
    relationships: list[tuple[str, str, str, float]],
    workspace: str,
) -> None:
    """Batch create relationships between concepts.

    Memgraph does not support parameterized relationship types, so we use
    explicit FOREACH branches per type. This preserves semantic graph structure
    (IS_A, PART_OF, CAUSES are distinct edge types) rather than collapsing
    everything into RELATED_TO with a property.
    """
    rel_maps: list[dict[str, object]] = [
        {
            "from_qn": from_qn,
            "to_qn": to_qn,
            "rel_type": rel_type,
            "strength": strength,
        }
        for from_qn, to_qn, rel_type, strength in relationships
    ]
    batch_size = settings.DOC_MEMGRAPH_BATCH_SIZE
    for batch in batched(rel_maps, batch_size):
        cypher = """
        UNWIND $rels as rel
        MATCH (a:Concept {qualified_name: rel.from_qn, workspace: $workspace})
        MATCH (b:Concept {qualified_name: rel.to_qn, workspace: $workspace})
        FOREACH (_ IN CASE WHEN rel.rel_type = 'RELATED_TO' THEN [1] ELSE [] END |
            MERGE (a)-[r:RELATED_TO]->(b) SET r.strength = rel.strength
        )
        FOREACH (_ IN CASE WHEN rel.rel_type = 'IS_A' THEN [1] ELSE [] END |
            MERGE (a)-[r:IS_A]->(b) SET r.strength = rel.strength
        )
        FOREACH (_ IN CASE WHEN rel.rel_type = 'PART_OF' THEN [1] ELSE [] END |
            MERGE (a)-[r:PART_OF]->(b) SET r.strength = rel.strength
        )
        FOREACH (_ IN CASE WHEN rel.rel_type = 'CAUSES' THEN [1] ELSE [] END |
            MERGE (a)-[r:CAUSES]->(b) SET r.strength = rel.strength
        )
        """
        ingestor.fetch_all(cypher, {"rels": list(batch), "workspace": workspace})
```

**Index Creation:**

Add the following helper to `codebase_rag/document/document_updater.py` and call it
once before the first concept extraction (guarded by an idempotency check):

```python
def _ensure_concept_indexes(ingestor: MemgraphIngestor) -> None:
    """Create indexes for Concept and Topic nodes if they do not exist."""
    cypher = """
    CREATE INDEX concept_qualified_name_index IF NOT EXISTS
    FOR (c:Concept) ON (c.qualified_name);

    CREATE INDEX concept_workspace_index IF NOT EXISTS
    FOR (c:Concept) ON (c.workspace);

    CREATE INDEX topic_qualified_name_index IF NOT EXISTS
    FOR (t:Topic) ON (t.qualified_name);

    CREATE INDEX topic_workspace_index IF NOT EXISTS
    FOR (t:Topic) ON (t.workspace);
    """
    try:
        ingestor.fetch_all(cypher)
    except Exception as e:
        logger.warning(f"Concept index creation failed (may already exist): {e}")
```

**Note on scope:** Phase 1 extracts relationships only between concepts co-occurring in the same chunk. Cross-document/cross-section relationship extraction is deferred to a future milestone (see Risks).

---

### Part 2: Query Classification Integration

This part integrates with the LLM-First Orchestration spec. No separate implementation needed.

#### 2.1 Query Intent (Already Defined)

The `QueryIntent` enum in `llm_query_planner.py` already includes document intents:

```python
class QueryIntent(StrEnum):
    # Code graph intents
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    SEMANTIC = "semantic"
    EXPLORATORY = "exploratory"
    VALIDATION = "validation"
    # Document graph intents
    DOC_SEMANTIC_SEARCH = "doc_semantic_search"
    DOC_GRAPH_TRAVERSAL = "doc_graph_traversal"
    DOC_COMPARISON = "doc_comparison"
    DOC_PROCEDURAL = "doc_procedural"
```

#### 2.2 Entity Extraction (Use LLM Plan)

**DO NOT implement regex-based concept extraction from queries.**

Instead, use `QueryPlan.expected_entities` from `LLMQueryPlanner.plan()`:

```python
# In query_router.py
from codebase_rag.orchestrator.llm_query_planner import LLMQueryPlanner

async def _query_document_only(self, request: QueryRequest) -> QueryResponse:
    """Query document graph using LLM-provided entities."""
    # Get the plan with entities already extracted by LLM
    if request.plan:
        concepts = request.plan.expected_entities
    else:
        # Fallback: get plan if not provided
        planner = LLMQueryPlanner()
        plan = await planner.plan(request.question)
        concepts = plan.expected_entities

    # Use concepts for graph traversal...
```

---

### Part 3: Graph Algorithm Support for Documents

#### 3.1 Document Graph Algorithms

**New file:** `codebase_rag/document/graph_algorithms.py`

```python
"""Graph algorithms for document knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from codebase_rag.config import settings
from . import logs as doc_ls

if TYPE_CHECKING:
    from codebase_rag.services import QueryProtocol


@dataclass
class ConceptPath:
    """Path between two concepts in the document graph."""

    source: str
    target: str
    path: list[str]
    path_length: int
    relationships: list[str]
    formatted_path: str


class DocumentGraphAlgorithms:
    """Graph algorithms for document knowledge graphs."""

    __slots__ = ("graph", "workspace", "max_path_depth")

    def __init__(
        self,
        graph: QueryProtocol,
        workspace: str = "default",
        max_path_depth: int = 5,
    ) -> None:
        self.graph = graph
        self.workspace = workspace
        # Clamp depth to configured limits
        self.max_path_depth = max(
            1,
            min(
                max_path_depth,
                getattr(settings, "DOC_GRAPH_MAX_PATH_DEPTH", 5),
            ),
        )

    async def find_shortest_path(
        self,
        source_concept: str,
        target_concept: str,
    ) -> ConceptPath | None:
        """Find shortest path between two concepts.

        Returns None if graph is not available or no path exists.

        Args:
            source_concept: Source concept name (not qualified name).
            target_concept: Target concept name (not qualified name).

        Returns:
            ConceptPath if found, None otherwise.
        """
        source_qn = f"{self.workspace}:{source_concept}"
        target_qn = f"{self.workspace}:{target_concept}"
        max_depth = self.max_path_depth

        if self.graph is None:
            logger.warning("Document graph not available for shortest path")
            return None

        logger.info(
            doc_ls.DOC_SHORTEST_PATH_QUERY.format(
                source=source_concept,
                target=target_concept,
            )
        )

        try:
            # NOTE: Memgraph does not support parameterized depth bounds in
            # variable-length patterns. max_depth is validated and clamped above.
            # This follows the pattern used in codebase_rag/graph_algorithms.py.
            query = f"""
            MATCH path = shortestPath(
                (source:Concept)-[:RELATED_TO|IS_A|PART_OF|CAUSES*1..{max_depth}]-(target:Concept)
            )
            WHERE source.qualified_name = $source_qn
              AND target.qualified_name = $target_qn
              AND source.workspace = $workspace
              AND target.workspace = $workspace
            RETURN
                [n in nodes(path) | n.name] as concept_names,
                [r in relationships(path) | type(r)] as rel_types,
                length(path) as path_length
            """
            results = await self.graph.fetch_all_async(
                query,
                {
                    "source_qn": source_qn,
                    "target_qn": target_qn,
                    "workspace": self.workspace,
                },
            )

            if not results:
                logger.info(
                    doc_ls.DOC_SHORTEST_PATH_NONE.format(
                        source=source_concept,
                        target=target_concept,
                    )
                )
                return None

            row = results[0]
            path_obj = ConceptPath(
                source=source_concept,
                target=target_concept,
                path=row["concept_names"],
                path_length=row["path_length"],
                relationships=row["rel_types"],
                formatted_path=self._generate_path_description(row),
            )
            logger.info(
                doc_ls.DOC_SHORTEST_PATH_FOUND.format(
                    source=source_concept,
                    target=target_concept,
                    length=path_obj.path_length,
                )
            )
            return path_obj

        except Exception as e:
            logger.error(doc_ls.DOC_GRAPH_ALGO_ERROR.format(error=e))
            return None

    async def find_related_concepts(
        self,
        concept: str,
        limit: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Find concepts related to a given concept.

        Args:
            concept: Concept name (not qualified name).
            limit: Maximum number of related concepts to return.

        Returns:
            List of (concept_name, relationship_type, strength) tuples.
        """
        if self.graph is None:
            logger.warning("Document graph not available for related concepts")
            return []

        concept_qn = f"{self.workspace}:{concept}"
        logger.info(doc_ls.DOC_RELATED_CONCEPTS_QUERY.format(concept=concept))

        try:
            query = """
            MATCH (c:Concept)-[r:RELATED_TO|IS_A|PART_OF|CAUSES]-(related:Concept)
            WHERE c.qualified_name = $concept_qn
              AND c.workspace = $workspace
              AND related.workspace = $workspace
            RETURN
                related.name as concept,
                type(r) as relationship,
                coalesce(r.strength, 0.5) as strength
            ORDER BY strength DESC
            LIMIT $limit
            """
            results = await self.graph.fetch_all_async(
                query,
                {
                    "concept_qn": concept_qn,
                    "limit": limit,
                    "workspace": self.workspace,
                },
            )
            return [
                (r["concept"], r["relationship"], r["strength"])
                for r in results
            ]
        except Exception as e:
            logger.error(doc_ls.DOC_GRAPH_ALGO_ERROR.format(error=e))
            return []

    async def find_concept_neighbors(
        self,
        concept: str,
        depth: int = 1,
    ) -> dict[str, list[str]]:
        """Find all concepts within N hops of a concept.

        Args:
            concept: Concept name (not qualified name).
            depth: Maximum hop distance (clamped to configured max).

        Returns:
            Dict with "neighbors" key containing list of concept names.
        """
        if self.graph is None:
            logger.warning("Document graph not available for neighbor search")
            return {"neighbors": []}

        concept_qn = f"{self.workspace}:{concept}"
        validated_depth = max(
            1,
            min(depth, getattr(settings, "DOC_GRAPH_MAX_PATH_DEPTH", 5)),
        )

        logger.info(
            doc_ls.DOC_NEIGHBORS_QUERY.format(concept=concept, depth=validated_depth)
        )

        try:
            # NOTE: Depth is validated and clamped. See note in find_shortest_path.
            query = f"""
            MATCH (c:Concept)-[:RELATED_TO|IS_A|PART_OF|CAUSES*1..{validated_depth}]-(neighbor:Concept)
            WHERE c.qualified_name = $concept_qn
              AND c.workspace = $workspace
              AND neighbor.workspace = $workspace
            RETURN DISTINCT neighbor.name as neighbor
            """
            results = await self.graph.fetch_all_async(
                query,
                {"concept_qn": concept_qn, "workspace": self.workspace},
            )
            return {"neighbors": [r["neighbor"] for r in results]}
        except Exception as e:
            logger.error(doc_ls.DOC_GRAPH_ALGO_ERROR.format(error=e))
            return {"neighbors": []}

    @staticmethod
    def _generate_path_description(row: dict[str, object]) -> str:
        """Generate human-readable description of a concept path."""
        names_obj = row.get("concept_names")
        rels_obj = row.get("rel_types")
        names = names_obj if isinstance(names_obj, list) else []
        rels = rels_obj if isinstance(rels_obj, list) else []
        if not names:
            return "No path found."
        parts = [str(names[0])]
        for i, rel in enumerate(rels):
            if i + 1 < len(names):
                parts.append(f"  --[{str(rel).lower()}]--> {names[i + 1]}")
        return "\n".join(parts)
```

#### 3.2 Tool Integration

**Update:** `codebase_rag/tools/document_query.py`

Update the existing `create_query_document_graph_tool` factory to add graph
traversal support while preserving the existing factory pattern and
initialization logic (`MemgraphIngestor` directly, not nonexistent helpers).

```python
"""Document query tool with graph traversal support."""

from __future__ import annotations

from loguru import logger
from pydantic_ai import Tool

from codebase_rag import constants as cs
from codebase_rag.config import settings
from codebase_rag.document.graph_algorithms import DocumentGraphAlgorithms
from codebase_rag.orchestrator.llm_query_planner import LLMQueryPlanner, QueryIntent
from codebase_rag.shared.query_router import QueryMode, QueryRequest, QueryRouter
from codebase_rag.tools import tool_descriptions as td


def create_query_document_graph_tool(
    query_router: QueryRouter | None = None,
) -> Tool:
    """Create query_document_graph tool with graph traversal support.

    The LLM agent decides intent and extracts entities via QueryPlan.
    This tool does NOT perform its own intent classification.
    """

    async def query_document_graph(
        natural_language_query: str,
        top_k: int = 5,
        include_paths: bool = False,
        source_concept: str | None = None,
        target_concept: str | None = None,
    ) -> str:
        """Query document graph with optional graph traversal support.

        Args:
            natural_language_query: The user's question about documents.
            top_k: Number of results to return.
            include_paths: If True, attempt graph traversal between concepts.
            source_concept: Optional explicit source concept for path queries.
            target_concept: Optional explicit target concept for path queries.

        Returns:
            Formatted results from document graph query.
        """
        logger.info(f"Querying document graph: {natural_language_query[:50]}...")

        router = query_router
        if router is None:
            router = _create_document_query_router()
            if router is None:
                return cs.MSG_SEMANTIC_NO_RESULTS.format(query=natural_language_query)

        workspace = getattr(router, "workspace", "default")

        # Get LLM plan for intent and entity extraction
        planner = LLMQueryPlanner()
        plan = await planner.plan(natural_language_query)

        # Handle graph traversal intent
        if plan.intent == QueryIntent.DOC_GRAPH_TRAVERSAL or include_paths:
            concepts = plan.expected_entities

            # Use explicit concepts if provided
            if source_concept and target_concept:
                concepts = [source_concept, target_concept]
            elif source_concept:
                concepts = [source_concept] + concepts[:1]

            if len(concepts) >= 2:
                algo = DocumentGraphAlgorithms(
                    graph=router.doc_graph,
                    workspace=workspace,
                )
                path = await algo.find_shortest_path(concepts[0], concepts[1])
                if path:
                    return _format_path_result(path)

            if len(concepts) == 1:
                algo = DocumentGraphAlgorithms(
                    graph=router.doc_graph,
                    workspace=workspace,
                )
                related = await algo.find_related_concepts(concepts[0])
                if related:
                    return _format_related_concepts(related)

        # Default: semantic search
        request = QueryRequest(
            question=natural_language_query,
            mode=QueryMode.DOCUMENT_ONLY,
            top_k=top_k,
            plan=plan,  # Pass plan for routing decisions
        )

        try:
            response = router.query(request)
            if not response.sources:
                return f"No relevant documents found for: {natural_language_query}"

            result_lines = ["**Document Query Results:**\n"]
            for i, source in enumerate(response.sources, 1):
                result_lines.append(
                    f"{i}. **{source.qualified_name or source.path}** "
                    f"({source.node_type or 'Section'})"
                )
                if source.line_range:
                    result_lines.append(
                        f"   Lines: {source.line_range[0]}-{source.line_range[1]}"
                    )
            result_lines.append(f"\n\n**Answer:**\n{response.answer}")
            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"Document query failed: {e}")
            return f"Document query failed: {e}"

    return Tool(
        query_document_graph,
        name=td.AgenticToolName.QUERY_DOCUMENT_GRAPH,
        description=td.QUERY_DOCUMENT_GRAPH,
    )


def _format_path_result(path) -> str:
    lines = [
        "Found path between concepts:",
        "",
        path.formatted_path,
        "",
        f"Path length: {path.path_length} hops",
    ]
    return "\n".join(lines)


def _format_related_concepts(related) -> str:
    if not related:
        return "No related concepts found."
    lines = ["Related concepts:"]
    for name, rel_type, strength in related:
        lines.append(f"  - {name} ({rel_type}, strength: {strength:.2f})")
    return "\n".join(lines)
```

**Note:** `_create_document_query_router()` in the existing file already uses
`MemgraphIngestor` directly. Keep that implementation; do not replace it with
nonexistent `get_document_query_protocol()` helpers.

#### 3.3 Query Router Update

**Update:** `codebase_rag/shared/query_router.py`

Add `plan` field to `QueryRequest`. Since `query_router.py` already has
`from __future__ import annotations`, forward references work without
runtime imports:

```python
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codebase_rag.orchestrator.llm_query_planner import QueryPlan


@dataclass
class QueryRequest:
    """Explicit query request with mode specification."""

    question: str
    mode: QueryMode
    validate: bool = False
    include_metadata: bool = True
    top_k: int = 5
    scope: str = "all"
    use_orchestrator: bool = False
    plan: QueryPlan | None = None  # LLM-provided plan with entities
    forced: bool = False  # If True, bypass current_mode guards (LLM cross-mode)
```

Add graph traversal routing to `_query_document_only()`. Update mode guard
to respect `forced` (aligns with Part 5 of orchestration spec):

```python
from codebase_rag.document.graph_algorithms import ConceptPath
from codebase_rag.orchestrator.llm_query_planner import QueryIntent


def _format_path_result(path: ConceptPath) -> str:
    lines = [
        "Found path between concepts:",
        "",
        path.formatted_path,
        "",
        f"Path length: {path.path_length} hops",
    ]
    return "\n".join(lines)


def _format_related_concepts(
    related: list[tuple[str, str, float]],
) -> str:
    if not related:
        return "No related concepts found."
    lines = ["Related concepts:"]
    for name, rel_type, strength in related:
        lines.append(f"  - {name} ({rel_type}, strength: {strength:.2f})")
    return "\n".join(lines)


async def _query_document_only(self, request: QueryRequest) -> QueryResponse:
    """Query document graph, optionally using graph traversal."""
    if self.current_mode == QueryMode.CODE_ONLY and not request.forced:
        return QueryResponse(
            answer="Document queries are disabled in CODE_ONLY mode.",
            sources=[],
            mode=request.mode,
            warnings=["Document queries disabled in CODE_ONLY mode"],
        )

    if not self.doc_graph:
        return QueryResponse(
            answer="Document graph is not available.",
            sources=[],
            mode=request.mode,
            warnings=["Document graph unavailable"],
        )

    if not self.doc_vector:
        return QueryResponse(
            answer="Document vector backend is not available.",
            sources=[],
            mode=request.mode,
            warnings=["Document vector backend unavailable"],
        )

    logger.info(f"Querying document graph: {request.question}")

    try:
        # Use LLM-provided plan for routing decisions
        if request.plan and request.plan.intent == QueryIntent.DOC_GRAPH_TRAVERSAL:
            from codebase_rag.document.graph_algorithms import DocumentGraphAlgorithms

            workspace = getattr(self, "workspace", "default")
            algo = DocumentGraphAlgorithms(self.doc_graph, workspace=workspace)
            concepts = request.plan.expected_entities

            if len(concepts) >= 2:
                path = await algo.find_shortest_path(concepts[0], concepts[1])
                if path:
                    return QueryResponse(
                        answer=_format_path_result(path),
                        sources=[],
                        mode=request.mode,
                    )
            elif len(concepts) == 1:
                related = await algo.find_related_concepts(concepts[0])
                if related:
                    return QueryResponse(
                        answer=_format_related_concepts(related),
                        sources=[],
                        mode=request.mode,
                    )

        # Default: semantic/vector search
        results = self._fetch_document_results(request)
        return self._build_document_response(request, results)

    except Exception as e:
        return QueryResponse(
            answer=f"Document search failed: {e}",
            sources=[],
            mode=request.mode,
            warnings=[str(e)],
        )
```

**Note:** `_query_document_only` becomes `async` because graph algorithm
methods are async (using `fetch_all_async`). Update `QueryRouter.query()`
dispatch to handle this. Options:

1. Make `query()` async and `await` `_query_document_only()` when mode is `DOCUMENT_ONLY`.
2. Use `asyncio.run()` or `asyncio.get_event_loop().run_until_complete()` inside
   `query()` for the async branch (less preferred, may conflict with running loops).

Preferred: Add `async def query_async(...)` alongside sync `query()`, and have
the document tool call `query_async()` directly.

---

### Part 4: Real-Time Updater Integration

The project supports real-time document updates via `REALTIME_DOCS_ENABLED`. Concept extraction must integrate with this path.

**New behavior for real-time updates:**

1. When a document is modified, the real-time updater re-runs `_process_document()` or `_process_document_async()`.
2. The document updater deletes old `Chunk`, `Section`, and `Document` nodes before recreating them.
3. **Concept cleanup must be added:** Before deleting chunks, remove `MENTIONS` relationships from those chunks and orphan `Concept` nodes that are no longer mentioned by any chunk in the workspace.
4. **Concept re-extraction:** After new chunks are created, run `_extract_and_store_concepts()` for the new chunks only.

**Implementation:** Add `_cleanup_concepts_for_document()` to `DocumentGraphUpdater`:

```python
def _cleanup_concepts_for_document(
    self,
    document_path: str,
    ingestor: MemgraphIngestor,
) -> None:
    """Remove orphaned concepts after document chunks are deleted.

    This is called during real-time document updates to clean up
    concepts that are no longer referenced by any chunk.
    """
    logger.info(doc_ls.DOC_CONCEPT_CLEANUP_START.format(doc_path=document_path))

    cypher = """
    MATCH (d:Document {path: $doc_path, workspace: $workspace})-[:CONTAINS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[m:MENTIONS]->(concept:Concept)
    DELETE m
    WITH DISTINCT concept
    WHERE concept IS NOT NULL
      AND NOT EXISTS {
        MATCH (:Chunk)-[:MENTIONS]->(concept)
      }
    DETACH DELETE concept
    RETURN count(concept) as removed_count
    """
    result = ingestor.fetch_all(
        cypher,
        {"doc_path": document_path, "workspace": self.workspace},
    )

    removed_count = result[0].get("removed_count", 0) if result else 0
    logger.info(doc_ls.DOC_CONCEPT_CLEANUP_DONE.format(count=removed_count))
```

---

## Implementation Plan

### Phase 1: Concept Extraction (Week 1-2)

| Task | Files | Priority |
|------|-------|----------|
| Add `CONCEPT` and `TOPIC` to `NodeLabel`/`RelationshipType` | `constants.py` | High |
| Update `_NODE_LABEL_UNIQUE_KEYS` to use `QUALIFIED_NAME` | `constants.py` | High |
| Create `codebase_rag/document/logs.py` with document log constants | `document/logs.py` (new) | High |
| Define `ExtractedConcept`, `ConceptRelationship`, `ExtractionResult` Pydantic models | `document/concept_extraction.py` (new) | High |
| Implement `LLMConceptExtractor` with Pydantic output | `document/concept_extraction.py` | High |
| Add concept storage to document updater (batched via UNWIND) | `document/document_updater.py` | High |
| Add `_cleanup_concepts_for_document` for real-time updates | `document/document_updater.py` | High |
| Add concept index creation on first extraction | `document/document_updater.py` | High |
| Add tests for concept extraction | `tests/test_concept_extraction.py` (new) | Medium |

### Phase 2: Graph Algorithms (Week 2-3)

| Task | Files | Priority |
|------|-------|----------|
| Implement `DocumentGraphAlgorithms` class | `document/graph_algorithms.py` (new) | High |
| Add shortest path algorithm (validated Cypher with f-string depth) | `document/graph_algorithms.py` | High |
| Add related concepts finder | `document/graph_algorithms.py` | Medium |
| Update `query_document_graph` tool with graph traversal | `tools/document_query.py` | High |
| Add `plan` field to `QueryRequest` | `shared/query_router.py` | High |
| Add graph traversal routing to `query_router._query_document_only()` | `shared/query_router.py` | High |
| Add integration tests | `tests/test_document_graph_algorithms.py` (new) | Medium |

### Phase 3: Integration & Testing (Week 3-4)

| Task | Priority |
|------|----------|
| End-to-end testing with real document repos | High |
| Real-time updater concept cleanup validation | High |
| Performance benchmarking for concept extraction | Medium |
| Documentation updates | Medium |

---

## Configuration

### New Environment Variables

```bash
# Concept extraction settings
DOC_CONCEPT_EXTRACTION_ENABLED=true
DOC_CONCEPT_MIN_CONFIDENCE=0.7
DOC_CONCEPT_EXTRACTION_CONCURRENCY=10

# Graph algorithm settings
DOC_GRAPH_MAX_PATH_DEPTH=5
DOC_GRAPH_MAX_NEIGHBORS=50
```

### New Settings in `config.py`

Add under the existing `# DOCUMENT GRAPHRAG` block:

```python
from pydantic import Field

# Document concept extraction
DOC_CONCEPT_EXTRACTION_ENABLED: bool = True
DOC_CONCEPT_MIN_CONFIDENCE: float = Field(default=0.7, ge=0.0, le=1.0)
DOC_CONCEPT_EXTRACTION_CONCURRENCY: int = Field(default=10, ge=1, le=50)

# Document graph algorithms
DOC_GRAPH_MAX_PATH_DEPTH: int = Field(default=5, ge=1, le=10)
DOC_GRAPH_MAX_NEIGHBORS: int = Field(default=50, ge=1, le=500)
```

---

## Backward Compatibility

1. **Schema Migration**: Existing document graphs will have no `Concept` nodes initially. The system will:
   - Fall back to semantic search when no concepts exist
   - Support lazy concept extraction on first query (optional)

2. **API Compatibility**: All new parameters are optional with sensible defaults:
   - `include_paths=False` in `query_document_graph`
   - Concept extraction disabled if `DOC_CONCEPT_EXTRACTION_ENABLED=false`
   - `plan=None` in `QueryRequest` (defaults to existing behavior)

3. **Query Mode Compatibility**: Graph traversal is a new fallback action, not replacing existing modes.

4. **Graceful Degradation for Missing Concepts**: When `include_paths=True` but the graph has no `Concept` nodes, the tool falls back to semantic search rather than returning "No related concepts found."

5. **Real-Time Updater**: Existing real-time document update flows continue to work. Concept cleanup is additive and only runs when `DOC_CONCEPT_EXTRACTION_ENABLED=true`.

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| "Shortest path" queries resolved via graph | 0% | 80% |
| Concept extraction precision | N/A | >85% |
| Query classification accuracy (via LLM planner) | ~60% | >90% |
| Average query response time (graph traversal) | N/A | <2s |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM concept extraction is slow | Medium | Medium | Batch processing, async extraction, optional lazy extraction |
| Concept extraction noise | Medium | Low | Confidence thresholds, manual review |
| Graph traversal performance | Low | Medium | Index on `Concept.qualified_name`, limit depth, workspace filtering |
| LLM API costs for concept extraction | Medium | Low | Use local model by default |
| Cross-chunk relationships missed | High | Medium | Documented as Phase 1 limitation; global relationship extraction in future milestone |
| Real-time updater concept drift | Medium | Medium | `_cleanup_concepts_for_document` removes orphan concepts on every update |

---

## Appendix A: Cypher Queries for Concept Schema

```cypher
// Create indexes for performance
CREATE INDEX concept_qualified_name_index IF NOT EXISTS
FOR (c:Concept) ON (c.qualified_name);

CREATE INDEX concept_workspace_index IF NOT EXISTS
FOR (c:Concept) ON (c.workspace);

CREATE INDEX topic_qualified_name_index IF NOT EXISTS
FOR (t:Topic) ON (t.qualified_name);

CREATE INDEX topic_workspace_index IF NOT EXISTS
FOR (t:Topic) ON (t.workspace);

// Example query: Find all concepts mentioned in a document
MATCH (d:Document {path: $doc_path, workspace: $workspace})-[:HAS_SECTION]->(s:Section)
      -[:HAS_CHUNK]->(ch:Chunk)-[:MENTIONS]->(c:Concept)
WHERE c.workspace = $workspace
RETURN DISTINCT c.name, count(ch) as mention_count
ORDER BY mention_count DESC;
```

---

## Appendix B: Example Workflow

**Query:** "What is the shortest path between categorical thinking and decision making?"

**Processing:**
1. `LLMQueryPlanner.plan()` identifies:
   - `intent = QueryIntent.DOC_GRAPH_TRAVERSAL`
   - `expected_entities = ["categorical thinking", "decision making"]`
   - `methods = [QueryMethod.GRAPH_ALGORITHMS]`
2. `DocumentGraphAlgorithms.find_shortest_path()` executes Cypher
3. Path found:
   ```
   categorical thinking --[RELATED_TO]--> mental models --[PART_OF]--> decision making
   ```
4. Response:
   ```
   Found path between concepts:

   categorical thinking
     --[related_to]--> mental models
     --[part_of]--> decision making

   Path length: 2 hops
   ```
