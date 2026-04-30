"""JSON Graph Query operations.

Provides semantic search, graph traversal, and relationship analysis
for JSON-ingested entity data.

Integrates with:
- Existing embedding infrastructure for semantic search
- Memgraph vector index for similarity queries
- Consistent Cypher patterns with cypher_queries.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from .config import settings
from .services.graph_service import MemgraphIngestor


@dataclass
class JsonEntityResult:
    """Result from JSON entity query."""

    unique_id: str
    name: str
    entity_type: str
    entity_category: str
    entity_subtype: str
    entity_emoji: str
    description: str
    analogy: str = ""
    example: str = ""
    pagerank_score: float = 0.1
    combined_score: float = 0.0
    vector_score: float = 0.0
    text_score: float = 0.0
    community_id: int = -1
    emoji: str = ""

    def to_dict(self) -> dict[str, Any]:
        desc = self.description
        if len(desc) > 200:
            desc = desc[:200] + "..."
        return {
            "unique_id": self.unique_id,
            "name": self.name,
            "type": self.entity_type,
            "entity_category": self.entity_category,
            "entity_subtype": self.entity_subtype,
            "entity_emoji": self.entity_emoji,
            "emoji": self.emoji,
            "description": desc,
            "analogy": self.analogy,
            "example": self.example,
            "pagerank_score": self.pagerank_score,
            "combined_score": self.combined_score,
            "vector_score": self.vector_score,
            "text_score": self.text_score,
            "community_id": self.community_id,
        }


@dataclass
class JsonRelationshipResult:
    """Result from JSON relationship query."""

    from_entity: str
    to_entity: str
    relationship_type: str
    verb: str
    category: str = "RELATED_TO"
    emoji: str = "🔗"
    strength: float = 0.5
    from_type: str = ""
    to_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_entity,
            "to": self.to_entity,
            "relationship_type": self.relationship_type,
            "category": self.category,
            "emoji": self.emoji,
            # Legacy aliases for backward compatibility
            "relationship_category": self.category,
            "relationship_emoji": self.emoji,
            "verb": self.verb,
            "strength": self.strength,
            "from_type": self.from_type,
            "to_type": self.to_type,
        }


class JsonGraphQueryEngine:
    """Query engine for JSON graph entities and relationships."""

    def __init__(
        self,
        executor: MemgraphIngestor,
        embedding_provider: Any | None = None,
        min_similarity: float = 0.5,
    ) -> None:
        self.executor = executor
        self.embedding_provider = embedding_provider
        self.min_similarity = min_similarity

    def _fetch_records(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute Cypher and return records."""
        return self.executor.fetch_all(cypher, params or {})

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        entity_category: str | None = None,
    ) -> list[JsonEntityResult]:
        """Execute semantic search on JSON entities.

        Uses vector similarity + graph signals (PageRank) for ranking.
        Falls back to keyword search if embeddings or vector index
        are unavailable.

        Args:
            query: Natural language query.
            top_k: Maximum results.
            entity_category: Optional filter by MECE category.

        Returns:
            List of JsonEntityResult ranked by combined score.
        """
        if self.embedding_provider is None:
            logger.debug("Semantic search unavailable: no embedding provider")
            return self.keyword_search(query, top_k, entity_category)

        try:
            query_embedding = self.embedding_provider.embed(query)
        except Exception as exc:
            logger.warning(f"Failed to generate query embedding: {exc}")
            return self.keyword_search(query, top_k, entity_category)

        try:
            category_filter = ""
            params: dict[str, Any] = {
                "embedding": query_embedding,
                "top_k": top_k * 3,
                "limit": top_k,
            }

            if entity_category:
                category_filter = "AND n.entity_category = $category"
                params["category"] = entity_category

            cypher = f"""
            CALL vector_search.search($index_name, $top_k, $embedding)
            YIELD node AS n, similarity AS sim
            WHERE sim >= $min_similarity {category_filter}
            RETURN id(n) AS node_id,
                   n.unique_id AS unique_id,
                   n.name AS name,
                   n.type AS entity_type,
                   n.entity_category AS entity_category,
                   n.entity_subtype AS entity_subtype,
                   n.entity_emoji AS entity_emoji,
                   n.emoji AS emoji,
                   n.description AS description,
                   n.analogy AS analogy,
                   n.example AS example,
                   COALESCE(n.pagerank_score, 0.1) AS pagerank_score,
                   COALESCE(n.community_id, -1) AS community_id,
                   sim AS similarity
            ORDER BY (sim * 0.7) + (COALESCE(n.pagerank_score, 0.1) * 0.3) DESC
            LIMIT $limit
            """
            params["index_name"] = settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME
            params["min_similarity"] = self.min_similarity

            records = self._fetch_records(cypher, params)

            results: list[JsonEntityResult] = []
            for record in records:
                vector_score = record.get("similarity", 0.0)
                pagerank_score = record.get("pagerank_score", 0.1)
                combined_score = vector_score * 0.7 + min(pagerank_score, 1.0) * 0.3

                results.append(
                    JsonEntityResult(
                        unique_id=record.get("unique_id", ""),
                        name=record.get("name", ""),
                        entity_type=record.get("entity_type", "Entity"),
                        entity_category=record.get("entity_category", "ABSTRACT_CONCEPT"),
                        entity_subtype=record.get("entity_subtype", ""),
                        entity_emoji=record.get("entity_emoji", "💡"),
                        emoji=record.get("emoji", ""),
                        description=record.get("description", ""),
                        analogy=record.get("analogy", ""),
                        example=record.get("example", ""),
                        pagerank_score=pagerank_score,
                        combined_score=combined_score,
                        vector_score=vector_score,
                        text_score=0.0,
                        community_id=record.get("community_id", -1),
                    )
                )

            if not results:
                return self.keyword_search(query, top_k, entity_category)

            return results

        except Exception as exc:
            message = str(exc).lower()
            if "vector_search" in message or "vector index" in message:
                logger.debug(f"Vector search unavailable for JSON graph: {exc}")
            else:
                logger.warning(f"Semantic search failed: {exc}")
            return self.keyword_search(query, top_k, entity_category)

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        entity_category: str | None = None,
    ) -> list[JsonEntityResult]:
        """Execute keyword search on entity names and descriptions.

        Args:
            query: Search query.
            top_k: Maximum results.
            entity_category: Optional filter by MECE entity category.

        Returns:
            List of JsonEntityResult.
        """
        keywords = [w.lower() for w in query.split() if len(w) > 1][:5]

        if not keywords:
            return []

        category_filter = ""
        params: dict[str, Any] = {"keywords": keywords, "limit": top_k}

        if entity_category:
            category_filter = "AND n.entity_category = $category"
            params["category"] = entity_category

        cypher = f"""
        MATCH (n:JsonEntity)
        WHERE ANY(kw IN $keywords WHERE
            toLower(n.name) CONTAINS kw
            OR toLower(n.unique_id) CONTAINS kw
            OR toLower(COALESCE(n.description, '')) CONTAINS kw
            OR toLower(COALESCE(n.analogy, '')) CONTAINS kw)
        {category_filter}
        RETURN n.unique_id AS unique_id,
               n.name AS name,
               n.type AS entity_type,
               n.entity_category AS entity_category,
               n.entity_subtype AS entity_subtype,
               n.entity_emoji AS entity_emoji,
               n.emoji AS emoji,
               n.description AS description,
               n.analogy AS analogy,
               n.example AS example,
               COALESCE(n.pagerank_score, 0.1) AS pagerank_score
        LIMIT $limit
        """

        records = self._fetch_records(cypher, params)

        return [
            JsonEntityResult(
                unique_id=r.get("unique_id", ""),
                name=r.get("name", ""),
                entity_type=r.get("entity_type", "Entity"),
                entity_category=r.get("entity_category", "ABSTRACT_CONCEPT"),
                entity_subtype=r.get("entity_subtype", ""),
                entity_emoji=r.get("entity_emoji", "💡"),
                emoji=r.get("emoji", ""),
                description=r.get("description", ""),
                analogy=r.get("analogy", ""),
                example=r.get("example", ""),
                pagerank_score=r.get("pagerank_score", 0.1),
                combined_score=0.5,
                vector_score=0.0,
                text_score=1.0,
            )
            for r in records
        ]

    def find_related_entities(
        self,
        entity_identifier: str,
        relationship_types: list[str] | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        max_depth: int = 2,
        top_k: int = 20,
    ) -> list[JsonEntityResult]:
        """Find entities related to a given entity via relationships.

        Args:
            entity_identifier: Name or unique_id of the source entity.
            relationship_types: Optional filter by relationship categories.
            direction: Direction of traversal.
            max_depth: Maximum traversal depth.
            top_k: Maximum results.

        Returns:
            List of related JsonEntityResult with relationship context.
        """
        rel_types = relationship_types or [
            "HIERARCHICAL",
            "COMPOSITIONAL",
            "CONTEXTUAL",
            "ATTRIBUTIVE",
            "COMPARATIVE",
            "SEQUENTIAL",
            "CAUSAL",
            "ANALOGICAL",
        ]
        rel_pattern = "|".join(rel_types)

        return_fields = """
            related.unique_id AS unique_id,
            related.name AS name,
            related.type AS entity_type,
            related.entity_category AS entity_category,
            related.entity_subtype AS entity_subtype,
            related.entity_emoji AS entity_emoji,
            related.emoji AS emoji,
            related.description AS description,
            related.analogy AS analogy,
            related.example AS example,
            COALESCE(related.pagerank_score, 0.1) AS pagerank_score,
            length(path) AS depth
        """

        where_clause = """
            start.name = $identifier OR start.unique_id = $identifier
            OR start.name STARTS WITH $identifier OR start.unique_id STARTS WITH $identifier
        """

        if direction == "both":
            pattern = f"-[:{rel_pattern}*1..{max_depth}]-"
            cypher = f"""
            MATCH (start:JsonEntity)
            WHERE {where_clause}
            MATCH path = (start){pattern}(related:JsonEntity)
            RETURN DISTINCT {return_fields}
            ORDER BY depth ASC, pagerank_score DESC
            LIMIT $limit
            """
        elif direction == "outgoing":
            cypher = f"""
            MATCH (start:JsonEntity)
            WHERE {where_clause}
            MATCH path = (start)-[:{rel_pattern}*1..{max_depth}]->(related:JsonEntity)
            RETURN DISTINCT {return_fields}
            UNION
            MATCH (start:JsonEntity)
            WHERE {where_clause}
            MATCH path = (start)<-[:{rel_pattern}*1..{max_depth}]-(related:JsonEntity)
            WHERE ALL(r IN relationships(path) WHERE r.is_symmetric = true)
            RETURN DISTINCT {return_fields}
            ORDER BY depth ASC, pagerank_score DESC
            LIMIT $limit
            """
        else:  # incoming
            cypher = f"""
            MATCH (start:JsonEntity)
            WHERE {where_clause}
            MATCH path = (start)<-[:{rel_pattern}*1..{max_depth}]-(related:JsonEntity)
            RETURN DISTINCT {return_fields}
            UNION
            MATCH (start:JsonEntity)
            WHERE {where_clause}
            MATCH path = (start)-[:{rel_pattern}*1..{max_depth}]->(related:JsonEntity)
            WHERE ALL(r IN relationships(path) WHERE r.is_symmetric = true)
            RETURN DISTINCT {return_fields}
            ORDER BY depth ASC, pagerank_score DESC
            LIMIT $limit
            """

        records = self._fetch_records(
            cypher, {"identifier": entity_identifier, "limit": top_k}
        )

        return [
            JsonEntityResult(
                unique_id=r.get("unique_id", ""),
                name=r.get("name", ""),
                entity_type=r.get("entity_type", "Entity"),
                entity_category=r.get("entity_category", "ABSTRACT_CONCEPT"),
                entity_subtype=r.get("entity_subtype", ""),
                entity_emoji=r.get("entity_emoji", "💡"),
                emoji=r.get("emoji", ""),
                description=r.get("description", ""),
                analogy=r.get("analogy", ""),
                example=r.get("example", ""),
                pagerank_score=r.get("pagerank_score", 0.1),
                combined_score=1.0 / (r.get("depth", 1) + 1),
            )
            for r in records
        ]

    def find_relationships(
        self,
        entity_identifier: str,
        relationship_category: str | None = None,
        verb: str | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "both",
    ) -> list[JsonRelationshipResult]:
        """Find relationships involving a specific entity.

        Args:
            entity_identifier: Entity name or unique_id.
            relationship_category: Optional filter by category (CAUSAL, etc.).
            verb: Optional filter by verb (influences, enables, etc.).
            direction: Direction of relationships to find.

        Returns:
            List of JsonRelationshipResult.
        """
        category_filter = ""
        verb_filter = ""
        params: dict[str, Any] = {"identifier": entity_identifier}

        if relationship_category:
            # Support both new 'category' and legacy 'relationship_category' properties
            category_filter = "AND COALESCE(r.category, r.relationship_category, type(r)) = $category"
            params["category"] = relationship_category

        if verb:
            verb_filter = "AND r.verb = $verb"
            params["verb"] = verb

        base_where = """(start.name = $identifier OR start.unique_id = $identifier
           OR start.name STARTS WITH $identifier OR start.unique_id STARTS WITH $identifier)"""
        return_clause = """RETURN type(r) AS relationship_type,
               COALESCE(r.category, r.relationship_category, "RELATED_TO") AS category,
               COALESCE(r.emoji, r.relationship_emoji, "🔗") AS emoji,
               r.verb AS verb,
               r.strength AS strength"""

        if direction == "outgoing":
            cypher = f"""
            MATCH (start:JsonEntity)-[r]->(target:JsonEntity)
            WHERE {base_where}
            {category_filter} {verb_filter}
            {return_clause},
                   start.name AS from_name,
                   start.type AS from_type,
                   target.name AS to_name,
                   target.type AS to_type
            UNION
            MATCH (start:JsonEntity)<-[r]-(target:JsonEntity)
            WHERE {base_where}
            AND r.is_symmetric = true
            {category_filter} {verb_filter}
            {return_clause},
                   start.name AS from_name,
                   start.type AS from_type,
                   target.name AS to_name,
                   target.type AS to_type
            """
        elif direction == "incoming":
            cypher = f"""
            MATCH (source:JsonEntity)-[r]->(start:JsonEntity)
            WHERE {base_where}
            {category_filter} {verb_filter}
            {return_clause},
                   source.name AS from_name,
                   source.type AS from_type,
                   start.name AS to_name,
                   start.type AS to_type
            UNION
            MATCH (source:JsonEntity)<-[r]-(start:JsonEntity)
            WHERE {base_where}
            AND r.is_symmetric = true
            {category_filter} {verb_filter}
            {return_clause},
                   source.name AS from_name,
                   source.type AS from_type,
                   start.name AS to_name,
                   start.type AS to_type
            """
        else:
            cypher = f"""
            MATCH (start:JsonEntity)-[r]-(other:JsonEntity)
            WHERE {base_where}
            {category_filter} {verb_filter}
            {return_clause},
                   start.name AS from_name,
                   start.type AS from_type,
                   other.name AS to_name,
                   other.type AS to_type
            """

        records = self._fetch_records(cypher, params)

        return [
            JsonRelationshipResult(
                from_entity=r.get("from_name", ""),
                to_entity=r.get("to_name", ""),
                relationship_type=r.get("relationship_type", "RELATED_TO"),
                category=r.get("category", "RELATED_TO"),
                emoji=r.get("emoji", "🔗"),
                verb=r.get("verb", "related-to"),
                strength=r.get("strength", 0.5),
                from_type=r.get("from_type", ""),
                to_type=r.get("to_type", ""),
            )
            for r in records
        ]

    def get_entities_by_category(
        self,
        entity_category: str,
        top_k: int = 50,
    ) -> list[JsonEntityResult]:
        """Get all entities of a specific MECE category.

        Args:
            entity_category: One of the 7 MECE categories.
            top_k: Maximum results.

        Returns:
            List of JsonEntityResult.
        """
        cypher = """
        MATCH (n:JsonEntity {entity_category: $category})
        RETURN n.unique_id AS unique_id,
               n.name AS name,
               n.type AS entity_type,
               n.entity_category AS entity_category,
               n.entity_subtype AS entity_subtype,
               n.entity_emoji AS entity_emoji,
               n.emoji AS emoji,
               n.description AS description,
               n.analogy AS analogy,
               n.example AS example,
               COALESCE(n.pagerank_score, 0.1) AS pagerank_score
        ORDER BY n.pagerank_score DESC
        LIMIT $limit
        """

        records = self._fetch_records(
            cypher, {"category": entity_category, "limit": top_k}
        )

        return [
            JsonEntityResult(
                unique_id=r.get("unique_id", ""),
                name=r.get("name", ""),
                entity_type=r.get("entity_type", "Entity"),
                entity_category=r.get("entity_category", "ABSTRACT_CONCEPT"),
                entity_subtype=r.get("entity_subtype", ""),
                entity_emoji=r.get("entity_emoji", "💡"),
                emoji=r.get("emoji", ""),
                description=r.get("description", ""),
                analogy=r.get("analogy", ""),
                example=r.get("example", ""),
                pagerank_score=r.get("pagerank_score", 0.1),
                combined_score=r.get("pagerank_score", 0.1),
            )
            for r in records
        ]

    def get_important_entities(
        self,
        top_k: int = 20,
        entity_category: str | None = None,
    ) -> list[JsonEntityResult]:
        """Get most important entities by PageRank score.

        Args:
            top_k: Maximum results.
            entity_category: Optional filter by category.

        Returns:
            List of JsonEntityResult ordered by PageRank.
        """
        category_filter = ""
        params: dict[str, Any] = {"limit": top_k}

        if entity_category:
            category_filter = "WHERE n.entity_category = $category"
            params["category"] = entity_category

        cypher = f"""
        MATCH (n:JsonEntity)
        {category_filter}
        RETURN n.unique_id AS unique_id,
               n.name AS name,
               n.type AS entity_type,
               n.entity_category AS entity_category,
               n.entity_subtype AS entity_subtype,
               n.entity_emoji AS entity_emoji,
               n.emoji AS emoji,
               n.description AS description,
               n.pagerank_score AS pagerank_score
        ORDER BY n.pagerank_score DESC
        LIMIT $limit
        """

        records = self._fetch_records(cypher, params)

        return [
            JsonEntityResult(
                unique_id=r.get("unique_id", ""),
                name=r.get("name", ""),
                entity_type=r.get("entity_type", "Entity"),
                entity_category=r.get("entity_category", "ABSTRACT_CONCEPT"),
                entity_subtype=r.get("entity_subtype", ""),
                entity_emoji=r.get("entity_emoji", "💡"),
                emoji=r.get("emoji", ""),
                description=r.get("description", ""),
                analogy="",
                example="",
                pagerank_score=r.get("pagerank_score", 0.0),
                combined_score=r.get("pagerank_score", 0.0),
            )
            for r in records
        ]

    def get_entity_taxonomy(
        self,
        root_entity: str | None = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Get hierarchical taxonomy starting from an entity.

        Args:
            root_entity: Optional root entity identifier. If None, returns full taxonomy.
            max_depth: Maximum depth for hierarchy.

        Returns:
            Nested dictionary representing the taxonomy.
        """
        if root_entity:
            cypher = f"""
            MATCH path = (root:JsonEntity)-[*1..{max_depth}]->(descendant:JsonEntity)
            WHERE root.name = $identifier OR root.unique_id = $identifier
               OR root.name STARTS WITH $identifier OR root.unique_id STARTS WITH $identifier
              AND ALL(r IN relationships(path) WHERE r.category IN ['HIERARCHICAL', 'COMPOSITIONAL'])
            RETURN root.name AS root_name,
                   descendant.name AS descendant_name,
                   descendant.unique_id AS descendant_id,
                   descendant.type AS descendant_type,
                   COALESCE(relationships(path)[-1].category, type(relationships(path)[-1])) AS relationship_type,
                   length(path) AS depth
            ORDER BY depth
            """
            records = self._fetch_records(cypher, {"identifier": root_entity})
        else:
            cypher = f"""
            MATCH path = (root:JsonEntity)-[*1..{max_depth}]->(descendant:JsonEntity)
            WHERE ALL(r IN relationships(path) WHERE r.category IN ['HIERARCHICAL', 'COMPOSITIONAL'])
              AND NOT EXISTS {{
                MATCH (parent:JsonEntity)-[pr]->(root)
                WHERE pr.category IN ['HIERARCHICAL', 'COMPOSITIONAL']
              }}
            RETURN root.name AS root_name,
                   descendant.name AS descendant_name,
                   descendant.unique_id AS descendant_id,
                   descendant.type AS descendant_type,
                   COALESCE(relationships(path)[-1].category, type(relationships(path)[-1])) AS relationship_type,
                   length(path) AS depth
            ORDER BY root_name, depth
            """
            records = self._fetch_records(cypher)

        taxonomy: dict[str, Any] = {}
        for r in records:
            root = r.get("root_name", "")
            if root not in taxonomy:
                taxonomy[root] = {"children": [], "type": "root"}

            taxonomy[root]["children"].append({
                "name": r.get("descendant_name", ""),
                "unique_id": r.get("descendant_id", ""),
                "type": r.get("descendant_type", "Entity"),
                "relationship": r.get("relationship_type", ""),
                "depth": r.get("depth", 1),
            })

        return taxonomy
