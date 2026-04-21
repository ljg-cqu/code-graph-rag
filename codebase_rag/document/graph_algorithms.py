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
