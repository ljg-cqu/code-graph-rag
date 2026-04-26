"""JSON Graph query tool for interactive agent."""
from __future__ import annotations

from loguru import logger

from codebase_rag.compat.pydantic_ai import Tool

from .. import constants as cs
from ..config import settings
from ..json_ingestion import _create_json_ingestor
from ..json_queries import JsonGraphQueryEngine
from . import tool_descriptions as td


def create_query_json_graph_tool() -> Tool:
    """Create query_json_graph tool for Pydantic AI agent.

    Usage: Query the JSON knowledge graph for entities and relationships.
    Example: "What competencies does a CTO need?"
    """

    async def query_json_graph(
        natural_language_query: str,
        top_k: int = 10,
        entity_category: str | None = None,
    ) -> str:
        """Query JSON knowledge graph for entities and relationships.

        Args:
            natural_language_query: Question about JSON-ingested domain concepts.
            top_k: Number of results to return.
            entity_category: Optional MECE category filter
                (CONCRETE_ENTITY, EVENT_PROCESS, INFORMATION_EXPRESSION,
                PROPERTY_ATTRIBUTE, SYSTEM_STRUCTURE, AGENT_ROLE, ABSTRACT_CONCEPT).

        Returns:
            Formatted results from JSON graph query.
        """
        logger.info(f"Querying JSON graph: {natural_language_query[:50]}...")

        try:
            from ..embedder import get_embedding_provider_instance

            embedding_provider = get_embedding_provider_instance()
        except Exception as exc:
            logger.debug(f"Embedding provider unavailable for JSON query: {exc}")
            embedding_provider = None

        try:
            with _create_json_ingestor(
                settings.JSON_MEMGRAPH_BATCH_SIZE
            ) as executor:
                engine = JsonGraphQueryEngine(
                    executor=executor,
                    embedding_provider=embedding_provider,
                )

                results = engine.semantic_search(
                    natural_language_query,
                    top_k=top_k,
                    entity_category=entity_category,
                )

                if not results:
                    return cs.MSG_SEMANTIC_NO_RESULTS.format(
                        query=natural_language_query
                    )

                lines = [
                    f"**JSON Graph Results ({len(results)} entities):**\n"
                ]

                for result in results[:top_k]:
                    category_emoji = cs.ENTITY_CATEGORY_EMOJI_MAP.get(
                        result.entity_category, "💡"
                    )
                    display_emoji = f"{result.emoji} " if result.emoji else ""
                    score_note = ""
                    if result.vector_score > 0:
                        score_note = f" (similarity: {result.combined_score:.2f})"

                    lines.append(
                        f"- **{display_emoji}{result.name}** "
                        f"[{result.entity_category} {category_emoji}]{score_note}\n"
                        f"  {result.description[:120]}"
                        f"{'...' if len(result.description) > 120 else ''}"
                    )

                return "\n".join(lines)

        except Exception as exc:
            logger.error(f"JSON graph query failed: {exc}")
            return f"JSON graph query failed: {exc}"

    return Tool(
        query_json_graph,
        name=td.AgenticToolName.QUERY_JSON_GRAPH,
        description=td.QUERY_JSON_GRAPH,
    )
