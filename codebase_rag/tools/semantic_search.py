from __future__ import annotations

from loguru import logger
from pydantic_ai import Tool

from .. import constants as cs
from .. import logs as ls
from ..cypher_queries import CYPHER_GET_FUNCTION_SOURCE_LOCATION
from ..types_defs import SemanticSearchResult
from . import tool_descriptions as td


def _semantic_search_keyword_fallback(query: str, top_k: int) -> list[SemanticSearchResult]:
    """Fallback keyword-based search when semantic search unavailable.

    Uses CONTAINS queries on function/class names and docstrings.
    """
    from ..config import settings
    from ..services.graph_service import MemgraphIngestor

    try:
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        ) as ingestor:
            # Extract meaningful keywords (filter out stopwords)
            stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of'}
            keywords = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in stopwords]
            if not keywords:
                return []

            # Use most specific keyword (longest)
            keyword = max(keywords, key=len)

            cypher = """
            MATCH (n:Function|Class|Method)
            WHERE n.name CONTAINS $keyword
               OR n.qualified_name CONTAINS $keyword
               OR n.docstring CONTAINS $keyword
            RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
                   n.name AS name, labels(n)[0] AS node_type
            LIMIT $limit
            """
            results = ingestor.fetch_all(cypher, {"keyword": keyword, "limit": top_k})

            return [
                SemanticSearchResult(
                    node_id=r["node_id"],
                    qualified_name=r["qualified_name"],
                    name=r["name"],
                    type=r["node_type"],
                    similarity=0.5,  # Neutral score for keyword matches
                )
                for r in results
            ]
    except Exception as e:
        logger.error(f"Keyword fallback failed: {e}")
        return []


def _search_with_hybrid_retriever(query: str, top_k: int) -> list[SemanticSearchResult]:
    """Level 2: HybridRetriever search with vector + graph signals."""
    from ..config import settings
    from ..memgraph_advanced import create_hybrid_retriever
    from ..memgraph_advanced.hybrid_retrieval import HybridSearchResult
    from ..services.graph_service import MemgraphIngestor

    effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K

    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=cs.SEMANTIC_BATCH_SIZE,
    ) as ingestor:
        # Use factory function with shared dependencies
        retriever = create_hybrid_retriever(ingestor)

        hybrid_results: list[HybridSearchResult] = retriever.search(
            query, top_k=effective_top_k
        )

        return [
            SemanticSearchResult(
                node_id=result.node_id,
                qualified_name=result.qualified_name,
                name=result.name,
                type=result.node_type,
                similarity=round(result.combined_score, 3),
            )
            for result in hybrid_results
        ]


def _search_direct_vector(query: str, top_k: int) -> list[SemanticSearchResult]:
    """Level 3: Direct vector search without HybridRetriever."""
    from ..config import settings
    from ..embeddings import get_embedding_provider
    from ..services.graph_service import MemgraphIngestor
    from ..vector_backend import get_shared_backend

    config = settings.active_embedding_config
    provider = get_embedding_provider(
        provider=config.provider,
        model_id=config.model_id,
    )

    query_embedding = provider.embed(query)
    backend = get_shared_backend()

    # Direct vector search
    vector_results = backend.search(query_embedding, top_k=top_k)

    # Fetch metadata from graph
    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
    ) as ingestor:
        node_ids = [nid for nid, _ in vector_results]
        if not node_ids:
            return []

        cypher = """
        MATCH (n)
        WHERE id(n) IN $node_ids
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS node_type
        """
        metadata = ingestor.fetch_all(cypher, {"node_ids": node_ids})
        metadata_map = {r["node_id"]: r for r in metadata}

        return [
            SemanticSearchResult(
                node_id=nid,
                qualified_name=metadata_map.get(nid, {}).get("qualified_name", "?"),
                name=metadata_map.get(nid, {}).get("name", "?"),
                type=metadata_map.get(nid, {}).get("node_type", "?"),
                similarity=round(score, 3),
            )
            for nid, score in vector_results
        ]


def semantic_code_search(query: str, top_k: int = 5) -> list[SemanticSearchResult]:
    """Search codebase using semantic similarity with comprehensive fallback chain.

    Includes caching layer to avoid redundant embedding generation for
    identical queries.

    Fallback order:
    0. Cache check (fastest)
    1. HybridRetriever (vector + graph signals) - best quality
    2. Direct vector search - if HybridRetriever fails
    3. Keyword-based graph search - always available fallback
    4. Empty with warning - last resort
    """
    from ..utils.semantic_cache import get_semantic_cache

    # Level 0: Cache check
    cache = get_semantic_cache()
    cached = cache.get(query, top_k)
    if cached is not None:
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return cached

    # Level 1: HybridRetriever (best quality)
    try:
        results = _search_with_hybrid_retriever(query, top_k)
        if results:
            logger.info(ls.SEMANTIC_FOUND.format(count=len(results), query=query))
            cache.put(query, top_k, results)
            return results
    except Exception as e:
        logger.warning(f"HybridRetriever failed: {e}")

    # Level 2: Direct vector search (if embeddings exist)
    try:
        results = _search_direct_vector(query, top_k)
        if results:
            logger.info(f"Direct vector search found {len(results)} results for: {query}")
            cache.put(query, top_k, results)
            return results
    except Exception as e:
        logger.warning(f"Direct vector search failed: {e}")

    # Level 3: Keyword fallback (always available)
    try:
        results = _semantic_search_keyword_fallback(query, top_k)
        if results:
            logger.info(f"Keyword fallback found {len(results)} results for: {query}")
            cache.put(query, top_k, results)
            return results
    except Exception as e:
        logger.warning(f"Keyword fallback failed: {e}")

    # Level 4: Empty with clear message
    logger.warning(f"All search methods failed for query: {query}")
    return []


def get_function_source_code(node_id: int) -> str | None:
    try:
        from ..config import settings
        from ..services.graph_service import MemgraphIngestor
        from ..utils.source_extraction import (
            extract_source_lines,
            validate_source_location,
        )

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=cs.SEMANTIC_BATCH_SIZE,
        ) as ingestor:
            results = ingestor._execute_query(
                CYPHER_GET_FUNCTION_SOURCE_LOCATION, {"node_id": node_id}
            )

            if not results:
                logger.warning(ls.SEMANTIC_NODE_NOT_FOUND.format(id=node_id))
                return None

            result = results[0]
            file_path = result.get("path")
            start_line = result.get("start_line")
            end_line = result.get("end_line")

            is_valid, file_path_obj = validate_source_location(
                file_path, start_line, end_line
            )
            if not is_valid or file_path_obj is None:
                logger.warning(ls.SEMANTIC_INVALID_LOCATION.format(id=node_id))
                return None

            return extract_source_lines(file_path_obj, start_line, end_line)

    except Exception as e:
        logger.error(ls.SEMANTIC_SOURCE_FAILED.format(id=node_id, error=e))
        return None


def create_semantic_search_tool() -> Tool:
    async def semantic_search_functions(query: str, top_k: int = 5) -> str:
        logger.info(ls.SEMANTIC_TOOL_SEARCH.format(query=query))

        results = semantic_code_search(query, top_k)

        if not results:
            return cs.MSG_SEMANTIC_NO_RESULTS.format(query=query)

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result['qualified_name']} (type: {result['type']}, similarity: {result['similarity']})"
            )

        response = cs.MSG_SEMANTIC_RESULT_HEADER.format(count=len(results), query=query)
        response += "\n".join(formatted_results)
        response += cs.MSG_SEMANTIC_RESULT_FOOTER

        return response

    return Tool(
        semantic_search_functions,
        name=td.AgenticToolName.SEMANTIC_SEARCH,
        description=td.SEMANTIC_SEARCH,
    )


def create_get_function_source_tool() -> Tool:
    async def get_function_source_by_id(node_id: int) -> str:
        logger.info(ls.SEMANTIC_TOOL_SOURCE.format(id=node_id))

        source_code = get_function_source_code(node_id)

        if source_code is None:
            return cs.MSG_SEMANTIC_SOURCE_UNAVAILABLE.format(id=node_id)

        return cs.MSG_SEMANTIC_SOURCE_FORMAT.format(id=node_id, code=source_code)

    return Tool(
        get_function_source_by_id,
        name=td.AgenticToolName.GET_FUNCTION_SOURCE,
        description=td.GET_FUNCTION_SOURCE,
    )
