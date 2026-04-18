from __future__ import annotations

from loguru import logger
from pydantic_ai import Tool

from .. import constants as cs
from .. import exceptions as ex
from .. import logs as ls
from ..cypher_queries import CYPHER_GET_FUNCTION_SOURCE_LOCATION
from ..types_defs import SemanticSearchResult
from ..utils.dependencies import has_semantic_dependencies
from . import tool_descriptions as td


def semantic_code_search(query: str, top_k: int = 5) -> list[SemanticSearchResult]:
    if not has_semantic_dependencies():
        logger.warning(ex.SEMANTIC_EXTRA)
        return []

    try:
        from ..config import settings
        from ..embeddings import get_embedding_provider
        from ..memgraph_advanced import HybridRetriever
        from ..services.graph_service import MemgraphIngestor
        from ..vector_backend import get_shared_backend

        effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K

        config = settings.active_embedding_config
        provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
        )

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=cs.SEMANTIC_BATCH_SIZE,
        ) as ingestor:
            retriever = HybridRetriever(
                graph_ingestor=ingestor,
                vector_backend=get_shared_backend(),
                embedding_provider=provider,
                config=settings.hybrid_retrieval_config,
            )

            hybrid_results = retriever.search(query, top_k=effective_top_k)

            formatted_results: list[SemanticSearchResult] = []
            for result in hybrid_results:
                formatted_results.append(
                    SemanticSearchResult(
                        node_id=result.node_id,
                        qualified_name=result.qualified_name,
                        name=result.name,
                        type=result.node_type,
                        similarity=round(result.combined_score, 3),
                    )
                )

            logger.info(
                ls.SEMANTIC_FOUND.format(count=len(formatted_results), query=query)
            )
            return formatted_results

    except Exception as e:
        logger.error(ls.SEMANTIC_FAILED.format(query=query, error=e))
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
