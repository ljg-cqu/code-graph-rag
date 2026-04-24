from __future__ import annotations

import hashlib
import time
from enum import StrEnum

from loguru import logger
from pydantic import BaseModel, Field

from codebase_rag.config import settings


class QueryMethod(StrEnum):
    SEMANTIC_SEARCH = "semantic_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    KEYWORD_SEARCH = "keyword_search"
    VECTOR_DIRECT = "vector_direct"
    GRAPH_ALGORITHMS = "graph_algorithms"
    GRAPH_NAVIGATION = "graph_navigation"


class GraphAlgorithm(StrEnum):
    """Specific graph algorithm to execute when method is GRAPH_ALGORITHMS."""

    SHORTEST_PATH = "shortest_path"
    SIMILARITY = "similarity"
    COMMUNITY = "community"
    BFS = "bfs"
    NONE = "none"


class QueryIntent(StrEnum):
    """Unified intent enum for both code and document queries."""

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


class QueryPlan(BaseModel):
    """Structured query plan produced by the LLM."""

    methods: list[QueryMethod] = Field(
        ...,
        description="Ordered list of methods to execute (primary first, max 3).",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation for method selection.",
    )
    fallback_methods: list[QueryMethod] = Field(
        default_factory=list,
        description="Methods to try if primary methods fail or return empty.",
    )
    expected_entities: list[str] = Field(
        default_factory=list,
        description="Key entities (functions, classes, concepts) identified from the query.",
    )
    requires_file_read: bool = Field(
        default=False,
        description="Whether the query likely requires reading source files.",
    )
    intent: QueryIntent | None = Field(
        default=None,
        description="Classified intent type for routing decisions.",
    )
    algorithm: GraphAlgorithm = Field(
        default=GraphAlgorithm.NONE,
        description="Specific graph algorithm when method is graph_algorithms.",
    )


# TTL cache for validated plans (query hash -> (plan, timestamp))
_PLAN_CACHE: dict[str, tuple[QueryPlan, float]] = {}
_PLAN_CACHE_TTL = 3600  # 1 hour TTL
_PLAN_CACHE_MAX_SIZE = 1000


class LLMQueryPlanner:
    """Uses LLM to plan query execution instead of keyword-based intent classification.

    Implements lazy initialization and caching for efficiency.
    """

    __slots__ = ("agent", "_model_id")

    SYSTEM_PROMPT = """You are a query planner for a codebase analysis system. Given a user's natural language query and the available query methods, select the best methods to execute.

Available methods:
- semantic_search: Find code entities by meaning (embeddings). Best for "what does X do?", "find code about Y".
- graph_traversal: Follow relationships in the graph (calls, imports, inherits). Best for "who calls X?", "what imports Y?".
- keyword_search: Fast text matching on names and qualified_names. Best for exact name lookups.
- vector_direct: Direct vector similarity without graph. Best for finding similar code snippets.
- graph_algorithms: Run algorithms (shortest path, similarity, community). Best for "path from A to B", "most central function".
- graph_navigation: Browse graph neighborhoods. Best for "show me around X", "context of Y".

Available graph algorithms (only when method is graph_algorithms):
- shortest_path: Find shortest path between two entities.
- similarity: Find structurally or semantically similar nodes.
- community: Detect communities or clusters.
- bfs: Breadth-first search neighborhood context.
- none: No specific algorithm.

Respond with a JSON object matching this structure:
{
  "methods": ["method_name", ...],
  "reasoning": "brief explanation",
  "fallback_methods": ["method_name", ...],
  "expected_entities": ["entity_name", ...],
  "requires_file_read": true/false,
  "intent": "intent_type",
  "algorithm": "algorithm_name"
}

Rules:
- Select 1-3 primary methods. Do not select more than 3.
- Use fallback_methods for methods to try if primary methods return empty.
- expected_entities should list specific functions, classes, or modules mentioned.
- intent should be one of: structural, functional, semantic, exploratory, validation, doc_semantic_search, doc_graph_traversal, doc_comparison, doc_procedural.
- algorithm should be one of: shortest_path, similarity, community, bfs, none. Only set when "graph_algorithms" is in methods.
"""

    def __init__(self) -> None:
        self.agent = None
        self._model_id: str | None = None

    def _initialize_agent(self) -> None:
        """Lazy initialization of the LLM agent."""
        if self.agent is not None:
            return

        try:
            config = settings.active_orchestrator_config
        except AttributeError:
            try:
                config = settings.active_cypher_config
            except AttributeError:
                from codebase_rag.config import ModelConfig

                config = ModelConfig(
                    provider=getattr(settings, "ACTIVE_ORCHESTRATOR_PROVIDER", "openai"),
                    model_id=getattr(settings, "ACTIVE_ORCHESTRATOR_MODEL", "gpt-4o-mini"),
                )

        from codebase_rag.services.llm import _create_chat_model

        llm = _create_chat_model(config)
        self._model_id = config.model_id

        from codebase_rag.compat.pydantic_ai import Agent

        self.agent = Agent(
            model=llm,
            system_prompt=self.SYSTEM_PROMPT,
            output_type=QueryPlan,
            retries=getattr(settings, "AGENT_RETRIES", 2),
        )

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _get_cached_plan(self, cache_key: str) -> QueryPlan | None:
        """Retrieve cached plan if still valid."""
        if cache_key in _PLAN_CACHE:
            plan, timestamp = _PLAN_CACHE[cache_key]
            if time.time() - timestamp < _PLAN_CACHE_TTL:
                return plan
            del _PLAN_CACHE[cache_key]
        return None

    def _cache_plan(self, cache_key: str, plan: QueryPlan) -> None:
        """Cache a successful plan with size limit."""
        if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX_SIZE:
            oldest_key = min(_PLAN_CACHE.keys(), key=lambda k: _PLAN_CACHE[k][1])
            del _PLAN_CACHE[oldest_key]

        _PLAN_CACHE[cache_key] = (plan, time.time())

    async def plan(self, query: str) -> QueryPlan:
        """Generate a query plan using LLM intelligence.

        Uses caching to reduce API calls for similar queries.

        Args:
            query: User's natural language query.

        Returns:
            QueryPlan with methods, reasoning, and entities.
        """
        self._initialize_agent()

        cache_key = self._get_cache_key(query)
        cached = self._get_cached_plan(cache_key)
        if cached is not None:
            logger.debug(f"Using cached query plan for: {query[:50]}...")
            return cached

        logger.info(f"Planning query execution for: {query[:50]}...")

        try:
            result = await self.agent.run(query)
            plan = result.output

            if not plan.methods:
                plan.methods = [QueryMethod.SEMANTIC_SEARCH]

            if QueryMethod.GRAPH_ALGORITHMS in plan.methods and plan.algorithm == GraphAlgorithm.NONE:
                plan.algorithm = GraphAlgorithm.SIMILARITY

            self._cache_plan(cache_key, plan)

            logger.info(
                f"Query plan: methods={[m.value for m in plan.methods]}, "
                f"intent={plan.intent}, entities={plan.expected_entities[:3]}"
            )
            return plan

        except Exception as e:
            logger.warning(f"LLM query planning failed: {e}. Using fallback plan.")
            return QueryPlan(
                methods=[QueryMethod.SEMANTIC_SEARCH],
                reasoning="Fallback plan due to LLM planning failure",
                fallback_methods=[QueryMethod.KEYWORD_SEARCH],
                expected_entities=[],
                requires_file_read=False,
                intent=None,
                algorithm=GraphAlgorithm.NONE,
            )

    def clear_cache(self) -> None:
        """Clear the plan cache."""
        _PLAN_CACHE.clear()
