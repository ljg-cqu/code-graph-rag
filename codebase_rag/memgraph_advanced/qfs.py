"""Query-Focused Summarization using community detection."""

import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..config import settings
from ..embeddings import get_embedding_provider
from ..providers import get_provider_from_config
from ..services import QueryProtocol
from ..services.graph_service import MemgraphIngestor
from ..utils.query_utils import extract_keywords


def _coerce_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


@dataclass
class CommunitySummary:
    """Summary of a detected community."""

    community_id: int
    node_count: int
    representative_nodes: list[str]
    summary_text: str
    key_functions: list[str]
    key_classes: list[str]
    embedding: list[float] | None = None


class CommunityQFS:
    """Query-Focused Summarization using community detection."""

    @staticmethod
    def _is_missing_procedure_error(error: Exception) -> bool:
        message = str(error).lower()
        return "there is no procedure named" in message or (
            "procedure" in message and "not found" in message
        )

    def __init__(self, ingestor: QueryProtocol | None = None):
        self._ingestor = ingestor
        self.provider = get_provider_from_config(settings.active_orchestrator_config)
        self.llm = self.provider.create_model(
            settings.active_orchestrator_config.model_id
        )

    @contextmanager
    def _with_ingestor(self) -> Generator[QueryProtocol, None, None]:
        if self._ingestor is not None:
            yield self._ingestor
        else:
            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
            ) as ingestor:
                yield ingestor

    def _complete_prompt(self, prompt: str) -> str:
        complete = getattr(self.llm, "complete", None)
        if not callable(complete):
            raise TypeError("Configured orchestrator model does not support complete()")
        response = complete(prompt)
        return response.strip() if isinstance(response, str) else str(response).strip()

    def build_community_summaries(self, min_size: int = 5) -> list[CommunitySummary]:
        """
        Build summaries for each detected community.

        Args:
            min_size: Minimum community size to include

        Returns:
            List of community summaries
        """
        cypher = """
        // First, ensure communities are detected
        CALL leiden_community_detection.get()
        YIELD node, community_id
        SET node.community_id = community_id

        // Collect nodes by community
        WITH community_id, collect(node) AS nodes
        WHERE size(nodes) >= $min_size

        // Get representative nodes (highest PageRank in each community)
        UNWIND nodes AS n
        WITH community_id, nodes, n
        ORDER BY COALESCE(n.pagerank_score, 0) DESC
        WITH community_id, nodes, collect(n)[0..5] AS representatives

        // Calculate community statistics
        WITH community_id,
             size(nodes) AS node_count,
             [r IN representatives | r.qualified_name] AS rep_names,
             [n IN nodes WHERE n:Function] AS functions,
             [n IN nodes WHERE n:Class] AS classes

        RETURN community_id, node_count, rep_names,
               size(functions) AS function_count,
               size(classes) AS class_count,
               [f IN functions | f.qualified_name][0..10] AS key_functions,
               [c IN classes | c.qualified_name][0..5] AS key_classes
        ORDER BY node_count DESC
        """

        params = {"min_size": min_size}

        with self._with_ingestor() as ingestor:
            try:
                records = ingestor.fetch_all(cypher, params)
            except Exception as exc:
                if self._is_missing_procedure_error(exc):
                    logger.info(
                        "Skipping community summaries (community detection is not supported in your Memgraph edition)"
                    )
                    return []
                raise

        config = settings.active_embedding_config
        embed_provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
        )

        summaries = []
        for record in records:
            # Generate summary text using LLM
            summary_text = self._generate_community_summary(record)
            key_functions = _coerce_str_list(record.get("key_functions"))
            key_classes = _coerce_str_list(record.get("key_classes"))

            # Pre-compute community embedding for query-time semantic ranking
            comm_text = f"{summary_text} {', '.join(key_functions[:3])} {', '.join(key_classes[:2])}"
            try:
                comm_embedding = embed_provider.embed(comm_text)
            except Exception:
                comm_embedding = None

            summaries.append(
                CommunitySummary(
                    community_id=_coerce_int(record.get("community_id"), 0),
                    node_count=_coerce_int(record.get("node_count"), 0),
                    representative_nodes=_coerce_str_list(record.get("rep_names")),
                    summary_text=summary_text,
                    key_functions=key_functions,
                    key_classes=key_classes,
                    embedding=comm_embedding,
                )
            )

        return summaries

    def _generate_community_summary(self, community_data: dict[str, Any]) -> str:
        """Generate natural language summary of a community using LLM."""
        prompt = f"""
        Generate a concise summary of this code community:

        Community ID: {community_data["community_id"]}
        Size: {community_data["node_count"]} nodes
        Key functions: {", ".join(community_data["key_functions"][:5])}
        Key classes: {", ".join(community_data["key_classes"][:3])}
        Representative nodes: {", ".join(community_data["rep_names"])}

        Summarize what this module/component likely does in 2-3 sentences.
        """

        return self._complete_prompt(prompt)

    def query_focused_summary(
        self, question: str, top_communities: int = 3, min_community_size: int = 5
    ) -> str:
        """
        Generate query-focused summary from community summaries.

        Args:
            question: User question
            top_communities: Number of top communities to include
            min_community_size: Minimum community size to consider

        Returns:
            Summarized answer
        """
        # First get all community summaries
        all_communities = self.build_community_summaries(min_size=min_community_size)
        if not all_communities:
            return "No code communities are available for summarization in the current graph."

        # Rank communities by relevance to question
        ranked_communities = self._rank_communities_by_relevance(
            question, all_communities
        )

        # Take top N communities
        top_comms = ranked_communities[:top_communities]

        # Generate final summary
        final_summary = self._generate_answer_from_communities(question, top_comms)

        return final_summary

    def _rank_communities_by_relevance(
        self, question: str, communities: list[CommunitySummary]
    ) -> list[CommunitySummary]:
        """Rank communities by semantic similarity + keyword overlap."""
        config = settings.active_embedding_config
        embed_provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
        )
        query_embedding = embed_provider.embed(question)

        keywords = extract_keywords(question, max_keywords=5)
        scored: list[tuple[float, CommunitySummary]] = []

        for comm in communities:
            # Semantic score: cosine similarity between query and pre-computed community embedding
            semantic_score = 0.0
            if comm.embedding is not None:
                semantic_score = _cosine_similarity(query_embedding, comm.embedding)

            # Keyword overlap score (secondary)
            keyword_score = 0
            summary_lower = comm.summary_text.lower()
            for kw in keywords:
                if kw in summary_lower:
                    keyword_score += 1
                if kw in [fn.lower() for fn in comm.key_functions]:
                    keyword_score += 2
                if kw in [c.lower() for c in comm.key_classes]:
                    keyword_score += 2

            combined_score = semantic_score * 0.7 + (keyword_score / max(len(keywords), 1)) * 0.3
            scored.append((combined_score, comm))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [comm for _, comm in scored]

    def _generate_answer_from_communities(
        self, question: str, communities: list[CommunitySummary]
    ) -> str:
        """Generate final answer from top relevant communities."""
        comm_info = "\n\n".join(
            [
                f"Community {comm.community_id}:\n{comm.summary_text}\nKey functions: {', '.join(comm.key_functions[:3])}"
                for comm in communities
            ]
        )

        prompt = f"""
        Answer the user's question using the following information about relevant code communities:

        {comm_info}

        User question: {question}

        Provide a clear, concise answer based only on the information above. If you don't have enough information, say so.
        """

        return self._complete_prompt(prompt)
