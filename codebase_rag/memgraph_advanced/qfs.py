"""Query-Focused Summarization using community detection."""

from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..config import settings
from ..providers import get_provider_from_config
from ..services.graph_service import MemgraphIngestor


@dataclass
class CommunitySummary:
    """Summary of a detected community."""

    community_id: int
    node_count: int
    representative_nodes: list[str]
    summary_text: str
    key_functions: list[str]
    key_classes: list[str]


class CommunityQFS:
    """Query-Focused Summarization using community detection."""

    @staticmethod
    def _is_missing_procedure_error(error: Exception) -> bool:
        message = str(error).lower()
        return "there is no procedure named" in message or (
            "procedure" in message and "not found" in message
        )

    def __init__(self):
        self.provider = get_provider_from_config(settings.active_orchestrator_config)
        self.llm = self.provider.create_model(
            settings.active_orchestrator_config.model_id
        )

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
        CALL graph_algorithms.leiden(
            "CALLS",
            "OUTGOING",
            { community_property: "community_id", weight_property: "weight" }
        ) YIELD node, community_id

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

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        ) as ingestor:
            try:
                records = ingestor.fetch_all(cypher, params)
            except Exception as exc:
                if self._is_missing_procedure_error(exc):
                    logger.info(
                        "Skipping community summaries (community detection is not supported in your Memgraph edition)"
                    )
                    return []
                raise

        summaries = []
        for record in records:
            # Generate summary text using LLM
            summary_text = self._generate_community_summary(record)

            summaries.append(
                CommunitySummary(
                    community_id=record["community_id"],
                    node_count=record["node_count"],
                    representative_nodes=record["rep_names"],
                    summary_text=summary_text,
                    key_functions=record["key_functions"],
                    key_classes=record["key_classes"],
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

        response = self.llm.complete(prompt)
        return response.strip()

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
        """Rank communities by relevance to the user question using LLM."""
        # For simplicity, we'll do keyword matching for now, can be enhanced with embeddings
        keywords = question.lower().split()
        scored = []

        for comm in communities:
            score = 0
            summary_text = comm.summary_text.lower()
            for kw in keywords:
                if kw in summary_text:
                    score += 1
                if kw in [fn.lower() for fn in comm.key_functions]:
                    score += 2
                if kw in [c.lower() for c in comm.key_classes]:
                    score += 2
            if score > 0:
                scored.append((-score, comm))

        # Sort by score descending
        scored.sort()
        if not scored:
            logger.debug(
                "No communities matched query keywords, returning all communities in original order"
            )
        return [comm for (score, comm) in scored] + [
            comm for comm in communities if comm not in [c for (s, c) in scored]
        ]

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

        response = self.llm.complete(prompt)
        return response.strip()
