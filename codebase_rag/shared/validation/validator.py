"""Base validation engine for bidirectional code↔document validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..query_router import ValidationReport

if TYPE_CHECKING:
    from ...services.graph_service import MemgraphIngestor


class BaseValidator(ABC):
    """
    Abstract base for validation engines.

    Provides common infrastructure for bidirectional validation.
    """

    def __init__(
        self,
        code_graph: MemgraphIngestor,
        doc_graph: MemgraphIngestor,
    ) -> None:
        self.code_graph = code_graph
        self.doc_graph = doc_graph

    @abstractmethod
    def validate(
        self,
        query: str | None = None,
        document_path: str | None = None,
        scope: str = "all",
    ) -> ValidationReport:
        """
        Run validation.

        Args:
            query: Natural language query (used to find relevant docs)
            document_path: Specific document to validate (optional)
            scope: Validation scope ("all", "sections", "claims")

        Returns:
            ValidationReport with results
        """
        pass

    @abstractmethod
    def generate_summary(self, report: ValidationReport) -> str:
        """
        Generate human-readable summary of validation results.

        Args:
            report: ValidationReport to summarize

        Returns:
            Human-readable summary string
        """
        pass

    def _execute_code_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a query on the code graph."""
        # Placeholder - actual implementation would use MemgraphIngestor
        return []

    def _execute_doc_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a query on the document graph."""
        # Placeholder - actual implementation would use MemgraphIngestor
        return []


__all__ = ["BaseValidator"]