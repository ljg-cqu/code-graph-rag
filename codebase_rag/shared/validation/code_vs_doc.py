"""Code → Document validation.

Validates CODE against DOCUMENT specifications.
Document is SOURCE OF TRUTH.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..query_router import ValidationResult, ValidationReport
from .validator import BaseValidator

if TYPE_CHECKING:
    from ...services.graph_service import MemgraphIngestor


class CodeVsDocValidator(BaseValidator):
    """
    Validate CODE against DOCUMENT specifications.

    Document is SOURCE OF TRUTH.

    Use cases:
    - API spec compliance (OpenAPI, gRPC proto)
    - Regulatory requirements (HIPAA, GDPR)
    - Contract verification
    """

    def __init__(
        self,
        code_graph: MemgraphIngestor,
        doc_graph: MemgraphIngestor,
    ) -> None:
        super().__init__(code_graph, doc_graph)

    def validate(
        self,
        query: str | None = None,
        document_path: str | None = None,
        scope: str = "all",
    ) -> ValidationReport:
        """
        Validate code implementation against document spec.

        Args:
            query: Natural language query (used to find relevant docs)
            document_path: Specific document to validate (optional)
            scope: Validation scope ("all", "sections", "claims")

        Returns:
            ValidationReport with validation results
        """
        results: list[ValidationResult] = []

        # Extract requirements from document(s)
        if document_path:
            requirements = self._extract_requirements(document_path)
        elif query:
            doc_paths = self._find_relevant_docs(query)
            requirements = []
            for p in doc_paths:
                requirements.extend(self._extract_requirements(p))
        else:
            raise ValueError("Either query or document_path required")

        # Verify each requirement in code
        for req in requirements:
            implemented = self._verify_in_code(req)
            results.append(
                ValidationResult(
                    element=req.get("description", str(req)),
                    status="VALID" if implemented else "MISSING",
                    direction="CODE_VS_DOC",
                    suggestion=None if implemented else f"Implement {req.get('description', req)}",
                )
            )

        passed = sum(1 for r in results if r.status == "VALID")
        failed = len(results) - passed

        return ValidationReport(
            total=len(results),
            passed=passed,
            failed=failed,
            direction="CODE_VS_DOC",
            results=results,
        )

    def _extract_requirements(self, document_path: str) -> list[dict]:
        """
        Extract requirements from specification document.

        Uses LLM for this (requires semantic understanding).
        Only called during validation, not indexing.
        """
        # TODO: Implement actual requirement extraction
        # This would use LLM to parse the document
        return []

    def _find_relevant_docs(self, query: str) -> list[str]:
        """Find documents relevant to the query."""
        # TODO: Use semantic search on document graph
        return []

    def _verify_in_code(self, requirement: dict) -> bool:
        """
        Verify requirement exists in code graph.

        This is deterministic (no LLM needed).
        """
        # TODO: Query code graph to verify implementation
        return False

    def generate_summary(self, report: ValidationReport) -> str:
        """Generate human-readable summary."""
        if report.total == 0:
            return "No requirements found to validate."

        status_emoji = "✅" if report.failed == 0 else "⚠️"

        summary = f"""{status_emoji} **Code vs Document Validation Results**

- **Total requirements:** {report.total}
- **Passed:** {report.passed}
- **Failed:** {report.failed}
- **Accuracy:** {report.accuracy_score:.1%}

"""

        if report.failed > 0:
            summary += "\n**Missing implementations:**\n"
            for r in report.results:
                if r.status == "MISSING":
                    summary += f"- {r.element}\n"
                    if r.suggestion:
                        summary += f"  → {r.suggestion}\n"

        return summary


__all__ = ["CodeVsDocValidator"]