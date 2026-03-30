"""Document → Code validation.

Validates DOCUMENTS against actual CODE.
Code is SOURCE OF TRUTH.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..query_router import ValidationResult, ValidationReport
from .validator import BaseValidator

if TYPE_CHECKING:
    from ...services.graph_service import MemgraphIngestor


class DocVsCodeValidator(BaseValidator):
    """
    Validate DOCUMENT against actual CODE.

    Code is SOURCE OF TRUTH.

    Use cases:
    - API documentation accuracy
    - Tutorial correctness
    - Finding outdated references
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
        Validate document accuracy against code.

        Args:
            query: Natural language query (used to find relevant docs)
            document_path: Specific document to validate (optional)
            scope: Validation scope ("all", "sections", "claims")

        Returns:
            ValidationReport with validation results
        """
        results: list[ValidationResult] = []

        # Extract claims from document(s)
        if document_path:
            claims = self._extract_claims(document_path)
        elif query:
            doc_paths = self._find_relevant_docs(query)
            claims = []
            for p in doc_paths:
                claims.extend(self._extract_claims(p))
        else:
            raise ValueError("Either query or document_path required")

        # Verify each claim against code
        for claim in claims:
            code_confirms = self._verify_claim(claim)
            results.append(
                ValidationResult(
                    element=claim.get("description", str(claim)),
                    status="ACCURATE" if code_confirms else "OUTDATED",
                    direction="DOC_VS_CODE",
                    suggestion=self._suggest_fix(claim) if not code_confirms else None,
                )
            )

        passed = sum(1 for r in results if r.status == "ACCURATE")
        failed = len(results) - passed

        return ValidationReport(
            total=len(results),
            passed=passed,
            failed=failed,
            direction="DOC_VS_CODE",
            results=results,
        )

    def _extract_claims(self, document_path: str) -> list[dict]:
        """
        Extract factual claims from document.

        Uses LLM for this (requires semantic understanding).
        Only called during validation, not indexing.
        """
        # TODO: Implement actual claim extraction
        # This would use LLM to parse the document
        return []

    def _find_relevant_docs(self, query: str) -> list[str]:
        """Find documents relevant to the query."""
        # TODO: Use semantic search on document graph
        return []

    def _verify_claim(self, claim: dict) -> bool:
        """
        Verify claim against code graph.

        This is deterministic (no LLM needed).
        """
        # TODO: Query code graph to verify claim
        return False

    def _suggest_fix(self, claim: dict) -> str:
        """
        Suggest fix for outdated claim.

        Uses LLM to generate suggestion based on actual code.
        """
        # TODO: Implement suggestion generation
        return "Update the documentation to reflect current implementation."

    def generate_summary(self, report: ValidationReport) -> str:
        """Generate human-readable summary."""
        if report.total == 0:
            return "No claims found to validate."

        status_emoji = "✅" if report.failed == 0 else "⚠️"

        summary = f"""{status_emoji} **Document vs Code Validation Results**

- **Total claims:** {report.total}
- **Accurate:** {report.passed}
- **Outdated:** {report.failed}
- **Accuracy:** {report.accuracy_score:.1%}

"""

        if report.failed > 0:
            summary += "\n**Outdated documentation:**\n"
            for r in report.results:
                if r.status == "OUTDATED":
                    summary += f"- {r.element}\n"
                    if r.suggestion:
                        summary += f"  → {r.suggestion}\n"

        return summary


__all__ = ["DocVsCodeValidator"]