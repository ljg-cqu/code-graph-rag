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
        # Query document graph for the document content
        query = """
        MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)
        WHERE d.path = $path OR d.path CONTAINS $path
        OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
        RETURN s.title as section_title, s.content as section_content,
               collect(c.content) as chunk_contents,
               d.path as document_path
        """
        results = self._execute_doc_query(query, {"path": document_path})

        # Extract claims from document content
        # This is a simplified implementation - full implementation would use LLM
        claims: list[dict] = []
        for result in results:
            section_title = result.get("section_title", "")
            section_content = result.get("section_content", "")
            chunks = result.get("chunk_contents", [])

            # Combine content for analysis
            all_content = [section_content] + [c for c in chunks if c]
            text = " ".join(c for c in all_content if c)

            # Basic claim extraction: look for sentences with code references
            import re
            # Look for code-like references (function names, class names, API endpoints)
            code_patterns = [
                r'`([^`]+)`',  # Inline code
                r'`([^`]+)`',  # Function/class references
                r'([A-Z][a-zA-Z]+(?:\.[a-zA-Z]+)+)',  # Qualified names
                r'([a-z_]+\(.*?\))',  # Function calls
                r'(/[a-z0-9/_-]+)',  # API endpoints
            ]

            for pattern in code_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    if match and len(match) > 2:
                        claims.append({
                            "description": f"Reference to '{match}' in {section_title}",
                            "code_reference": match,
                            "source_section": section_title,
                            "source_document": document_path,
                        })

        # Deduplicate claims by code reference
        seen_refs: set[str] = set()
        unique_claims: list[dict] = []
        for claim in claims:
            ref = claim.get("code_reference", "")
            if ref and ref not in seen_refs:
                unique_claims.append(claim)
                seen_refs.add(ref)

        return unique_claims[:20]  # Limit to 20 claims

    def _find_relevant_docs(self, query: str) -> list[str]:
        """Find documents relevant to the query."""
        from ...document.tools.document_search import search_documents_by_keywords

        # Extract keywords from query
        keywords = [w for w in query.lower().split() if len(w) > 3][:5]

        results = search_documents_by_keywords(
            ingestor=self.doc_graph,
            keywords=keywords,
            workspace="default",
            limit=10,
        )

        # Return unique document paths
        doc_paths: list[str] = []
        seen: set[str] = set()
        for result in results:
            path = result.get("document_path", "")
            if path and path not in seen:
                doc_paths.append(path)
                seen.add(path)

        return doc_paths

    def _verify_claim(self, claim: dict) -> bool:
        """
        Verify claim against code graph.

        This is deterministic (no LLM needed).
        """
        code_reference = claim.get("code_reference", "")
        description = claim.get("description", "")

        if not code_reference:
            return False

        # Query code graph to verify the reference exists
        # Check for functions, classes, methods
        query = """
        MATCH (n)
        WHERE (n:Function OR n:Class OR n:Method OR n:Variable)
          AND (n.name = $ref
           OR n.qualified_name = $ref
           OR n.qualified_name CONTAINS $ref
           OR n.name CONTAINS $ref)
        RETURN count(n) as count
        """
        results = self._execute_code_query(query, {"ref": code_reference})
        if results and results[0].get("count", 0) > 0:
            return True

        # Check for API endpoint in routes
        if code_reference.startswith("/"):
            query = """
            MATCH (n:Function)
            WHERE n.name CONTAINS 'route'
               OR n.decorators CONTAINS 'route'
               OR n.decorators CONTAINS 'get'
               OR n.decorators CONTAINS 'post'
            RETURN n.name as name, n.decorators as decorators
            LIMIT 20
            """
            route_results = self._execute_code_query(query, {})
            for route in route_results:
                decorators = route.get("decorators", "")
                if code_reference in str(decorators):
                    return True

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