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
        # Query document graph for the document content
        query = """
        MATCH (d:Document {path: $path})-[:CONTAINS_SECTION]->(s:Section)
        OPTIONAL MATCH (s)-[:CONTAINS_SECTION]->(subsection:Section)
        RETURN s.title as title, s.content as content,
               s.qualified_name as qualified_name,
               collect(subsection.title) as subsections
        """
        results = self._execute_doc_query(query, {"path": document_path})

        if not results:
            # Try to find by partial path match
            query = """
            MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)
            WHERE d.path CONTAINS $path
            RETURN s.title as title, s.content as content,
                   s.qualified_name as qualified_name,
                   d.path as document_path
            """
            results = self._execute_doc_query(query, {"path": document_path})

        # Extract requirements from document sections
        # This is a simplified implementation - full implementation would use LLM
        requirements: list[dict] = []
        for result in results:
            content = result.get("content", "")
            title = result.get("title", "")
            if content:
                # Basic requirement extraction: look for sections that contain
                # keywords like "must", "shall", "required", "should"
                import re
                requirement_patterns = [
                    r'(?:must|shall|required|should)\s+([^.]+)',
                    r'([A-Z][^.]*\b(?:must|shall|required|should)\b[^.]*)',
                ]
                for pattern in requirement_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1]
                        if match and len(match) > 10:
                            requirements.append({
                                "description": match.strip(),
                                "source_section": title,
                                "source_document": document_path,
                            })

        # If no requirements found via pattern matching, use sections as requirements
        if not requirements:
            for result in results:
                title = result.get("title", "")
                if title and title.lower() not in ("introduction", "overview", "background"):
                    requirements.append({
                        "description": f"Section: {title}",
                        "source_section": title,
                        "source_document": document_path,
                    })

        return requirements[:20]  # Limit to 20 requirements for now

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

    def _verify_in_code(self, requirement: dict) -> bool:
        """
        Verify requirement exists in code graph.

        This is deterministic (no LLM needed).
        """
        description = requirement.get("description", "")
        source_section = requirement.get("source_section", "")

        # Extract potential function/class names from the requirement
        import re
        # Look for CamelCase words (likely class/function names)
        names = re.findall(r'\b([A-Z][a-zA-Z]+[a-z][a-zA-Z]*)\b', description)
        # Also look for snake_case words
        names.extend(re.findall(r'\b([a-z]+_[a-z_]+)\b', description))
        # Add section title as a potential name
        if source_section:
            # Convert section title to possible function/class name
            section_name = source_section.replace(" ", "_").replace("-", "_").lower()
            names.append(section_name)

        if not names:
            # No names to search for
            return False

        # Query code graph for matching functions/classes
        for name in names[:5]:  # Limit to first 5 names
            query = """
            MATCH (n)
            WHERE (n:Function OR n:Class OR n:Method)
              AND (toLower(n.name) CONTAINS toLower($name)
               OR toLower(n.qualified_name) CONTAINS toLower($name))
            RETURN count(n) as count
            """
            results = self._execute_code_query(query, {"name": name})
            if results and results[0].get("count", 0) > 0:
                return True

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