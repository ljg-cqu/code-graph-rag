"""Document validation tools for interactive agent.

Provides Pydantic AI Tool factories for validating code against specifications
and documentation against code.
"""

from __future__ import annotations

from loguru import logger
from pydantic_ai import Tool

from .. import constants as cs
from ..config import settings
from ..shared.query_router import QueryMode, QueryRequest, QueryRouter
from ..shared.validation.api import ValidationRequest, ValidationTriggerAPI
from . import tool_descriptions as td


def create_validate_code_against_spec_tool(
    query_router: QueryRouter | None = None,
) -> Tool:
    """Create validate_code_against_spec tool for Pydantic AI agent.

    Usage: Validate CODE against DOCUMENT specifications.
    Example: "Does code implement all endpoints in OpenAPI spec?"
    """

    async def validate_code_against_spec(
        spec_document_path: str,
        scope: str = "all",
        max_cost_usd: float = 0.50,
        dry_run: bool = False,
    ) -> str:
        """Validate code against specification document.

        Args:
            spec_document_path: Path to the specification document
            scope: Validation scope (all, sections, claims)
            max_cost_usd: Maximum cost budget in USD
            dry_run: If true, only estimate cost without running

        Returns:
            Validation report or cost estimate
        """
        logger.info(f"Validating code against spec: {spec_document_path}")

        # Create validation API for cost estimation
        validation_api = ValidationTriggerAPI(
            llm_provider=settings.active_orchestrator_config.provider,
        )

        # Request validation with cost estimation
        validation_request = ValidationRequest(
            document_path=spec_document_path,
            mode="CODE_VS_DOC",
            scope=scope,
            max_cost_usd=max_cost_usd,
            dry_run=dry_run,
        )

        try:
            trigger_result = await validation_api.request_validation(validation_request)

            if not trigger_result.accepted:
                cost_info = ""
                if trigger_result.cost_estimate:
                    cost_info = f" (Estimated: ${trigger_result.cost_estimate.estimated_cost_usd:.2f})"
                return f"Validation not executed: {trigger_result.message}{cost_info}"

            # Execute validation via QueryRouter
            router = query_router
            if router is None:
                router = _create_validation_query_router()
                if router is None:
                    return "Validation failed: Could not connect to graphs"

            request = QueryRequest(
                question=f"Validate {spec_document_path} against code",
                mode=QueryMode.CODE_VS_DOC,
                scope=scope,
            )
            response = router.query(request)

            if response.validation_report:
                report = response.validation_report
                return (
                    f"**Code vs Document Validation Report**\n\n"
                    f"- Total requirements: {report.total}\n"
                    f"- Passed: {report.passed}\n"
                    f"- Failed: {report.failed}\n"
                    f"- Accuracy: {report.accuracy_score:.1%}\n\n"
                    f"{response.answer}"
                )

            return response.answer

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return f"Validation failed: {e}"

    return Tool(
        validate_code_against_spec,
        name="validate_code_against_spec",
        description=td.MCP_VALIDATE_CODE_AGAINST_SPEC,
    )


def create_validate_doc_against_code_tool(
    query_router: QueryRouter | None = None,
) -> Tool:
    """Create validate_doc_against_code tool for Pydantic AI agent.

    Usage: Validate DOCUMENT against actual CODE.
    Example: "Is docs/api.md still accurate?"
    """

    async def validate_doc_against_code(
        document_path: str,
        scope: str = "all",
        max_cost_usd: float = 0.50,
        dry_run: bool = False,
    ) -> str:
        """Validate documentation against actual code.

        Args:
            document_path: Path to the document to validate
            scope: Validation scope (all, sections, claims)
            max_cost_usd: Maximum cost budget in USD
            dry_run: If true, only estimate cost without running

        Returns:
            Validation report or cost estimate
        """
        logger.info(f"Validating doc against code: {document_path}")

        # Create validation API for cost estimation
        validation_api = ValidationTriggerAPI(
            llm_provider=settings.active_orchestrator_config.provider,
        )

        # Request validation with cost estimation
        validation_request = ValidationRequest(
            document_path=document_path,
            mode="DOC_VS_CODE",
            scope=scope,
            max_cost_usd=max_cost_usd,
            dry_run=dry_run,
        )

        try:
            trigger_result = await validation_api.request_validation(validation_request)

            if not trigger_result.accepted:
                cost_info = ""
                if trigger_result.cost_estimate:
                    cost_info = f" (Estimated: ${trigger_result.cost_estimate.estimated_cost_usd:.2f})"
                return f"Validation not executed: {trigger_result.message}{cost_info}"

            # Execute validation via QueryRouter
            router = query_router
            if router is None:
                router = _create_validation_query_router()
                if router is None:
                    return "Validation failed: Could not connect to graphs"

            request = QueryRequest(
                question=f"Validate {document_path} against code",
                mode=QueryMode.DOC_VS_CODE,
                scope=scope,
            )
            response = router.query(request)

            if response.validation_report:
                report = response.validation_report
                return (
                    f"**Document vs Code Validation Report**\n\n"
                    f"- Total claims: {report.total}\n"
                    f"- Accurate: {report.passed}\n"
                    f"- Outdated: {report.failed}\n"
                    f"- Accuracy: {report.accuracy_score:.1%}\n\n"
                    f"{response.answer}"
                )

            return response.answer

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return f"Validation failed: {e}"

    return Tool(
        validate_doc_against_code,
        name="validate_doc_against_code",
        description=td.MCP_VALIDATE_DOC_AGAINST_CODE,
    )


def _create_validation_query_router() -> QueryRouter | None:
    """Create QueryRouter with both graph connections for validation."""
    try:
        from ..services.graph_service import MemgraphIngestor

        code_ingestor = MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=100,
        )

        doc_ingestor = MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            batch_size=100,
        )

        return QueryRouter(
            code_graph=code_ingestor,
            doc_graph=doc_ingestor,
        )

    except Exception as e:
        logger.warning(f"Could not create validation query router: {e}")
        return None


__all__ = [
    "create_validate_code_against_spec_tool",
    "create_validate_doc_against_code_tool",
]