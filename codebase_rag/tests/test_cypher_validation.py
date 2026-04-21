import pytest

from codebase_rag import exceptions as ex
from codebase_rag.services.llm import _validate_cypher_read_only_async


class TestValidateCypherReadOnlyAsync:
    """Test the mechanical fast-path of async Cypher validation.

    The hybrid validator has two layers:
    1. Mechanical fast-path (tested here) — no LLM required
    2. LLM semantic validation — tested separately with @pytest.mark.slow
    """

    @pytest.mark.asyncio
    async def test_rejects_multiple_semicolons(self) -> None:
        with pytest.raises(ex.LLMGenerationError, match="Multiple"):
            await _validate_cypher_read_only_async(
                "MATCH (n) RETURN n; MATCH (m) RETURN m;"
            )

    @pytest.mark.asyncio
    async def test_rejects_over_window_function(self) -> None:
        with pytest.raises(ex.LLMGenerationError, match="OVER"):
            await _validate_cypher_read_only_async(
                "MATCH (n) RETURN n.name, count(*) OVER(PARTITION BY n.type)"
            )

    @pytest.mark.asyncio
    async def test_rejects_using_parallel_execution(self) -> None:
        with pytest.raises(ex.LLMGenerationError, match="PARALLEL"):
            await _validate_cypher_read_only_async(
                "MATCH (n) RETURN n USING PARALLEL EXECUTION"
            )

    @pytest.mark.asyncio
    async def test_allows_safe_match_query(self) -> None:
        await _validate_cypher_read_only_async("MATCH (n) RETURN n;")

    @pytest.mark.asyncio
    async def test_allows_safe_match_with_where(self) -> None:
        await _validate_cypher_read_only_async(
            "MATCH (n:Function) WHERE n.name = 'foo' RETURN n;"
        )

    @pytest.mark.asyncio
    async def test_allows_single_semicolon(self) -> None:
        await _validate_cypher_read_only_async("MATCH (n) RETURN n;")

    @pytest.mark.asyncio
    async def test_allows_query_without_semicolon(self) -> None:
        await _validate_cypher_read_only_async("MATCH (n) RETURN n")

    @pytest.mark.asyncio
    async def test_rejects_mixed_case_over(self) -> None:
        with pytest.raises(ex.LLMGenerationError, match="OVER"):
            await _validate_cypher_read_only_async(
                "MATCH (n) RETURN over(n.name)"
            )

    @pytest.mark.asyncio
    async def test_rejects_mixed_case_parallel(self) -> None:
        with pytest.raises(ex.LLMGenerationError, match="PARALLEL"):
            await _validate_cypher_read_only_async(
                "MATCH (n) RETURN n using parallel execution"
            )
