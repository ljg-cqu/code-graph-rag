from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from .. import constants as cs
from .. import exceptions as ex
from .. import logs as ls
from ..compat.pydantic_ai import Agent, UsageLimits
from ..config import settings
from ..rate_limiter import QuotaStatus, get_rate_limiter
from ..services import QueryProtocol
from ..services.graph_service import MemgraphIngestor
from ..utils.source_extraction import extract_source_lines

_FUNCTIONS_WITHOUT_DOCSTRINGS_QUERY = """
MATCH (n:Function|Method)
WHERE n.docstring IS NULL
  AND n.start_line IS NOT NULL
  AND n.end_line IS NOT NULL
  AND n.path IS NOT NULL
RETURN id(n) AS node_id,
       n.qualified_name AS qualified_name,
       n.name AS name,
       n.path AS file_path,
       n.start_line AS start_line,
       n.end_line AS end_line,
       n.pagerank_score AS pagerank_score
ORDER BY {order_by}
LIMIT $limit
"""

_UPDATE_DOCSTRING_QUERY = f"""
MATCH (n)
WHERE id(n) = $node_id
SET n.docstring = $docstring,
    n.{cs.KEY_DOCSTRING_GENERATED} = true,
    n.{cs.KEY_DOCSTRING_GENERATED_AT} = $generated_at
"""

_DOCSTRING_SYSTEM_PROMPT = """You are an expert Python documentation generator.
Given a function's source code, generate a clear, concise docstring that:
1. Explains the function's purpose in one sentence
2. Describes each parameter (if any): name and type
3. Describes the return value (if any)
4. Notes any important side effects or exceptions raised

Follow Google-style docstring format. Keep under 150 words.
Do not include triple backticks in the output."""

_DOCSTRING_USER_TEMPLATE = """Generate a docstring for this function:

```python
{source_code}
```

Output only the docstring string, nothing else."""


def _build_functions_query(priority: Literal["high-pagerank", "alphabetical"]) -> str:
    order_by = (
        "n.pagerank_score DESC"
        if priority == "high-pagerank"
        else "n.qualified_name ASC"
    )
    return _FUNCTIONS_WITHOUT_DOCSTRINGS_QUERY.format(order_by=order_by)


class DocstringInferenceResult(BaseModel):
    """Result of a docstring inference batch run."""

    processed: int
    succeeded: int
    failed: int
    errors: list[str]


class DocstringInferer:
    """Generate docstrings for code functions lacking them using an LLM."""

    __slots__ = ("agent", "_provider", "_model", "_ingestor")

    def __init__(self, ingestor: QueryProtocol | None = None) -> None:
        try:
            config = settings.active_cypher_config
            llm = self._create_chat_model(config)

            self.agent = Agent(
                model=llm,
                system_prompt=_DOCSTRING_SYSTEM_PROMPT,
                output_type=str,
                retries=settings.AGENT_RETRIES,
            )
            self._provider = config.provider
            self._model = config.model_id
            self._ingestor = ingestor

            if settings.RATE_LIMIT_ENABLED:
                limiter = get_rate_limiter()
                rpm = self._get_provider_rpm(config.provider)
                limiter.register_provider(
                    config.provider,
                    config.model_id,
                    requests_per_minute=rpm,
                )
        except Exception as e:
            raise ex.LLMGenerationError(ex.LLM_INIT_CYPHER.format(error=e)) from e

    @staticmethod
    def _create_chat_model(config: object) -> object:
        """Create a chat/completion model for docstring generation."""
        from ..services.llm import _create_chat_model as create_model

        return create_model(config)  # type: ignore[arg-type]

    def _get_provider_rpm(self, provider: str) -> float:
        provider_rpm = {
            cs.Provider.OPENAI: settings.OPENAI_REQUESTS_PER_MINUTE,
            cs.Provider.ANTHROPIC: settings.ANTHROPIC_REQUESTS_PER_MINUTE,
            cs.Provider.GOOGLE: settings.GOOGLE_REQUESTS_PER_MINUTE,
            "doubao": settings.DOUBAO_REQUESTS_PER_MINUTE,
            cs.Provider.OLLAMA: settings.OLLAMA_REQUESTS_PER_MINUTE,
        }
        return provider_rpm.get(provider.lower(), settings.RATE_LIMIT_REQUESTS_PER_MINUTE)

    async def infer_for_functions(
        self,
        limit: int | None = None,
        batch_size: int | None = None,
        priority: Literal["high-pagerank", "alphabetical"] | None = None,
    ) -> DocstringInferenceResult:
        if not settings.DOCSTRING_INFERENCE_ENABLED:
            logger.info("Docstring inference is disabled (set CGR_DOCSTRING_INFERENCE_ENABLED=true to enable)")
            return DocstringInferenceResult(
                processed=0, succeeded=0, failed=0, errors=[]
            )
        """Infer docstrings for functions lacking them.

        Args:
            limit: Maximum number of functions to process.
            batch_size: Number of functions to process in one batch.
            priority: Ordering priority for function selection.

        Returns:
            Result summary with processed, succeeded, failed counts and errors.
        """
        resolved_limit = limit or settings.DOCSTRING_INFERENCE_BATCH_SIZE
        resolved_batch_size = batch_size or settings.DOCSTRING_INFERENCE_BATCH_SIZE
        resolved_priority = priority or settings.DOCSTRING_INFERENCE_PRIORITY
        max_per_run = settings.DOCSTRING_INFERENCE_MAX_PER_RUN
        resolved_limit = min(resolved_limit, max_per_run)

        functions = self._fetch_functions_without_docstrings(
            resolved_limit, resolved_priority
        )
        if not functions:
            return DocstringInferenceResult(
                processed=0, succeeded=0, failed=0, errors=[]
            )

        logger.info(ls.DOCSTRING_INFERENCE_START, count=len(functions))

        errors: list[str] = []
        succeeded = 0
        failed = 0

        for func in functions:
            func_name = func.get("name", "unknown")
            func_qn = func.get("qualified_name", "unknown")
            logger.debug(ls.DOCSTRING_INFERENCE_FUNCTION, name=func_name, qn=func_qn)

            if settings.RATE_LIMIT_ENABLED:
                limiter = get_rate_limiter()
                status = limiter.check_quota(self._provider, self._model)
                if status == QuotaStatus.EXHAUSTED:
                    logger.warning(
                        ls.DOCSTRING_INFERENCE_QUOTA_EXHAUSTED,
                        provider=f"{self._provider}/{self._model}",
                    )
                    errors.append(
                        f"Quota exhausted for {self._provider}/{self._model}"
                    )
                    break

            try:
                source = self._read_function_source(func)
                if not source:
                    logger.warning(
                        ls.DOCSTRING_INFERENCE_SOURCE_ERROR,
                        name=func_name,
                        error="No source code found",
                    )
                    failed += 1
                    errors.append(f"No source for {func_qn}")
                    continue

                docstring = await self._generate_docstring(source)
                self._update_node_docstring(
                    int(func["node_id"]),
                    docstring,
                )
                succeeded += 1
                logger.debug(ls.DOCSTRING_INFERENCE_SUCCESS, name=func_name)

            except Exception as e:
                logger.warning(
                    ls.DOCSTRING_INFERENCE_LLM_ERROR,
                    name=func_name,
                    error=str(e),
                )
                failed += 1
                errors.append(f"{func_qn}: {e}")

            if (succeeded + failed) % resolved_batch_size == 0:
                logger.info(
                    ls.DOCSTRING_INFERENCE_BATCH_COMPLETE,
                    succeeded=succeeded,
                    failed=failed,
                )

        logger.info(
            ls.DOCSTRING_INFERENCE_COMPLETE,
            processed=len(functions),
            succeeded=succeeded,
            failed=failed,
        )

        return DocstringInferenceResult(
            processed=len(functions),
            succeeded=succeeded,
            failed=failed,
            errors=errors,
        )

    def _fetch_functions_without_docstrings(
        self,
        limit: int,
        priority: Literal["high-pagerank", "alphabetical"],
    ) -> list[dict]:
        query = _build_functions_query(priority)
        with self._get_ingestor() as ingestor:
            return ingestor.fetch_all(query, {"limit": limit})

    def _update_node_docstring(
        self,
        node_id: int,
        docstring: str,
    ) -> None:
        generated_at = datetime.now(UTC).isoformat()
        with self._get_ingestor() as ingestor:
            ingestor.execute_write(
                _UPDATE_DOCSTRING_QUERY,
                {
                    "node_id": node_id,
                    "docstring": docstring,
                    "generated_at": generated_at,
                },
            )

    @staticmethod
    def _read_function_source(func: dict) -> str | None:
        file_path = func.get("file_path")
        start_line = func.get("start_line")
        end_line = func.get("end_line")

        if not file_path or not isinstance(start_line, int) or not isinstance(end_line, int):
            return None

        return extract_source_lines(Path(file_path), start_line, end_line)

    async def _generate_docstring(self, source_code: str) -> str:
        prompt = _DOCSTRING_USER_TEMPLATE.format(source_code=source_code)
        result = await self.agent.run(
            prompt,
            usage_limits=UsageLimits(response_tokens_limit=256),
        )
        output = result.output if isinstance(result.output, str) else str(result.output)
        return output.strip()

    def _get_ingestor(self) -> QueryProtocol:
        if self._ingestor is not None:
            return self._ingestor
        return MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        )
