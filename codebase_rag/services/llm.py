from __future__ import annotations

import hashlib
import re
import time
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, DeferredToolRequests, Tool
from pydantic_ai.usage import UsageLimits

from .. import constants as cs
from .. import exceptions as ex
from .. import logs as ls
from ..config import ModelConfig, settings
from ..prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
    build_cypher_repair_prompt,
    build_rag_orchestrator_prompt,
)
from ..providers import get_provider_from_config

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def _create_provider_model(config: ModelConfig) -> Model:
    provider = get_provider_from_config(config)
    return provider.create_model(config.model_id)


def _clean_cypher_response(response_text: str) -> str:
    """Clean LLM response to extract pure Cypher query.

    Handles markdown formatting that models sometimes output:
    - Triple backticks (```cypher ... ```)
    - Bold text (**Cypher Query:**)
    - Headers and other markdown
    - Removes unsupported features like parallel execution
    - Removes comments
    - Only keeps a single query (multiple queries are not supported)
    """
    query = response_text.strip()

    # Extract content from code blocks (```cypher ... ``` or ``` ... ```)
    if "```" in query:
        parts = query.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if block.lower().startswith("cypher"):
                block = block[len("cypher") :]
            query = block.strip()
    else:
        # Remove markdown bold/headers (e.g., **Cypher Query:**)
        while "**" in query:
            start = query.index("**")
            end = query.find("**", start + 2)
            if end == -1:
                break
            after = end + 2
            if after < len(query) and query[after] == ":":
                after += 1
            query = query[:start] + query[after:].lstrip()
        # Remove single backticks
        query = query.replace(cs.CYPHER_BACKTICK, "")
        # Remove "cypher" prefix if present
        if query.lower().startswith(cs.CYPHER_PREFIX):
            query = query[len(cs.CYPHER_PREFIX) :].strip()

    # Remove unsupported parallel execution clause
    query = query.replace("USING PARALLEL EXECUTION", "").strip()

    # Remove all line comments (--)
    lines = []
    for line in query.split("\n"):
        comment_start = line.find("--")
        if comment_start != -1:
            line = line[:comment_start]
        lines.append(line.rstrip())
    query = "\n".join([line_content for line_content in lines if line_content.strip()])

    # Remove all block comments (/* ... */)

    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL).strip()

    # Only keep the first query (multiple semicolon-separated queries are not supported)
    if ";" in query:
        query = query.split(";")[0].strip() + ";"

    # Remove any leading empty lines left after removing the above
    while query.startswith("\n"):
        query = query[1:].strip()

    # Replace unsupported | label/relationship syntax with IN clause pattern
    # First handle node label patterns like (n:Label1|Label2)
    label_pattern = re.compile(r"\((\w+):([a-zA-Z0-9_|]+)\)")
    matches = list(label_pattern.finditer(query))
    where_clauses = []
    for match in reversed(matches):
        var_name = match.group(1)
        labels_part = match.group(2)
        if "|" in labels_part:
            labels = [f"'{label.strip()}'" for label in labels_part.split("|")]
            where_clause = f"labels({var_name})[0] IN [{', '.join(labels)}]"
            where_clauses.append(where_clause)
            # Replace the (var:Label1|Label2) with (var)
            query = query[: match.start()] + f"({var_name})" + query[match.end() :]

    # Handle all relationship type patterns with | syntax:
    # ()-[r:REL1|REL2]-(), ()-[:REL1|REL2]-(), ()->[r:REL1|REL2]-(), ()-[r:REL1|REL2]->()
    rel_pattern = re.compile(r"(<?-)\[(\w*):([a-zA-Z0-9_|]+)\](->?)")
    rel_matches = list(rel_pattern.finditer(query))
    for match in reversed(rel_matches):
        left_arrow = match.group(1)
        rel_var = match.group(2) or "r"
        rel_types_part = match.group(3)
        right_arrow = match.group(4)
        if "|" in rel_types_part:
            rel_types = [
                f"'{rel_type.strip()}'" for rel_type in rel_types_part.split("|")
            ]
            where_clause = f"type({rel_var}) IN [{', '.join(rel_types)}]"
            where_clauses.append(where_clause)
            # Replace the [var:REL1|REL2] with [var], preserving direction arrows
            new_rel_part = f"{left_arrow}[{rel_var}]{right_arrow}"
            query = query[: match.start()] + new_rel_part + query[match.end() :]

    # Add the where clauses if any
    if where_clauses:
        if "WHERE" in query.upper():
            # Insert after existing WHERE
            insert_pos = query.upper().find("WHERE") + len("WHERE")
            query = (
                query[:insert_pos]
                + " "
                + " AND ".join(where_clauses)
                + " AND "
                + query[insert_pos:]
            )
        else:
            # Insert WHERE clause before RETURN
            return_pos = query.upper().find("RETURN")
            if return_pos != -1:
                query = (
                    query[:return_pos]
                    + " WHERE "
                    + " AND ".join(where_clauses)
                    + " "
                    + query[return_pos:]
                )

    # Detect and discard trailing broken query fragments that leak past
    # markdown extraction. LLMs sometimes concatenate two queries where
    # the second starts with a fragment (no MATCH keyword at beginning).
    # This happens when the LLM generates output like:
    #   ```cypher MATCH (c:Class)...RETURN...; .qualified_name)...LIMIT 50;```
    # After markdown extraction and semicolon splitting, the first query
    # is valid but a trailing fragment without MATCH may remain.
    if query and not query.lstrip().upper().startswith("MATCH"):
        logger.warning(f"Discarding trailing Cypher fragment without MATCH: {query[:80]}")
        return ""

    if not query.endswith(cs.CYPHER_SEMICOLON):
        query += cs.CYPHER_SEMICOLON
    return query


# Cache for LLM-validated Cypher queries
_CYPHER_LLM_VALIDATION_CACHE: dict[str, tuple[bool, float]] = {}
_CYPHER_LLM_VALIDATION_CACHE_TTL = 3600  # 1 hour


class CypherSafetyAssessment(BaseModel):
    """Structured output for Cypher safety validation."""

    safe: bool = Field(..., description="Whether the query is read-only safe")
    reasoning: str = Field(default="", description="Brief explanation")


# Singleton agent for Cypher safety validation
_CYPHER_SAFETY_AGENT: Agent | None = None


async def _llm_validate_cypher_safety(query: str) -> bool:
    """Use LLM to validate Cypher query safety semantically.

    Uses structured output (Pydantic model) instead of string parsing.
    Agent is created once and reused for efficiency.
    """
    global _CYPHER_SAFETY_AGENT

    if _CYPHER_SAFETY_AGENT is None:
        system_prompt = """You validate Cypher queries for read-only safety. A query is UNSAFE if it:
- Modifies data (CREATE, DELETE, DETACH DELETE, SET, REMOVE, MERGE that creates new nodes)
- Modifies schema (CREATE INDEX, DROP INDEX, CREATE CONSTRAINT, etc.)
- Uses non-read-only CALL procedures (write procedures, admin procedures)
- Contains multiple statements separated by semicolons

A query is SAFE if it only:
- Reads data (MATCH, RETURN, WITH, OPTIONAL MATCH, UNWIND)
- Uses read-only CALL procedures
- Uses COUNT, COLLECT, etc.

Respond with a JSON object: {"safe": true/false, "reasoning": "brief explanation"}"""

        config = settings.active_cypher_config
        llm = _create_provider_model(config)
        _CYPHER_SAFETY_AGENT = Agent(
            model=llm,
            system_prompt=system_prompt,
            output_type=CypherSafetyAssessment,
            retries=1,
        )

    try:
        result = await _CYPHER_SAFETY_AGENT.run(query)
        return result.output.safe
    except Exception as e:
        logger.warning(f"LLM Cypher validation failed: {e}. Defaulting to unsafe.")
        return False


async def _validate_cypher_read_only_async(query: str) -> None:
    """Validate Cypher query for read-only safety.

    Uses hybrid approach: mechanical fast-path + LLM semantic validation.
    """
    upper_query = query.upper()

    # Fast-path mechanical checks (zero false positives)
    if upper_query.count(";") > 1:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="Multiple semicolon-separated queries",
                query=query,
            )
        )
    if "OVER(" in upper_query:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="OVER() window function",
                query=query,
            )
        )
    if "USING PARALLEL EXECUTION" in upper_query:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="USING PARALLEL EXECUTION",
                query=query,
            )
        )

    # Check cache for LLM validation
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    if query_hash in _CYPHER_LLM_VALIDATION_CACHE:
        is_safe, cached_time = _CYPHER_LLM_VALIDATION_CACHE[query_hash]
        if time.time() - cached_time < _CYPHER_LLM_VALIDATION_CACHE_TTL:
            if is_safe:
                return
            raise ex.LLMGenerationError(
                ex.LLM_DANGEROUS_QUERY.format(
                    keyword="Unsafe operation (cached)",
                    query=query,
                )
            )

    # LLM semantic validation
    is_safe = await _llm_validate_cypher_safety(query)

    # Cache result
    _CYPHER_LLM_VALIDATION_CACHE[query_hash] = (is_safe, time.time())

    if not is_safe:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="Unsafe write operation detected by LLM",
                query=query,
            )
        )


class CypherGenerator:
    __slots__ = ("agent", "_fallback_agent")

    def __init__(self) -> None:
        try:
            config = settings.active_cypher_config
            llm = _create_provider_model(config)

            system_prompt = (
                LOCAL_CYPHER_SYSTEM_PROMPT
                if config.provider == cs.Provider.OLLAMA
                else CYPHER_SYSTEM_PROMPT
            )

            self.agent = Agent(
                model=llm,
                system_prompt=system_prompt,
                output_type=str,
                retries=settings.AGENT_RETRIES,
            )
            self._fallback_agent = None
        except Exception as e:
            raise ex.LLMGenerationError(ex.LLM_INIT_CYPHER.format(error=e)) from e

    async def _run_query_prompt(self, prompt: str) -> str:
        result = await self.agent.run(prompt, usage_limits=UsageLimits(request_limit=settings.AGENT_REQUEST_LIMIT))
        if (
            not isinstance(result.output, str)
            or cs.CYPHER_MATCH_KEYWORD not in result.output.upper()
        ):
            raise ex.LLMGenerationError(
                ex.LLM_INVALID_QUERY.format(output=result.output)
            )

        query = _clean_cypher_response(result.output)
        await _validate_cypher_read_only_async(query)
        return query

    async def generate(self, natural_language_query: str) -> str:
        logger.info(ls.CYPHER_GENERATING.format(query=natural_language_query))
        try:
            query = await self._run_query_prompt(natural_language_query)
            logger.info(ls.CYPHER_GENERATED.format(query=query))
            return query
        except Exception as e:
            logger.error(ls.CYPHER_ERROR.format(error=e))
            raise ex.LLMGenerationError(ex.LLM_GENERATION_FAILED.format(error=e)) from e

    async def repair(
        self,
        natural_language_query: str,
        failed_query: str,
        error_message: str,
    ) -> str:
        logger.info(ls.CYPHER_REPAIRING.format(query=natural_language_query))
        try:
            query = await self._run_query_prompt(
                build_cypher_repair_prompt(
                    natural_language_query,
                    failed_query,
                    error_message,
                )
            )
            logger.info(ls.CYPHER_REPAIRED.format(query=query))
            return query
        except Exception as e:
            logger.error(ls.CYPHER_ERROR.format(error=e))
            raise ex.LLMGenerationError(ex.LLM_GENERATION_FAILED.format(error=e)) from e

    async def generate_fallback(
        self,
        user_query: str,
        error_message: str,
    ) -> str | None:
        """Generate a conservative fallback Cypher query when primary generation fails.

        Args:
            user_query: Original natural language query.
            error_message: Error from the primary Cypher generation attempt.

        Returns:
            A conservative Cypher query string, or None if generation fails.
        """
        if self._fallback_agent is None:
            from pydantic_ai import Agent

            self._fallback_agent = Agent(
                model=self.agent.model,
                system_prompt=(
                    "You generate conservative, simple Cypher queries. "
                    "Use only MATCH, WHERE, RETURN, LIMIT, ORDER BY. "
                    "No CALL procedures, no complex patterns. "
                    "Return ONLY the Cypher query, no markdown, no explanation."
                ),
                retries=1,
            )

        fallback_prompt = f"""The primary Cypher generation failed with this error: {error_message}

User query: {user_query}

Generate a SIMPLE, conservative Cypher query that will likely work. Guidelines:
- Use only MATCH, WHERE, RETURN, LIMIT, ORDER BY
- Do not use CALL procedures
- Do not use variable-length paths longer than 3 hops
- Prefer exact name/qualified_name matching over complex patterns
- If unsure, return a query that matches nodes by name CONTAINS and returns their qualified_name and path

Return ONLY the Cypher query string, no markdown, no explanation."""

        try:
            result = await self._fallback_agent.run(fallback_prompt)
            query = str(result.output).strip()
            if query.startswith("```"):
                query = query.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return query
        except Exception as e:
            logger.warning(f"Fallback Cypher generation failed: {e}")
            return None


def create_rag_orchestrator_with_config(
    config: ModelConfig,
    tools: list[Tool],
    system_prompt: str | None = None,
    output_type: object | None = None,
    mode: str | None = None,
) -> Agent:
    try:
        llm = _create_provider_model(config)

        return Agent(
            model=llm,
            system_prompt=system_prompt or build_rag_orchestrator_prompt(tools, mode=mode),
            tools=tools,
            retries=settings.AGENT_RETRIES,
            output_retries=settings.ORCHESTRATOR_OUTPUT_RETRIES,
            output_type=output_type or [str, DeferredToolRequests],
        )
    except Exception as e:
        raise ex.LLMGenerationError(ex.LLM_INIT_ORCHESTRATOR.format(error=e)) from e


def create_rag_orchestrator(tools: list[Tool], mode: str | None = None) -> Agent:
    return create_rag_orchestrator_with_config(
        settings.active_orchestrator_config, tools, mode=mode
    )


def create_compression_agent() -> Agent:
    """Create a PydanticAI agent for semantic context distillation."""
    from ..compression_prompts import COMPRESSION_SYSTEM_PROMPT
    from ..compression_schemas import CompressedState

    role = settings.SEMANTIC_COMPRESSION_MODEL_ROLE
    config = getattr(settings, f"active_{role}_config", settings.active_orchestrator_config)
    llm = _create_provider_model(config)

    return Agent(
        model=llm,
        system_prompt=COMPRESSION_SYSTEM_PROMPT,
        output_type=CompressedState,
        retries=settings.AGENT_RETRIES,
    )
