from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger
from pydantic_ai import Agent, DeferredToolRequests, Tool

from .. import constants as cs
from .. import exceptions as ex
from .. import logs as ls
from ..config import ModelConfig, settings
from ..prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
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
    import re

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

    if not query.endswith(cs.CYPHER_SEMICOLON):
        query += cs.CYPHER_SEMICOLON
    return query


_COMMENT_OR_WS = r"(?:\s|//[^\n]*|/\*.*?\*/)+"


def _build_keyword_pattern(keyword: str) -> re.Pattern[str]:
    parts = keyword.split()
    if len(parts) == 1:
        return re.compile(rf"\b{re.escape(parts[0])}\b")
    joined = _COMMENT_OR_WS.join(re.escape(p) for p in parts)
    return re.compile(rf"\b{joined}\b", re.DOTALL)


_CYPHER_DANGEROUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (kw, _build_keyword_pattern(kw)) for kw in cs.CYPHER_DANGEROUS_KEYWORDS
]


def _validate_cypher_read_only(query: str) -> None:
    upper_query = query.upper()

    # Block multiple queries
    if upper_query.count(";") > 1:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="Multiple semicolon-separated queries", query=query
            )
        )

    # Block unsupported Memgraph Community features
    if "OVER(" in upper_query:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(keyword="OVER() window function", query=query)
        )
    if "USING PARALLEL EXECUTION" in upper_query:
        raise ex.LLMGenerationError(
            ex.LLM_DANGEROUS_QUERY.format(
                keyword="USING PARALLEL EXECUTION", query=query
            )
        )

    # Block dangerous write operations
    for keyword, pattern in _CYPHER_DANGEROUS_PATTERNS:
        if pattern.search(upper_query):
            raise ex.LLMGenerationError(
                ex.LLM_DANGEROUS_QUERY.format(keyword=keyword, query=query)
            )


class CypherGenerator:
    __slots__ = ("agent",)

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
        except Exception as e:
            raise ex.LLMGenerationError(ex.LLM_INIT_CYPHER.format(error=e)) from e

    async def generate(self, natural_language_query: str) -> str:
        logger.info(ls.CYPHER_GENERATING.format(query=natural_language_query))
        try:
            result = await self.agent.run(natural_language_query)
            if (
                not isinstance(result.output, str)
                or cs.CYPHER_MATCH_KEYWORD not in result.output.upper()
            ):
                raise ex.LLMGenerationError(
                    ex.LLM_INVALID_QUERY.format(output=result.output)
                )

            query = _clean_cypher_response(result.output)
            _validate_cypher_read_only(query)
            logger.info(ls.CYPHER_GENERATED.format(query=query))
            return query
        except Exception as e:
            logger.error(ls.CYPHER_ERROR.format(error=e))
            raise ex.LLMGenerationError(ex.LLM_GENERATION_FAILED.format(error=e)) from e


def create_rag_orchestrator(tools: list[Tool]) -> Agent:
    try:
        config = settings.active_orchestrator_config
        llm = _create_provider_model(config)

        return Agent(
            model=llm,
            system_prompt=build_rag_orchestrator_prompt(tools),
            tools=tools,
            retries=settings.AGENT_RETRIES,
            output_retries=settings.ORCHESTRATOR_OUTPUT_RETRIES,
            output_type=[str, DeferredToolRequests],
        )
    except Exception as e:
        raise ex.LLMGenerationError(ex.LLM_INIT_ORCHESTRATOR.format(error=e)) from e
