#!/usr/bin/env python3
"""Test LLM and embedding configuration validity."""

import os

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI


def _create_client(api_key_env: str, endpoint_env: str) -> OpenAI:
    return OpenAI(
        api_key=os.getenv(api_key_env),
        base_url=os.getenv(endpoint_env),
    )


def _test_chat_model(name: str, api_key_env: str, endpoint_env: str, model_env: str) -> None:
    logger.info("=== Testing {} ===", name)
    try:
        model_name = os.getenv(model_env)
        if not model_name:
            raise ValueError(f"{model_env} is not set")
        oai_client = _create_client(api_key_env, endpoint_env)
        response = oai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Return only 'OK' if you receive this.",
                }
            ],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        logger.info("{} works. Response: {}", name, result)
    except Exception as error:
        logger.error("{} failed: {}", name, error)


def _test_embedding_api() -> None:
    logger.info("=== Testing Embedding API ===")
    try:
        model_name = os.getenv("EMBEDDING_MODEL")
        if not model_name:
            raise ValueError("EMBEDDING_MODEL is not set")
        oai_client = _create_client("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL")
        response = oai_client.embeddings.create(
            model=model_name,
            input=["Test text"],
        )
        dimension = len(response.data[0].embedding)
        logger.info("Embedding API works. Dimension returned: {}", dimension)
        expected_doc_dim = int(os.getenv("DOC_MEMGRAPH_VECTOR_DIM", "768"))
        if dimension != expected_doc_dim:
            logger.warning(
                "Embedding dimension {} does not match DOC_MEMGRAPH_VECTOR_DIM {}",
                dimension,
                expected_doc_dim,
            )
            logger.warning(
                "Update DOC_MEMGRAPH_VECTOR_DIM to match the configured embedding model output dimension"
            )
    except Exception as error:
        logger.error("Embedding API failed: {}", error)


def _log_configuration_issues() -> None:
    logger.info("=== Configuration Issues Found ===")
    issues: list[str] = []
    if os.getenv("DOC_MEMGRAPH_VECTOR_DIM") == "768":
        issues.append(
            "DOC_MEMGRAPH_VECTOR_DIM is set to 768 but text-embedding-v4 outputs 1024 dimensions. Fix: change DOC_MEMGRAPH_VECTOR_DIM=1024 in .env"
        )

    if issues:
        for issue in issues:
            logger.warning(issue)
    else:
        logger.info("No configuration issues found.")


def main() -> None:
    load_dotenv()
    _test_chat_model(
        "Orchestrator LLM",
        "ORCHESTRATOR_API_KEY",
        "ORCHESTRATOR_ENDPOINT",
        "ORCHESTRATOR_MODEL",
    )
    _test_chat_model(
        "Cypher LLM",
        "CYPHER_API_KEY",
        "CYPHER_ENDPOINT",
        "CYPHER_MODEL",
    )
    _test_embedding_api()
    _log_configuration_issues()


if __name__ == "__main__":
    main()
