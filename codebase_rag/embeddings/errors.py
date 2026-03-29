"""User-facing error messages for embedding errors.

This module provides formatted error messages for common embedding
error scenarios, helping users understand and resolve issues.
"""

from __future__ import annotations

from ..exceptions import EmbeddingErrorCode

# Provider-specific authentication help messages
AUTH_SOLUTIONS: dict[str, str] = {
    "openai": """
  - Set OPENAI_API_KEY or EMBEDDING_API_KEY environment variable
  - Get your API key at: https://platform.openai.com/api-keys
  - Verify: echo $OPENAI_API_KEY""",
    "google_gla": """
  - Set GOOGLE_API_KEY or EMBEDDING_API_KEY environment variable
  - Get your API key at: https://aistudio.google.com/apikey
  - Verify: echo $GOOGLE_API_KEY""",
    "google_vertex": """
  - Set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path
  - Or use: gcloud auth application-default login
  - Verify: gcloud auth list
  - Ensure service account has Vertex AI permissions""",
    "ollama": """
  - No authentication required for local Ollama
  - Ensure Ollama is running: ollama serve
  - Verify: curl http://localhost:11434/api/tags""",
    "local": """
  - No authentication required for local models
  - Ensure transformers/torch is installed
  - Model will download on first use""",
}

# User-facing error message templates
USER_FACING_MESSAGES: dict[EmbeddingErrorCode, str] = {
    # Configuration errors (1xx)
    EmbeddingErrorCode.INPUT_VALIDATION: """
Input validation failed: {reason}
  Input: {input_preview}...

Possible causes:
  - Empty or whitespace-only input
  - Input exceeds maximum length ({max_chars} characters)
  - Invalid characters detected

Solutions:
  1. Check your input text for issues
  2. Split large inputs into smaller chunks
  3. Clean input of invalid characters
""",
    EmbeddingErrorCode.MODEL_NOT_FOUND: """
Model '{model}' not found for provider '{provider}'.

Available models:
  {available_models}

Solutions:
  1. Check the model name for typos
  2. Use one of the supported models listed above
  3. For Ollama: run 'ollama pull {model}' to download the model
""",
    EmbeddingErrorCode.PROVIDER_NOT_FOUND: """
Embedding provider '{provider}' not found.

Supported providers: local, openai, google, ollama

Solutions:
  1. Check EMBEDDING_PROVIDER value for typos
  2. Ensure provider package is installed:
     - openai: pip install openai
     - google: pip install google-generativeai
     - ollama: Ensure Ollama is running locally
""",
    EmbeddingErrorCode.MODEL_NOT_SUPPORTED: """
Model '{model}' is not supported by provider '{provider}'.

Supported models for this provider:
  {supported_models}

Solutions:
  1. Choose a supported model from the list above
  2. Switch to a different provider that supports this model
""",
    EmbeddingErrorCode.AUTHENTICATION_FAILED: """
Authentication failed for {provider} ({auth_method}).

Solutions for {provider}:
{auth_solutions}

Common fixes:
  1. Verify your API key is valid and not expired
  2. Check environment variables are set correctly
  3. For Google Vertex AI, verify service account permissions
""",
    # Runtime errors (2xx)
    EmbeddingErrorCode.CONNECTION_FAILED: """
Cannot connect to {provider} at {endpoint}.

Troubleshooting steps:
  1. Check if the service is running
  2. Verify network connectivity: curl -v {endpoint}
  3. Check firewall and proxy settings
  4. For Ollama: ollama serve
  5. For cloud services: check service status page

Original error: {original_error}
""",
    EmbeddingErrorCode.TIMEOUT: """
Embedding request timed out after {timeout_seconds}s for {provider}.

This usually indicates:
  - Network latency issues
  - Server overload
  - Large batch size causing processing delays

Solutions:
  1. Reduce batch size (current: {batch_size})
  2. Increase timeout: EMBEDDING_TIMEOUT={timeout_seconds_doubled}
  3. Try again later if service is overloaded
  4. For local models, consider using a smaller model
""",
    EmbeddingErrorCode.RATE_LIMITED: """
Rate limit exceeded for {provider} ({limit_type}).

Wait {retry_after}s before retrying.

Rate limits for {provider}:
  {rate_limit_info}

Solutions:
  1. Reduce request frequency
  2. Use smaller batch sizes
  3. For OpenAI: check your tier at https://platform.openai.com/account/rate-limits
  4. For Google: check quotas at https://console.cloud.google.com/apis/credentials
  5. Consider upgrading your API plan
""",
    EmbeddingErrorCode.QUOTA_EXCEEDED: """
Quota exceeded for {provider} ({quota_type}).

Your {quota_type} quota has been exhausted.

Solutions:
  1. Wait for quota reset ({reset_time})
  2. Upgrade your plan at {billing_url}
  3. Switch to a different provider temporarily
  4. Use local embeddings: EMBEDDING_PROVIDER=local
""",
    EmbeddingErrorCode.BATCH_SIZE_EXCEEDED: """
Batch size {batch_size} exceeds maximum {max_batch_size} for {provider}.

Provider limits:
  - OpenAI: 2048 texts per batch
  - Google GLA: 100 texts per batch
  - Google Vertex: 250 texts per batch
  - Ollama: No hard limit (varies by model)

Solutions:
  1. Reduce batch size to {recommended_size}
  2. Let the provider auto-chunk: set batch_size=0
""",
    EmbeddingErrorCode.GENERATION_FAILED: """
Embedding generation failed for {provider}/{model}.

Reason: {reason}

Input that caused the error:
  {input_preview}

Solutions:
  1. Check if the model supports your input language
  2. Verify input is not corrupted
  3. Try with different input to isolate the issue
  4. Check provider status: {status_url}
""",
    # Data errors (3xx)
    EmbeddingErrorCode.DIMENSION_MISMATCH: """
Vector dimension mismatch detected.

  Current database: {existing_dim} dimensions (from model: {existing_model})
  Requested model: {configured_dim} dimensions ({model})

This happens when switching embedding models.

Options:
  1. Use a model with {existing_dim} dimensions:
     - Compatible models: {compatible_models}

  2. Re-create the vector index with new dimensions:
     WARNING: This will delete all existing embeddings!
     Time estimate: {reembedding_time} for {node_count} nodes

     Run: codegraph embedding reindex --model {model}

  3. Keep the existing model:
     Set EMBEDDING_MODEL={existing_model}
""",
}


def get_user_facing_message(
    error_code: EmbeddingErrorCode,
    **kwargs,
) -> str:
    """Format a user-facing error message with context.

    Args:
        error_code: The error code for the message.
        **kwargs: Context variables for message formatting.

    Returns:
        Formatted error message string.
    """
    template = USER_FACING_MESSAGES.get(error_code, "Unknown error: {error_code}")
    return template.format(error_code=error_code, **kwargs)


def get_auth_solutions(provider: str) -> str:
    """Get authentication help for a provider.

    Args:
        provider: Provider name.

    Returns:
        Authentication solutions string.
    """
    return AUTH_SOLUTIONS.get(provider, AUTH_SOLUTIONS.get("local", ""))


__all__ = [
    "USER_FACING_MESSAGES",
    "AUTH_SOLUTIONS",
    "get_user_facing_message",
    "get_auth_solutions",
]