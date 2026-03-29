# (H) Provider validation errors
GOOGLE_GLA_NO_KEY = (
    "Gemini GLA provider requires api_key. "
    "Set ORCHESTRATOR_API_KEY or CYPHER_API_KEY in .env file."
)
GOOGLE_VERTEX_NO_PROJECT = (
    "Gemini Vertex provider requires project_id. "
    "Set ORCHESTRATOR_PROJECT_ID or CYPHER_PROJECT_ID in .env file."
)
OPENAI_NO_KEY = (
    "OpenAI provider requires api_key. "
    "Set ORCHESTRATOR_API_KEY or CYPHER_API_KEY in .env file."
)
ANTHROPIC_NO_KEY = (
    "Anthropic provider requires api_key. "
    "Set ORCHESTRATOR_API_KEY or CYPHER_API_KEY in .env file."
)
AZURE_NO_KEY = "Azure OpenAI provider requires api_key. Set AZURE_API_KEY in .env file."
AZURE_NO_ENDPOINT = (
    "Azure OpenAI provider requires endpoint. Set AZURE_OPENAI_ENDPOINT in .env file."
)
OLLAMA_NOT_RUNNING = (
    "Ollama server not responding at {endpoint}. "
    "Make sure Ollama is running: ollama serve"
)
UNKNOWN_PROVIDER = "Unknown provider '{provider}'. Available providers: {available}"

# (H) Dependency errors
SEMANTIC_EXTRA = "Semantic search requires 'semantic' extra: uv sync --extra semantic"

# (H) Configuration errors
PROVIDER_EMPTY = "Provider name cannot be empty in 'provider:model' format."
MODEL_ID_EMPTY = "Model ID cannot be empty."
MODEL_FORMAT_INVALID = (
    "Model must be specified as 'provider:model' (e.g., openai:gpt-4o)."
)
BATCH_SIZE_POSITIVE = "batch_size must be a positive integer"
CONFIG = "{role} configuration error: {error}"

# (H) Graph loading errors
GRAPH_FILE_NOT_FOUND = "Graph file not found: {path}"
FAILED_TO_LOAD_DATA = "Failed to load data from file"
NODES_NOT_LOADED = "Nodes should be loaded"
RELATIONSHIPS_NOT_LOADED = "Relationships should be loaded"
DATA_NOT_LOADED = "Data should be loaded"

# (H) Parser errors
NO_LANGUAGES = "No Tree-sitter languages available."

# (H) LLM errors
LLM_INIT_CYPHER = "Failed to initialize CypherGenerator: {error}"
LLM_INVALID_QUERY = "LLM did not generate a valid query. Output: {output}"
LLM_DANGEROUS_QUERY = "LLM generated a destructive Cypher query (found '{keyword}'). Query rejected: {query}"
LLM_GENERATION_FAILED = "Cypher generation failed: {error}"
LLM_INIT_ORCHESTRATOR = "Failed to initialize RAG Orchestrator: {error}"

# (H) Graph service errors
BATCH_SIZE = "batch_size must be a positive integer"
CONN = "Not connected to Memgraph."
AUTH_INCOMPLETE = (
    "Both username and password are required for authentication. "
    "Either provide both or neither."
)

# (H) Access control errors (used with raise)
ACCESS_DENIED = "Access denied: Cannot access files outside the project root."
DOC_UNSUPPORTED_PROVIDER = "DocumentAnalyzer does not support the 'local' LLM provider."


# (H) Exception classes
class LLMGenerationError(Exception):
    pass


# (H) Embedding exception hierarchy
from enum import StrEnum


class EmbeddingErrorCode(StrEnum):
    """Error codes for programmatic handling of embedding errors."""

    # Configuration errors (1xx)
    INPUT_VALIDATION = "E100"
    MODEL_NOT_FOUND = "E101"
    PROVIDER_NOT_FOUND = "E102"
    MODEL_NOT_SUPPORTED = "E103"
    AUTHENTICATION_FAILED = "E104"

    # Runtime errors (2xx)
    CONNECTION_FAILED = "E200"
    TIMEOUT = "E201"
    RATE_LIMITED = "E202"
    QUOTA_EXCEEDED = "E203"
    BATCH_SIZE_EXCEEDED = "E204"
    GENERATION_FAILED = "E205"

    # Data errors (3xx)
    DIMENSION_MISMATCH = "E300"


class EmbeddingError(Exception):
    """Base exception for embedding errors."""

    def __init__(
        self,
        message: str,
        *,
        error_code: EmbeddingErrorCode,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.provider = provider
        self.model = model


class InputValidationError(EmbeddingError):
    """Raised when input validation fails (empty text, invalid characters, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        input_text: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.INPUT_VALIDATION,
            provider=provider,
            model=model,
        )
        self.input_text = input_text
        self.reason = reason


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when the specified model does not exist."""

    def __init__(
        self,
        model: str,
        *,
        provider: str | None = None,
        available_models: list[str] | None = None,
    ) -> None:
        super().__init__(
            f"Model '{model}' not found for provider '{provider}'",
            error_code=EmbeddingErrorCode.MODEL_NOT_FOUND,
            provider=provider,
            model=model,
        )
        self.available_models = available_models


class EmbeddingProviderNotFoundError(EmbeddingError):
    """Raised when the embedding provider is not installed or configured."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Embedding provider '{provider}' not found",
            error_code=EmbeddingErrorCode.PROVIDER_NOT_FOUND,
            provider=provider,
        )


class EmbeddingModelNotSupportedError(EmbeddingError):
    """Raised when the model is not supported by the provider."""

    def __init__(
        self,
        model: str,
        provider: str,
        supported_models: list[str],
    ) -> None:
        super().__init__(
            f"Model '{model}' not supported by provider '{provider}'",
            error_code=EmbeddingErrorCode.MODEL_NOT_SUPPORTED,
            provider=provider,
            model=model,
        )
        self.supported_models = supported_models


class EmbeddingAuthenticationError(EmbeddingError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        auth_method: str | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.AUTHENTICATION_FAILED,
            provider=provider,
            model=model,
        )
        self.auth_method = auth_method


class EmbeddingConnectionError(EmbeddingError):
    """Raised when connection to embedding service fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.CONNECTION_FAILED,
            provider=provider,
            model=model,
        )
        self.endpoint = endpoint
        self.original_error = original_error


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when embedding request times out."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        timeout_seconds: float,
        operation: str = "embedding",
    ) -> None:
        super().__init__(
            f"Embedding {operation} timed out after {timeout_seconds}s",
            error_code=EmbeddingErrorCode.TIMEOUT,
            provider=provider,
            model=model,
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        retry_after: float | None = None,
        limit_type: str | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.RATE_LIMITED,
            provider=provider,
            model=model,
        )
        self.retry_after = retry_after
        self.limit_type = limit_type


class EmbeddingQuotaExceededError(EmbeddingError):
    """Raised when quota/billing limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        quota_type: str | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.QUOTA_EXCEEDED,
            provider=provider,
            model=model,
        )
        self.quota_type = quota_type


class BatchSizeExceededError(EmbeddingError):
    """Raised when batch size exceeds provider limit."""

    def __init__(
        self,
        batch_size: int,
        max_batch_size: int,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            f"Batch size {batch_size} exceeds maximum {max_batch_size}",
            error_code=EmbeddingErrorCode.BATCH_SIZE_EXCEEDED,
            provider=provider,
            model=model,
        )
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails for other reasons."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        input_text: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.GENERATION_FAILED,
            provider=provider,
            model=model,
        )
        self.input_text = input_text
        self.original_error = original_error


class DimensionMismatchError(EmbeddingError):
    """Raised when embedding dimension doesn't match vector index."""

    def __init__(
        self,
        existing_dim: int,
        configured_dim: int,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        existing_model: str | None = None,
    ) -> None:
        super().__init__(
            message
            or f"Dimension mismatch: existing={existing_dim}, configured={configured_dim}",
            error_code=EmbeddingErrorCode.DIMENSION_MISMATCH,
            provider=provider,
            model=model,
        )
        self.existing_dim = existing_dim
        self.configured_dim = configured_dim
        self.existing_model = existing_model
