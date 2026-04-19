from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict, Unpack

from dotenv import load_dotenv
from loguru import logger
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import constants as cs
from . import exceptions as ex
from . import logs
from .types_defs import CgrignorePatterns, EmbeddingConfigKwargs, ModelConfigKwargs

# Support ENV_FILE environment variable for custom env file location
_env_file = os.environ.get("ENV_FILE")
if _env_file:
    if os.path.isfile(_env_file):
        load_dotenv(_env_file)
        logger.debug(f"Loaded environment from ENV_FILE: {_env_file}")
    else:
        logger.warning(
            f"ENV_FILE is set to '{_env_file}' but file not found. "
            f"Falling back to default .env file."
        )
        load_dotenv()  # Fallback to default
else:
    load_dotenv()  # Default: load from current directory


class ApiKeyInfoEntry(TypedDict):
    env_var: str
    url: str
    name: str


API_KEY_INFO: dict[str, ApiKeyInfoEntry] = {
    cs.Provider.OPENAI: {
        "env_var": "OPENAI_API_KEY",
        "url": "https://platform.openai.com/api-keys",
        "name": "OpenAI",
    },
    cs.Provider.ANTHROPIC: {
        "env_var": "ANTHROPIC_API_KEY",
        "url": "https://console.anthropic.com/settings/keys",
        "name": "Anthropic",
    },
    cs.Provider.GOOGLE: {
        "env_var": "GOOGLE_API_KEY",
        "url": "https://console.cloud.google.com/apis/credentials",
        "name": "Google AI",
    },
    cs.Provider.AZURE: {
        "env_var": "AZURE_API_KEY",
        "url": "https://portal.azure.com/",
        "name": "Azure OpenAI",
    },
    cs.Provider.COHERE: {
        "env_var": "COHERE_API_KEY",
        "url": "https://dashboard.cohere.com/api-keys",
        "name": "Cohere",
    },
}


def format_missing_api_key_errors(
    provider: str, role: str = cs.DEFAULT_MODEL_ROLE
) -> str:
    provider_lower = provider.lower()

    if provider_lower in API_KEY_INFO:
        info = API_KEY_INFO[provider_lower]
        env_var = info["env_var"]
        url = info["url"]
        name = info["name"]
    else:
        env_var = f"{provider.upper()}_API_KEY"
        url = f"your {provider} provider's website"
        name = provider.capitalize()

    role_msg = f" for {role}" if role != cs.DEFAULT_MODEL_ROLE else ""

    error_msg = f"""
─── API Key Missing ───────────────────────────────────────────────

  Error: {env_var} environment variable is not set.
         This is required to use {name}{role_msg}.

  To fix this:

  1. Get your API key from:
     {url}

  2. Set it in your environment:
     export {env_var}='your-key-here'

     Or add it to your .env file in the project root:
     {env_var}=your-key-here

  3. Alternatively, you can use a local model with Ollama:
     (No API key required)

───────────────────────────────────────────────────────────────────
""".strip()  # noqa: W293
    return error_msg


LOCAL_PROVIDERS = frozenset({cs.Provider.OLLAMA, cs.Provider.LOCAL, cs.Provider.VLLM})


def _model_config_from_mapping(entry: Mapping[object, object]) -> ModelConfig:
    if not all(isinstance(key, str) for key in entry):
        raise ValueError("ModelConfig mapping keys must be strings")
    provider = entry.get("provider")
    model_id = entry.get("model_id")
    if not isinstance(provider, str) or not isinstance(model_id, str):
        raise ValueError("ModelConfig mappings require string provider and model_id")

    api_key = entry.get("api_key")
    endpoint = entry.get("endpoint")
    project_id = entry.get("project_id")
    region = entry.get("region")
    provider_type = entry.get("provider_type")
    thinking_budget = entry.get("thinking_budget")
    service_account_file = entry.get("service_account_file")

    return ModelConfig(
        provider=provider,
        model_id=model_id,
        api_key=api_key if isinstance(api_key, str) else None,
        endpoint=endpoint if isinstance(endpoint, str) else None,
        project_id=project_id if isinstance(project_id, str) else None,
        region=region if isinstance(region, str) else None,
        provider_type=provider_type if isinstance(provider_type, str) else None,
        thinking_budget=thinking_budget if isinstance(thinking_budget, int) else None,
        service_account_file=(
            service_account_file if isinstance(service_account_file, str) else None
        ),
    )


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    api_key: str | None = None
    endpoint: str | None = None
    project_id: str | None = None
    region: str | None = None
    provider_type: str | None = None
    thinking_budget: int | None = None
    service_account_file: str | None = None

    def to_update_kwargs(self) -> ModelConfigKwargs:
        result = asdict(self)
        del result[cs.FIELD_PROVIDER]
        del result[cs.FIELD_MODEL_ID]
        return ModelConfigKwargs(**result)

    def validate_api_key(self, role: str = cs.DEFAULT_MODEL_ROLE) -> None:
        provider_lower = self.provider.lower()
        provider_env_keys = {
            cs.Provider.ANTHROPIC: cs.ENV_ANTHROPIC_API_KEY,
            cs.Provider.AZURE: cs.ENV_AZURE_API_KEY,
        }
        env_key = provider_env_keys.get(provider_lower)
        if (
            provider_lower in LOCAL_PROVIDERS
            or (
                provider_lower == cs.Provider.GOOGLE
                and self.provider_type == cs.GoogleProviderType.VERTEX
            )
            or (env_key and os.environ.get(env_key))
        ):
            return
        if (
            not self.api_key
            or not self.api_key.strip()
            or self.api_key == cs.DEFAULT_API_KEY
        ):
            error_msg = format_missing_api_key_errors(self.provider, role)
            raise ValueError(error_msg)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider.

    Attributes:
        provider: Provider name (local, openai, google, ollama).
        model_id: Model identifier (e.g., "microsoft/unixcoder-base").
        dimension: Embedding dimension. None for auto-detect.
        api_key: API key for external providers.
        endpoint: Custom endpoint URL.
        keep_alive: Ollama model keep-alive duration.
        project_id: Google Cloud project ID (for Vertex AI).
        region: Google Cloud region.
        provider_type: Google provider type (gla/vertex).
        service_account_file: Path to Google service account JSON.
        device: Device for local models (auto/cpu/cuda).
        ssl_verify: SSL verification — True (default), False to disable, or path to CA bundle.
        proxy: Optional HTTP proxy URL for external API requests.
        fallback_to_local: Fall back to local embedding model on API failure.
        fallback_model: Local model to use when fallback is triggered.
    """

    provider: str
    model_id: str
    dimension: int | None = None
    api_key: str | None = None
    endpoint: str | None = None
    keep_alive: str | None = None
    project_id: str | None = None
    region: str | None = None
    provider_type: str | None = None
    service_account_file: str | None = None
    device: str | None = None
    ssl_verify: bool = True
    proxy: str | None = None
    fallback_to_local: bool = True
    fallback_model: str = "BAAI/bge-large-en-v1.5"

    def to_update_kwargs(self) -> EmbeddingConfigKwargs:
        result = asdict(self)
        del result[cs.FIELD_PROVIDER]
        del result[cs.FIELD_MODEL_ID]
        del result["dimension"]
        return EmbeddingConfigKwargs(**result)


@dataclass
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval.

    Weights control the contribution of each signal to the final combined score.
    All weights must sum to approximately 1.0.

    The text signal uses keyword-based matching (CONTAINS on name, qualified_name,
    and docstring) rather than full-text indexing, which is sufficient for
    enriching vector search results with explicit term matches.
    """

    vector_weight: float = 0.60
    text_weight: float = 0.15
    pagerank_weight: float = 0.20
    community_weight: float = 0.05
    top_k: int = 10
    max_context_depth: int = 2
    min_similarity_threshold: float = 0.1

    def __post_init__(self) -> None:
        """Validate that weights are in valid range and sum approximately to 1.0."""
        weights = [
            self.vector_weight,
            self.text_weight,
            self.pagerank_weight,
            self.community_weight,
        ]
        weight_names = [
            "vector_weight",
            "text_weight",
            "pagerank_weight",
            "community_weight",
        ]
        for i, w in enumerate(weights):
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"{weight_names[i]} must be between 0 and 1, got {w}")

        total = sum(weights)
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Hybrid weights must sum to 1.0, got {total:.3f} "
                f"(vector={self.vector_weight}, text={self.text_weight}, "
                f"pagerank={self.pagerank_weight}, community={self.community_weight})"
            )


@dataclass
class PathAnalysisConfig:
    """Configuration for path analysis."""

    max_paths: int = 3
    max_path_length: int = 10
    bottleneck_threshold: float = 0.01


@dataclass
class QFSConfig:
    """Configuration for query-focused summarization."""

    top_communities: int = 3
    min_community_size: int = 5
    summary_max_tokens: int = 500


class AppConfig(BaseSettings):
    """
    (H) All settings are loaded from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        # Note: Environment is loaded at module level via load_dotenv()
        # with support for ENV_FILE environment variable. This avoids
        # double-loading and ensures ENV_FILE takes precedence.
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
    )

    MEMGRAPH_HOST: str = "localhost"
    MEMGRAPH_PORT: int = 7687
    MEMGRAPH_HTTP_PORT: int = 7444
    MEMGRAPH_USERNAME: str | None = None
    MEMGRAPH_PASSWORD: str | None = None
    LAB_PORT: int = 3000
    MEMGRAPH_BATCH_SIZE: int = 1000
    MEMGRAPH_QUERY_MAX_RETRIES: int = Field(default=3, ge=0)
    MEMGRAPH_RETRY_BASE_DELAY: float = Field(default=0.25, gt=0)
    MEMGRAPH_CONNECTION_TIMEOUT: int = Field(default=600, gt=0)
    MEMGRAPH_QUERY_TIMEOUT: int = Field(default=120, gt=0)
    MEMGRAPH_USE_DYNAMIC_ALGORITHMS: bool | None = None
    AGENT_RETRIES: int = 3
    AGENT_REQUEST_LIMIT: int | None = None
    ORCHESTRATOR_OUTPUT_RETRIES: int = 100

    @property
    def memgraph(self) -> dict:
        """Memgraph configuration as a dict for easy access."""
        return {
            "host": self.MEMGRAPH_HOST,
            "port": self.MEMGRAPH_PORT,
            "http_port": self.MEMGRAPH_HTTP_PORT,
            "username": self.MEMGRAPH_USERNAME,
            "password": self.MEMGRAPH_PASSWORD,
            "batch_size": self.MEMGRAPH_BATCH_SIZE,
            "use_dynamic_algorithms": self.MEMGRAPH_USE_DYNAMIC_ALGORITHMS,
        }

    # Advanced Memgraph retrieval/algorithm configurations
    _hybrid_retrieval_config: HybridRetrievalConfig | None = None
    _path_analysis_config: PathAnalysisConfig | None = None
    _qfs_config: QFSConfig | None = None

    @property
    def hybrid_retrieval_config(self) -> HybridRetrievalConfig:
        """Get hybrid retrieval configuration instance aligned with VECTOR_SEARCH_TOP_K."""
        if not self._hybrid_retrieval_config:
            self._hybrid_retrieval_config = HybridRetrievalConfig(top_k=self.VECTOR_SEARCH_TOP_K)
        return self._hybrid_retrieval_config

    @property
    def path_analysis_config(self) -> PathAnalysisConfig:
        """Get path analysis configuration instance."""
        if not self._path_analysis_config:
            self._path_analysis_config = PathAnalysisConfig()
        return self._path_analysis_config

    @property
    def qfs_config(self) -> QFSConfig:
        """Get query-focused summarization configuration instance."""
        if not self._qfs_config:
            self._qfs_config = QFSConfig()
        return self._qfs_config

    ORCHESTRATOR_PROVIDER: str = ""
    ORCHESTRATOR_MODEL: str = ""
    ORCHESTRATOR_API_KEY: str | None = None
    ORCHESTRATOR_ENDPOINT: str | None = None
    ORCHESTRATOR_PROJECT_ID: str | None = None
    ORCHESTRATOR_REGION: str = cs.DEFAULT_REGION
    ORCHESTRATOR_PROVIDER_TYPE: cs.GoogleProviderType | None = None
    ORCHESTRATOR_THINKING_BUDGET: int | None = None
    ORCHESTRATOR_SERVICE_ACCOUNT_FILE: str | None = None

    CYPHER_PROVIDER: str = ""
    CYPHER_MODEL: str = ""
    CYPHER_API_KEY: str | None = None
    CYPHER_ENDPOINT: str | None = None
    CYPHER_PROJECT_ID: str | None = None
    CYPHER_REGION: str = cs.DEFAULT_REGION
    CYPHER_PROVIDER_TYPE: cs.GoogleProviderType | None = None
    CYPHER_THINKING_BUDGET: int | None = None
    CYPHER_SERVICE_ACCOUNT_FILE: str | None = None

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    @property
    def ollama_endpoint(self) -> str:
        return f"{self.OLLAMA_BASE_URL.rstrip('/')}/v1"

    TARGET_REPO_PATH: str = "."
    SHELL_COMMAND_TIMEOUT: int = 30
    PYTHON_INSPECT_TIMEOUT: int = Field(default=10, gt=0)
    SHELL_COMMAND_ALLOWLIST: frozenset[str] = frozenset(
        {
            "ls",
            "rg",
            "cat",
            "git",
            "echo",
            "pwd",
            "pytest",
            "mypy",
            "ruff",
            "uv",
            "find",
            "pre-commit",
            "rm",
            "cp",
            "mv",
            "mkdir",
            "rmdir",
            "wc",
            "head",
            "tail",
            "sort",
            "uniq",
            "cut",
            "tr",
            "xargs",
            "awk",
            "sed",
            "tee",
        }
    )
    SHELL_READ_ONLY_COMMANDS: frozenset[str] = frozenset(
        {
            "ls",
            "cat",
            "find",
            "pwd",
            "rg",
            "echo",
            "wc",
            "head",
            "tail",
            "sort",
            "uniq",
            "cut",
            "tr",
        }
    )
    SHELL_SAFE_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
        {
            "status",
            "log",
            "diff",
            "show",
            "ls-files",
            "remote",
            "config",
            "branch",
        }
    )

    # Embedding cache (backend-agnostic)
    EMBEDDING_CACHE_DIR: str = "./.embedding_cache"

    # Vector backend setting retained for compatibility; only Memgraph is supported
    VECTOR_STORE_BACKEND: str = "memgraph"

    # Memgraph native vector settings
    MEMGRAPH_VECTOR_INDEX_NAME: str = "code_embeddings"
    MEMGRAPH_VECTOR_DIM: int = 768  # Must match embedding model
    MEMGRAPH_VECTOR_CAPACITY: int = 100000  # REQUIRED - estimate ~2x function count
    MEMGRAPH_VECTOR_METRIC: str = "cos"  # Options: l2sq, cos, ip, pearson
    MEMGRAPH_VECTOR_SCALAR_KIND: str = "f32"  # Options: f32, f64, f16, bf16, f8

    # Graph Algorithm Configuration
    ALGORITHM_RUN_POST_INGESTION: bool = True
    ALGORITHM_ENABLE_PAGERANK: bool = True
    ALGORITHM_ENABLE_COMMUNITY_DETECTION: bool = True
    ALGORITHM_COMMUNITY_ALGORITHM: str = "leiden"

    # Unified vector settings
    VECTOR_SEARCH_TOP_K: int = 5
    VECTOR_EMBEDDING_BATCH_SIZE: int = 50
    VECTOR_MIN_SIMILARITY: float = 0.0

    # Embedding provider configuration
    EMBEDDING_PROVIDER: str = "local"  # Options: local, openai, google, ollama
    EMBEDDING_MODEL: str = "microsoft/unixcoder-base"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_ENDPOINT: str | None = None  # Custom endpoint or base URL
    EMBEDDING_BASE_URL: str | None = (
        None  # Alias for EMBEDDING_ENDPOINT (OpenAI-compatible APIs, root or /embeddings URL)
    )
    EMBEDDING_KEEP_ALIVE: str | None = (
        None  # Ollama: keep model loaded duration (e.g., "5m")
    )
    EMBEDDING_PROJECT_ID: str | None = None  # Google Vertex AI
    EMBEDDING_REGION: str = "us-central1"
    EMBEDDING_PROVIDER_TYPE: str | None = None  # Google: gla/vertex
    EMBEDDING_SERVICE_ACCOUNT_FILE: str | None = None  # Google service account
    EMBEDDING_DEVICE: str = "auto"  # Local: auto/cpu/cuda
    EMBEDDING_SSL_VERIFY: bool = True
    EMBEDDING_PROXY: str | None = None
    EMBEDDING_FALLBACK_TO_LOCAL: bool = True
    EMBEDDING_FALLBACK_MODEL: str = "BAAI/bge-large-en-v1.5"

    EMBEDDING_MAX_LENGTH: int = 512
    EMBEDDING_PROGRESS_INTERVAL: int = 10

    # Embedding chunking strategy
    EMBEDDING_CHUNKING_STRATEGY: Literal[
        "truncate", "chunk", "hierarchical", "error"
    ] = "chunk"
    EMBEDDING_CHUNK_OVERLAP_TOKENS: int = 32
    EMBEDDING_MAX_CHUNKS_PER_NODE: int = 5
    EMBEDDING_SKIP_BINARY_FILES: bool = True

    FLUSH_THREAD_POOL_SIZE: int = Field(default=4, gt=0)
    FILE_FLUSH_INTERVAL: int = Field(default=500, gt=0)

    # Parallel indexing settings
    PARALLEL_INDEXING_WORKERS: int = Field(default=30, gt=0)
    """Number of parallel workers to use for codebase indexing.
    Auto-optimized at runtime: will not exceed available CPU cores or number of changed files.
    Set to 1 to disable parallel processing entirely (sequential mode)."""

    INDEXING_WORKER_TIMEOUT: int = Field(default=3600, gt=0)
    """Timeout in seconds for worker processes during indexing.
    Prevents indefinite hangs when processing large files or stuck workers.
    Default: 3600 (1 hour). Increase for very large codebases."""

    RUN_INGESTION_QUALITY_CHECKS: bool = True
    """Whether to run post-ingestion data quality validation checks after indexing completes."""

    INCLUDE_BUILTIN_CALLS: bool = False
    """Whether to include CALLS edges to built-in functions.
    When False (default), calls to builtin functions (qualified_name starting with 'builtin.')
    are filtered out during ingestion to reduce graph noise.
    Set to True for complete call graph analysis."""

    MAX_MISSING_EMBEDDINGS_PCT: float = Field(default=2.0, gt=0, lt=100)
    """Maximum allowed percentage of nodes missing embeddings before quality check fails."""

    CACHE_MAX_ENTRIES: int = 1000
    CACHE_MAX_MEMORY_MB: int = 500
    CACHE_EVICTION_DIVISOR: int = 10
    CACHE_MEMORY_THRESHOLD_RATIO: float = 0.8

    # ─────────────────────────────────────────────────────────
    # DOCUMENT GRAPHRAG (NEW)
    # ─────────────────────────────────────────────────────────
    DOC_MEMGRAPH_HOST: str = "localhost"
    DOC_MEMGRAPH_PORT: int = 7688
    DOC_MEMGRAPH_HTTP_PORT: int = 7445
    DOC_MEMGRAPH_USERNAME: str | None = None
    DOC_MEMGRAPH_PASSWORD: str | None = None
    DOC_MEMGRAPH_BATCH_SIZE: int = 1000
    DOC_MEMGRAPH_VECTOR_DIM: int = 768
    DOC_MEMGRAPH_USE_DYNAMIC_ALGORITHMS: bool | None = None
    DOC_MEMGRAPH_MEMORY_LIMIT: str = "4GB"  # Memory limit for document graph container
    DOC_MEMGRAPH_CONNECTION_TIMEOUT: int = Field(default=600, gt=0)
    DOC_MAX_CHUNKS_PER_DOCUMENT: int = (
        5000  # Maximum chunks per document to prevent memory exhaustion
    )
    DOC_LAB_PORT: int = 3001  # Memgraph Lab for document graph
    DOC_VECTOR_STORE_BACKEND: str = "memgraph"

    @property
    def doc_memgraph(self) -> dict:
        """Document Memgraph configuration as a dict for easy access."""
        return {
            "host": self.DOC_MEMGRAPH_HOST,
            "port": self.DOC_MEMGRAPH_PORT,
            "http_port": self.DOC_MEMGRAPH_HTTP_PORT,
            "username": self.DOC_MEMGRAPH_USERNAME,
            "password": self.DOC_MEMGRAPH_PASSWORD,
            "batch_size": self.DOC_MEMGRAPH_BATCH_SIZE,
            "use_dynamic_algorithms": self.DOC_MEMGRAPH_USE_DYNAMIC_ALGORITHMS,
        }

    DOC_SUPPORTED_EXTENSIONS: list[str] = Field(
        default=[".md", ".rst", ".txt", ".pdf", ".docx"]
    )
    DOC_ENABLE_PDF_EXTRACTION: bool = True
    DOC_MAX_FILE_SIZE_MB: int = Field(default=50, gt=0)
    DOC_EXTRACTION_TIMEOUT_SECONDS: int = Field(default=30, gt=0)
    DOC_ENABLED: bool = True  # Master switch for document features

    # Document vector settings
    DOC_MEMGRAPH_VECTOR_INDEX_NAME: str = "doc_embeddings"
    DOC_MEMGRAPH_VECTOR_CAPACITY: int = 100000
    DOC_VECTOR_SEARCH_TOP_K: int = 5

    # ─────────────────────────────────────────────────────────
    # JSON GRAPHRAG (NEW)
    # ─────────────────────────────────────────────────────────
    JSON_MEMGRAPH_HOST: str = "localhost"
    JSON_MEMGRAPH_PORT: int = 7689
    JSON_MEMGRAPH_HTTP_PORT: int = 7446
    JSON_MEMGRAPH_USERNAME: str | None = None
    JSON_MEMGRAPH_PASSWORD: str | None = None
    JSON_MEMGRAPH_BATCH_SIZE: int = 1000
    JSON_MEMGRAPH_VECTOR_DIM: int = 768
    JSON_MEMGRAPH_USE_DYNAMIC_ALGORITHMS: bool | None = None
    JSON_MEMGRAPH_MEMORY_LIMIT: str = "2GB"  # Memory limit for JSON graph container
    JSON_MEMGRAPH_CONNECTION_TIMEOUT: int = Field(default=600, gt=0)
    JSON_LAB_PORT: int = 3002  # Memgraph Lab for JSON graph
    JSON_VECTOR_STORE_BACKEND: str = "memgraph"
    JSON_ENABLED: bool = True  # Master switch for JSON features

    @property
    def json_memgraph(self) -> dict:
        """JSON Memgraph configuration as a dict for easy access."""
        return {
            "host": self.JSON_MEMGRAPH_HOST,
            "port": self.JSON_MEMGRAPH_PORT,
            "http_port": self.JSON_MEMGRAPH_HTTP_PORT,
            "username": self.JSON_MEMGRAPH_USERNAME,
            "password": self.JSON_MEMGRAPH_PASSWORD,
            "batch_size": self.JSON_MEMGRAPH_BATCH_SIZE,
            "use_dynamic_algorithms": self.JSON_MEMGRAPH_USE_DYNAMIC_ALGORITHMS,
        }

    # JSON vector settings
    JSON_MEMGRAPH_VECTOR_INDEX_NAME: str = "json_embeddings"
    JSON_MEMGRAPH_VECTOR_CAPACITY: int = 100000
    JSON_VECTOR_SEARCH_TOP_K: int = 5
    # JSON ingestion parallel worker count (round-robin connection pool)
    JSON_PARALLEL_WORKERS: int = 10

    # Real-time updater (extended)
    REALTIME_DEBOUNCE_SECONDS: float = Field(
        default=5.0, gt=0,
        validation_alias=AliasChoices("CGR_REALTIME_DEBOUNCE", "CGR_REALTIME_DEBOUNCE_SECONDS")
    )
    REALTIME_MAX_WAIT_SECONDS: float = Field(
        default=30.0, gt=0,
        validation_alias=AliasChoices("CGR_REALTIME_MAX_WAIT", "CGR_REALTIME_MAX_WAIT_SECONDS")
    )
    REALTIME_BATCH_SIZE: int = Field(default=100, gt=0, validation_alias="CGR_REALTIME_BATCH_SIZE")
    REALTIME_UPDATER_ENABLED: bool = Field(default=False, validation_alias="CGR_REALTIME_UPDATER")
    REALTIME_CODE_ENABLED: bool = Field(default=True, validation_alias="CGR_REALTIME_CODE")
    REALTIME_DOCS_ENABLED: bool = Field(default=False, validation_alias="CGR_REALTIME_DOCS")
    REALTIME_JSON_ENABLED: bool = Field(default=False, validation_alias="CGR_REALTIME_JSON")
    REALTIME_IGNORE_OPENED_EVENTS: bool = Field(default=True, validation_alias="CGR_REALTIME_IGNORE_OPENED")
    REALTIME_LOG_SKIPPED_DOCUMENTS: bool = Field(default=False, validation_alias="CGR_REALTIME_LOG_SKIPPED")

    QUERY_RESULT_MAX_TOKENS: int = Field(default=16000, gt=0)
    QUERY_RESULT_ROW_CAP: int = Field(default=500, gt=0)
    QUERY_RESULT_TRUNCATION_STRATEGY: Literal["fifo", "relevance", "balanced"] = (
        "balanced"
    )
    QUERY_RESULT_MAX_ROW_TOKENS: int = 2000
    QUERY_RESULT_MIN_ROWS: int = 5
    QUERY_RESULT_DIVERSITY_BUDGET_PCT: float = 0.15

    # ─────────────────────────────────────────────────────────
    # Query Method Optimization Configuration
    # ─────────────────────────────────────────────────────────
    # These settings control the enhanced query orchestrator:
    # - Intent classification confidence threshold
    # - Minimum methods required for sufficient results
    # - Circuit breaker for failing methods
    # - Graph data integrity verification

    QUERY_INTENT_CONFIDENCE_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias="CGR_QUERY_INTENT_CONFIDENCE_THRESHOLD",
    )
    """Minimum confidence score for intent-based method selection.
    Below this threshold, all available methods are executed."""

    QUERY_MIN_METHODS: int = Field(
        default=2,
        ge=1,
        le=6,
        validation_alias="CGR_QUERY_MIN_METHODS",
    )
    """Minimum number of successful methods required before early termination."""

    QUERY_CIRCUIT_BREAKER_THRESHOLD: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias="CGR_QUERY_CIRCUIT_BREAKER_THRESHOLD",
    )
    """Number of consecutive failures before a method is temporarily disabled."""

    QUERY_INTEGRITY_CHECK_COUNT: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias="CGR_QUERY_INTEGRITY_CHECK_COUNT",
    )
    """Number of top results to spot-check for graph-to-disk integrity."""

    QUERY_ENABLE_INTEGRITY_CHECK: bool = Field(
        default=True,
        validation_alias="CGR_QUERY_ENABLE_INTEGRITY_CHECK",
    )
    """Enable graph data integrity verification (file existence, line bounds)."""

    QUERY_TEMPLATE_FALLBACK_ENABLED: bool = Field(
        default=True,
        validation_alias="CGR_QUERY_TEMPLATE_FALLBACK_ENABLED",
    )
    """Enable template-based Cypher fallback when LLM generation fails."""

    # Visibility and logging
    LOG_TRUNCATION_DETAILS: bool = True
    RETURN_TRUNCATION_METADATA: bool = True
    LOG_QUALITY_CHECK_STACKTRACES: bool = False
    """Whether to log full stack traces for quality check failures.
    Enable for debugging complex issues."""

    OLLAMA_HEALTH_TIMEOUT: float = 5.0

    _active_orchestrator: ModelConfig | None = None
    _active_cypher: ModelConfig | None = None
    _active_embedding: EmbeddingConfig | None = None
    _active_worker_llms: list[ModelConfig] | None = None

    QUIET: bool = Field(False, validation_alias="CGR_QUIET")

    # Logging configuration
    LOG_LEVEL: str = Field("INFO", validation_alias="CGR_LOG_LEVEL")

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in allowed_levels:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of: {', '.join(allowed_levels)}"
            )
        return upper_v

    # Yolo mode via environment (for MCP server and persistent settings)
    CGR_YOLO_MODE: bool = False

    # Global file access configuration
    ENABLE_GLOBAL_FILE_ACCESS: bool = True
    GLOBAL_FILE_ACCESS_WRITE_REQUIRES_APPROVAL: bool = True

    MCP_HTTP_HOST: str = "127.0.0.1"
    MCP_HTTP_PORT: int = 8080
    MCP_HTTP_ENDPOINT_PATH: str = "/mcp"

    # Parallel Sub-Agent Configuration
    CGR_MAX_PARALLEL_WORKERS: int = 30
    CGR_DEFAULT_PARALLEL_WORKERS: int = 20
    CGR_ALLOW_DYNAMIC_MAX_OVERRIDE: bool = True
    CGR_AUTO_SCALE_WORKERS: bool = True
    CGR_SUBAGENT_TIMEOUT: int = 300
    CGR_SUBAGENT_ALLOW_WRITE: bool = False
    CGR_AUTO_SPLIT_ENABLED: bool = True
    CGR_SUBAGENT_RETRY_ATTEMPTS: int = 2

    # Automatic Concurrency Detection (no explicit user request needed)
    CGR_AUTO_PARALLEL_ENABLED: bool = True
    CGR_PARALLEL_ELIGIBILITY_THRESHOLD: float = 0.6
    CGR_SIMPLE_TASK_MODEL: str | None = None
    CGR_AGGREGATION_DEDUPLICATION_ENABLED: bool = True
    CGR_PARALLEL_METRICS_ENABLED: bool = True
    CGR_PARALLEL_MAX_QUEUE_SIZE: int = 200

    # ─────────────────────────────────────────────────────────
    # Enhanced Parallel Execution Configuration (NEW)
    # ─────────────────────────────────────────────────────────

    # Strict write detection mode: when True, uses the current broad regex patterns
    # (more false positives, more conservative). When False (default), uses the
    # enhanced Layer 1-4 detection with contextual understanding.
    CGR_PARALLEL_WRITE_DETECTION_STRICT: bool = False

    # Minimum subtasks required to justify parallel execution overhead
    CGR_PARALLEL_MIN_SUBTASKS: int = Field(default=2, gt=0)

    # Enable adaptive threshold adjustment based on historical parallel execution success rates
    # When True, replaces the existing confidence-multiplier calibration with threshold-based adjustment
    CGR_PARALLEL_ADAPTIVE_THRESHOLD: bool = True

    # Enable file type hint extraction for better task splitting
    # When True, _extract_file_type_hints() is used in _collect_scoped_files() Strategy 2
    CGR_PARALLEL_FILE_TYPE_HINTS: bool = True

    # Include codebase context (file count, languages, repo size) in LLM eligibility analysis
    # When True, codebase stats are computed once per agent initialization and included in the system prompt
    CGR_PARALLEL_CODEBASE_CONTEXT: bool = True

    # Worker LLM Configuration for Sub-Agents
    CGR_WORKER_LLMS: str | list[str | dict] = Field(default_factory=list)
    CGR_WORKER_LLM_ASSIGNMENT_STRATEGY: Literal["round-robin"] = "round-robin"

    # Context Window Management Configuration
    DEFAULT_CONTEXT_WINDOW: int = Field(default=256000, gt=0)
    ORCHESTRATOR_CONTEXT_WINDOW: int | None = Field(default=None, gt=0)
    CYPHER_CONTEXT_WINDOW: int | None = Field(default=None, gt=0)

    # Context Window Compression Configuration
    CONTEXT_COMPRESSION_AUTO_TRIGGER_PCT: float = Field(
        default=75.0, gt=0, lt=100,
        description="Trigger compression when context reaches this % of max_context (reduced from 85% for safety)"
    )
    CONTEXT_COMPRESSION_HYSTERESIS_PCT: float = Field(default=5.0, gt=0, lt=20)
    CONTEXT_COMPRESSION_MIN_RETENTION_SCORE: float = Field(default=70.0, gt=0, lt=100)
    CONTEXT_COMPRESSION_PARALLEL_WORKERS: int = Field(default=10, gt=0)
    CONTEXT_COMPRESSION_AGGRESSIVE_RETENTION_THRESHOLD: float = Field(
        default=50.0, gt=0, lt=100
    )
    CONTEXT_COMPRESSION_ARCHIVE_TTL_HOURS: int = Field(default=24, gt=0)
    CONTEXT_COMPRESSION_ENABLED: bool = True
    CONTEXT_COMPRESSION_SYSTEM_RESERVE_PCT: float = Field(
        default=20.0, gt=0, lt=50,
        description="Percentage of context window to reserve for system prompt, tools, and response buffer"
    )

    # Semantic Compression Configuration
    SEMANTIC_COMPRESSION_ENABLED: bool = True
    SEMANTIC_COMPRESSION_MODEL_ROLE: Literal["orchestrator", "cypher"] = "orchestrator"
    SEMANTIC_COMPRESSION_MAX_RECENT_MESSAGES: int = Field(default=4, ge=1, le=10)
    SEMANTIC_COMPRESSION_TARGET_PCT: float = Field(default=75.0, gt=0, lt=100)
    SEMANTIC_COMPRESSION_TIMEOUT_SECONDS: int = Field(default=30, gt=0)
    SEMANTIC_COMPRESSION_FALLBACK_ON_ERROR: bool = True
    SEMANTIC_COMPRESSION_VERBATIM_BUDGET_PCT: float = Field(default=40.0, gt=0, lt=80)
    SEMANTIC_COMPRESSION_LLM_INPUT_CAP: int = Field(default=50000, gt=1000)
    SEMANTIC_COMPRESSION_PRESERVE_VERBATIM_ROLES: frozenset[str] = Field(
        default_factory=lambda: frozenset({"system", "tool", "function"})
    )

    # ─────────────────────────────────────────────────────────
    # Model Management Feature Configuration (Phase 3 placeholders)
    # ─────────────────────────────────────────────────────────

    # Custom model catalog file path (JSON/YAML). When set, the catalog is loaded from this file
    # instead of the static MODEL_CATALOG. Must be validated as a valid file path at startup if provided.
    CGR_MODEL_CATALOG_PATH: str | None = None

    # Disable dynamic model discovery (Ollama querying, provider API integration).
    # Only the static MODEL_CATALOG is used.
    CGR_DISABLE_MODEL_DISCOVERY: bool = False

    def _get_default_config(self, role: str) -> ModelConfig:
        role_upper = role.upper()

        provider = getattr(self, f"{role_upper}_PROVIDER", None)
        model = getattr(self, f"{role_upper}_MODEL", None)

        if provider and model:
            return ModelConfig(
                provider=provider.lower(),
                model_id=model,
                api_key=getattr(self, f"{role_upper}_API_KEY", None),
                endpoint=getattr(self, f"{role_upper}_ENDPOINT", None),
                project_id=getattr(self, f"{role_upper}_PROJECT_ID", None),
                region=getattr(self, f"{role_upper}_REGION", cs.DEFAULT_REGION),
                provider_type=getattr(self, f"{role_upper}_PROVIDER_TYPE", None),
                thinking_budget=getattr(self, f"{role_upper}_THINKING_BUDGET", None),
                service_account_file=getattr(
                    self, f"{role_upper}_SERVICE_ACCOUNT_FILE", None
                ),
            )

        return ModelConfig(
            provider=cs.Provider.OLLAMA,
            model_id=cs.DEFAULT_MODEL,
            endpoint=self.ollama_endpoint,
            api_key=cs.DEFAULT_API_KEY,
        )

    def _get_default_orchestrator_config(self) -> ModelConfig:
        return self._get_default_config(cs.ModelRole.ORCHESTRATOR)

    def _get_default_cypher_config(self) -> ModelConfig:
        return self._get_default_config(cs.ModelRole.CYPHER)

    def _get_model_config_for_provider(self, provider: str, model: str) -> ModelConfig:
        """
        Get a ModelConfig instance for a given provider and model, using default settings for the provider.

        Args:
            provider: LLM provider name
            model: Model ID

        Returns:
            Populated ModelConfig object
        """
        provider_lower = provider.lower()

        # Look for provider-specific default configs
        api_key = None
        endpoint = None
        project_id = None
        region = cs.DEFAULT_REGION
        provider_type = None
        thinking_budget = None
        service_account_file = None

        # Check for provider-specific API keys from environment
        if provider_lower == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", cs.DEFAULT_API_KEY)
        elif provider_lower == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", cs.DEFAULT_API_KEY)
        elif provider_lower == "google":
            api_key = os.environ.get("GOOGLE_API_KEY", cs.DEFAULT_API_KEY)
        elif provider_lower == "azure":
            api_key = os.environ.get("AZURE_API_KEY", cs.DEFAULT_API_KEY)
        elif provider_lower == "ollama":
            endpoint = self.ollama_endpoint
            api_key = cs.DEFAULT_API_KEY

        return ModelConfig(
            provider=provider_lower,
            model_id=model,
            api_key=api_key,
            endpoint=endpoint,
            project_id=project_id,
            region=region,
            provider_type=provider_type,
            thinking_budget=thinking_budget,
            service_account_file=service_account_file,
        )

    @property
    def active_worker_llms(self) -> list[ModelConfig]:
        """Get the list of validated active worker LLMs for sub-agents."""
        if self._active_worker_llms is not None:
            return self._active_worker_llms

        # Parse from CGR_WORKER_LLMS config
        worker_llms_config = self.CGR_WORKER_LLMS
        parsed_llms: list[ModelConfig] = []

        if isinstance(worker_llms_config, str):
            config_str = worker_llms_config.strip()
            # Strip surrounding quotes that may come from .env file
            if (config_str.startswith("'") and config_str.endswith("'")) or \
               (config_str.startswith('"') and config_str.endswith('"')):
                config_str = config_str[1:-1]
            if config_str:
                # Check if it's a JSON array or object
                if config_str.startswith("[") or config_str.startswith("{"):
                    logger.debug("Detected JSON format for CGR_WORKER_LLMS")
                    try:
                        import json
                        import re

                        # Strip comments from JSON (lines starting with #)
                        # Handle both single-line comments and inline comments
                        lines = config_str.split("\n")
                        cleaned_lines = []
                        for line in lines:
                            # Remove full-line comments and inline comments
                            # Match # not inside a string
                            cleaned = re.sub(r'(?<!\")\s*#.*$', '', line)
                            if cleaned.strip():
                                cleaned_lines.append(cleaned)
                        cleaned_json = "\n".join(cleaned_lines)

                        # Remove trailing commas before ] or } (left behind after comment removal)
                        cleaned_json = re.sub(r',(\s*[\]\}])', r'\1', cleaned_json)

                        # Debug: log the cleaned JSON before parsing
                        logger.debug(f"Cleaned JSON for CGR_WORKER_LLMS: {cleaned_json[:200]}...")

                        parsed = json.loads(cleaned_json)
                        if isinstance(parsed, list):
                            for entry in parsed:
                                if isinstance(entry, str):
                                    provider, model = self.parse_model_string(entry)
                                    parsed_llms.append(
                                        self._get_model_config_for_provider(
                                            provider, model
                                        )
                                    )
                                elif isinstance(entry, dict):
                                    parsed_llms.append(_model_config_from_mapping(entry))
                        elif isinstance(parsed, dict):
                            # Single object wrapped in braces
                            parsed_llms.append(_model_config_from_mapping(parsed))
                    except json.JSONDecodeError as e:
                        # Invalid JSON, fall through to comma-separated parsing
                        logger.warning(
                            f"CGR_WORKER_LLMS JSON parsing failed: {e}. "
                            "Falling back to comma-separated format."
                        )
                        # Debug: show the problematic JSON around the error position
                        if e.lineno and e.lineno <= len(cleaned_lines):
                            start = max(0, e.lineno - 2)
                            end = min(len(cleaned_lines), e.lineno + 1)
                            context = "\n".join(f"  {i+1}: {cleaned_lines[i]}" for i in range(start, end))
                            logger.debug(f"JSON context around line {e.lineno}:\n{context}")
                else:
                    # Split comma-separated list (e.g., "openai:gpt-4o,anthropic:claude-3")
                    entries = [
                        entry.strip()
                        for entry in worker_llms_config.split(",")
                        if entry.strip()
                    ]
                    for entry in entries:
                        provider, model = self.parse_model_string(entry)
                        parsed_llms.append(
                            self._get_model_config_for_provider(provider, model)
                        )
        elif isinstance(worker_llms_config, list):
            for entry in worker_llms_config:
                if isinstance(entry, str):
                    provider, model = self.parse_model_string(entry)
                    parsed_llms.append(
                        self._get_model_config_for_provider(provider, model)
                    )
                elif isinstance(entry, dict):
                    # Full ModelConfig dict
                    parsed_llms.append(_model_config_from_mapping(entry))

        # Validate all parsed LLMs
        valid_llms = []
        for llm_config in parsed_llms:
            try:
                llm_config.validate_api_key(role="worker")
                valid_llms.append(llm_config)
            except ValueError as e:
                logger.warning(f"Skipping invalid worker LLM config: {str(e)}")

        self._active_worker_llms = valid_llms
        return valid_llms

    def set_worker_llms(self, llms: list[str | ModelConfig | dict]) -> None:
        """
        Dynamically set the list of worker LLMs for sub-agents.

        Args:
            llms: List of LLM configurations, either in provider:model string format,
                  ModelConfig objects, or ModelConfig dictionaries
        """
        parsed_llms: list[ModelConfig] = []

        for llm_entry in llms:
            if isinstance(llm_entry, ModelConfig):
                parsed_llms.append(llm_entry)
            elif isinstance(llm_entry, str):
                provider, model = self.parse_model_string(llm_entry)
                parsed_llms.append(self._get_model_config_for_provider(provider, model))
            elif isinstance(llm_entry, dict):
                parsed_llms.append(_model_config_from_mapping(llm_entry))

        # Validate all configs
        valid_llms = []
        for llm_config in parsed_llms:
            try:
                llm_config.validate_api_key(role="worker")
                valid_llms.append(llm_config)
            except ValueError as e:
                logger.warning(f"Skipping invalid worker LLM config: {str(e)}")

        self._active_worker_llms = valid_llms if valid_llms else None
        if valid_llms:
            logger.info(f"Set {len(valid_llms)} worker LLMs successfully")
        else:
            logger.info(
                "No valid worker LLMs configured, falling back to orchestrator LLM"
            )

    @property
    def active_orchestrator_config(self) -> ModelConfig:
        return self._active_orchestrator or self._get_default_orchestrator_config()

    @property
    def active_cypher_config(self) -> ModelConfig:
        return self._active_cypher or self._get_default_cypher_config()

    def set_orchestrator(
        self, provider: str, model: str, **kwargs: Unpack[ModelConfigKwargs]
    ) -> None:
        config = ModelConfig(provider=provider.lower(), model_id=model, **kwargs)
        self._active_orchestrator = config

    def set_cypher(
        self, provider: str, model: str, **kwargs: Unpack[ModelConfigKwargs]
    ) -> None:
        config = ModelConfig(provider=provider.lower(), model_id=model, **kwargs)
        self._active_cypher = config

    def _get_default_embedding_config(self) -> EmbeddingConfig:
        """Get default embedding configuration from environment."""
        return EmbeddingConfig(
            provider=self.EMBEDDING_PROVIDER.lower(),
            model_id=self.EMBEDDING_MODEL,
            api_key=self._get_effective_embedding_api_key(),
            endpoint=self._get_effective_embedding_endpoint(),
            keep_alive=self.EMBEDDING_KEEP_ALIVE,
            project_id=self.EMBEDDING_PROJECT_ID,
            region=self.EMBEDDING_REGION,
            provider_type=self.EMBEDDING_PROVIDER_TYPE,
            service_account_file=self.EMBEDDING_SERVICE_ACCOUNT_FILE,
            device=self.EMBEDDING_DEVICE,
            ssl_verify=self.EMBEDDING_SSL_VERIFY,
            proxy=self.EMBEDDING_PROXY,
            fallback_to_local=self.EMBEDDING_FALLBACK_TO_LOCAL,
            fallback_model=self.EMBEDDING_FALLBACK_MODEL,
        )

    @property
    def active_embedding_config(self) -> EmbeddingConfig:
        """Get the active embedding configuration."""
        return self._active_embedding or self._get_default_embedding_config()

    def set_embedding(
        self,
        provider: str,
        model_id: str,
        dimension: int | None = None,
        **kwargs: Unpack[EmbeddingConfigKwargs],
    ) -> None:
        """Set the active embedding provider configuration."""
        config = EmbeddingConfig(
            provider=provider.lower(),
            model_id=model_id,
            dimension=dimension,
            **kwargs,
        )
        self._active_embedding = config

    def _get_effective_embedding_api_key(self) -> str | None:
        """Resolve embedding API key with fallback hierarchy.

        Fallback order:
        1. EMBEDDING_API_KEY (embedding-specific key)
        2. {PROVIDER}_API_KEY (provider-specific key)
        3. OPENAI_API_KEY (legacy fallback for OpenAI-compatible)

        Returns:
            Resolved API key or None.
        """
        # 1. Embedding-specific key takes precedence
        if self.EMBEDDING_API_KEY:
            return self.EMBEDDING_API_KEY

        # 2. Provider-specific key
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        provider_lower = self.EMBEDDING_PROVIDER.lower()
        if provider_lower in provider_key_map:
            key = os.environ.get(provider_key_map[provider_lower])
            if key:
                return key

        # 3. OpenAI fallback (for OpenAI-compatible APIs)
        return os.environ.get("OPENAI_API_KEY")

    def _get_effective_embedding_endpoint(self) -> str | None:
        """Resolve embedding endpoint with fallback.

        Fallback order:
        1. EMBEDDING_ENDPOINT (embedding-specific endpoint)
        2. EMBEDDING_BASE_URL (alias for OpenAI-compatible APIs)
        3. OLLAMA_BASE_URL (for Ollama provider)

        Returns:
            Resolved endpoint URL or None.
        """
        if self.EMBEDDING_ENDPOINT:
            return self.EMBEDDING_ENDPOINT

        if self.EMBEDDING_BASE_URL:
            return self.EMBEDDING_BASE_URL

        if self.EMBEDDING_PROVIDER.lower() == "ollama":
            return self.ollama_endpoint

        return None

    def _get_model_dimension(self, provider: str, model_id: str) -> int:
        """Get dimension for model with provider context.

        Args:
            provider: Provider name (openai, google, ollama, local).
            model_id: Model identifier.

        Returns:
            Known dimension or default (768) if unknown.
        """
        return cs.EMBEDDING_MODEL_DIMENSIONS.get(model_id, 768)

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        """Alias for get_effective_vector_dim("code") for backward compatibility."""
        return self.get_effective_vector_dim("code")

    @property
    def DOC_EMBEDDING_DIMENSION(self) -> int:
        """Effective embedding dimension for document graph (follows standard precedence logic)."""
        return self.get_effective_vector_dim("document")

    @property
    def JSON_EMBEDDING_DIMENSION(self) -> int:
        """Effective embedding dimension for JSON graph (follows standard precedence logic)."""
        return self.get_effective_vector_dim("json")

    def get_effective_vector_dim(self, graph_type: str = "code") -> int:
        """Return effective dimension with proper precedence for specified graph type.

        Args:
            graph_type: One of "code", "document", "json"

        Consistent precedence order for all graph types:
            1. <GRAPH_TYPE>_MEMGRAPH_VECTOR_DIM if set (explicit override)
            2. Auto-detect from embedding model (shared across all graphs)

        Returns:
            Effective vector dimension for the requested graph type.
        """
        # Map graph type to corresponding env var (consistent naming convention)
        env_var_map = {
            "code": "MEMGRAPH_VECTOR_DIM",
            "document": "DOC_MEMGRAPH_VECTOR_DIM",
            "json": "JSON_MEMGRAPH_VECTOR_DIM",
        }
        if graph_type not in env_var_map:
            raise ValueError(
                f"Invalid graph_type: {graph_type}. Must be one of {list(env_var_map.keys())}"
            )

        # Check for explicit override first (same logic for all graph types)
        env_dim = os.environ.get(env_var_map[graph_type])
        if env_dim:
            try:
                return int(env_dim)
            except ValueError:
                pass

        # Auto-detect from model as universal fallback for all graph types
        return self._get_model_dimension(
            self.EMBEDDING_PROVIDER,
            self.EMBEDDING_MODEL,
        )

    def parse_model_string(self, model_string: str) -> tuple[str, str]:
        if ":" not in model_string:
            return cs.Provider.OLLAMA, model_string
        provider, model = model_string.split(":", 1)
        if not provider:
            raise ValueError(ex.PROVIDER_EMPTY)
        return provider.lower(), model

    def resolve_batch_size(self, batch_size: int | None) -> int:
        resolved = self.MEMGRAPH_BATCH_SIZE if batch_size is None else batch_size
        if resolved < 1:
            raise ValueError(ex.BATCH_SIZE_POSITIVE)
        return resolved

    @field_validator("EMBEDDING_MAX_LENGTH")
    @classmethod
    def validate_embedding_max_length(cls, v: int) -> int:
        """Validate EMBEDDING_MAX_LENGTH against UniXcoder context limit."""
        max_context = cs.UNIXCODER_MAX_CONTEXT - 4  # Reserve for special tokens
        if v > max_context:
            raise ValueError(
                f"EMBEDDING_MAX_LENGTH ({v}) must be <= {max_context} "
                f"(UNIXCODER_MAX_CONTEXT - 4 for special tokens)"
            )
        if v < 64:
            raise ValueError(
                f"EMBEDDING_MAX_LENGTH ({v}) must be >= 64 for meaningful embeddings"
            )
        return v

    @field_validator("EMBEDDING_CHUNK_OVERLAP_TOKENS")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Validate overlap is less than max length."""
        max_length = info.data.get("EMBEDDING_MAX_LENGTH", 512)
        if v >= max_length:
            raise ValueError(
                f"EMBEDDING_CHUNK_OVERLAP_TOKENS ({v}) must be < "
                f"EMBEDDING_MAX_LENGTH ({max_length})"
            )
        if v < 0:
            raise ValueError(f"EMBEDDING_CHUNK_OVERLAP_TOKENS ({v}) must be >= 0")
        return v

    @field_validator("QUERY_RESULT_MAX_ROW_TOKENS")
    @classmethod
    def validate_max_row_tokens(cls, v: int, info) -> int:
        """Validate max row tokens is less than max tokens."""
        max_tokens = info.data.get("QUERY_RESULT_MAX_TOKENS", 16000)
        if v >= max_tokens:
            raise ValueError(
                f"QUERY_RESULT_MAX_ROW_TOKENS ({v}) must be < "
                f"QUERY_RESULT_MAX_TOKENS ({max_tokens})"
            )
        if v < 100:
            raise ValueError(
                f"QUERY_RESULT_MAX_ROW_TOKENS ({v}) must be >= 100 for useful results"
            )
        return v

    @field_validator("QUERY_RESULT_DIVERSITY_BUDGET_PCT")
    @classmethod
    def validate_diversity_budget(cls, v: float) -> float:
        """Validate diversity budget is in valid range."""
        if not 0 <= v <= 0.5:
            raise ValueError(
                f"QUERY_RESULT_DIVERSITY_BUDGET_PCT ({v}) must be between 0 and 0.5"
            )
        return v

    @field_validator("QUERY_RESULT_MIN_ROWS")
    @classmethod
    def validate_min_rows(cls, v: int) -> int:
        """Validate min rows is reasonable."""
        if v < 1:
            raise ValueError(f"QUERY_RESULT_MIN_ROWS ({v}) must be >= 1")
        if v > 50:
            raise ValueError(f"QUERY_RESULT_MIN_ROWS ({v}) must be <= 50")
        return v

    @field_validator("DOC_SUPPORTED_EXTENSIONS", mode="before")
    @classmethod
    def parse_doc_extensions(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated or JSON array env var to list."""
        if isinstance(v, str):
            # Try JSON parse first (pydantic-settings 2.x uses JSON for list fields)
            if v.startswith("["):
                import json

                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [ext.strip().lower() for ext in parsed if ext.strip()]
                except json.JSONDecodeError:
                    pass
            # Fallback to comma-separated
            return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
        return v

    @staticmethod
    def _validate_memgraph_only_backend(value: str, field_name: str) -> str:
        backend = value.lower()
        if backend != "memgraph":
            raise ValueError(
                f"{field_name}={backend!r} is not supported. Only 'memgraph' is available."
            )
        return backend

    @field_validator(
        "VECTOR_STORE_BACKEND",
        "DOC_VECTOR_STORE_BACKEND",
        "JSON_VECTOR_STORE_BACKEND",
    )
    @classmethod
    def validate_vector_backend(cls, v: str, info) -> str:
        """Validate vector backend configuration matches the runtime backend."""
        return cls._validate_memgraph_only_backend(v, info.field_name)

    @field_validator("DOC_MAX_FILE_SIZE_MB")
    @classmethod
    def validate_doc_max_file_size(cls, v: int) -> int:
        """Validate max file size is reasonable."""
        if v < 1:
            raise ValueError(f"DOC_MAX_FILE_SIZE_MB ({v}) must be >= 1")
        if v > 500:
            raise ValueError(f"DOC_MAX_FILE_SIZE_MB ({v}) must be <= 500")
        return v


settings = AppConfig()

CGRIGNORE_FILENAME = ".cgrignore"


EMPTY_CGRIGNORE = CgrignorePatterns(exclude=frozenset(), unignore=frozenset())


def load_cgrignore_patterns(repo_path: Path) -> CgrignorePatterns:
    ignore_file = repo_path / CGRIGNORE_FILENAME
    if not ignore_file.is_file():
        return EMPTY_CGRIGNORE

    exclude: set[str] = set()
    unignore: set[str] = set()
    try:
        with ignore_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("!"):
                    unignore.add(line[1:].strip())
                else:
                    exclude.add(line)
        if exclude or unignore:
            logger.info(
                logs.CGRIGNORE_LOADED.format(
                    exclude_count=len(exclude),
                    unignore_count=len(unignore),
                    path=ignore_file,
                )
            )
        return CgrignorePatterns(
            exclude=frozenset(exclude),
            unignore=frozenset(unignore),
        )
    except OSError as e:
        logger.warning(logs.CGRIGNORE_READ_FAILED.format(path=ignore_file, error=e))
        return EMPTY_CGRIGNORE
