from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict, Unpack

from dotenv import load_dotenv
from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import constants as cs
from . import exceptions as ex
from . import logs
from .types_defs import CgrignorePatterns, EmbeddingConfigKwargs, ModelConfigKwargs

load_dotenv()


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

    def to_update_kwargs(self) -> EmbeddingConfigKwargs:
        result = asdict(self)
        del result[cs.FIELD_PROVIDER]
        del result[cs.FIELD_MODEL_ID]
        del result["dimension"]
        return EmbeddingConfigKwargs(**result)


class AppConfig(BaseSettings):
    """
    (H) All settings are loaded from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    MEMGRAPH_HOST: str = "localhost"
    MEMGRAPH_PORT: int = 7687
    MEMGRAPH_HTTP_PORT: int = 7444
    MEMGRAPH_USERNAME: str | None = None
    MEMGRAPH_PASSWORD: str | None = None
    LAB_PORT: int = 3000
    MEMGRAPH_BATCH_SIZE: int = 1000
    AGENT_RETRIES: int = 3
    ORCHESTRATOR_OUTPUT_RETRIES: int = 100

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

    QDRANT_DB_PATH: str = "./.qdrant_code_embeddings"  # Legacy: used only when backend=qdrant
    QDRANT_COLLECTION_NAME: str = "code_embeddings"
    QDRANT_VECTOR_DIM: int = 768
    QDRANT_TOP_K: int = 5
    QDRANT_UPSERT_RETRIES: int = Field(default=3, gt=0)
    QDRANT_RETRY_BASE_DELAY: float = Field(default=0.5, gt=0)
    QDRANT_URI: str | None = None
    QDRANT_BATCH_SIZE: int = Field(default=50, gt=0)

    # Vector backend selection
    VECTOR_STORE_BACKEND: str = "memgraph"  # Options: "memgraph" (default), "qdrant"

    # Memgraph native vector settings
    MEMGRAPH_VECTOR_INDEX_NAME: str = "code_embeddings"
    MEMGRAPH_VECTOR_DIM: int = 768  # Must match embedding model
    MEMGRAPH_VECTOR_CAPACITY: int = 100000  # REQUIRED - estimate ~2x function count
    MEMGRAPH_VECTOR_METRIC: str = "cos"  # Options: l2sq, cos, ip, pearson
    MEMGRAPH_VECTOR_SCALAR_KIND: str = "f32"  # Options: f32, f64, f16, bf16, f8

    # Unified vector settings
    VECTOR_SEARCH_TOP_K: int = 5
    VECTOR_EMBEDDING_BATCH_SIZE: int = 50
    VECTOR_MIN_SIMILARITY: float = 0.0

    # Embedding provider configuration
    EMBEDDING_PROVIDER: str = "local"  # Options: local, openai, google, ollama
    EMBEDDING_MODEL: str = "microsoft/unixcoder-base"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_ENDPOINT: str | None = None  # Custom endpoint URL
    EMBEDDING_BASE_URL: str | None = None  # Alias for EMBEDDING_ENDPOINT (OpenAI-compatible APIs)
    EMBEDDING_KEEP_ALIVE: str | None = None  # Ollama: keep model loaded duration (e.g., "5m")
    EMBEDDING_PROJECT_ID: str | None = None  # Google Vertex AI
    EMBEDDING_REGION: str = "us-central1"
    EMBEDDING_PROVIDER_TYPE: str | None = None  # Google: gla/vertex
    EMBEDDING_SERVICE_ACCOUNT_FILE: str | None = None  # Google service account
    EMBEDDING_DEVICE: str = "auto"  # Local: auto/cpu/cuda

    EMBEDDING_MAX_LENGTH: int = 512
    EMBEDDING_PROGRESS_INTERVAL: int = 10

    # Embedding chunking strategy
    EMBEDDING_CHUNKING_STRATEGY: Literal["truncate", "chunk", "hierarchical", "error"] = "chunk"
    EMBEDDING_CHUNK_OVERLAP_TOKENS: int = 32
    EMBEDDING_MAX_CHUNKS_PER_NODE: int = 5
    EMBEDDING_SKIP_BINARY_FILES: bool = True

    FLUSH_THREAD_POOL_SIZE: int = Field(default=4, gt=0)
    FILE_FLUSH_INTERVAL: int = Field(default=500, gt=0)

    CACHE_MAX_ENTRIES: int = 1000
    CACHE_MAX_MEMORY_MB: int = 500
    CACHE_EVICTION_DIVISOR: int = 10
    CACHE_MEMORY_THRESHOLD_RATIO: float = 0.8

    QUERY_RESULT_MAX_TOKENS: int = Field(default=16000, gt=0)
    QUERY_RESULT_ROW_CAP: int = Field(default=500, gt=0)
    QUERY_RESULT_TRUNCATION_STRATEGY: Literal["fifo", "relevance", "balanced"] = "balanced"
    QUERY_RESULT_MAX_ROW_TOKENS: int = 2000
    QUERY_RESULT_MIN_ROWS: int = 5
    QUERY_RESULT_DIVERSITY_BUDGET_PCT: float = 0.15

    # Visibility and logging
    LOG_TRUNCATION_DETAILS: bool = True
    RETURN_TRUNCATION_METADATA: bool = True

    OLLAMA_HEALTH_TIMEOUT: float = 5.0

    _active_orchestrator: ModelConfig | None = None
    _active_cypher: ModelConfig | None = None
    _active_embedding: EmbeddingConfig | None = None

    QUIET: bool = Field(False, validation_alias="CGR_QUIET")

    MCP_HTTP_HOST: str = "0.0.0.0"
    MCP_HTTP_PORT: int = 8080
    MCP_HTTP_ENDPOINT_PATH: str = "/mcp"

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

    def get_effective_vector_dim(self) -> int:
        """Return effective dimension with proper precedence.

        Precedence order:
        1. MEMGRAPH_VECTOR_DIM if set (explicit override)
        2. Auto-detect from embedding model

        Returns:
            Effective vector dimension.
        """
        # If explicitly set via env/config, use that
        # Check if it's set to something other than default
        env_dim = os.environ.get("MEMGRAPH_VECTOR_DIM")
        if env_dim:
            try:
                return int(env_dim)
            except ValueError:
                pass

        # Auto-detect from model
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
            raise ValueError(
                f"EMBEDDING_CHUNK_OVERLAP_TOKENS ({v}) must be >= 0"
            )
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
