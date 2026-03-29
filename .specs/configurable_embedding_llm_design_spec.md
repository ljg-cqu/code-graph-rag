# Configurable Embedding LLM Design Specification

> **Status**: Draft - Under Review (Round 2)
> **Created**: 2026-03-29
> **Location**: `.specs/configurable_embedding_llm_design_spec.md`

---

## Overview

This specification defines the architecture for making Code-Graph-RAG's embedding system configurable, supporting multiple embedding providers (OpenAI, Google, Ollama, local transformers) instead of the hardcoded UniXcoder model.

---

## Current Architecture

### Embedding System (Hardcoded)

```
embedder.py
├── UniXcoder (microsoft/unixcoder-base) - FIXED
├── 768-dimensional vectors - FIXED
├── torch + transformers dependency - REQUIRED
└── EmbeddingCache - disk-based JSON cache
```

### LLM Provider System (Configurable, NOT Used for Embeddings)

```
providers/base.py
├── ModelProvider (abstract base)
│   ├── GoogleProvider → GoogleModel
│   ├── OpenAIProvider → OpenAIChatModel
│   ├── AnthropicProvider → AnthropicModel
│   ├── OllamaProvider → OpenAIChatModel
│   └── AzureOpenAIProvider → OpenAIChatModel
├── ModelConfig (dataclass)
└── PROVIDER_REGISTRY (dict)
```

### Configuration

```
config.py (AppConfig)
├── ORCHESTRATOR_PROVIDER/MODEL - for RAG orchestration
├── CYPHER_PROVIDER/MODEL - for Cypher generation
├── MEMGRAPH_VECTOR_DIM: int = 768 - hardcoded to UniXcoder
├── EMBEDDING_CACHE_DIR - cache location
└── NO EMBEDDING_PROVIDER/MODEL - MISSING
```

---

## Problems with Current Approach

1. **No flexibility**: Users cannot choose embedding models
2. **Heavy dependency**: torch + transformers required for embeddings
3. **No API options**: Cannot use OpenAI/Google embedding APIs
4. **Fixed dimensions**: 768-dim hardcoded, incompatible with other models
5. **Local-only**: No remote embedding service support

---

## Proposed Architecture

### 1. EmbeddingProvider Abstract Base

```python
# codebase_rag/embeddings/base.py

from abc import ABC, abstractmethod
from .. import constants as cs

class EmbeddingProvider(ABC):
    """Abstract base for embedding providers.

    Design Divergence from LLM Providers:
    Unlike the LLM provider system which uses a create_model() factory method
    to instantiate model objects, embedding providers directly implement embed()
    methods. This is because:
    1. Embedding is a single operation, not a stateful conversation
    2. No need for separate model instances - providers are stateless
    3. Simpler API: get_provider() -> embed() vs get_provider() -> create_model() -> embed()
    4. Batching is handled internally by embed_batch() rather than separate model instances
    """

    __slots__ = ("config", "model_id", "dimension")

    def __init__(self, model_id: str, dimension: int, **config: str | int | None) -> None:
        self.config = config
        self.model_id = model_id
        self.dimension = dimension

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Chunking Behavior:
        When len(texts) > batch_size, this method automatically chunks the input
        into multiple batches of size batch_size (last batch may be smaller).
        Each chunk is processed sequentially with rate limiting applied between
        chunks. The returned list maintains the same order as the input texts.

        Example:
            texts = ["a", "b", "c", "d", "e"] with batch_size=2
            -> Processes ["a", "b"], then ["c", "d"], then ["e"]
            -> Returns [[emb_a], [emb_b], [emb_c], [emb_d], [emb_e]]

        Args:
            texts: List of strings to embed.
            batch_size: Maximum number of texts per API call. Defaults to 32.

        Returns:
            List of embeddings in the same order as input texts.
        """
        ...

    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider configuration."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> cs.EmbeddingProvider:
        """Return provider identifier as EmbeddingProvider enum."""
        ...
```

### 2. Provider Implementations

#### OpenAI Embeddings

```python
# codebase_rag/embeddings/openai.py

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding models."""

    SUPPORTED_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    MAX_BATCH_SIZE = 2048  # OpenAI's actual limit
    MAX_TOKENS_PER_INPUT = 8191

    def __init__(
        self,
        model_id: str = "text-embedding-3-small",
        api_key: str | None = None,
        endpoint: str | None = None,
        dimensions: int | None = None,  # For text-embedding-3-* models
        truncate_strategy: str = "end",
    ) -> None:
        dimension = dimensions or self.SUPPORTED_MODELS.get(model_id, 1536)
        super().__init__(model_id, dimension, api_key=api_key, endpoint=endpoint)
        ...
```

#### Google Embeddings

```python
# codebase_rag/embeddings/google.py

from typing import Literal
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.impersonated_credentials import Credentials as ImpersonatedCredentials

class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding models via GLA (Generative Language API) or Vertex AI."""

    # GLA models (API key authentication)
    GLA_MODELS = {
        "text-embedding-004": 768,
        "embedding-001": 768,
    }

    # Vertex AI models (ADC/project authentication)
    VERTEX_MODELS = {
        "text-embedding-004": 768,
        "text-embedding-005": 3072,  # Verified: text-embedding-005 uses 3072 dimensions
    }

    BATCH_LIMITS = {
        "gla": {"max_texts": 100, "max_tokens_per_text": 20000},
        "vertex": {"max_texts": 250, "max_tokens_per_text": 30000},
    }

    # Required scopes for service account credentials
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    def __init__(
        self,
        model_id: str = "text-embedding-004",
        api_key: str | None = None,
        project_id: str | None = None,
        region: str = "us-central1",
        provider_type: Literal["gla", "vertex"] = "gla",
        # Advanced auth options
        credentials: ga_credentials.Credentials | None = None,
        service_account_file: str | None = None,
        impersonate_service_account: str | None = None,
        quota_project: str | None = None,
    ) -> None:
        # Determine dimension based on model and provider type
        dimension = self._get_model_dimension(model_id, provider_type)
        super().__init__(model_id, dimension, api_key=api_key, project_id=project_id,
                         region=region, provider_type=provider_type,
                         credentials=credentials, service_account_file=service_account_file,
                         impersonate_service_account=impersonate_service_account,
                         quota_project=quota_project)
        self._client = None

    def _get_model_dimension(self, model_id: str, provider_type: str) -> int:
        """Get model dimension based on provider type."""
        if provider_type == "gla":
            return self.GLA_MODELS.get(model_id, 768)
        else:
            return self.VERTEX_MODELS.get(model_id, 768)

    def validate_config(self) -> None:
        """
        Validate provider configuration.

        GLA: Requires API key ONLY (no project_id, no ADC)
        Vertex: Requires ADC/project credentials (API key not supported)
        """
        if self.provider_type == "gla":
            if not self.api_key:
                raise EmbeddingAuthenticationError(
                    "GLA provider requires EMBEDDING_API_KEY or GOOGLE_API_KEY. "
                    "Set via: export EMBEDDING_API_KEY=your-api-key",
                    provider="google",
                    model=self.model_id,
                )
            # GLA does not use project_id - warn if provided
            if self.project_id:
                import warnings
                warnings.warn(
                    "GLA provider does not use project_id. "
                    "For Vertex AI, set EMBEDDING_PROVIDER_TYPE=vertex",
                    UserWarning,
                )
        else:  # vertex
            # Vertex requires project credentials via ADC, service account, or explicit credentials
            has_auth = (
                self.credentials is not None
                or self.service_account_file is not None
                or self.impersonate_service_account is not None
                or self._has_adc()
            )
            if not has_auth:
                raise EmbeddingAuthenticationError(
                    "Vertex AI provider requires authentication. Options:\n"
                    "  1. Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json\n"
                    "  2. Set EMBEDDING_SERVICE_ACCOUNT_FILE=/path/to/service-account.json\n"
                    "  3. Set EMBEDDING_IMPERSONATE_SERVICE_ACCOUNT=service-account@project.iam\n"
                    "  4. Run `gcloud auth application-default login` for ADC\n"
                    "Also set EMBEDDING_PROJECT_ID=your-gcp-project",
                    provider="google",
                    model=self.model_id,
                )
            if not self.project_id:
                raise EmbeddingAuthenticationError(
                    "Vertex AI requires EMBEDDING_PROJECT_ID to be set.",
                    provider="google",
                    model=self.model_id,
                )

    def _has_adc(self) -> bool:
        """Check if Application Default Credentials are available."""
        try:
            import google.auth
            google.auth.default()
            return True
        except google.auth.exceptions.DefaultCredentialsError:
            return False

    def _get_credentials(self) -> ga_credentials.Credentials:
        """Get credentials with ADC fallback. Applies to Vertex only."""
        creds: ga_credentials.Credentials | None = None

        # Priority 1: Explicitly provided credentials
        if self.credentials:
            creds = self.credentials

        # Priority 2: Service account file
        elif self.service_account_file:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_file,
                scopes=self.SCOPES,
            )
            # Apply quota project to service account credentials
            if self.quota_project or self.project_id:
                creds = creds.with_quota_project(self.quota_project or self.project_id)

        # Priority 3: Impersonate service account
        elif self.impersonate_service_account:
            creds = self._create_impersonated_credentials()

        # Priority 4: ADC (Application Default Credentials)
        else:
            creds, _ = google.auth.default(
                quota_project_id=self.quota_project or self.project_id,
                scopes=self.SCOPES,
            )

        return creds

    def _create_impersonated_credentials(self) -> ImpersonatedCredentials:
        """Create impersonated credentials for a target service account."""
        # Get source credentials (ADC or service account)
        if self.service_account_file:
            from google.oauth2 import service_account
            source_creds = service_account.Credentials.from_service_account_file(
                self.service_account_file,
                scopes=self.SCOPES,
            )
        else:
            # Use ADC as source
            source_creds, _ = google.auth.default(scopes=self.SCOPES)

        # Create impersonated credentials
        target_principal = self.impersonate_service_account
        return ImpersonatedCredentials(
            source_credentials=source_creds,
            target_principal=target_principal,
            target_scopes=self.SCOPES,
            quota_project_id=self.quota_project or self.project_id,
        )

    def _create_client(self):
        """Create the appropriate client based on provider type."""
        if self.provider_type == "gla":
            return self._create_gla_client()
        else:
            return self._create_vertex_client()

    def _create_gla_client(self):
        """Create GLA client using Generative Language API."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        return genai

    def _create_vertex_client(self):
        """Create Vertex AI client with credentials."""
        from vertexai import init as vertex_init
        from vertexai.language_models import TextEmbeddingModel

        creds = self._get_credentials()
        vertex_init(
            project=self.project_id,
            location=self.region,
            credentials=creds,
        )
        return TextEmbeddingModel.from_pretrained(self.model_id)

    @property
    def provider_name(self) -> str:
        return "google"

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if self._client is None:
            self._client = self._create_client()

        if self.provider_type == "gla":
            result = self._client.embed_content(
                model=f"models/{self.model_id}",
                content=text,
            )
            return result["embedding"]
        else:
            embeddings = self._client.get_embeddings([text])
            return embeddings[0].values

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if self._client is None:
            self._client = self._create_client()

        limits = self.BATCH_LIMITS[self.provider_type]
        max_texts = limits["max_texts"]

        if len(texts) > max_texts:
            raise BatchSizeExceededError(
                f"{self.provider_type.upper()} supports max {max_texts} texts per batch, "
                f"got {len(texts)}"
            )

        if self.provider_type == "gla":
            result = self._client.embed_content(
                model=f"models/{self.model_id}",
                content=texts,
            )
            return result["embedding"]
        else:
            embeddings = self._client.get_embeddings(texts)
            return [e.values for e in embeddings]
```

#### Ollama Embeddings

```python
# codebase_rag/embeddings/ollama.py

import httpx
from ..exceptions import EmbeddingModelNotFoundError, EmbeddingConnectionError

class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding models."""

    KNOWN_MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        model_id: str = "nomic-embed-text",
        endpoint: str = "http://localhost:11434",
        keep_alive: str | None = None,
    ) -> None:
        # CRITICAL: Initialize endpoint FIRST before any API calls
        self._endpoint = endpoint.rstrip("/")
        self._keep_alive = keep_alive

        # Detect dimension using initialized endpoint
        if model_id in self.KNOWN_MODEL_DIMENSIONS:
            dimension = self.KNOWN_MODEL_DIMENSIONS[model_id]
        else:
            # Probe with test embedding (requires self._endpoint to be set)
            dimension = self._probe_dimension(model_id)

        super().__init__(model_id, dimension, endpoint=endpoint, keep_alive=keep_alive)

    def _probe_dimension(self, model_id: str) -> int:
        """Probe model dimension by generating a test embedding.

        Raises:
            EmbeddingModelNotFoundError: If model is not pulled in Ollama.
            EmbeddingConnectionError: If Ollama is not reachable.
        """
        try:
            embedding = self._raw_embed("probe")
            return len(embedding)
        except EmbeddingModelNotFoundError:
            raise
        except Exception as e:
            # Check if it's a model not found error
            if self._is_model_not_found(str(e)):
                raise EmbeddingModelNotFoundError(
                    f"Model '{model_id}' not found in Ollama. Pull it first: ollama pull {model_id}",
                    provider="ollama",
                    model=model_id,
                )
            raise EmbeddingConnectionError(
                f"Failed to connect to Ollama at {self._endpoint}: {e}",
                provider="ollama",
                model=model_id,
            )

    def _is_model_not_found(self, error_message: str) -> bool:
        """Check if error indicates model not found."""
        return "not found" in error_message.lower() or "404" in error_message

    def _build_payload(self, input_data: str | list[str]) -> dict:
        """Build API payload, conditionally omitting keep_alive if None."""
        payload = {
            "model": self.model_id,
            "input": input_data,
        }
        # CRITICAL: Only include keep_alive if explicitly set
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        return payload

    def _raw_embed(self, text: str) -> list[float]:
        """Generate embedding without validation (used for probing)."""
        response = self._call_api("/api/embed", self._build_payload(text))
        return response["embeddings"][0]

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self._call_api("/api/embed", self._build_payload(text))
        return response["embeddings"][0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts using /api/embed batch support.

        Ollama's /api/embed endpoint natively supports batch input via list.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._call_api("/api/embed", self._build_payload(batch))
            all_embeddings.extend(response["embeddings"])

        return all_embeddings

    def _call_api(self, path: str, payload: dict) -> dict:
        """Call Ollama API with error handling."""
        url = f"{self._endpoint}{path}"
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_body = e.response.text
                if "not found" in error_body.lower() or "model" in error_body.lower():
                    raise EmbeddingModelNotFoundError(
                        f"Model '{self.model_id}' not found in Ollama. Pull it first: ollama pull {self.model_id}",
                        provider="ollama",
                        model=self.model_id,
                    )
            raise
        except httpx.ConnectError as e:
            raise EmbeddingConnectionError(
                f"Cannot connect to Ollama at {self._endpoint}. "
                f"Ensure Ollama is running: ollama serve",
                provider="ollama",
                model=self.model_id,
            )
```

#### Local Transformers (UniXcoder - Existing)

```python
# codebase_rag/embeddings/local.py

class LocalEmbeddingProvider(EmbeddingProvider):
    """Local transformers-based embedding (UniXcoder, etc.)."""

    SUPPORTED_MODELS = {
        "microsoft/unixcoder-base": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(
        self,
        model_id: str = "microsoft/unixcoder-base",
        device: str = "auto",  # auto, cpu, cuda
    ) -> None:
        dimension = self.SUPPORTED_MODELS.get(model_id, 768)
        super().__init__(model_id, dimension, device=device)
        ...
```

### 3. EmbeddingConfig Dataclass

```python
# codebase_rag/config.py (additions)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider."""

    provider: str  # openai, google, ollama, local
    model_id: str
    dimension: int | None = None  # None = auto-detect from model/provider
    api_key: str | None = None
    endpoint: str | None = None
    keep_alive: str | None = None  # for Ollama: keep model loaded duration
    project_id: str | None = None
    region: str | None = None
    provider_type: str | None = None  # for Google: gla/vertex
    service_account_file: str | None = None
    device: str | None = None  # for local: auto/cpu/cuda
```

### 4. AppConfig Extensions

```python
# codebase_rag/config.py (additions to AppConfig)

class AppConfig(BaseSettings):
    # ... existing fields ...

    # Embedding configuration (NEW)
    EMBEDDING_PROVIDER: str = "local"  # local, openai, google, ollama
    EMBEDDING_MODEL: str = "microsoft/unixcoder-base"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_ENDPOINT: str | None = None
    EMBEDDING_KEEP_ALIVE: str | None = None  # Ollama: keep model loaded duration (e.g., "5m")
    EMBEDDING_PROJECT_ID: str | None = None
    EMBEDDING_REGION: str = "us-central1"
    EMBEDDING_PROVIDER_TYPE: str | None = None  # gla/vertex
    EMBEDDING_SERVICE_ACCOUNT_FILE: str | None = None  # Google service account
    EMBEDDING_DEVICE: str = "auto"  # auto/cpu/cuda (for local provider)

    # Vector dimension configuration
    # NOTE: In-memory mutation of MEMGRAPH_VECTOR_DIM is non-persistent.
    # To make dimension changes permanent, set via environment variable.
    MEMGRAPH_VECTOR_DIM: int | None = None  # None = auto-detect from model
    QDRANT_VECTOR_DIM: int | None = None  # DEPRECATED: Use VECTOR_DIMENSION instead

    # Unified vector dimension (recommended for new configurations)
    VECTOR_DIMENSION: int | None = None  # Unified across all backends

    _active_embedding: EmbeddingConfig | None = None

    @property
    def active_embedding_config(self) -> EmbeddingConfig:
        return self._active_embedding or self._get_default_embedding_config()

    def _get_effective_api_key(self) -> str | None:
        """Resolve API key with fallback hierarchy.

        Fallback order:
        1. EMBEDDING_API_KEY (embedding-specific key)
        2. {PROVIDER}_API_KEY (provider-specific key, e.g., OPENAI_API_KEY, GOOGLE_API_KEY)
        3. OPENAI_API_KEY (legacy fallback for OpenAI-compatible providers)

        Returns:
            Resolved API key or None if not configured.
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
        provider_key_env = provider_key_map.get(self.EMBEDDING_PROVIDER)
        if provider_key_env:
            key = os.environ.get(provider_key_env)
            if key:
                return key

        # 3. OpenAI fallback (for OpenAI-compatible APIs)
        return os.environ.get("OPENAI_API_KEY")

    def _get_effective_ollama_endpoint(self) -> str:
        """Resolve Ollama endpoint with fallback.

        Fallback order:
        1. EMBEDDING_ENDPOINT (embedding-specific endpoint)
        2. OLLAMA_BASE_URL (Ollama CLI default)

        Returns:
            Resolved endpoint URL.
        """
        if self.EMBEDDING_ENDPOINT:
            return self.EMBEDDING_ENDPOINT
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def _get_model_dimension(self, provider: str, model_id: str) -> int:
        """Get dimension for model with provider context.

        Args:
            provider: Provider name (openai, google, ollama, local)
            model_id: Model identifier

        Returns:
            Known dimension or default (768) if unknown.
        """
        # Provider-specific dimension lookup
        provider_dimensions = {
            "openai": {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            },
            "google": {
                "text-embedding-004": 768,
                "text-embedding-005": 768,  # May be 3072, verify with Google docs
                "embedding-001": 768,
            },
            "ollama": {
                "nomic-embed-text": 768,
                "mxbai-embed-large": 1024,
                "all-minilm": 384,
                "snowflake-arctic-embed": 1024,
            },
            "local": {
                "microsoft/unixcoder-base": 768,
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-large-en-v1.5": 1024,
            },
        }

        dims = provider_dimensions.get(provider, {})
        return dims.get(model_id, 768)  # Default to 768 (UniXcoder)

    def get_effective_vector_dim(self) -> int:
        """Return effective dimension with proper precedence.

        Precedence order:
        1. VECTOR_DIMENSION (unified, recommended)
        2. MEMGRAPH_VECTOR_DIM or QDRANT_VECTOR_DIM (backend-specific)
        3. Auto-detect from embedding model (if None sentinel)

        NOTE: In-memory mutation of MEMGRAPH_VECTOR_DIM is non-persistent.
        Changes via settings.MEMGRAPH_VECTOR_DIM = X only last for the
        current process. For persistent changes, set the environment variable.

        Returns:
            Effective vector dimension.
        """
        # 1. Unified dimension takes precedence
        if self.VECTOR_DIMENSION is not None:
            return self.VECTOR_DIMENSION

        # 2. Backend-specific dimensions
        backend = os.environ.get("VECTOR_STORE_BACKEND", "memgraph")
        if backend == "memgraph" and self.MEMGRAPH_VECTOR_DIM is not None:
            return self.MEMGRAPH_VECTOR_DIM
        if backend == "qdrant" and self.QDRANT_VECTOR_DIM is not None:
            return self.QDRANT_VECTOR_DIM

        # 3. Auto-detect from model (None sentinel triggers detection)
        return self._get_model_dimension(
            self.EMBEDDING_PROVIDER,
            self.EMBEDDING_MODEL
        )
```

### 5. Provider Registry

```python
# codebase_rag/embeddings/__init__.py

# Add enum to align with existing Provider pattern
from ..constants import EmbeddingProvider as EmbeddingProviderEnum

EMBEDDING_PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    EmbeddingProviderEnum.LOCAL: LocalEmbeddingProvider,
    EmbeddingProviderEnum.OPENAI: OpenAIEmbeddingProvider,
    EmbeddingProviderEnum.GOOGLE: GoogleEmbeddingProvider,
    EmbeddingProviderEnum.OLLAMA: OllamaEmbeddingProvider,
}

def get_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Factory function to create embedding provider."""
    ...

def get_embedding_provider_from_config(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create provider from config dataclass."""
    ...
```

### 6. Updated embedder.py

```python
# codebase_rag/embedder.py (refactored)

_embedding_provider: EmbeddingProvider | None = None

def get_embedding_provider_instance() -> EmbeddingProvider:
    """Get or create embedding provider singleton."""
    global _embedding_provider
    if _embedding_provider is None:
        config = settings.active_embedding_config
        _embedding_provider = get_embedding_provider_from_config(config)
        _embedding_provider.validate_config()
    return _embedding_provider

def embed_code(code: str, max_length: int | None = None) -> list[float]:
    """Generate embedding using configured provider."""
    cache = get_embedding_cache()
    provider = get_embedding_provider_instance()

    # Cache key includes model_id
    cache_key = cache._content_hash(code, provider.model_id)
    if (cached := cache.get(cache_key)) is not None:
        return cached

    embedding = provider.embed(code)
    cache.put(cache_key, embedding)
    return embedding
```

### 7. Constants Updates

```python
# codebase_rag/constants.py (additions)

class EmbeddingProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"

EMBEDDING_MODEL_DIMENSIONS = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Google
    "text-embedding-004": 768,
    "embedding-001": 768,
    # Local
    "microsoft/unixcoder-base": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
}
```

---

## Cache Architecture (Updated)

### Problem with Original Design

Original spec proposed `sha256(model_id:content)` which breaks backward compatibility.

### Recommended Approach: Per-Model Cache Files

```python
import fcntl
import hashlib
import tempfile
from urllib.parse import quote
from pathlib import Path

class EmbeddingCache:
    """Per-model embedding cache with file locking and version compatibility."""

    __slots__ = ("_cache", "_path", "_model_id", "_dimension", "_dirty")
    CACHE_VERSION = 2
    LEGACY_VERSION = 1
    MIN_SUPPORTED_VERSION = 1

    def __init__(self, path: Path, model_id: str, dimension: int) -> None:
        self._cache: dict[str, list[float]] = {}
        self._path = path
        self._model_id = model_id
        self._dimension = dimension
        self._dirty: bool = False

    def _cache_file_for_model(self) -> Path:
        """Return model-specific cache file path with safe encoding.

        Uses URL encoding to handle all special characters safely.
        """
        # Use urllib.parse.quote for robust encoding of model_id
        # This handles /, :, -, . and all other special characters
        safe_model_id = quote(self._model_id, safe="")
        return self._path.parent / f"{safe_model_id}.json"

    def _content_hash(self, content: str, model_id: str) -> str:
        """Generate cache key hash for content + model_id combination.

        The hash ensures different models get different cache keys,
        preventing semantic incompatibility issues.

        Args:
            content: The text content to hash.
            model_id: The model identifier to include in hash.

        Returns:
            SHA256 hash string for use as cache key.
        """
        combined = f"{model_id}:{content}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def save(self) -> None:
        """Save cache to disk with atomic write and file locking.

        Uses atomic write (write to temp, then rename) to prevent
        corruption from concurrent access or process termination.
        """
        if not self._dirty and not self._cache:
            return

        path = self._cache_file_for_model()
        data = {
            "_metadata": {
                "model_id": self._model_id,
                "dimension": self._dimension,
                "version": self.CACHE_VERSION,
                "count": len(self._cache),
                "created_with_model": self._model_id,
            },
            "embeddings": self._cache,
        }
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.stem}.tmp",
            suffix=".json"
        )
        try:
            with open(fd, "w") as f:
                json.dump(data, f)
            # Atomic rename on POSIX systems
            Path(temp_path).replace(path)
            self._dirty = False
        except Exception:
            # Clean up temp file on failure
            Path(temp_path).unlink(missing_ok=True)
            raise

    def load(self) -> None:
        """Load cache from disk with version and model validation.

        Validates:
        - Cache version compatibility
        - Model ID match (semantic compatibility)
        - Dimension match (vector compatibility)

        On mismatch, the invalid cache file is deleted to prevent reuse.
        """
        path = self._cache_file_for_model()
        if not path.exists():
            # Check for legacy cache to migrate
            legacy_path = self._path  # original .embedding_cache.json
            if legacy_path.exists():
                self._migrate_legacy(legacy_path)
            return

        with path.open("r") as f:
            # Acquire shared lock for reading
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        meta = data.get("_metadata", {})

        # Version compatibility check
        cache_version = meta.get("version", self.LEGACY_VERSION)
        if cache_version < self.MIN_SUPPORTED_VERSION:
            logger.warning(
                f"Cache version {cache_version} too old (min: {self.MIN_SUPPORTED_VERSION}), "
                "deleting and starting fresh"
            )
            self._delete_cache_file(path)
            return

        if cache_version > self.CACHE_VERSION:
            logger.warning(
                f"Cache version {cache_version} newer than supported {self.CACHE_VERSION}, "
                "deleting to avoid compatibility issues"
            )
            self._delete_cache_file(path)
            return

        self._cache = data.get("embeddings", {})
```

---

## Dimension Handling (Updated)

### Startup Behavior

`initialize()` is called in the following scenarios:

1. **Server startup**: When MemgraphVectorStore is instantiated via `get_vector_store()`
2. **After index recreation**: When `recreate_vector_indexes()` completes
3. **After provider switch**: When `switch_embedding_provider()` triggers re-initialization

The initialization flow:
```
App Start -> get_vector_store() -> VectorStore.__init__() -> initialize()
                                                    |
                                        Check dimension consistency
                                                    |
                                        Raise DimensionMismatchError if mismatch
```

### Dimension Validation and Error Messages

```python
# vector_store_memgraph.py

def initialize(self) -> None:
    """Create vector indexes with dimension validation.

    Called automatically on startup. Raises DimensionMismatchError if
    existing embeddings have incompatible dimensions with configured provider.
    """
    dimension = settings.get_effective_vector_dim()

    # Check existing embeddings for dimension mismatch
    existing_dim = self._get_existing_embedding_dimension()
    if existing_dim is not None and existing_dim != dimension:
        raise DimensionMismatchError(
            existing_dim=existing_dim,
            configured_dim=dimension,
            message=self._format_dimension_mismatch_message(existing_dim, dimension),
        )

    # Create indexes if not exist
    self._create_vector_indexes(dimension)

    # Run consistency check
    if not self.validate_embedding_consistency():
        raise EmbeddingConsistencyError(
            "Embeddings are inconsistent or corrupted. Run reembed_all() to fix."
        )

def _format_dimension_mismatch_message(self, existing_dim: int, configured_dim: int) -> str:
    """Generate actionable error message with specific commands and warnings."""
    node_count = self._get_embedding_node_count()
    estimated_time = self._estimate_reembedding_time(node_count)
    cache_path = self._get_cache_file_path()

    return f"""
ERROR: Dimension Mismatch - Existing: {existing_dim}d, Configured: {configured_dim}d

Your embedding model produces {configured_dim}-dimensional vectors, but the database
contains {existing_dim}-dimensional embeddings from a previous model.

NODES AFFECTED: {node_count:,} nodes with embeddings
ESTIMATED RE-EMBEDDING TIME: {estimated_time}
CACHE FILE TO BE DELETED: {cache_path}

OPTIONS:

1. Use a model with {existing_dim} dimensions:
   export EMBEDDING_MODEL=<compatible_model>
   # See: codebase_rag constants EMBEDDING_MODEL_DIMENSIONS

2. Recreate indexes and re-embed (DESTRUCTIVE - clears all vectors):
   from codebase_rag import switch_embedding_provider
   switch_embedding_provider(
       provider="{settings.EMBEDDING_PROVIDER}",
       model_id="{settings.EMBEDDING_MODEL}",
       force=True  # Acknowledge data loss
   )
   # Time estimate: {estimated_time}

3. Manual step-by-step (if you want to backup first):
   # 1. Backup your data
   # 2. Drop indexes: DROP VECTOR INDEX function_embedding_index;
   # 3. Clear vectors: MATCH (n) SET n.embedding = NULL;
   # 4. Delete cache: rm {cache_path}
   # 5. Reinitialize: vector_store.initialize()
   # 6. Re-embed: reembed_all()

DATA LOSS WARNING: Option 2 and 3 will DELETE ALL EXISTING VECTORS.
The cache file will be DELETED (not just cleared).
"""

def _get_embedding_node_count(self) -> int:
    """Count nodes with embeddings for user feedback."""
    result = self._execute_query("""
        MATCH (n) WHERE n.embedding IS NOT NULL
        RETURN count(n) as count
    """)
    return result[0]["count"] if result else 0

def _estimate_reembedding_time(self, node_count: int) -> str:
    """Estimate re-embedding time based on node count.

    Rough estimates:
    - Local (UniXcoder): ~100 nodes/sec
    - OpenAI API: ~500 nodes/sec (with batching)
    - Google API: ~250 nodes/sec
    - Ollama: ~50 nodes/sec (depends on model)
    """
    rates = {
        "local": 100,
        "openai": 500,
        "google": 250,
        "ollama": 50,
    }
    rate = rates.get(settings.EMBEDDING_PROVIDER, 100)
    seconds = node_count / rate

    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes"
    else:
        return f"{seconds / 3600:.1f} hours"

def _get_cache_file_path(self) -> Path:
    """Return the cache file path for the current model."""
    safe_model_id = settings.EMBEDDING_MODEL.replace("/", "_").replace(":", "_")
    return Path(settings.EMBEDDING_CACHE_DIR) / f"{safe_model_id}.json"
```

### Existing Dimension Detection

```python
def _get_existing_embedding_dimension(self) -> int | None:
    """Detect dimension of existing embeddings with multiple fallbacks.

    Detection priority:
    1. Vector index metadata (fastest)
    2. Sample existing embedding from database
    3. Cache file metadata
    4. Return None if no embeddings exist

    Returns:
        Dimension int if embeddings exist, None if database is empty.
    """
    # Priority 1: Check vector index configuration
    index_dim = self._get_vector_index_dimension()
    if index_dim is not None:
        return index_dim

    # Priority 2: Sample an existing embedding
    sample_dim = self._sample_embedding_dimension()
    if sample_dim is not None:
        return sample_dim

    # Priority 3: Check cache file metadata
    cache_dim = self._get_cache_dimension()
    if cache_dim is not None:
        return cache_dim

    # No existing embeddings found
    return None

def _get_vector_index_dimension(self) -> int | None:
    """Get dimension from Memgraph vector index metadata."""
    try:
        # Try to get index info from Memgraph
        result = self._execute_query("""
            SHOW INDEX INFO
            WHERE index_type = 'VECTOR'
        """)
        if result:
            # Return the dimension of the first vector index
            return result[0].get("dimension")
    except Exception:
        # Memgraph version may not support SHOW INDEX INFO
        pass
    return None

def _sample_embedding_dimension(self) -> int | None:
    """Sample a single embedding to detect dimension."""
    result = self._execute_query("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN n.embedding as emb
        LIMIT 1
    """)
    if result and result[0].get("emb"):
        return len(result[0]["emb"])
    return None

def _get_cache_dimension(self) -> int | None:
    """Get dimension from cache file metadata."""
    cache_path = self._get_cache_file_path()
    if not cache_path.exists():
        # Check legacy cache location
        legacy_path = Path(settings.EMBEDDING_CACHE_DIR) / ".embedding_cache.json"
        if legacy_path.exists():
            return self._read_legacy_cache_dimension(legacy_path)
        return None

    try:
        with cache_path.open("r") as f:
            data = json.load(f)
        return data.get("_metadata", {}).get("dimension")
    except (json.JSONDecodeError, KeyError):
        return None

def _read_legacy_cache_dimension(self, path: Path) -> int | None:
    """Read dimension from legacy cache format."""
    # Legacy cache stored vectors directly without metadata
    # Assume 768 (original UniXcoder dimension) if legacy cache exists
    return 768 if path.exists() else None
```

### Consistency Validation

```python
def validate_embedding_consistency(self) -> bool:
    """Validate embedding consistency across the database.

    Checks:
    1. All embeddings have the same dimension
    2. Dimension matches configured provider
    3. No null/empty embeddings on indexed nodes

    Returns:
        True if consistent, False if issues detected.
    """
    configured_dim = settings.get_effective_vector_dim()

    # Check for dimension consistency
    result = self._execute_query("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        WITH n, size(n.embedding) as dim
        RETURN dim, count(n) as count
        ORDER BY count DESC
    """)

    if not result:
        # No embeddings, consistent by default
        return True

    # Check if all dimensions match
    dimensions = {r["dim"] for r in result}
    if len(dimensions) > 1:
        logger.warning(
            f"Inconsistent embedding dimensions found: {dimensions}. "
            f"Counts: {[(r['dim'], r['count']) for r in result]}"
        )
        return False

    # Check if dimension matches configuration
    actual_dim = result[0]["dim"]
    if actual_dim != configured_dim:
        logger.warning(
            f"Embedding dimension {actual_dim} does not match "
            f"configured dimension {configured_dim}"
        )
        return False

    return True
```

### Cache Deletion on Mismatch

```python
def _delete_cache_on_mismatch(self) -> None:
    """Delete cache file when dimension mismatch is detected.

    IMPORTANT: The cache file is DELETED, not just cleared in memory.
    This prevents stale cache data from causing issues after provider switch.
    """
    cache_path = self._get_cache_file_path()

    if cache_path.exists():
        try:
            cache_path.unlink()  # Delete the file
            logger.info(f"Deleted cache file due to dimension mismatch: {cache_path}")
        except OSError as e:
            logger.warning(f"Failed to delete cache file {cache_path}: {e}")

    # Also delete legacy cache if it exists
    legacy_path = Path(settings.EMBEDDING_CACHE_DIR) / ".embedding_cache.json"
    if legacy_path.exists():
        try:
            legacy_path.unlink()
            logger.info(f"Deleted legacy cache file: {legacy_path}")
        except OSError as e:
            logger.warning(f"Failed to delete legacy cache: {e}")
```

### Re-embedding Workflow

```python
def reembed_all(self, batch_size: int = 100) -> dict[str, int]:
    """Re-embed all nodes with current embedding provider.

    TIME COST DOCUMENTATION:
    - Local (UniXcoder): ~100 nodes/second
    - OpenAI API: ~500 nodes/second (batched, rate-limited)
    - Google API: ~250 nodes/second
    - Ollama: ~50 nodes/second

    For a codebase with 10,000 functions:
    - Local: ~100 seconds (~2 minutes)
    - OpenAI: ~20 seconds
    - Google: ~40 seconds
    - Ollama: ~200 seconds (~3 minutes)

    Args:
        batch_size: Number of nodes to process per batch. Defaults to 100.

    Returns:
        Dict with counts: {"total": N, "success": M, "failed": K}
    """
    from codebase_rag.embedder import get_embedding_provider_instance

    provider = get_embedding_provider_instance()
    configured_dim = provider.dimension

    # Clear existing embeddings first
    self._execute_query("""
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        SET n.embedding = NULL
    """)

    # Delete cache file
    self._delete_cache_on_mismatch()

    # Recreate indexes with correct dimension
    self.recreate_vector_indexes(configured_dim)

    # Get all nodes that need embedding
    nodes = self._execute_query("""
        MATCH (n)
        WHERE n:Function OR n:Method OR n:Class
        RETURN elementId(n) as id, n.source_code as source
    """)

    total = len(nodes)
    success = 0
    failed = 0

    logger.info(f"Starting re-embedding of {total} nodes with {settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}")

    for i in range(0, total, batch_size):
        batch = nodes[i:i + batch_size]
        sources = [n["source"] for n in batch if n["source"]]

        try:
            embeddings = provider.embed_batch(sources)

            for node, embedding in zip(batch, embeddings):
                self._execute_query("""
                    MATCH (n)
                    WHERE elementId(n) = $id
                    SET n.embedding = $embedding
                """, id=node["id"], embedding=embedding)
                success += 1

        except Exception as e:
            logger.error(f"Batch {i//batch_size} failed: {e}")
            failed += len(batch)

        # Progress logging
        if (i // batch_size) % 10 == 0:
            logger.info(f"Re-embedding progress: {success}/{total} ({100*success//total}%)")

    logger.info(f"Re-embedding complete: {success} success, {failed} failed")
    return {"total": total, "success": success, "failed": failed}
```

### Provider Switching Workflow

```python
# codebase_rag/embeddings/switching.py

def switch_embedding_provider(
    provider: str,
    model_id: str,
    force: bool = False,
    reembed: bool = True,
) -> SwitchResult:
    """Switch embedding provider with complete migration workflow.

    This is the RECOMMENDED way to change embedding providers. It handles:
    1. Configuration update
    2. Dimension mismatch detection
    3. Cache invalidation (file DELETED)
    4. Vector index recreation
    5. Re-embedding all nodes (optional)

    Args:
        provider: Provider name ("local", "openai", "google", "ollama")
        model_id: Model identifier (e.g., "text-embedding-3-small")
        force: If True, proceed even with dimension mismatch (data loss)
        reembed: If True, re-embed all nodes after switch

    Returns:
        SwitchResult with status and statistics

    Raises:
        DimensionMismatchError: If dimension mismatch and force=False

    Example:
        # Safe switch (will raise if dimension mismatch)
        result = switch_embedding_provider("openai", "text-embedding-3-small")

        # Force switch with re-embedding
        result = switch_embedding_provider(
            provider="openai",
            model_id="text-embedding-3-small",
            force=True,
            reembed=True
        )
        print(f"Re-embedded {result.nodes_reembedded} nodes in {result.duration}")
    """
    from codebase_rag.config import settings
    from codebase_rag.embedder import get_embedding_provider_instance

    old_provider = settings.EMBEDDING_PROVIDER
    old_model = settings.EMBEDDING_MODEL

    # Get new provider instance to check dimension
    new_config = EmbeddingConfig(provider=provider, model_id=model_id)
    new_provider = get_embedding_provider_from_config(new_config)
    new_dim = new_provider.dimension

    # Check for dimension mismatch
    vector_store = get_vector_store()
    existing_dim = vector_store._get_existing_embedding_dimension()

    if existing_dim is not None and existing_dim != new_dim:
        if not force:
            # Delete cache file before raising error
            vector_store._delete_cache_on_mismatch()

            raise DimensionMismatchError(
                existing_dim=existing_dim,
                configured_dim=new_dim,
                message=vector_store._format_dimension_mismatch_message(
                    existing_dim, new_dim
                ),
            )

        # Force=True: user acknowledges data loss
        logger.warning(
            f"Dimension mismatch forced: {existing_dim}d -> {new_dim}d. "
            "All existing vectors will be deleted."
        )

    # Update configuration
    settings.EMBEDDING_PROVIDER = provider
    settings.EMBEDDING_MODEL = model_id

    # Delete old cache file (complete deletion, not just clear)
    vector_store._delete_cache_on_mismatch()

    # Clear global provider singleton
    global _embedding_provider
    _embedding_provider = None

    # Recreate vector indexes if dimension changed
    if existing_dim != new_dim:
        vector_store.recreate_vector_indexes(new_dim)

    # Re-embed all nodes if requested
    nodes_reembedded = 0
    if reembed:
        result = vector_store.reembed_all()
        nodes_reembedded = result["success"]

    return SwitchResult(
        old_provider=old_provider,
        old_model=old_model,
        new_provider=provider,
        new_model=model_id,
        old_dimension=existing_dim,
        new_dimension=new_dim,
        nodes_reembedded=nodes_reembedded,
    )


@dataclass
class SwitchResult:
    """Result of embedding provider switch."""
    old_provider: str
    old_model: str
    new_provider: str
    new_model: str
    old_dimension: int | None
    new_dimension: int
    nodes_reembedded: int

    @property
    def dimension_changed(self) -> bool:
        return self.old_dimension != self.new_dimension
```

### Index Recreation

```python
def recreate_vector_indexes(self, new_dimension: int) -> None:
    """Drop and recreate all vector indexes with new dimension.

    WARNING: This operation clears all embedding vectors from the database.
    Use switch_embedding_provider() for a complete migration workflow.
    """
    # Drop existing indexes
    for label in self.LABELS_TO_INDEX:
        index_name = f"{label.lower()}_embedding_index"
        self._execute_query(f"DROP VECTOR INDEX {index_name} IF EXISTS;")

    # Clear embeddings on nodes
    self._execute_query("""
        MATCH (n) WHERE n.embedding IS NOT NULL
        SET n.embedding = NULL;
    """)

    # Create new indexes with new dimension
    settings.MEMGRAPH_VECTOR_DIM = new_dimension
    self.initialize()
```

---

## Error Handling (Updated)

### Exception Hierarchy

```python
# codebase_rag/exceptions.py

from enum import StrEnum


class EmbeddingErrorCode(StrEnum):
    """Error codes for programmatic handling."""
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


class ModelNotFoundError(EmbeddingError):
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
        endpoint: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.CONNECTION_FAILED,
            provider=provider,
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
        limit_type: str | None = None,  # "rpm", "tpm", etc.
    ) -> None:
        super().__init__(
            message,
            error_code=EmbeddingErrorCode.RATE_LIMITED,
            provider=provider,
            model=model,
        )
        self.retry_after = retry_after  # Seconds to wait before retry
        self.limit_type = limit_type


class EmbeddingQuotaExceededError(EmbeddingError):
    """Raised when quota/billing limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        quota_type: str | None = None,  # "monthly", "daily", etc.
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
        *,
        provider: str | None = None,
        model: str | None = None,
        existing_model: str | None = None,
    ) -> None:
        super().__init__(
            f"Dimension mismatch: existing={existing_dim}, configured={configured_dim}",
            error_code=EmbeddingErrorCode.DIMENSION_MISMATCH,
            provider=provider,
            model=model,
        )
        self.existing_dim = existing_dim
        self.configured_dim = configured_dim
        self.existing_model = existing_model
```

### User-Facing Error Messages

```python
# codebase_rag/embeddings/errors.py

from ..exceptions import EmbeddingErrorCode


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
  2. Increase timeout: EMBEDDING_TIMEOUT={timeout_seconds * 2}
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


def get_user_facing_message(
    error_code: EmbeddingErrorCode,
    **kwargs,
) -> str:
    """Format a user-facing error message with context."""
    template = USER_FACING_MESSAGES.get(error_code, "Unknown error: {error_code}")
    return template.format(**kwargs)
```

### Error Usage Example

```python
# Example usage in providers

from ..exceptions import (
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    BatchSizeExceededError,
    InputValidationError,
)

class OpenAIEmbeddingProvider:
    MAX_BATCH_SIZE = 2048

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        # Validate batch size
        if len(texts) > self.MAX_BATCH_SIZE:
            raise BatchSizeExceededError(
                batch_size=len(texts),
                max_batch_size=self.MAX_BATCH_SIZE,
                provider="openai",
                model=self.model_id,
            )

        # Validate inputs
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise InputValidationError(
                    f"Empty text at index {i}",
                    provider="openai",
                    model=self.model_id,
                    input_text=text,
                    reason="empty_input",
                )

        try:
            return self._call_api(texts)
        except RateLimitError as e:
            raise EmbeddingRateLimitError(
                "OpenAI rate limit exceeded",
                provider="openai",
                model=self.model_id,
                retry_after=e.retry_after,
                limit_type="rpm",
            )
        except TimeoutError as e:
            raise EmbeddingTimeoutError(
                provider="openai",
                model=self.model_id,
                timeout_seconds=self.timeout,
                operation="batch_embed",
            )
```## Rate Limiting Infrastructure

```python
# codebase_rag/embeddings/rate_limiter.py

import time
import random

class AdaptiveRateLimiter:
    """Token bucket rate limiter with adaptive backoff."""

    def __init__(
        self,
        requests_per_minute: int = 500,
        tokens_per_minute: int = 300_000,
    ):
        self.rpm_bucket = TokenBucket(requests_per_minute, 60.0)
        self.tpm_bucket = TokenBucket(tokens_per_minute, 60.0)
        self._consecutive_429s: int = 0

    def acquire(self, tokens: int) -> float:
        """Wait until capacity available."""
        wait_time = max(
            self.rpm_bucket.acquire(1),
            self.tpm_bucket.acquire(tokens)
        )
        if wait_time > 0:
            time.sleep(wait_time)
        return wait_time

    def handle_429(self, retry_after: float | None) -> float:
        """Handle rate limit response with adaptive backoff."""
        self._consecutive_429s += 1
        if retry_after:
            backoff = retry_after
        else:
            base = min(60.0, 2.0 ** self._consecutive_429s)
            backoff = base + random.uniform(0, 1)
        time.sleep(backoff)
        return backoff

    def reset_429_counter(self) -> None:
        self._consecutive_429s = 0
```

---

## File Structure

```
codebase_rag/
├── embeddings/
│   ├── __init__.py          # Registry, factory functions
│   ├── base.py              # EmbeddingProvider abstract base
│   ├── openai.py            # OpenAI implementation
│   ├── google.py            # Google implementation
│   ├── ollama.py            # Ollama implementation
│   ├── local.py             # Local transformers (UniXcoder)
│   ├── rate_limiter.py      # API rate limiting
│   ├── errors.py            # User-facing error messages
│   └── protocols.py         # Type protocols
├── embedder.py              # Refactored to use providers
├── config.py                # Add EmbeddingConfig, new settings
├── constants.py             # Add embedding constants, enum
├── exceptions.py            # Add embedding exceptions
├── vector_backend.py        # Update for dynamic dimension
├── vector_store_memgraph.py # Update for dynamic dimension
└── unixcoder.py             # Keep for local provider
```

---

## Configuration Examples

### Local (UniXcoder - Default)

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=microsoft/unixcoder-base
```

### OpenAI

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...
```

### Google GLA

```bash
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_API_KEY=...
```

### Google Vertex AI

```bash
EMBEDDING_PROVIDER=google
EMBEDDING_PROVIDER_TYPE=vertex
EMBEDDING_MODEL=text-embedding-005
EMBEDDING_PROJECT_ID=my-project
EMBEDDING_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
```

### Ollama

```bash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_ENDPOINT=http://localhost:11434
EMBEDDING_KEEP_ALIVE=5m  # Optional: keep model loaded for 5 minutes (default: unset)
```

---

## Backward Compatibility

### Guaranteed

1. **Default behavior unchanged**: `local` + `microsoft/unixcoder-base` = 768-dim
2. **embedder.py functions preserved**: `embed_code`, `embed_code_batch` work identically
3. **Cache format migrated**: Legacy cache auto-migrated to new format
4. **Vector backend unchanged**: Protocol interface same
5. **MEMGRAPH_VECTOR_DIM preserved**: Default 768, can override

### Migration Path

1. **Phase 1**: Add `embeddings/` package, keep UniXcoder as default
2. **Phase 2**: Add OpenAI, Google, Ollama providers
3. **Phase 3**: Update vector backends for dynamic dimension
4. **Phase 4**: Add MCP tools for embedding configuration
5. **Phase 5**: Update documentation

---

## Testing Strategy

### Unit Tests (35+)

```python
# test_embedding_providers.py

class TestEmbeddingProviderBase:
    def test_slots_include_config()
    def test_dimension_property()

class TestLocalEmbeddingProvider:
    def test_embed_returns_correct_dimension()
    def test_embed_batch_handles_empty_list()
    def test_embed_batch_respects_batch_size()
    def test_device_auto_selection()

class TestOpenAIEmbeddingProvider:
    def test_embed_calls_api_correctly()
    def test_model_detection_for_dimensions()
    def test_api_key_validation()
    def test_rate_limiter_integration()
    def test_429_handling()
    def test_token_truncation()

class TestGoogleEmbeddingProvider:
    def test_gla_provider_initialization()
    def test_vertex_provider_initialization()
    def test_adc_credentials()
    def test_service_account_auth()

class TestOllamaEmbeddingProvider:
    def test_endpoint_validation()
    def test_batch_embedding()
    def test_dimension_probing()
    def test_keep_alive_parameter()

class TestEmbeddingProviderFactory:
    def test_get_provider_returns_correct_type()
    def test_get_provider_from_config()
    def test_register_provider_adds_to_registry()
```

### Integration Tests (20+)

```python
# test_embeddings_integration.py

def test_embedding_with_memgraph_backend()
def test_embedding_with_qdrant_backend()
def test_dimension_synced_with_vector_index()
def test_embedding_cache_with_different_providers()
def test_provider_switch_invalidates_cache()
def test_provider_switch_recreates_vector_index()
def test_dimension_mismatch_detection()
def test_code_index_to_embedding_to_storage_flow()
```

### Cache Tests (11+)

```python
# test_cache_invalidation.py

def test_cache_key_includes_model_id()
def test_cache_hit_same_provider_same_model()
def test_cache_miss_different_provider()
def test_cache_survives_restart()
def test_cache_format_backward_compatible()
def test_cache_corruption_recovery()
def test_cache_migration_to_model_aware_format()
def test_cache_dimension_metadata()
```

### Dimension Mismatch Tests (10+)

```python
# test_dimension_mismatch.py

def test_dimension_mismatch_detection_on_insert()
def test_existing_vectors_detect_dimension()
def test_provider_switch_detects_dimension_change()
def test_dimension_validation_before_batch_insert()
def test_vector_index_recreation_on_dimension_change()
```

### Authentication & Auth Errors Tests (8 tests)

```python
# test_errors.py

class TestAuthenticationErrors:
    def test_openai_invalid_api_key_format()
    def test_google_invalid_api_key_format()
    def test_google_expired_credentials()
    def test_google_revoked_service_account_key()
    def test_ollama_missing_auth_when_required()
    def test_vertex_missing_project_id()
    def test_vertex_invalid_service_account_file()
    def test_missing_all_auth_raises_clear_error()
```

### Network & Connectivity Tests (6 tests)

```python
# test_errors.py

class TestNetworkErrors:
    def test_openai_connection_refused()
    def test_ollama_dns_failure()
    def test_google_api_timeout()
    def test_retry_logic_on_transient_failure()
    def test_partial_batch_failure_handling()
    def test_circuit_breaker_triggers_after_max_retries()
```

### Concurrency Tests (4 tests)

```python
# test_concurrency.py

class TestConcurrency:
    def test_singleton_initialization_race_condition()
    def test_cache_concurrent_read_write_safety()
    def test_provider_hot_swap_thread_safety()
    def test_rate_limiter_concurrent_requests()
```

### Edge Inputs Tests (5 tests)

```python
# test_edge_inputs.py

class TestEdgeInputs:
    def test_empty_string_input()
    def test_whitespace_only_input()
    def test_unicode_handling_emoji_cjk()
    def test_special_characters_sql_injection_attempt()
    def test_token_limit_exceeded_truncation()
```

### Config & Precedence Tests (4 tests)

```python
# test_config.py

class TestConfigPrecedence:
    def test_env_var_overrides_dataclass_fields()
    def test_dataclass_overrides_defaults()
    def test_missing_optional_fields_use_defaults()
    def test_invalid_config_values_raise_validation_error()
```

### Test Summary

| Category | Count | File |
|----------|-------|------|
| Unit Tests | 35+ | test_embedding_providers.py |
| Integration Tests | 20+ | test_embeddings_integration.py |
| Cache Tests | 11+ | test_cache_invalidation.py |
| Dimension Mismatch Tests | 10+ | test_dimension_mismatch.py |
| Authentication & Auth Errors Tests | 8 | test_errors.py |
| Network & Connectivity Tests | 6 | test_errors.py |
| Concurrency Tests | 4 | test_concurrency.py |
| Edge Inputs Tests | 5 | test_edge_inputs.py |
| Config & Precedence Tests | 4 | test_config.py |
| **Total** | **100+** | |

### Recommended Test Files

- `test_embedding_providers.py` - Unit tests for all providers
- `test_embeddings_integration.py` - Integration tests with vector backends
- `test_cache_invalidation.py` - Cache behavior tests
- `test_dimension_mismatch.py` - Dimension handling tests
- `test_errors.py` - Authentication and network error tests (NEW)
- `test_concurrency.py` - Thread safety and race condition tests (NEW)
- `test_edge_inputs.py` - Edge case input handling tests (NEW)
- `test_config.py` - Configuration precedence tests

---

## Implementation Checklist

### Phase 1: Core Infrastructure

- [ ] Create `embeddings/base.py` with `EmbeddingProvider` abstract class
- [ ] Create `embeddings/local.py` wrapping existing UniXcoder
- [ ] Create `embeddings/__init__.py` with registry and factory
- [ ] Create `embeddings/protocols.py` with type protocols
- [ ] Update `config.py` with `EmbeddingConfig` and new settings
- [ ] Update `constants.py` with embedding constants and enum
- [ ] Add `EmbeddingProvider` enum to align with existing `Provider`
- [ ] Add `EmbeddingError` exception hierarchy to `exceptions.py`
- [ ] Refactor `embedder.py` to use provider system
- [ ] Audit and update all files importing from `embedder.py`
- [ ] Add `validate_embedding_config()` startup check
- [ ] Ensure backward compatibility (default = local/unixcoder)

### Phase 2: External Providers

- [ ] Create `embeddings/openai.py` with rate limiting
- [ ] Create `embeddings/google.py` (GLA + Vertex + ADC)
- [ ] Create `embeddings/ollama.py` with dimension probing
- [ ] Create `embeddings/rate_limiter.py`
- [ ] Create `embeddings/errors.py` with user-facing messages

> **Note**: Phase 2 must complete before Phase 3 because Phase 3's dimension
> validation requires the `EMBEDDING_MODEL_DIMENSIONS` mappings defined by
> provider implementations.

### Phase 3: Cache & Dimension

- [ ] Update `EmbeddingCache` for per-model files
- [ ] Add cache migration logic for legacy format
- [ ] Add large cache migration performance consideration (batch processing)
- [ ] Implement `_get_existing_embedding_dimension()` in vector backends
- [ ] Add logic to clear old embeddings when switching providers
- [ ] Update `vector_backend.py` for dynamic dimension
- [ ] Update `vector_store_memgraph.py` with dimension validation
- [ ] Add `recreate_vector_indexes()` method
- [ ] Move `DimensionMismatchError` to exceptions (defined in Phase 1)

### Phase 4: Testing

- [ ] Add 35+ unit tests for providers
- [ ] Add 20+ integration tests
- [ ] Add 11+ cache tests
- [ ] Add 10+ dimension mismatch tests
- [ ] Add provider switching mid-session tests
- [ ] Add singleton lifecycle tests (creation, reuse, reset)
- [ ] Add conditional import fallback tests (torch unavailable)
- [ ] Verify backward compatibility tests pass

### Phase 5: MCP Tools

- [ ] Add `get_embedding_status` tool
- [ ] Add `set_embedding_provider` tool
- [ ] Add `change_embedding_dimension` tool

> **Note**: MCP tools are implemented before documentation so docs can reference
> the actual tool implementations.

### Phase 6: Documentation

- [ ] Update MCP server docs
- [ ] Update Claude Code setup docs
- [ ] Create embeddings provider guide

---

## Risk Assessment

### Low Risk

- Abstract base class design
- Registry pattern (well-established)
- Configuration extension (additive)

### Medium Risk

- Dimension mismatch handling
- Cache compatibility with model changes
- API rate limiting

### High Risk

- Breaking existing behavior (must test thoroughly)
- Vector index recreation on dimension change

### Mitigation Strategies

1. **Default = existing**: UniXcoder local remains default
2. **Dimension validation**: Warn before creating mismatched index
3. **Cache migration**: Auto-migrate legacy caches
4. **Comprehensive tests**: 80+ tests ensure no regression

---

## Round 2 Review Summary

**Review Date**: 2026-03-29
**Workers**: 10/10 completed

### Worker 1: Abstract Base Review

**Score**: 8/10

| Issue | Fix |
|-------|-----|
| `provider_name` returns `str` not enum | Change to `cs.EmbeddingProvider` |
| `**config` missing type annotation | Add `**config: str \| int \| None` |
| No design divergence documentation | Add docstring explaining lack of `create_model()` |
| `embed_batch` chunking undefined | Clarify behavior when `texts > batch_size` |

### Worker 2: Cache Architecture Review

**Score**: 6/10

| Issue | Fix |
|-------|-----|
| Model ID sanitization incomplete | Use `urllib.parse.quote()` or base64 |
| `_migrate_legacy` undefined | Add explicit migration method |
| Only validates dimension, not model_id | Validate both for semantic compatibility |
| Dimension mismatch doesn't persist clear | Delete invalid cache file, not just clear memory |
| `_content_hash` signature change undefined | Define new signature with `model_id` param |
| No version upgrade path | Add version compatibility check |
| No file locking for concurrent access | Add advisory locking or atomic writes |

### Worker 3: Dimension Handling Review

**Score**: 5/10

| Issue | Fix |
|-------|-----|
| No resolution workflow for mismatches | Add `switch_embedding_provider()` with complete flow |
| Missing `_get_existing_embedding_dimension()` | Define implementation with fallbacks |
| No re-embedding after clear | Add `reembed_all()` method, document time cost |
| Error messages not actionable | Include specific commands, data loss warnings, time estimates |
| Cache file persists after mismatch | Delete mismatched cache file |
| No consistency validation | Add `validate_embedding_consistency()` check |
| Startup behavior undefined | Document when `initialize()` runs |

### Worker 4: Error Handling Review

**Score**: 5/10

| Issue | Fix |
|-------|-----|
| Duplicate exception naming | Consolidate `EmbeddingDimensionMismatchError` and `DimensionMismatchError` |
| Missing critical exceptions | Add `InputValidationError`, `BatchSizeExceededError`, `TimeoutError`, etc. |
| Only 3 user-facing messages | Add 15+ more scenarios |
| Rate limit error missing `retry_after` | Add attribute for programmatic handling |
| No error codes | Add `error_code` for programmatic handling |

### Worker 5: OpenAI Provider Review

**Score**: 4/10

| Issue | Fix |
|-------|-----|
| Rate limiter not integrated with provider | Add explicit integration in `__init__` |
| Unlimited 429 counter growth | Add circuit breaker (MAX_CONSECUTIVE_429S) |
| Token truncation completely missing | Implement with tiktoken |
| Batch implementation missing | Add OpenAI-specific batch handling |
| No retry loop implementation | Add `_call_with_retry()` method |
| No success callback to reset counter | Reset on successful API call |

### Worker 6: Google Provider Review

**Score**: 5/10

| Issue | Fix |
|-------|-----|
| GLA/Vertex auth confusion | GLA requires API key only, Vertex requires ADC/project |
| Impersonation parameter never used | Add `impersonated_credentials` implementation |
| Missing scopes for service account | Add `scopes=["https://www.googleapis.com/auth/cloud-platform"]` |
| No client creation code | Add `_create_client()` method |
| Quota project not applied to SA | Add to service account instantiation |
| text-embedding-005 dimension wrong | Verify actual dimensions (may be 3072, not 768) |

### Worker 7: Ollama Provider Review

**Score**: 5/10

| Issue | Fix |
|-------|-----|
| Dimension probing before `super().__init__()` | Move endpoint initialization first, add `_probe_dimension()` |
| `keep_alive: None` sent as `null` | Conditionally omit from payload |
| Missing `embed_batch()` for `/api/embed` | Add batch implementation |
| Model not pulled detection | Parse Ollama 404 response for "not found" |
| `keep_alive` not in config | Add `EMBEDDING_KEEP_ALIVE` env var |

### Worker 8: Configuration Review

**Score**: 6/10

| Issue | Fix |
|-------|-----|
| Missing `EMBEDDING_SERVICE_ACCOUNT_FILE` | Add to AppConfig |
| Undefined API key fallback hierarchy | Document `EMBEDDING_API_KEY` → `{PROVIDER}_API_KEY` |
| `OLLAMA_BASE_URL` ignored for embeddings | Add fallback for Ollama provider |
| Default detection unreliable | Use `None` as sentinel for auto-detect |
| `QDRANT_VECTOR_DIM` not addressed | Unify or deprecate |
| Dimension lookup not provider-aware | Add provider context to lookup |
| In-memory mutation non-persistent | Document or require env var change |

### Worker 9: Test Coverage Review

**Score**: 6/10

| Issue | Fix |
|-------|-----|
| ~76 tests, not quite 80+ | Increase target to 100+ tests |
| Missing authentication failure tests | Add 8 tests for API key errors per provider |
| Missing network failure tests | Add 6 tests for connection/timeout |
| Missing concurrency tests | Add 4 tests for thread safety |
| Missing edge input tests | Add 5 tests for empty/unicode/limits |
| Missing config precedence tests | Add 4 tests for env var vs dataclass |

### Worker 10: Implementation Phases Review

**Score**: 7/10

| Issue | Fix |
|-------|-----|
| Phase 2 ↔ Phase 3 dependency | Merge or add explicit dependency note |
| Missing items in Phase 1 | Add `protocols.py`, exceptions, import audit |
| Phase order issue | Swap Phase 5 and Phase 6 (MCP tools before docs) |
| Missing cache warm-up strategy | Add large cache migration consideration |
| Missing config validation | Add `validate_embedding_config()` startup check |

### Consolidated Critical Fixes

1. **Cache migration**: Define `_migrate_legacy()` explicitly, validate both `model_id` and `dimension`
2. **Token truncation**: Implement with tiktoken for OpenAI, add retry loop with circuit breaker
3. **Dimension workflow**: Add complete `switch_embedding_provider()` workflow with re-embedding
4. **Errors**: Merge duplicate exceptions, expand user messages from 3 to 15+
5. **Config**: Use `None` as sentinel for auto-detect, add missing config fields
6. **Google auth**: GLA requires API key only, Vertex requires ADC/project - separate validation
7. **Ollama probing**: Move endpoint init first, add `_probe_dimension()` with error handling
8. **Test coverage**: Increase from ~76 to 100+ tests, add auth/network/concurrency tests

### Worker Scores Summary

| Worker | Area | Score |
|--------|------|-------|
| 1 | Abstract Base | 8/10 |
| 2 | Cache Architecture | 6/10 |
| 3 | Dimension Handling | 5/10 |
| 4 | Error Handling | 5/10 |
| 5 | OpenAI Provider | 4/10 |
| 6 | Google Provider | 5/10 |
| 7 | Ollama Provider | 5/10 |
| 8 | Configuration | 6/10 |
| 9 | Test Coverage | 6/10 |
| 10 | Implementation Phases | 7/10 |

**Average Score**: 5.7/10 - Implementation needs refinement before coding

### Top Priority Fixes

| Priority | Area | Fix |
|----------|------|-----|
| 1 | OpenAI | Implement token truncation with tiktoken |
| 2 | Dimension | Add `switch_embedding_provider()` workflow |
| 3 | Google | Separate GLA/Vertex auth validation |
| 4 | Ollama | Fix dimension probing initialization order |
| 5 | Cache | Define `_migrate_legacy()` method |
| 6 | Errors | Consolidate exceptions, expand messages |

---

## References

- OpenAI Embeddings API: https://platform.openai.com/docs/guides/embeddings
- Google AI Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
- Memgraph Vector Index: https://memgraph.com/docs/reference-guide/indexing
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- UniXcoder: https://github.com/microsoft/unixcoder