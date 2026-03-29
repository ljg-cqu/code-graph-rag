"""Google embedding provider supporting GLA and Vertex AI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .. import constants as cs
from ..exceptions import EmbeddingAuthenticationError, EmbeddingGenerationError
from .base import EmbeddingProvider

if TYPE_CHECKING:
    import httpx

# Known Google embedding model dimensions
GOOGLE_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}

# Default models
DEFAULT_GOOGLE_MODEL = "text-embedding-004"

# Provider types
GLA_PROVIDER = "gla"  # Generative Language API (AI Studio)
VERTEX_PROVIDER = "vertex"  # Vertex AI

# Batch limits per provider type
BATCH_LIMITS = {
    "gla": {"max_texts": 100, "max_tokens_per_text": 20000},
    "vertex": {"max_texts": 250, "max_tokens_per_text": 30000},
}


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google embedding provider supporting GLA and Vertex AI.

    This provider supports two authentication modes:
    - GLA (Generative Language API): Uses API key via GOOGLE_API_KEY
    - Vertex AI: Uses service account or default credentials

    Attributes:
        model_id: Google model identifier (e.g., text-embedding-004).
        dimension: Embedding dimension (auto-detected from model).
        api_key: Google API key for GLA.
        project_id: Google Cloud project ID (Vertex AI).
        region: Google Cloud region (Vertex AI).
        provider_type: "gla" or "vertex".
        service_account_file: Path to service account JSON file.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_GOOGLE_MODEL,
        dimension: int | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        provider_type: str | None = None,
        service_account_file: str | None = None,
        # Additional config parameters that may be passed by factory (ignored)
        keep_alive: str | None = None,
        device: str | None = None,
    ) -> None:
        # Determine dimension from known models or default
        if dimension is None:
            dimension = GOOGLE_MODEL_DIMENSIONS.get(model_id, 768)

        super().__init__(
            model_id,
            dimension,
            api_key=api_key,
            endpoint=endpoint,
            project_id=project_id,
            region=region,
            provider_type=provider_type,
            service_account_file=service_account_file,
        )
        self._api_key = api_key
        self._project_id = project_id
        self._region = region or "us-central1"
        self._provider_type = provider_type or GLA_PROVIDER
        self._service_account_file = service_account_file
        self._endpoint = endpoint
        self._client: httpx.Client | None = None
        self._token: str | None = None

    @property
    def provider_name(self) -> cs.EmbeddingProvider:
        return cs.EmbeddingProvider.GOOGLE

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=60.0)
        return self._client

    def _get_auth_token(self) -> str:
        """Get authentication token for Vertex AI.

        Returns:
            Bearer token for Vertex AI.
        """
        if self._token:
            return self._token

        # Try to get token from service account
        if self._service_account_file:
            import json

            try:
                with open(self._service_account_file) as f:
                    creds = json.load(f)
                # Would need google-auth library for proper token generation
                # For simplicity, assume user has gcloud configured
                import subprocess

                result = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self._token = result.stdout.strip()
                return self._token
            except Exception as e:
                raise EmbeddingAuthenticationError(
                    f"Failed to get Vertex AI token: {e}",
                    provider="google",
                    model=self.model_id,
                ) from e

        # Try gcloud default application credentials
        try:
            import subprocess

            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self._token = result.stdout.strip()
            return self._token
        except Exception as e:
            raise EmbeddingAuthenticationError(
                f"Failed to get Google auth token: {e}. "
                "Ensure gcloud is configured or use GLA provider type.",
                provider="google",
                model=self.model_id,
            ) from e

    def validate_config(self) -> None:
        """Validate Google provider configuration.

        Raises:
            EmbeddingAuthenticationError: If required credentials are missing.
        """
        if self._provider_type == GLA_PROVIDER:
            if not self._api_key or not self._api_key.strip():
                raise EmbeddingAuthenticationError(
                    "Google GLA embedding requires an API key. "
                    "Set EMBEDDING_API_KEY or GOOGLE_API_KEY environment variable.",
                    provider="google",
                    model=self.model_id,
                )
        elif self._provider_type == VERTEX_PROVIDER:
            if not self._project_id:
                raise EmbeddingAuthenticationError(
                    "Vertex AI embedding requires a project ID. "
                    "Set EMBEDDING_PROJECT_ID environment variable.",
                    provider="google",
                    model=self.model_id,
                )
        else:
            raise EmbeddingAuthenticationError(
                f"Unknown Google provider type: {self._provider_type}. "
                "Use 'gla' or 'vertex'.",
                provider="google",
                model=self.model_id,
            )

    def _get_endpoint(self) -> str:
        """Get the API endpoint based on provider type.

        Returns:
            API endpoint URL.
        """
        if self._endpoint:
            return self._endpoint

        if self._provider_type == GLA_PROVIDER:
            return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:embedContent"

        # Vertex AI endpoint
        return (
            f"https://{self._region}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project_id}/locations/{self._region}/"
            f"publishers/google/models/{self.model_id}:predict"
        )

    def _make_request_gla(self, texts: list[str]) -> list[list[float]]:
        """Make embedding request using GLA API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        client = self._get_client()
        endpoint = self._get_endpoint()
        all_embeddings: list[list[float]] = []

        # GLA processes one text at a time
        for text in texts:
            params = {"key": self._api_key}
            payload = {
                "model": f"models/{self.model_id}",
                "content": {
                    "parts": [{"text": text}],
                },
            }

            try:
                response = client.post(endpoint, params=params, json=payload)
                response.raise_for_status()
                data = response.json()

                embedding = data.get("embedding", {}).get("values", [])
                if not embedding:
                    raise EmbeddingGenerationError(
                        "Google GLA returned empty embedding",
                        provider="google",
                        model=self.model_id,
                    )
                all_embeddings.append(embedding)

            except Exception as e:
                raise EmbeddingGenerationError(
                    f"Google GLA embedding request failed: {e}",
                    provider="google",
                    model=self.model_id,
                ) from e

        return all_embeddings

    def _make_request_vertex(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Make embedding request using Vertex AI API.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per request.

        Returns:
            List of embedding vectors.
        """
        client = self._get_client()
        endpoint = self._get_endpoint()
        token = self._get_auth_token()
        all_embeddings: list[list[float]] = []

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Vertex AI supports batch embedding
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            instances = [{"content": text} for text in batch]

            payload = {
                "instances": instances,
                "parameters": {
                    "outputDimensionality": self.dimension,
                },
            }

            try:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                predictions = data.get("predictions", [])
                for pred in predictions:
                    embedding = pred.get("embeddings", [])
                    if not embedding:
                        raise EmbeddingGenerationError(
                            "Vertex AI returned empty embedding",
                            provider="google",
                            model=self.model_id,
                        )
                    all_embeddings.append(embedding)

            except Exception as e:
                raise EmbeddingGenerationError(
                    f"Vertex AI embedding request failed: {e}",
                    provider="google",
                    model=self.model_id,
                ) from e

        return all_embeddings

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if self._provider_type == GLA_PROVIDER:
            embeddings = self._make_request_gla([text])
        else:
            embeddings = self._make_request_vertex([text], batch_size=1)
        return embeddings[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts per API call. Defaults to 32.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        if self._provider_type == GLA_PROVIDER:
            # GLA processes one at a time
            return self._make_request_gla(texts)
        else:
            # Vertex supports batch
            return self._make_request_vertex(texts, batch_size)