from __future__ import annotations

import os
from abc import ABC, abstractmethod
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import (
    AnthropicProvider as PydanticAnthropicProvider,
)
from pydantic_ai.providers.azure import AzureProvider as PydanticAzureProvider
from pydantic_ai.providers.google import GoogleProvider as PydanticGoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .. import constants as cs
from .. import exceptions as ex
from .. import logs as ls
from ..config import ModelConfig, settings


class ModelProvider(ABC):
    __slots__ = ("config",)
    MODEL_CONTEXT_WINDOWS: dict[str, int] = {}

    def __init__(self, **config: str | int | None) -> None:
        self.config = config

    @abstractmethod
    def create_model(
        self, model_id: str, **kwargs: str | int | None
    ) -> GoogleModel | OpenAIResponsesModel | OpenAIChatModel | AnthropicModel:
        pass

    @abstractmethod
    def validate_config(self) -> None:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> cs.Provider:
        pass

    def get_model_context_window(self, model_id: str) -> int:
        """Get the context window size for a given model ID.

        Precedence order:
        1. Provider/model-specific environment variable override: {PROVIDER}_{MODEL}_CONTEXT_WINDOW
        2. Pre-defined model context window map for the provider
        3. Global default context window (256k)
        """
        import re

        # Normalize model ID: convert to uppercase, replace spaces, hyphens, periods with underscores
        normalized_model = re.sub(r"[\s\-\.]", "_", model_id).upper()
        # Check for provider/model-specific env var
        env_var_name = (
            f"{self.provider_name.value.upper()}_{normalized_model}_CONTEXT_WINDOW"
        )
        env_value = os.environ.get(env_var_name)
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(
                    f"Invalid value for {env_var_name}, falling back to default"
                )

        # Check provider-specific model map
        if model_id in self.MODEL_CONTEXT_WINDOWS:
            return self.MODEL_CONTEXT_WINDOWS[model_id]
        if normalized_model in self.MODEL_CONTEXT_WINDOWS:
            return self.MODEL_CONTEXT_WINDOWS[normalized_model]
        for model_prefix, window_size in self.MODEL_CONTEXT_WINDOWS.items():
            if normalized_model.startswith(model_prefix.rstrip("*").upper()):
                return window_size

        # Fall back to global default
        return settings.DEFAULT_CONTEXT_WINDOW


def _resolve_api_key(api_key: str | None, env_var: str) -> str | None:
    if api_key and api_key != cs.DEFAULT_API_KEY:
        return api_key
    env_key = os.environ.get(env_var)
    if env_key:
        return env_key
    return None


class GoogleProvider(ModelProvider):
    __slots__ = (
        "api_key",
        "provider_type",
        "project_id",
        "region",
        "service_account_file",
        "thinking_budget",
    )

    MODEL_CONTEXT_WINDOWS = {
        "gemini-2.5-pro*": 1048576,
        "gemini-2.5-flash*": 1048576,
        "gemini-1.5-pro*": 2097152,
        "gemini-1.5-flash*": 1048576,
    }

    def __init__(
        self,
        api_key: str | None = None,
        provider_type: cs.GoogleProviderType = cs.GoogleProviderType.GLA,
        project_id: str | None = None,
        region: str = cs.DEFAULT_REGION,
        service_account_file: str | None = None,
        thinking_budget: int | None = None,
        **kwargs: str | int | None,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = _resolve_api_key(api_key, cs.ENV_GOOGLE_API_KEY)
        self.provider_type = provider_type
        self.project_id = project_id
        self.region = region
        self.service_account_file = service_account_file
        self.thinking_budget = thinking_budget

    @property
    def provider_name(self) -> cs.Provider:
        return cs.Provider.GOOGLE

    def validate_config(self) -> None:
        if self.provider_type == cs.GoogleProviderType.GLA and not self.api_key:
            raise ValueError(ex.GOOGLE_GLA_NO_KEY)
        if self.provider_type == cs.GoogleProviderType.VERTEX and not self.project_id:
            raise ValueError(ex.GOOGLE_VERTEX_NO_PROJECT)

    def create_model(self, model_id: str, **kwargs: str | int | None) -> GoogleModel:
        self.validate_config()

        if self.provider_type == cs.GoogleProviderType.VERTEX:
            credentials = None
            if self.service_account_file:
                # (H) Convert service account file to credentials object for pydantic-ai
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_file,
                    scopes=[cs.GOOGLE_CLOUD_SCOPE],
                )
            provider = PydanticGoogleProvider(
                project=self.project_id,
                location=self.region,
                credentials=credentials,
            )
        else:
            # (H) api_key is guaranteed to be set by validate_config for gla type
            assert self.api_key is not None
            provider = PydanticGoogleProvider(api_key=self.api_key)

        if self.thinking_budget is None:
            return GoogleModel(model_id, provider=provider)
        model_settings = GoogleModelSettings(
            google_thinking_config={"thinking_budget": int(self.thinking_budget)}
        )
        return GoogleModel(model_id, provider=provider, settings=model_settings)


class OpenAIProvider(ModelProvider):
    __slots__ = ("api_key", "endpoint")

    MODEL_CONTEXT_WINDOWS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 128000,
    }

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = cs.OPENAI_DEFAULT_ENDPOINT,
        **kwargs: str | int | None,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = _resolve_api_key(api_key, cs.ENV_OPENAI_API_KEY)
        self.endpoint = endpoint

    @property
    def provider_name(self) -> cs.Provider:
        return cs.Provider.OPENAI

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(ex.OPENAI_NO_KEY)

    def create_model(
        self, model_id: str, **kwargs: str | int | None
    ) -> OpenAIChatModel:
        self.validate_config()

        provider = PydanticOpenAIProvider(api_key=self.api_key, base_url=self.endpoint)
        return OpenAIChatModel(model_id, provider=provider)


class OllamaProvider(ModelProvider):
    __slots__ = ("endpoint", "api_key")

    MODEL_CONTEXT_WINDOWS = {
        "llama3.1*": 128000,
        "llama3*": 8192,
        "mistral-nemo*": 128000,
        "gemma2*": 128000,
        "qwen2*": 128000,
        "phi3*": 128000,
    }

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str = cs.DEFAULT_API_KEY,
        **kwargs: str | int | None,
    ) -> None:
        super().__init__(**kwargs)
        self.endpoint = endpoint or settings.ollama_endpoint
        self.api_key = api_key

    @property
    def provider_name(self) -> cs.Provider:
        return cs.Provider.OLLAMA

    def validate_config(self) -> None:
        base_url = self.endpoint.rstrip(cs.V1_PATH).rstrip("/")

        if not check_ollama_running(base_url):
            raise ValueError(ex.OLLAMA_NOT_RUNNING.format(endpoint=base_url))

    def create_model(
        self, model_id: str, **kwargs: str | int | None
    ) -> OpenAIChatModel:
        self.validate_config()

        provider = PydanticOpenAIProvider(api_key=self.api_key, base_url=self.endpoint)
        return OpenAIChatModel(model_id, provider=provider)


class AnthropicProvider(ModelProvider):
    __slots__ = ("api_key",)

    MODEL_CONTEXT_WINDOWS = {
        "claude-3-5-sonnet*": 200000,
        "claude-3-opus*": 200000,
        "claude-3-sonnet*": 200000,
        "claude-3-haiku*": 200000,
        "claude-2.1*": 200000,
        "claude-2.0*": 100000,
    }

    def __init__(
        self,
        api_key: str | None = None,
        **kwargs: str | int | None,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = _resolve_api_key(api_key, cs.ENV_ANTHROPIC_API_KEY)

    @property
    def provider_name(self) -> cs.Provider:
        return cs.Provider.ANTHROPIC

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(ex.ANTHROPIC_NO_KEY)

    def create_model(self, model_id: str, **kwargs: str | int | None) -> AnthropicModel:
        self.validate_config()
        # (H) api_key is guaranteed to be set by validate_config
        assert self.api_key is not None
        provider = PydanticAnthropicProvider(api_key=self.api_key)
        return AnthropicModel(model_id, provider=provider)


class AzureOpenAIProvider(ModelProvider):
    __slots__ = ("api_key", "endpoint", "api_version")

    # Azure OpenAI uses same model context windows as regular OpenAI
    MODEL_CONTEXT_WINDOWS = OpenAIProvider.MODEL_CONTEXT_WINDOWS

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        **kwargs: str | int | None,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = _resolve_api_key(api_key, cs.ENV_AZURE_API_KEY)
        self.endpoint = endpoint or os.environ.get(cs.ENV_AZURE_ENDPOINT)
        self.api_version = api_version or os.environ.get(cs.ENV_AZURE_API_VERSION)

    @property
    def provider_name(self) -> cs.Provider:
        return cs.Provider.AZURE

    def validate_config(self) -> None:
        if not self.api_key:
            raise ValueError(ex.AZURE_NO_KEY)
        if not self.endpoint:
            raise ValueError(ex.AZURE_NO_ENDPOINT)

    def create_model(
        self, model_id: str, **kwargs: str | int | None
    ) -> OpenAIChatModel:
        self.validate_config()
        # (H) api_key and endpoint are guaranteed to be set by validate_config
        assert self.api_key is not None
        assert self.endpoint is not None
        provider = PydanticAzureProvider(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )
        return OpenAIChatModel(model_id, provider=provider)


PROVIDER_REGISTRY: dict[str, type[ModelProvider]] = {
    cs.Provider.GOOGLE: GoogleProvider,
    cs.Provider.OPENAI: OpenAIProvider,
    cs.Provider.OLLAMA: OllamaProvider,
    cs.Provider.ANTHROPIC: AnthropicProvider,
    cs.Provider.AZURE: AzureOpenAIProvider,
}


def get_provider(
    provider_name: str | cs.Provider, **config: str | int | None
) -> ModelProvider:
    provider_key = str(provider_name)
    if provider_key not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            ex.UNKNOWN_PROVIDER.format(provider=provider_name, available=available)
        )

    provider_class = PROVIDER_REGISTRY[provider_key]
    return provider_class(**config)


def get_provider_from_config(config: ModelConfig) -> ModelProvider:
    return get_provider(
        config.provider,
        api_key=config.api_key,
        endpoint=config.endpoint,
        project_id=config.project_id,
        region=config.region,
        provider_type=config.provider_type,
        thinking_budget=config.thinking_budget,
        service_account_file=config.service_account_file,
    )


def register_provider(name: str, provider_class: type[ModelProvider]) -> None:
    PROVIDER_REGISTRY[name] = provider_class
    logger.info(ls.PROVIDER_REGISTERED.format(name=name))


def list_providers() -> list[str]:
    return list(PROVIDER_REGISTRY.keys())


def check_ollama_running(endpoint: str | None = None) -> bool:
    endpoint = endpoint or settings.OLLAMA_BASE_URL
    try:
        health_url = urljoin(endpoint, cs.OLLAMA_HEALTH_PATH)
        with httpx.Client(timeout=settings.OLLAMA_HEALTH_TIMEOUT) as client:
            response = client.get(health_url)
            return response.status_code == cs.HTTP_OK
    except (httpx.RequestError, httpx.TimeoutException):
        return False
