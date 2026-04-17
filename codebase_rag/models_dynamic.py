"""Dynamic model catalog builder combining static catalog with .env configurations.

Builds a runtime model catalog by merging the static MODEL_CATALOG with
models configured via environment variables (ORCHESTRATOR_*, CYPHER_*,
CGR_WORKER_LLMS). All entries are DynamicModelInfo frozen dataclass instances
for uniform processing.
"""

from __future__ import annotations

import json
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

from . import constants as cs
from .models_catalog import MODEL_CATALOG, PROVIDER_DISPLAY_NAMES

if TYPE_CHECKING:
    from .types_defs import ModelInfo


def _load_external_catalog(file_path: str) -> dict[str, list[ModelInfo]]:
    """Load model catalog from external JSON or YAML file.

    The file must contain a dictionary mapping provider keys to lists of model
    objects. Each model object must have the following fields matching ModelInfo:
      - provider (str): Provider identifier (e.g., "openai", "google")
      - model_id (str): Model identifier (e.g., "gpt-4o", "gemini-2.5-pro")
      - display_name (str): Human-readable name (e.g., "GPT-4o", "Gemini 2.5 Pro")
      - context_window (int): Default context window in tokens
      - description (str): Short description
      - requires_api_key (bool): Whether the model requires an API key
      - is_local (bool): Whether the model runs locally (e.g., Ollama)
      - pricing_tier (str): One of "free", "low", "medium", "high"

    Example JSON:
        {
            "openai": [
                {
                    "provider": "openai",
                    "model_id": "gpt-4o",
                    "display_name": "GPT-4o",
                    "context_window": 128000,
                    "description": "Multimodal, fast",
                    "requires_api_key": true,
                    "is_local": false,
                    "pricing_tier": "medium"
                }
            ]
        }

    Args:
        file_path: Path to JSON or YAML file.

    Returns:
        Dictionary mapping providers to lists of ModelInfo objects.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid or schema validation fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"External catalog file not found: {file_path}")

    # Read file
    content = path.read_text(encoding="utf-8")

    # Parse based on extension
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(content)
        except ImportError:
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
    elif path.suffix.lower() == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            "Use .json, .yaml, or .yml"
        )

    # Validate top-level structure
    if not isinstance(data, dict):
        raise ValueError(
            f"Catalog must be a dictionary (provider -> model list), got {type(data)}"
        )

    from .types_defs import ModelInfo

    catalog: dict[str, list[ModelInfo]] = {}

    for provider_key, model_list in data.items():
        if not isinstance(model_list, list):
            raise ValueError(
                f"Provider '{provider_key}' value must be a list, got {type(model_list)}"
            )

        provider_models = []
        for i, model_data in enumerate(model_list):
            if not isinstance(model_data, dict):
                raise ValueError(
                    f"Provider '{provider_key}' model[{i}] must be a dictionary, got {type(model_data)}"
                )

            # Convert to ModelInfo (validates fields via NamedTuple constructor)
            try:
                model_info = ModelInfo(
                    provider=model_data["provider"],
                    model_id=model_data["model_id"],
                    display_name=model_data["display_name"],
                    context_window=model_data["context_window"],
                    description=model_data["description"],
                    requires_api_key=model_data["requires_api_key"],
                    is_local=model_data["is_local"],
                    pricing_tier=model_data["pricing_tier"],
                )
            except KeyError as e:
                raise ValueError(
                    f"Provider '{provider_key}' model[{i}] missing required field: {e}"
                )
            except TypeError as e:
                raise ValueError(
                    f"Provider '{provider_key}' model[{i}] field type error: {e}"
                )

            provider_models.append(model_info)

        catalog[provider_key] = provider_models

    logger.info(f"Loaded external catalog from {file_path}: {len(catalog)} provider(s)")
    return catalog


@dataclass(frozen=True)
class DynamicModelInfo:
    """Enhanced model info with dynamic configuration.

    NOT a subclass of ModelInfo (NamedTuple) — uses frozen dataclass
    to allow new fields with defaults while preserving immutability.
    """

    provider: str
    model_id: str
    display_name: str
    context_window: int
    description: str
    requires_api_key: bool
    is_local: bool
    pricing_tier: Literal["free", "low", "medium", "high"]
    # New fields for dynamic models
    is_configured: bool = False
    endpoint: str | None = None
    source: Literal["static", "env_orchestrator", "env_cypher", "env_worker"] = (
        "static"
    )

    @classmethod
    def from_model_info(
        cls, info: ModelInfo, **overrides: object
    ) -> DynamicModelInfo:
        """Convert a static ModelInfo (NamedTuple) to DynamicModelInfo.

        Note: context_window should be resolved via get_model_context_window()
        at build time, not copied from the static ModelInfo. The static
        context_window is the baseline but may be overridden by env vars
        (e.g., OPENAI_GPT_4O_CONTEXT_WINDOW) or provider MODEL_CONTEXT_WINDOWS
        dict entries. Callers should pass the resolved value in **overrides.
        """
        return cls(
            provider=info.provider,
            model_id=info.model_id,
            display_name=info.display_name,
            context_window=overrides.pop("context_window", info.context_window),
            description=info.description,
            requires_api_key=info.requires_api_key,
            is_local=info.is_local,
            pricing_tier=info.pricing_tier,
            **overrides,
        )


def _check_api_key_available(model_info: ModelInfo | DynamicModelInfo) -> bool:
    """Check whether an API key is available for a static catalog model.

    Reuses the existing API_KEY_INFO dict from config.py (which already maps
    providers to their env var names and URLs) instead of creating a new mapping.

    Uses the same logic as ModelConfig.validate_api_key() and LOCAL_PROVIDERS:
    - Local providers (ollama, local, vllm) never require API keys → True
    - Google Vertex (provider_type=vertex) uses project_id → check project_id instead
    - All other providers: check their environment variable

    Returns True if the model can be used without additional user configuration.
    """
    from .config import API_KEY_INFO, LOCAL_PROVIDERS

    provider = model_info.provider.lower()

    # Local providers never need API keys
    if provider in LOCAL_PROVIDERS:
        return True

    # Non-local providers without API key requirement in catalog are always unconfigured
    if not model_info.requires_api_key:
        return True

    # Reuse existing API_KEY_INFO mapping (already maps providers → env var names)
    info = API_KEY_INFO.get(provider)
    if info and os.environ.get(info["env_var"]):
        return True

    # Special case for Google Vertex (uses project_id instead of API key)
    if provider == "google" and (
        os.environ.get("GOOGLE_PROJECT_ID") or
        os.environ.get("ORCHESTRATOR_PROJECT_ID") or
        os.environ.get("CYPHER_PROJECT_ID")
    ):
        return True

    return False


def _find_model_in_list(
    models: list[DynamicModelInfo], model_id: str
) -> int:
    """Find index of a model by model_id in a list. Returns -1 if not found."""
    for i, m in enumerate(models):
        if m.model_id == model_id:
            return i
    return -1


def _extract_configured_models_from_env() -> list[DynamicModelInfo]:
    """Extract models explicitly configured in .env variables.

    Only includes models that are explicitly set by the user (not defaults).
    Default fallback models (Ollama/llama3.2) are excluded to avoid duplicates
    with the static catalog.

    Returns:
        List of DynamicModelInfo objects from .env configuration.
    """
    from .config import settings

    configured: list[DynamicModelInfo] = []
    seen: set[tuple[str, str]] = set()  # (provider, model_id) for dedup

    logger.debug("Discovering models from .env configuration")

    # 1. Orchestrator model — only if explicitly configured
    if settings.ORCHESTRATOR_PROVIDER and settings.ORCHESTRATOR_MODEL:
        model_info = _build_dynamic_model_from_config(
            provider=settings.ORCHESTRATOR_PROVIDER.lower(),
            model_id=settings.ORCHESTRATOR_MODEL,
            endpoint=settings.ORCHESTRATOR_ENDPOINT,
            api_key=settings.ORCHESTRATOR_API_KEY,
            source="env_orchestrator",
        )
        if model_info:
            key = (model_info.provider, model_info.model_id)
            if key not in seen:
                logger.debug(f"Discovered orchestrator model: {model_info.provider}:{model_info.model_id}")
                configured.append(model_info)
                seen.add(key)

    # 2. Cypher model — only if explicitly configured
    if settings.CYPHER_PROVIDER and settings.CYPHER_MODEL:
        model_info = _build_dynamic_model_from_config(
            provider=settings.CYPHER_PROVIDER.lower(),
            model_id=settings.CYPHER_MODEL,
            endpoint=settings.CYPHER_ENDPOINT,
            api_key=settings.CYPHER_API_KEY,
            source="env_cypher",
        )
        if model_info:
            key = (model_info.provider, model_info.model_id)
            if key not in seen:
                logger.debug(f"Discovered cypher model: {model_info.provider}:{model_info.model_id}")
                configured.append(model_info)
                seen.add(key)

    # 3. Worker LLMs
    worker_llms_list = settings.active_worker_llms
    logger.debug(f"Processing {len(worker_llms_list)} worker LLM configs")
    for worker_config in worker_llms_list:
        model_info = _build_dynamic_model_from_config(
            provider=worker_config.provider,
            model_id=worker_config.model_id,
            endpoint=worker_config.endpoint,
            api_key=worker_config.api_key,
            source="env_worker",
        )
        if model_info:
            key = (model_info.provider, model_info.model_id)
            if key not in seen:
                logger.debug(f"Discovered worker model: {model_info.provider}:{model_info.model_id}")
                configured.append(model_info)
                seen.add(key)

    logger.info(f"Discovered {len(configured)} model(s) from .env configuration")
    return configured


def _build_dynamic_model_from_config(
    provider: str,
    model_id: str,
    endpoint: str | None,
    api_key: str | None,
    source: Literal["env_orchestrator", "env_cypher", "env_worker"],
) -> DynamicModelInfo | None:
    """Build a DynamicModelInfo from configuration values.

    For unknown providers, treats them as OpenAI-compatible endpoints.
    """
    from .providers.base import PROVIDER_REGISTRY, get_provider
    from . import constants as cs

    # Handle unknown providers by treating them as OpenAI-compatible
    effective_provider = provider
    if provider not in PROVIDER_REGISTRY:
        logger.warning(
            f"Provider '{provider}' not in registry. Treating as OpenAI-compatible endpoint."
        )
        effective_provider = cs.Provider.OPENAI

    # Resolve context window via the provider's get_model_context_window()
    try:
        provider_instance = get_provider(effective_provider)
        context_window = provider_instance.get_model_context_window(model_id)
    except Exception as e:
        logger.debug(
            f"Context window resolution failed for {provider}:{model_id}: {e}. "
            "Using fallback 256000."
        )
        context_window = 256000  # Fallback

    # Check if API key is available
    has_api_key = False
    if api_key and api_key.strip() and api_key != cs.DEFAULT_API_KEY:
        has_api_key = True
    else:
        # Check environment for the provider's API key
        # Use original provider name (not effective_provider) to look up the correct env var
        # For unknown providers like "dashscope", this ensures we check DASHSCOPE_API_KEY
        # rather than OPENAI_API_KEY
        from .config import API_KEY_INFO, LOCAL_PROVIDERS

        info = API_KEY_INFO.get(provider)
        if info and os.environ.get(info["env_var"]):
            has_api_key = True
        elif provider in LOCAL_PROVIDERS:
            has_api_key = True
        # Special case for Google Vertex (uses project_id instead of API key)
        elif provider == "google" and (
            os.environ.get("GOOGLE_PROJECT_ID") or
            os.environ.get("ORCHESTRATOR_PROJECT_ID") or
            os.environ.get("CYPHER_PROJECT_ID")
        ):
            has_api_key = True
        # For unknown providers not in API_KEY_INFO, check {PROVIDER}_API_KEY format
        elif provider not in API_KEY_INFO:
            env_var_name = f"{provider.upper()}_API_KEY"
            if os.environ.get(env_var_name):
                has_api_key = True

    # Generate display name
    display_name = _generate_display_name(model_id)

    logger.debug(
        f"Built dynamic model: {provider}:{model_id} "
        f"(configured={has_api_key}, source={source})"
    )

    # Determine if this is a local provider using the original provider name
    from .config import LOCAL_PROVIDERS

    is_local_provider = provider in LOCAL_PROVIDERS

    return DynamicModelInfo(
        provider=provider,
        model_id=model_id,
        display_name=display_name,
        context_window=context_window,
        description="",
        requires_api_key=not is_local_provider,
        is_local=is_local_provider,
        pricing_tier="medium",
        is_configured=has_api_key,
        endpoint=endpoint,
        source=source,
    )


def _generate_display_name(model_id: str) -> str:
    """Generate a human-readable display name from a model ID.

    Examples:
        qwen3-max → Qwen3 Max
        glm-5.1 → Glm 5.1
        MiMo-V2-Pro → MiMo V2 Pro
        gpt-4o → Gpt 4o
    """
    # Replace separators with spaces, then title-case keeping known acronyms
    name = model_id.replace("-", " ").replace("_", " ")
    parts = name.split()
    result = []
    for part in parts:
        # Keep all-uppercase parts as-is (e.g., "MiMo" stays "MiMo", "GPT" stays "GPT")
        if part.isupper() and len(part) > 1:
            result.append(part)
        else:
            result.append(part.capitalize())
    return " ".join(result)


def build_dynamic_model_catalog() -> dict[str, list[DynamicModelInfo]]:
    """Build unified model catalog combining static and dynamic (.env) models.

    All entries are converted to DynamicModelInfo for uniform processing.
    Context windows are resolved dynamically via get_model_context_window()
    at build time (env var overrides and MODEL_CONTEXT_WINDOWS dict are applied).

    Configuration options (see .env.example):
    - CGR_MODEL_CATALOG_PATH: Load catalog from external JSON/YAML file instead of static catalog
    - CGR_DISABLE_MODEL_DISCOVERY: Skip dynamic model discovery from .env configuration

    Priority order:
    1. .env configured models (validated and working) — marked is_configured=True
    2. Static catalog models (for discovery) — marked is_configured based on API key availability

    Returns:
        Dictionary mapping providers to lists of DynamicModelInfo objects
    """
    logger.debug("Building dynamic model catalog")
    from .config import settings
    from .providers.base import get_provider

    # 1. Load base catalog (external file or static)
    catalog: dict[str, list[DynamicModelInfo]] = {}
    used_external = False

    if settings.CGR_MODEL_CATALOG_PATH:
        try:
            external_catalog = _load_external_catalog(settings.CGR_MODEL_CATALOG_PATH)
            logger.info(f"Using external catalog from {settings.CGR_MODEL_CATALOG_PATH}")

            # Convert external catalog ModelInfo entries to DynamicModelInfo with resolved context windows
            for provider_key, models in external_catalog.items():
                try:
                    provider_instance = get_provider(provider_key)
                except Exception:
                    # If provider can't be instantiated, use static context windows
                    provider_instance = None

                catalog[provider_key] = [
                    DynamicModelInfo.from_model_info(
                        m,
                        is_configured=_check_api_key_available(m),
                        context_window=(
                            provider_instance.get_model_context_window(m.model_id)
                            if provider_instance
                            else m.context_window
                        ),
                        source="external",
                    )
                    for m in models
                ]

            used_external = True
        except Exception as e:
            logger.error(
                f"Failed to load external catalog from {settings.CGR_MODEL_CATALOG_PATH}: {e}. "
                "Falling back to static catalog."
            )
            # Fall through to static catalog loading
            catalog = {}

    # If external catalog not set or failed, load static catalog
    if not catalog:
        for provider_key, models in MODEL_CATALOG.items():
            try:
                provider_instance = get_provider(provider_key)
            except Exception:
                # If provider can't be instantiated, use static context windows
                provider_instance = None

            catalog[provider_key] = [
                DynamicModelInfo.from_model_info(
                    m,
                    is_configured=_check_api_key_available(m),
                    context_window=(
                        provider_instance.get_model_context_window(m.model_id)
                        if provider_instance
                        else m.context_window
                    ),
                    source="static",
                )
                for m in models
            ]

    # 2. Add/override with .env configured models (unless disabled)
    if not settings.CGR_DISABLE_MODEL_DISCOVERY:
        configured_models = _extract_configured_models_from_env()

        for model_info in configured_models:
            provider = model_info.provider
            if provider not in catalog:
                catalog[provider] = []

            # Check if this exact model already exists in catalog
            existing_idx = _find_model_in_list(catalog[provider], model_info.model_id)
            if existing_idx >= 0:
                # Update existing entry with dynamic info (preserving is_configured=True)
                catalog[provider][existing_idx] = model_info
            else:
                # Add new model to provider list
                catalog[provider].append(model_info)
    else:
        logger.debug("Dynamic model discovery disabled (CGR_DISABLE_MODEL_DISCOVERY=True)")

    total_models = sum(len(v) for v in catalog.values())
    source = "external" if used_external else "static"
    discovery = "enabled" if not settings.CGR_DISABLE_MODEL_DISCOVERY else "disabled"
    logger.debug(
        f"Built dynamic catalog with {total_models} model(s) across {len(catalog)} provider(s) "
        f"(source: {source}, discovery: {discovery})"
    )
    return catalog
