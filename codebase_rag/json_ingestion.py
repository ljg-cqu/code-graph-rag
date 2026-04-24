from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import jsonschema
from loguru import logger

from . import constants as cs
from .config import settings
from .embedder import EmbeddingCache, get_embedding_provider_instance
from .embeddings.base import EmbeddingProvider
from .schemas import IngestionResult, JSONEntity, JSONRelationship, UpdateResult
from .services.graph_service import MemgraphIngestor

__all__ = [
    "ingest_json_data",
    "delete_dataset",
    "delete_entities_by_source_file",
    "handle_json_update_event",
    "validate_json_input",
    "load_json_files",
    "recreate_json_vector_index",
    "are_json_embeddings_available",
    "IngestionResult",
    "UpdateResult",
]

config = settings

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ingestion_schema.json"
)
with open(SCHEMA_PATH, encoding="utf-8") as schema_file:
    INGESTION_SCHEMA = json.load(schema_file)

JSON_ENTITY_LABEL = "JsonEntity"
EMBEDDING_VERSION = 1

# Lazy-initialized embedding provider and cache
_embedding_provider: EmbeddingProvider | None = None
_embedding_cache: EmbeddingCache | None = None


def _get_embedding_provider() -> EmbeddingProvider:
    """Get the embedding provider instance lazily."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = get_embedding_provider_instance()
    return _embedding_provider


def _get_embedding_cache() -> EmbeddingCache:
    """Get the embedding cache instance lazily."""
    global _embedding_cache
    if _embedding_cache is None:
        provider = _get_embedding_provider()
        _embedding_cache = EmbeddingCache(
            dimension=getattr(
                provider, "dimension", settings.get_effective_vector_dim("json")
            )
        )
    return _embedding_cache


def are_json_embeddings_available() -> tuple[bool, str | None]:
    """Check if JSON embeddings are available.

    Returns:
        Tuple of (is_available, reason_if_not_available)
    """
    if not settings.JSON_EMBEDDINGS_ENABLED:
        return False, "JSON embeddings disabled via configuration"

    try:
        _get_embedding_provider()
        return True, None
    except Exception as e:
        return False, f"Embedding provider not available: {e}"


@dataclass(frozen=True)
class PreparedJsonFile:
    path: Path
    dataset_id: str
    metadata: dict[str, Any]
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


@dataclass
class DatasetReferences:
    ids: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    ambiguous_names: set[str] = field(default_factory=set)


@dataclass
class OperationSummary:
    ingested: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _create_json_ingestor(batch_size: int) -> MemgraphIngestor:
    effective_batch_size = batch_size or settings.JSON_MEMGRAPH_BATCH_SIZE
    return MemgraphIngestor(
        host=settings.JSON_MEMGRAPH_HOST,
        port=settings.JSON_MEMGRAPH_PORT,
        batch_size=effective_batch_size,
        username=settings.JSON_MEMGRAPH_USERNAME,
        password=settings.JSON_MEMGRAPH_PASSWORD,
    )


def _canonical_entity_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return slug or "entity"


def _build_unique_id(dataset_id: str, entity_id: str) -> str:
    return f"{dataset_id}::{entity_id}"


def _escape_identifier(identifier: str) -> str:
    return identifier.replace("`", "``")


def _split_relationship_targets(target: str) -> list[str]:
    if "," not in target:
        return [target.strip()]

    targets = [part.strip() for part in target.split(",") if part.strip()]
    return targets or [target.strip()]


def _batch_operation(data: dict[str, Any]) -> str:
    raw_operation = data.get("operation") or data.get("metadata", {}).get("operation")
    if raw_operation is None:
        return "add"
    return str(raw_operation).strip().lower() or "add"


def _batch_last_updated(data: dict[str, Any]) -> str | None:
    raw_last_updated = data.get("last_updated") or data.get("metadata", {}).get(
        "last_updated"
    )
    if raw_last_updated is None:
        return None
    value = str(raw_last_updated).strip()
    return value or None


def _metadata_properties(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"dataset_id", "default_entity_labels"}
    }


def _extract_relationships(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_relationships = data.get("relationships", [])
    if isinstance(raw_relationships, dict):
        relationship_items = raw_relationships.get("relationships", [])
    else:
        relationship_items = raw_relationships

    normalized_relationships: list[dict[str, Any]] = []
    for relationship in relationship_items:
        target = relationship.get("target")
        if not isinstance(target, str):
            normalized_relationships.append(copy.deepcopy(relationship))
            continue

        for normalized_target in _split_relationship_targets(target):
            normalized_relationship = copy.deepcopy(relationship)
            normalized_relationship["target"] = normalized_target
            normalized_relationships.append(normalized_relationship)

    return normalized_relationships


def _normalize_entities(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    default_labels = list(data.get("metadata", {}).get("default_entity_labels") or [])
    batch_operation = _batch_operation(data)
    batch_last_updated = _batch_last_updated(data)
    seen_ids: set[str] = set()

    for index, entity in enumerate(data.get("entities", [])):
        if entity.get("properties") is None:
            entity["properties"] = {}

        if not entity.get("id") and entity.get("name"):
            entity["id"] = _canonical_entity_id(str(entity["name"]))

        if default_labels and not entity.get("labels"):
            entity["labels"] = list(default_labels)

        if not entity.get("operation"):
            entity["operation"] = batch_operation

        if batch_last_updated and not entity.get("last_updated"):
            entity["last_updated"] = batch_last_updated

        entity_id = entity.get("id")
        if entity_id in seen_ids:
            errors.append(f"Duplicate entity ID detected: {entity_id}")
        elif entity_id is not None:
            seen_ids.add(str(entity_id))

        try:
            JSONEntity(**entity)
        except Exception as exc:
            errors.append(f"Entity {index} validation failed: {exc}")

    return errors


def _normalize_relationships(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    batch_operation = _batch_operation(data)
    batch_last_updated = _batch_last_updated(data)
    relationships = _extract_relationships(data)
    data["relationships"] = relationships

    for index, relationship in enumerate(relationships):
        if relationship.get("properties") is None:
            relationship["properties"] = {}

        if not relationship.get("operation"):
            relationship["operation"] = batch_operation

        if batch_last_updated and not relationship.get("last_updated"):
            relationship["last_updated"] = batch_last_updated

        try:
            JSONRelationship(**relationship)
        except Exception as exc:
            errors.append(f"Relationship {index} validation failed: {exc}")

    return errors


def _validate_relationship_references(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    entity_refs = {
        str(entity_ref)
        for entity in entities
        for entity_ref in (entity.get("id"), entity.get("name"))
        if entity_ref is not None
    }

    for relationship in relationships:
        source = str(relationship["source"])
        target = str(relationship["target"])
        if source not in entity_refs:
            errors.append(
                f"Relationship source references non-existent entity: {source}"
            )
        if target not in entity_refs:
            errors.append(
                f"Relationship target references non-existent entity: {target}"
            )

    return errors


def validate_json_input(
    data: dict[str, Any],
    allow_external_references: bool = False,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        normalized_data = copy.deepcopy(data)
        jsonschema.validate(instance=normalized_data, schema=INGESTION_SCHEMA)

        errors.extend(_normalize_entities(normalized_data))
        errors.extend(_normalize_relationships(normalized_data))

        if not allow_external_references:
            errors.extend(
                _validate_relationship_references(
                    normalized_data.get("entities", []),
                    normalized_data.get("relationships", []),
                )
            )

        if errors:
            return False, None, errors

        return True, normalized_data, []
    except jsonschema.exceptions.ValidationError as exc:
        errors.append(f"Schema validation failed: {exc}")
        return False, None, errors
    except Exception as exc:
        errors.append(f"Validation error: {exc}")
        return False, None, errors


def load_json_files(
    input_path: str, exclude_patterns: list[str] | None = None
) -> list[tuple[Path, dict[str, Any]]]:
    json_files, load_errors, skip_count = _load_json_files_with_errors(
        input_path, exclude_patterns
    )
    if skip_count:
        logger.debug(f"Skipped {skip_count} non-entity JSON file(s)")
    for error in load_errors:
        logger.warning(error)
    return json_files


class JSONFilePurpose(StrEnum):
    """Classification of JSON file purposes for ingestion."""

    ENTITY_DATA = "entity_data"
    JSON_SCHEMA = "json_schema"
    CONFIG = "config"
    PACKAGE_METADATA = "package_metadata"
    CACHE = "cache"
    TELEMETRY = "telemetry"
    VERSION = "version"
    RAW_ARRAY = "raw_array"
    METADATA_ONLY = "metadata_only"
    RELATIONSHIP_DOCS = "relationship_docs"
    DATA_WRAPPER = "data_wrapper"
    GRAPH_FORMAT = "graph_format"
    SINGLE_ENTITY = "single_entity"
    UNKNOWN = "unknown"


def _detect_json_purpose(data: Any, file_path: Path) -> tuple[JSONFilePurpose, str | None]:
    """
    Detect the purpose/type of a JSON file and provide guidance.

    Returns:
        Tuple of (purpose_type, guidance_message)
    """
    if not isinstance(data, dict):
        if isinstance(data, list):
            return JSONFilePurpose.RAW_ARRAY, "This appears to be a raw JSON array. For ingestion, wrap it in an object with 'entities' and optional 'metadata' fields."
        return JSONFilePurpose.UNKNOWN, "Unrecognized JSON format. Expected an object with 'entities' and/or 'relationships' arrays."

    # Check for schema definition
    if "$schema" in data or "definitions" in data or "properties" in data:
        return JSONFilePurpose.JSON_SCHEMA, "This appears to be a JSON Schema file, not entity data. Schema files define structure but don't contain ingestable entities."

    # Check for configuration
    if any(k in data for k in ["config", "settings", "options", "parameters"]):
        return JSONFilePurpose.CONFIG, "This appears to be a configuration file, not entity data. Configuration files control behavior but don't define entities."

    # Check for metadata-only
    if "metadata" in data and "entities" not in data and "relationships" not in data:
        return JSONFilePurpose.METADATA_ONLY, "This file contains only metadata. For ingestion, add an 'entities' array with the actual entity definitions."

    # Check for valid ingestion payload
    has_entities = "entities" in data and isinstance(data.get("entities"), list)
    has_relationships = "relationships" in data and isinstance(data.get("relationships"), list)

    if has_entities or has_relationships:
        return JSONFilePurpose.ENTITY_DATA, None  # Valid format, no guidance needed

    # Check for relationship documentation (common in this repo)
    if "relationships" in data and not isinstance(data.get("relationships"), list):
        return JSONFilePurpose.RELATIONSHIP_DOCS, "This appears to document relationships but not in the ingestable format. The 'relationships' field should be an array of relationship objects."

    # Check for other common patterns
    if "data" in data and isinstance(data.get("data"), list):
        return JSONFilePurpose.DATA_WRAPPER, "This has a 'data' array. For ingestion, rename 'data' to 'entities' or wrap the array appropriately."

    if "nodes" in data and "links" in data:
        return JSONFilePurpose.GRAPH_FORMAT, "This appears to be a graph format (nodes/links). For ingestion, rename 'nodes' to 'entities' and 'links' to 'relationships'."

    if "name" in data and "description" in data and len(data) <= 5:
        return JSONFilePurpose.SINGLE_ENTITY, "This appears to be a single entity object. For ingestion, wrap it in an 'entities' array."

    return JSONFilePurpose.UNKNOWN, f"Unrecognized JSON format. Keys found: {list(data.keys())[:5]}. Expected 'entities' and/or 'relationships' arrays."


def _load_json_files_with_errors(
    input_path: str, exclude_patterns: list[str] | None = None, filter_preset: str = "lenient"
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str], int]:
    path = Path(input_path)
    json_files: list[tuple[Path, dict[str, Any]]] = []
    load_errors: list[str] = []
    skip_count = 0

    # Common JSON files to exclude by default
    # Includes: virtual environments, package managers, IDE files, build artifacts,
    # and known non-CGR JSON patterns (AWS SDK metadata, etc.)
    # Note: Patterns use fnmatch syntax. For directory matching, the pattern should
    # match the relative path from the base directory.
    default_exclude_patterns = {
        # Virtual environment directories (match any depth)
        ".venv/*",
        ".venv/**/*",
        "venv/*",
        "venv/**/*",
        "env/*",
        "env/**/*",
        "*/.venv/*",
        "*/venv/*",
        "*/env/*",
        # pip packages
        "site-packages/*",
        "site-packages/**/*",
        "*/site-packages/*",
        "*/site-packages/**/*",
        "dist-packages/*",
        "dist-packages/**/*",
        "*/dist-packages/*",
        "*/dist-packages/**/*",
        # Package manager directories
        "node_modules/*",
        "node_modules/**/*",
        "*/node_modules/*",
        "*/node_modules/**/*",
        ".poetry/*",
        ".poetry/**/*",
        "*/.poetry/*",
        "*/.poetry/**/*",
        ".cache/pip/*",
        ".cache/pip/**/*",
        "*/.cache/pip/*",
        "*/.cache/pip/**/*",
        "package-lock.json",
        "*/package-lock.json",
        "yarn.lock",
        "*/yarn.lock",
        "pnpm-lock.yaml",
        "*/pnpm-lock.yaml",
        "poetry.lock",
        "*/poetry.lock",
        # Build artifacts
        "*.min.json",
        "tsconfig*.json",
        "*/tsconfig*.json",
        ".eslintrc*.json",
        "*/.eslintrc*.json",
        "prettierrc*.json",
        "*/prettierrc*.json",
        ".vscode/*",
        ".vscode/**/*",
        "*/.vscode/*",
        "*/.vscode/**/*",
        ".idea/*",
        ".idea/**/*",
        "*/.idea/*",
        "*/.idea/**/*",
        "build/*",
        "build/**/*",
        "*/build/*",
        "*/build/**/*",
        "dist/*",
        "dist/**/*",
        "*/dist/*",
        "*/dist/**/*",
        "coverage/*",
        "coverage/**/*",
        "*/coverage/*",
        "*/coverage/**/*",
        "*.egg-info/*",
        "*.egg-info/**/*",
        "*/*.egg-info/*",
        "*/*.egg-info/**/*",
        "benchmarks/results/*",
        "benchmarks/results/**/*",
        "*/benchmarks/results/*",
        "*/benchmarks/results/**/*",
        # Known non-CGR JSON patterns (AWS SDK metadata, etc.)
        "paginators-*.json",
        "*/paginators-*.json",
        "examples-*.json",
        "*/examples-*.json",
        "service-*.json",
        "*/service-*.json",
        "waiters-*.json",
        "*/waiters-*.json",
        "resources-*.json",
        "*/resources-*.json",
        "endpoints-*.json",
        "*/endpoints-*.json",
        "tools/*",
        "tools/**/*",
        "*/tools/*",
        "*/tools/**/*",
        # Note: .cgr, .embedding_cache, .tmp_cache_*, .cgr-* handled by should_exclude()
    }

    # Apply filter preset
    if filter_preset == "none":
        # No default filtering - only use user-provided patterns
        all_exclude_patterns: set[str] = set()
        logger.debug("JSON filter preset 'none': skipping all default filters")
    elif filter_preset == "strict":
        # Strict mode: only process .cgr.json files
        # We'll handle this specially in should_exclude
        all_exclude_patterns = default_exclude_patterns.copy()
        logger.debug("JSON filter preset 'strict': only processing .cgr.json files")
    else:
        # lenient (default): use default patterns
        all_exclude_patterns = default_exclude_patterns.copy()

    # Combine with user-provided excludes
    if exclude_patterns:
        all_exclude_patterns.update(set(exclude_patterns))

    base_path = path if path.is_dir() else path.parent

    def looks_like_ingestion_payload(data: Any) -> bool:
        return isinstance(data, dict) and "entities" in data

    def should_exclude(file_path: Path) -> str | None:
        if filter_preset == "strict":
            if not file_path.name.endswith(".cgr.json"):
                return "strict_mode"

        if file_path.name.startswith(".tmp_cache_") and file_path.suffix == ".json":
            return "tmp_cache"

        if file_path.name.startswith(".cgr-") and file_path.suffix == ".json":
            return "cgr_internal"

        if any(
            part == ".embedding_cache" or part.endswith(".egg-info")
            for part in file_path.parts
        ):
            return "embedding_cache_or_egg_info"

        for part in file_path.parts[:-1]:
            if part.startswith(".") and part != ".cgr":
                return "hidden_directory"

        try:
            relative_path = str(file_path.relative_to(base_path))
        except ValueError:
            relative_path = str(file_path)

        for pattern in all_exclude_patterns:
            if fnmatch(relative_path, pattern):
                return f"pattern:{pattern}"
        return None

    if path.is_file() and path.suffix == ".json":
        exclude_reason = should_exclude(path)
        if exclude_reason:
            logger.debug(
                cs.JSON_INGEST_SKIP_EXCLUDED.format(path=path, reason=exclude_reason)
            )
            return [], [], 1
        try:
            with open(path, encoding="utf-8") as json_file:
                data = json.load(json_file)
                purpose, guidance = _detect_json_purpose(data, path)
                if purpose == JSONFilePurpose.ENTITY_DATA:
                    json_files.append((path, data))
                elif guidance:
                    skip_count += 1
                    logger.debug(f"Skipping {path}: {guidance}")
        except json.JSONDecodeError as exc:
            skip_count += 1
            logger.debug(cs.JSON_INGEST_SKIP_PARSE_ERROR.format(path=path, error=exc))
            load_errors.append(f"Skipping invalid JSON file {path}: {exc}")
        except OSError as exc:
            skip_count += 1
            logger.debug(cs.JSON_INGEST_SKIP_IO_ERROR.format(path=path, error=exc))
            load_errors.append(f"Skipping unreadable JSON file {path}: {exc}")
    elif path.is_dir():
        skipped_files_with_guidance: list[tuple[Path, str, JSONFilePurpose]] = []
        exclude_reason_counts: dict[str, int] = {}
        for file_path in path.rglob("*.json"):
            exclude_reason = should_exclude(file_path)
            if exclude_reason:
                exclude_reason_counts[exclude_reason] = exclude_reason_counts.get(exclude_reason, 0) + 1
                continue
            try:
                with open(file_path, encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    purpose, guidance = _detect_json_purpose(data, file_path)
                    if purpose == JSONFilePurpose.ENTITY_DATA:
                        json_files.append((file_path, data))
                    elif guidance:
                        skipped_files_with_guidance.append((file_path, guidance, purpose))
            except json.JSONDecodeError as exc:
                skip_count += 1
                logger.debug(cs.JSON_INGEST_SKIP_PARSE_ERROR.format(path=file_path, error=exc))
                load_errors.append(f"Skipping invalid JSON file {file_path}: {exc}")
            except OSError as exc:
                skip_count += 1
                logger.debug(cs.JSON_INGEST_SKIP_IO_ERROR.format(path=file_path, error=exc))
                load_errors.append(f"Skipping unreadable JSON file {file_path}: {exc}")

        if exclude_reason_counts:
            summary = ", ".join(
                f"{count} {reason}" for reason, count in sorted(exclude_reason_counts.items())
            )
            logger.info(cs.JSON_INGEST_SKIP_REASON_SUMMARY.format(reasons=summary))
            skip_count += sum(exclude_reason_counts.values())

        if skipped_files_with_guidance:
            purpose_counts: dict[str, int] = {}
            for fp, guidance, purpose in skipped_files_with_guidance:
                purpose_counts[purpose.value] = purpose_counts.get(purpose.value, 0) + 1
                logger.debug(f"Skipping {fp}: {guidance}")

            if purpose_counts:
                summary = ", ".join(f"{count} {purpose.replace('_', ' ')}"
                                   for purpose, count in sorted(purpose_counts.items()))
                logger.info(f"JSON files skipped: {summary}")
            skip_count += len(skipped_files_with_guidance)
    else:
        raise ValueError(
            f"Invalid input path: {input_path} (must be .json file or directory containing JSON files)"
        )

    return json_files, load_errors, skip_count


def generate_embeddings_for_entities(
    entities: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], list[str]]:
    """Generate embeddings for entities with graceful fallback.

    Returns:
        Tuple of (entity_id_to_embedding_dict, warnings_list)
    """
    embeddings: dict[str, list[float]] = {}
    warnings: list[str] = []

    available, reason = are_json_embeddings_available()
    if not available:
        msg = f"Embeddings not generated: {reason}"
        if settings.JSON_EMBEDDINGS_REQUIRED:
            raise RuntimeError(msg)
        warnings.append(msg)
        return embeddings, warnings

    texts: list[str] = []
    entity_ids: list[str] = []
    for entity in entities:
        if str(entity.get("operation") or "add").lower() == "delete":
            continue
        entity_id = str(entity["id"])
        name = str(entity.get("name", ""))
        description = str(entity.get("properties", {}).get("description") or name)
        texts.append(f"{name} - {description}")
        entity_ids.append(entity_id)

    if not texts:
        return embeddings, warnings

    try:
        cache = _get_embedding_cache()
        cached_embeddings = cache.get_many(texts)
        uncached_texts: list[str] = []
        uncached_ids: list[str] = []

        for index, (text, entity_id) in enumerate(zip(texts, entity_ids)):
            if index in cached_embeddings:
                embeddings[entity_id] = cached_embeddings[index]
            else:
                uncached_texts.append(text)
                uncached_ids.append(entity_id)

        if uncached_texts:
            provider = _get_embedding_provider()
            generated_embeddings = provider.embed_batch(uncached_texts)
            for index, (entity_id, embedding) in enumerate(
                zip(uncached_ids, generated_embeddings)
            ):
                embeddings[entity_id] = embedding
                cache.put(uncached_texts[index], embedding)
    except Exception as exc:
        msg = f"Embedding generation failed: {exc}"
        if settings.JSON_EMBEDDINGS_REQUIRED:
            raise RuntimeError(msg) from exc
        warnings.append(msg)

    return embeddings, warnings


def _prepare_json_file(
    file_data: tuple[Path, dict[str, Any]],
    dataset_id: str | None,
    metadata_override: dict[str, Any] | None,
) -> tuple[PreparedJsonFile | None, list[str]]:
    file_path, data = file_data
    normalized_data = copy.deepcopy(data)
    metadata = normalized_data.setdefault("metadata", {})

    if metadata_override:
        metadata.update(metadata_override)
    if dataset_id:
        metadata["dataset_id"] = dataset_id

    valid, validated_data, errors = validate_json_input(
        normalized_data,
        allow_external_references=True,
    )
    if not valid or validated_data is None:
        return None, [f"{file_path}: {error}" for error in errors]

    current_dataset_id = str(validated_data["metadata"]["dataset_id"])
    prepared = PreparedJsonFile(
        path=file_path,
        dataset_id=current_dataset_id,
        metadata=validated_data.get("metadata", {}),
        entities=validated_data.get("entities", []),
        relationships=validated_data.get("relationships", []),
    )
    return prepared, []


def _validate_prepared_files(prepared_files: list[PreparedJsonFile]) -> list[str]:
    errors: list[str] = []
    dataset_entity_sources: dict[str, dict[str, Path]] = defaultdict(dict)
    dataset_entity_refs: dict[str, set[str]] = defaultdict(set)
    dataset_name_counts: dict[str, dict[str, int]] = defaultdict(dict)

    for prepared_file in prepared_files:
        for entity in prepared_file.entities:
            entity_id = str(entity["id"])
            entity_name = str(entity["name"])
            current_path = prepared_file.path
            existing_path = dataset_entity_sources[prepared_file.dataset_id].get(
                entity_id
            )
            if existing_path is not None and existing_path != current_path:
                errors.append(
                    "Duplicate entity ID across prepared files: "
                    f"dataset={prepared_file.dataset_id}, entity_id={entity_id}, "
                    f"files={existing_path} and {current_path}"
                )
            else:
                dataset_entity_sources[prepared_file.dataset_id][entity_id] = (
                    current_path
                )

            dataset_entity_refs[prepared_file.dataset_id].add(entity_id)
            dataset_entity_refs[prepared_file.dataset_id].add(entity_name)
            dataset_name_counts[prepared_file.dataset_id][entity_name] = (
                dataset_name_counts[prepared_file.dataset_id].get(entity_name, 0) + 1
            )

    dataset_ambiguous_names = {
        dataset_id: {name for name, count in name_counts.items() if count > 1}
        for dataset_id, name_counts in dataset_name_counts.items()
    }

    for prepared_file in prepared_files:
        entity_refs = dataset_entity_refs[prepared_file.dataset_id]
        ambiguous_names = dataset_ambiguous_names.get(prepared_file.dataset_id, set())

        for index, relationship in enumerate(prepared_file.relationships, start=1):
            source = str(relationship["source"])
            target = str(relationship["target"])

            if source in ambiguous_names:
                errors.append(
                    f"{prepared_file.path}: Relationship {index} source references "
                    f"ambiguous entity name in dataset '{prepared_file.dataset_id}': {source}"
                )
            elif source not in entity_refs:
                errors.append(
                    f"{prepared_file.path}: Relationship {index} source references "
                    f"non-existent entity in dataset '{prepared_file.dataset_id}': {source}"
                )

            if target in ambiguous_names:
                errors.append(
                    f"{prepared_file.path}: Relationship {index} target references "
                    f"ambiguous entity name in dataset '{prepared_file.dataset_id}': {target}"
                )
            elif target not in entity_refs:
                errors.append(
                    f"{prepared_file.path}: Relationship {index} target references "
                    f"non-existent entity in dataset '{prepared_file.dataset_id}': {target}"
                )

    return errors


def _build_dataset_references(
    prepared_files: list[PreparedJsonFile],
) -> dict[str, DatasetReferences]:
    references_by_dataset: dict[str, DatasetReferences] = defaultdict(DatasetReferences)

    for prepared_file in prepared_files:
        references = references_by_dataset[prepared_file.dataset_id]
        for entity in prepared_file.entities:
            entity_id = str(entity["id"])
            unique_id = _build_unique_id(prepared_file.dataset_id, entity_id)
            references.ids[entity_id] = unique_id

            name = str(entity["name"])
            if name in references.ambiguous_names:
                continue

            existing_unique_id = references.names.get(name)
            if existing_unique_id is not None and existing_unique_id != unique_id:
                references.names.pop(name, None)
                references.ambiguous_names.add(name)
            else:
                references.names[name] = unique_id

    return dict(references_by_dataset)


def _fetch_existing_entities(
    graph_connection: MemgraphIngestor,
    dataset_id: str,
) -> dict[str, str | None]:
    rows = graph_connection.fetch_all(
        """
        MATCH (n:JsonEntity {dataset_id: $dataset_id})
        RETURN n.unique_id AS unique_id, n.last_updated AS last_updated
        """,
        {"dataset_id": dataset_id},
    )
    return {
        str(row["unique_id"]): (
            str(row["last_updated"]) if row.get("last_updated") is not None else None
        )
        for row in rows
        if row.get("unique_id") is not None
    }


def _fetch_existing_relationships(
    graph_connection: MemgraphIngestor,
    dataset_id: str,
) -> dict[tuple[str, str, str], str | None]:
    rows = graph_connection.fetch_all(
        """
        MATCH (s:JsonEntity {dataset_id: $dataset_id})-[r {dataset_id: $dataset_id}]->(t:JsonEntity {dataset_id: $dataset_id})
        RETURN s.unique_id AS source, type(r) AS rel_type, t.unique_id AS target, r.last_updated AS last_updated
        """,
        {"dataset_id": dataset_id},
    )
    return {
        (str(row["source"]), str(row["rel_type"]), str(row["target"])): (
            str(row["last_updated"]) if row.get("last_updated") is not None else None
        )
        for row in rows
        if row.get("source") is not None
        and row.get("rel_type") is not None
        and row.get("target") is not None
    }


def _existing_is_newer_or_equal(
    existing_last_updated: str | None,
    incoming_last_updated: str | None,
) -> bool:
    if existing_last_updated is None or incoming_last_updated is None:
        return False
    return existing_last_updated >= incoming_last_updated


def _entity_labels(entity: dict[str, Any]) -> list[str]:
    entity_type = str(entity.get("type") or "Entity")
    labels = [JSON_ENTITY_LABEL, entity_type, *(entity.get("labels") or [])]
    return list(dict.fromkeys(label for label in labels if label))


def _entity_properties(
    dataset_id: str,
    entity: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    entity_id = str(entity["id"])
    name = str(entity["name"])
    entity_type = str(entity.get("type") or "Entity")
    labels = _entity_labels(entity)
    properties = dict(_metadata_properties(metadata))
    properties.update(copy.deepcopy(entity.get("properties") or {}))
    properties["id"] = entity_id
    properties["entity_id"] = entity_id
    properties["unique_id"] = _build_unique_id(dataset_id, entity_id)
    properties["name"] = name
    properties["type"] = entity_type
    properties["dataset_id"] = dataset_id
    properties["entity_labels"] = labels

    if entity.get("last_updated"):
        properties["last_updated"] = str(entity["last_updated"])

    return properties


def _relationship_properties(
    dataset_id: str,
    relationship: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    properties = dict(_metadata_properties(metadata))
    properties.update(copy.deepcopy(relationship.get("properties") or {}))
    properties["dataset_id"] = dataset_id

    if relationship.get("id"):
        properties["relationship_id"] = str(relationship["id"])
    if relationship.get("last_updated"):
        properties["last_updated"] = str(relationship["last_updated"])
    if relationship.get("confidence") is not None:
        properties["confidence"] = relationship["confidence"]
    if relationship.get("explanation"):
        properties["explanation"] = str(relationship["explanation"])
    if relationship.get("isInferred") is not None:
        properties["isInferred"] = bool(relationship["isInferred"])

    return properties


def _merge_entity(
    graph_connection: MemgraphIngestor,
    dataset_id: str,
    entity: dict[str, Any],
    metadata: dict[str, Any],
    embedding: list[float] | None,
) -> None:
    labels = _entity_labels(entity)
    label_string = ":".join(f"`{_escape_identifier(label)}`" for label in labels)
    query_lines = [
        f"MERGE (n:{label_string} {{unique_id: $unique_id}})",
        "SET n += $properties",
    ]
    params: dict[str, Any] = {
        "unique_id": _build_unique_id(dataset_id, str(entity["id"])),
        "properties": _entity_properties(dataset_id, entity, metadata),
    }

    if embedding is not None:
        query_lines.extend(
            [
                "SET n.embedding = $embedding",
                "SET n.embedding_model = $embedding_model",
                "SET n.embedding_version = $embedding_version",
            ]
        )
        params["embedding"] = embedding
        params["embedding_model"] = settings.EMBEDDING_MODEL
        params["embedding_version"] = EMBEDDING_VERSION

    query_lines.append("RETURN id(n) AS node_id")
    graph_connection.fetch_all("\n".join(query_lines), params)


def ingest_entities(
    dataset_id: str,
    entities: list[dict[str, Any]],
    entity_embeddings: dict[str, list[float]],
    metadata: dict[str, Any],
    skip_existing: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
    graph_connection: MemgraphIngestor | None = None,
) -> OperationSummary:
    summary = OperationSummary()
    existing_nodes: dict[str, str | None] = {}

    if graph_connection is not None and (not dry_run or skip_existing or incremental):
        try:
            existing_nodes = _fetch_existing_entities(graph_connection, dataset_id)
        except Exception as exc:
            logger.warning(f"Could not fetch existing JSON entities: {exc}")

    for entity in entities:
        operation = str(entity.get("operation") or "add").lower()
        unique_id = _build_unique_id(dataset_id, str(entity["id"]))
        entity_last_updated = (
            str(entity["last_updated"]) if entity.get("last_updated") else None
        )
        exists = unique_id in existing_nodes

        if operation == "delete":
            if dry_run:
                summary.deleted += 1
                continue
            if graph_connection is None:
                summary.failed += 1
                summary.errors.append(
                    f"Failed to delete entity {entity['id']}: no graph connection"
                )
                continue
            try:
                rows = graph_connection.fetch_all(
                    """
                    MATCH (n:JsonEntity {unique_id: $unique_id, dataset_id: $dataset_id})
                    WITH collect(n) AS nodes
                    FOREACH (node IN nodes | DETACH DELETE node)
                    RETURN size(nodes) AS deleted
                    """,
                    {"unique_id": unique_id, "dataset_id": dataset_id},
                )
                deleted_count = int(rows[0].get("deleted", 0)) if rows else 0
                if deleted_count > 0:
                    summary.deleted += deleted_count
                else:
                    summary.skipped += 1
            except Exception as exc:
                summary.failed += 1
                summary.errors.append(f"Failed to delete entity {entity['id']}: {exc}")
            continue

        if skip_existing and exists:
            summary.skipped += 1
            continue

        if (
            incremental
            and exists
            and _existing_is_newer_or_equal(
                existing_nodes.get(unique_id), entity_last_updated
            )
        ):
            summary.skipped += 1
            continue

        if dry_run:
            if exists or operation == "update":
                summary.updated += 1 if exists else 0
                summary.ingested += 0 if exists else 1
            else:
                summary.ingested += 1
            continue

        if graph_connection is None:
            summary.failed += 1
            summary.errors.append(
                f"Failed to ingest entity {entity['id']}: no graph connection"
            )
            continue

        try:
            _merge_entity(
                graph_connection,
                dataset_id,
                entity,
                metadata,
                entity_embeddings.get(str(entity["id"])),
            )
            if exists:
                summary.updated += 1
            else:
                summary.ingested += 1
        except Exception as exc:
            summary.failed += 1
            summary.errors.append(f"Failed to ingest entity {entity['id']}: {exc}")

    return summary


def _lookup_entity_reference(
    dataset_id: str,
    reference: str,
    dataset_references: DatasetReferences,
    graph_connection: MemgraphIngestor | None,
    lookup_cache: dict[str, str | None],
) -> tuple[str | None, str | None]:
    if reference in dataset_references.ids:
        return dataset_references.ids[reference], None
    if reference in dataset_references.names:
        return dataset_references.names[reference], None
    if reference in dataset_references.ambiguous_names:
        return None, f"Reference '{reference}' is ambiguous in dataset '{dataset_id}'"
    if reference in lookup_cache:
        if lookup_cache[reference] is None:
            return None, f"Could not resolve relationship reference: {reference}"
        return lookup_cache[reference], None
    if graph_connection is None:
        return None, f"Could not resolve relationship reference: {reference}"

    rows = graph_connection.fetch_all(
        """
        MATCH (n:JsonEntity {dataset_id: $dataset_id})
        WHERE n.entity_id = $reference OR n.id = $reference OR n.name = $reference
        RETURN n.unique_id AS unique_id
        LIMIT 2
        """,
        {"dataset_id": dataset_id, "reference": reference},
    )

    if len(rows) == 1 and rows[0].get("unique_id") is not None:
        resolved_unique_id = str(rows[0]["unique_id"])
        lookup_cache[reference] = resolved_unique_id
        return resolved_unique_id, None

    lookup_cache[reference] = None
    if len(rows) > 1:
        return (
            None,
            f"Reference '{reference}' is ambiguous in graph for dataset '{dataset_id}'",
        )
    return None, f"Could not resolve relationship reference: {reference}"


def ingest_relationships(
    dataset_id: str,
    relationships: list[dict[str, Any]],
    dataset_references: DatasetReferences,
    metadata: dict[str, Any],
    skip_existing: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
    graph_connection: MemgraphIngestor | None = None,
) -> OperationSummary:
    summary = OperationSummary()
    existing_relationships: dict[tuple[str, str, str], str | None] = {}
    lookup_cache: dict[str, str | None] = {}

    if graph_connection is not None and (not dry_run or skip_existing or incremental):
        try:
            existing_relationships = _fetch_existing_relationships(
                graph_connection,
                dataset_id,
            )
        except Exception as exc:
            logger.warning(f"Could not fetch existing JSON relationships: {exc}")

    for relationship in relationships:
        source_ref = str(relationship["source"])
        target_ref = str(relationship["target"])
        rel_type = str(relationship["relationship"])
        operation = str(relationship.get("operation") or "add").lower()
        relationship_last_updated = (
            str(relationship["last_updated"])
            if relationship.get("last_updated")
            else None
        )

        source_unique_id, source_error = _lookup_entity_reference(
            dataset_id,
            source_ref,
            dataset_references,
            graph_connection,
            lookup_cache,
        )
        if source_unique_id is None:
            summary.failed += 1
            summary.errors.append(
                source_error or f"Could not resolve source: {source_ref}"
            )
            continue

        target_unique_id, target_error = _lookup_entity_reference(
            dataset_id,
            target_ref,
            dataset_references,
            graph_connection,
            lookup_cache,
        )
        if target_unique_id is None:
            summary.failed += 1
            summary.errors.append(
                target_error or f"Could not resolve target: {target_ref}"
            )
            continue

        relationship_key = (source_unique_id, rel_type, target_unique_id)
        exists = relationship_key in existing_relationships

        if operation == "delete":
            if dry_run:
                summary.deleted += 1
                continue
            if graph_connection is None:
                summary.failed += 1
                summary.errors.append(
                    f"Failed to delete relationship {source_ref} -> {target_ref}: no graph connection"
                )
                continue
            try:
                rows = graph_connection.fetch_all(
                    f"""
                    MATCH (a:JsonEntity {{unique_id: $source_id, dataset_id: $dataset_id}})-[r:`{_escape_identifier(rel_type)}` {{dataset_id: $dataset_id}}]->(b:JsonEntity {{unique_id: $target_id, dataset_id: $dataset_id}})
                    WITH collect(r) AS relationships
                    FOREACH (relationship IN relationships | DELETE relationship)
                    RETURN size(relationships) AS deleted
                    """,
                    {
                        "dataset_id": dataset_id,
                        "source_id": source_unique_id,
                        "target_id": target_unique_id,
                    },
                )
                deleted_count = int(rows[0].get("deleted", 0)) if rows else 0
                if deleted_count > 0:
                    summary.deleted += deleted_count
                else:
                    summary.skipped += 1
            except Exception as exc:
                summary.failed += 1
                summary.errors.append(
                    f"Failed to delete relationship {source_ref} -> {target_ref}: {exc}"
                )
            continue

        if skip_existing and exists:
            summary.skipped += 1
            continue

        if (
            incremental
            and exists
            and _existing_is_newer_or_equal(
                existing_relationships.get(relationship_key), relationship_last_updated
            )
        ):
            summary.skipped += 1
            continue

        if dry_run:
            if exists or operation == "update":
                summary.updated += 1 if exists else 0
                summary.ingested += 0 if exists else 1
            else:
                summary.ingested += 1
            continue

        if graph_connection is None:
            summary.failed += 1
            summary.errors.append(
                f"Failed to ingest relationship {source_ref} -> {target_ref}: no graph connection"
            )
            continue

        try:
            graph_connection.fetch_all(
                f"""
                MATCH (a:JsonEntity {{unique_id: $source_id, dataset_id: $dataset_id}}),
                      (b:JsonEntity {{unique_id: $target_id, dataset_id: $dataset_id}})
                MERGE (a)-[r:`{_escape_identifier(rel_type)}` {{dataset_id: $dataset_id}}]->(b)
                SET r += $properties
                RETURN id(r) AS relationship_id
                """,
                {
                    "dataset_id": dataset_id,
                    "source_id": source_unique_id,
                    "target_id": target_unique_id,
                    "properties": _relationship_properties(
                        dataset_id,
                        relationship,
                        metadata,
                    ),
                },
            )
            if exists:
                summary.updated += 1
            else:
                summary.ingested += 1
        except Exception as exc:
            summary.failed += 1
            summary.errors.append(
                f"Failed to ingest relationship {source_ref} -> {target_ref}: {exc}"
            )

    return summary


def _ensure_json_vector_index(batch_size: int) -> None:
    recreate_json_vector_index(batch_size=batch_size)


def _find_vector_index(
    rows: list[dict[str, Any]],
    index_name: str,
) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("index_name") or "") == index_name:
            return row
    return None


def _read_vector_index_dimension(index_info: dict[str, Any] | None) -> int | None:
    if index_info is None:
        return None

    raw_dimension = index_info.get("dimension")
    if raw_dimension is None:
        return None

    try:
        return int(raw_dimension)
    except (TypeError, ValueError):
        return None


def recreate_json_vector_index(
    batch_size: int,
    dimension: int | None = None,
    clear_existing_embeddings: bool = True,
    force_recreate: bool = False,
) -> None:
    if not settings.JSON_EMBEDDINGS_ENABLED:
        logger.info("JSON embeddings disabled, skipping vector index creation")
        return

    available, reason = are_json_embeddings_available()
    if not available:
        logger.info(
            f"JSON embeddings not available ({reason}), skipping vector index creation"
        )
        return

    effective_dimension = dimension or settings.get_effective_vector_dim("json")
    index_name = settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME
    capacity = settings.JSON_MEMGRAPH_VECTOR_CAPACITY
    cypher = f"""
    CREATE VECTOR INDEX {index_name}
    ON :{JSON_ENTITY_LABEL}(embedding)
    WITH CONFIG {{
        "dimension": {effective_dimension},
        "capacity": {capacity},
        "metric": "cos"
    }};
    """

    try:
        with _create_json_ingestor(batch_size) as graph_connection:
            existing_index = None
            try:
                existing_indexes = graph_connection.fetch_all("SHOW VECTOR INDEX INFO;")
                existing_index = _find_vector_index(existing_indexes, index_name)
            except Exception:
                existing_index = None

            existing_dimension = _read_vector_index_dimension(existing_index)
            needs_recreate = force_recreate

            if existing_index is not None and not force_recreate:
                if existing_dimension == effective_dimension:
                    logger.info(f"Vector index '{index_name}' already exists")
                    return

                logger.warning(
                    f"Vector index '{index_name}' has dimension {existing_dimension}, "
                    f"recreating it for dimension {effective_dimension}"
                )
                needs_recreate = True

            if existing_index is not None and needs_recreate:
                if clear_existing_embeddings:
                    graph_connection.execute_write(
                        f"""
                        MATCH (n:{JSON_ENTITY_LABEL})
                        SET n.embedding = NULL,
                            n.embedding_model = NULL,
                            n.embedding_version = NULL
                        """,
                        {},
                    )
                try:
                    graph_connection.execute_write(
                        f"DROP VECTOR INDEX {index_name};",
                        {},
                    )
                except Exception as exc:
                    logger.debug(
                        f"Failed to drop JSON vector index '{index_name}': {exc}"
                    )

            graph_connection.execute_write(cypher, {})
            logger.info(
                f"Created JSON vector index '{index_name}' "
                f"(dim={effective_dimension}, capacity={capacity})"
            )
    except Exception as exc:
        error_message = str(exc).lower()
        if "already exists" in error_message or "duplicate" in error_message:
            logger.info(f"Vector index '{index_name}' already exists")
            return
        logger.warning(f"Failed to create JSON vector index '{index_name}': {exc}")


def _merge_summary_into_result(
    result: IngestionResult,
    summary: OperationSummary,
    is_entity_summary: bool,
) -> None:
    if is_entity_summary:
        result.entities_ingested += summary.ingested
        result.entities_updated += summary.updated
        result.entities_deleted += summary.deleted
        result.entities_skipped += summary.skipped
        result.entities_failed += summary.failed
    else:
        result.relationships_ingested += summary.ingested
        result.relationships_updated += summary.updated
        result.relationships_deleted += summary.deleted
        result.relationships_skipped += summary.skipped
        result.relationships_failed += summary.failed

    result.errors.extend(summary.errors)


def _ingest_entity_file(
    prepared_file: PreparedJsonFile,
    batch_size: int,
    skip_existing: bool,
    dry_run: bool,
    incremental: bool,
) -> OperationSummary:
    entity_embeddings: dict[str, list[float]] = {}
    summary = OperationSummary()

    if not dry_run:
        entity_embeddings, embed_warnings = generate_embeddings_for_entities(
            prepared_file.entities
        )
        for warning in embed_warnings:
            logger.warning(warning)
        summary.errors.extend(embed_warnings)

    if dry_run:
        ingest_summary = ingest_entities(
            prepared_file.dataset_id,
            prepared_file.entities,
            entity_embeddings,
            prepared_file.metadata,
            skip_existing=skip_existing,
            dry_run=True,
            incremental=incremental,
            graph_connection=None,
        )
        summary.ingested += ingest_summary.ingested
        summary.updated += ingest_summary.updated
        summary.deleted += ingest_summary.deleted
        summary.skipped += ingest_summary.skipped
        summary.failed += ingest_summary.failed
        summary.errors.extend(ingest_summary.errors)
        return summary

    with _create_json_ingestor(batch_size) as graph_connection:
        graph_connection.ensure_constraints()
        ingest_summary = ingest_entities(
            prepared_file.dataset_id,
            prepared_file.entities,
            entity_embeddings,
            prepared_file.metadata,
            skip_existing=skip_existing,
            dry_run=False,
            incremental=incremental,
            graph_connection=graph_connection,
        )

    summary.ingested += ingest_summary.ingested
    summary.updated += ingest_summary.updated
    summary.deleted += ingest_summary.deleted
    summary.skipped += ingest_summary.skipped
    summary.failed += ingest_summary.failed
    summary.errors.extend(ingest_summary.errors)
    return summary


def _ingest_relationship_file(
    prepared_file: PreparedJsonFile,
    dataset_references: DatasetReferences,
    batch_size: int,
    skip_existing: bool,
    dry_run: bool,
    incremental: bool,
) -> OperationSummary:
    if dry_run:
        return ingest_relationships(
            prepared_file.dataset_id,
            prepared_file.relationships,
            dataset_references,
            prepared_file.metadata,
            skip_existing=skip_existing,
            dry_run=True,
            incremental=incremental,
            graph_connection=None,
        )

    with _create_json_ingestor(batch_size) as graph_connection:
        graph_connection.ensure_constraints()
        return ingest_relationships(
            prepared_file.dataset_id,
            prepared_file.relationships,
            dataset_references,
            prepared_file.metadata,
            skip_existing=skip_existing,
            dry_run=False,
            incremental=incremental,
            graph_connection=graph_connection,
        )


def delete_dataset(
    dataset_id: str,
    dry_run: bool = False,
) -> tuple[bool, int, int, list[str]]:
    errors: list[str] = []
    nodes_deleted = 0
    relationships_deleted = 0

    try:
        with _create_json_ingestor(
            settings.JSON_MEMGRAPH_BATCH_SIZE
        ) as graph_connection:
            relationship_rows = graph_connection.fetch_all(
                """
                MATCH ()-[r]->()
                WHERE r.dataset_id = $dataset_id
                RETURN count(r) AS deleted
                """,
                {"dataset_id": dataset_id},
            )
            relationships_deleted = (
                int(relationship_rows[0].get("deleted", 0)) if relationship_rows else 0
            )

            node_rows = graph_connection.fetch_all(
                """
                MATCH (n:JsonEntity {dataset_id: $dataset_id})
                RETURN count(n) AS deleted
                """,
                {"dataset_id": dataset_id},
            )
            nodes_deleted = int(node_rows[0].get("deleted", 0)) if node_rows else 0

            if not dry_run:
                if relationships_deleted > 0:
                    graph_connection.execute_write(
                        """
                        MATCH ()-[r]->()
                        WHERE r.dataset_id = $dataset_id
                        DELETE r
                        """,
                        {"dataset_id": dataset_id},
                    )
                if nodes_deleted > 0:
                    graph_connection.execute_write(
                        """
                        MATCH (n:JsonEntity {dataset_id: $dataset_id})
                        DETACH DELETE n
                        """,
                        {"dataset_id": dataset_id},
                    )

        logger.info(
            f"Deleted dataset {dataset_id}: {nodes_deleted} nodes, {relationships_deleted} relationships"
        )
        return True, nodes_deleted, relationships_deleted, errors
    except Exception as exc:
        errors.append(f"Failed to delete dataset {dataset_id}: {exc}")
        return False, nodes_deleted, relationships_deleted, errors


def delete_entities_by_source_file(
    dataset_id: str,
    source_file: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> OperationSummary:
    """
    Delete all JSON entities associated with a specific source file.

    Args:
        dataset_id: Dataset identifier
        source_file: Source file path to match against entity source_file property
        batch_size: Graph connection batch size
        dry_run: If True, only count without deleting

    Returns:
        OperationSummary with deletion counts
    """
    summary = OperationSummary()

    try:
        with _create_json_ingestor(batch_size) as graph_connection:
            graph_connection.ensure_constraints()
            rows = graph_connection.fetch_all(
                """
                MATCH (n:JsonEntity {dataset_id: $dataset_id, source_file: $source_file})
                RETURN n.unique_id AS unique_id
                """,
                {"dataset_id": dataset_id, "source_file": source_file},
            )

            if dry_run:
                summary.deleted = len(rows)
                return summary

            for row in rows:
                unique_id = str(row["unique_id"])
                graph_connection.execute_write(
                    """
                    MATCH (n:JsonEntity {unique_id: $unique_id, dataset_id: $dataset_id})
                    DETACH DELETE n
                    """,
                    {"unique_id": unique_id, "dataset_id": dataset_id},
                )
                summary.deleted += 1

        logger.info(
            f"Deleted {summary.deleted} entities for source_file={source_file} in dataset={dataset_id}"
        )
    except Exception as exc:
        summary.failed += 1
        summary.errors.append(f"Failed to delete entities for {source_file}: {exc}")

    return summary


def ingest_json_data(
    input_path: str = "",
    dataset_id: str | None = None,
    skip_existing: bool = False,
    batch_size: int = 100,
    incremental: bool = False,
    dry_run: bool = False,
    conflict_resolution: str = "last-write-wins",
    pre_loaded_data: list[tuple[Path, dict[str, Any]]] | None = None,
    parallel_workers: int = settings.JSON_PARALLEL_WORKERS,
    metadata_override: dict[str, Any] | None = None,
    exclude_patterns: list[str] | None = None,
    filter_preset: str = "lenient",
) -> IngestionResult:
    del conflict_resolution

    result = IngestionResult(dataset_ids=[dataset_id] if dataset_id else [], dry_run=dry_run)

    if not settings.JSON_ENABLED:
        result.errors.append("JSON ingestion disabled via JSON_ENABLED=False")
        logger.error("JSON ingestion is disabled")
        return result

    try:
        load_errors: list[str] = []
        skip_count = 0
        if pre_loaded_data is not None:
            json_files = pre_loaded_data
        else:
            json_files, load_errors, skip_count = _load_json_files_with_errors(
                input_path, exclude_patterns, filter_preset
            )

        result.files_skipped += skip_count
        result.errors.extend(load_errors)
        logger.info(f"Loaded {len(json_files)} JSON file(s) for ingestion")

        if not json_files:
            logger.info("No valid JSON files found for ingestion")
            return result

        prepared_files: list[PreparedJsonFile] = []
        for file_data in json_files:
            prepared_file, errors = _prepare_json_file(
                file_data,
                dataset_id,
                metadata_override,
            )
            if prepared_file is None:
                result.files_skipped += 1
                result.errors.extend(errors)
                continue

            prepared_files.append(prepared_file)
            result.files_processed += 1
            result.entities_processed += len(prepared_file.entities)
            result.relationships_processed += len(prepared_file.relationships)

        # Collect all unique dataset IDs from prepared files
        result.dataset_ids = sorted({pf.dataset_id for pf in prepared_files})

        batch_errors = _validate_prepared_files(prepared_files)
        if batch_errors:
            result.errors.extend(batch_errors)
            return result

        if prepared_files and not dry_run:
            embed_available, _ = are_json_embeddings_available()
            if embed_available:
                _ensure_json_vector_index(batch_size)

        dataset_references = _build_dataset_references(prepared_files)

        if parallel_workers > 1 and len(prepared_files) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=parallel_workers
            ) as executor:
                entity_futures = [
                    executor.submit(
                        _ingest_entity_file,
                        prepared_file,
                        batch_size,
                        skip_existing,
                        dry_run,
                        incremental,
                    )
                    for prepared_file in prepared_files
                ]
                for future in concurrent.futures.as_completed(entity_futures):
                    try:
                        worker_result = future.result()
                        _merge_summary_into_result(result, worker_result, is_entity_summary=True)
                    except Exception as exc:
                        logger.error(f"Worker failed: {exc}")
                        failure = OperationSummary()
                        failure.failed = 1
                        failure.errors.append(f"Worker execution failed: {exc}")
                        _merge_summary_into_result(result, failure, is_entity_summary=True)

                relationship_futures = [
                    executor.submit(
                        _ingest_relationship_file,
                        prepared_file,
                        dataset_references.get(
                            prepared_file.dataset_id,
                            DatasetReferences(),
                        ),
                        batch_size,
                        skip_existing,
                        dry_run,
                        incremental,
                    )
                    for prepared_file in prepared_files
                ]
                for future in concurrent.futures.as_completed(relationship_futures):
                    try:
                        worker_result = future.result()
                        _merge_summary_into_result(result, worker_result, is_entity_summary=False)
                    except Exception as exc:
                        logger.error(f"Relationship worker failed: {exc}")
                        failure = OperationSummary()
                        failure.failed = 1
                        failure.errors.append(f"Relationship worker execution failed: {exc}")
                        _merge_summary_into_result(result, failure, is_entity_summary=False)
        else:
            for prepared_file in prepared_files:
                _merge_summary_into_result(
                    result,
                    _ingest_entity_file(
                        prepared_file,
                        batch_size,
                        skip_existing,
                        dry_run,
                        incremental,
                    ),
                    is_entity_summary=True,
                )

            for prepared_file in prepared_files:
                _merge_summary_into_result(
                    result,
                    _ingest_relationship_file(
                        prepared_file,
                        dataset_references.get(
                            prepared_file.dataset_id,
                            DatasetReferences(),
                        ),
                        batch_size,
                        skip_existing,
                        dry_run,
                        incremental,
                    ),
                    is_entity_summary=False,
                )

        # Log completion with guidance if nothing was ingested
        if result.entities_ingested == 0 and result.files_processed > 0:
            logger.warning(
                f"Ingestion completed with {result.files_processed} file(s) processed "
                f"but 0 entities ingested. This usually means:"
            )
            logger.warning(
                "  1. The JSON files don't match the expected ingestion format "
                "(needs 'entities' array with entity objects)"
            )
            logger.warning(
                "  2. The files are documentation, configuration, or schema files rather than entity data"
            )
            logger.warning(
                "  3. Check the 'entities' array is present and contains valid entity objects with 'id' and 'name' fields"
            )

        logger.info(
            "Ingestion completed: "
            f"{result.files_processed} files processed, "
            f"{result.files_skipped} files skipped, "
            f"{result.entities_ingested} entities ingested, "
            f"{result.relationships_ingested} relationships ingested"
        )
        return result
    except Exception as exc:
        result.errors.append(f"Ingestion failed: {exc}")
        logger.error(f"Ingestion failed: {exc}", exc_info=True)
        return result


def handle_json_update_event(
    event: dict[str, Any],
    dataset_id: str,
    conflict_resolution: str = "last-write-wins",
    dry_run: bool = False,
    metadata: dict[str, Any] | None = None,
) -> UpdateResult:
    event_id = event.get("id")
    operation = str(event.get("operation", "add")).lower()
    processed_at = datetime.now(UTC).isoformat()
    last_updated = str(event.get("timestamp") or processed_at)

    logger.info(
        f"Processing update event {event_id}: operation={operation}, dataset={dataset_id}"
    )

    merged_metadata: dict[str, Any] = {"dataset_id": dataset_id}
    if metadata:
        merged_metadata.update(metadata)

    json_data = {
        "metadata": merged_metadata,
        "operation": operation,
        "last_updated": last_updated,
        "entities": event.get("entities", []),
        "relationships": event.get("relationships", []),
    }

    valid, _, validation_errors = validate_json_input(
        json_data,
        allow_external_references=True,
    )
    if not valid:
        return UpdateResult(
            dataset_id=dataset_id,
            operation=operation,
            event_id=event_id,
            processed_at=processed_at,
            errors=validation_errors,
            entities_failed=len(json_data.get("entities", [])),
            relationships_failed=len(json_data.get("relationships", [])),
            dry_run=dry_run,
        )

    result = ingest_json_data(
        pre_loaded_data=[(Path("event_stream.json"), json_data)],
        dataset_id=dataset_id,
        incremental=True,
        conflict_resolution=conflict_resolution,
        dry_run=dry_run,
        parallel_workers=1,
        exclude_patterns=None,
    )

    return UpdateResult(
        **result.model_dump(),
        operation=operation,
        event_id=event_id,
        processed_at=processed_at,
    )
