from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import jsonschema
from loguru import logger
from tqdm import tqdm

from .config import settings
from .embedder import EmbeddingCache, get_embedding_provider_instance
from .schemas import IngestionResult, UpdateResult
from .services.graph_service import MemgraphIngestor
from .vector_store import _get_backend as get_vector_store_instance

__all__ = [
    "ingest_json_data",
    "delete_dataset",
    "handle_json_update_event",
    "validate_json_input",
    "load_json_files",
    "IngestionResult",
    "UpdateResult",
]

config = settings

# Load official ingestion schema (single source of truth)
SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ingestion_schema.json"
)
with open(SCHEMA_PATH) as f:
    INGESTION_SCHEMA = json.load(f)

embedding_provider = get_embedding_provider_instance()
embedding_cache = EmbeddingCache()
vector_store = get_vector_store_instance()


# Round-robin Memgraph connection pool for parallel workers (thread-safe)
class MemgraphConnectionPool:
    def __init__(self, host: str, port: int, pool_size: int = 10):
        self.pool_size = pool_size
        self.connections = [
            MemgraphIngestor(host=host, port=port) for _ in range(pool_size)
        ]
        self._lock = threading.Lock()
        self._counter = 0

    def get(self) -> MemgraphIngestor:
        """Get next connection in round-robin fashion (thread-safe)"""
        with self._lock:
            conn = self.connections[self._counter % self.pool_size]
            self._counter += 1
            return conn


# Initialize pool with connection count matching configured parallel worker count
graph_pool = MemgraphConnectionPool(
    host=settings.JSON_MEMGRAPH_HOST,
    port=settings.JSON_MEMGRAPH_PORT,
    pool_size=settings.JSON_PARALLEL_WORKERS,
)
# Backward compatibility: single connection for existing non-parallel code
graph_service = graph_pool.get()


def validate_json_input(
    data: dict[str, Any],
) -> tuple[bool, dict | None, list[str]]:
    """
    Validate JSON input directly against the official ingestion_schema.json.
    NO CONVERSION - data must match schema exactly.
    """
    errors = []
    try:
        jsonschema.validate(instance=data, schema=INGESTION_SCHEMA)
        return True, data, []
    except jsonschema.exceptions.ValidationError as e:
        errors.append(f"Schema validation failed: {str(e)}")
        return False, None, errors
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, None, errors


def load_json_files(input_path: str) -> list[tuple[Path, dict[str, Any]]]:
    """Load JSON files from a path (single file or directory)."""
    path = Path(input_path)
    json_files = []

    if path.is_file() and path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            json_files.append((path, json.load(f)))
    elif path.is_dir():
        for file_path in path.rglob("*.json"):
            with open(file_path, encoding="utf-8") as f:
                json_files.append((file_path, json.load(f)))
    else:
        raise ValueError(
            f"Invalid input path: {input_path} (must be .json file or directory containing JSON files)"
        )

    return json_files


def generate_embeddings_for_entities(
    entities: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], list[str]]:
    """Generate embeddings for entity descriptions.
    Uses properties.description, falls back to name if missing.
    """
    embeddings = {}
    errors = []

    texts = []
    entity_ids = []
    for entity in entities:
        entity_id = entity["id"]
        name = entity.get("name", "")
        description = entity.get("properties", {}).get("description", name)
        text = f"{name} - {description}"
        texts.append(text)
        entity_ids.append(entity_id)

    # Check cache first
    cached_embeddings = embedding_cache.get_many(texts)
    to_generate = []
    to_generate_ids = []

    for i, (text, eid) in enumerate(zip(texts, entity_ids)):
        if i in cached_embeddings:
            embeddings[eid] = cached_embeddings[i]
        else:
            to_generate.append(text)
            to_generate_ids.append(eid)

    # Generate missing embeddings
    if to_generate:
        try:
            generated = embedding_provider.embed_batch(to_generate)
            for idx, (eid, emb) in enumerate(zip(to_generate_ids, generated)):
                embeddings[eid] = emb
                embedding_cache.put(to_generate[idx], emb)
        except Exception as e:
            errors.append(f"Embedding generation failed: {str(e)}")

    return embeddings, errors


def generate_embeddings_for_relationships(
    relationships: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], list[str]]:
    """Generate embeddings for relationships (if they have description)."""
    embeddings = {}
    errors = []

    texts = []
    rel_keys = []
    for rel in relationships:
        # Create unique key for relationship
        key = f"{rel['source']}-{rel['target']}-{rel['relationship']}"
        description = rel.get("properties", {}).get("description", "")
        if not description:
            # Fallback to relationship type and entity reference
            description = (
                f"{rel['relationship']} between {rel['source']} and {rel['target']}"
            )
        text = description
        texts.append(text)
        rel_keys.append(key)

    # Check cache first
    cached_embeddings = embedding_cache.get_batch(texts)
    to_generate = []
    to_generate_keys = []

    for i, (text, key) in enumerate(zip(texts, rel_keys)):
        if cached_embeddings[i] is not None:
            embeddings[key] = cached_embeddings[i]
        else:
            to_generate.append(text)
            to_generate_keys.append(key)

    # Generate missing embeddings
    if to_generate:
        try:
            generated = embedding_provider.encode_batch(to_generate)
            for key, emb in zip(to_generate_keys, generated):
                embeddings[key] = emb
                embedding_cache.put(to_generate[i], emb)
        except Exception as e:
            errors.append(f"Relationship embedding generation failed: {str(e)}")

    return embeddings, errors


def generate_embeddings_for_relationships(
    relationships: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], list[str]]:
    """Generate embeddings for relationship descriptions (if present)."""
    embeddings = {}
    errors = []

    texts = []
    rel_ids = []
    for rel in relationships:
        # Use explanation if present, fall back to description in properties
        text = rel.get("explanation") or rel.get("properties", {}).get("description")
        if not text or not text.strip():
            continue

        texts.append(text)
        # Generate relationship ID
        rel_id = (
            rel.get("id") or f"{rel['source']}_{rel['target']}_{rel['relationship']}"
        )
        rel_ids.append(rel_id)

    # Check cache first
    cached_embeddings = embedding_cache.get_many(texts)
    to_generate = []
    to_generate_ids = []

    for i, (text, rid) in enumerate(zip(texts, rel_ids)):
        if i in cached_embeddings:
            embeddings[rid] = cached_embeddings[i]
        else:
            to_generate.append(text)
            to_generate_ids.append(rid)

    # Generate missing embeddings
    if to_generate:
        try:
            generated = embedding_provider.embed_batch(to_generate)
            for idx, (rid, emb) in enumerate(zip(to_generate_ids, generated)):
                embeddings[rid] = emb
                embedding_cache.put(to_generate[idx], emb)
        except Exception as e:
            errors.append(f"Relationship embedding generation failed: {str(e)}")

    return embeddings, errors


def ingest_entities(
    dataset_id: str,
    entities: list[dict[str, Any]],
    entity_embeddings: dict[str, list[float]],
    skip_existing: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
    conflict_resolution: str = "last-write-wins",
    last_updated_threshold: str | None = None,
    graph_connection: MemgraphIngestor | None = None,
) -> tuple[int, int, int, int, list[str]]:
    """
    Ingest entities directly from ingestion schema format into Memgraph.
    Uses thread-local connection from round-robin pool for parallel processing.
    """
    ingested = 0
    updated = 0
    skipped = 0
    failed = 0
    errors = []
    graph_conn = graph_connection or graph_pool.get()

    # Get existing entity IDs for skip_existing check
    existing_nodes: dict[str, str] = {}
    if skip_existing:
        try:
            result = graph_conn._execute_query(
                "MATCH (n {dataset_id: $dataset_id}) RETURN n.unique_id AS id, n.last_updated AS last_updated",
                {"dataset_id": dataset_id},
            )
            existing_nodes = {row["id"]: row["last_updated"] for row in result}
        except Exception as e:
            logger.warning(f"Could not fetch existing nodes: {str(e)}")

    for entity in tqdm(entities, desc="Ingesting entities"):
        entity_id = entity["id"]
        name = entity.get("name", "")
        entity_type = entity.get("type", "Entity")
        labels = entity.get("labels", [])
        properties = entity.get("properties", {})
        last_updated = entity.get("last_updated", None)

        # Add required metadata properties
        properties["name"] = name
        properties["type"] = entity_type
        properties["dataset_id"] = dataset_id
        if last_updated:
            properties["last_updated"] = last_updated

        # Unique ID includes dataset_id to avoid collisions
        unique_id = f"{dataset_id}::{entity_id}"

        # Skip if existing
        if skip_existing and unique_id in existing_nodes:
            skipped += 1
            continue

        # Incremental update check
        if incremental and last_updated_threshold and unique_id in existing_nodes:
            if existing_nodes[unique_id] >= last_updated_threshold:
                skipped += 1
                continue

        # Merge node into graph
        if not dry_run:
            # Build labels string, escape each label with backticks for spaces/special chars
            all_labels = list(set(labels + [entity_type]))
            escaped_labels = [f"`{label}`" for label in all_labels]
            label_string = ":".join(escaped_labels)
            if not label_string:
                label_string = "`Entity`"

            # Build merge query with parameters to avoid escaping issues
            query = f"MERGE (n:{label_string} {{unique_id: $unique_id}}) SET n += $properties RETURN n"
            params = {"unique_id": unique_id, "properties": properties}

            try:
                result = graph_conn._execute_query(query, params)
                # Check if node was created or updated
                if result and len(result) > 0 and "n" in result[0]:
                    # MERGE succeeded, count as ingested/updated
                    ingested += 1
                else:
                    failed += 1
                    errors.append(f"Failed to ingest entity {entity_id}")
                    continue
            except Exception as e:
                failed += 1
                errors.append(f"Failed to ingest entity {entity_id}: {str(e)}")
                continue
        else:
            # Dry run: just count
            ingested += 1

        # Add to vector store
        if entity_id in entity_embeddings and not dry_run:
            try:
                vector_store.add_item(
                    id=unique_id,
                    embedding=entity_embeddings[entity_id],
                    metadata={
                        "entity_id": entity_id,
                        "dataset_id": dataset_id,
                        "name": name,
                        "type": entity_type,
                        "labels": all_labels,
                        "description": properties.get("description", name),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to add entity to vector store: {str(e)}")

    return ingested, updated, skipped, failed, errors


def ingest_relationships(
    dataset_id: str,
    relationships: list[dict[str, Any]],
    entity_name_to_id: dict[str, str],
    rel_embeddings: dict[str, list[float]],
    skip_existing: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
    conflict_resolution: str = "last-write-wins",
    last_updated_threshold: str | None = None,
    graph_connection: MemgraphIngestor | None = None,
) -> tuple[int, int, int, int, list[str]]:
    """
    Ingest relationships directly from ingestion schema format.
    Resolves source/target references that can be either ID or name.
    Uses thread-local connection from round-robin pool for parallel processing.
    """
    ingested = 0
    updated = 0
    skipped = 0
    failed = 0
    errors = []
    graph_conn = graph_connection or graph_pool.get()

    # Build full ID map (original ID -> unique dataset ID)
    full_id_map = {}
    for entity_id in entity_name_to_id.values():
        full_id_map[entity_id] = f"{dataset_id}::{entity_id}"

    # Get existing relationships for skip check
    existing_rels: set[tuple[str, str, str]] = set()
    if skip_existing:
        try:
            result = graph_conn._execute_query(
                """
                MATCH (s {dataset_id: $dataset_id})-[r {dataset_id: $dataset_id}]->(t {dataset_id: $dataset_id})
                RETURN s.unique_id AS source, type(r) AS rel_type, t.unique_id AS target
                """,
                {"dataset_id": dataset_id},
            )
            existing_rels = {
                (row["source"], row["rel_type"], row["target"]) for row in result
            }
        except Exception as e:
            logger.warning(f"Could not fetch existing relationships: {str(e)}")

    for rel in tqdm(relationships, desc="Ingesting relationships"):
        source_ref = rel["source"]
        target_ref = rel["target"]
        rel_type = rel["relationship"]
        properties = rel.get("properties", {})
        last_updated = rel.get("last_updated", None)

        # Resolve references (can be ID or name)
        source_id = entity_name_to_id.get(source_ref)
        if not source_id:
            # Try to match by partial name
            for name, eid in entity_name_to_id.items():
                if source_ref in name or name in source_ref:
                    source_id = eid
                    break

        target_id = entity_name_to_id.get(target_ref)
        if not target_id:
            # Try to match by partial name
            for name, eid in entity_name_to_id.items():
                if target_ref in name or name in target_ref:
                    target_id = eid
                    break

        if not source_id or not target_id:
            failed += 1
            errors.append(
                f"Could not resolve relationship reference: {source_ref} -> {target_ref}"
            )
            continue

        # Get full unique IDs
        full_source_id = full_id_map[source_id]
        full_target_id = full_id_map[target_id]

        # Skip existing
        rel_key = (full_source_id, rel_type, full_target_id)
        if skip_existing and rel_key in existing_rels:
            skipped += 1
            continue

        # Add metadata properties
        properties["dataset_id"] = dataset_id
        if last_updated:
            properties["last_updated"] = last_updated

        # Merge relationship into graph
        if not dry_run:
            try:
                # Build merge relationship query directly, escape rel_type with backticks
                escaped_rel_type = f"`{rel_type}`"
                # Build merge query with parameters to avoid escaping issues
                query = f"""
                MATCH (a {{unique_id: $source_id}}), (b {{unique_id: $target_id}})
                MERGE (a)-[r:{escaped_rel_type}]->(b)
                SET r += $properties
                RETURN r, r.created_at IS NOT NULL AS was_created
                """
                params = {
                    "source_id": full_source_id,
                    "target_id": full_target_id,
                    "properties": properties,
                }
                result = graph_conn._execute_query(query, params)
                if result and len(result) > 0 and "r" in result[0]:
                    # MERGE succeeded, count as ingested/updated
                    ingested += 1
                else:
                    failed += 1
                    errors.append(
                        f"Failed to ingest relationship {source_ref} -> {target_ref}"
                    )
                    continue
            except Exception as e:
                failed += 1
                errors.append(
                    f"Failed to ingest relationship {source_ref} -> {target_ref}: {str(e)}"
                )
                continue
        else:
            # Dry run: just count
            ingested += 1

    return ingested, updated, skipped, failed, errors


def delete_dataset(
    dataset_id: str, dry_run: bool = False
) -> tuple[bool, int, int, list[str]]:
    """Delete all nodes, relationships, and vector entries for a dataset."""
    errors = []
    nodes_deleted = 0
    rels_deleted = 0

    try:
        if not dry_run:
            # Delete relationships first
            rel_query = """
            MATCH ()-[r]->()
            WHERE r.source_dataset = $dataset_id
            DELETE r
            RETURN count(r) as deleted
            """
            rel_result = graph_service._execute_query(
                rel_query, {"dataset_id": dataset_id}
            )
            rels_deleted = rel_result[0][0] if rel_result else 0

            # Delete nodes
            node_query = """
            MATCH (n)
            WHERE n.source_dataset = $dataset_id
            DELETE n
            RETURN count(n) as deleted
            """
            node_result = graph_service._execute_query(
                node_query, {"dataset_id": dataset_id}
            )
            nodes_deleted = node_result[0][0] if node_result else 0

            # Delete vector entries
            vector_store.delete_by_metadata({"dataset_id": dataset_id})

        logger.info(
            f"Deleted dataset {dataset_id}: {nodes_deleted} nodes, {rels_deleted} relationships"
        )
        return True, nodes_deleted, rels_deleted, errors

    except Exception as e:
        errors.append(f"Failed to delete dataset {dataset_id}: {str(e)}")
        return False, nodes_deleted, rels_deleted, errors


def ingest_json_data(
    input_path: str = "",
    dataset_id: str | None = None,
    skip_existing: bool = False,
    batch_size: int = 100,
    incremental: bool = False,
    dry_run: bool = False,
    conflict_resolution: str = "last-write-wins",
    pre_loaded_data: list[tuple[Path, dict[str, Any]]] | None = None,
    # New parameters (backward compatible defaults)
    parallel_workers: int = settings.JSON_PARALLEL_WORKERS,
    metadata_override: dict[str, Any] | None = None,
) -> IngestionResult:
    """
    Ingest JSON data into graph and vector database.
    Exact schema match with ingestion_schema.json required - no conversions.
    Supports parallel processing with configurable worker count.
    """
    import concurrent.futures

    result = IngestionResult(dataset_id=dataset_id or "", dry_run=dry_run)

    def process_single_file(
        file_data: tuple[Path, dict[str, Any]],
    ) -> tuple[IngestionResult, bool]:
        """Process a single JSON file, returns partial result and success flag"""
        file_path, data = file_data
        partial_result = IngestionResult(dataset_id=dataset_id or "", dry_run=dry_run)
        # Get round-robin connection from pool for this worker thread
        graph_conn = graph_pool.get()

        # Validate input against official schema
        valid, validated_data, validation_errors = validate_json_input(data)
        if not valid or not validated_data:
            partial_result.errors.extend(validation_errors)
            logger.error(f"Validation failed for {file_path}: {validation_errors}")
            return partial_result, False

        # Apply metadata overrides if provided (e.g. workspace ID)
        if metadata_override:
            if "metadata" not in validated_data:
                validated_data["metadata"] = {}
            validated_data["metadata"].update(metadata_override)

        # Auto-generate missing IDs for entities (use name as base)
        for entity in validated_data.get("entities", []):
            if "id" not in entity:
                # Generate deterministic ID from name
                entity_id = entity["name"].replace(" ", "_").replace("/", "_").lower()
                entity["id"] = entity_id

        # Override dataset ID if provided
        current_dataset_id = dataset_id or validated_data["metadata"].get("dataset_id")
        if not current_dataset_id:
            partial_result.errors.append("Missing required 'dataset_id' in metadata")
            return partial_result, False
        partial_result.dataset_id = current_dataset_id

        entities = validated_data.get("entities", [])
        relationships = validated_data.get("relationships", [])

        # Count entities and relationships to process
        partial_result.entities_processed += len(entities)
        partial_result.relationships_processed += len(relationships)

        # Build name-to-ID map for relationship resolution
        entity_name_to_id: dict[str, str] = {}
        for entity in entities:
            entity_id = entity["id"]
            name = entity.get("name")
            if name:
                entity_name_to_id[name] = entity_id
            # Also add ID as a reference
            entity_name_to_id[entity_id] = entity_id

        # Generate embeddings
        entity_embeddings, embed_errors = generate_embeddings_for_entities(entities)
        partial_result.errors.extend(embed_errors)

        rel_embeddings, rel_embed_errors = generate_embeddings_for_relationships(
            relationships
        )
        partial_result.errors.extend(rel_embed_errors)

        # Ingest entities with thread-local connection
        e_ingested, e_updated, e_skipped, e_failed, e_errors = ingest_entities(
            current_dataset_id,
            entities,
            entity_embeddings,
            skip_existing=skip_existing,
            dry_run=dry_run,
            incremental=incremental,
            conflict_resolution=conflict_resolution,
            last_updated_threshold=validated_data["metadata"].get("last_updated"),
            graph_connection=graph_conn,
        )
        partial_result.entities_ingested += e_ingested
        partial_result.entities_updated += e_updated
        partial_result.entities_skipped += e_skipped
        partial_result.entities_failed += e_failed
        partial_result.errors.extend(e_errors)

        # Ingest relationships with thread-local connection
        r_ingested, r_updated, r_skipped, r_failed, r_errors = ingest_relationships(
            current_dataset_id,
            relationships,
            entity_name_to_id,
            rel_embeddings,
            skip_existing=skip_existing,
            dry_run=dry_run,
            incremental=incremental,
            conflict_resolution=conflict_resolution,
            last_updated_threshold=validated_data["metadata"].get("last_updated"),
            graph_connection=graph_conn,
        )
        partial_result.relationships_ingested += r_ingested
        partial_result.relationships_updated += r_updated
        partial_result.relationships_skipped += r_skipped
        partial_result.relationships_failed += r_failed
        partial_result.errors.extend(r_errors)

        return partial_result, True

    try:
        with graph_service:
            # Load JSON files or use pre-loaded data
            if pre_loaded_data is not None:
                json_files = pre_loaded_data
            else:
                json_files = load_json_files(input_path)
            logger.info(f"Loaded {len(json_files)} JSON file(s) for ingestion")

            # Process files in parallel if workers > 1
            if parallel_workers > 1 and len(json_files) > 1:
                logger.info(
                    f"Processing files with {parallel_workers} parallel workers"
                )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=parallel_workers
                ) as executor:
                    # Submit all files for processing
                    future_to_file = {
                        executor.submit(process_single_file, file_data): file_data
                        for file_data in json_files
                    }

                    # Aggregate results as they complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        partial_result, success = future.result()
                        if success:
                            result.files_processed += 1
                        else:
                            result.files_skipped += 1

                        # Merge partial results into main result
                        result.entities_processed += partial_result.entities_processed
                        result.entities_ingested += partial_result.entities_ingested
                        result.entities_updated += partial_result.entities_updated
                        result.entities_skipped += partial_result.entities_skipped
                        result.entities_failed += partial_result.entities_failed
                        result.relationships_processed += (
                            partial_result.relationships_processed
                        )
                        result.relationships_ingested += (
                            partial_result.relationships_ingested
                        )
                        result.relationships_updated += (
                            partial_result.relationships_updated
                        )
                        result.relationships_skipped += (
                            partial_result.relationships_skipped
                        )
                        result.relationships_failed += (
                            partial_result.relationships_failed
                        )
                        result.errors.extend(partial_result.errors)
                        if partial_result.dataset_id and not result.dataset_id:
                            result.dataset_id = partial_result.dataset_id
            else:
                # Single worker processing (original logic, backward compatible)
                logger.info("Processing files with single worker")
                for file_path, data in json_files:
                    partial_result, success = process_single_file((file_path, data))
                    if success:
                        result.files_processed += 1
                    else:
                        result.files_skipped += 1

                    # Merge partial result
                    result.entities_processed += partial_result.entities_processed
                    result.entities_ingested += partial_result.entities_ingested
                    result.entities_updated += partial_result.entities_updated
                    result.entities_skipped += partial_result.entities_skipped
                    result.entities_failed += partial_result.entities_failed
                    result.relationships_processed += (
                        partial_result.relationships_processed
                    )
                    result.relationships_ingested += (
                        partial_result.relationships_ingested
                    )
                    result.relationships_updated += partial_result.relationships_updated
                    result.relationships_skipped += partial_result.relationships_skipped
                    result.relationships_failed += partial_result.relationships_failed
                    result.errors.extend(partial_result.errors)
                    if partial_result.dataset_id and not result.dataset_id:
                        result.dataset_id = partial_result.dataset_id

            logger.info(
                f"Ingestion completed: {result.files_processed} files processed, {result.files_skipped} files skipped, "
                f"{result.entities_ingested} entities ingested, {result.relationships_ingested} relationships ingested"
            )
            return result

    except Exception as e:
        result.errors.append(f"Ingestion failed: {str(e)}")
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        return result


def handle_json_update_event(
    event: dict[str, Any],
    dataset_id: str,
    conflict_resolution: str = "last-write-wins",
    dry_run: bool = False,
) -> UpdateResult:
    """Process single incremental JSON update event from streaming source (Kafka/RabbitMQ/webhook)."""
    from datetime import datetime

    from .schemas import UpdateResult

    event_id = event.get("id")
    operation = event.get("operation", "add").lower()

    logger.info(
        f"Processing update event {event_id}: operation={operation}, dataset={dataset_id}"
    )

    # Create temporary JSON input structure from event
    json_data = {
        "metadata": {
            "dataset_id": dataset_id,
            "operation": operation,
            "last_updated": event.get("timestamp", datetime.utcnow().isoformat()),
        },
        "entities": event.get("entities", []),
        "relationships": event.get("relationships", []),
    }

    # Validate event data
    valid, validated_data, validation_errors = validate_json_input(json_data)
    if not valid or not validated_data:
        return UpdateResult(
            dataset_id=dataset_id,
            operation=operation,
            event_id=event_id,
            processed_at=datetime.utcnow().isoformat(),
            errors=validation_errors,
            entities_failed=len(json_data.get("entities", [])),
            relationships_failed=len(json_data.get("relationships", [])),
            dry_run=dry_run,
        )

    # Ingest event data with incremental mode enabled
    result = ingest_json_data(
        pre_loaded_data=[(Path("event_stream.json"), json_data)],
        dataset_id=dataset_id,
        incremental=True,
        conflict_resolution=conflict_resolution,
        dry_run=dry_run,
    )

    # Convert to UpdateResult
    return UpdateResult(
        **result.model_dump(),
        operation=operation,
        event_id=event_id,
        processed_at=datetime.utcnow().isoformat(),
    )
